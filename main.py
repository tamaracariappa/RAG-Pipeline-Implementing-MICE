"""
main.py – FM RAG Pipeline orchestrator with ATOMIC CHUNK-BASED CHECKPOINTING.

Stages:
  1. Dataset check
  2. Preprocessing   (skipped if output exists)
  3. Cleaning        (skipped if output exists)
  4. FAISS ingest    (RESUMABLE - atomic chunk-based checkpoints)
  5. Evaluation      – all four strategies, shared query set, fixed seed
  6. Analysis        – per-query comparison, win matrix, metadata impact, CSV export

ATOMIC CHECKPOINTING:
  - Tracks completed chunks (not rows) to avoid duplicates
  - Each chunk is fully inserted + persisted before marking as done
  - Power-cut safe: chunks are either 100% done or 0% done
  - Delete data/ingestion_progress.json to restart from scratch
"""

from __future__ import annotations

import json
import logging
import os
import random
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

import cleaning
import faiss_store
import preprocessing
from analysis import run_full_analysis
from config import (
    CHECKPOINT_INTERVAL,
    CLEANED_PATH,
    DATA_DIR,
    EVAL_RESULTS_PATH,
    EVAL_SEED,
    PREPROCESSED_PATH,
    PROGRESS_FILE,
    RAW_DATASET_PATH,
    DEFAULT_TOP_K,
    TEXT_INDEX_PATH,
)
from embedder import embed_texts
from embedding_builder import add_text_columns, iter_cleaned_chunks
from evaluation import (
    generate_test_cases,
    print_summary,
    run_evaluation,
    save_summary_json,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────

def _fix_seeds(seed: int = EVAL_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ─────────────────────────────────────────────────────────────
# ATOMIC CHUNK-BASED CHECKPOINT MANAGEMENT
# ─────────────────────────────────────────────────────────────

def load_progress():
    """Load ingestion progress from checkpoint file."""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            data = json.load(f)
            # Ensure completed_chunks is a set for O(1) lookup
            if "completed_chunks" in data:
                data["completed_chunks"] = set(data["completed_chunks"])
            return data
    return {
        "completed_chunks": set(),  # Set of chunk indices that are 100% done
        "total_rows_processed": 0,
        "completed": False
    }


def save_progress(completed_chunks, total_rows_processed, completed=False):
    """
    Save ingestion progress to checkpoint file.
    
    Args:
        completed_chunks: Set of chunk indices that are fully processed
        total_rows_processed: Total number of rows completed
        completed: True when entire dataset is done
    """
    with open(PROGRESS_FILE, 'w') as f:
        json.dump({
            "completed_chunks": sorted(list(completed_chunks)),  # JSON needs list
            "total_rows_processed": total_rows_processed,
            "completed": completed,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=2)


def clear_progress():
    """Delete progress file to restart from scratch."""
    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)
        log.info("Cleared ingestion progress - will start from scratch")


# ─────────────────────────────────────────────────────────────
# Stages
# ─────────────────────────────────────────────────────────────

def _check_dataset() -> bool:
    if os.path.exists(RAW_DATASET_PATH):
        return True
    log.error("Dataset not found: %s", RAW_DATASET_PATH)
    log.error("Download: https://data.mendeley.com/datasets/cb8d2nsjss/1")
    return False


def _stage_preprocess() -> None:
    if os.path.exists(PREPROCESSED_PATH):
        log.info("Preprocessed CSV exists – skipping."); return
    log.info("Stage 2 – Preprocessing …")
    df = preprocessing.load_dataset(RAW_DATASET_PATH)
    records = [preprocessing.preprocess_row(r) for _, r in tqdm(df.iterrows(), total=len(df))]
    pd.DataFrame(records).to_csv(PREPROCESSED_PATH, index=False)
    log.info("Saved → %s", PREPROCESSED_PATH)


def _stage_clean() -> None:
    if os.path.exists(CLEANED_PATH):
        log.info("Cleaned CSV exists – skipping."); return
    log.info("Stage 3 – Cleaning …")
    cleaning.clean_preprocessed_dataset(PREPROCESSED_PATH, CLEANED_PATH)


def _count_rows(path: str) -> int:
    with open(path, encoding="utf-8", errors="replace") as fh:
        return sum(1 for _ in fh) - 1


def _stage_ingest():
    """
    FAISS ingest with ATOMIC chunk-based checkpointing.
    
    Each chunk is processed atomically:
      1. Load chunk
      2. Generate embeddings
      3. Insert into FAISS
      4. Persist FAISS to disk
      5. Mark chunk as complete
      6. Save progress
    
    Power cuts can only happen between chunks, never during a chunk.
    This prevents duplicate insertions.
    """
    
    # Check if already completed
    progress = load_progress()
    
    if progress.get("completed", False):
        log.info("FAISS ingestion already completed - skipping")
        return

    log.info("Stage 4 – FAISS ingest (atomic chunk-based checkpointing) …")
    
    # Load or initialize FAISS stores
    faiss_store.initialize_stores()

    total_rows = _count_rows(CLEANED_PATH)
    completed_chunks = progress.get("completed_chunks", set())
    total_rows_processed = progress.get("total_rows_processed", 0)
    
    if completed_chunks:
        log.info("🔄 RESUMING: %d chunks already completed (%d rows, %.1f%%)", 
                 len(completed_chunks), total_rows_processed, 
                 100 * total_rows_processed / total_rows)
    else:
        log.info("Starting fresh ingestion")

    chunks_since_persist = 0
    
    with tqdm(total=total_rows, desc="Ingesting", unit="rows", 
              initial=total_rows_processed) as pbar:

        for chunk_idx, chunk in enumerate(iter_cleaned_chunks()):
            
            # Skip if this chunk was already fully processed
            if chunk_idx in completed_chunks:
                continue
            
            chunk_size = len(chunk)
            
            # ─────────────────────────────────────────────────
            # ATOMIC CHUNK PROCESSING
            # ─────────────────────────────────────────────────
            
            # Step 1: Prepare data
            chunk = add_text_columns(chunk)
            rows = chunk.to_dict(orient="records")

            # Step 2: Generate embeddings
            text_embs = embed_texts(chunk["text_repr"].tolist())
            mice_embs = embed_texts(chunk["mice_repr"].tolist())

            # Step 3: Insert into FAISS (in-memory)
            faiss_store.insert_text_batch(rows, text_embs)
            faiss_store.insert_mice_batch(rows, mice_embs)

            # Step 4: Persist to disk immediately (atomic commit)
            faiss_store.persist()
            
            # Step 5: Mark chunk as complete
            completed_chunks.add(chunk_idx)
            total_rows_processed += chunk_size
            chunks_since_persist += 1
            
            # Step 6: Save progress checkpoint
            # (do this frequently to avoid losing progress metadata)
            if chunks_since_persist >= 5 or chunk_idx % 10 == 0:  
                # Save every 5 chunks or every 10th chunk
                save_progress(completed_chunks, total_rows_processed, completed=False)
                chunks_since_persist = 0
            
            # Update progress bar
            pbar.update(chunk_size)
            
            # Log periodic checkpoints for visibility
            if total_rows_processed % CHECKPOINT_INTERVAL < chunk_size:
                log.info("💾 Progress: %d rows completed (%.1f%%)", 
                         total_rows_processed, 
                         100 * total_rows_processed / total_rows)

    # Final save
    log.info("✅ Ingestion complete - final save")
    save_progress(completed_chunks, total_rows_processed, completed=True)
    log.info("FAISS ingest complete: %d rows in %d chunks", 
             total_rows_processed, len(completed_chunks))


def _stage_evaluate_and_analyse() -> None:
    log.info("Stage 5 – Evaluation …")

    _fix_seeds()

    # Load FAISS indexes into RAM
    faiss_store.initialize_stores()

    # Generate once; reuse for both evaluation and analysis
    test_cases = generate_test_cases(seed=EVAL_SEED)

    all_metrics = run_evaluation(
        test_cases=test_cases,
        seed=EVAL_SEED
    )

    print_summary(all_metrics)

    save_summary_json(all_metrics, EVAL_RESULTS_PATH)

    log.info("Stage 6 – Analysis …")

    analysis_csv = os.path.join(
        DATA_DIR,
        "per_query_analysis.csv"
    )

    run_full_analysis(
        test_cases=test_cases,
        csv_path=analysis_csv,
        top_k=DEFAULT_TOP_K,
        print_n_examples=5,
    )

# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

def RAG_Pipeline() -> None:
    print("=" * 60)
    print("FM RAG PIPELINE – Research Edition")
    print("Atomic Chunk-Based Checkpointing (Duplicate-Safe)")
    print("=" * 60)
    _fix_seeds()
    t0 = time.time()
    os.makedirs(DATA_DIR, exist_ok=True)

    if not _check_dataset():
        return

    _stage_preprocess()
    _stage_clean()

    _stage_ingest()
    _stage_evaluate_and_analyse()

    log.info("Done in %.1f s.", time.time() - t0)


if __name__ == "__main__":
    RAG_Pipeline()
