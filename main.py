"""
main.py - FM RAG Pipeline orchestrator with ATOMIC CHUNK-BASED CHECKPOINTING.

Stages:
  1. Dataset check
  2. Preprocessing   (skipped if output exists)
  3. Cleaning        (skipped if output exists)
  4. FAISS ingest    (RESUMABLE - atomic chunk-based checkpoints)
  5. Evaluation      - strategies A / B / C, shared query set, fixed seed
  6. Analysis        - per-query comparison, win matrix, metadata impact, CSV export

MODEL SELECTION:
  python main.py --model bge_base | bge_m3 | jina_v3

  The key is resolved in config.py at import time and propagates automatically
  to the embedder, the declared dimension, the FAISS index paths, the
  checkpoint file and the evaluation output.  Preprocessing and cleaning are
  model-independent and are shared - never re-run per model.

TWO-PASS INGEST:
  Exactly ONE model is ingested per invocation (the key config resolved), and
  within that model exactly ONE representation is resident at a time:

      Pass 1  TEXT  -> embed -> add -> persist text.index + text_metadata.pkl
      (release)
      Pass 2  MICE  -> embed -> add -> persist mice.index + mice_metadata.pkl

  Holding one index instead of two halves peak resident memory - 7.26 GiB
  rather than 14.52 GiB at 2.5M x 768, and 9.68 vs 19.36 GiB at dim 1024.

ATOMIC CHECKPOINTING:
  - Tracks completed chunks (not rows) to avoid duplicates
  - Each pass keeps its own chunk set and resumes independently
  - persist_track() and save_progress() are adjacent statements, index first,
    so a crash can only leave the index AHEAD of the checkpoint - never behind.
    _reconcile() adopts such a chunk on the next run instead of re-embedding it
  - Checkpoints are per model: data/embeddings/<model>/ingestion_progress.json
  - Delete that file to restart that model's ingest from scratch
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shutil
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

import cleaning
import faiss_store
import preprocessing
from analysis import print_model_matrix, run_full_analysis
from config import (
    ACTIVE_MODEL_KEY,
    CHECKPOINT_INTERVAL,
    CLEANED_PATH,
    CSV_CHUNK_SIZE,
    DATA_DIR,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    EVAL_RESULTS_PATH,
    EVAL_SEED,
    MODEL_DIR,
    MODEL_REGISTRY,
    PER_QUERY_CSV_PATH,
    PREPROCESSED_PATH,
    PROGRESS_FILE,
    RAW_DATASET_PATH,
    DEFAULT_TOP_K,
    ensure_model_dir,
)
from embedder import embed_texts
from embedding_builder import (
    build_mice_representation,
    build_text_representation,
    iter_cleaned_chunks,
)
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

# On-disk shape (one file per model):
#
#   {"chunk_size": 100000,
#    "text": {"completed_chunks": [0, 1, 2], "rows": 300000, "done": false},
#    "mice": {"completed_chunks": [],        "rows": 0,      "done": false},
#    "completed": false,
#    "timestamp": "2026-08-28 14:03:11"}
#
# Each pass owns its chunk set, so TEXT can be finished while MICE has not
# started.  "completed" is derived (both passes done) and exists so a human -
# and the old tooling - can read completion at a glance.


def _blank_progress() -> dict:
    return {
        "chunk_size": CSV_CHUNK_SIZE,
        "text": {"completed_chunks": set(), "rows": 0, "done": False},
        "mice": {"completed_chunks": set(), "rows": 0, "done": False},
    }


def load_progress() -> dict:
    """
    Load this model's ingestion progress.

    Understands the pre-two-pass layout as well: that loop embedded TEXT and
    MICE inside the same chunk, so a legacy chunk set applies to both passes.
    """
    if not os.path.exists(PROGRESS_FILE):
        return _blank_progress()

    with open(PROGRESS_FILE, "r") as f:
        data = json.load(f)

    if "completed_chunks" in data:                      # legacy flat layout
        chunks = set(data.get("completed_chunks", []))
        rows   = int(data.get("total_rows_processed", 0))
        done   = bool(data.get("completed", False))
        return {
            "chunk_size": int(data.get("chunk_size", 10000)),
            "text": {"completed_chunks": set(chunks), "rows": rows, "done": done},
            "mice": {"completed_chunks": set(chunks), "rows": rows, "done": done},
        }

    progress = _blank_progress()
    progress["chunk_size"] = int(data.get("chunk_size", CSV_CHUNK_SIZE))
    for track in faiss_store.TRACKS:
        entry = data.get(track, {})
        progress[track] = {
            "completed_chunks": set(entry.get("completed_chunks", [])),
            "rows": int(entry.get("rows", 0)),
            "done": bool(entry.get("done", False)),
        }
    return progress


def save_progress(progress: dict) -> None:
    """Write the ACTIVE MODEL's checkpoint.  Sets are serialised as sorted lists."""
    ensure_model_dir()

    payload = {
        "chunk_size": progress["chunk_size"],
        "completed":  all(progress[t]["done"] for t in faiss_store.TRACKS),
        "timestamp":  time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    for track in faiss_store.TRACKS:
        state = progress[track]
        payload[track] = {
            "completed_chunks": sorted(state["completed_chunks"]),
            "rows":             state["rows"],
            "done":             state["done"],
        }

    with open(PROGRESS_FILE, "w") as f:
        json.dump(payload, f, indent=2)


def _check_chunk_size(progress: dict) -> None:
    """
    Refuse to resume a checkpoint written at a different CSV_CHUNK_SIZE.

    Chunk indices only mean something relative to the chunk size.  Resuming a
    10k-row checkpoint under a 100k-row config would treat chunk 5 as rows
    500k-600k when it actually held rows 50k-60k, skipping most of the corpus
    and duplicating the rest.  A fully completed checkpoint is exempt: nothing
    is left to resume, so the recorded size is only history.
    """
    recorded = progress.get("chunk_size", CSV_CHUNK_SIZE)
    if recorded == CSV_CHUNK_SIZE:
        return
    if all(progress[t]["done"] for t in faiss_store.TRACKS):
        return

    raise RuntimeError(
        f"Checkpoint for {ACTIVE_MODEL_KEY} was written with CSV_CHUNK_SIZE="
        f"{recorded}, but config.py now says {CSV_CHUNK_SIZE}. Chunk indices are "
        f"relative to the chunk size, so resuming would skip and duplicate rows. "
        f"Rebuild this model's index with:\n"
        f"    python main.py --model {ACTIVE_MODEL_KEY} --reset-progress"
    )


def clear_progress():
    """Delete the active model's progress file to restart from scratch."""
    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)
        log.info("Cleared ingestion progress for %s - will start from scratch",
                 ACTIVE_MODEL_KEY)


# ─────────────────────────────────────────────────────────────
# One-off migration of the pre-multi-model layout
# ─────────────────────────────────────────────────────────────

_LEGACY_FILES = [
    "text.index",
    "mice.index",
    "text_metadata.pkl",
    "mice_metadata.pkl",
    "ingestion_progress.json",
    "eval_results.json",
    # data/per_query_analysis.csv is deliberately NOT migrated: the Streamlit
    # dashboard hard-codes that path (loaders/data_loader.py), and the
    # frontend is out of scope.  New runs write a per-model copy alongside
    # the index instead.
]


def migrate_legacy_bge_base(dry_run: bool = False) -> None:
    """
    Move the original flat data/*.index + *.pkl artefacts into
    data/embeddings/bge_base/ so the existing, already-ingested baseline is
    reused instead of rebuilt.

    Refuses to overwrite: if a destination file already exists the migration
    aborts and nothing is moved.
    """
    dest = os.path.join(DATA_DIR, "embeddings", "bge_base")
    present = [f for f in _LEGACY_FILES if os.path.exists(os.path.join(DATA_DIR, f))]

    if not present:
        log.info("No legacy artefacts in %s - nothing to migrate.", DATA_DIR)
        return

    clashes = [f for f in present if os.path.exists(os.path.join(dest, f))]
    if clashes:
        raise RuntimeError(
            f"Refusing to migrate: {dest} already contains {clashes}. "
            f"Move or delete them first - this tool never overwrites indexes."
        )

    log.info("Migrating %d legacy artefact(s) → %s", len(present), dest)
    for name in present:
        src = os.path.join(DATA_DIR, name)
        log.info("  %s%s", name, "  (dry run)" if dry_run else "")
        if not dry_run:
            os.makedirs(dest, exist_ok=True)
            shutil.move(src, os.path.join(dest, name))
    if not dry_run:
        log.info("Migration complete.")


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
        log.info("Preprocessed CSV exists - skipping."); return
    log.info("Stage 2 - Preprocessing …")
    df = preprocessing.load_dataset(RAW_DATASET_PATH)
    records = [preprocessing.preprocess_row(r) for _, r in tqdm(df.iterrows(), total=len(df))]
    pd.DataFrame(records).to_csv(PREPROCESSED_PATH, index=False)
    log.info("Saved → %s", PREPROCESSED_PATH)


def _stage_clean() -> None:
    if os.path.exists(CLEANED_PATH):
        log.info("Cleaned CSV exists - skipping."); return
    log.info("Stage 3 - Cleaning …")
    cleaning.clean_preprocessed_dataset(PREPROCESSED_PATH, CLEANED_PATH)


def _count_rows(path: str) -> int:
    with open(path, encoding="utf-8", errors="replace") as fh:
        return sum(1 for _ in fh) - 1


# One representation per pass.  Only the column being embedded is built, so a
# pass never materialises the other representation's strings.
_REPR_BUILDERS = {
    "text": build_text_representation,
    "mice": build_mice_representation,
}


def _reconcile(track: str, state: dict, index_ntotal: int) -> None:
    """
    Repair the crash window between persist_track() and save_progress().

    Those two writes are adjacent but still two writes.  Because the index is
    written FIRST, the only reachable inconsistency is an index that is ahead
    of the checkpoint.  Chunks are processed in ascending order and each is
    fully added before its persist, so index.ntotal always lands on a chunk
    boundary: the extra rows are exactly the next chunk, and adopting it is
    cheaper and safer than embedding it a second time.
    """
    if index_ntotal == state["rows"]:
        return

    if index_ntotal < state["rows"]:
        raise RuntimeError(
            f"{track} index holds {index_ntotal} vectors but the checkpoint "
            f"claims {state['rows']} rows. The index is behind its checkpoint, "
            f"which the commit order cannot produce - the index file was "
            f"probably replaced or truncated. Rebuild with:\n"
            f"    python main.py --model {ACTIVE_MODEL_KEY} --reset-progress"
        )

    recovered  = index_ntotal - state["rows"]
    next_chunk = max(state["completed_chunks"]) + 1 if state["completed_chunks"] else 0
    state["completed_chunks"].add(next_chunk)
    state["rows"] = index_ntotal
    log.warning(
        "Pass %s: index held %d row(s) past the checkpoint - chunk %d was "
        "persisted before its progress write. Adopted it; not re-embedding.",
        track, recovered, next_chunk,
    )


def _ingest_pass(track: str, total_rows: int, progress: dict) -> None:
    """
    Embed and index ONE representation over the whole corpus.

    Only this track's index is resident; the other stays on disk.  The index
    and its metadata are released at the end so the next pass starts clean.
    """
    state = progress[track]

    if state["done"]:
        log.info("Pass %-4s already complete (%d rows) - skipping.",
                 track, state["rows"])
        return

    # mmap=False: this pass WRITES the index, and adding to a mapped index
    # would copy the whole buffer into owned memory, undoing the reservation.
    faiss_store.initialize_store(track, n_reserve=total_rows, mmap=False)
    index = faiss_store.text_index if track == "text" else faiss_store.mice_index
    _reconcile(track, state, index.ntotal)

    if state["completed_chunks"]:
        log.info("Pass %-4s RESUMING at %d / %d rows (%.1f%%).", track,
                 state["rows"], total_rows, 100 * state["rows"] / total_rows)
    else:
        log.info("Pass %-4s starting fresh.", track)

    build     = _REPR_BUILDERS[track]
    completed = state["completed_chunks"]

    with tqdm(total=total_rows, desc=f"Ingest {track}", unit="rows",
              initial=state["rows"]) as pbar:

        for chunk_idx, chunk in enumerate(iter_cleaned_chunks()):

            if chunk_idx in completed:
                continue

            n_rows = len(chunk)

            texts = chunk.apply(build, axis=1).tolist()
            rows  = chunk.to_dict(orient="records")
            faiss_store.insert_batch(track, rows, embed_texts(texts))
            del texts, rows

            # -- ATOMIC COMMIT -----------------------------------
            # Index first, checkpoint immediately after, nothing between.
            # This ordering makes the only possible crash state an index
            # ahead of its checkpoint, which _reconcile() repairs. Writing
            # the checkpoint first would instead lose rows silently.
            faiss_store.persist_track(track)
            completed.add(chunk_idx)
            state["rows"] += n_rows
            save_progress(progress)
            # ----------------------------------------------------

            pbar.update(n_rows)

            if state["rows"] % CHECKPOINT_INTERVAL < n_rows:
                log.info("  %s: %d / %d rows (%.1f%%)", track, state["rows"],
                         total_rows, 100 * state["rows"] / total_rows)

    state["done"] = True
    save_progress(progress)
    log.info("Pass %-4s complete: %d rows in %d chunks.",
             track, state["rows"], len(completed))

    faiss_store.release(track)


def _stage_ingest():
    """
    FAISS ingest for the ONE active model - TEXT pass, then MICE pass.

    Each pass walks the cleaned CSV independently and commits per chunk:
      1. Read chunk                       (CSV_CHUNK_SIZE rows)
      2. Build this pass's representation (the other is never built)
      3. Embed
      4. Add to that track's index
      5. persist_track()  +  save_progress()   <- adjacent, index first
    """
    progress = load_progress()
    _check_chunk_size(progress)

    if all(progress[t]["done"] for t in faiss_store.TRACKS):
        log.info("FAISS ingestion already completed for %s - skipping.",
                 ACTIVE_MODEL_KEY)
        return

    log.info("Stage 4 - FAISS ingest for %s (two-pass, per-chunk atomic commit) …",
             ACTIVE_MODEL_KEY)

    ensure_model_dir()

    # Count first: the row total pre-sizes the code buffer so each pass
    # allocates once instead of realloc-and-copying on every growth step
    # (see faiss_store.reserve_capacity).
    total_rows = _count_rows(CLEANED_PATH)
    log.info("Corpus: %d rows, %d per chunk (%d chunks per pass).",
             total_rows, CSV_CHUNK_SIZE, -(-total_rows // CSV_CHUNK_SIZE))

    for track in faiss_store.TRACKS:
        _ingest_pass(track, total_rows, progress)

    log.info("FAISS ingest complete for %s: %d rows per index.",
             ACTIVE_MODEL_KEY, progress["text"]["rows"])


def _stage_evaluate_and_analyse() -> None:
    log.info("Stage 5 - Evaluation …")

    _fix_seeds()

    # Memory-map both indexes (the default).  Evaluation only searches, so the
    # vectors stay on disk and the OS pages in what the queries touch instead
    # of resident-loading 14.52 GiB.
    faiss_store.initialize_stores()

    # Generate once; reuse for both evaluation and analysis
    test_cases = generate_test_cases(seed=EVAL_SEED)

    all_metrics = run_evaluation(
        test_cases=test_cases,
        seed=EVAL_SEED
    )

    print_summary(all_metrics)

    save_summary_json(all_metrics, EVAL_RESULTS_PATH)

    log.info("Stage 6 - Analysis …")

    run_full_analysis(
        test_cases=test_cases,
        csv_path=PER_QUERY_CSV_PATH,
        top_k=DEFAULT_TOP_K,
        print_n_examples=5,
    )

    # Cross-model table, populated by whichever models have been evaluated.
    print_model_matrix()

# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

def RAG_Pipeline(skip_ingest: bool = False) -> None:
    print("=" * 60)
    print("FM RAG PIPELINE - Research Edition")
    print(f"Model: {ACTIVE_MODEL_KEY}  ({EMBEDDING_MODEL}, dim {EMBEDDING_DIM})")
    print(f"Store: {MODEL_DIR}")
    print("Atomic Chunk-Based Checkpointing (Duplicate-Safe)")
    print("=" * 60)
    _fix_seeds()
    t0 = time.time()
    os.makedirs(DATA_DIR, exist_ok=True)
    ensure_model_dir()

    if not _check_dataset():
        return

    # Model-independent - shared by every model, never re-run per model.
    _stage_preprocess()
    _stage_clean()

    if not skip_ingest:
        _stage_ingest()
    _stage_evaluate_and_analyse()

    log.info("Done in %.1f s.", time.time() - t0)


def _cli() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="FM RAG pipeline - MICE across embedding models."
    )
    ap.add_argument(
        "--model", default=ACTIVE_MODEL_KEY, choices=list(MODEL_REGISTRY),
        help="embedding model key (resolved by config.py at import time)",
    )
    ap.add_argument("--reset-progress", action="store_true",
                    help="delete this model's ingestion checkpoint first")
    ap.add_argument("--skip-ingest", action="store_true",
                    help="evaluate/analyse only; assume the index is built")
    ap.add_argument("--migrate-legacy", action="store_true",
                    help="move the flat data/*.index artefacts into "
                         "data/embeddings/bge_base/ (never overwrites)")
    ap.add_argument("--dry-run", action="store_true",
                    help="with --migrate-legacy: report moves without doing them")
    ap.add_argument("--compare", action="store_true",
                    help="print the MODEL × STRATEGY table + ΔMICE and exit")
    return ap.parse_args()


if __name__ == "__main__":
    args = _cli()

    if args.migrate_legacy:
        migrate_legacy_bge_base(dry_run=args.dry_run)
    elif args.compare:
        print_model_matrix()
    else:
        if args.reset_progress:
            clear_progress()
        RAG_Pipeline(skip_ingest=args.skip_ingest)
