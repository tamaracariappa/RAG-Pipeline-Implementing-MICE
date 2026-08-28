"""
config.py - Central configuration for the FM RAG Pipeline.
All hyperparameters and paths live here; nothing else is hard-coded.

MULTI-MODEL EXPERIMENT
──────────────────────
The embedding model is the single experimental variable.  Everything else
(dataset, preprocessing, cleaning, TEXT/MICE representations, query set,
relevance judgments, top-k, FAISS index type, normalisation, retrieval
logic, metrics) is held constant across models.

The active model is resolved once, at import time, from - in priority order:
  1. environment variable  MICE_EMBEDDING_MODEL
  2. CLI flag              --model <key>   /   --model=<key>
  3. DEFAULT_MODEL_KEY

Resolving here (rather than in each entry point) means every module that
already does `from config import ...` becomes model-aware with no changes,
and processes that never pass --model (e.g. the Streamlit app) keep the
default model exactly as before.
"""

import os
import sys

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "data")

RAW_DATASET_FILENAME = (
    "Facility Management Unified Classification Database (FMUCD).csv"
)
RAW_DATASET_PATH  = os.path.join(DATA_DIR, RAW_DATASET_FILENAME)
PREPROCESSED_PATH = os.path.join(DATA_DIR, "preprocessed.csv")
CLEANED_PATH      = os.path.join(DATA_DIR, "preprocessed_clean.csv")

# Root for all model-specific artefacts (indexes, metadata, checkpoints,
# evaluation output).  One sub-directory per registry key.
EMBEDDINGS_ROOT = os.path.join(DATA_DIR, "embeddings")


# ─────────────────────────────────────────────────────────────
# MODEL REGISTRY
# ─────────────────────────────────────────────────────────────
# Per-model fields:
#   model_id            HuggingFace repo id
#   dim                 native embedding dimension - DECLARED here, VERIFIED
#                       at runtime in embedder.get_model() and again against
#                       the FAISS index in faiss_store; never trusted blindly
#   trust_remote_code   required by models shipping custom modelling code
#   query_prefix        string concatenated to the query before encoding
#   doc_prefix          string concatenated to the document before encoding
#   query_encode_kwargs extra kwargs passed to SentenceTransformer.encode()
#   doc_encode_kwargs   for the query / document side respectively
#   batch_size          encode batch size - a MEMORY knob only.  It does not
#                       alter the representation and is not a quality-tuning
#                       parameter; larger models simply do not fit at 128.
#
# Encoding procedures follow each model's OFFICIAL model card:
#   bge_base     query-side instruction prefix, documents raw.
#   bge_m3       no instruction on either side ("the BGE-M3 model no longer
#                requires adding instructions to the queries").
#   jina_v3      task-specific LoRA adapters: retrieval.query / retrieval.passage.

MODEL_REGISTRY = {
    "bge_base": {
        "model_id":            "BAAI/bge-base-en-v1.5",
        "dim":                 768,
        "trust_remote_code":   False,
        "query_prefix":        "Represent this sentence for searching relevant passages: ",
        "doc_prefix":          "",
        "query_encode_kwargs": {},
        "doc_encode_kwargs":   {},
        "batch_size":          128,
    },
    "bge_m3": {
        "model_id":            "BAAI/bge-m3",
        "dim":                 1024,
        "trust_remote_code":   False,
        "query_prefix":        "",
        "doc_prefix":          "",
        "query_encode_kwargs": {},
        "doc_encode_kwargs":   {},
        "batch_size":          64,
    },
    "jina_v3": {
        "model_id":            "jinaai/jina-embeddings-v3",
        "dim":                 1024,
        "trust_remote_code":   True,
        "query_prefix":        "",
        "doc_prefix":          "",
        "query_encode_kwargs": {"task": "retrieval.query"},
        "doc_encode_kwargs":   {"task": "retrieval.passage"},
        "batch_size":          64,
    },
}

DEFAULT_MODEL_KEY = "bge_base"


def _resolve_model_key() -> str:
    """Resolve the active model key from env var, then CLI flag, then default."""
    key = os.environ.get("MICE_EMBEDDING_MODEL")

    if not key:
        argv = sys.argv[1:]
        for i, arg in enumerate(argv):
            if arg == "--model" and i + 1 < len(argv):
                key = argv[i + 1]
                break
            if arg.startswith("--model="):
                key = arg.split("=", 1)[1]
                break

    key = key or DEFAULT_MODEL_KEY

    if key not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown embedding model key {key!r}. "
            f"Choose one of: {', '.join(MODEL_REGISTRY)}"
        )

    # Propagate so late imports / subprocesses agree on the same model.
    os.environ["MICE_EMBEDDING_MODEL"] = key
    return key


ACTIVE_MODEL_KEY = _resolve_model_key()
MODEL_SPEC       = MODEL_REGISTRY[ACTIVE_MODEL_KEY]

# ─────────────────────────────────────────────────────────────
# EMBEDDING
# ─────────────────────────────────────────────────────────────
EMBEDDING_MODEL      = MODEL_SPEC["model_id"]
EMBEDDING_DIM        = MODEL_SPEC["dim"]   # cross-checked at runtime
EMBED_BATCH_SIZE     = MODEL_SPEC["batch_size"]
NORMALIZE_EMBEDDINGS = True      # L2-normalise → cosine == dot product

# ─────────────────────────────────────────────────────────────
# DATA LOADING  (controls peak RAM usage)
# ─────────────────────────────────────────────────────────────
# Rows per pandas read_csv chunk.  Also the ingest's commit granularity: each
# chunk is embedded, added and persisted as one unit, so a larger chunk means
# proportionally fewer full-index rewrites (the ingest's dominant disk cost).
# Changing this invalidates a PARTIAL ingestion_progress.json, because chunk
# indices are recorded against it - main.py detects the mismatch and refuses
# rather than silently mixing two chunk sizes.
CSV_CHUNK_SIZE = 100000

# -----------------------------
# FAISS  (model-specific - dimensions differ, indexes must never be shared)
# -----------------------------

MODEL_DIR = os.path.join(EMBEDDINGS_ROOT, ACTIVE_MODEL_KEY)

TEXT_INDEX_PATH = os.path.join(MODEL_DIR, "text.index")
MICE_INDEX_PATH = os.path.join(MODEL_DIR, "mice.index")

TEXT_METADATA_PATH = os.path.join(MODEL_DIR, "text_metadata.pkl")
MICE_METADATA_PATH = os.path.join(MODEL_DIR, "mice_metadata.pkl")

# ─────────────────────────────────────────────────────────────
# EVALUATION  (model-specific output; identical query set + judgments)
# ─────────────────────────────────────────────────────────────
EVAL_RESULTS_PATH  = os.path.join(MODEL_DIR, "eval_results.json")
PER_QUERY_CSV_PATH = os.path.join(MODEL_DIR, "per_query_analysis.csv")

EVAL_SAMPLE_SIZE = 500          # number of auto-generated test queries
EVAL_TOP_K_LIST  = [1, 5, 10]  # k values for Recall@k and NDCG@k
EVAL_SEED        = 42

# Strategies compared in the experiment.  B' was removed: its implementation
# was byte-for-byte the same candidate-fetch + Python metadata filter as B,
# so it was a duplicate condition rather than a distinct pre-filter strategy.
STRATEGIES = ("A", "B", "C")


# -----------------------------
# RETRIEVAL
# -----------------------------

DEFAULT_TOP_K = 10

POST_FILTER_MULTIPLIER = 20

# -----------------------------
# CHECKPOINTING (for resume after power failure) - model-specific, so one
# model can never resume from, overwrite, or be confused with another's state
# -----------------------------

CHECKPOINT_INTERVAL = 50000       # Save progress every N rows
PROGRESS_FILE = os.path.join(MODEL_DIR, "ingestion_progress.json")


def ensure_model_dir() -> str:
    """Create (if needed) and return the active model's storage directory."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    return MODEL_DIR
