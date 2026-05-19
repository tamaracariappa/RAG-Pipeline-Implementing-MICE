"""
config.py - Central configuration for the FM RAG Pipeline.
All hyperparameters and paths live here; nothing else is hard-coded.
"""

import os

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
EVAL_RESULTS_PATH = os.path.join(DATA_DIR, "eval_results.json")

# ─────────────────────────────────────────────────────────────
# EMBEDDING
# ─────────────────────────────────────────────────────────────
EMBEDDING_MODEL      = "BAAI/bge-base-en-v1.5"
EMBEDDING_DIM        = 768       # bge-base output dimension
EMBED_BATCH_SIZE = 128           # Increased for GPU (was 64)
NORMALIZE_EMBEDDINGS = True      # L2-normalise → cosine == dot product

# ─────────────────────────────────────────────────────────────
# DATA LOADING  (controls peak RAM usage)
# ─────────────────────────────────────────────────────────────
CSV_CHUNK_SIZE = 10000          # rows per pandas read_csv chunk

# -----------------------------
# FAISS
# -----------------------------

TEXT_INDEX_PATH = os.path.join(DATA_DIR, "text.index")
MICE_INDEX_PATH = os.path.join(DATA_DIR, "mice.index")

TEXT_METADATA_PATH = os.path.join(DATA_DIR, "text_metadata.pkl")
MICE_METADATA_PATH = os.path.join(DATA_DIR, "mice_metadata.pkl")

# ─────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────
EVAL_SAMPLE_SIZE = 500          # number of auto-generated test queries
EVAL_TOP_K_LIST  = [1, 5, 10]  # k values for Recall@k and NDCG@k
EVAL_SEED        = 42


# -----------------------------
# RETRIEVAL
# -----------------------------

DEFAULT_TOP_K = 10

POST_FILTER_MULTIPLIER = 20

# -----------------------------
# CHECKPOINTING (for resume after power failure)
# -----------------------------

CHECKPOINT_INTERVAL = 50000       # Save progress every N rows
PROGRESS_FILE = os.path.join(DATA_DIR, "ingestion_progress.json")
