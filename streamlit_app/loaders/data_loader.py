"""
loaders/data_loader.py – Safe wrappers around existing project modules.

All imports are guarded so the Streamlit app loads even if FAISS indexes
or the cleaned CSV are not yet built (shows a friendly "not ready" message
instead of a hard crash).
"""

from __future__ import annotations

import os
import json
import pickle
import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

log = logging.getLogger(__name__)


# ── Project root (one level above streamlit_app/) ────────────
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)


def _project_path(*parts: str) -> str:
    return os.path.join(PROJECT_ROOT, *parts)


# ─────────────────────────────────────────────────────────────
# Config access
# ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_config() -> dict:
    """Return key config values as a plain dict (no import side-effects)."""
    try:
        import config as cfg  # noqa: F401
        return {
            "embedding_model":  cfg.EMBEDDING_MODEL,
            "embedding_dim":    cfg.EMBEDDING_DIM,
            "embed_batch_size": cfg.EMBED_BATCH_SIZE,
            "default_top_k":    cfg.DEFAULT_TOP_K,
            "eval_top_k_list":  cfg.EVAL_TOP_K_LIST,
            "cleaned_path":     cfg.CLEANED_PATH,
            "text_index_path":  cfg.TEXT_INDEX_PATH,
            "mice_index_path":  cfg.MICE_INDEX_PATH,
            "text_meta_path":   cfg.TEXT_METADATA_PATH,
            "mice_meta_path":   cfg.MICE_METADATA_PATH,
            "eval_results_path": cfg.EVAL_RESULTS_PATH,
        }
    except Exception as exc:
        log.warning("Could not load config.py: %s", exc)
        return {}


# ─────────────────────────────────────────────────────────────
# Cleaned CSV
# ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False, ttl=3600)
def load_cleaned_df(nrows: int = 5000) -> Optional[pd.DataFrame]:
    """
    Load the first *nrows* rows of the cleaned CSV.
    Returns None if the file doesn't exist yet.
    """
    cfg = load_config()
    path = cfg.get("cleaned_path", _project_path("data", "preprocessed_clean.csv"))
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, dtype=str, nrows=nrows).fillna("")
        return df
    except Exception as exc:
        log.error("Failed to load cleaned CSV: %s", exc)
        return None


# ─────────────────────────────────────────────────────────────
# FAISS metadata (pkl)
# ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_text_metadata() -> Optional[list]:
    cfg = load_config()
    path = cfg.get("text_meta_path",
                   _project_path("data", "text_metadata.pkl"))
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as fh:
            return pickle.load(fh)
    except Exception as exc:
        log.error("Failed to load text metadata: %s", exc)
        return None


@st.cache_resource(show_spinner=False)
def load_mice_metadata() -> Optional[list]:
    cfg = load_config()
    path = cfg.get("mice_meta_path",
                   _project_path("data", "mice_metadata.pkl"))
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as fh:
            return pickle.load(fh)
    except Exception as exc:
        log.error("Failed to load MICE metadata: %s", exc)
        return None


# ─────────────────────────────────────────────────────────────
# FAISS index vectors (sampled)
# ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _load_faiss_indexes():
    """Initialize FAISS stores once and cache the handle."""
    try:
        import faiss_store
        faiss_store.initialize_stores()
        return faiss_store
    except Exception as exc:
        log.error("FAISS store initialization failed: %s", exc)
        return None


def sample_text_vectors(n: int = 1000) -> Tuple[Optional[np.ndarray], Optional[list]]:
    """
    Return (vectors, metadata) for a random sample of *n* TEXT index entries.
    Vectors are reconstructed from the FAISS flat index.
    """
    import faiss_store as fs
    fs_mod = _load_faiss_indexes()
    if fs_mod is None or fs.text_index is None:
        return None, None

    total = fs.text_index.ntotal
    if total == 0:
        return None, None

    n = min(n, total)
    rng = np.random.default_rng(42)
    indices = rng.choice(total, size=n, replace=False).tolist()

    try:
        vecs = np.zeros((n, fs.text_index.d), dtype=np.float32)
        for i, idx in enumerate(indices):
            fs.text_index.reconstruct(int(idx), vecs[i])
        meta_sample = [fs.text_metadata[i] for i in indices]
        return vecs, meta_sample
    except Exception as exc:
        log.error("Vector reconstruction failed: %s", exc)
        return None, None


def sample_mice_vectors(n: int = 1000) -> Tuple[Optional[np.ndarray], Optional[list]]:
    """Same as sample_text_vectors but for the MICE index."""
    import faiss_store as fs
    fs_mod = _load_faiss_indexes()
    if fs_mod is None or fs.mice_index is None:
        return None, None

    total = fs.mice_index.ntotal
    if total == 0:
        return None, None

    n = min(n, total)
    rng = np.random.default_rng(42)
    indices = rng.choice(total, size=n, replace=False).tolist()

    try:
        vecs = np.zeros((n, fs.mice_index.d), dtype=np.float32)
        for i, idx in enumerate(indices):
            fs.mice_index.reconstruct(int(idx), vecs[i])
        meta_sample = [fs.mice_metadata[i] for i in indices]
        return vecs, meta_sample
    except Exception as exc:
        log.error("MICE vector reconstruction failed: %s", exc)
        return None, None


# ─────────────────────────────────────────────────────────────
# Evaluation results
# ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_eval_results() -> Optional[dict]:
    cfg = load_config()
    path = cfg.get("eval_results_path",
                   _project_path("data", "eval_results.json"))
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:
        log.error("Failed to load eval_results.json: %s", exc)
        return None


@st.cache_data(show_spinner=False)
def load_per_query_csv() -> Optional[pd.DataFrame]:
    path = _project_path("data", "per_query_analysis.csv")
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        log.error("Failed to load per_query_analysis.csv: %s", exc)
        return None


# ─────────────────────────────────────────────────────────────
# Index statistics
# ─────────────────────────────────────────────────────────────

def get_index_stats() -> dict:
    """Return basic stats about the loaded FAISS indexes."""
    try:
        import faiss_store as fs
        _load_faiss_indexes()
        return {
            "text_total": fs.text_index.ntotal if fs.text_index else 0,
            "mice_total": fs.mice_index.ntotal if fs.mice_index else 0,
            "dim":        fs.text_index.d      if fs.text_index else 0,
        }
    except Exception:
        return {"text_total": 0, "mice_total": 0, "dim": 0}
