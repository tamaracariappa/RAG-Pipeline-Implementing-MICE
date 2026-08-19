"""
faiss_store.py - FAISS IndexFlatIP storage, one index pair per embedding model.

Index type and normalisation are experimental controls and never vary:
  faiss.IndexFlatIP over L2-normalised vectors (inner product == cosine).

Only the dimension varies, because the embedding model varies.  Paths come
from config and are already model-scoped (data/embeddings/<model_key>/…), so
one model can never read, resume from, or overwrite another model's index.
Every load and every insert re-validates:

    registry dim == embedding dim == FAISS index dim
"""

import faiss
import numpy as np
import pickle
import os

from config import (
    ACTIVE_MODEL_KEY,
    EMBEDDING_DIM,
    TEXT_INDEX_PATH,
    MICE_INDEX_PATH,
    TEXT_METADATA_PATH,
    MICE_METADATA_PATH,
    ensure_model_dir,
)

# -----------------------------
# GLOBALS
# -----------------------------

text_index = None
mice_index = None

text_metadata = None
mice_metadata = None


# -----------------------------
# CREATE INDEX
# -----------------------------

def create_index():
    return faiss.IndexFlatIP(EMBEDDING_DIM)


def _assert_dim(actual: int, what: str) -> None:
    if actual != EMBEDDING_DIM:
        raise RuntimeError(
            f"Dimension mismatch for model {ACTIVE_MODEL_KEY!r}: {what} has "
            f"dim {actual}, registry declares {EMBEDDING_DIM}. Refusing to "
            f"truncate, pad or project. Check that the index directory "
            f"belongs to this model."
        )


# -----------------------------
# SAVE / LOAD
# -----------------------------

def save_index(index, path):
    ensure_model_dir()
    faiss.write_index(index, path)


def load_index(path):
    index = faiss.read_index(path)
    _assert_dim(index.d, os.path.basename(path))
    return index


def save_metadata(metadata, path):
    ensure_model_dir()
    with open(path, "wb") as f:
        pickle.dump(metadata, f)


def load_metadata(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# -----------------------------
# INITIALIZE
# -----------------------------

def initialize_stores():
    global text_index
    global mice_index
    global text_metadata
    global mice_metadata

    if os.path.exists(TEXT_INDEX_PATH):
        text_index = load_index(TEXT_INDEX_PATH)
        text_metadata = load_metadata(TEXT_METADATA_PATH)

    else:
        text_index = create_index()
        text_metadata = []

    if os.path.exists(MICE_INDEX_PATH):
        mice_index = load_index(MICE_INDEX_PATH)
        mice_metadata = load_metadata(MICE_METADATA_PATH)

    else:
        mice_index = create_index()
        mice_metadata = []


# -----------------------------
# INSERT
# -----------------------------

def insert_text_batch(rows, embeddings):
    global text_index
    global text_metadata

    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    _assert_dim(embeddings.shape[1], "TEXT embedding batch")

    text_index.add(embeddings)

    for row in rows:
        text_metadata.append(row)


def insert_mice_batch(rows, embeddings):
    global mice_index
    global mice_metadata

    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    _assert_dim(embeddings.shape[1], "MICE embedding batch")

    mice_index.add(embeddings)

    for row in rows:
        mice_metadata.append(row)


# -----------------------------
# SAVE ALL
# -----------------------------

def persist():
    save_index(text_index, TEXT_INDEX_PATH)
    save_index(mice_index, MICE_INDEX_PATH)

    save_metadata(text_metadata, TEXT_METADATA_PATH)
    save_metadata(mice_metadata, MICE_METADATA_PATH)


# -----------------------------
# SEARCH
# -----------------------------

def search_text(query_vector, top_k):
    assert_initialized()
    scores, indices = text_index.search(
        np.array([query_vector]).astype(np.float32),
        top_k
    )

    return scores[0], indices[0], text_metadata


def search_mice(query_vector, top_k):
    assert_initialized()
    scores, indices = mice_index.search(
        np.array([query_vector]).astype(np.float32),
        top_k
    )

    return scores[0], indices[0], mice_metadata


# -----------------------------
# VALIDATION
# -----------------------------

def assert_initialized():

    if text_index is None:
        raise RuntimeError(
            "FAISS text index not initialized."
        )

    if mice_index is None:
        raise RuntimeError(
            "FAISS MICE index not initialized."
        )