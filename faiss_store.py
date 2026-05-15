import faiss
import numpy as np
import pandas as pd
import pickle
import os

from config import (
    EMBEDDING_DIM,
    TEXT_INDEX_PATH,
    MICE_INDEX_PATH,
    TEXT_METADATA_PATH,
    MICE_METADATA_PATH,
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


# -----------------------------
# SAVE / LOAD
# -----------------------------

def save_index(index, path):
    faiss.write_index(index, path)


def load_index(path):
    return faiss.read_index(path)


def save_metadata(metadata, path):
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

    embeddings = embeddings.astype(np.float32)

    text_index.add(embeddings)

    for row in rows:
        text_metadata.append(row)


def insert_mice_batch(rows, embeddings):
    global mice_index
    global mice_metadata

    embeddings = embeddings.astype(np.float32)

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