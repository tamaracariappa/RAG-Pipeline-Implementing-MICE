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
import gc
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

def create_index(n_reserve: int = 0):
    """
    Build an empty IndexFlatIP, optionally pre-sized for *n_reserve* vectors.

    n_reserve = 0 (default) creates the index exactly as before - used by the
    read-only paths (evaluation, retrieval, the Streamlit loaders) that never
    grow the index and must not pay for a reservation they will not use.
    """
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    reserve_capacity(index, n_reserve)
    return index


def reserve_capacity(index, n_reserve: int) -> None:
    """
    Pre-size the flat code buffer to hold *n_reserve* vectors, then shrink the
    logical length back to whatever the index already contains.

    IndexFlatCodes stores vectors in one contiguous byte buffer.  Growing it
    incrementally makes MSVC's allocator realloc at a 1.5x growth factor and
    copy the whole buffer, so peak RSS momentarily reaches ~2.5x the final
    index size.  At 2.5M x 768 floats (7.26 GiB) that spike is fatal on a
    12 GiB machine.  Resizing up once and back down reserves the capacity in a
    single allocation: std::vector::resize() downwards never releases capacity,
    so the buffer keeps its room and add() never has to reallocate.

    Shrinking back to the CURRENT length rather than to 0 is what makes this
    safe on resume: a loaded index already holds ntotal vectors, and truncating
    its buffer to 0 would leave ntotal pointing at codes that no longer exist,
    silently corrupting the index.  For a fresh index ntotal is 0, so this is
    exactly the resize(n * code_size) -> resize(0) sequence.
    """
    if n_reserve <= 0:
        return

    kept = index.ntotal * index.code_size
    want = max(n_reserve * index.code_size, kept)

    index.codes.resize(want)
    index.codes.resize(kept)


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


def load_index(path, mmap: bool = True):
    """
    Read an index from disk.  mmap=True requests IO_FLAG_MMAP; mmap=False
    always performs a plain resident read.

    MEASURED CAVEAT - IO_FLAG_MMAP DOES NOT MAP A FLAT INDEX.
    faiss defines IO_FLAG_MMAP as IO_FLAG_SKIP_IVF_DATA | IO_FLAG_READ_ONLY:
    it controls how an IVF index's inverted lists are opened, and IndexFlat
    has no inverted lists.  On faiss-cpu 1.13.2 the flag is accepted and then
    ignored for IndexFlatIP - the whole file is still read into RAM.  Verified
    two ways on a 2150 MB index:

        resident read consumed 2113 MB;  IO_FLAG_MMAP consumed 2125 MB
        the .index file could be DELETED immediately after the mapped load,
        so no file handle or mapping was being held

    The flag is kept because it is free, harmless and correct-if-supported:
    results are bit-identical either way, and a future faiss that maps flat
    storage would make this path pay off with no code change.  But do NOT
    budget RAM as though evaluation streams from disk today - it does not.

    ponytail: mmap is a no-op for IndexFlatIP; the only way to search 2.5M
    vectors without holding the whole index is to shard it and search the
    shards in sequence, merging top-k.  That preserves exact IndexFlatIP
    cosine semantics (an exact search over a partition of the corpus, merged,
    equals an exact search over the whole), so it stays inside the
    experimental controls.  Do that if evaluation has to fit in RAM.

    mmap=False is REQUIRED by the ingest regardless: writing to an index that
    had been opened read-only would copy the buffer into owned memory and
    discard the single allocation reserve_capacity() just made.
    """
    index = faiss.read_index(path, faiss.IO_FLAG_MMAP) if mmap else faiss.read_index(path)
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

TRACKS = ("text", "mice")


def _paths(track: str):
    """(index_path, metadata_path) for a track."""
    if track == "text":
        return TEXT_INDEX_PATH, TEXT_METADATA_PATH
    if track == "mice":
        return MICE_INDEX_PATH, MICE_METADATA_PATH
    raise ValueError(f"Unknown track {track!r}. Choose 'text' or 'mice'.")


def initialize_store(track: str, n_reserve: int = 0, mmap: bool = True):
    """
    Load or create ONE track's index + metadata.

    The ingest runs one representation at a time, so only that track's index
    is ever resident; the other stays on disk.  At 2.5M x 768 that is 7.26 GiB
    held instead of 14.52 GiB.

    mmap=False + n_reserve>0 is the write path (ingest).  mmap=True with the
    default n_reserve=0 is the read path.  A load that requested mapping is
    never reserved: reserving is only meaningful for an index about to grow.
    """
    global text_index, mice_index, text_metadata, mice_metadata

    index_path, metadata_path = _paths(track)

    if os.path.exists(index_path):
        index = load_index(index_path, mmap=mmap)
        metadata = load_metadata(metadata_path)
        if not mmap:
            reserve_capacity(index, n_reserve)
    else:
        index = create_index(n_reserve)
        metadata = []

    if track == "text":
        text_index, text_metadata = index, metadata
    else:
        mice_index, mice_metadata = index, metadata


def initialize_stores(n_reserve: int = 0, mmap: bool = True):
    """
    Load or create BOTH stores.  Used by the read paths - evaluation,
    retrieval and the Streamlit loaders - which need to search either index.

    Defaults are the read-only profile: IO_FLAG_MMAP is requested and nothing
    is reserved.  Note that the flag does not actually map an IndexFlat - see
    load_index() - so evaluation still resident-loads both indexes.

    n_reserve: expected FINAL row count, threaded in from main._count_rows().
    Only meaningful together with mmap=False, i.e. on a write path.

    The ingest does NOT use this - it calls initialize_store() per track so
    only one index is resident at a time.
    """
    for track in TRACKS:
        initialize_store(track, n_reserve, mmap=mmap)


def release(track: str) -> None:
    """
    Drop a track's index and metadata and return the memory to the allocator.

    Called between ingest passes: the TEXT index and its 2.5M-record metadata
    list must be gone before the MICE pass starts allocating its own.
    """
    global text_index, mice_index, text_metadata, mice_metadata

    if track == "text":
        text_index, text_metadata = None, None
    elif track == "mice":
        mice_index, mice_metadata = None, None
    else:
        raise ValueError(f"Unknown track {track!r}. Choose 'text' or 'mice'.")

    gc.collect()


# -----------------------------
# INSERT
# -----------------------------

def insert_batch(track: str, rows, embeddings):
    """Add one chunk's vectors + metadata to a single track."""
    index = text_index if track == "text" else mice_index
    metadata = text_metadata if track == "text" else mice_metadata

    if index is None:
        raise RuntimeError(
            f"FAISS {track} index not initialized - call initialize_store({track!r}) first."
        )

    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    _assert_dim(embeddings.shape[1], f"{track.upper()} embedding batch")

    index.add(embeddings)
    metadata.extend(rows)


def insert_text_batch(rows, embeddings):
    insert_batch("text", rows, embeddings)


def insert_mice_batch(rows, embeddings):
    insert_batch("mice", rows, embeddings)


# -----------------------------
# SAVE
# -----------------------------

def persist_track(track: str) -> None:
    """
    Write ONE track's index + metadata to disk.

    This is the ingest's commit point: main.py calls it and then immediately
    records the chunk in the progress file, so the two artefacts can never
    drift by more than the crash window between those two adjacent statements.
    """
    index_path, metadata_path = _paths(track)
    index = text_index if track == "text" else mice_index
    metadata = text_metadata if track == "text" else mice_metadata

    save_index(index, index_path)
    save_metadata(metadata, metadata_path)


def persist():
    """Write both tracks.  Kept for callers that hold both stores."""
    for track in TRACKS:
        persist_track(track)


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