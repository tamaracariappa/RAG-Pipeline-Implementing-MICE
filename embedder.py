"""
embedder.py - Singleton embedding model for the active registry entry.

Public interface is unchanged - the rest of the application never learns
which embedding model is active:

    embed_texts(texts, show_progress=False) -> (n, dim) float32
    embed_query(query)                      -> (dim,)  float32

All model-specific behaviour is data, not code: it comes from
config.MODEL_SPEC (see the MODEL_REGISTRY docstring in config.py).  Each
model's OFFICIAL retrieval encoding procedure is applied:

  bge_base     query-side instruction prefix ("Represent this sentence …"),
               documents encoded raw.  Unchanged from the original baseline.
  bge_m3       no instruction on either side.
  jina_v3      task LoRA adapters: retrieval.query / retrieval.passage.

The document text itself (TEXT / MICE representations) is identical for
every model - only the encoding procedure differs.
"""

from __future__ import annotations

from typing import List
import os
import threading

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from config import (
    ACTIVE_MODEL_KEY,
    EMBEDDING_MODEL,
    EMBEDDING_DIM,
    EMBED_BATCH_SIZE,
    MODEL_SPEC,
    NORMALIZE_EMBEDDINGS,
)

# ─────────────────────────────────────────────────────────────
# Singleton
# ─────────────────────────────────────────────────────────────
_model: SentenceTransformer | None = None
_model_lock = threading.Lock()


def _device() -> str:
    """
    Encoding device.  Not an experimental variable - embeddings are the same
    representation either way; the device only changes throughput.

    MICE_DEVICE overrides auto-detection:
        auto (default)  CUDA when available, otherwise CPU
        cpu             force CPU
        cuda / cuda:N   force GPU; raises if torch reports no CUDA device,
                        rather than silently degrading to CPU
    """
    want = os.environ.get("MICE_DEVICE", "auto").strip().lower() or "auto"

    if want == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    if want.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"MICE_DEVICE={want!r} requested but torch reports no CUDA device "
            f"(torch {torch.__version__}). This is a CPU-only torch build "
            f"and/or there is no NVIDIA GPU. Install a CUDA build "
            f"(https://pytorch.org/get-started/locally/) or use --device cpu."
        )

    return want


def _batch_size() -> int:
    """Encode batch size.  MICE_BATCH_SIZE overrides the registry value.
    A memory/throughput knob only - it does not change the representation,
    so it is not an experimental variable."""
    override = os.environ.get("MICE_BATCH_SIZE", "").strip()
    if override:
        return int(override)
    return EMBED_BATCH_SIZE


def get_model() -> SentenceTransformer:
    """
    Thread-safe singleton model loader.

    Prevents Streamlit reruns from trying to initialize the
    SentenceTransformer multiple times simultaneously.

    Raises:
        RuntimeError: if the model's actual output dimension disagrees with
                      the dimension declared in MODEL_REGISTRY.
    """
    global _model

    if _model is not None:
        return _model

    with _model_lock:

        # another thread may have loaded it already
        if _model is not None:
            return _model

        device = _device()
        print(f"[Embedder] Loading model: {EMBEDDING_MODEL} "
              f"(key={ACTIVE_MODEL_KEY}, device={device})")

        model = SentenceTransformer(
            EMBEDDING_MODEL,
            device=device,
            trust_remote_code=MODEL_SPEC["trust_remote_code"],
        )

        actual_dim = model.get_sentence_embedding_dimension()

        if actual_dim != EMBEDDING_DIM:
            raise RuntimeError(
                f"Model {EMBEDDING_MODEL} produced dim {actual_dim} != "
                f"registry dim {EMBEDDING_DIM} for key {ACTIVE_MODEL_KEY!r}. "
                f"Fix MODEL_REGISTRY rather than truncating or padding."
            )

        print(f"[Embedder] Ready. Embedding dim: {actual_dim}")

        _model = model

    return _model


# ─────────────────────────────────────────────────────────────
# Internal encoding
# ─────────────────────────────────────────────────────────────

def _prepare(texts: List[str], prefix: str) -> List[str]:
    """Apply the model's literal string decoration (instruction prefix)."""
    if prefix:
        texts = [prefix + t for t in texts]
    return texts


def _encode(texts: List[str], *, is_query: bool, show_progress: bool) -> np.ndarray:
    model = get_model()

    prefix = MODEL_SPEC["query_prefix"] if is_query else MODEL_SPEC["doc_prefix"]
    kwargs = MODEL_SPEC["query_encode_kwargs"] if is_query else MODEL_SPEC["doc_encode_kwargs"]

    prepared = _prepare(list(texts), prefix)

    embeddings = model.encode(
        prepared,
        batch_size=_batch_size(),
        show_progress_bar=show_progress,
        normalize_embeddings=NORMALIZE_EMBEDDINGS,
        convert_to_numpy=True,
        **kwargs,
    )
    return np.asarray(embeddings, dtype=np.float32)


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────

def embed_texts(texts: List[str], show_progress: bool = False) -> np.ndarray:
    """
    Encode a list of *document* strings using the active model's official
    document/passage procedure.

    Uses EMBED_BATCH_SIZE chunks to keep GPU/CPU memory bounded.

    Args:
        texts:         Arbitrary-length list of strings.
        show_progress: Display tqdm bar (useful during ingest).

    Returns:
        float32 ndarray of shape (len(texts), EMBEDDING_DIM).
        Vectors are L2-normalised when NORMALIZE_EMBEDDINGS is True.
    """
    return _encode(texts, is_query=False, show_progress=show_progress)


def embed_query(query: str) -> np.ndarray:
    """
    Encode a single *query* string using the active model's official query
    procedure (instruction prefix / task adapter).

    Returns:
        float32 ndarray of shape (EMBEDDING_DIM,).
    """
    return _encode([query], is_query=True, show_progress=False)[0]
