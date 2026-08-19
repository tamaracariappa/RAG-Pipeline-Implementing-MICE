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
  nv_embed_v2  instruction supplied via `prompt=` (so sentence-transformers
               excludes it from the pooled span), EOS appended to every
               input, tokenizer.padding_side="right"; passages carry no
               instruction.

The document text itself (TEXT / MICE representations) is identical for
every model - only the encoding procedure differs.
"""

from __future__ import annotations

from typing import List
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
    """CUDA when available, otherwise CPU.  Device is not an experimental
    variable and must not hard-fail on CPU-only environments."""
    return "cuda" if torch.cuda.is_available() else "cpu"


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

        if MODEL_SPEC["max_seq_length"]:
            model.max_seq_length = MODEL_SPEC["max_seq_length"]
        if MODEL_SPEC["padding_side"]:
            model.tokenizer.padding_side = MODEL_SPEC["padding_side"]

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

def _prepare(texts: List[str], prefix: str, append_eos: bool) -> List[str]:
    """Apply the model's literal string decorations (prefix / EOS token)."""
    if prefix:
        texts = [prefix + t for t in texts]
    if append_eos:
        eos = get_model().tokenizer.eos_token or ""
        if eos:
            texts = [t + eos for t in texts]
    return texts


def _encode(texts: List[str], *, is_query: bool, show_progress: bool) -> np.ndarray:
    model = get_model()

    prefix = MODEL_SPEC["query_prefix"] if is_query else MODEL_SPEC["doc_prefix"]
    kwargs = MODEL_SPEC["query_encode_kwargs"] if is_query else MODEL_SPEC["doc_encode_kwargs"]

    prepared = _prepare(list(texts), prefix, MODEL_SPEC["append_eos"])

    embeddings = model.encode(
        prepared,
        batch_size=EMBED_BATCH_SIZE,
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
    procedure (instruction prefix / task adapter / instruction prompt).

    Returns:
        float32 ndarray of shape (EMBEDDING_DIM,).
    """
    return _encode([query], is_query=True, show_progress=False)[0]
