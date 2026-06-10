"""
embedder.py - Singleton embedding model (BAAI/bge-base-en-v1.5).

Responsibilities:
  - Load the model once and reuse it (singleton pattern).
  - Batch-encode arbitrary-length text lists into float32 vectors.
  - Provide a query encoder that prepends the BGE instruction prefix.

BGE-v1.5 note:
  For asymmetric retrieval tasks, BGE recommends prepending an instruction
  to the *query* side only.  Document sides are encoded without a prefix.
"""

from __future__ import annotations

from typing import List
import threading

import numpy as np
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL, EMBEDDING_DIM, EMBED_BATCH_SIZE, NORMALIZE_EMBEDDINGS

# ─────────────────────────────────────────────────────────────
# Singleton
# ─────────────────────────────────────────────────────────────
_model: SentenceTransformer | None = None
_model_lock = threading.Lock()

# Instruction prefix recommended by BGE for retrieval queries
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def get_model() -> SentenceTransformer:
    """
    Thread-safe singleton model loader.

    Prevents Streamlit reruns from trying to initialize the
    SentenceTransformer multiple times simultaneously.
    """
    global _model

    if _model is not None:
        return _model

    with _model_lock:

        # another thread may have loaded it already
        if _model is not None:
            return _model

        print(f"[Embedder] Loading model: {EMBEDDING_MODEL}")

        model = SentenceTransformer(
            EMBEDDING_MODEL,
            device="cuda"
        )

        actual_dim = model.get_sentence_embedding_dimension()

        if actual_dim != EMBEDDING_DIM:
            raise RuntimeError(
                f"Model dim {actual_dim} ≠ "
                f"config EMBEDDING_DIM {EMBEDDING_DIM}"
            )

        print(f"[Embedder] Ready. Embedding dim: {actual_dim}")

        _model = model

    return _model


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────

def embed_texts(texts: List[str], show_progress: bool = False) -> np.ndarray:
    """
    Encode a list of *document* strings.

    No instruction prefix is applied (documents are encoded as-is).
    Uses EMBED_BATCH_SIZE chunks to keep GPU/CPU memory bounded.

    Args:
        texts:         Arbitrary-length list of strings.
        show_progress: Display tqdm bar (useful during ingest).

    Returns:
        float32 ndarray of shape (len(texts), EMBEDDING_DIM).
        Vectors are L2-normalised when NORMALIZE_EMBEDDINGS is True.
    """
    model = get_model()
    embeddings = model.encode(
        texts,
        batch_size=EMBED_BATCH_SIZE,
        show_progress_bar=show_progress,
        normalize_embeddings=NORMALIZE_EMBEDDINGS,
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)


def embed_query(query: str) -> np.ndarray:
    """
    Encode a single *query* string with the BGE instruction prefix.

    Returns:
        float32 ndarray of shape (EMBEDDING_DIM,).
    """
    prefixed = _BGE_QUERY_PREFIX + query
    return embed_texts([prefixed])[0]
