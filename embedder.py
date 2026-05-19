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

import numpy as np
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL, EMBEDDING_DIM, EMBED_BATCH_SIZE, NORMALIZE_EMBEDDINGS

# ─────────────────────────────────────────────────────────────
# Singleton
# ─────────────────────────────────────────────────────────────
_model: SentenceTransformer | None = None

# Instruction prefix recommended by BGE for retrieval queries
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def get_model() -> SentenceTransformer:
    """Return the singleton SentenceTransformer, loading it on first call."""
    global _model
    if _model is None:
        print(f"[Embedder] Loading model: {EMBEDDING_MODEL}")
        _model = SentenceTransformer(EMBEDDING_MODEL)
        actual_dim = _model.get_sentence_embedding_dimension()
        if actual_dim != EMBEDDING_DIM:
            raise RuntimeError(
                f"Model dim {actual_dim} ≠ config EMBEDDING_DIM {EMBEDDING_DIM}. "
                "Update config.py."
            )
        print(f"[Embedder] Ready. Embedding dim: {actual_dim}")
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
