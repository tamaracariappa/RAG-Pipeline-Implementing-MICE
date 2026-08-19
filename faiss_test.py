"""
faiss_test.py - five-document sanity check of the active embedding model.

    python faiss_test.py --model bge_m3

Uses a throwaway in-memory index; the real indexes are never touched.
"""

import numpy as np
import faiss

from config import ACTIVE_MODEL_KEY, EMBEDDING_DIM, EMBEDDING_MODEL
from embedder import embed_texts, embed_query

print(f"Model: {ACTIVE_MODEL_KEY} ({EMBEDDING_MODEL}, declared dim {EMBEDDING_DIM})")

# -----------------------------------
# SAMPLE DOCUMENTS
# -----------------------------------

documents = [
    "research building hvac maintenance work order for cooling failure",
    "electrical panel inspection in teaching facility",
    "plumbing leak repair in student dormitory",
    "fire protection system inspection in laboratory",
    "roof repair after water damage"
]

# -----------------------------------
# EMBED DOCUMENTS
# -----------------------------------

print("Generating embeddings...")

doc_embeddings = embed_texts(documents)

print("Shape:", doc_embeddings.shape)

# -----------------------------------
# CREATE FAISS INDEX
# -----------------------------------

dim = doc_embeddings.shape[1]

assert dim == EMBEDDING_DIM, (
    f"Model emitted dim {dim}, registry declares {EMBEDDING_DIM} "
    f"for {ACTIVE_MODEL_KEY!r}"
)

index = faiss.IndexFlatIP(dim)

index.add(doc_embeddings.astype(np.float32))

print("FAISS index size:", index.ntotal)

# -----------------------------------
# TEST QUERY
# -----------------------------------

query = "hvac cooling issue in research facility"

query_embedding = embed_query(query)

scores, indices = index.search(
    np.array([query_embedding]).astype(np.float32),
    3
)

# -----------------------------------
# RESULTS
# -----------------------------------

print("\nQUERY:")
print(query)

print("\nTOP RESULTS:")

for score, idx in zip(scores[0], indices[0]):

    print("\nScore:", round(float(score), 4))
    print(documents[idx])