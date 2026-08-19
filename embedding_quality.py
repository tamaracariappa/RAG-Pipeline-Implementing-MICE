"""
embedding_quality.py - offline sanity checks over dumped embedding arrays.

Operates on .npy dumps, which are now model-scoped:
    data/embeddings/<model_key>/{text_only,metadata_embedded}.npy

    python embedding_quality.py --model bge_m3

No pipeline stage writes these dumps automatically; if they are absent the
checks are skipped rather than raising.
"""

import os

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from config import ACTIVE_MODEL_KEY, EMBEDDING_MODEL, MODEL_DIR


# -----------------------------
# 1. SANITY CHECK
# -----------------------------
def sanity_check(embedding_path):
    emb = np.load(embedding_path)

    print("\n" + "=" * 50)
    print(f"SANITY CHECK → {embedding_path}")
    print("=" * 50)

    print("Shape:", emb.shape)
    print("Dtype:", emb.dtype)

    norms = np.linalg.norm(emb, axis=1)
    print("Mean norm:", norms.mean())
    print("Std norm:", norms.std())

    print("Has NaN:", np.isnan(emb).any())
    print("Has Inf:", np.isinf(emb).any())


# -----------------------------
# 2. RETRIEVAL TEST
# -----------------------------
def retrieval_test(csv_path, embedding_path, query_idx=100, top_k=5):
    df = pd.read_csv(csv_path)
    emb = np.load(embedding_path)

    query_vec = emb[query_idx].reshape(1, -1)

    sims = cosine_similarity(query_vec, emb)[0]
    top_indices = sims.argsort()[-top_k-1:][::-1]

    print("\n" + "=" * 50)
    print(f"RETRIEVAL TEST → {csv_path}")
    print("=" * 50)

    print("\nQUERY:")
    print(df.iloc[query_idx]["text"])

    print("\nTOP MATCHES:")
    for i in top_indices:
        print(f"\nScore: {sims[i]:.4f}")
        print(df.iloc[i]["text"][:200])


# -----------------------------
# 3. AVERAGE SIMILARITY
# -----------------------------
def average_similarity(embedding_path, sample_size=1000):
    emb = np.load(embedding_path)

    emb_sample = emb[:sample_size]
    sims = cosine_similarity(emb_sample, emb_sample)

    print("\n" + "=" * 50)
    print(f"AVERAGE SIMILARITY → {embedding_path}")
    print("=" * 50)

    print("Average similarity:", sims.mean())


# -----------------------------
# MAIN RUNNER
# -----------------------------
if __name__ == "__main__":

    print(f"Model: {ACTIVE_MODEL_KEY} ({EMBEDDING_MODEL})")
    print(f"Looking in: {MODEL_DIR}")

    for stem in ("text_only", "metadata_embedded"):
        npy = os.path.join(MODEL_DIR, f"{stem}.npy")
        csv = os.path.join(MODEL_DIR, f"{stem}.csv")

        if not os.path.exists(npy):
            print(f"\nSkipping {stem}: {npy} not found.")
            continue

        sanity_check(npy)
        if os.path.exists(csv):
            retrieval_test(csv, npy)
        average_similarity(npy)