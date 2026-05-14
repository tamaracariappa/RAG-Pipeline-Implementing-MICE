import pandas as pd

import faiss_store

from embedder import embed_texts
from embedding_builder import add_text_columns
from retrieval import strategy_a

# -----------------------------------
# LOAD SMALL SAMPLE
# -----------------------------------

print("Loading dataset sample...")

df = pd.read_csv(
    "data/preprocessed_clean.csv",
    nrows=1000
).fillna("")

# -----------------------------------
# BUILD REPRESENTATIONS
# -----------------------------------

print("Building text representations...")

df = add_text_columns(df)

rows = df.to_dict(orient="records")

# -----------------------------------
# INITIALIZE FAISS
# -----------------------------------

faiss_store.initialize_stores()

# -----------------------------------
# EMBEDDINGS
# -----------------------------------

print("Generating embeddings...")

text_embs = embed_texts(
    df["text_repr"].tolist(),
    show_progress=True
)

mice_embs = embed_texts(
    df["mice_repr"].tolist(),
    show_progress=True
)

# -----------------------------------
# INSERT
# -----------------------------------

print("Inserting into FAISS...")

faiss_store.insert_text_batch(
    rows,
    text_embs
)

faiss_store.insert_mice_batch(
    rows,
    mice_embs
)

# -----------------------------------
# SAVE
# -----------------------------------

print("Persisting indexes...")

faiss_store.persist()

# -----------------------------------
# TEST RETRIEVAL
# -----------------------------------

query = "hvac cooling failure in research building"

print("\nQUERY:")
print(query)

results = strategy_a(query, top_k=5)

print("\nTOP RESULTS:\n")

for r in results:

    print(f"Score: {r.score:.4f}")
    print(f"WOID: {r.woid}")
    print(f"Type: {r.wo_type}")
    print(f"Equipment: {r.equipment}")
    print(f"Description: {r.wo_description[:150]}")
    print("-" * 60)

print("\nMini pipeline test successful.")