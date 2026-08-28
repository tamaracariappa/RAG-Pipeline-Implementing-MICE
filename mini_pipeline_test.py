"""
mini_pipeline_test.py - small-sample smoke test for one embedding model.

    python mini_pipeline_test.py --model bge_base
    python mini_pipeline_test.py --model bge_m3
    python mini_pipeline_test.py --model jina_v3

GPU:
    --device cuda      force GPU; fails loudly if torch reports no CUDA device
    --device cuda:1    pick a specific GPU
    --device cpu       force CPU
    --device auto      (default) GPU when available, else CPU

    --batch-size N     override the registry batch size.  Purely a memory knob
                       - it does not change the representation.  A GPU takes a
                       far larger batch than the CPU defaults.

FAISS stays on the CPU: the project pins faiss-cpu, and the index type is an
experimental control.  --device moves the encoder only.

Verifies, on a small sample only:
  - the model loads
  - the TEXT representation is unchanged
  - the MICE representation is unchanged
  - document + query embeddings are generated
  - dimensions match: registry == model output == FAISS index
  - embeddings are L2-normalised and finite (no NaN / Inf)
  - a temporary FAISS IndexFlatIP works
  - strategies A / B / C all retrieve
  - the model-specific storage paths resolve and are writable
  - the encoder actually landed on the requested device

It NEVER touches the real indexes: FAISS state is built in memory, and the
path check writes into a throwaway sub-directory that is removed afterwards.
"""

from __future__ import annotations

import argparse
import os
import pickle
import shutil
import sys
import time

import faiss
import numpy as np
import pandas as pd
import torch

import faiss_store

from config import (
    ACTIVE_MODEL_KEY,
    CLEANED_PATH,
    EMBED_BATCH_SIZE,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    MODEL_DIR,
    MODEL_REGISTRY,
    NORMALIZE_EMBEDDINGS,
)
from embedder import embed_query, embed_texts, get_model
from embedding_builder import (
    add_text_columns,
    build_mice_representation,
    build_text_representation,
)
from retrieval import FilterConfig, strategy_a, strategy_b, strategy_c

# Expected representations for a synthetic row.  These are the CONTROL:
# the TEXT and MICE templates must be byte-identical for every model.
_FIXTURE_ROW = pd.Series({
    "BuildingID":    "A050",
    "BuildingName":  "Chemistry",
    "Type":          "Research",
    "equipment":     "HVAC",
    "WOStartDate":   "2019-03-14",
    "WOEndDate":     "2019-03-16",
    "WODescription": "Replace failed chilled water pump",
})
_EXPECTED_TEXT = "Chemistry | Research | Replace failed chilled water pump"
_EXPECTED_MICE = (
    "building id: a050. building name: chemistry. facility type: research. "
    "equipment system: hvac. work period: 2019-03-14 to 2019-03-16. "
    "work order description: replace failed chilled water pump."
)

_failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{'  - ' + detail if detail else ''}")
    if not ok:
        _failures.append(name)


def _describe_gpus() -> str:
    if not torch.cuda.is_available():
        return (f"no CUDA device visible to torch {torch.__version__} "
                f"({'CPU-only build' if '+cpu' in torch.__version__ else 'no GPU / driver'})")
    parts = []
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        parts.append(f"cuda:{i} {p.name} {p.total_memory / 1e9:.1f} GB")
    return " | ".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=ACTIVE_MODEL_KEY, choices=list(MODEL_REGISTRY),
                    help="embedding model key (resolved by config.py at import)")
    ap.add_argument("--n", type=int, default=200, help="sample rows (default 200)")
    ap.add_argument("--device", default="auto",
                    help="auto (default) | cpu | cuda | cuda:N. "
                         "'cuda' fails loudly rather than falling back to CPU.")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="override the registry encode batch size (memory knob "
                         "only; does not change the representation)")
    args = ap.parse_args()

    if args.model != ACTIVE_MODEL_KEY:
        # config resolves --model at import time; disagreement means the flag
        # was shadowed by MICE_EMBEDDING_MODEL in the environment.
        print(f"ERROR: --model {args.model!r} but config resolved "
              f"{ACTIVE_MODEL_KEY!r}. Unset MICE_EMBEDDING_MODEL.", file=sys.stderr)
        return 2

    # Must be set before the first get_model() call - the embedder reads these
    # when it lazily builds its singleton.
    os.environ["MICE_DEVICE"] = args.device
    if args.batch_size:
        os.environ["MICE_BATCH_SIZE"] = str(args.batch_size)

    batch = args.batch_size or EMBED_BATCH_SIZE

    print("=" * 70)
    print(f"MINI PIPELINE TEST - {ACTIVE_MODEL_KEY} ({EMBEDDING_MODEL})")
    print(f"declared dim {EMBEDDING_DIM} | sample {args.n} rows | batch {batch}")
    print(f"requested device: {args.device}")
    print(f"torch sees: {_describe_gpus()}")
    print("=" * 70)

    # ── 1. Model loads ────────────────────────────────────────
    print("\n1. Model load")
    try:
        model = get_model()
    except RuntimeError as exc:
        # The embedder refuses to silently downgrade cuda -> cpu.
        print(f"  [FAIL] model loads on requested device\n    {exc}")
        return 1

    actual_dim = model.get_sentence_embedding_dimension()
    check("model loads", model is not None)
    check("model dim == registry dim", actual_dim == EMBEDDING_DIM,
          f"{actual_dim} vs {EMBEDDING_DIM}")

    dev = str(next(model.parameters()).device)
    check("encoder is on the requested device",
          args.device == "auto" or dev.split(":")[0] == args.device.split(":")[0],
          f"weights on {dev}, requested {args.device}")
    if dev.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    # ── 2. Representations unchanged ──────────────────────────
    print("\n2. Representations (model-independent control)")
    check("TEXT representation unchanged",
          build_text_representation(_FIXTURE_ROW) == _EXPECTED_TEXT,
          build_text_representation(_FIXTURE_ROW))
    check("MICE representation unchanged",
          build_mice_representation(_FIXTURE_ROW) == _EXPECTED_MICE,
          build_mice_representation(_FIXTURE_ROW))

    # ── 3. Sample + embeddings ────────────────────────────────
    print(f"\n3. Embedding {args.n} sample rows")
    df = pd.read_csv(CLEANED_PATH, dtype=str, nrows=args.n).fillna("")
    df = add_text_columns(df)
    rows = df.to_dict(orient="records")

    t0 = time.perf_counter()
    text_embs = embed_texts(df["text_repr"].tolist(), show_progress=True)
    mice_embs = embed_texts(df["mice_repr"].tolist(), show_progress=True)
    elapsed = time.perf_counter() - t0
    q_vec = embed_query("hvac cooling failure in research building")

    n_texts = 2 * len(df)
    rate = n_texts / elapsed if elapsed else 0.0
    print(f"  throughput: {n_texts} texts in {elapsed:.1f}s = {rate:.1f} texts/s")
    if rate:
        # 2 representations x 2,536,923 rows for a full ingest of this model.
        print(f"  => full corpus (5,073,846 texts) would take "
              f"~{5_073_846 / rate / 3600:.1f} h at this rate")
    if dev.startswith("cuda"):
        print(f"  peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB "
              f"allocated, {torch.cuda.max_memory_reserved() / 1e9:.2f} GB reserved")

    check("document embeddings generated", text_embs.shape[0] == len(df),
          str(text_embs.shape))
    check("MICE embeddings generated", mice_embs.shape[0] == len(df),
          str(mice_embs.shape))
    check("query embedding generated", q_vec.shape == (EMBEDDING_DIM,),
          str(q_vec.shape))

    for label, arr in (("TEXT", text_embs), ("MICE", mice_embs)):
        check(f"{label} dim correct", arr.shape[1] == EMBEDDING_DIM,
              f"{arr.shape[1]} vs {EMBEDDING_DIM}")
        check(f"{label} dtype float32", arr.dtype == np.float32, str(arr.dtype))
        check(f"{label} finite (no NaN/Inf)", bool(np.isfinite(arr).all()))
        norms = np.linalg.norm(arr, axis=1)
        if NORMALIZE_EMBEDDINGS:
            check(f"{label} L2-normalised",
                  bool(np.allclose(norms, 1.0, atol=1e-3)),
                  f"min {norms.min():.6f} max {norms.max():.6f}")

    check("query finite (no NaN/Inf)", bool(np.isfinite(q_vec).all()))
    if NORMALIZE_EMBEDDINGS:
        qn = float(np.linalg.norm(q_vec))
        check("query L2-normalised", abs(qn - 1.0) < 1e-3, f"norm {qn:.6f}")

    # ── 4. Temporary in-memory FAISS index ────────────────────
    print("\n4. Temporary FAISS IndexFlatIP (in memory - real indexes untouched)")
    text_index = faiss.IndexFlatIP(EMBEDDING_DIM)
    mice_index = faiss.IndexFlatIP(EMBEDDING_DIM)
    text_index.add(np.ascontiguousarray(text_embs, dtype=np.float32))
    mice_index.add(np.ascontiguousarray(mice_embs, dtype=np.float32))

    check("FAISS index dim == registry dim", text_index.d == EMBEDDING_DIM,
          f"{text_index.d} vs {EMBEDDING_DIM}")
    check("all rows indexed", text_index.ntotal == len(df) == mice_index.ntotal,
          f"text {text_index.ntotal} mice {mice_index.ntotal}")

    # Point the retrieval layer at the temporary indexes.
    faiss_store.text_index    = text_index
    faiss_store.mice_index    = mice_index
    faiss_store.text_metadata = rows
    faiss_store.mice_metadata = rows

    # ── 5. Strategies A / B / C ───────────────────────────────
    print("\n5. Retrieval strategies")
    query = "hvac cooling failure in research building"
    fc = FilterConfig(equipment=df.iloc[0]["equipment"] or None)

    res_a = strategy_a(query, top_k=5)
    res_b = strategy_b(query, fc, top_k=5)
    res_c = strategy_c(query, top_k=5)

    check("Strategy A returns results", len(res_a) > 0, f"{len(res_a)} hits")
    check("Strategy B runs (filter may legitimately empty it)",
          isinstance(res_b, list), f"{len(res_b)} hits, filter={fc}")
    check("Strategy C returns results", len(res_c) > 0, f"{len(res_c)} hits")
    check("A scores are valid cosine values",
          all(-1.01 <= r.score <= 1.01 for r in res_a))
    check("C scores are valid cosine values",
          all(-1.01 <= r.score <= 1.01 for r in res_c))

    print("\n  Strategy A top-3:")
    for r in res_a[:3]:
        print(f"    {r.score:.4f}  {r.woid}  {r.equipment}  "
              f"{r.wo_description[:60]}")
    print("  Strategy C top-3:")
    for r in res_c[:3]:
        print(f"    {r.score:.4f}  {r.woid}  {r.equipment}  "
              f"{r.wo_description[:60]}")

    # ── 6. Model-specific storage paths ───────────────────────
    print("\n6. Model-specific storage paths")
    check("MODEL_DIR is model-scoped", MODEL_DIR.endswith(ACTIVE_MODEL_KEY), MODEL_DIR)

    tmp_dir = os.path.join(MODEL_DIR, "_mini_test_tmp")
    try:
        os.makedirs(tmp_dir, exist_ok=True)
        idx_path = os.path.join(tmp_dir, "text.index")
        meta_path = os.path.join(tmp_dir, "text_metadata.pkl")
        faiss.write_index(text_index, idx_path)
        with open(meta_path, "wb") as fh:
            pickle.dump(rows[:5], fh)
        reloaded = faiss.read_index(idx_path)
        check("index round-trips through the model directory",
              reloaded.d == EMBEDDING_DIM and reloaded.ntotal == text_index.ntotal,
              f"dim {reloaded.d} ntotal {reloaded.ntotal}")
        check("metadata round-trips", os.path.getsize(meta_path) > 0)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    check("temporary artefacts cleaned up", not os.path.exists(tmp_dir))

    # ── Summary ───────────────────────────────────────────────
    print("\n" + "=" * 70)
    if _failures:
        print(f"FAILED ({len(_failures)}): {', '.join(_failures)}")
        return 1
    print(f"ALL CHECKS PASSED - {ACTIVE_MODEL_KEY} (dim {EMBEDDING_DIM})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
