#!/usr/bin/env python
"""
precompute_pca.py - one-off PCA precomputation for the evaluation dashboard.

The dashboard must never open a FAISS index: the indexes are hundreds of MB and
the app targets a 12 GB laptop. This script does the expensive work once,
offline, and writes a small static JSON that the app can read instantly.

Usage
-----
    python scripts/precompute_pca.py                     # all models
    python scripts/precompute_pca.py --models bge_m3     # one model
    python scripts/precompute_pca.py --sample 2000       # bigger sample
    python scripts/precompute_pca.py --force             # overwrite existing

Reads, per model, from data/embeddings/<model>/:
    text.index / text_metadata.pkl
    mice.index / mice_metadata.pkl

Writes:
    data/embeddings/<model>/pca_cache.json        2-D projection of the sample
    data/embeddings/<model>/vector_sample_50.json 50 rows with their raw vectors

The sample is drawn with a fixed seed, so re-running on an unchanged index
reproduces the same projection. Labels are stored as a category dictionary plus
integer codes rather than repeated strings, which keeps the file small enough to
parse in milliseconds.

The 50-row vector sample is a subset of the same draw, so it costs no extra
index pass: those vectors are already reconstructed for the PCA. It is what the
dashboard's Vector Inspector reads to show one work order's text beside the
float array it was embedded into.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EMBED_ROOT = PROJECT_ROOT / "data" / "embeddings"

MODELS = ("bge_base", "bge_m3", "jina_v3")
TRACKS = ("text", "mice")

# Metadata keys the dashboard offers as "colour by" dimensions.
LABEL_KEYS = ("equipment", "Type", "BuildingName")

SEED = 42
DESC_CHARS = 70

# Vector Inspector: how many rows to carry, and at what precision. The vectors
# are unit-normalised, so a 1024-d component sits around 0.03 and five decimals
# keep three significant figures - enough to plot, and it holds the file under
# a megabyte.
SAMPLE_ROWS = 50
VECTOR_DECIMALS = 5


def _encode_labels(values: list[str]) -> dict:
    """Store labels as {cats: [...], codes: [...]} instead of repeated strings."""
    categories: list[str] = []
    lookup: dict[str, int] = {}
    codes: list[int] = []
    for value in values:
        code = lookup.get(value)
        if code is None:
            code = len(categories)
            lookup[value] = code
            categories.append(value)
        codes.append(code)
    return {"cats": categories, "codes": codes}


def project_track(
    model: str, track: str, sample: int
) -> tuple[dict | None, dict | None, str | None]:
    """
    PCA one track of one model.

    Returns (pca_payload, inspector_sample, error_message). The second element
    is the SAMPLE_ROWS subset with its full float vectors, carved out of the
    same draw so the index is read once for both.
    """
    import faiss
    from sklearn.decomposition import PCA

    # The stored metadata rows are the raw cleaned-CSV records; neither track
    # keeps the string it embedded. Rebuilding both with the ingest's own
    # builders is what guarantees the inspector shows the exact text that
    # produced the vector next to it.
    sys.path.insert(0, str(PROJECT_ROOT))
    from embedding_builder import (
        build_mice_representation,
        build_text_representation,
    )

    index_path = EMBED_ROOT / model / f"{track}.index"
    meta_path = EMBED_ROOT / model / f"{track}_metadata.pkl"

    missing = [p.name for p in (index_path, meta_path) if not p.exists()]
    if missing:
        return None, None, f"missing {', '.join(missing)}"

    index = faiss.read_index(str(index_path))
    with open(meta_path, "rb") as fh:
        metadata = pickle.load(fh)

    total = int(index.ntotal)
    if total == 0 or not metadata:
        return None, None, "index or metadata is empty"

    usable = min(total, len(metadata))
    take = min(sample, usable)
    rng = np.random.default_rng(SEED)
    picks = sorted(rng.choice(usable, size=take, replace=False).tolist())

    vectors = np.zeros((take, index.d), dtype=np.float32)
    try:
        for row, idx in enumerate(picks):
            index.reconstruct(int(idx), vectors[row])
    except RuntimeError as exc:
        # Non-flat indexes need a direct map before reconstruct() works.
        return None, None, f"cannot reconstruct vectors ({exc}); a flat index is required"

    pca = PCA(n_components=2, random_state=SEED)
    coords = pca.fit_transform(vectors)

    rows = [metadata[i] for i in picks]

    # Inspector subset. Seeded off `take`, which is identical for both tracks,
    # so text row i and MICE row i are the same work order - the comparison the
    # inspector draws only means anything if they line up.
    sub = sorted(np.random.default_rng(SEED).choice(
        take, size=min(SAMPLE_ROWS, take), replace=False).tolist())
    inspector = {
        "dim": int(index.d),
        "woid": [str(rows[j].get("WOID") or "") for j in sub],
        "text": [build_text_representation(rows[j]) for j in sub],
        "mice": [build_mice_representation(rows[j]) for j in sub],
        "vectors": [[round(float(v), VECTOR_DECIMALS) for v in vectors[j]]
                    for j in sub],
    }
    payload = {
        "dim": int(index.d),
        "n_total": total,
        "n_sampled": take,
        "explained": [round(float(v), 6) for v in pca.explained_variance_ratio_],
        "x": [round(float(v), 4) for v in coords[:, 0]],
        "y": [round(float(v), 4) for v in coords[:, 1]],
        "labels": {
            key: _encode_labels([str(r.get(key) or "unknown") for r in rows])
            for key in LABEL_KEYS
        },
        "woid": [str(r.get("WOID") or "") for r in rows],
        "desc": [str(r.get("WODescription") or "")[:DESC_CHARS] for r in rows],
    }
    return payload, inspector, None


def _write(path: Path, document: dict, model: str) -> None:
    path.write_text(json.dumps(document, separators=(",", ":")), encoding="utf-8")
    print(f"  {model}: wrote {path.relative_to(PROJECT_ROOT)} "
          f"({path.stat().st_size / 1024:,.0f} KB)")


def build_model(model: str, sample: int, force: bool) -> bool:
    """Write one model's pca_cache.json and vector_sample_50.json."""
    pca_path = EMBED_ROOT / model / "pca_cache.json"
    sample_path = EMBED_ROOT / model / "vector_sample_50.json"
    if pca_path.exists() and sample_path.exists() and not force:
        print(f"  {model}: both caches already exist (use --force to rebuild)")
        return True

    if not (EMBED_ROOT / model).is_dir():
        print(f"  {model}: no directory at {EMBED_ROOT / model}", file=sys.stderr)
        return False

    tracks, samples, failures = {}, {}, []
    for track in TRACKS:
        payload, inspector, err = project_track(model, track, sample)
        if err:
            failures.append(f"{track}: {err}")
            print(f"  {model}/{track}: skipped - {err}", file=sys.stderr)
        else:
            tracks[track] = payload
            samples[track] = inspector
            print(f"  {model}/{track}: {payload['n_sampled']:,} of "
                  f"{payload['n_total']:,} vectors, {payload['dim']}-d, "
                  f"{sum(payload['explained']):.1%} variance retained")

    if not tracks:
        print(f"  {model}: nothing written ({'; '.join(failures)})", file=sys.stderr)
        return False

    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    _write(pca_path, {
        "model": model,
        "generated_utc": stamp,
        "seed": SEED,
        "label_keys": list(LABEL_KEYS),
        "tracks": tracks,
    }, model)

    # One row list shared by both tracks - same work orders, same order. The
    # WOIDs are compared rather than assumed: a mismatch would silently pair a
    # raw vector with another row's MICE vector.
    ids = [samples[t]["woid"] for t in samples]
    if len({tuple(i) for i in ids}) != 1:
        print(f"  {model}: text and MICE samples cover different rows - "
              f"{sample_path.name} not written", file=sys.stderr)
        return False

    lead = samples[next(iter(samples))]
    _write(sample_path, {
        "model": model,
        "generated_utc": stamp,
        "seed": SEED,
        "n": len(lead["woid"]),
        "dim": lead["dim"],
        "rows": [
            {"woid": w, "text": t, "mice": m}
            for w, t, m in zip(lead["woid"], lead["text"], lead["mice"])
        ],
        "vectors": {track: s["vectors"] for track, s in samples.items()},
    }, model)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Precompute the dashboard's static vector artifacts: a 2-D "
                    "PCA projection and a 50-row sample with raw vectors."
    )
    parser.add_argument("--models", nargs="+", default=list(MODELS),
                        help=f"model keys to build (default: {' '.join(MODELS)})")
    parser.add_argument("--sample", type=int, default=1000,
                        help="vectors to sample per track (default: 1000)")
    parser.add_argument("--force", action="store_true",
                        help="rebuild caches that already exist")
    args = parser.parse_args()

    if args.sample < 2:
        parser.error("--sample must be at least 2 for a 2-component PCA")

    print("Precomputing into data/embeddings/<model>/: "
          "pca_cache.json, vector_sample_50.json")
    ok = [build_model(m, args.sample, args.force) for m in args.models]

    built = sum(ok)
    print(f"\n{built} of {len(ok)} model(s) have their caches.")
    if built == 0:
        print("Nothing was built. The FAISS indexes are build artifacts - run the "
              "ingestion pipeline first so text.index/mice.index exist.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
