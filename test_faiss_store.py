"""
test_faiss_store.py - regression checks for the memory-optimised FAISS store.

Run:  python test_faiss_store.py          (exit 0 = all pass)

Covers the parts of faiss_store.py that are easy to get subtly wrong:
  - capacity reservation must not change what the index contains
  - reserving on an index that ALREADY holds vectors must preserve them
    (a plain resize(0) would truncate the buffer while ntotal kept pointing
    at the missing codes - silent corruption, not a crash)
  - the experimental controls are untouched: IndexFlatIP, exact cosine over
    L2-normalised vectors, bit-identical scores before and after reservation

No pytest, no fixtures - a plain assert script against a throwaway directory.
The real indexes are never opened.
"""

import os
import shutil
import tempfile

import faiss
import numpy as np

import faiss_store
from config import EMBEDDING_DIM

_failures = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{'  - ' + detail if detail else ''}")
    if not ok:
        _failures.append(name)


def _unit(n, d=None, seed=0):
    """n L2-normalised random vectors - the representation the pipeline stores."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, d or EMBEDDING_DIM)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return np.ascontiguousarray(v, dtype=np.float32)


def test_reserve_on_empty():
    print("\n1. reserve on a fresh index")
    idx = faiss_store.create_index(n_reserve=5000)
    check("stays empty after reservation", idx.ntotal == 0, f"ntotal={idx.ntotal}")
    check("still IndexFlatIP", isinstance(idx, faiss.IndexFlatIP))
    check("dim preserved", idx.d == EMBEDDING_DIM, f"d={idx.d}")

    v = _unit(100)
    idx.add(v)
    check("add works after reservation", idx.ntotal == 100, f"ntotal={idx.ntotal}")

    scores, _ = idx.search(v[:1], 1)
    check("self-similarity == 1.0 (cosine control intact)",
          abs(float(scores[0][0]) - 1.0) < 1e-4, f"score={float(scores[0][0]):.6f}")


def test_reserve_preserves_existing_rows():
    """The resume case. This is the one a naive resize(0) breaks."""
    print("\n2. reserve on an index that already holds vectors (resume path)")
    idx = faiss_store.create_index()
    v = _unit(250, seed=7)
    idx.add(v)

    before_scores, before_ids = idx.search(v[:5], 3)

    faiss_store.reserve_capacity(idx, 100_000)      # simulate resumed ingest

    check("ntotal unchanged", idx.ntotal == 250, f"ntotal={idx.ntotal}")

    after_scores, after_ids = idx.search(v[:5], 3)
    check("stored vectors survived reservation",
          np.array_equal(before_ids, after_ids))
    check("scores bit-identical after reservation",
          np.array_equal(before_scores, after_scores))

    recon = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    idx.reconstruct(0, recon)
    check("reconstruct(0) matches the vector originally added",
          np.allclose(recon, v[0], atol=1e-6))

    idx.add(_unit(50, seed=8))
    check("can keep adding after reservation", idx.ntotal == 300, f"ntotal={idx.ntotal}")


def test_reserve_is_noop_when_zero():
    print("\n3. n_reserve=0 leaves behaviour exactly as before")
    a = faiss_store.create_index()
    b = faiss_store.create_index(n_reserve=0)
    v = _unit(40, seed=3)
    a.add(v); b.add(v)
    sa, _ = a.search(v[:3], 5)
    sb, _ = b.search(v[:3], 5)
    check("identical results with and without the parameter",
          np.array_equal(sa, sb))


def test_initialize_stores_roundtrip():
    print("\n4. initialize_stores(n_reserve=...) round-trips through disk")
    tmp = tempfile.mkdtemp(prefix="mice_faiss_test_")
    orig = (faiss_store.TEXT_INDEX_PATH, faiss_store.MICE_INDEX_PATH,
            faiss_store.TEXT_METADATA_PATH, faiss_store.MICE_METADATA_PATH)
    try:
        faiss_store.TEXT_INDEX_PATH = os.path.join(tmp, "text.index")
        faiss_store.MICE_INDEX_PATH = os.path.join(tmp, "mice.index")
        faiss_store.TEXT_METADATA_PATH = os.path.join(tmp, "text_metadata.pkl")
        faiss_store.MICE_METADATA_PATH = os.path.join(tmp, "mice_metadata.pkl")

        faiss_store.initialize_stores(n_reserve=10_000)
        check("fresh stores start empty", faiss_store.text_index.ntotal == 0)

        rows = [{"WOID": f"WO{i}", "BuildingID": "A050", "BuildingName": "chemistry",
                 "Type": "research", "equipment": "hvac",
                 "WODescription": "replace failed chilled water pump",
                 "WOStartDate": "2019-03-14", "WOEndDate": "ONGOING"}
                for i in range(120)]
        tv, mv = _unit(120, seed=11), _unit(120, seed=12)
        faiss_store.insert_text_batch(rows, tv)
        faiss_store.insert_mice_batch(rows, mv)
        faiss_store.persist()

        probe = tv[:4].copy()
        want_s, want_i, _ = faiss_store.search_text(probe[0], 3)

        # Simulate a resumed run: reload from disk WITH a reservation.
        faiss_store.text_index = faiss_store.mice_index = None
        faiss_store.initialize_stores(n_reserve=10_000)

        check("reloaded text index has all rows",
              faiss_store.text_index.ntotal == 120,
              f"ntotal={faiss_store.text_index.ntotal}")
        check("reloaded mice index has all rows",
              faiss_store.mice_index.ntotal == 120)
        check("metadata reloaded", len(faiss_store.text_metadata) == 120)

        got_s, got_i, _ = faiss_store.search_text(probe[0], 3)
        check("search results identical across persist+reserve+reload",
              np.array_equal(want_i, got_i) and np.allclose(want_s, got_s, atol=1e-6))

        faiss_store.insert_text_batch(rows[:10], _unit(10, seed=13))
        check("resumed index still accepts inserts",
              faiss_store.text_index.ntotal == 130,
              f"ntotal={faiss_store.text_index.ntotal}")
    finally:
        (faiss_store.TEXT_INDEX_PATH, faiss_store.MICE_INDEX_PATH,
         faiss_store.TEXT_METADATA_PATH, faiss_store.MICE_METADATA_PATH) = orig
        faiss_store.text_index = faiss_store.mice_index = None
        faiss_store.text_metadata = faiss_store.mice_metadata = None
        shutil.rmtree(tmp, ignore_errors=True)


def _avail_bytes():
    """Free physical RAM, via GlobalMemoryStatusEx. Noisy but directional."""
    import ctypes
    class M(ctypes.Structure):
        _fields_ = ([("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] +
                    [(n, ctypes.c_ulonglong) for n in
                     ("ullTotalPhys", "ullAvailPhys", "ullTotalPageFile",
                      "ullAvailPageFile", "ullTotalVirtual", "ullAvailVirtual",
                      "ullAvailExtendedVirtual")])
    m = M(); m.dwLength = ctypes.sizeof(M)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullAvailPhys


def test_mmap_load():
    print("\n5. memory-mapped load (evaluation / retrieval path)")
    tmp = tempfile.mkdtemp(prefix="mice_mmap_test_")
    try:
        path = os.path.join(tmp, "text.index")
        n = 40_000                                   # 40k x 768 x 4B = 123 MB
        vecs = _unit(n, seed=21)
        idx = faiss_store.create_index()
        idx.add(vecs)
        faiss.write_index(idx, path)
        del idx

        probe = vecs[:3].copy()

        resident = faiss_store.load_index(path, mmap=False)
        want_s, want_i = resident.search(probe, 5)
        del resident

        mapped = faiss_store.load_index(path, mmap=True)
        got_s, got_i = mapped.search(probe, 5)

        check("mapped index reports the same row count", mapped.ntotal == n,
              f"ntotal={mapped.ntotal}")
        check("mapped index keeps dim", mapped.d == EMBEDDING_DIM)
        check("mapped search returns identical neighbours",
              np.array_equal(want_i, got_i))
        check("mapped search returns identical scores",
              np.allclose(want_s, got_s, atol=1e-6))

        recon = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        mapped.reconstruct(7, recon)
        check("reconstruct() works on a mapped index (Streamlit needs it)",
              np.allclose(recon, vecs[7], atol=1e-6))
        del mapped

        # IO_FLAG_MMAP does NOT map a flat index (see load_index docstring).
        # Pin the measurement so this test starts failing the day faiss gains
        # real flat-index mapping - that is a change worth noticing.
        payload_mb = n * EMBEDDING_DIM * 4 / 1e6
        before = _avail_bytes()
        a = faiss_store.load_index(path, mmap=False)
        resident_mb = (before - _avail_bytes()) / 1e6
        del a

        before = _avail_bytes()
        b = faiss_store.load_index(path, mmap=True)
        mapped_mb = (before - _avail_bytes()) / 1e6
        del b

        print(f"      payload {payload_mb:.0f} MB | resident load {resident_mb:.0f} MB "
              f"| IO_FLAG_MMAP load {mapped_mb:.0f} MB")
        check("IO_FLAG_MMAP still resident-loads a flat index "
              "(documented faiss behaviour, not a saving)",
              mapped_mb > payload_mb * 0.5,
              f"{mapped_mb:.0f} MB consumed for a {payload_mb:.0f} MB payload")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_ingest_path_is_not_mapped():
    print("\n6. ingest opens its track unmapped and writable")
    tmp = tempfile.mkdtemp(prefix="mice_ingest_open_test_")
    orig = (faiss_store.TEXT_INDEX_PATH, faiss_store.TEXT_METADATA_PATH)
    try:
        faiss_store.TEXT_INDEX_PATH = os.path.join(tmp, "text.index")
        faiss_store.TEXT_METADATA_PATH = os.path.join(tmp, "text_metadata.pkl")

        faiss_store.initialize_store("text", n_reserve=5000, mmap=False)
        faiss_store.insert_batch("text", [{"WOID": f"WO{i}"} for i in range(60)],
                                 _unit(60, seed=31))
        faiss_store.persist_track("text")
        check("fresh unmapped track accepts inserts",
              faiss_store.text_index.ntotal == 60,
              f"ntotal={faiss_store.text_index.ntotal}")

        # Reopen the way a resumed ingest does, then keep writing.
        faiss_store.initialize_store("text", n_reserve=5000, mmap=False)
        faiss_store.insert_batch("text", [{"WOID": f"WO{i}"} for i in range(40)],
                                 _unit(40, seed=32))
        check("resumed unmapped track keeps accepting inserts",
              faiss_store.text_index.ntotal == 100,
              f"ntotal={faiss_store.text_index.ntotal}")

        faiss_store.release("text")
        check("release() drops the index", faiss_store.text_index is None)
    finally:
        (faiss_store.TEXT_INDEX_PATH, faiss_store.TEXT_METADATA_PATH) = orig
        faiss_store.text_index = None
        faiss_store.text_metadata = None
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    print("=" * 62)
    print(f"faiss_store regression checks  (dim {EMBEDDING_DIM}, faiss {faiss.__version__})")
    print("=" * 62)

    test_reserve_on_empty()
    test_reserve_preserves_existing_rows()
    test_reserve_is_noop_when_zero()
    test_initialize_stores_roundtrip()
    test_mmap_load()
    test_ingest_path_is_not_mapped()

    print("\n" + "=" * 62)
    if _failures:
        print(f"FAILED ({len(_failures)}): {', '.join(_failures)}")
        raise SystemExit(1)
    print("ALL CHECKS PASSED")
