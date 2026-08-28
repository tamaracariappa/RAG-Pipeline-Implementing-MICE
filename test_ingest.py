"""
test_ingest.py - regression checks for the two-pass FAISS ingest.

Run:  python test_ingest.py          (exit 0 = all pass)

Exercises the orchestration in main.py, not the encoder: embed_texts is
replaced with a deterministic stub so the whole suite runs in seconds against
a synthetic CSV in a temp directory.  The real corpus and the real indexes are
never opened.

What it pins down:
  - a fresh run indexes every row exactly once, in both passes
  - metadata carries the dataset columns and NOT the derived repr columns
  - an interrupted run resumes without duplicating or skipping rows
  - _reconcile() adopts a chunk that was persisted before its checkpoint
    (the crash window between persist_track and save_progress)
  - resuming under a changed CSV_CHUNK_SIZE is refused, not silently wrong
  - a pre-two-pass checkpoint is understood
"""

import io
import json
import os
import pickle
import shutil
import tempfile

import numpy as np
import pandas as pd

import faiss_store
import main
from config import EMBEDDING_DIM

_failures = []
N_ROWS, CHUNK = 250, 100          # 3 chunks: 100 / 100 / 50


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{'  - ' + detail if detail else ''}")
    if not ok:
        _failures.append(name)


def _fake_embed(texts, show_progress=False):
    """Deterministic unit vectors keyed off the text - no model, no network."""
    out = np.empty((len(texts), EMBEDDING_DIM), dtype=np.float32)
    for i, t in enumerate(texts):
        rng = np.random.default_rng(abs(hash(t)) % (2**32))
        v = rng.standard_normal(EMBEDDING_DIM).astype(np.float32)
        out[i] = v / np.linalg.norm(v)
    return out


def _write_csv(path, n=N_ROWS):
    pd.DataFrame([{
        "BuildingID":    f"A{100 + i % 7}",
        "BuildingName":  f"building {i % 5}",
        "Type":          ["research", "teaching", "other"][i % 3],
        "WOID":          f"WO{i:06d}",
        "WODescription": f"replace failed chilled water pump unit {i}",
        "WOStartDate":   "2019-03-14",
        "WOEndDate":     "ONGOING",
        "equipment":     ["hvac", "plumbing", "electrical"][i % 3],
    } for i in range(n)]).to_csv(path, index=False)


class Harness:
    """Point main.py and faiss_store at a throwaway directory."""

    def __enter__(self):
        self.tmp = tempfile.mkdtemp(prefix="mice_ingest_test_")
        self.csv = os.path.join(self.tmp, "clean.csv")
        _write_csv(self.csv)

        self._saved = {
            "embed":    main.embed_texts,
            "iter":     main.iter_cleaned_chunks,
            "cleaned":  main.CLEANED_PATH,
            "progress": main.PROGRESS_FILE,
            "ensure":   main.ensure_model_dir,
            "chunk":    main.CSV_CHUNK_SIZE,
            "paths":    (faiss_store.TEXT_INDEX_PATH, faiss_store.MICE_INDEX_PATH,
                         faiss_store.TEXT_METADATA_PATH, faiss_store.MICE_METADATA_PATH),
        }

        main.embed_texts = _fake_embed
        main.iter_cleaned_chunks = self._chunks
        main.CLEANED_PATH = self.csv
        main.PROGRESS_FILE = os.path.join(self.tmp, "ingestion_progress.json")
        main.ensure_model_dir = lambda: self.tmp
        main.CSV_CHUNK_SIZE = CHUNK

        faiss_store.TEXT_INDEX_PATH = os.path.join(self.tmp, "text.index")
        faiss_store.MICE_INDEX_PATH = os.path.join(self.tmp, "mice.index")
        faiss_store.TEXT_METADATA_PATH = os.path.join(self.tmp, "text_metadata.pkl")
        faiss_store.MICE_METADATA_PATH = os.path.join(self.tmp, "mice_metadata.pkl")
        return self

    def __exit__(self, *exc):
        main.embed_texts = self._saved["embed"]
        main.iter_cleaned_chunks = self._saved["iter"]
        main.CLEANED_PATH = self._saved["cleaned"]
        main.PROGRESS_FILE = self._saved["progress"]
        main.ensure_model_dir = self._saved["ensure"]
        main.CSV_CHUNK_SIZE = self._saved["chunk"]
        (faiss_store.TEXT_INDEX_PATH, faiss_store.MICE_INDEX_PATH,
         faiss_store.TEXT_METADATA_PATH, faiss_store.MICE_METADATA_PATH) = self._saved["paths"]
        faiss_store.text_index = faiss_store.mice_index = None
        faiss_store.text_metadata = faiss_store.mice_metadata = None
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _chunks(self):
        for chunk in pd.read_csv(self.csv, dtype=str, chunksize=main.CSV_CHUNK_SIZE):
            yield chunk.fillna("")

    def progress(self):
        with io.open(main.PROGRESS_FILE, encoding="utf-8") as f:
            return json.load(f)

    def count(self, track):
        """ntotal for one track, or None if that pass never wrote its index."""
        import faiss
        path = (faiss_store.TEXT_INDEX_PATH if track == "text"
                else faiss_store.MICE_INDEX_PATH)
        if not os.path.exists(path):
            return None
        return faiss.read_index(path).ntotal

    def counts(self):
        return self.count("text"), self.count("mice")


def test_fresh_run():
    print("\n1. fresh two-pass ingest")
    with Harness() as h:
        main._stage_ingest()
        t, m = h.counts()
        check("TEXT index has every row exactly once", t == N_ROWS, f"ntotal={t}")
        check("MICE index has every row exactly once", m == N_ROWS, f"ntotal={m}")

        p = h.progress()
        check("both passes marked done",
              p["text"]["done"] and p["mice"]["done"])
        check("derived 'completed' flag set", p["completed"] is True)
        check("chunk_size recorded", p["chunk_size"] == CHUNK, str(p["chunk_size"]))
        check("3 chunks per pass", len(p["text"]["completed_chunks"]) == 3,
              str(p["text"]["completed_chunks"]))

        with open(faiss_store.TEXT_METADATA_PATH, "rb") as f:
            meta = pickle.load(f)
        check("metadata has one record per row", len(meta) == N_ROWS, str(len(meta)))
        check("metadata keeps the dataset columns",
              {"WOID", "BuildingID", "Type", "equipment", "WODescription"} <= set(meta[0]))
        check("metadata does NOT carry derived repr columns",
              "text_repr" not in meta[0] and "mice_repr" not in meta[0],
              str(list(meta[0])))


def test_resume_after_interruption():
    print("\n2. resume after an interrupted run")
    with Harness() as h:
        # Run only the TEXT pass, then stop as if the process died.
        progress = main.load_progress()
        main._ingest_pass("text", N_ROWS, progress)

        p = h.progress()
        check("TEXT pass done, MICE pass not started",
              p["text"]["done"] and not p["mice"]["done"])
        check("mice.index not written yet",
              not os.path.exists(faiss_store.MICE_INDEX_PATH))

        # Resume: TEXT must be skipped, MICE must run.
        main._stage_ingest()
        t, m = h.counts()
        check("TEXT not double-ingested on resume", t == N_ROWS, f"ntotal={t}")
        check("MICE completed on resume", m == N_ROWS, f"ntotal={m}")


def test_reconcile_crash_window():
    print("\n3. crash between persist_track() and save_progress()")
    with Harness() as h:
        progress = main.load_progress()
        main._ingest_pass("text", N_ROWS, progress)

        # Simulate the crash window: the index holds chunk 2, but the
        # checkpoint was never updated for it.
        p = h.progress()
        p["text"]["completed_chunks"] = [0, 1]
        p["text"]["rows"] = 200
        p["text"]["done"] = False
        p["completed"] = False
        with io.open(main.PROGRESS_FILE, "w", encoding="utf-8") as f:
            json.dump(p, f)

        reloaded = main.load_progress()
        main._ingest_pass("text", N_ROWS, reloaded)

        t = h.count("text")
        check("chunk adopted, not re-embedded (no duplicates)", t == N_ROWS,
              f"ntotal={t} (would be {N_ROWS + 50} if re-added)")
        check("checkpoint caught up to the index",
              h.progress()["text"]["rows"] == N_ROWS)


def test_chunk_size_change_refused():
    print("\n4. resuming under a changed CSV_CHUNK_SIZE is refused")
    with Harness() as h:
        progress = main.load_progress()
        main._ingest_pass("text", N_ROWS, progress)      # partial: mice not done

        main.CSV_CHUNK_SIZE = CHUNK * 10                 # config changed underneath
        try:
            main._stage_ingest()
            check("raises on chunk-size mismatch", False, "no exception raised")
        except RuntimeError as exc:
            check("raises on chunk-size mismatch", True)
            check("error names --reset-progress", "--reset-progress" in str(exc))
        finally:
            main.CSV_CHUNK_SIZE = CHUNK


def test_legacy_checkpoint_understood():
    print("\n5. pre-two-pass checkpoint is migrated, not misread")
    with Harness() as h:
        with io.open(main.PROGRESS_FILE, "w", encoding="utf-8") as f:
            json.dump({"completed_chunks": [0, 1, 2],
                       "total_rows_processed": N_ROWS,
                       "completed": True,
                       "timestamp": "2026-05-23 08:03:27"}, f)

        p = main.load_progress()
        check("legacy chunks applied to BOTH passes",
              p["text"]["completed_chunks"] == {0, 1, 2}
              and p["mice"]["completed_chunks"] == {0, 1, 2})
        check("legacy completion recognised",
              p["text"]["done"] and p["mice"]["done"])
        check("legacy chunk_size assumed 10000", p["chunk_size"] == 10000,
              str(p["chunk_size"]))

        # A completed legacy run is exempt from the chunk-size guard.
        main._check_chunk_size(p)
        check("completed legacy checkpoint passes the chunk-size guard", True)


if __name__ == "__main__":
    print("=" * 62)
    print(f"two-pass ingest regression checks  (dim {EMBEDDING_DIM})")
    print("=" * 62)

    test_fresh_run()
    test_resume_after_interruption()
    test_reconcile_crash_window()
    test_chunk_size_change_refused()
    test_legacy_checkpoint_understood()

    print("\n" + "=" * 62)
    if _failures:
        print(f"FAILED ({len(_failures)}): {', '.join(_failures)}")
        raise SystemExit(1)
    print("ALL CHECKS PASSED")
