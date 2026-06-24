"""
loaders/retrieval_runner.py - Thin wrapper around existing retrieval.py.

Provides a single timed_retrieve() function used by the Live Query page.
All actual retrieval logic lives in the parent project's retrieval.py.

Fix (2026-06-24):
    Removed @st.cache_resource from FAISS initialization.  The decorator
    caused initialize_stores() to mutate globals on a cached copy of the
    faiss_store module that was different from the module object seen by
    retrieval.py, so assert_initialized() always raised and every strategy
    silently returned ([], 0.0).

    The replacement (_init_faiss_once) uses a plain module-level sentinel
    so initialization still only happens once per process, but it operates
    on the same sys.modules['faiss_store'] instance that retrieval.py uses.
"""

from __future__ import annotations

import sys
import time
import logging
from typing import Dict, List, Optional, Tuple

import streamlit as st

log = logging.getLogger(__name__)

# Module-level sentinel — replaces @st.cache_resource
_faiss_initialized: bool = False
_faiss_init_error:  str  = ""


def _init_faiss_once() -> bool:
    """
    Initialize FAISS stores exactly once per process.

    Works by importing faiss_store through the normal import machinery so
    that the module object stored in sys.modules is the same one that
    retrieval.py will later import.  Mutating globals on that shared object
    is visible to all callers.

    Returns True on success, False on failure.
    """
    global _faiss_initialized, _faiss_init_error

    if _faiss_initialized:
        return True

    try:
        import faiss_store  # noqa: PLC0415  (intentional late import)
        faiss_store.initialize_stores()
        _faiss_initialized = True
        log.info("FAISS stores initialized successfully.")
        return True

    except Exception as exc:
        _faiss_init_error = str(exc)
        log.error("FAISS init failed: %s", exc)
        return False


# ─────────────────────────────────────────────────────────────
# Strategy runner
# ─────────────────────────────────────────────────────────────

def run_strategy(
    strategy: str,
    query: str,
    top_k: int = 10,
    filter_config=None,
) -> Tuple[List, float]:
    """
    Run a single retrieval strategy and return (results, latency_ms).

    Args:
        strategy:      'A', 'B', 'B_prime', or 'C'
        query:         Raw query string
        top_k:         Number of results
        filter_config: FilterConfig (required for B / B_prime)

    Returns:
        (results, latency_ms) — results is a list of RetrievalResult objects.
        Returns ([], 0.0) on error.
    """
    if not _init_faiss_once():
        log.error(
            "Skipping strategy %s — FAISS not ready: %s",
            strategy, _faiss_init_error,
        )
        return [], 0.0

    try:
        from retrieval import (          # noqa: PLC0415
            strategy_a, strategy_b, strategy_b_prime, strategy_c,
            FilterConfig, deduplicate,
        )
        fc = filter_config or FilterConfig()

        t0 = time.perf_counter()
        if strategy == "A":
            results = strategy_a(query, top_k)
        elif strategy == "B":
            results = strategy_b(query, fc, top_k)
        elif strategy == "B_prime":
            results = strategy_b_prime(query, fc, top_k)
        elif strategy == "C":
            results = strategy_c(query, top_k)
        else:
            raise ValueError(f"Unknown strategy: {strategy!r}")
        latency_ms = (time.perf_counter() - t0) * 1000

        return deduplicate(results), round(latency_ms, 1)

    except Exception as exc:
        log.error("Strategy %s failed: %s", strategy, exc, exc_info=True)
        return [], 0.0


def run_all_strategies(
    query: str,
    top_k: int = 10,
    filter_config=None,
) -> Dict[str, Tuple[List, float]]:
    """
    Run all four strategies against *query*.

    Returns:
        {strategy_name: (results, latency_ms), …}
    """
    out = {}
    for s in ("A", "B", "B_prime", "C"):
        out[s] = run_strategy(s, query, top_k, filter_config)
    return out


def classify_query(query: str):
    """
    Classify a query using the existing query_router.classify_query().
    Returns a QueryAnalysis object, or None on error.
    """
    try:
        from query_router import classify_query as _cq  # noqa: PLC0415
        return _cq(query)
    except Exception as exc:
        log.error("Query classification failed: %s", exc)
        return None


def get_faiss_init_error() -> str:
    """Return the FAISS initialization error message, or empty string if OK."""
    return _faiss_init_error