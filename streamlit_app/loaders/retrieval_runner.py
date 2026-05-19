"""
loaders/retrieval_runner.py - Thin wrapper around existing retrieval.py.

Provides a single timed_retrieve() function used by the Live Query page.
All actual retrieval logic lives in the parent project's retrieval.py.
"""

from __future__ import annotations

import time
import logging
from typing import Dict, List, Optional, Tuple

import streamlit as st

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Lazy FAISS initialization
# ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _ensure_faiss_ready() -> bool:
    """Initialize FAISS stores once; return True if successful."""
    try:
        import faiss_store
        faiss_store.initialize_stores()
        return True
    except Exception as exc:
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
        (results, latency_ms) - results is a list of RetrievalResult objects.
        Returns ([], 0.0) on error.
    """
    if not _ensure_faiss_ready():
        return [], 0.0

    try:
        from retrieval import (
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
        log.error("Strategy %s failed: %s", strategy, exc)
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
        from query_router import classify_query as _cq
        return _cq(query)
    except Exception as exc:
        log.error("Query classification failed: %s", exc)
        return None
