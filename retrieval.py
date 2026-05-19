"""
retrieval.py - Four retrieval strategies + query routing integration.

Core strategies A / B / B' / C are UNCHANGED.
route_and_retrieve() is the single integration point for the QueryRouter:
it accepts a raw query, classifies it, selects the appropriate strategy,
and returns results alongside the routing decision for downstream analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import faiss_store

from config import (
    DEFAULT_TOP_K,
    POST_FILTER_MULTIPLIER,
)

from embedder import embed_query


# ─────────────────────────────────────────────────────────────
# Data types
# ─────────────────────────────────────────────────────────────

@dataclass
class FilterConfig:
    """Metadata constraints for Strategy B / B'. None = unconstrained."""
    building_id:   Optional[str] = None
    building_type: Optional[str] = None
    equipment:     Optional[str] = None

    def is_empty(self) -> bool:
        return all(v is None for v in (self.building_id, self.building_type, self.equipment))


@dataclass
class RetrievalResult:
    woid:           str
    building_id:    str
    building_name:  str
    wo_type:        str
    equipment:      str
    wo_description: str
    score:          float
    strategy:       str

    def as_dict(self) -> Dict:
        return {
            "woid":           self.woid,
            "building_id":    self.building_id,
            "building_name":  self.building_name,
            "type":           self.wo_type,
            "equipment":      self.equipment,
            "wo_description": self.wo_description,
            "score":          round(self.score, 6),
            "strategy":       self.strategy,
        }


# ─────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────


def _python_filter(result: RetrievalResult, fc: FilterConfig) -> bool:
    if fc.building_id   and result.building_id != fc.building_id:
        return False
    if fc.building_type and result.wo_type      != fc.building_type:
        return False
    if fc.equipment     and result.equipment    != fc.equipment:
        return False
    return True


def deduplicate(results: List[RetrievalResult]) -> List[RetrievalResult]:
    """Keep highest-scoring entry per WOID."""
    seen: Set[str] = set()
    out:  List[RetrievalResult] = []
    for r in sorted(results, key=lambda x: x.score, reverse=True):
        if r.woid not in seen:
            seen.add(r.woid)
            out.append(r)
    return out


# ─────────────────────────────────────────────────────────────
# Strategy A - Baseline
# ─────────────────────────────────────────────────────────────

def strategy_a(query: str, top_k: int = DEFAULT_TOP_K):

    vec = embed_query(query)

    scores, indices, metadata = faiss_store.search_text(vec, top_k)

    results = []

    for score, idx in zip(scores, indices):
        if idx < 0 or idx >= len(metadata):
            continue

        row = metadata[idx]

        results.append(
            RetrievalResult(
                woid=row["WOID"],
                building_id=row["BuildingID"],
                building_name=row["BuildingName"],
                wo_type=row["Type"],
                equipment=row["equipment"],
                wo_description=row["WODescription"],
                score=float(score),
                strategy="A"
            )
        )

    return deduplicate(results)

# ─────────────────────────────────────────────────────────────
# Strategy B - Post-filter
# ─────────────────────────────────────────────────────────────

def strategy_b(query, filter_config, top_k=DEFAULT_TOP_K):

    fetch_k = top_k * POST_FILTER_MULTIPLIER

    results = strategy_a(query, fetch_k)

    filtered = [
        r for r in results
        if _python_filter(r, filter_config)
    ]

    for r in filtered:
        r.strategy = "B"

    return filtered[:top_k]


# ─────────────────────────────────────────────────────────────
# Strategy B' - Pre-filter
# ─────────────────────────────────────────────────────────────

def strategy_b_prime(query, filter_config, top_k=DEFAULT_TOP_K):

    results = strategy_a(query, top_k * POST_FILTER_MULTIPLIER)

    filtered = [
        r for r in results
        if _python_filter(r, filter_config)
    ]

    for r in filtered:
        r.strategy = "B_prime"

    return filtered[:top_k]


# ─────────────────────────────────────────────────────────────
# Strategy C - MICE
# ─────────────────────────────────────────────────────────────

def strategy_c(query: str, top_k: int = DEFAULT_TOP_K):

    vec = embed_query(f"work order description: {query}")

    scores, indices, metadata = faiss_store.search_mice(vec, top_k)

    results = []

    for score, idx in zip(scores, indices):

        if idx < 0 or idx >= len(metadata):
            continue
            
        row = metadata[idx]

        results.append(
            RetrievalResult(
                woid=row["WOID"],
                building_id=row["BuildingID"],
                building_name=row["BuildingName"],
                wo_type=row["Type"],
                equipment=row["equipment"],
                wo_description=row["WODescription"],
                score=float(score),
                strategy="C"
            )
        )

    return deduplicate(results)


# ─────────────────────────────────────────────────────────────
# Routing integration point  (used by query_router.py)
# ─────────────────────────────────────────────────────────────

def route_and_retrieve(
    query: str,
    strategy: str,
    filter_config: Optional[FilterConfig] = None,
    top_k: int = DEFAULT_TOP_K,
) -> List[RetrievalResult]:
    """
    Dispatch to a named strategy.  Called by QueryRouter after classification.

    Args:
        query:         Raw query string (no prefix applied here).
        strategy:      One of 'A', 'B', 'B_prime', 'C'.
        filter_config: Required for B / B'; ignored for A and C.
        top_k:         Number of results to return.

    Returns:
        Deduplicated RetrievalResult list.

    Raises:
        ValueError: Unknown strategy name.
    """
    fc = filter_config or FilterConfig()
    if strategy == "A":
        return strategy_a(query, top_k)
    if strategy == "B":
        return strategy_b(query, fc, top_k)
    if strategy == "B_prime":
        return strategy_b_prime(query, fc, top_k)
    if strategy == "C":
        return strategy_c(query, top_k)
    raise ValueError(f"Unknown strategy: {strategy!r}. Choose A / B / B_prime / C.")
