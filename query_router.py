"""
query_router.py - Lightweight rule-based query classifier and strategy router.

Design constraints:
  - Zero ML inference: pure regex + vocabulary lookup.
  - Operates on raw query text only; no external state needed.
  - Deterministic: same input always produces the same routing decision.
  - Returns a FilterConfig alongside the strategy label so the caller
    can pass both directly to route_and_retrieve().

Routing logic:
  Query contains equipment AND/OR type tokens  →  Strategy B_prime (pre-filter)
  Query contains only building_id pattern      →  Strategy B       (post-filter)
  No metadata tokens detected                  →  Strategy A       (baseline)
  Caller explicitly requests MICE              →  Strategy C

Note: The router is optional.  The four strategies can always be called
directly (e.g., for controlled experiments where strategy is fixed).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, FrozenSet, List, Optional, Tuple

from retrieval import FilterConfig, RetrievalResult, route_and_retrieve
from config import DEFAULT_TOP_K


# ─────────────────────────────────────────────────────────────
# Vocabulary
# ─────────────────────────────────────────────────────────────

# Equipment categories drawn directly from the cleaned dataset
EQUIPMENT_VOCAB: FrozenSet[str] = frozenset({
    "hvac", "plumbing", "electrical", "elevator", "fire protection",
    "interior finishes", "interior construction", "furnishings", "equipment",
    "telecommunications", "site", "conveying", "structural", "roofing",
    "exterior", "specialties",
})

# Facility type tokens
TYPE_VOCAB: FrozenSet[str] = frozenset({
    "research", "teaching", "mixed teaching research",
    "student experience", "other",
})

# Regex for building ID patterns (e.g. A050, C201, J920)
_BUILDING_ID_RE = re.compile(r'\b([A-Z]\d{3,4})\b')

# Pre-tokenise multi-word vocab entries for O(1) substring lookup
_EQUIPMENT_SORTED = sorted(EQUIPMENT_VOCAB, key=len, reverse=True)
_TYPE_SORTED      = sorted(TYPE_VOCAB,      key=len, reverse=True)


# ─────────────────────────────────────────────────────────────
# QueryAnalysis
# ─────────────────────────────────────────────────────────────

class QuerySignal(Enum):
    NONE        = "none"
    EQUIPMENT   = "equipment"
    TYPE        = "type"
    BUILDING_ID = "building_id"
    MIXED       = "mixed"          # equipment + type


@dataclass
class QueryAnalysis:
    """
    Result of classifying a raw query string.

    recommended_strategy:
      'A'       - no metadata detected
      'B'       - building_id only (post-filter is safer for ID lookups)
      'B_prime' - equipment and/or type detected (pre-filter is efficient)
      'C'       - caller-requested; not set by the router automatically
    """
    query:                   str
    signal:                  QuerySignal
    extracted_equipment:     Optional[str]
    extracted_type:          Optional[str]
    extracted_building_id:   Optional[str]
    filter_config:           FilterConfig
    recommended_strategy:    str            # 'A' | 'B' | 'B_prime' | 'C'


# ─────────────────────────────────────────────────────────────
# Classifier
# ─────────────────────────────────────────────────────────────

def _find_in_vocab(text: str, vocab_sorted: List[str]) -> Optional[str]:
    """Return the first (longest) vocabulary term found in *text*."""
    for term in vocab_sorted:
        # Use word-boundary matching where possible
        pattern = r'\b' + re.escape(term) + r'\b'
        if re.search(pattern, text):
            return term
    return None


def classify_query(query: str) -> QueryAnalysis:
    """
    Classify *query* and return a QueryAnalysis with routing recommendation.

    Rules (applied in priority order):
      1. Extract building_id (regex).
      2. Extract equipment term (vocabulary lookup).
      3. Extract type term (vocabulary lookup).
      4. Route: MIXED → B_prime, EQUIPMENT/TYPE → B_prime,
                BUILDING_ID → B, NONE → A.
    """
    q_lower = query.lower()

    # 1. Building ID
    bid_match = _BUILDING_ID_RE.search(query)   # search original case
    building_id = bid_match.group(1) if bid_match else None

    # 2. Equipment
    equipment = _find_in_vocab(q_lower, _EQUIPMENT_SORTED)

    # 3. Type
    btype = _find_in_vocab(q_lower, _TYPE_SORTED)

    # 4. Signal + strategy
    has_equip = equipment is not None
    has_type  = btype     is not None
    has_bid   = building_id is not None

    if has_equip and has_type:
        signal   = QuerySignal.MIXED
        strategy = "B_prime"
    elif has_equip or has_type:
        signal   = QuerySignal.EQUIPMENT if has_equip else QuerySignal.TYPE
        strategy = "B_prime"
    elif has_bid:
        signal   = QuerySignal.BUILDING_ID
        strategy = "B"
    else:
        signal   = QuerySignal.NONE
        strategy = "A"

    filter_cfg = FilterConfig(
        building_id   = building_id,
        building_type = btype,
        equipment     = equipment,
    )

    return QueryAnalysis(
        query                 = query,
        signal                = signal,
        extracted_equipment   = equipment,
        extracted_type        = btype,
        extracted_building_id = building_id,
        filter_config         = filter_cfg,
        recommended_strategy  = strategy,
    )


# ─────────────────────────────────────────────────────────────
# Router
# ─────────────────────────────────────────────────────────────

def router_retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    force_strategy: Optional[str] = None,
) -> Tuple[List[RetrievalResult], QueryAnalysis]:
    """
    Classify *query* and retrieve using the recommended (or forced) strategy.

    Args:
        query:          Raw natural-language query.
        top_k:          Number of results to return.
        force_strategy: Override the router's recommendation.
                        Useful for ablation: pass 'C' to always use MICE.

    Returns:
        (results, analysis) where analysis carries the routing decision.
    """
    analysis  = classify_query(query)
    strategy  = force_strategy or analysis.recommended_strategy
    results   = route_and_retrieve(
        query         = query,
        strategy      = strategy,
        filter_config = analysis.filter_config,
        top_k         = top_k,
    )
    return results, analysis


# ─────────────────────────────────────────────────────────────
# CLI helper
# ─────────────────────────────────────────────────────────────

def explain_routing(query: str) -> None:
    """Print a human-readable routing explanation for *query*."""
    a = classify_query(query)
    print(f"\nQuery   : {a.query}")
    print(f"Signal  : {a.signal.value}")
    print(f"Building: {a.extracted_building_id or '—'}")
    print(f"Type    : {a.extracted_type        or '—'}")
    print(f"Equip   : {a.extracted_equipment   or '—'}")
    print(f"→ Strategy {a.recommended_strategy}")
