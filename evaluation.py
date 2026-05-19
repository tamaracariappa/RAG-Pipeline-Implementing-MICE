"""
evaluation.py - Research-grade evaluation for the FM RAG pipeline.

Replaces self-retrieval with a rigorous multi-document relevance model:
  - Relevance is defined by (equipment × type) topic grouping.
  - Two query sub-types ensure both signals are tested:
      semantic    (40%): stripped description, no metadata hint, no filter
      constrained (60%): equipment/type prepended, filter_config populated
  - Metrics are broken out per query type → ablation-ready.
  - Per-query wall-clock latency (mean, std, p95) reported per strategy.
  - All seeds are explicit; output is fully deterministic.
"""

from __future__ import annotations

import json
import logging
import random
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from config import (
    CLEANED_PATH,
    EVAL_RESULTS_PATH,
    EVAL_SAMPLE_SIZE,
    EVAL_SEED,
    EVAL_TOP_K_LIST,
)
from retrieval import (
    FilterConfig,
    RetrievalResult,
    deduplicate,
    strategy_a,
    strategy_b,
    strategy_b_prime,
    strategy_c,
)

log = logging.getLogger(__name__)

_CONSTRAINED_FRACTION = 0.60   # fraction of test cases with metadata hints
_MIN_GROUP_SIZE       = 5       # minimum docs per (equipment, type) topic


# ─────────────────────────────────────────────────────────────
# TestCase
# ─────────────────────────────────────────────────────────────

@dataclass
class TestCase:
    """
    A single evaluation query with its ground-truth relevant document set.

    query_type:
      'semantic'    - no metadata hint in query text; filter_config all-None.
                      Tests whether pure ANN recall is sufficient.
      'constrained' - equipment/type tokens in query text; filter_config set.
                      Tests whether metadata-aware strategies outperform A.
    """
    query_text:      str
    relevant_woids:  Set[str]
    filter_config:   FilterConfig
    query_type:      str                  # 'semantic' | 'constrained'
    metadata_group:  Tuple[str, str]      # (equipment, type) defining relevance
    description:     str = ""


# ─────────────────────────────────────────────────────────────
# EvalDatasetBuilder
# ─────────────────────────────────────────────────────────────

class EvalDatasetBuilder:
    """
    Builds a deterministic, multi-document relevance test set from the
    cleaned CSV with no LLM involvement.

    Relevance model:
      All documents sharing the same (equipment, type) tuple are mutually
      relevant.  This is unambiguous, reproducible, and directly tests
      the metadata contribution hypothesis.
    """

    _BUILDING_CODE = re.compile(r'\b[a-z]\d{3,4}\b')
    _ZONE          = re.compile(r'\bzone\s*\d+\b')
    _INTERNAL_CODE = re.compile(r'\b(pm|cppainter|waycas|nspi|wo\d+|elelct|csb)\b')
    _DASH_CODE     = re.compile(r'\b\w{2,}-\w{2,}-\w{2,}\b')
    _WHITESPACE    = re.compile(r'\s{2,}')

    def __init__(
        self,
        df: pd.DataFrame,
        min_group_size: int = _MIN_GROUP_SIZE,
        n_queries: int      = EVAL_SAMPLE_SIZE,
        seed: int           = EVAL_SEED,
    ) -> None:
        self._df  = df.copy()
        self._min = min_group_size
        self._n   = n_queries
        self._rng = random.Random(seed)
        np.random.seed(seed)

    # ── Query text factories ──────────────────────────────────

    def _strip_identifiers(self, desc: str, building_name: str) -> str:
        """Remove building-specific tokens so the query generalises."""
        q = desc.lower()
        if building_name and building_name not in ("unknown_building", ""):
            q = q.replace(building_name.lower(), "")
        q = self._BUILDING_CODE.sub(" ", q)
        q = self._ZONE.sub(" ", q)
        q = self._INTERNAL_CODE.sub(" ", q)
        q = self._DASH_CODE.sub(" ", q)
        q = self._WHITESPACE.sub(" ", q).strip(" --|.,")
        return q

    def _semantic_query(self, desc: str, building_name: str) -> str:
        return self._strip_identifiers(desc, building_name)

    def _constrained_query(
        self,
        desc: str,
        building_name: str,
        equipment: str,
        btype: str,
    ) -> str:
        """Prepend equipment/type tokens to the stripped description."""
        base   = self._strip_identifiers(desc, building_name)
        tokens = [
            t for t in [equipment, btype]
            if t and t not in ("", "unknown_type", "UNKNOWN_TYPE")
        ]
        prefix = " ".join(tokens)
        return f"{prefix}: {base}" if prefix else base

    # ── Public build() ────────────────────────────────────────

    def build(self) -> List[TestCase]:
        """
        1. Group cleaned CSV by (equipment, Type).
        2. Discard groups < min_group_size or with placeholder values.
        3. Shuffle groups deterministically.
        4. For each group: seed doc → query; remaining WOIDs → relevant set.
        5. Assign constrained / semantic query type by position.
        """
        groups = [
            (key, grp.reset_index(drop=True))
            for key, grp in self._df.groupby(["equipment", "Type"])
            if (
                len(grp) >= self._min
                and key[0] not in ("", "UNKNOWN_TYPE")
                and key[1] not in ("", "UNKNOWN_TYPE")
            )
        ]

        if not groups:
            raise ValueError(
                "No valid (equipment × Type) groups found. "
                "Check dataset quality."
            )

        self._rng.shuffle(groups)

        n_constrained = int(self._n * _CONSTRAINED_FRACTION)
        cases: List[TestCase] = []

        for (equipment, btype), grp in groups:
            if len(cases) >= self._n:
                break

            seed_row  = grp.iloc[self._rng.randint(0, len(grp) - 1)]
            rel_woids = set(grp["WOID"]) - {seed_row["WOID"]}

            if len(rel_woids) < 2:
                continue

            constrained = len(cases) < n_constrained

            if constrained:
                query_text = self._constrained_query(
                    seed_row["WODescription"],
                    seed_row["BuildingName"],
                    equipment, btype,
                )
                filter_cfg = FilterConfig(
                    building_type = btype     if btype     not in ("", "UNKNOWN_TYPE") else None,
                    equipment     = equipment if equipment != ""                        else None,
                )
                qtype = "constrained"
            else:
                query_text = self._semantic_query(
                    seed_row["WODescription"],
                    seed_row["BuildingName"],
                )
                filter_cfg = FilterConfig()
                qtype      = "semantic"

            if len(query_text) < 10:
                continue

            cases.append(TestCase(
                query_text     = query_text,
                relevant_woids = rel_woids,
                filter_config  = filter_cfg,
                query_type     = qtype,
                metadata_group = (equipment, btype),
                description    = f"[{qtype}] {equipment}/{btype} src:{seed_row['WOID']}",
            ))

        log.info(
            "Built %d test cases  (%d constrained / %d semantic).",
            len(cases),
            sum(1 for c in cases if c.query_type == "constrained"),
            sum(1 for c in cases if c.query_type == "semantic"),
        )
        return cases


def generate_test_cases(
    csv_path: str = CLEANED_PATH,
    n: int        = EVAL_SAMPLE_SIZE,
    seed: int     = EVAL_SEED,
) -> List[TestCase]:
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    return EvalDatasetBuilder(df, n_queries=n, seed=seed).build()


# ─────────────────────────────────────────────────────────────
# Metric functions
# ─────────────────────────────────────────────────────────────

def recall_at_k(retrieved: List[RetrievalResult], relevant: Set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len({r.woid for r in retrieved[:k]} & relevant) / len(relevant)


def mrr(retrieved: List[RetrievalResult], relevant: Set[str]) -> float:
    for rank, r in enumerate(retrieved, start=1):
        if r.woid in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: List[RetrievalResult], relevant: Set[str], k: int) -> float:
    dcg = sum(
        1.0 / np.log2(rank + 1)
        for rank, r in enumerate(retrieved[:k], start=1)
        if r.woid in relevant
    )
    idcg = sum(1.0 / np.log2(r + 1) for r in range(1, min(len(relevant), k) + 1))
    return dcg / idcg if idcg > 0 else 0.0


# ─────────────────────────────────────────────────────────────
# StrategyMetrics
# ─────────────────────────────────────────────────────────────

@dataclass
class StrategyMetrics:
    strategy:           str
    n_queries:          int   = 0
    mrr_score:          float = 0.0
    recall:             Dict[int, float] = field(default_factory=dict)
    ndcg:               Dict[int, float] = field(default_factory=dict)
    # Ablation breakdowns
    recall_semantic:    Dict[int, float] = field(default_factory=dict)
    recall_constrained: Dict[int, float] = field(default_factory=dict)
    ndcg_semantic:      Dict[int, float] = field(default_factory=dict)
    ndcg_constrained:   Dict[int, float] = field(default_factory=dict)
    # Latency
    latency_mean_s: float = 0.0
    latency_std_s:  float = 0.0
    latency_p95_s:  float = 0.0


# ─────────────────────────────────────────────────────────────
# Per-strategy evaluation loop
# ─────────────────────────────────────────────────────────────

def _evaluate_one_strategy(
    name: str,
    retrieve_fn: Callable[[TestCase], List[RetrievalResult]],
    test_cases: List[TestCase],
    top_k_list: List[int],
) -> StrategyMetrics:

    def _zero() -> Dict[int, float]:
        return {k: 0.0 for k in top_k_list}

    recall_all = _zero(); ndcg_all = _zero()
    recall_sem = _zero(); ndcg_sem = _zero()
    recall_con = _zero(); ndcg_con = _zero()
    mrr_acc = 0.0
    latencies: List[float] = []
    valid = n_sem = n_con = 0

    for tc in test_cases:
        try:
            t0      = time.perf_counter()
            results = retrieve_fn(tc)
            elapsed = time.perf_counter() - t0
        except Exception as exc:
            log.warning("Strategy %s | '%s' failed: %s", name, tc.description, exc)
            continue

        results = deduplicate(results)
        latencies.append(elapsed)
        valid += 1
        is_sem = tc.query_type == "semantic"
        is_sem and (n_sem := n_sem + 1) or (n_con := n_con + 1)  # noqa

        for k in top_k_list:
            r = recall_at_k(results, tc.relevant_woids, k)
            d = ndcg_at_k(results,   tc.relevant_woids, k)
            recall_all[k] += r; ndcg_all[k] += d
            if is_sem:
                recall_sem[k] += r; ndcg_sem[k] += d
            else:
                recall_con[k] += r; ndcg_con[k] += d

        mrr_acc += mrr(results, tc.relevant_woids)

    n  = max(valid,  1)
    ns = max(n_sem,  1)
    nc = max(n_con,  1)
    lat = np.array(latencies) if latencies else np.zeros(1)

    return StrategyMetrics(
        strategy           = name,
        n_queries          = valid,
        mrr_score          = mrr_acc / n,
        recall             = {k: recall_all[k] / n  for k in top_k_list},
        ndcg               = {k: ndcg_all[k]   / n  for k in top_k_list},
        recall_semantic    = {k: recall_sem[k]  / ns for k in top_k_list},
        recall_constrained = {k: recall_con[k]  / nc for k in top_k_list},
        ndcg_semantic      = {k: ndcg_sem[k]    / ns for k in top_k_list},
        ndcg_constrained   = {k: ndcg_con[k]    / nc for k in top_k_list},
        latency_mean_s     = float(lat.mean()),
        latency_std_s      = float(lat.std()),
        latency_p95_s      = float(np.percentile(lat, 95)),
    )


# ─────────────────────────────────────────────────────────────
# Full evaluation runner
# ─────────────────────────────────────────────────────────────

def run_evaluation(
    test_cases: Optional[List[TestCase]] = None,
    top_k_list: List[int]                = EVAL_TOP_K_LIST,
    seed: int                            = EVAL_SEED,
) -> List[StrategyMetrics]:
    """
    Evaluate strategies A, B, B', C on the same query set.
    All strategies share identical top_k, query text, and FilterConfig.
    """
    random.seed(seed)
    np.random.seed(seed)

    if test_cases is None:
        log.info("Generating evaluation dataset …")
        test_cases = generate_test_cases(seed=seed)

    log.info("Evaluating %d queries | k=%s", len(test_cases), top_k_list)
    max_k = max(top_k_list)

    def run_a(tc: TestCase)       -> List[RetrievalResult]:
        return strategy_a(tc.query_text, top_k=max_k)

    def run_b(tc: TestCase)       -> List[RetrievalResult]:
        return strategy_b(tc.query_text, tc.filter_config, top_k=max_k)

    def run_b_prime(tc: TestCase) -> List[RetrievalResult]:
        return strategy_b_prime(tc.query_text, tc.filter_config, top_k=max_k)

    def run_c(tc: TestCase)       -> List[RetrievalResult]:
        return strategy_c(tc.query_text, top_k=max_k)

    all_metrics: List[StrategyMetrics] = []
    for label, fn in [("A", run_a), ("B", run_b), ("B_prime", run_b_prime), ("C", run_c)]:
        log.info("Evaluating Strategy %s …", label)
        m = _evaluate_one_strategy(label, fn, test_cases, top_k_list)
        all_metrics.append(m)

    return all_metrics


# ─────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────

def print_summary(
    all_metrics: List[StrategyMetrics],
    top_k_list: List[int] = EVAL_TOP_K_LIST,
) -> None:
    k_vals = sorted(top_k_list)
    max_k  = max(k_vals)
    W      = 10

    hdr  = f"{'Strategy':<10} {'MRR':>{W}}"
    for k in k_vals:
        hdr += f"  {'Rec@'+str(k):>{W}}  {'NDCG@'+str(k):>{W}}"
    hdr += f"  {'Rec_sem':>{W}}  {'Rec_con':>{W}}  {'Lat(ms)':>{W}}  {'p95(ms)':>{W}}"

    sep = "─" * len(hdr)
    print(f"\n{'='*len(hdr)}\nEXPERIMENT SUMMARY\n{'='*len(hdr)}")
    print(hdr)
    print(sep)

    for m in all_metrics:
        row = f"{m.strategy:<10} {m.mrr_score:>{W}.4f}"
        for k in k_vals:
            row += f"  {m.recall.get(k,0):>{W}.4f}  {m.ndcg.get(k,0):>{W}.4f}"
        row += (
            f"  {m.recall_semantic.get(max_k,0):>{W}.4f}"
            f"  {m.recall_constrained.get(max_k,0):>{W}.4f}"
            f"  {m.latency_mean_s*1000:>{W}.1f}"
            f"  {m.latency_p95_s*1000:>{W}.1f}"
        )
        print(row)

    n     = all_metrics[0].n_queries if all_metrics else 0
    n_con = int(n * _CONSTRAINED_FRACTION)
    print(f"{'='*len(hdr)}")
    print(f"  Queries: {n} total  ({n - n_con} semantic / {n_con} constrained)\n")


def save_summary_json(
    all_metrics: List[StrategyMetrics],
    path: str = EVAL_RESULTS_PATH,
) -> None:
    payload = []
    for m in all_metrics:
        payload.append({
            "strategy":           m.strategy,
            "n_queries":          m.n_queries,
            "mrr":                round(m.mrr_score, 6),
            "recall":             {str(k): round(v, 6) for k, v in m.recall.items()},
            "ndcg":               {str(k): round(v, 6) for k, v in m.ndcg.items()},
            "recall_semantic":    {str(k): round(v, 6) for k, v in m.recall_semantic.items()},
            "recall_constrained": {str(k): round(v, 6) for k, v in m.recall_constrained.items()},
            "ndcg_semantic":      {str(k): round(v, 6) for k, v in m.ndcg_semantic.items()},
            "ndcg_constrained":   {str(k): round(v, 6) for k, v in m.ndcg_constrained.items()},
            "latency": {
                "mean_ms": round(m.latency_mean_s * 1000, 3),
                "std_ms":  round(m.latency_std_s  * 1000, 3),
                "p95_ms":  round(m.latency_p95_s  * 1000, 3),
            },
        })
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    log.info("Results saved → %s", path)
