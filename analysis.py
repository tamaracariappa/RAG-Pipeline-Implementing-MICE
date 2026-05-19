"""
analysis.py - Result analysis tools for the FM RAG experiment.

Provides:
  QueryComparison         - per-query results across all four strategies
  compare_all_strategies  - run all strategies on one query, return comparison
  find_metadata_impact    - cases where metadata strategies beat A by ≥ threshold
  compute_win_matrix      - pairwise strategy win/loss/tie counts
  print_query_comparison  - human-readable per-query table
  export_comparisons_csv  - full per-query CSV for offline analysis
  run_full_analysis       - convenience wrapper used in main.py

All functions are stateless and accept plain Python / dataclass inputs.
No Milvus state is accessed here; results are passed in as arguments.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from evaluation import (
    TestCase,
    StrategyMetrics,
    recall_at_k,
    mrr,
    ndcg_at_k,
)
from retrieval import (
    FilterConfig,
    RetrievalResult,
    strategy_a,
    strategy_b,
    strategy_b_prime,
    strategy_c,
    deduplicate,
)
from config import DEFAULT_TOP_K, EVAL_TOP_K_LIST

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# QueryComparison
# ─────────────────────────────────────────────────────────────

@dataclass
class QueryComparison:
    """
    Side-by-side retrieval results and metrics for one test case
    across all four strategies.
    """
    test_case:   TestCase
    results:     Dict[str, List[RetrievalResult]]   # strategy → results
    # Recall@max_k for each strategy
    recall:      Dict[str, float] = field(default_factory=dict)
    mrr_scores:  Dict[str, float] = field(default_factory=dict)
    ndcg:        Dict[str, float] = field(default_factory=dict)
    # Rank of first relevant hit per strategy (None = not found)
    first_hit:   Dict[str, Optional[int]] = field(default_factory=dict)

    @property
    def metadata_gain(self) -> float:
        """
        Max recall improvement of metadata strategies (B, B', C) over A.
        Positive = metadata helps; negative = metadata hurts.
        """
        a = self.recall.get("A", 0.0)
        best_meta = max(
            self.recall.get("B",       0.0),
            self.recall.get("B_prime", 0.0),
            self.recall.get("C",       0.0),
        )
        return best_meta - a

    @property
    def best_strategy(self) -> str:
        """Strategy with highest recall."""
        return max(self.recall, key=lambda s: self.recall[s])

    @property
    def worst_strategy(self) -> str:
        """Strategy with lowest recall."""
        return min(self.recall, key=lambda s: self.recall[s])


# ─────────────────────────────────────────────────────────────
# Per-query comparison builder
# ─────────────────────────────────────────────────────────────

def compare_all_strategies(
    test_case: TestCase,
    top_k: int = DEFAULT_TOP_K,
    k_for_metrics: int = DEFAULT_TOP_K,
) -> QueryComparison:
    """
    Run all four strategies on *test_case* and return a QueryComparison.

    Args:
        test_case:      The query + relevant_woids + filter_config.
        top_k:          Number of candidates to fetch from each strategy.
        k_for_metrics:  k used when computing Recall@k / NDCG@k.
    """
    fc = test_case.filter_config

    results: Dict[str, List[RetrievalResult]] = {
        "A":       deduplicate(strategy_a(test_case.query_text, top_k)),
        "B":       deduplicate(strategy_b(test_case.query_text, fc, top_k)),
        "B_prime": deduplicate(strategy_b_prime(test_case.query_text, fc, top_k)),
        "C":       deduplicate(strategy_c(test_case.query_text, top_k)),
    }

    rel   = test_case.relevant_woids
    k     = k_for_metrics
    recall    = {s: recall_at_k(r, rel, k) for s, r in results.items()}
    mrr_sc    = {s: mrr(r, rel)            for s, r in results.items()}
    ndcg_sc   = {s: ndcg_at_k(r, rel, k)  for s, r in results.items()}

    first_hit: Dict[str, Optional[int]] = {}
    for s, r in results.items():
        hit = next((i + 1 for i, x in enumerate(r) if x.woid in rel), None)
        first_hit[s] = hit

    return QueryComparison(
        test_case  = test_case,
        results    = results,
        recall     = recall,
        mrr_scores = mrr_sc,
        ndcg       = ndcg_sc,
        first_hit  = first_hit,
    )


def build_comparisons(
    test_cases: List[TestCase],
    top_k: int = DEFAULT_TOP_K,
) -> List[QueryComparison]:
    """Run compare_all_strategies for every test case."""
    comparisons = []
    for i, tc in enumerate(test_cases, 1):
        try:
            cmp = compare_all_strategies(tc, top_k=top_k)
            comparisons.append(cmp)
        except Exception as exc:
            log.warning("Comparison failed for '%s': %s", tc.description, exc)
        if i % 50 == 0:
            log.info("  Compared %d / %d queries …", i, len(test_cases))
    return comparisons


# ─────────────────────────────────────────────────────────────
# Analysis functions
# ─────────────────────────────────────────────────────────────

def find_metadata_impact(
    comparisons: List[QueryComparison],
    gain_threshold: float = 0.2,
) -> List[QueryComparison]:
    """
    Return comparisons where any metadata strategy (B, B', C) improves
    Recall over A by at least *gain_threshold*.

    Sorted by metadata_gain descending.
    """
    impactful = [c for c in comparisons if c.metadata_gain >= gain_threshold]
    return sorted(impactful, key=lambda c: c.metadata_gain, reverse=True)


def find_metadata_hurt(
    comparisons: List[QueryComparison],
    loss_threshold: float = 0.2,
) -> List[QueryComparison]:
    """
    Return comparisons where ALL metadata strategies perform worse than A
    by at least *loss_threshold*.  Identifies cases where metadata filtering
    is counter-productive.
    """
    hurt = []
    for c in comparisons:
        a = c.recall.get("A", 0.0)
        meta_max = max(
            c.recall.get("B", 0.0),
            c.recall.get("B_prime", 0.0),
            c.recall.get("C", 0.0),
        )
        if a - meta_max >= loss_threshold:
            hurt.append(c)
    return sorted(hurt, key=lambda c: c.metadata_gain)


def compute_win_matrix(
    comparisons: List[QueryComparison],
) -> Dict[Tuple[str, str], Dict[str, int]]:
    """
    Pairwise win / loss / tie counts for all strategy pairs.

    Returns:
      {(A, B): {"wins": n, "losses": m, "ties": t}, ...}
      where "wins" = number of queries where A's recall > B's recall.
    """
    strategies = ["A", "B", "B_prime", "C"]
    matrix: Dict[Tuple[str, str], Dict[str, int]] = {}

    for i, s1 in enumerate(strategies):
        for s2 in strategies[i + 1:]:
            wins = losses = ties = 0
            for c in comparisons:
                r1 = c.recall.get(s1, 0.0)
                r2 = c.recall.get(s2, 0.0)
                if r1 > r2:
                    wins += 1
                elif r2 > r1:
                    losses += 1
                else:
                    ties += 1
            matrix[(s1, s2)] = {"wins": wins, "losses": losses, "ties": ties}

    return matrix


# ─────────────────────────────────────────────────────────────
# Printing / reporting
# ─────────────────────────────────────────────────────────────

def print_query_comparison(cmp: QueryComparison, show_hits: int = 3) -> None:
    """Print a formatted per-query comparison table."""
    tc = cmp.test_case
    print(f"\n{'─'*70}")
    print(f"Query  : {tc.query_text[:120]}")
    print(f"Type   : {tc.query_type}  |  Group: {tc.metadata_group}")
    print(f"Relevant WOIDs ({len(tc.relevant_woids)}): {list(tc.relevant_woids)[:5]} …")
    print(f"{'Strategy':<12} {'Recall':>8} {'MRR':>8} {'NDCG':>8} {'1stHit':>8}")
    print(f"{'─'*50}")
    for s in ("A", "B", "B_prime", "C"):
        fh = cmp.first_hit.get(s)
        print(
            f"{s:<12}"
            f"{cmp.recall.get(s, 0):>8.4f}"
            f"{cmp.mrr_scores.get(s, 0):>8.4f}"
            f"{cmp.ndcg.get(s, 0):>8.4f}"
            f"{'#'+str(fh) if fh else '—':>8}"
        )
    print(f"\n  Metadata gain: {cmp.metadata_gain:+.4f}  |  Best: {cmp.best_strategy}")
    if show_hits:
        best_s   = cmp.best_strategy
        top_hits = cmp.results.get(best_s, [])[:show_hits]
        print(f"\n  Top-{show_hits} from Strategy {best_s}:")
        for i, r in enumerate(top_hits, 1):
            marker = "✓" if r.woid in tc.relevant_woids else " "
            print(f"    {i}. [{marker}] {r.woid}  score={r.score:.4f}  {r.wo_description[:60]}")


def print_win_matrix(matrix: Dict[Tuple[str, str], Dict[str, int]]) -> None:
    """Print the pairwise win/loss/tie table."""
    print(f"\n{'='*55}\nPAIRWISE WIN / LOSS / TIE\n{'='*55}")
    print(f"{'Pair':<16} {'Wins':>6} {'Losses':>8} {'Ties':>6}")
    print("─" * 40)
    for (s1, s2), counts in sorted(matrix.items()):
        print(
            f"{s1} vs {s2:<10}"
            f"{counts['wins']:>6}"
            f"{counts['losses']:>8}"
            f"{counts['ties']:>6}"
        )
    print()


def print_impact_summary(comparisons: List[QueryComparison]) -> None:
    """Print counts of queries where metadata helps vs hurts."""
    gains  = find_metadata_impact(comparisons, gain_threshold=0.1)
    hurts  = find_metadata_hurt(comparisons,   loss_threshold=0.1)
    neutral = len(comparisons) - len(gains) - len(hurts)

    print(f"\n{'='*55}\nMETADATA IMPACT SUMMARY\n{'='*55}")
    print(f"  Total queries  : {len(comparisons)}")
    print(f"  Metadata helps : {len(gains)}  ({len(gains)/max(len(comparisons),1):.1%})")
    print(f"  Metadata hurts : {len(hurts)}  ({len(hurts)/max(len(comparisons),1):.1%})")
    print(f"  Neutral        : {neutral}  ({neutral/max(len(comparisons),1):.1%})")

    if gains:
        top = gains[0]
        print(f"\n  Largest gain  : {top.metadata_gain:+.4f}  [{top.test_case.description}]")
    if hurts:
        worst = hurts[0]
        print(f"  Largest loss  : {worst.metadata_gain:+.4f}  [{worst.test_case.description}]")
    print()


# ─────────────────────────────────────────────────────────────
# CSV export
# ─────────────────────────────────────────────────────────────

def export_comparisons_csv(
    comparisons: List[QueryComparison],
    path: str,
    strategies: Tuple[str, ...] = ("A", "B", "B_prime", "C"),
) -> None:
    """
    Write one row per (query, strategy) to a CSV file.
    Suitable for statistical analysis in R / Python / Excel.
    """
    rows = []
    for cmp in comparisons:
        tc = cmp.test_case
        for s in strategies:
            rows.append({
                "description":    tc.description,
                "query_type":     tc.query_type,
                "equipment":      tc.metadata_group[0],
                "facility_type":  tc.metadata_group[1],
                "n_relevant":     len(tc.relevant_woids),
                "query_text":     tc.query_text[:200],
                "strategy":       s,
                "recall":         round(cmp.recall.get(s, 0.0), 6),
                "mrr":            round(cmp.mrr_scores.get(s, 0.0), 6),
                "ndcg":           round(cmp.ndcg.get(s, 0.0), 6),
                "first_hit_rank": cmp.first_hit.get(s),
                "metadata_gain":  round(cmp.metadata_gain, 6),
            })

    if not rows:
        log.warning("No comparisons to export.")
        return

    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    log.info("Per-query CSV saved → %s  (%d rows)", path, len(rows))


# ─────────────────────────────────────────────────────────────
# Convenience wrapper
# ─────────────────────────────────────────────────────────────

def run_full_analysis(
    test_cases: List[TestCase],
    csv_path: str,
    top_k: int = DEFAULT_TOP_K,
    print_n_examples: int = 5,
) -> List[QueryComparison]:
    """
    Build comparisons, print summaries, export CSV.
    Called from main.py after run_evaluation().

    Args:
        test_cases:      Same set used in run_evaluation().
        csv_path:        Output path for per-query CSV.
        top_k:           Retrieval depth for comparisons.
        print_n_examples: Number of individual query comparisons to print.

    Returns:
        List of QueryComparison objects for further inspection.
    """
    log.info("Running full analysis on %d queries …", len(test_cases))
    comparisons = build_comparisons(test_cases, top_k=top_k)

    # Print a sample of per-query breakdowns
    impact_cases = find_metadata_impact(comparisons, gain_threshold=0.2)
    for cmp in (impact_cases or comparisons)[:print_n_examples]:
        print_query_comparison(cmp)

    # Aggregate summaries
    matrix = compute_win_matrix(comparisons)
    print_win_matrix(matrix)
    print_impact_summary(comparisons)

    # Export full CSV
    export_comparisons_csv(comparisons, csv_path)

    return comparisons
