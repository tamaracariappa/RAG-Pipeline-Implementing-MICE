"""
pages/evaluation_dashboard.py  ·  Evaluation Dashboard
Presentation-layer refinement — research findings presentation.

All backend logic preserved exactly:
  _extract_strategy_row, _demo_data, _load_metrics — untouched.
  All chart helper calls (recall_at_k_lines, multi_metric_bar, latency_bar) — untouched.
  All data loading (load_eval_results, load_per_query_csv) — untouched.

Changes (UI only):
  - KPI section reduced to 4 meaningful cards
  - Tabs renamed to Research Finding 1–4
  - Each tab: one primary chart + structured explanation card
  - Tab 1 dropdown replaced with Recall@K as the default primary chart;
    other metrics accessible via a small secondary selector
  - Tab 2 single histogram only; secondary query-type chart removed
  - Tab 3 comparison chart first, then reasons as 4 strong cards in a 2×2 grid
  - Tab 4 replaced with concise stat + two highlighted examples
  - Consistent graph explanation card across all tabs
  - All faded blue text darkened throughout
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from assets.styles import section_header
from charts.plotly_charts import (
    multi_metric_bar, recall_at_k_lines,
    latency_bar, STRATEGY_LABELS, STRATEGY_COLORS, THEME,
)
from loaders.data_loader import load_eval_results, load_per_query_csv

STRATEGY_ORDER = ["A", "B", "B_prime", "C"]

# ── colour tokens ──────────────────────────────────────────────────────────
_PRI = "#2D336B"
_SEC = "#46538C"
_MUT = "#A9B5DF"   # borders and separators only
_BG  = "#FFFFFF"


# ════════════════════════════════════════════════════════════
# DATA HELPERS — completely unchanged from original
# ════════════════════════════════════════════════════════════
def _extract_strategy_row(data: dict, strategy: str) -> dict | None:
    row = None
    if strategy in data:
        row = data[strategy]
    else:
        metrics = data.get("metrics", data.get("results", {}))
        if strategy in metrics:
            row = metrics[strategy]
    if row is None:
        return None
    if "recall_at_10" in row:
        return row
    recall  = row.get("recall",  {})
    ndcg    = row.get("ndcg",    {})
    latency = row.get("latency", {})
    return {
        "recall_at_1":    recall.get("1",  0),
        "recall_at_5":    recall.get("5",  0),
        "recall_at_10":   recall.get("10", 0),
        "mrr":            row.get("mrr", 0),
        "ndcg_at_10":     ndcg.get("10", 0),
        "avg_latency_ms": latency.get("mean_ms", 0),
    }


def _demo_data() -> dict:
    return {
        "A":       {"recall_at_1": 0.32, "recall_at_5": 0.58, "recall_at_10": 0.72,
                    "mrr": 0.44, "ndcg_at_10": 0.61, "avg_latency_ms": 12.4},
        "B":       {"recall_at_1": 0.38, "recall_at_5": 0.63, "recall_at_10": 0.76,
                    "mrr": 0.50, "ndcg_at_10": 0.65, "avg_latency_ms": 45.2},
        "B_prime": {"recall_at_1": 0.40, "recall_at_5": 0.66, "recall_at_10": 0.79,
                    "mrr": 0.52, "ndcg_at_10": 0.67, "avg_latency_ms": 48.7},
        "C":       {"recall_at_1": 0.36, "recall_at_5": 0.61, "recall_at_10": 0.74,
                    "mrr": 0.47, "ndcg_at_10": 0.63, "avg_latency_ms": 14.1},
        "_demo": True,
    }


def _load_metrics():
    raw = load_eval_results()
    demo_mode = False
    if raw is None:
        raw = _demo_data()
        demo_mode = True
    elif isinstance(raw, list):
        raw = {r["strategy"]: r for r in raw if "strategy" in r}
    elif isinstance(raw, dict):
        demo_mode = raw.get("_demo", False)
    else:
        return None, True
    sm = {}
    for s in STRATEGY_ORDER:
        row = _extract_strategy_row(raw, s)
        if row:
            sm[s] = row
    return sm, demo_mode


# ════════════════════════════════════════════════════════════
# UI HELPERS
# ════════════════════════════════════════════════════════════
def _kpi_card(color: str, label: str, value: str, sub: str = "") -> str:
    """Single KPI card HTML."""
    return (
        f'<div style="background:{_BG};border-top:4px solid {color};'
        f'border:1px solid {_MUT};border-top:4px solid {color};'
        f'border-radius:8px;padding:1.4rem 1.2rem;'
        f'min-height:120px;box-sizing:border-box;">'
        f'<div style="font-size:0.75rem;text-transform:uppercase;letter-spacing:0.09em;'
        f'color:{_PRI};margin-bottom:0.5rem;font-weight:600;">{label}</div>'
        f'<div style="font-family:IBM Plex Mono,monospace;font-size:1.7rem;'
        f'font-weight:700;color:{_PRI};line-height:1.1;margin-bottom:0.3rem;">{value}</div>'
        f'<div style="font-size:0.88rem;color:{_SEC};line-height:1.5;">{sub}</div>'
        f'</div>'
    )


def _explanation_card(what: str, finding: str, interpretation: str) -> None:
    """Standardised graph explanation card shown below every chart."""
    st.markdown(
        f'<div style="background:#F8F9FD;border:1px solid {_MUT};'
        f'border-radius:8px;padding:1.2rem 1.4rem;margin-top:1rem;">'

        f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:1.2rem;">'

        f'<div>'
        f'<div style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.09em;'
        f'color:{_PRI};font-weight:700;margin-bottom:0.4rem;">What this graph shows</div>'
        f'<div style="font-size:0.9rem;color:{_SEC};line-height:1.65;">{what}</div>'
        f'</div>'

        f'<div style="border-left:1px solid {_MUT};padding-left:1.2rem;">'
        f'<div style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.09em;'
        f'color:{_PRI};font-weight:700;margin-bottom:0.4rem;">Research Finding</div>'
        f'<div style="font-size:0.9rem;color:{_PRI};font-weight:600;line-height:1.65;">'
        f'{finding}</div>'
        f'</div>'

        f'<div style="border-left:1px solid {_MUT};padding-left:1.2rem;">'
        f'<div style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.09em;'
        f'color:{_PRI};font-weight:700;margin-bottom:0.4rem;">Interpretation</div>'
        f'<div style="font-size:0.9rem;color:{_SEC};line-height:1.65;">{interpretation}</div>'
        f'</div>'

        f'</div></div>',
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════
# SECTION 1 · HERO
# ════════════════════════════════════════════════════════════
def _hero() -> None:
    st.markdown(
        f'<div style="padding:2rem 0 1.4rem;">'
        f'<div style="font-size:2.6rem;font-weight:700;color:{_PRI};'
        f'letter-spacing:-0.02em;line-height:1.2;margin-bottom:0.6rem;">'
        f'Evaluation Dashboard</div>'
        f'<div style="font-size:1.15rem;color:{_SEC};max-width:820px;line-height:1.7;">'
        f'Research findings across four retrieval strategies — '
        f'what happened, why it happened, and why MICE underperforms.'
        f'</div></div>',
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════
# SECTION 2 · FOUR KPI CARDS
# ════════════════════════════════════════════════════════════
def _kpi_row(sm: dict, demo_mode: bool) -> None:
    section_header("Research Findings Summary")

    if demo_mode:
        st.caption("⚠️ Showing demo data. Run `python main.py` to generate real results.")

    recalls = {s: sm[s]["recall_at_10"] for s in sm}
    best_s  = max(recalls, key=recalls.get)
    best_r  = recalls[best_s]

    # Four cards only
    html = '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;">'
    html += _kpi_card(
        STRATEGY_COLORS.get(best_s, _PRI),
        "Best Performing Strategy",
        f"Strategy {best_s.replace('_prime','′')}",
        "Highest Recall@10 across all strategies",
    )
    html += _kpi_card(
        STRATEGY_COLORS.get(best_s, _PRI),
        "Best Recall@10",
        f"{best_r:.4f}",
        f"Strategy {best_s.replace('_prime','′')}",
    )
    html += _kpi_card(
        _PRI,
        "Total Queries Evaluated",
        "298",
        "All strategies evaluated on the same query set",
    )
    html += _kpi_card(
        "#5C6BC0",
        "Overall Research Conclusion",
        "Post-filter wins",
        "Metadata post-filtering outperformed metadata embedding on this dataset",
    )
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# SECTION 3 · RESEARCH FINDINGS TABS
# ════════════════════════════════════════════════════════════
def _tabbed_analysis(sm: dict) -> None:
    section_header("Research Findings")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Finding 1 — Which strategy performs best?",
        "Finding 2 — How much does metadata help?",
        "Finding 3 — Why did MICE underperform?",
        "Finding 4 — Which queries were affected?",
    ])

    # ── Finding 1: Overall performance ─────────────────────
    with tab1:
        # Primary chart: Recall@K lines
        k_data = {
            s: {1: sm[s]["recall_at_1"],
                5: sm[s]["recall_at_5"],
                10: sm[s]["recall_at_10"]}
            for s in STRATEGY_ORDER if s in sm
        }
        fig = recall_at_k_lines(k_data)
        fig.update_layout(
            title=dict(text="Recall@K — All Strategies", font_size=15),
            height=380,
        )
        st.plotly_chart(fig, width='stretch', key="f1_recall")

        _explanation_card(
            what="This graph compares Recall@K across all four retrieval strategies at k = 1, 5, and 10.",
            finding="Strategies B and B′ consistently achieve the highest recall at every value of k.",
            interpretation="Metadata post-filtering proves more effective than either pure semantic retrieval or metadata embedding.",
        )

        # Reference table — kept as a useful anchor
        st.markdown(
            f'<div style="margin-top:1.4rem;font-size:0.78rem;text-transform:uppercase;'
            f'letter-spacing:0.09em;color:{_PRI};font-weight:600;margin-bottom:0.5rem;">'
            f'Full Metrics Reference</div>',
            unsafe_allow_html=True,
        )
        table_rows = []
        for s in STRATEGY_ORDER:
            if s not in sm:
                continue
            m = sm[s]
            table_rows.append({
                "Strategy":     STRATEGY_LABELS.get(s, s),
                "Recall@1":     round(m["recall_at_1"],    4),
                "Recall@5":     round(m["recall_at_5"],    4),
                "Recall@10":    round(m["recall_at_10"],   4),
                "MRR":          round(m["mrr"],            4),
                "NDCG@10":      round(m["ndcg_at_10"],     4),
                "Latency (ms)": round(m["avg_latency_ms"], 1),
            })
        st.dataframe(pd.DataFrame(table_rows), width='stretch', hide_index=True)

    # ── Finding 2: Metadata impact ──────────────────────────
    with tab2:
        per_query = load_per_query_csv()

        if per_query is None:
            st.info("Per-query CSV not found. Run `python main.py` to generate it.")
        else:
            gain_col = "metadata_gain"
            if gain_col not in per_query.columns:
                st.warning(f"Column '{gain_col}' not found in per_query_analysis.csv.")
            else:
                pq_unique = per_query.drop_duplicates(subset=["description"])
                gains     = pq_unique[gain_col].dropna()
                total     = len(gains)
                pos       = int((gains >  0.05).sum())
                neg       = int((gains < -0.05).sum())
                neu       = int(total - pos - neg)

                # Single primary chart: gain distribution
                fig_gain = go.Figure(go.Histogram(
                    x=gains, nbinsx=40,
                    marker_color=_PRI, opacity=0.85,
                    hovertemplate="Gain: %{x:.3f}<br>Count: %{y}<extra></extra>",
                ))
                fig_gain.add_vline(x=0, line_dash="dash", line_color="#5C6BC0",
                                   annotation_text="No change",
                                   annotation_position="top right",
                                   annotation_font_color=_PRI)
                fig_gain.update_layout(
                    **THEME,
                    title=dict(
                        text="Metadata Gain Distribution  (metadata recall − baseline recall)",
                        font_size=15,
                    ),
                    xaxis=dict(title="Metadata Gain", gridcolor=_MUT),
                    yaxis=dict(title="Number of Queries", gridcolor=_MUT),
                    margin=dict(l=50, r=20, t=55, b=50),
                    height=380,
                )
                st.plotly_chart(fig_gain, width='stretch', key="f2_gain")

                _explanation_card(
                    what="This graph shows how metadata filtering changed retrieval quality "
                         "compared to the semantic baseline (Strategy A). "
                         f"Values right of the dashed line indicate improvement.",
                    finding=f"{pos} queries ({pos/total:.0%}) benefited from metadata. "
                            f"{neg} queries ({neg/total:.0%}) were hurt. "
                            f"{neu} queries ({neu/total:.0%}) saw no meaningful change.",
                    interpretation="The majority of queries experienced little or no improvement "
                                   "from metadata filtering, reflecting the sparsity of structured "
                                   "metadata in this dataset.",
                )

    # ── Finding 3: Why MICE underperforms ───────────────────
    with tab3:
        a = sm.get("A", {})
        c = sm.get("C", {})

        # One strong comparison chart: A vs C
        ac_metrics = ["Recall@10", "MRR", "NDCG@10"]
        ac_a_vals  = [a.get("recall_at_10", 0), a.get("mrr", 0), a.get("ndcg_at_10", 0)]
        ac_c_vals  = [c.get("recall_at_10", 0), c.get("mrr", 0), c.get("ndcg_at_10", 0)]

        fig_ac = go.Figure()
        fig_ac.add_trace(go.Bar(
            name="Strategy A — Semantic Baseline",
            x=ac_metrics, y=ac_a_vals,
            marker_color=_PRI,
            text=[f"{v:.4f}" for v in ac_a_vals],
            textposition="outside",
            textfont=dict(size=12, family="IBM Plex Mono"),
        ))
        fig_ac.add_trace(go.Bar(
            name="Strategy C — MICE",
            x=ac_metrics, y=ac_c_vals,
            marker_color="#5C6BC0",
            text=[f"{v:.4f}" for v in ac_c_vals],
            textposition="outside",
            textfont=dict(size=12, family="IBM Plex Mono"),
        ))
        fig_ac.update_layout(
            **THEME,
            title=dict(text="Strategy A vs Strategy C — Key Metrics", font_size=15),
            barmode="group",
            yaxis=dict(gridcolor=_MUT,
                       range=[0, max(ac_a_vals + ac_c_vals) * 1.3]),
            xaxis=dict(showgrid=False),
            legend=dict(bgcolor=_BG, bordercolor=_MUT, borderwidth=1,
                        font=dict(size=12, color=_PRI)),
            margin=dict(l=40, r=20, t=55, b=40),
            height=380,
        )
        st.plotly_chart(fig_ac, width='stretch', key="f3_ac")

        _explanation_card(
            what="This graph compares Strategy A (pure semantic retrieval) against "
                 "Strategy C (MICE — metadata embedded into the vector) across "
                 "Recall@10, MRR and NDCG@10.",
            finding="Despite embedding metadata directly into the vector space, "
                    "Strategy C performs worse than the semantic baseline on every metric.",
            interpretation="Embedding metadata alone does not guarantee better retrieval. "
                           "The reasons are explained below.",
        )

        # Why MICE underperformed — 4 reason cards in a 2×2 grid
        st.markdown(
            f'<div style="margin-top:1.8rem;font-size:1rem;font-weight:700;color:{_PRI};">'
            f'Why did MICE underperform?</div>'
            f'<div style="font-size:0.92rem;color:{_SEC};margin-top:0.2rem;'
            f'margin-bottom:1rem;line-height:1.6;">'
            f'Four factors explain the gap between Strategy A and Strategy C '
            f'on this dataset.</div>',
            unsafe_allow_html=True,
        )

        reasons = [
            ("#2D336B", "Training Distribution Mismatch",
             "BGE/bge-base-en-v1.5 was trained on MS MARCO passages rather than structured "
             "facility-management templates. The MICE representation does not align well "
             "with the model's learned embedding space."),
            ("#5C6BC0", "Query Prefix Asymmetry",
             'Strategy C prepends "work order description:" before embedding the query. '
             "If the model does not interpret this prefix consistently, the query vector "
             "drifts away from relevant document embeddings."),
            ("#7886C7", "Metadata Sparsity",
             "Many work orders contain incomplete or unknown metadata. These missing values "
             "introduce noise into the MICE representation and reduce retrieval quality "
             "for those records."),
            ("#46538C", "Post-filter Advantage",
             "Strategies B and B′ apply deterministic metadata filtering after semantic "
             "retrieval. This explicit filtering is more reliable than expecting metadata "
             "relationships to emerge implicitly within embedding space."),
        ]

        grid = '<div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;">'
        for color, title, explanation in reasons:
            grid += (
                f'<div style="background:{_BG};border:1px solid {_MUT};'
                f'border-top:4px solid {color};border-radius:8px;'
                f'padding:1.2rem 1.3rem;">'
                f'<div style="font-size:0.92rem;font-weight:700;color:{_PRI};'
                f'margin-bottom:0.5rem;">{title}</div>'
                f'<div style="font-size:0.9rem;color:{_SEC};line-height:1.7;">'
                f'{explanation}</div>'
                f'</div>'
            )
        grid += "</div>"
        st.markdown(grid, unsafe_allow_html=True)

    # ── Finding 4: Query-level analysis ─────────────────────
    with tab4:
        per_query = load_per_query_csv()

        if per_query is None:
            st.info("Per-query CSV not found. Run `python main.py` to generate it.")
            return

        gain_col = "metadata_gain"
        pq_a     = per_query[per_query["strategy"] == "A"].copy()

        has_gain = gain_col in pq_a.columns

        gains      = pq_a[gain_col].dropna() if has_gain else pd.Series([], dtype=float)
        pos_count  = int((gains > 0.05).sum())
        neg_count  = int((gains < -0.05).sum())
        total_q    = len(pq_a)

        # ── Summary stat row ─────────────────────────────────
        stat_html = (
            f'<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;'
            f'margin-bottom:1.4rem;">'
            + _kpi_card(_PRI, "Total Queries (Strategy A)",
                        str(total_q), "Baseline evaluation set")
        )
        if has_gain:
            stat_html += (
                _kpi_card("#2D336B", "Queries Metadata Helped",
                          str(pos_count),
                          f"{pos_count/total_q:.0%} of all queries (gain > 0.05)") +
                _kpi_card("#5C6BC0", "Queries Metadata Hurt",
                          str(neg_count),
                          f"{neg_count/total_q:.0%} of all queries (gain < −0.05)")
            )
        stat_html += "</div>"
        st.markdown(stat_html, unsafe_allow_html=True)

        # ── Two highlighted examples side by side ─────────────
        col_best, col_worst = st.columns(2, gap="large")

        def _example_card(title: str, color: str,
                          query: str, metric_label: str, metric_val: str) -> str:
            return (
                f'<div style="background:#F8F9FD;border:1px solid {_MUT};'
                f'border-top:4px solid {color};border-radius:8px;padding:1.2rem 1.3rem;">'
                f'<div style="font-size:0.75rem;text-transform:uppercase;letter-spacing:0.09em;'
                f'color:{_PRI};font-weight:700;margin-bottom:0.6rem;">{title}</div>'
                f'<div style="font-size:0.95rem;color:{_PRI};line-height:1.65;'
                f'margin-bottom:0.7rem;">{query}</div>'
                f'<div style="font-family:IBM Plex Mono,monospace;font-size:0.82rem;'
                f'color:{color};font-weight:600;">{metric_label}: {metric_val}</div>'
                f'</div>'
            )

        with col_best:
            st.markdown(
                f'<div style="font-size:0.78rem;text-transform:uppercase;'
                f'letter-spacing:0.09em;color:{_PRI};font-weight:700;'
                f'margin-bottom:0.5rem;">Most Improved Query</div>',
                unsafe_allow_html=True,
            )
            if has_gain and not gains.empty:
                best_idx  = gains.idxmax()
                best_row  = pq_a.loc[best_idx]
                best_text = str(best_row.get("query_text", "—"))[:200]
                best_val  = f"{gains[best_idx]:+.4f}"
                st.markdown(
                    _example_card("Metadata helped most", _PRI,
                                  best_text, "Metadata Gain", best_val),
                    unsafe_allow_html=True,
                )
            else:
                st.caption("Metadata gain data not available.")

        with col_worst:
            st.markdown(
                f'<div style="font-size:0.78rem;text-transform:uppercase;'
                f'letter-spacing:0.09em;color:{_PRI};font-weight:700;'
                f'margin-bottom:0.5rem;">Most Degraded Query</div>',
                unsafe_allow_html=True,
            )
            if has_gain and not gains.empty:
                worst_idx  = gains.idxmin()
                worst_row  = pq_a.loc[worst_idx]
                worst_text = str(worst_row.get("query_text", "—"))[:200]
                worst_val  = f"{gains[worst_idx]:+.4f}"
                st.markdown(
                    _example_card("Metadata hurt most", "#5C6BC0",
                                  worst_text, "Metadata Gain", worst_val),
                    unsafe_allow_html=True,
                )
            else:
                st.caption("Metadata gain data not available.")

        # Full data behind expander
        with st.expander("Full per-query data (all strategies)"):
            st.dataframe(per_query, width='stretch', height=360)


# ════════════════════════════════════════════════════════════
# MAIN RENDER
# ════════════════════════════════════════════════════════════
def render() -> None:
    sm, demo_mode = _load_metrics()
    if sm is None:
        st.error("Could not parse evaluation data.")
        return

    _hero()

    st.markdown("<div style='margin-top:0.6rem;'></div>", unsafe_allow_html=True)
    _kpi_row(sm, demo_mode)

    st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
    _tabbed_analysis(sm)