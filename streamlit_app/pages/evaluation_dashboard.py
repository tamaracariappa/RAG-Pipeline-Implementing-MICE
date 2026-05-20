"""
pages/evaluation_dashboard.py - Metrics dashboard from real eval_results.json.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd

from assets.styles import section_header
from charts.plotly_charts import (
    multi_metric_bar,
    recall_at_k_lines,
    metric_bar_chart,
    STRATEGY_LABELS,
    STRATEGY_COLORS,
)
from loaders.data_loader import load_eval_results, load_per_query_csv


STRATEGY_ORDER = ["A", "B", "B_prime", "C"]

METRIC_DISPLAY = {
    "recall_at_10": "Recall@10",
    "recall_at_5":  "Recall@5",
    "recall_at_1":  "Recall@1",
    "mrr":          "MRR",
    "ndcg_at_10":   "NDCG@10",
    "avg_latency_ms": "Avg Latency (ms)",
}


def _extract_strategy_row(data: dict, strategy: str) -> dict | None:
    """
    Extract and normalize one strategy row from eval_results.json.

    Supports BOTH:

    Flat format:
    {
        "A": {
            "recall_at_1": 0.1,
            "recall_at_5": 0.2,
            "recall_at_10": 0.3,
            "mrr": 0.4,
            "ndcg_at_10": 0.5,
            "avg_latency_ms": 12.3
        }
    }

    Nested format (current evaluation output):
    {
        "strategy": "A",
        "mrr": 0.4,
        "recall": {"1": 0.1, "5": 0.2, "10": 0.3},
        "ndcg": {"1": 0.2, "5": 0.3, "10": 0.5},
        "latency": {"mean_ms": 12.3}
    }
    """

    row = None

    # --------------------------------------------------
    # Case 1: Flat format
    # --------------------------------------------------
    if strategy in data:
        row = data[strategy]

    else:
        metrics = data.get("metrics", data.get("results", {}))
        if strategy in metrics:
            row = metrics[strategy]

    if row is None:
        return None

    # --------------------------------------------------
    # Already flat → return unchanged
    # --------------------------------------------------
    if "recall_at_10" in row:
        return row

    # --------------------------------------------------
    # Nested → normalize into flat keys
    # --------------------------------------------------
    recall = row.get("recall", {})
    ndcg = row.get("ndcg", {})
    latency = row.get("latency", {})

    return {
        "recall_at_1": recall.get("1", 0),
        "recall_at_5": recall.get("5", 0),
        "recall_at_10": recall.get("10", 0),
        "mrr": row.get("mrr", 0),
        "ndcg_at_10": ndcg.get("10", 0),
        "avg_latency_ms": latency.get("mean_ms", 0),
    }


def _demo_data() -> dict:
    """Synthetic data shown when eval_results.json is not yet generated."""
    return {
        "A":       {"recall_at_1": 0.32, "recall_at_5": 0.58,
                    "recall_at_10": 0.72, "mrr": 0.44, "ndcg_at_10": 0.61,
                    "avg_latency_ms": 12.4},
        "B":       {"recall_at_1": 0.38, "recall_at_5": 0.63,
                    "recall_at_10": 0.76, "mrr": 0.50, "ndcg_at_10": 0.65,
                    "avg_latency_ms": 45.2},
        "B_prime": {"recall_at_1": 0.40, "recall_at_5": 0.66,
                    "recall_at_10": 0.79, "mrr": 0.52, "ndcg_at_10": 0.67,
                    "avg_latency_ms": 48.7},
        "C":       {"recall_at_1": 0.36, "recall_at_5": 0.61,
                    "recall_at_10": 0.74, "mrr": 0.47, "ndcg_at_10": 0.63,
                    "avg_latency_ms": 14.1},
        "_demo": True,
    }


def render():
    st.markdown('<div class="page-title">Evaluation Dashboard</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">'
        'Recall, MRR, NDCG and latency metrics across all four retrieval strategies.'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Load data ────────────────────────────────────────────
    raw = load_eval_results()

    if raw is None:
        st.warning(
            "⚠️ `data/eval_results.json` not found. "
            "Showing demo data. Run `python main.py` to generate real results."
        )
        raw = _demo_data()
        demo_mode = True

    else:
        # Handle list-based evaluation outputs
        if isinstance(raw, list):
            normalized = {}

            for row in raw:
                strategy = row.get("strategy")

                if strategy:
                    normalized[strategy] = row

            raw = normalized
            demo_mode = False

        elif isinstance(raw, dict):
            demo_mode = raw.get("_demo", False)

        else:
            st.error("Unsupported evaluation result format.")
            return

    # Build per-strategy rows
    strategy_metrics = {}
    for s in STRATEGY_ORDER:
        row = _extract_strategy_row(raw, s)
        if row:
            strategy_metrics[s] = row

    if not strategy_metrics:
        st.error("Could not parse metric data from eval_results.json.")
        return

    # ── Primary metric cards ─────────────────────────────────
    section_header("Primary Metrics · Recall@10")

    mc = st.columns(len(strategy_metrics))
    best_recall = max(
        m.get("recall_at_10", 0) for m in strategy_metrics.values())
    for col, s in zip(mc, STRATEGY_ORDER):
        if s not in strategy_metrics:
            continue
        m = strategy_metrics[s]
        r10 = m.get("recall_at_10", 0)
        is_best = abs(r10 - best_recall) < 1e-6
        color = STRATEGY_COLORS.get(s, "#7886C7")
        with col:
            st.markdown(f"""
            <div style="background:#FFFFFF;border:1px solid #A9B5DF;
                        border-top:3px solid {color};border-radius:6px;
                        padding:1rem;text-align:center;">
                <div style="font-family:'IBM Plex Mono',monospace;font-size:1.4rem;
                            font-weight:600;color:{color};">{s}</div>
                <div style="font-family:'IBM Plex Mono',monospace;font-size:2rem;
                            color:#2D336B;font-weight:600;">{r10:.4f}</div>
                <div style="font-size:0.68rem;color:#7886C7;
                            text-transform:uppercase;letter-spacing:0.06em;">Recall@10</div>
                {"<div style='font-size:0.7rem;color:#7886C7;margin-top:0.2rem;'>★ Best</div>" if is_best else ""}
            </div>""", unsafe_allow_html=True)

    # ── MRR and latency row ──────────────────────────────────
    st.markdown("<br/>", unsafe_allow_html=True)
    m2 = st.columns(len(strategy_metrics) * 2)
    col_pairs = list(zip(m2[::2], m2[1::2]))
    for (c_mrr, c_lat), s in zip(col_pairs, STRATEGY_ORDER):
        if s not in strategy_metrics:
            continue
        m = strategy_metrics[s]
        color = STRATEGY_COLORS.get(s, "#7886C7")
        with c_mrr:
            st.markdown(f"""
            <div style="background:#FFFFFF;border:1px solid #A9B5DF;
                        border-radius:4px;padding:0.7rem;text-align:center;">
                <div style="font-family:'IBM Plex Mono',monospace;
                            font-size:1.2rem;color:{color};">
                    {m.get('mrr', 0):.4f}</div>
                <div style="font-size:0.65rem;color:#7886C7;text-transform:uppercase;
                            letter-spacing:0.06em;">{s} · MRR</div>
            </div>""", unsafe_allow_html=True)
        with c_lat:
            st.markdown(f"""
            <div style="background:#FFFFFF;border:1px solid #A9B5DF;
                        border-radius:4px;padding:0.7rem;text-align:center;">
                <div style="font-family:'IBM Plex Mono',monospace;
                            font-size:1.2rem;color:#7886C7;">
                    {m.get('avg_latency_ms', 0):.1f}ms</div>
                <div style="font-size:0.65rem;color:#7886C7;text-transform:uppercase;
                            letter-spacing:0.06em;">{s} · Latency</div>
            </div>""", unsafe_allow_html=True)

    # ── Bar charts ───────────────────────────────────────────
    st.markdown("<br/>", unsafe_allow_html=True)
    section_header("Metric Comparison")

    chart_tab1, chart_tab2, chart_tab3 = st.tabs(
        ["Recall@k", "MRR & NDCG", "Latency"])

    with chart_tab1:
        # Recall@k line chart
        k_data: dict[str, dict[int, float]] = {}
        for s, m in strategy_metrics.items():
            k_data[s] = {}
            for k in [1, 5, 10]:
                key = f"recall_at_{k}"
                if key in m:
                    k_data[s][k] = m[key]

        if any(k_data.values()):
            fig = recall_at_k_lines(k_data)
            st.plotly_chart(fig, use_container_width=True)

        # Also as grouped bars
        records = [
            {
                "strategy":    s,
                "Recall@1":    strategy_metrics[s].get("recall_at_1",  0),
                "Recall@5":    strategy_metrics[s].get("recall_at_5",  0),
                "Recall@10":   strategy_metrics[s].get("recall_at_10", 0),
            }
            for s in STRATEGY_ORDER if s in strategy_metrics
        ]
        if records:
            fig2 = multi_metric_bar(records, ["Recall@1","Recall@5","Recall@10"])
            st.plotly_chart(fig2, use_container_width=True)

    with chart_tab2:
        records2 = [
            {
                "strategy": s,
                "MRR":      strategy_metrics[s].get("mrr", 0),
                "NDCG@10":  strategy_metrics[s].get("ndcg_at_10", 0),
            }
            for s in STRATEGY_ORDER if s in strategy_metrics
        ]
        if records2:
            fig3 = multi_metric_bar(records2, ["MRR", "NDCG@10"])
            st.plotly_chart(fig3, use_container_width=True)

    with chart_tab3:
        from charts.plotly_charts import latency_bar
        lats = {
            s: strategy_metrics[s].get("avg_latency_ms", 0)
            for s in STRATEGY_ORDER if s in strategy_metrics
        }
        fig4 = latency_bar(lats)
        st.plotly_chart(fig4, use_container_width=True)
        st.markdown("""
        <div style="font-size:0.78rem;color:#7886C7;line-height:1.6;">
            A and C are single-pass FAISS searches. B and B′ search an expanded pool
            (top_k × 20) before filtering — higher precision but higher latency.
        </div>""", unsafe_allow_html=True)

    # ── Summary table ────────────────────────────────────────
    st.markdown("<br/>", unsafe_allow_html=True)
    section_header("Full Metrics Table")

    table_rows = []
    for s in STRATEGY_ORDER:
        if s not in strategy_metrics:
            continue
        m = strategy_metrics[s]
        table_rows.append({
            "Strategy":       STRATEGY_LABELS.get(s, s),
            "Recall@1":       round(m.get("recall_at_1",    0), 4),
            "Recall@5":       round(m.get("recall_at_5",    0), 4),
            "Recall@10":      round(m.get("recall_at_10",   0), 4),
            "MRR":            round(m.get("mrr",            0), 4),
            "NDCG@10":        round(m.get("ndcg_at_10",     0), 4),
            "Latency (ms)":   round(m.get("avg_latency_ms", 0), 1),
        })

    df_table = pd.DataFrame(table_rows)
    st.dataframe(df_table, use_container_width=True, hide_index=True)

    # ── Per-query analysis ───────────────────────────────────
    per_query = load_per_query_csv()
    if per_query is not None:
        st.markdown("<br/>", unsafe_allow_html=True)
        section_header("Per-Query Analysis", "from per_query_analysis.csv")

        with st.expander("Metadata Impact Distribution"):
            gain_col = "metadata_gain"
            if gain_col in per_query.columns:
                import plotly.graph_objects as go
                from charts.plotly_charts import THEME
                # Only unique queries (one gain per query, not per strategy)
                pq_unique = per_query.drop_duplicates(subset=["description"])
                gains = pq_unique[gain_col].dropna()

                fig_gain = go.Figure(go.Histogram(
                    x=gains,
                    nbinsx=30,
                    marker_color="#2D336B",
                    opacity=0.85,
                ))
                fig_gain.update_layout(
                    **THEME,
                    title=dict(
                        text="Distribution of Metadata Gain (best_meta_recall − A_recall)",
                        font_size=12),
                    xaxis=dict(title="Metadata Gain", gridcolor="#A9B5DF"),
                    yaxis=dict(title="Query Count",   gridcolor="#A9B5DF"),
                    margin=dict(l=50, r=20, t=50, b=50),
                    height=300,
                )
                st.plotly_chart(fig_gain, use_container_width=True)

                pos = (gains > 0.1).sum()
                neg = (gains < -0.1).sum()
                neu = len(gains) - pos - neg
                st.markdown(f"""
                <div style="font-size:0.8rem;color:#7886C7;line-height:1.8;">
                    Metadata helps (+0.1): <strong style="color:#2D336B;">{pos}</strong>
                    &nbsp;·&nbsp;
                    Metadata hurts (−0.1): <strong style="color:#5C6BC0;">{neg}</strong>
                    &nbsp;·&nbsp;
                    Neutral: <strong style="color:#7886C7;">{neu}</strong>
                </div>""", unsafe_allow_html=True)

        with st.expander("Sample per-query rows"):
            st.dataframe(per_query.head(100), use_container_width=True, height=300)
    else:
        st.caption(
            "Per-query CSV not found. Run `python main.py` to generate "
            "`data/per_query_analysis.csv`."
        )
