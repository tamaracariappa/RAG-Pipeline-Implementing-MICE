"""
charts/plotly_charts.py - Reusable Plotly figures for FM-RAG visualizations.

All functions return go.Figure objects; Streamlit rendering is done in pages.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ── Shared theme ────────────────────────────────────────────────────────────
THEME = dict(
    paper_bgcolor="#FFF2F2",
    plot_bgcolor="#FFF2F2",
    font_color="#2D336B",
    font_family="IBM Plex Mono, monospace"
)

STRATEGY_COLORS = {
    "A":       "#2D336B",   # deep navy  – primary accent
    "B":       "#7886C7",   # medium blue – secondary accent
    "B_prime": "#A9B5DF",   # soft blue   – soft accent
    "C":       "#5C6BC0",   # indigo      – between accent2 and accent
}

STRATEGY_LABELS = {
    "A":       "Strategy A",
    "B":       "Strategy B",
    "B_prime": "Strategy B′",
    "C":       "Strategy C",
}


def _base_layout(**kwargs) -> dict:
    layout = {**THEME}
    layout.update(kwargs)
    return layout


# ─────────────────────────────────────────────────────────────
# Metric bar chart
# ─────────────────────────────────────────────────────────────

def metric_bar_chart(
    data: Dict[str, Dict[str, float]],
    metric_name: str = "Recall@10",
    title: str = "",
) -> go.Figure:
    """
    Grouped bar chart comparing strategies on a single metric.

    data: {strategy: {metric_key: value, …}}
    """
    strategies = list(data.keys())
    values     = [data[s].get(metric_name, 0.0) for s in strategies]
    colors     = [STRATEGY_COLORS.get(s, "#7886C7") for s in strategies]
    labels     = [STRATEGY_LABELS.get(s, s) for s in strategies]

    fig = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            marker_color=colors,
            text=[f"{v:.4f}" for v in values],
            textposition="outside",
            textfont=dict(size=11, family="IBM Plex Mono"),
        )
    )
    fig.update_layout(
        **_base_layout(
            title=dict(text=title, font_size=13),
            yaxis=dict(
                range=[0, max(values) * 1.25 if values else 1],
                gridcolor="#A9B5DF",
                showgrid=True,
            ),
            xaxis=dict(showgrid=False),
            showlegend=False,
            margin=dict(l=40, r=20, t=50, b=40),
            height=320,
        )
    )
    return fig


def multi_metric_bar(
    records: List[dict],
    metrics: List[str],
    strategy_col: str = "strategy",
) -> go.Figure:
    """
    Grouped bars: one group per metric, one bar per strategy.
    records: list of {strategy, metric1, metric2, …}
    """
    df = pd.DataFrame(records)
    fig = go.Figure()
    for metric in metrics:
        for _, row in df.iterrows():
            s = row[strategy_col]
            fig.add_trace(go.Bar(
                name=f"{STRATEGY_LABELS.get(s, s)} - {metric}",
                x=[metric],
                y=[row.get(metric, 0.0)],
                marker_color=STRATEGY_COLORS.get(s, "#7886C7"),
                legendgroup=s,
                showlegend=(metric == metrics[0]),
                text=[f"{row.get(metric, 0.0):.4f}"],
                textposition="outside",
            ))
    fig.update_layout(
        **_base_layout(
            barmode="group",
            legend=dict(bgcolor="#FFFFFF", bordercolor="#A9B5DF", borderwidth=1),
            margin=dict(l=40, r=20, t=30, b=40),
            height=360,
            yaxis=dict(gridcolor="#A9B5DF", showgrid=True),
            xaxis=dict(showgrid=False),
        )
    )
    return fig


# ─────────────────────────────────────────────────────────────
# Recall@k line chart
# ─────────────────────────────────────────────────────────────

def recall_at_k_lines(
    data: Dict[str, Dict[int, float]],
) -> go.Figure:
    """
    Line chart of Recall@k across k values.
    data: {strategy: {k: recall_value}}
    """
    fig = go.Figure()
    for s, kv in data.items():
        ks = sorted(kv.keys())
        vals = [kv[k] for k in ks]
        fig.add_trace(go.Scatter(
            x=ks,
            y=vals,
            mode="lines+markers",
            name=STRATEGY_LABELS.get(s, s),
            line=dict(color=STRATEGY_COLORS.get(s, "#7886C7"), width=2),
            marker=dict(size=7),
        ))
    fig.update_layout(
        **_base_layout(
            xaxis=dict(title="k", gridcolor="#A9B5DF", showgrid=True),
            yaxis=dict(title="Recall@k", range=[0, 1.05],
                       gridcolor="#A9B5DF", showgrid=True),
            legend=dict(bgcolor="#FFFFFF", bordercolor="#A9B5DF", borderwidth=1),
            margin=dict(l=50, r=20, t=30, b=50),
            height=340,
        )
    )
    return fig


# ─────────────────────────────────────────────────────────────
# PCA scatter plot
# ─────────────────────────────────────────────────────────────

def pca_scatter(
    coords: np.ndarray,
    labels: List[str],
    hover_texts: List[str],
    color_col: str = "label",
    title: str = "Embedding Space (PCA 2D)",
    explained_var: Optional[List[float]] = None,
) -> go.Figure:
    """
    Interactive 2D PCA scatter colored by category label.
    coords: (n, 2) numpy array of PCA-reduced coordinates.
    """
    unique_labels = sorted(set(labels))
    palette = px.colors.qualitative.Set2 + px.colors.qualitative.Pastel1

    fig = go.Figure()
    for i, lbl in enumerate(unique_labels):
        mask = np.array([l == lbl for l in labels])
        fig.add_trace(go.Scatter(
            x=coords[mask, 0],
            y=coords[mask, 1],
            mode="markers",
            name=lbl[:30],
            marker=dict(
                color=palette[i % len(palette)],
                size=5,
                opacity=0.75,
            ),
            text=[t for t, m in zip(hover_texts, mask) if m],
            hovertemplate="%{text}<extra></extra>",
        ))

    xlab = f"PC1 ({explained_var[0]:.1%} var)" if explained_var else "PC1"
    ylab = f"PC2 ({explained_var[1]:.1%} var)" if explained_var else "PC2"

    fig.update_layout(
        **_base_layout(
            title=dict(text=title, font_size=13),
            xaxis=dict(title=xlab, gridcolor="#A9B5DF", showgrid=True),
            yaxis=dict(title=ylab, gridcolor="#A9B5DF", showgrid=True),
            legend=dict(bgcolor="#FFFFFF", bordercolor="#A9B5DF",
                        borderwidth=1, font_size=10),
            margin=dict(l=50, r=20, t=50, b=50),
            height=520,
        )
    )
    return fig


# ─────────────────────────────────────────────────────────────
# Score distribution histogram
# ─────────────────────────────────────────────────────────────

def score_histogram(
    scores_by_strategy: Dict[str, List[float]],
    title: str = "Score Distribution",
) -> go.Figure:
    fig = go.Figure()
    for s, scores in scores_by_strategy.items():
        fig.add_trace(go.Histogram(
            x=scores,
            name=STRATEGY_LABELS.get(s, s),
            marker_color=STRATEGY_COLORS.get(s, "#7886C7"),
            opacity=0.7,
            nbinsx=30,
        ))
    fig.update_layout(
        **_base_layout(
            barmode="overlay",
            title=dict(text=title, font_size=13),
            xaxis=dict(title="Cosine Similarity Score", gridcolor="#A9B5DF"),
            yaxis=dict(title="Count", gridcolor="#A9B5DF"),
            legend=dict(bgcolor="#FFFFFF", bordercolor="#A9B5DF", borderwidth=1),
            margin=dict(l=50, r=20, t=50, b=50),
            height=300,
        )
    )
    return fig


# ─────────────────────────────────────────────────────────────
# Vector heatmap (truncated dimensions)
# ─────────────────────────────────────────────────────────────

def vector_heatmap(
    vectors: np.ndarray,
    n_dims: int = 20,
    title: str = "Embedding Vectors (first 20 dims)",
) -> go.Figure:
    """Display a sample of vectors as a heatmap."""
    vecs = vectors[:, :n_dims]
    fig = go.Figure(go.Heatmap(
        z=vecs,
        colorscale="RdBu",
        zmid=0,
        showscale=True,
        colorbar=dict(thickness=12, tickfont=dict(size=9)),
        hovertemplate="Row %{y} · Dim %{x}<br>Value: %{z:.4f}<extra></extra>",
    ))
    fig.update_layout(
        **_base_layout(
            title=dict(text=title, font_size=13),
            xaxis=dict(title="Dimension index", showgrid=False),
            yaxis=dict(title="Work order", showgrid=False),
            margin=dict(l=60, r=20, t=50, b=50),
            height=320,
        )
    )
    return fig


# ─────────────────────────────────────────────────────────────
# Latency comparison
# ─────────────────────────────────────────────────────────────

def latency_bar(latencies: Dict[str, float]) -> go.Figure:
    """Horizontal bar chart of retrieval latencies."""
    labels = [STRATEGY_LABELS.get(s, s) for s in latencies]
    values = list(latencies.values())
    colors = [STRATEGY_COLORS.get(s, "#7886C7") for s in latencies]

    fig = go.Figure(go.Bar(
        y=labels,
        x=values,
        orientation="h",
        marker_color=colors,
        text=[f"{v:.1f} ms" for v in values],
        textposition="outside",
        textfont=dict(size=10, family="IBM Plex Mono"),
    ))
    fig.update_layout(
        **_base_layout(
            xaxis=dict(title="Latency (ms)", gridcolor="#A9B5DF"),
            yaxis=dict(showgrid=False),
            margin=dict(l=20, r=80, t=20, b=40),
            height=220,
            showlegend=False,
        )
    )
    return fig
