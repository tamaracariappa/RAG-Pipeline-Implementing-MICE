"""
pages/embedding_viz.py - PCA 2D visualization of FAISS embedding spaces.

Samples vectors efficiently from FAISS flat indexes.
Never loads the full dataset into memory.
"""

from __future__ import annotations

import numpy as np
import streamlit as st

from assets.styles import section_header
from charts.plotly_charts import pca_scatter, vector_heatmap
from loaders.data_loader import (
    get_index_stats,
    sample_mice_vectors,
    sample_text_vectors,
)


def _run_pca(vecs: np.ndarray) -> tuple[np.ndarray, list[float]]:
    """PCA reduction to 2D using sklearn. Returns (coords, explained_variance)."""
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(vecs)
    return coords, pca.explained_variance_ratio_.tolist()


def _build_hover(meta: list[dict]) -> list[str]:
    texts = []
    for m in meta:
        woid  = m.get("WOID", "")
        equip = m.get("equipment", "")
        btype = m.get("Type", "")
        desc  = m.get("WODescription", "")[:80]
        texts.append(
            f"<b>{woid}</b><br>"
            f"Equipment: {equip}<br>"
            f"Type: {btype}<br>"
            f"<i>{desc}…</i>"
        )
    return texts


@st.cache_data(show_spinner=False, ttl=1800)
def _cached_pca(track: str, n_samples: int, color_by: str):
    """
    Cache-aware PCA computation.
    Returns (coords, labels, hover_texts, explained_var) or None on failure.
    """
    if track == "Text (Strategies A, B, B′)":
        vecs, meta = sample_text_vectors(n_samples)
    else:
        vecs, meta = sample_mice_vectors(n_samples)

    if vecs is None or meta is None or len(vecs) == 0:
        return None

    coords, expl = _run_pca(vecs)
    labels = [m.get(color_by, "unknown") or "unknown" for m in meta]
    hover  = _build_hover(meta)
    return coords, labels, hover, expl


def render():
    st.markdown('<div class="page-title">Embedding Visualization</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">'
        'PCA projection of the 768-dimensional embedding space. '
        'Points = individual work orders. Clustering indicates semantic similarity.'
        '</div>',
        unsafe_allow_html=True,
    )

    stats = get_index_stats()

    # ── Controls ─────────────────────────────────────────────
    section_header("Controls")
    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns(4)

    with ctrl1:
        track = st.selectbox(
            "Embedding Track",
            ["Text (Strategies A, B, B′)", "MICE (Strategy C)"],
        )
    with ctrl2:
        max_n = min(
            stats.get("text_total", 0) if "Text" in track
            else stats.get("mice_total", 0),
            5000,
        ) or 2000
        n_samples = st.slider("Sample size", 200, max_n,
                              min(1000, max_n), step=100)
    with ctrl3:
        color_by_label = st.selectbox("Color by", ["equipment", "Type"])
    with ctrl4:
        n_dims_heat = st.slider("Heatmap dims", 5, 30, 20)

    # ── PCA scatter ──────────────────────────────────────────
    section_header("2D Projection", "PCA")

    cache_key = f"{track}_{n_samples}_{color_by_label}"
    with st.spinner("Running PCA…"):
        result = _cached_pca(track, n_samples, color_by_label)

    if result is None:
        st.warning(
            "⚠️ FAISS index not found or empty. "
            "Build the indexes first with `python main.py`."
        )
        _show_pca_explainer()
        return

    coords, labels, hover, expl = result

    title = (
        f"{'TEXT' if 'Text' in track else 'MICE'} Embedding Space · "
        f"{n_samples} samples · colored by {color_by_label}"
    )
    fig = pca_scatter(coords, labels, hover,
                      title=title, explained_var=expl)
    st.plotly_chart(fig, use_container_width=True)

    # Explanation
    st.markdown(f"""
    <div style="font-size:0.78rem;color:#8890a8;line-height:1.6;
                background:#1a1d27;border:1px solid #2a2d3e;
                border-radius:4px;padding:0.8rem 1rem;">
        <b style="color:#e4e6f0;">Reading this chart:</b><br>
        Each dot is one work order encoded as a 768-dim vector and projected to 2D via PCA.
        Points that cluster together have similar embeddings → the retrieval engine ranks
        them as semantically close. Tight, well-separated clusters indicate the model
        has learned meaningful representations for {color_by_label} categories.<br><br>
        PC1 explains <b style="color:#4f8ef7;">{expl[0]:.1%}</b> of variance ·
        PC2 explains <b style="color:#4f8ef7;">{expl[1]:.1%}</b>
    </div>""", unsafe_allow_html=True)

    # ── MICE vs TEXT comparison ──────────────────────────────
    st.markdown("<br/>", unsafe_allow_html=True)
    section_header("Representation Comparison")
    with st.expander("Compare TEXT vs MICE clustering side-by-side"):
        c1, c2 = st.columns(2)
        with c1:
            with st.spinner("Computing TEXT PCA…"):
                text_result = _cached_pca(
                    "Text (Strategies A, B, B′)", n_samples, color_by_label)
            if text_result:
                tc, tl, th, te = text_result
                st.plotly_chart(
                    pca_scatter(tc, tl, th,
                                title=f"TEXT · {color_by_label}",
                                explained_var=te),
                    use_container_width=True,
                )
        with c2:
            with st.spinner("Computing MICE PCA…"):
                mice_result = _cached_pca(
                    "MICE (Strategy C)", n_samples, color_by_label)
            if mice_result:
                mc, ml, mh, me = mice_result
                st.plotly_chart(
                    pca_scatter(mc, ml, mh,
                                title=f"MICE · {color_by_label}",
                                explained_var=me),
                    use_container_width=True,
                )

        st.markdown("""
        <div style="font-size:0.78rem;color:#8890a8;line-height:1.6;margin-top:0.5rem;">
            MICE embeddings tend to form tighter, more separable clusters by
            metadata category because field labels ("equipment system: hvac") give the
            transformer a stable anchor for each dimension. TEXT embeddings capture
            richer free-text semantics but blur across metadata boundaries.
        </div>""", unsafe_allow_html=True)

    # ── Vector heatmap ───────────────────────────────────────
    st.markdown("<br/>", unsafe_allow_html=True)
    section_header("Raw Vector Values", f"first {n_dims_heat} dims · 20 samples")

    if result is not None:
        # Re-sample a tiny slice directly from the cached vectors
        vecs_small, meta_small = (
            sample_text_vectors(20) if "Text" in track
            else sample_mice_vectors(20)
        )
        if vecs_small is not None and len(vecs_small) > 0:
            fig_heat = vector_heatmap(
                vecs_small, n_dims=n_dims_heat,
                title=f"Embedding Vectors · {track} (first {n_dims_heat} dims, 20 rows)"
            )
            st.plotly_chart(fig_heat, use_container_width=True)
            st.markdown("""
            <div style="font-size:0.78rem;color:#8890a8;">
                Each row = one work order. Each column = one of 768 embedding dimensions.
                Red = positive activation · Blue = negative activation.
                Similar work orders produce similar color patterns.
            </div>""", unsafe_allow_html=True)


def _show_pca_explainer():
    """Static explainer shown when indexes are not built yet."""
    section_header("How Embeddings Work")
    st.markdown("""
    <div style="background:#1a1d27;border:1px solid #2a2d3e;border-radius:6px;
                padding:1.2rem;font-size:0.83rem;color:#8890a8;line-height:1.8;">
        <ol style="padding-left:1.2rem;">
            <li>Each work-order description is fed through BAAI/bge-base-en-v1.5.</li>
            <li>The model outputs a 768-dimensional float vector.</li>
            <li>Semantically similar texts produce vectors with high cosine similarity.</li>
            <li>PCA collapses 768 dims → 2 dims for visualization while preserving
                relative distances.</li>
            <li>MICE prepends structured labels before encoding, so metadata
                dimensions become spatially separable in the projected space.</li>
        </ol>
    </div>""", unsafe_allow_html=True)
