"""
pages/embedding_viz.py - Embedding Space Explorer.

Focused visual comparison of TEXT vs MICE embedding organisation.
All PCA computation, loaders, and chart functions preserved unchanged.
"""

from __future__ import annotations

import numpy as np
import streamlit as st

from charts.plotly_charts import pca_scatter
from loaders.data_loader import (
    get_index_stats,
    load_config,
    sample_mice_vectors,
    sample_text_vectors,
)


# ── PCA helpers (unchanged) ──────────────────────────────────

def _run_pca(vecs: np.ndarray) -> tuple[np.ndarray, list[float]]:
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
            f"<i>{desc}\u2026</i>"
        )
    return texts


@st.cache_data(show_spinner=False, ttl=1800)
def _cached_pca(track: str, n_samples: int, color_by: str):
    if track == "TEXT":
        vecs, meta = sample_text_vectors(n_samples)
    else:
        vecs, meta = sample_mice_vectors(n_samples)
    if vecs is None or meta is None or len(vecs) == 0:
        return None
    coords, expl = _run_pca(vecs)
    labels = [m.get(color_by, "unknown") or "unknown" for m in meta]
    hover  = _build_hover(meta)
    return coords, labels, hover, expl


# ── SVG icons (inline, no external dependencies) ─────────────

ICON_CHIP = '<svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#2D336B" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><rect x="7" y="7" width="10" height="10" rx="1"/><path d="M7 9H5M7 12H5M7 15H5M17 9h2M17 12h2M17 15h2M9 7V5M12 7V5M15 7V5M9 17v2M12 17v2M15 17v2"/></svg>'
ICON_BOX  = '<svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#2D336B" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/><polyline points="3.27 6.96 12 12.01 20.73 6.96"/><line x1="12" y1="22.08" x2="12" y2="12"/></svg>'
ICON_PROJ = '<svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#2D336B" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>'
ICON_DOTS = '<svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#2D336B" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="6" cy="6" r="1.5" fill="#2D336B"/><circle cx="12" cy="4" r="1.5" fill="#2D336B"/><circle cx="18" cy="8" r="1.5" fill="#2D336B"/><circle cx="5" cy="13" r="1.5" fill="#2D336B"/><circle cx="14" cy="16" r="1.5" fill="#2D336B"/><circle cx="19" cy="14" r="1.5" fill="#2D336B"/><circle cx="9" cy="19" r="1.5" fill="#2D336B"/><circle cx="16" cy="20" r="1.5" fill="#2D336B"/></svg>'

# Observation card icons
ICON_LINK   = '<svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#2D336B" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M12 2v3M12 19v3M4.22 4.22l2.12 2.12M17.66 17.66l2.12 2.12M2 12h3M19 12h3M4.22 19.78l2.12-2.12M17.66 6.34l2.12-2.12"/></svg>'
ICON_CLUSTER= '<svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#2D336B" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="6" height="6" rx="1"/><rect x="15" y="3" width="6" height="6" rx="1"/><rect x="3" y="15" width="6" height="6" rx="1"/><rect x="15" y="15" width="6" height="6" rx="1"/></svg>'
ICON_META   = '<svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#2D336B" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>'
ICON_NEIGH  = '<svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#2D336B" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="8" width="5" height="5" rx="1"/><rect x="10" y="3" width="5" height="5" rx="1"/><rect x="16" y="11" width="5" height="5" rx="1"/><rect x="8" y="15" width="5" height="5" rx="1"/><line x1="8" y1="10.5" x2="10" y2="5.5"/><line x1="15" y1="5.5" x2="16" y2="11"/><line x1="13" y1="15" x2="17.5" y2="14"/></svg>'


# ── Page ─────────────────────────────────────────────────────

def render():
    st.markdown("""
    <style>
    .main .block-container {
        max-width: 1600px !important;
        padding: 2rem 3rem 4rem !important;
    }

    /* ── Hero ── */
    .es-hero {
        padding: 2rem 0 1.8rem;
        border-bottom: 1px solid var(--border);
        margin-bottom: 2rem;
    }
    .es-eyebrow {
        font-family: var(--mono);
        font-size: 0.72rem;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: var(--accent2);
        margin-bottom: 0.55rem;
    }
    .es-title {
        font-size: 2.6rem;
        font-weight: 700;
        color: var(--accent);
        letter-spacing: -0.02em;
        line-height: 1.1;
        margin-bottom: 0.6rem;
    }
    .es-subtitle {
        font-size: 1.05rem;
        color: #4A5490;
        line-height: 1.6;
        margin: 0;
    }

    /* ── Section divider + label ── */
    .es-sec-div {
        margin: 2rem 0 1.4rem;
        border: none;
        border-top: 1px solid var(--border);
    }
    .es-sec-label {
        font-family: var(--mono);
        font-size: 0.68rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: var(--accent2);
        margin-bottom: 0.6rem;
    }

    /* ── Control bar ── */
    .es-control-heading {
        display: flex;
        align-items: baseline;
        gap: 0.55rem;
        margin-bottom: 1rem;
    }
    .es-control-heading-main {
        font-size: 1.05rem;
        font-weight: 700;
        color: var(--accent);
        letter-spacing: 0;
    }
    .es-control-heading-sub {
        font-family: var(--mono);
        font-size: 0.65rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--muted);
    }

    /* ── Container borders (st.container border=True) ── */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        background: #FFFFFF !important;
        overflow: hidden !important;
    }
    .es-plot-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 1rem 1.4rem 0.9rem;
        border-bottom: 1px solid var(--border);
    }
    .es-plot-track-label {
        font-family: var(--mono);
        font-size: 0.65rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: var(--muted);
        margin-bottom: 0.25rem;
    }
    .es-plot-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: var(--accent);
    }
    .es-plot-badges {
        display: flex;
        gap: 0.3rem;
        margin-bottom: 0.25rem;
        justify-content: flex-end;
    }
    .es-badge {
        font-family: var(--mono);
        font-size: 0.75rem;
        font-weight: 600;
        background: #2D336B;
        color: #FFF2F2;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
    }
    .es-badge.light {
        background: #A9B5DF;
        color: #2D336B;
    }
    .es-variance-note {
        font-family: var(--mono);
        font-size: 0.75rem;
        color: var(--muted);
        text-align: right;
    }

    /* ── Observation panel ── */
    .es-obs-outer {
        background: #FFFFFF;
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 1.6rem 1.8rem 1.4rem;
    }
    .es-obs-heading {
        font-family: var(--mono);
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: var(--accent);
        margin-bottom: 1.4rem;
    }
    .es-obs-cards {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1.2rem;
    }
    .es-obs-card {
        display: flex;
        flex-direction: column;
        gap: 0.6rem;
    }
    .es-obs-icon {
        width: 44px;
        height: 44px;
        background: #F0F3FF;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
    }
    .es-obs-card-title {
        font-size: 0.97rem;
        font-weight: 700;
        color: var(--accent);
        line-height: 1.3;
    }
    .es-obs-card-desc {
        font-size: 0.9rem;
        color: #4A5490;
        line-height: 1.55;
    }

    /* ── Representation difference ── */
    .es-repr-grid {
        display: grid;
        grid-template-columns: 5fr 7fr;
        gap: 1rem;
    }
    .es-repr-card {
        background: #FFFFFF;
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 1.5rem 1.8rem;
    }
    .es-repr-track {
        font-family: var(--mono);
        font-size: 0.68rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: var(--muted);
        margin-bottom: 0.5rem;
    }
    .es-repr-name {
        font-size: 1.4rem;
        font-weight: 700;
        color: var(--accent);
        margin-bottom: 1rem;
    }
    .es-repr-fields {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-bottom: 1rem;
    }
    .es-repr-field {
        font-family: var(--mono);
        font-size: 0.8rem;
        color: #2D336B;
        background: #F0F3FF;
        border: 1px solid var(--border);
        border-radius: 4px;
        padding: 0.18rem 0.5rem;
        white-space: nowrap;
    }
    .es-repr-arrow {
        font-size: 1.5rem;
        color: var(--accent2);
        margin: 0.6rem 0 0.5rem;
        display: block;
    }
    .es-repr-result {
        font-size: 1rem;
        font-weight: 700;
        color: #FFFFFF;
        background: #2D336B;
        border-radius: 6px;
        padding: 0.65rem 1rem;
        display: inline-block;
    }

    /* ── Footer ── */
    .es-footer {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
    }
    .es-footer-item {
        background: #FFFFFF;
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 1.1rem 1.3rem;
        display: flex;
        align-items: flex-start;
        gap: 0.9rem;
    }
    .es-footer-icon {
        width: 36px;
        height: 36px;
        background: #F0F3FF;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
        margin-top: 0.1rem;
    }
    .es-footer-text {}
    .es-footer-label {
        font-family: var(--mono);
        font-size: 0.65rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--muted);
        margin-bottom: 0.25rem;
    }
    .es-footer-value {
        font-size: 1rem;
        font-weight: 700;
        color: var(--accent);
        word-break: break-word;
        line-height: 1.3;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Load config ───────────────────────────────────────────
    stats  = get_index_stats()
    config = load_config()
    embed_model = config.get("embedding_model", "BAAI/bge-base-en-v1.5") or "BAAI/bge-base-en-v1.5"
    embed_dim   = stats.get("dim") or config.get("embedding_dim", 768) or 768

    # ════════════════════════════════════════════════════════
    # SECTION 1 — HERO
    # ════════════════════════════════════════════════════════
    st.markdown("""
    <div class="es-hero">
        <div class="es-eyebrow">FM-RAG Research Platform \u00b7 Embedding Space Explorer</div>
        <div class="es-title">Embedding Space Explorer</div>
        <div class="es-subtitle">
            Visual comparison of how semantic and metadata-aware representations
            organise Facility Management work orders in vector space.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════
    # SECTION 2 — CONTROL BAR
    # ════════════════════════════════════════════════════════
    st.markdown('<div class="es-control-label">Controls \u2014 apply to both visualisations</div>',
                unsafe_allow_html=True)

    max_text = min(stats.get("text_total", 0), 5000) or 2000
    max_mice = min(stats.get("mice_total", 0), 5000) or 2000
    max_n    = max(max_text, max_mice)

    _pad1, cc1, cc2, _pad2 = st.columns([1, 2, 4, 1], gap="large")
    with cc1:
        color_by = st.selectbox(
            "Color by",
            ["equipment", "Type"],
            key="es_color",
        )
    with cc2:
        n_samples = st.slider(
            "Sample size",
            200, max_n, min(1000, max_n), step=100,
            key="es_samples",
        )

    # ════════════════════════════════════════════════════════
    # SECTION 3 — SIDE-BY-SIDE COMPARISON
    # ════════════════════════════════════════════════════════
    with st.spinner("Computing both embedding spaces\u2026"):
        text_result = _cached_pca("TEXT", n_samples, color_by)
        mice_result = _cached_pca("MICE", n_samples, color_by)

    if text_result is None and mice_result is None:
        st.warning(
            "\u26a0\ufe0f FAISS indexes not found or empty. "
            "Build the indexes first with `python main.py`."
        )
        return

    t_var = f"PC1 {text_result[3][0]:.1%} \u00b7 PC2 {text_result[3][1]:.1%}" if text_result else "\u2014"
    m_var = f"PC1 {mice_result[3][0]:.1%} \u00b7 PC2 {mice_result[3][1]:.1%}" if mice_result else "\u2014"

    plot_col1, plot_col2 = st.columns(2, gap="medium")

    with plot_col1:
        with st.container(border=True):
            st.markdown(f"""
            <div class="es-plot-header">
                <div>
                    <div class="es-plot-track-label">Embedding Track 1</div>
                    <div class="es-plot-title">TEXT Embedding Space</div>
                </div>
                <div>
                    <div class="es-plot-badges">
                        <span class="es-badge">A</span>
                        <span class="es-badge">B</span>
                        <span class="es-badge">B\u2032</span>
                    </div>
                    <div class="es-variance-note">{t_var}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            if text_result:
                tc, tl, th, te = text_result
                fig_t = pca_scatter(
                    tc, tl, th,
                    title=f"TEXT \u00b7 {color_by} \u00b7 {n_samples:,} samples",
                    explained_var=te,
                )
                fig_t.update_layout(height=540)
                st.plotly_chart(fig_t, width='stretch')
            else:
                st.info("TEXT index unavailable.")

    with plot_col2:
        with st.container(border=True):
            st.markdown(f"""
            <div class="es-plot-header">
                <div>
                    <div class="es-plot-track-label">Embedding Track 2</div>
                    <div class="es-plot-title">MICE Embedding Space</div>
                </div>
                <div>
                    <div class="es-plot-badges">
                        <span class="es-badge light">C</span>
                    </div>
                    <div class="es-variance-note">{m_var}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            if mice_result:
                mc, ml, mh, me = mice_result
                fig_m = pca_scatter(
                    mc, ml, mh,
                    title=f"MICE \u00b7 {color_by} \u00b7 {n_samples:,} samples",
                    explained_var=me,
                )
                fig_m.update_layout(height=540)
                st.plotly_chart(fig_m, width='stretch')
            else:
                st.info("MICE index unavailable.")

    # ════════════════════════════════════════════════════════
    # SECTION 4 — OBSERVATION PANEL
    # ════════════════════════════════════════════════════════
    st.markdown('<hr class="es-sec-div">', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="es-obs-outer">
        <div class="es-obs-heading">What to look for</div>
        <div class="es-obs-cards">
            <div class="es-obs-card">
                <div class="es-obs-icon">{ICON_LINK}</div>
                <div class="es-obs-card-title">Similarity = Proximity</div>
                <div class="es-obs-card-desc">Points closer together represent more similar work orders in the embedding space.</div>
            </div>
            <div class="es-obs-card">
                <div class="es-obs-icon">{ICON_CLUSTER}</div>
                <div class="es-obs-card-title">Cluster Separation</div>
                <div class="es-obs-card-desc">Compare how clearly different equipment types form distinct groups in each space.</div>
            </div>
            <div class="es-obs-card">
                <div class="es-obs-icon">{ICON_META}</div>
                <div class="es-obs-card-title">Metadata Influence</div>
                <div class="es-obs-card-desc">Observe whether metadata categories form clearer groups in the MICE space.</div>
            </div>
            <div class="es-obs-card">
                <div class="es-obs-icon">{ICON_NEIGH}</div>
                <div class="es-obs-card-title">Neighbourhood Formation</div>
                <div class="es-obs-card-desc">Compare local neighbourhoods to see how relationships differ between TEXT and MICE.</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════
    # SECTION 5 — REPRESENTATION DIFFERENCE
    # ════════════════════════════════════════════════════════
    st.markdown('<hr class="es-sec-div">', unsafe_allow_html=True)
    st.markdown('<div class="es-sec-label">Why the spaces differ</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="es-repr-grid">
        <div class="es-repr-card">
            <div class="es-repr-track">Track 1 \u00b7 Strategies A, B, B\u2032</div>
            <div class="es-repr-name">TEXT</div>
            <div class="es-repr-fields">
                <span class="es-repr-field">Building Name</span>
                <span class="es-repr-field">Facility Type</span>
                <span class="es-repr-field">Work Description</span>
            </div>
            <div class="es-repr-arrow">\u2193</div>
            <span class="es-repr-result">Semantic Similarity</span>
        </div>
        <div class="es-repr-card">
            <div class="es-repr-track">Track 2 \u00b7 Strategy C</div>
            <div class="es-repr-name">MICE</div>
            <div class="es-repr-fields">
                <span class="es-repr-field">Building ID</span>
                <span class="es-repr-field">Building Name</span>
                <span class="es-repr-field">Facility Type</span>
                <span class="es-repr-field">Equipment System</span>
                <span class="es-repr-field">Work Description</span>
            </div>
            <div class="es-repr-arrow">\u2193</div>
            <span class="es-repr-result">Metadata-Aware Similarity</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════
    # SECTION 6 — FOOTER
    # ════════════════════════════════════════════════════════
    st.markdown('<hr class="es-sec-div">', unsafe_allow_html=True)

    samples_shown = n_samples if (text_result or mice_result) else 0
    st.markdown(f"""
    <div class="es-footer">
        <div class="es-footer-item">
            <div class="es-footer-icon">{ICON_CHIP}</div>
            <div class="es-footer-text">
                <div class="es-footer-label">Model</div>
                <div class="es-footer-value">{embed_model}</div>
            </div>
        </div>
        <div class="es-footer-item">
            <div class="es-footer-icon">{ICON_BOX}</div>
            <div class="es-footer-text">
                <div class="es-footer-label">Dimensions</div>
                <div class="es-footer-value">{embed_dim}</div>
            </div>
        </div>
        <div class="es-footer-item">
            <div class="es-footer-icon">{ICON_PROJ}</div>
            <div class="es-footer-text">
                <div class="es-footer-label">Projection</div>
                <div class="es-footer-value">PCA (2 components)</div>
            </div>
        </div>
        <div class="es-footer-item">
            <div class="es-footer-icon">{ICON_DOTS}</div>
            <div class="es-footer-text">
                <div class="es-footer-label">Samples Visualised</div>
                <div class="es-footer-value">{samples_shown:,}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)