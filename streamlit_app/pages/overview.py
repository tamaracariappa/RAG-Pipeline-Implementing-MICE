"""
pages/overview.py - FM-RAG Research Landing Dashboard.

Redesigned for university evaluators and research reviewers.
All existing loaders and session state behaviour preserved.
"""

import streamlit as st
from assets.styles import section_header
from loaders.data_loader import get_index_stats, load_config


# ── Static data ──────────────────────────────────────────────

PIPELINE_STAGES = [
    ("Dataset",       "~2.5M FM work orders"),
    ("Preprocessing", "Normalise & clean text"),
    ("Embeddings",    "BGE-base-en \u2192 768-dim"),
    ("FAISS",         "Dual flat IP indexes"),
    ("Retrieval",     "4 metadata strategies"),
    ("Evaluation",    "Recall \u00b7 MRR \u00b7 NDCG"),
]

STRATEGIES = [
    ("A",       "#2D336B", "Semantic Baseline",
     "Pure vector similarity. No metadata. Reference condition for all comparisons."),
    ("B",       "#4A5490", "Metadata Post-Filter",
     "Retrieve top-N semantically, then apply building/equipment filters in Python."),
    ("B\u2032", "#7886C7", "Metadata-Aware Filtering",
     "Expand candidate pool before filtering \u2014 reduces recall loss at boundary."),
    ("C",       "#A9B5DF", "MICE Embeddings",
     "Metadata-Infused Contextual Embeddings \u2014 facility context baked into vectors."),
]


def render():
    # ── Scoped CSS ───────────────────────────────────────────
    st.markdown("""
    <style>
    /* ── Widen content for 1920px ── */
    .main .block-container {
        max-width: 1600px !important;
        padding: 2.5rem 3rem 5rem !important;
    }

    /* ── Hero grid: left title + right RQ card, equal-height row ── */
    .ov-hero-grid {
        display: grid;
        grid-template-columns: 3fr 2fr;
        gap: 2rem;
        align-items: stretch;
        padding: 2.5rem 0 2rem;
        border-bottom: 1px solid var(--border);
        margin-bottom: 2rem;
    }
    .ov-hero-left {
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .ov-eyebrow {
        font-family: var(--mono);
        font-size: 0.72rem;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: var(--accent2);
        margin-bottom: 0.7rem;
    }
    .ov-title {
        font-size: 3rem;
        font-weight: 700;
        color: var(--accent);
        letter-spacing: -0.02em;
        line-height: 1.1;
        margin-bottom: 0.85rem;
    }
    .ov-subtitle {
        font-size: 1.1rem;
        color: #4A5490;
        line-height: 1.6;
        font-weight: 400;
        margin: 0;
    }

    /* ── Research question card ── */
    .ov-rq-card {
        background: #FFFFFF;
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 1.6rem 1.8rem;
        display: flex;
        flex-direction: column;
        justify-content: center;
        height: 100%;
        box-sizing: border-box;
    }
    .ov-rq-label {
        font-family: var(--mono);
        font-size: 0.68rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: var(--muted);
        margin-bottom: 0.55rem;
    }
    .ov-rq-text {
        font-size: 1.05rem;
        font-weight: 500;
        color: var(--accent);
        line-height: 1.65;
        margin: 0;
    }
    .ov-rq-text em {
        font-style: normal;
        font-weight: 600;
    }

    /* ── Section label + divider ── */
    .ov-sec-div {
        margin: 2rem 0 1.5rem;
        border: none;
        border-top: 1px solid var(--border);
    }
    .ov-sec-label {
        font-family: var(--mono);
        font-size: 0.68rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: var(--accent2);
        margin-bottom: 1rem;
    }

    /* ── Snapshot grid: 4 equal-height cards in pure CSS ── */
    .ov-snap-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
        align-items: stretch;
    }
    .ov-snap-card {
        background: #FFFFFF;
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 1.5rem 1.6rem;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        transition: border-color 0.18s, box-shadow 0.18s;
    }
    .ov-snap-card:hover {
        border-color: #2D336B;
        box-shadow: 0 2px 12px rgba(45,51,107,0.09);
    }
    .ov-snap-label {
        font-family: var(--mono);
        font-size: 0.65rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--muted);
        margin-bottom: 0.5rem;
    }
    .ov-snap-value {
        font-size: 2.2rem;
        font-weight: 700;
        font-family: var(--mono);
        color: var(--accent);
        line-height: 1.1;
        margin-bottom: 0.3rem;
        /* clamp so model name never overflows */
        word-break: break-word;
    }
    .ov-snap-value.ov-snap-value--sm {
        font-size: 1.25rem;
        padding-top: 0.45rem;
    }
    .ov-snap-sub {
        font-size: 0.88rem;
        color: var(--accent2);
        margin-top: auto;
    }

    /* ── Pipeline ── */
    .ov-pipeline {
        display: flex;
        align-items: stretch;
        background: #FFFFFF;
        border: 1px solid var(--border);
        border-radius: 8px;
        overflow: hidden;
    }
    .ov-pipe-cell {
        flex: 1;
        display: flex;
        align-items: stretch;
    }
    .ov-pipe-cell:last-child .ov-pipe-divider {
        display: none;
    }
    .ov-pipe-stage {
        flex: 1;
        padding: 1.1rem 1rem;
        transition: background 0.15s;
    }
    .ov-pipe-stage:hover {
        background: #F0F3FF;
    }
    .ov-pipe-divider {
        width: 1px;
        background: var(--border);
        flex-shrink: 0;
        align-self: stretch;
        position: relative;
    }
    .ov-pipe-arrow {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        width: 20px;
        height: 20px;
        background: #FFFFFF;
        border: 1px solid var(--border);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.55rem;
        color: var(--accent2);
        z-index: 2;
    }
    .ov-pipe-name {
        font-weight: 700;
        font-size: 0.92rem;
        color: var(--accent);
        margin-bottom: 0.3rem;
    }
    .ov-pipe-desc {
        font-size: 0.8rem;
        color: var(--muted);
        line-height: 1.45;
    }

    /* ── Strategy cards ── */
    .ov-strat-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
    }
    .ov-strat-card {
        background: #FFFFFF;
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 1.5rem 1.4rem 1.4rem;
        transition: border-color 0.18s, transform 0.15s, box-shadow 0.18s;
    }
    .ov-strat-card:hover {
        border-color: #2D336B;
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(45,51,107,0.11);
    }
    .ov-strat-letter {
        font-family: var(--mono);
        font-size: 2.6rem;
        font-weight: 700;
        line-height: 1;
        margin-bottom: 0.65rem;
    }
    .ov-strat-name {
        font-size: 0.97rem;
        font-weight: 600;
        color: var(--accent);
        margin-bottom: 0.5rem;
    }
    .ov-strat-desc {
        font-size: 0.9rem;
        color: #4A5490;
        line-height: 1.55;
    }

    /* ── Tech status cards ── */
    .ov-tech-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 0.75rem;
    }
    .ov-tech-card {
        background: #FFFFFF;
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 1rem 1.2rem;
    }
    .ov-tech-label {
        font-family: var(--mono);
        font-size: 0.65rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--muted);
        margin-bottom: 0.3rem;
    }
    .ov-tech-value {
        font-family: var(--mono);
        font-size: 1rem;
        font-weight: 600;
        color: var(--accent);
    }

    /* ── Architecture preview ── */
    .ov-arch-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1rem;
    }
    .ov-arch-track {
        background: #FFFFFF;
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 1.5rem 1.6rem;
    }
    .ov-arch-track-label {
        font-family: var(--mono);
        font-size: 0.65rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: var(--muted);
        margin-bottom: 0.4rem;
    }
    .ov-arch-track-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: var(--accent);
        margin-bottom: 0.75rem;
    }
    .ov-arch-badges {
        display: flex;
        gap: 0.4rem;
        flex-wrap: wrap;
        margin-bottom: 0.75rem;
    }
    .ov-arch-badge {
        font-family: var(--mono);
        font-size: 0.75rem;
        font-weight: 600;
        background: #2D336B;
        color: #FFF2F2;
        padding: 0.2rem 0.55rem;
        border-radius: 4px;
    }
    .ov-arch-badge.light {
        background: #A9B5DF;
        color: #2D336B;
    }
    .ov-arch-desc {
        font-size: 0.9rem;
        color: #4A5490;
        line-height: 1.6;
    }
    </style>
    """, unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════
    # SECTION 1 — HERO  (pure CSS grid — no st.columns)
    # ════════════════════════════════════════════════════════
    st.markdown("""
    <div class="ov-hero-grid">
        <div class="ov-hero-left">
            <div class="ov-eyebrow">Research Platform \u00b7 Facility Management AI</div>
            <div class="ov-title">FM-RAG Research Platform</div>
            <div class="ov-subtitle">
                Investigating whether structured metadata improves retrieval quality
                in Facility Management RAG systems.
            </div>
        </div>
        <div class="ov-rq-card">
            <div class="ov-rq-label">Research Question</div>
            <p class="ov-rq-text">
                Does incorporating <em>structured metadata</em> \u2014 building type,
                equipment system, and facility context \u2014 improve retrieval performance
                compared to <em>pure semantic search</em>?
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════
    # SECTION 2 — System Information  (pure CSS grid)
    # ════════════════════════════════════════════════════════
    stats  = get_index_stats()
    config = load_config()

    dataset_size  = stats.get("text_total") or stats.get("mice_total") or 0
    embed_model   = (config.get("embedding_model", "BAAI/bge-base-en-v1.5") or "").split("/")[-1]
    vector_dim    = stats.get("dim") or config.get("embedding_dim", 768)

    size_display  = f"{dataset_size:,}" if dataset_size else "~2.5M"
    model_display = embed_model or "BGE-base-en-v1.5"
    dim_display   = str(vector_dim) if vector_dim else "768"

    st.markdown(f"""
    <div class="ov-sec-label">System Information</div>
    <div class="ov-snap-grid">
        <div class="ov-snap-card">
            <div>
                <div class="ov-snap-label">Dataset Size</div>
                <div class="ov-snap-value">{size_display}</div>
            </div>
            <div class="ov-snap-sub">FM work orders indexed</div>
        </div>
        <div class="ov-snap-card">
            <div>
                <div class="ov-snap-label">Retrieval Strategies</div>
                <div class="ov-snap-value">4</div>
            </div>
            <div class="ov-snap-sub">A \u00b7 B \u00b7 B\u2032 \u00b7 C</div>
        </div>
        <div class="ov-snap-card">
            <div>
                <div class="ov-snap-label">Embedding Model</div>
                <div class="ov-snap-value ov-snap-value--sm">{model_display}</div>
            </div>
            <div class="ov-snap-sub">Sentence transformer</div>
        </div>
        <div class="ov-snap-card">
            <div>
                <div class="ov-snap-label">Vector Dimension</div>
                <div class="ov-snap-value">{dim_display}</div>
            </div>
            <div class="ov-snap-sub">Dense embedding size</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════
    # SECTION 3 — PIPELINE
    # ════════════════════════════════════════════════════════
    st.markdown('<hr class="ov-sec-div">', unsafe_allow_html=True)
    st.markdown('<div class="ov-sec-label">System Pipeline</div>', unsafe_allow_html=True)

    cells_html = ""
    for i, (name, desc) in enumerate(PIPELINE_STAGES):
        is_last = (i == len(PIPELINE_STAGES) - 1)
        divider = (
            "" if is_last else
            '<div class="ov-pipe-divider"><div class="ov-pipe-arrow">\u2192</div></div>'
        )
        cells_html += (
            f'<div class="ov-pipe-cell">'
            f'  <div class="ov-pipe-stage">'
            f'    <div class="ov-pipe-name">{name}</div>'
            f'    <div class="ov-pipe-desc">{desc}</div>'
            f'  </div>'
            f'  {divider}'
            f'</div>'
        )

    st.markdown(
        f'<div class="ov-pipeline">{cells_html}</div>',
        unsafe_allow_html=True,
    )

    # ════════════════════════════════════════════════════════
    # SECTION 4 — RETRIEVAL METHODOLOGY
    # ════════════════════════════════════════════════════════
    st.markdown('<hr class="ov-sec-div">', unsafe_allow_html=True)
    st.markdown('<div class="ov-sec-label">Retrieval Methodology</div>', unsafe_allow_html=True)

    cards_html = '<div class="ov-strat-grid">'
    for label, color, name, desc in STRATEGIES:
        cards_html += (
            f'<div class="ov-strat-card">'
            f'  <div class="ov-strat-letter" style="color:{color};">{label}</div>'
            f'  <div class="ov-strat-name">{name}</div>'
            f'  <div class="ov-strat-desc">{desc}</div>'
            f'</div>'
        )
    cards_html += "</div>"
    st.markdown(cards_html, unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════
    # SECTION 5 — ARCHITECTURE PREVIEW
    # ════════════════════════════════════════════════════════
    st.markdown('<hr class="ov-sec-div">', unsafe_allow_html=True)
    st.markdown('<div class="ov-sec-label">Architecture Overview</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="ov-arch-grid">
        <div class="ov-arch-track">
            <div class="ov-arch-track-label">Embedding Track 1</div>
            <div class="ov-arch-track-title">TEXT Track</div>
            <div class="ov-arch-badges">
                <span class="ov-arch-badge">A</span>
                <span class="ov-arch-badge">B</span>
                <span class="ov-arch-badge">B\u2032</span>
            </div>
            <div class="ov-arch-desc">
                Representation: <code>BuildingName | Type | WODescription</code><br><br>
                Semantic retrieval with optional metadata post-filtering.
                Metadata is applied as a constraint after vector search,
                preserving embedding quality.
            </div>
        </div>
        <div class="ov-arch-track">
            <div class="ov-arch-track-label">Embedding Track 2</div>
            <div class="ov-arch-track-title">MICE Track</div>
            <div class="ov-arch-badges">
                <span class="ov-arch-badge light">C</span>
            </div>
            <div class="ov-arch-desc">
                Representation includes building ID, name, facility type,
                equipment system, work period, and description.<br><br>
                Metadata is baked directly into vectors \u2014 retrieval is
                contextually richer but may dilute semantic signal.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top:0.9rem;'></div>", unsafe_allow_html=True)
    if st.button("View Full Architecture \u2192", key="ov_arch_btn"):
        st.session_state.active_page = "System Architecture"
        st.rerun()