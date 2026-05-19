"""
pages/overview.py - Landing page: FM-RAG pipeline overview.
"""

import streamlit as st
from assets.styles import section_header
from loaders.data_loader import get_index_stats, load_config


PIPELINE_STEPS = [
    ("01", "CSV Dataset",
     "Raw FM work-order records with building metadata, descriptions, dates."),
    ("02", "Preprocessing",
     "Normalize text, remove newlines/quotes, standardize field formats."),
    ("03", "Cleaning",
     "Drop empty WOIDs, remove duplicates, group missing BuildingIDs."),
    ("04", "Representation Construction",
     "Build TEXT repr (semantic) and MICE repr (metadata-infused) per row."),
    ("05", "Embedding Generation",
     "Encode both representations with BAAI/bge-base-en-v1.5 → 768-dim vectors."),
    ("06", "FAISS Vector Storage",
     "Two flat inner-product indexes: text.index and mice.index."),
    ("07", "Retrieval Strategies",
     "A (baseline) · B (post-filter) · B′ (pre-filter) · C (MICE search)."),
    ("08", "Evaluation",
     "Recall@k, MRR, NDCG across 500 auto-generated test queries."),
]

STRATEGY_SUMMARY = [
    ("A", "#4f8ef7", "Semantic only",
     "Pure vector similarity. No metadata filtering. Best baseline."),
    ("B", "#38c96e", "Post-filter",
     "Fetch top-N semantically, then apply metadata filters in Python."),
    ("B′","#f7a94f", "Pre-filter",
     "Expand the candidate pool, filter before returning top-k."),
    ("C", "#e06c75", "MICE",
     "Metadata-Infused Contextual Embeddings. Filter baked into the vector."),
]


def render():
    # ── Page header ─────────────────────────────────────────
    st.markdown('<div class="page-title">FM-RAG Research Platform</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">'
        'Facility Management Retrieval-Augmented Generation · '
        'Metadata-aware retrieval for ~2.5M work-order records'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Top-level stats ──────────────────────────────────────
    stats  = get_index_stats()
    config = load_config()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("TEXT Index", f"{stats['text_total']:,}" if stats['text_total'] else "—",
                  help="Vectors in the semantic FAISS index")
    with c2:
        st.metric("MICE Index", f"{stats['mice_total']:,}" if stats['mice_total'] else "—",
                  help="Vectors in the metadata-infused FAISS index")
    with c3:
        st.metric("Vector Dim", str(stats.get("dim") or
                                    config.get("embedding_dim", 768)))
    with c4:
        st.metric("Embedding Model",
                  (config.get("embedding_model","BAAI/bge-base-en-v1.5") or "")
                  .split("/")[-1])

    st.markdown("<br/>", unsafe_allow_html=True)

    # ── Pipeline flow ────────────────────────────────────────
    section_header("Pipeline Overview", "8 stages")

    left, right = st.columns([1, 1], gap="large")

    with left:
        for step in PIPELINE_STEPS[:4]:
            num, title, desc = step
            st.markdown(f"""
            <div class="pipeline-step">
                <div class="step-num">STAGE {num}</div>
                <div class="step-title">{title}</div>
                <div class="step-desc">{desc}</div>
            </div>""", unsafe_allow_html=True)

    with right:
        for step in PIPELINE_STEPS[4:]:
            num, title, desc = step
            st.markdown(f"""
            <div class="pipeline-step">
                <div class="step-num">STAGE {num}</div>
                <div class="step-title">{title}</div>
                <div class="step-desc">{desc}</div>
            </div>""", unsafe_allow_html=True)

    # ── Core research question ───────────────────────────────
    st.markdown("<br/>", unsafe_allow_html=True)
    section_header("Research Question")
    st.markdown("""
    <div style="background:#1a1d27;border:1px solid #2a2d3e;border-left:3px solid #4f8ef7;
                border-radius:4px;padding:1rem 1.4rem;font-size:0.9rem;
                color:#e4e6f0;line-height:1.7;">
        Does incorporating <strong>structured metadata</strong> (building type, equipment system)
        into the retrieval pipeline improve work-order retrieval recall
        compared to <strong>pure semantic</strong> vector search?
    </div>
    """, unsafe_allow_html=True)

    # ── Strategy summary cards ───────────────────────────────
    st.markdown("<br/>", unsafe_allow_html=True)
    section_header("Retrieval Strategies", "4 approaches")
    cols = st.columns(4)
    for col, (label, color, title, desc) in zip(cols, STRATEGY_SUMMARY):
        with col:
            st.markdown(f"""
            <div style="background:#1a1d27;border:1px solid #2a2d3e;
                        border-top:3px solid {color};border-radius:6px;
                        padding:1rem;">
                <div style="font-family:'IBM Plex Mono',monospace;font-size:1.5rem;
                            font-weight:600;color:{color};">{label}</div>
                <div style="font-size:0.75rem;font-weight:600;
                            text-transform:uppercase;letter-spacing:0.06em;
                            color:#e4e6f0;margin:0.3rem 0 0.6rem;">{title}</div>
                <div style="font-size:0.78rem;color:#8890a8;line-height:1.5;">{desc}</div>
            </div>""", unsafe_allow_html=True)

    # ── Dual embedding track ─────────────────────────────────
    st.markdown("<br/>", unsafe_allow_html=True)
    section_header("Dual Embedding Architecture")

    t, m = st.columns(2, gap="large")
    with t:
        st.markdown("""
        <div style="background:#1a1d27;border:1px solid #2a2d3e;
                    border-radius:6px;padding:1.1rem;">
            <div style="font-size:0.7rem;letter-spacing:0.1em;text-transform:uppercase;
                        color:#4f8ef7;margin-bottom:0.5rem;">TEXT Track · Strategies A, B, B′</div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:0.78rem;
                        color:#98c4fb;background:#0d1117;padding:0.7rem;
                        border-radius:4px;line-height:1.8;">
                BuildingName | Type | WODescription
            </div>
            <div style="font-size:0.78rem;color:#8890a8;margin-top:0.7rem;line-height:1.5;">
                Compact semantic signal. Metadata applied as a post-retrieval filter.
            </div>
        </div>""", unsafe_allow_html=True)

    with m:
        st.markdown("""
        <div style="background:#1a1d27;border:1px solid #2a2d3e;
                    border-radius:6px;padding:1.1rem;">
            <div style="font-size:0.7rem;letter-spacing:0.1em;text-transform:uppercase;
                        color:#e06c75;margin-bottom:0.5rem;">MICE Track · Strategy C</div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:0.78rem;
                        color:#f19898;background:#0d1117;padding:0.7rem;
                        border-radius:4px;line-height:1.8;">
                building id: … building name: …<br>
                facility type: … equipment system: …<br>
                work order description: …
            </div>
            <div style="font-size:0.78rem;color:#8890a8;margin-top:0.7rem;line-height:1.5;">
                Metadata baked directly into the embedding. No post-filtering needed.
            </div>
        </div>""", unsafe_allow_html=True)

    # ── Quick navigation ─────────────────────────────────────
    st.markdown("<br/>", unsafe_allow_html=True)
    section_header("Quick Navigation")
    nav = st.columns(4)
    nav_items = [
        ("📂", "Dataset Explorer",        "Inspect work orders"),
        ("⚡", "Live Query Testing",      "Run live retrieval"),
        ("📊", "Evaluation Dashboard",    "View metrics"),
        ("🔬", "Embedding Visualization", "Explore vector space"),
    ]
    for col, (icon, name, hint) in zip(nav, nav_items):
        with col:
            if st.button(f"{icon} {name}", help=hint,
                         use_container_width=True, key=f"quick_{name}"):
                st.session_state.active_page = name
                st.rerun()
