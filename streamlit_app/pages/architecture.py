"""
pages/architecture.py - System architecture reference page.
"""

import streamlit as st
from assets.styles import section_header


MODULE_TABLE = [
    ("preprocessing.py",      "CSV loading, text normalization, row preprocessing"),
    ("cleaning.py",           "Remove empty WOIDs, dedup, group missing BuildingIDs"),
    ("embedding_builder.py",  "TEXT and MICE representation factory functions"),
    ("embedder.py",           "BAAI/bge-base-en-v1.5 singleton; embed_texts(), embed_query()"),
    ("faiss_store.py",        "Two IndexFlatIP stores; insert, search, persist, load"),
    ("retrieval.py",          "Strategies A / B / B′ / C; FilterConfig; deduplicate()"),
    ("query_router.py",       "Regex + vocab classifier → strategy + FilterConfig"),
    ("evaluation.py",         "TestCase generation; Recall@k, MRR, NDCG; run_evaluation()"),
    ("analysis.py",           "QueryComparison; win matrix; metadata impact; CSV export"),
    ("main.py",               "Pipeline orchestrator with atomic chunk checkpointing"),
]

CONFIG_KEYS = [
    ("EMBEDDING_MODEL",      "BAAI/bge-base-en-v1.5",  "Sentence transformer for both tracks"),
    ("EMBEDDING_DIM",        "768",                     "Output vector dimension"),
    ("EMBED_BATCH_SIZE",     "128",                     "GPU-optimized encoding batch"),
    ("DEFAULT_TOP_K",        "10",                      "Default retrieval depth"),
    ("POST_FILTER_MULTIPLIER","20",                     "Candidate pool multiplier for B / B′"),
    ("EVAL_SAMPLE_SIZE",     "500",                     "Auto-generated test queries"),
    ("CSV_CHUNK_SIZE",       "10,000",                  "Rows per ingestion chunk"),
    ("CHECKPOINT_INTERVAL",  "50,000",                  "Progress log interval"),
]


def _code_block(text: str) -> None:
    st.markdown(
        f'<div style="background:#FFF2F2;border:1px solid #A9B5DF;'
        f'border-radius:4px;padding:0.8rem;font-family:\'IBM Plex Mono\',monospace;'
        f'font-size:0.78rem;color:#2D336B;white-space:pre-wrap;">{text}</div>',
        unsafe_allow_html=True,
    )


def render():
    st.markdown('<div class="page-title">System Architecture</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">'
        'Technical reference for the FM-RAG pipeline components, '
        'configuration, and checkpointing strategy.'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Module overview ──────────────────────────────────────
    section_header("Module Reference")

    import pandas as pd
    df_modules = pd.DataFrame(MODULE_TABLE, columns=["Module", "Responsibility"])
    st.dataframe(df_modules, width='stretch', hide_index=True, height=380)

    # ── Configuration ────────────────────────────────────────
    st.markdown("<br/>", unsafe_allow_html=True)
    section_header("Key Configuration (config.py)")

    df_cfg = pd.DataFrame(
        CONFIG_KEYS, columns=["Parameter", "Default", "Description"])
    st.dataframe(df_cfg, width='stretch', hide_index=True)

    # ── Dual-index architecture ──────────────────────────────
    st.markdown("<br/>", unsafe_allow_html=True)
    section_header("Dual Embedding Architecture")

    a1, a2 = st.columns(2, gap="large")

    with a1:
        st.markdown("""
        <div style="background:#FFFFFF;border:1px solid #A9B5DF;
                    border-left:3px solid #2D336B;border-radius:4px;
                    padding:1rem;">
            <div style="font-size:0.7rem;letter-spacing:0.1em;
                        text-transform:uppercase;color:#2D336B;
                        margin-bottom:0.6rem;">TEXT Track</div>
            <ul style="font-size:0.8rem;color:#7886C7;
                       padding-left:1.1rem;line-height:2.0;">
                <li>File: <code>text.index</code> + <code>text_metadata.pkl</code></li>
                <li>Repr: <code>BuildingName | Type | WODescription</code></li>
                <li>Search: <code>faiss_store.search_text()</code></li>
                <li>Used by: A, B, B′</li>
            </ul>
        </div>""", unsafe_allow_html=True)

    with a2:
        st.markdown("""
        <div style="background:#FFFFFF;border:1px solid #A9B5DF;
                    border-left:3px solid #7886C7;border-radius:4px;
                    padding:1rem;">
            <div style="font-size:0.7rem;letter-spacing:0.1em;
                        text-transform:uppercase;color:#7886C7;
                        margin-bottom:0.6rem;">MICE Track</div>
            <ul style="font-size:0.8rem;color:#7886C7;
                       padding-left:1.1rem;line-height:2.0;">
                <li>File: <code>mice.index</code> + <code>mice_metadata.pkl</code></li>
                <li>Repr: labelled sentence template (all 6 fields)</li>
                <li>Search: <code>faiss_store.search_mice()</code></li>
                <li>Used by: C only</li>
            </ul>
        </div>""", unsafe_allow_html=True)

    # ── MICE template ────────────────────────────────────────
    st.markdown("<br/>", unsafe_allow_html=True)
    section_header("MICE Template")
    _code_block(
        "building id: {building_id}.\n"
        "building name: {building_name}.\n"
        "facility type: {btype}.\n"
        "equipment system: {equipment}.\n"
        "work period: {start} to {end}.\n"
        "work order description: {description}."
    )
    st.markdown("""
    <div style="font-size:0.78rem;color:#7886C7;margin-top:0.5rem;line-height:1.6;">
        Lowercase field labels match BGE's MSMARCO training distribution.
        Field ordering: categorical → temporal → free-text places high-selectivity
        fields early in the token sequence where positional attention is strongest.
    </div>""", unsafe_allow_html=True)

    # ── Atomic checkpointing ─────────────────────────────────
    st.markdown("<br/>", unsafe_allow_html=True)
    section_header("Atomic Chunk-Based Checkpointing")

    st.markdown("""
    <div style="background:#FFFFFF;border:1px solid #A9B5DF;
                border-radius:6px;padding:1.1rem;">
        <div style="font-size:0.8rem;color:#7886C7;line-height:1.9;">
            Each CSV chunk is processed as an <strong style="color:#2D336B;">atomic unit</strong>:
        </div>
    </div>""", unsafe_allow_html=True)

    steps = [
        "1. Load chunk from CSV",
        "2. Build TEXT + MICE representations",
        "3. Embed both tracks (GPU-batched)",
        "4. Insert into FAISS in-memory",
        "5. Persist indexes to disk (atomic commit)",
        "6. Mark chunk complete in ingestion_progress.json",
    ]
    for i, step in enumerate(steps):
        color = "#2D336B" if i < 4 else "#7886C7"
        st.markdown(
            f'<div style="font-family:\'IBM Plex Mono\',monospace;'
            f'font-size:0.78rem;color:{color};padding:0.15rem 0;">{step}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("""
    <div style="font-size:0.78rem;color:#7886C7;margin-top:0.6rem;line-height:1.6;">
        Power cuts can only interrupt <em>between</em> chunks.
        No partial or duplicate vectors are written.
        Delete <code>data/ingestion_progress.json</code> to restart from scratch.
    </div>""", unsafe_allow_html=True)

    # ── Query router design ──────────────────────────────────
    st.markdown("<br/>", unsafe_allow_html=True)
    section_header("Query Router Design")

    st.markdown("""
    <div style="background:#FFFFFF;border:1px solid #A9B5DF;
                border-radius:6px;padding:1.1rem;">
        <div style="font-size:0.8rem;color:#7886C7;line-height:1.8;">
            <strong style="color:#2D336B;">Zero ML inference.</strong>
            Pure regex + vocabulary lookup → deterministic, &lt;1 ms latency.<br><br>
            <strong style="color:#A9B5DF;">Equipment or type vocabulary hit</strong>
            → Strategy B′ (pre-filter)<br>
            <strong style="color:#7886C7;">Building ID pattern (e.g. A050)</strong>
            → Strategy B (post-filter)<br>
            <strong style="color:#2D336B;">No metadata signals detected</strong>
            → Strategy A (semantic baseline)<br>
            <strong style="color:#5C6BC0;">Caller explicit override</strong>
            → Strategy C (MICE)
        </div>
    </div>""", unsafe_allow_html=True)

    # ── Evaluation design ────────────────────────────────────
    st.markdown("<br/>", unsafe_allow_html=True)
    section_header("Evaluation Design")

    st.markdown("""
    <div style="background:#FFFFFF;border:1px solid #A9B5DF;
                border-radius:6px;padding:1.1rem;">
        <ul style="font-size:0.8rem;color:#7886C7;
                   padding-left:1.2rem;line-height:2.0;margin:0;">
            <li>500 auto-generated test queries (seed=42, reproducible)</li>
            <li>Queries drawn from four types: semantic, equipment, type, mixed</li>
            <li>Relevant WOIDs identified by metadata group overlap</li>
            <li>Metrics: Recall@1/5/10, MRR, NDCG@10</li>
            <li>All strategies evaluated on the <em>same</em> query set</li>
            <li>Results exported to <code>eval_results.json</code> and
                <code>per_query_analysis.csv</code></li>
        </ul>
    </div>""", unsafe_allow_html=True)

    # ── Future directions ────────────────────────────────────
    st.markdown("<br/>", unsafe_allow_html=True)
    section_header("Future Directions")

    fd1, fd2 = st.columns(2, gap="large")
    with fd1:
        st.markdown("""
        <div style="background:#FFFFFF;border:1px solid #A9B5DF;
                    border-radius:6px;padding:1rem;">
            <div style="font-size:0.72rem;letter-spacing:0.08em;
                        text-transform:uppercase;color:#2D336B;
                        margin-bottom:0.5rem;">Retrieval</div>
            <ul style="font-size:0.78rem;color:#7886C7;
                       padding-left:1rem;line-height:1.9;margin:0;">
                <li>Hybrid BM25 + dense retrieval (RRF fusion)</li>
                <li>FAISS IVF index for sub-linear search at scale</li>
                <li>Fine-tuned bi-encoder on FM domain</li>
                <li>Adaptive top-k based on query confidence</li>
            </ul>
        </div>""", unsafe_allow_html=True)
    with fd2:
        st.markdown("""
        <div style="background:#FFFFFF;border:1px solid #A9B5DF;
                    border-radius:6px;padding:1rem;">
            <div style="font-size:0.72rem;letter-spacing:0.08em;
                        text-transform:uppercase;color:#7886C7;
                        margin-bottom:0.5rem;">System</div>
            <ul style="font-size:0.78rem;color:#7886C7;
                       padding-left:1rem;line-height:1.9;margin:0;">
                <li>REST API wrapper around route_and_retrieve()</li>
                <li>Incremental index updates without full rebuild</li>
                <li>LLM answer generation layer (RAG completion)</li>
                <li>Feedback loop for relevance annotation</li>
            </ul>
        </div>""", unsafe_allow_html=True)
