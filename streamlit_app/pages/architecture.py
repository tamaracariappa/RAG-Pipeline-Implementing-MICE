"""
pages/architecture.py  ·  System & Retrieval Architecture
Merged from architecture.py + strategy_explainer.py.
Hero = Retrieval Strategies (the core research contribution).
All backend logic untouched.

Refinement pass:
- Strategy cards: equal height via flexbox, larger fonts, dark text, centred flows
- Removed emojis from arch cards, checkpoint steps, and module list
- Replaced Streamlit pages list with a directory-tree code block
- Removed _evaluation_design section
- Darkened all secondary/body text: #7886C7 → #46538C where it carries meaning
- Emoji-free checkpoint steps
- Light blue (#A9B5DF) reserved for borders and subtle separators only
"""

from __future__ import annotations
import streamlit as st
from assets.styles import section_header
from loaders.data_loader import load_config

# ── colour tokens ──────────────────────────────────────────────────────────
_PRI  = "#2D336B"   # primary text
_SEC  = "#46538C"   # secondary text (darker than before)
_MUT  = "#A9B5DF"   # muted — borders and separators only
_BG   = "#FFFFFF"   # card background

# ── Strategy data ──────────────────────────────────────────────────────────
STRATEGIES = [
    {
        "key": "A", "color": "#2D336B",
        "title": "Semantic Baseline",
        "summary": "Pure vector similarity — query embedded, compared to every TEXT index vector.",
        "index": "TEXT", "filter": "None",
        "pros": ["Fastest — single FAISS pass", "No metadata needed at query time"],
        "cons": ["Cannot distinguish facility context", "May return contextually wrong records"],
        "flow": ["Query", "embed_query()", "FAISS · TEXT index", "Top-k results"],
    },
    {
        "key": "B", "color": "#7886C7",
        "title": "Semantic + Post-Filter",
        "summary": "Fetch top_k × 20 from TEXT index, then trim with Python metadata filters.",
        "index": "TEXT", "filter": "Python post-filter",
        "pros": ["Works with any metadata combination", "No retraining needed"],
        "cons": ["Larger pool — slower than A", "Over-filtering can reduce result count"],
        "flow": ["Query + FilterConfig", "embed_query()", "FAISS · top_k×20", "Python filter", "Top-k results"],
    },
    {
        "key": "B′", "color": "#A9B5DF",
        "title": "Pre-Filter",
        "summary": "Expanded pool from TEXT index, metadata filter applied before final top-k ranking.",
        "index": "TEXT", "filter": "Python pre-selection",
        "pros": ["Semantic ranking preserved within filtered set", "Better precision on selective filters"],
        "cons": ["Still requires expanded candidate pool", "Depends on filter selectivity"],
        "flow": ["Query + FilterConfig", "embed_query()", "FAISS · expanded pool", "Filter → re-rank", "Top-k results"],
    },
    {
        "key": "C", "color": "#5C6BC0",
        "title": "MICE — Metadata-Infused",
        "summary": "Metadata embedded into the vector via structured template. MICE index queried directly.",
        "index": "MICE", "filter": "None (implicit in vector)",
        "pros": ["Metadata selectivity baked into similarity score", "Single-pass — no post-filtering overhead"],
        "cons": ["Requires second index (storage overhead)", "Query must match document template style"],
        "flow": ['"work order description: " + query', "embed_query()", "FAISS · MICE index", "Top-k results"],
    },
]

# ── Module reference (no emojis) ───────────────────────────────────────────
MODULE_TABLE = [
    ("preprocessing.py",     "CSV loading, text normalisation, row preprocessing"),
    ("cleaning.py",          "Remove empty WOIDs, deduplication, missing BuildingID grouping"),
    ("embedding_builder.py", "TEXT and MICE representation factory functions"),
    ("embedder.py",          "BGE singleton; embed_texts() and embed_query()"),
    ("faiss_store.py",       "Two IndexFlatIP stores — insert, search, persist, load"),
    ("retrieval.py",         "Strategies A / B / B′ / C; FilterConfig; deduplicate()"),
    ("query_router.py",      "Regex + vocab classifier — strategy selection + FilterConfig"),
    ("evaluation.py",        "TestCase generation; Recall@k, MRR, NDCG; run_evaluation()"),
    ("analysis.py",          "QueryComparison; win matrix; metadata impact; CSV export"),
    ("main.py",              "Pipeline orchestrator with atomic chunk checkpointing"),
]

CONFIG_KEYS = [
    ("Embedding Model",    "BAAI/bge-base-en-v1.5"),
    ("Vector Dimension",   "768"),
    ("Batch Size",         "128"),
    ("Default Top-K",      "10"),
    ("Filter Multiplier",  "×20  (B / B′ pool)"),
    ("Eval Sample Size",   "500 queries"),
    ("CSV Chunk Size",     "10,000 rows"),
    ("Checkpoint Interval","50,000 rows"),
]

# ── Checkpoint steps (no emojis) ───────────────────────────────────────────
CHECKPOINT_STEPS = [
    ("01", "Load\nChunk",       "Read CSV slice\n~10,000 rows"),
    ("02", "Build\nReprs",      "TEXT + MICE\nstrings"),
    ("03", "Generate\nEmbeds",  "GPU-batched\nBGE encoding"),
    ("04", "Insert\nFAISS",     "Add to both\nin-memory indexes"),
    ("05", "Persist\nDisk",     "Atomic write\nboth index files"),
    ("06", "Mark\nComplete",    "Update\ningestion_progress.json"),
]

# ── Architecture cards (no emojis) ─────────────────────────────────────────
ARCH_CARDS = [
    ("#2D336B", "Dataset",    "2.5M facility management work orders from FMUCD"),
    ("#5C6BC0", "Embeddings", "Two representations per record — TEXT and MICE"),
    ("#7886C7", "FAISS",      "Two IndexFlatIP stores — exact cosine search"),
    ("#A9B5DF", "Retrieval",  "Four strategies with router-based dispatch"),
    ("#2D336B", "Evaluation", "Recall@k, MRR, NDCG across 298 test queries"),
]

# ── Directory tree (replaces Streamlit pages list) ─────────────────────────
DIRECTORY_TREE = """\
streamlit_app/
├── app.py                      ← Entry point and page router
├── requirements_app.txt        ← App-specific dependencies
├── assets/
│   └── styles.py               ← Global CSS and helper renderers
├── components/
│   └── sidebar.py              ← Navigation sidebar
├── pages/
│   ├── overview.py             ← Landing page / pipeline overview
│   ├── dataset_explorer.py     ← Work order inspector and representation comparison
│   ├── embedding_viz.py        ← PCA scatter of FAISS vector spaces
│   ├── live_query.py           ← Live retrieval across all four strategies
│   ├── vector_storage.py       ← How FAISS stores embeddings
│   ├── architecture.py         ← Strategies, pipeline, and configuration
│   └── evaluation_dashboard.py ← Metrics from eval_results.json
├── charts/
│   └── plotly_charts.py        ← Reusable Plotly figure builders
├── loaders/
│   ├── data_loader.py          ← Cached wrappers for CSV, FAISS, and evaluation
│   └── retrieval_runner.py     ← Calls retrieval strategies from the app
└── utils/                      ← Reserved for future utilities"""

BACKEND_TREE = """\
RAG-Pipeline-Implementing-MICE/
├── preprocessing.py    ← CSV loading, text normalisation, row preprocessing
├── cleaning.py         ← WOID deduplication, missing BuildingID grouping
├── embedding_builder.py← TEXT and MICE representation, factory functions
├── embedder.py         ← BGE singleton; embed_texts() and embed_query()
├── faiss_store.py      ← Two IndexFlatIP stores —
│                          insert, search, persist, load
├── retrieval.py        ← Strategies A / B / B′ / C;
│                          FilterConfig; deduplicate()
├── query_router.py     ← Regex + vocab classifier;
│                          strategy selection
├── evaluation.py       ← Recall@k, MRR, NDCG;
│                          run_evaluation()
├── analysis.py         ← Win matrix, metadata impact,
│                          CSV export
├── main.py             ← Pipeline orchestrator with atomic chunk checkpointing
└── data/
    ├── text.index / mice.index
    ├── eval_results.json
    └── per_query_analysis.csv"""


# ── Helper: centred vertical flow ──────────────────────────────────────────
def _flow_card(steps: list, color: str) -> str:
    """Build a vertically centred flowchart HTML string."""
    parts = []
    for i, step in enumerate(steps):
        parts.append(
            f'<div style="background:#FAFBFF;border:1px solid {color};'
            f'border-radius:4px;padding:0.4rem 0.75rem;'
            f'font-family:IBM Plex Mono,monospace;font-size:0.75rem;'
            f'color:{_PRI};white-space:nowrap;text-align:center;">{step}</div>'
        )
        if i < len(steps) - 1:
            parts.append(
                f'<div style="color:{color};font-size:0.9rem;'
                f'line-height:1;text-align:center;">↓</div>'
            )
    return (
        '<div style="display:flex;flex-direction:column;'
        'align-items:center;gap:0.25rem;">' +
        "".join(parts) + "</div>"
    )


# ════════════════════════════════════════════════════════════
# SECTION 1 · HERO
# ════════════════════════════════════════════════════════════
def _hero() -> None:
    st.markdown(f"""
    <div style="padding:2rem 0 1.6rem;">
        <div style="font-size:2.6rem;font-weight:700;color:{_PRI};
                    letter-spacing:-0.02em;line-height:1.2;margin-bottom:0.6rem;">
            Retrieval Strategies
        </div>
        <div style="font-size:1.15rem;color:{_SEC};max-width:820px;line-height:1.7;">
            Four approaches to facility management work-order retrieval —
            from pure semantics to metadata-infused embeddings.
        </div>
    </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# SECTION 2 · STRATEGY CARDS  (equal height, larger fonts)
# ════════════════════════════════════════════════════════════
def _strategy_cards() -> None:
    # Render all four cards inside a single CSS grid so heights are
    # governed by the same grid row — equal height guaranteed.
    cards_html = '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;">'

    for s in STRATEGIES:
        pros_html = "".join(
            f'<div style="font-size:0.88rem;color:{_PRI};'
            f'margin:0.25rem 0;line-height:1.55;">&#10003; {p}</div>'
            for p in s["pros"]
        )
        cons_html = "".join(
            f'<div style="font-size:0.88rem;color:{_SEC};'
            f'margin:0.25rem 0;line-height:1.55;">&#10007; {c}</div>'
            for c in s["cons"]
        )
        flow_html = _flow_card(s["flow"], s["color"])

        cards_html += (
            f'<div style="background:{_BG};border:1px solid {_MUT};'
            f'border-top:4px solid {s["color"]};border-radius:10px;'
            f'padding:1.4rem 1.2rem;display:flex;flex-direction:column;gap:0;">'

            # Strategy key + title
            f'<div style="font-family:IBM Plex Mono,monospace;font-size:1.7rem;'
            f'font-weight:700;color:{s["color"]};margin-bottom:0.15rem;">'
            f'Strategy {s["key"]}</div>'

            f'<div style="font-size:0.85rem;font-weight:700;text-transform:uppercase;'
            f'letter-spacing:0.07em;color:{_PRI};margin-bottom:0.7rem;">{s["title"]}</div>'

            # Summary
            f'<div style="font-size:0.92rem;color:{_SEC};line-height:1.65;'
            f'margin-bottom:1rem;">{s["summary"]}</div>'

            # Flow label
            f'<div style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.09em;'
            f'color:{_PRI};margin-bottom:0.45rem;font-weight:600;">Execution Flow</div>'

            # Centred flow diagram
            f'{flow_html}'

            # Strengths
            f'<div style="margin-top:1rem;font-size:0.72rem;text-transform:uppercase;'
            f'letter-spacing:0.09em;color:{_PRI};margin-bottom:0.35rem;'
            f'font-weight:600;">Strengths</div>'
            f'{pros_html}'

            # Limitations
            f'<div style="margin-top:0.7rem;font-size:0.72rem;text-transform:uppercase;'
            f'letter-spacing:0.09em;color:{_PRI};margin-bottom:0.35rem;'
            f'font-weight:600;">Limitations</div>'
            f'{cons_html}'

            # Footer — push to bottom with margin-top:auto
            f'<div style="margin-top:auto;padding-top:0.9rem;'
            f'border-top:1px solid #F0F2FA;font-size:0.82rem;color:{_SEC};">'
            f'Index: <strong style="color:{_PRI};">{s["index"]}</strong>'
            f'&nbsp;·&nbsp;'
            f'Filter: <strong style="color:{_PRI};">{s["filter"]}</strong>'
            f'</div>'

            f'</div>'
        )

    cards_html += "</div>"
    st.markdown(cards_html, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# SECTION 3 · QUERY ROUTER
# ════════════════════════════════════════════════════════════
def _query_router() -> None:
    section_header("Query Router", "zero-latency dispatch")

    routing = [
        ("#A9B5DF", "Equipment or Type vocabulary hit",  "→ Strategy B′", "Pre-filter path"),
        ("#7886C7", "Building ID pattern detected",      "→ Strategy B",  "Post-filter path"),
        ("#2D336B", "No metadata signals detected",      "→ Strategy A",  "Semantic baseline"),
        ("#5C6BC0", "Caller explicit override",          "→ Strategy C",  "MICE path"),
    ]

    cards_html = '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;">'
    for color, signal, dest, label in routing:
        cards_html += (
            f'<div style="background:{_BG};border:1px solid {_MUT};'
            f'border-left:4px solid {color};border-radius:8px;'
            f'padding:1rem 1.1rem;">'
            f'<div style="font-size:0.88rem;color:{_SEC};'
            f'margin-bottom:0.5rem;line-height:1.6;">{signal}</div>'
            f'<div style="font-family:IBM Plex Mono,monospace;font-size:1.05rem;'
            f'font-weight:700;color:{color};margin-bottom:0.25rem;">{dest}</div>'
            f'<div style="font-size:0.82rem;color:{_PRI};font-weight:500;">{label}</div>'
            f'</div>'
        )
    cards_html += "</div>"
    st.markdown(cards_html, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="font-size:0.92rem;color:{_SEC};margin-top:0.7rem;line-height:1.7;">
        Pure regex + vocabulary lookup —
        <strong style="color:{_PRI};">zero ML inference</strong>, &lt;1 ms latency.
        Used in production; evaluation always compares all four strategies directly.
    </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# SECTION 4 · ARCHITECTURE AT A GLANCE  (no emojis)
# ════════════════════════════════════════════════════════════
def _arch_glance() -> None:
    section_header("Architecture at a Glance")

    cards_html = '<div style="display:grid;grid-template-columns:repeat(5,1fr);gap:1rem;">'
    for color, title, desc in ARCH_CARDS:
        cards_html += (
            f'<div style="background:{_BG};border:1px solid {_MUT};'
            f'border-top:4px solid {color};border-radius:8px;'
            f'padding:1.2rem 1rem;text-align:center;'
            f'min-height:100px;box-sizing:border-box;">'
            f'<div style="font-size:1rem;font-weight:700;color:{_PRI};'
            f'margin-bottom:0.4rem;">{title}</div>'
            f'<div style="font-size:0.88rem;color:{_SEC};line-height:1.6;">{desc}</div>'
            f'</div>'
        )
    cards_html += "</div>"
    st.markdown(cards_html, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# SECTION 5 · PIPELINE COMPONENTS
# ════════════════════════════════════════════════════════════
def _pipeline_components() -> None:
    section_header("Pipeline Components")

    # Single HTML block — flex row forces both panels to identical height
    LABEL = (
        f'font-size:0.78rem;text-transform:uppercase;letter-spacing:0.09em;'
        f'color:{_PRI};margin-bottom:0.8rem;font-weight:600;'
    )
    BLOCK = (
        f'background:#F8F9FD;border:1px solid {_MUT};border-radius:8px;'
        f'padding:1rem 1.2rem;font-family:IBM Plex Mono,monospace;'
        f'font-size:0.82rem;color:{_PRI};line-height:1.75;'
        f'height:100%;box-sizing:border-box;'
    )
    st.markdown(
        f'<div style="display:flex;gap:2rem;align-items:stretch;">'

        # Left — backend pipeline
        f'<div style="flex:1;display:flex;flex-direction:column;">'
        f'<div style="{LABEL}">Backend Pipeline</div>'
        f'<div style="{BLOCK}white-space:pre-wrap;word-break:break-word;flex:1;">'
        f'{BACKEND_TREE}</div>'
        f'</div>'

        # Right — project structure
        f'<div style="flex:1;display:flex;flex-direction:column;">'
        f'<div style="{LABEL}">Project Structure</div>'
        f'<div style="{BLOCK}white-space:pre;overflow-x:auto;flex:1;">'
        f'{DIRECTORY_TREE}</div>'
        f'</div>'

        f'</div>',
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════
# SECTION 6 · KEY CONFIGURATION
# ════════════════════════════════════════════════════════════
def _config_cards() -> None:
    section_header("Key Configuration")

    for row in [CONFIG_KEYS[:4], CONFIG_KEYS[4:]]:
        cards_html = '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;">'
        for label, value in row:
            cards_html += (
                f'<div style="background:{_BG};border:1px solid {_MUT};'
                f'border-radius:8px;padding:1.1rem 1.2rem;'
                f'min-height:90px;box-sizing:border-box;">'
                f'<div style="font-size:0.75rem;text-transform:uppercase;'
                f'letter-spacing:0.09em;color:{_PRI};margin-bottom:0.45rem;">{label}</div>'
                f'<div style="font-family:IBM Plex Mono,monospace;font-size:1.05rem;'
                f'font-weight:700;color:{_PRI};">{value}</div>'
                f'</div>'
            )
        cards_html += "</div>"
        st.markdown(cards_html, unsafe_allow_html=True)
        st.markdown("<div style='margin-bottom:0.7rem;'></div>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# SECTION 7 · CHECKPOINTING  (no emojis, step numbers instead)
# ════════════════════════════════════════════════════════════
def _checkpointing() -> None:
    section_header("Atomic Chunk-Based Checkpointing")

    n = len(CHECKPOINT_STEPS)
    widths = [1 if i % 2 == 0 else 0.15 for i in range(n * 2 - 1)]
    cols = st.columns(widths)

    for i, (num, title, desc) in enumerate(CHECKPOINT_STEPS):
        with cols[i * 2]:
            border = _PRI if i in (0, n - 1) else _SEC
            st.markdown(
                f'<div style="background:{_BG};border:1px solid {_MUT};'
                f'border-top:3px solid {border};border-radius:8px;'
                f'padding:1.1rem 0.8rem;text-align:center;">'
                f'<div style="font-family:IBM Plex Mono,monospace;font-size:0.72rem;'
                f'color:{_PRI};letter-spacing:0.1em;margin-bottom:0.35rem;">{num}</div>'
                f'<div style="font-size:0.95rem;font-weight:700;color:{_PRI};'
                f'white-space:pre-line;line-height:1.35;margin-bottom:0.35rem;">{title}</div>'
                f'<div style="font-size:0.82rem;color:{_SEC};'
                f'white-space:pre-line;line-height:1.5;">{desc}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        if i < n - 1:
            with cols[i * 2 + 1]:
                st.markdown(
                    f'<div style="display:flex;align-items:center;justify-content:center;'
                    f'height:100%;"><span style="font-size:1.3rem;color:{_SEC};">→</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    st.markdown(
        f'<div style="display:grid;grid-template-columns:1fr 1fr;'
        f'gap:0.7rem;margin-top:0.9rem;">'
        f'<div style="background:{_BG};border:1px solid {_MUT};'
        f'border-left:4px solid {_PRI};border-radius:0 8px 8px 0;'
        f'padding:0.8rem 1.1rem;font-size:0.92rem;color:{_PRI};line-height:1.65;">'
        f'Power failures only interrupt <strong>between</strong> chunks — '
        f'no partial or duplicate vectors are written.</div>'
        f'<div style="background:{_BG};border:1px solid {_MUT};'
        f'border-left:4px solid #5C6BC0;border-radius:0 8px 8px 0;'
        f'padding:0.8rem 1.1rem;font-size:0.92rem;color:{_PRI};line-height:1.65;">'
        f'Completed chunks are never reprocessed. '
        f'Delete <code>ingestion_progress.json</code> to restart from scratch.</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════
# MAIN RENDER  (evaluation_design removed)
# ════════════════════════════════════════════════════════════
def render() -> None:
    _hero()
    _strategy_cards()

    st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
    _query_router()

    st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
    _arch_glance()

    st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
    _pipeline_components()

    st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
    _config_cards()

    st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
    _checkpointing()