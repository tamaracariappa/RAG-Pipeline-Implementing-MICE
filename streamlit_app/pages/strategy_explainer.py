"""
pages/strategy_explainer.py - Visual comparison of the four retrieval strategies.
"""

import streamlit as st

from assets.styles import section_header


STRATEGIES = [
    {
        "key":    "A",
        "color":  "#4f8ef7",
        "label":  "Strategy A",
        "title":  "Semantic Baseline",
        "what":   "Pure vector similarity. Query is embedded and compared against "
                  "all TEXT index vectors. Top-k by cosine score are returned.",
        "index":  "TEXT index only",
        "filter": "None",
        "pros": [
            "Fastest retrieval (single FAISS search)",
            "No metadata required at query time",
            "Best recall for queries with no metadata context",
        ],
        "cons": [
            "Cannot distinguish facility types or equipment without metadata",
            "Returns semantically similar but contextually wrong results",
        ],
        "flow": [
            ("Query", "#4f8ef7"),
            ("embed_query()", "#4f8ef7"),
            ("FAISS search · TEXT index", "#4f8ef7"),
            ("Top-k results", "#4f8ef7"),
        ],
    },
    {
        "key":    "B",
        "color":  "#38c96e",
        "label":  "Strategy B",
        "title":  "Semantic + Post-Filter",
        "what":   "Fetch a large candidate pool from the TEXT index (top_k × 20), "
                  "then apply Python-level metadata filters to trim to top_k.",
        "index":  "TEXT index",
        "filter": "Python post-filter",
        "pros": [
            "Compatible with any metadata field combination",
            "No need to retrain embeddings",
            "Filter precision independent of embedding quality",
        ],
        "cons": [
            "Larger candidate pool → slower than A",
            "May miss relevant results if pool is too small",
            "Over-filtering can leave fewer than top_k results",
        ],
        "flow": [
            ("Query + FilterConfig", "#38c96e"),
            ("embed_query()", "#38c96e"),
            ("FAISS search · top_k × 20", "#38c96e"),
            ("Python metadata filter", "#38c96e"),
            ("Top-k filtered results", "#38c96e"),
        ],
    },
    {
        "key":    "B_prime",
        "color":  "#f7a94f",
        "label":  "Strategy B′",
        "title":  "Pre-Filter",
        "what":   "Similar to B but the metadata filtering is positioned before "
                  "the final top-k selection. Semantically strongest filtered "
                  "candidates are ranked first.",
        "index":  "TEXT index",
        "filter": "Python pre-selection filter",
        "pros": [
            "Retains semantic ranking within filtered set",
            "Better precision than B when filter is selective",
            "No index modification needed",
        ],
        "cons": [
            "Still requires expanded candidate pool",
            "Effectiveness depends on filter selectivity",
        ],
        "flow": [
            ("Query + FilterConfig", "#f7a94f"),
            ("embed_query()", "#f7a94f"),
            ("FAISS search · expanded pool", "#f7a94f"),
            ("Filter → semantic re-rank", "#f7a94f"),
            ("Top-k results", "#f7a94f"),
        ],
    },
    {
        "key":    "C",
        "color":  "#e06c75",
        "label":  "Strategy C",
        "title":  "MICE — Metadata-Infused",
        "what":   "Metadata is embedded into the vector itself using a structured "
                  "sentence template. The MICE index is queried directly with a "
                  "prefixed query. No post-filtering required.",
        "index":  "MICE index",
        "filter": "None (implicit in vector)",
        "pros": [
            "Metadata selectivity is part of the similarity score",
            "Single-pass retrieval — no post-filtering overhead",
            "Potentially higher recall when metadata is strongly indicative",
        ],
        "cons": [
            "Requires a second index (storage overhead)",
            "Query prefix must match document template style",
            "Training distribution sensitivity",
        ],
        "flow": [
            ("Query", "#e06c75"),
            ('"work order description: " + query', "#e06c75"),
            ("embed_query()", "#e06c75"),
            ("FAISS search · MICE index", "#e06c75"),
            ("Top-k results", "#e06c75"),
        ],
    },
]


def _flow_diagram(steps: list) -> None:
    parts = []
    for i, (label, color) in enumerate(steps):
        parts.append(f"""
        <div style="background:#0d1117;border:1px solid {color};
                    border-radius:4px;padding:0.4rem 0.65rem;
                    font-size:0.72rem;color:{color};
                    font-family:'IBM Plex Mono',monospace;
                    white-space:nowrap;">{label}</div>""")
        if i < len(steps) - 1:
            parts.append(
                f'<div style="color:{color};font-size:0.9rem;">↓</div>')
    st.markdown(
        '<div style="display:flex;flex-direction:column;'
        'align-items:flex-start;gap:0.3rem;">'
        + "".join(parts) + "</div>",
        unsafe_allow_html=True,
    )


def render():
    st.markdown('<div class="page-title">Retrieval Strategies</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">'
        'Four approaches to work-order retrieval — from pure semantics '
        'to metadata-infused embeddings.'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── At-a-glance comparison ───────────────────────────────
    section_header("At a Glance", "4 strategies")

    cols = st.columns(4)
    for col, s in zip(cols, STRATEGIES):
        with col:
            pros_html = "".join(
                f'<div style="font-size:0.73rem;color:#38c96e;margin:0.15rem 0;">'
                f'✓ {p}</div>' for p in s["pros"][:2])
            cons_html = "".join(
                f'<div style="font-size:0.73rem;color:#e06c75;margin:0.15rem 0;">'
                f'✗ {c}</div>' for c in s["cons"][:2])
            st.markdown(f"""
            <div style="background:#1a1d27;border:1px solid #2a2d3e;
                        border-top:3px solid {s['color']};border-radius:6px;
                        padding:1rem;min-height:260px;">
                <div style="font-family:'IBM Plex Mono',monospace;font-size:1.4rem;
                            font-weight:600;color:{s['color']};">{s['key']}</div>
                <div style="font-size:0.72rem;font-weight:600;
                            text-transform:uppercase;letter-spacing:0.06em;
                            color:#e4e6f0;margin:0.3rem 0 0.6rem;">{s['title']}</div>
                <div style="font-size:0.75rem;color:#8890a8;
                            line-height:1.5;margin-bottom:0.7rem;">{s['what'][:150]}…</div>
                {pros_html}
                {cons_html}
            </div>""", unsafe_allow_html=True)

    # ── Detailed strategy cards ──────────────────────────────
    st.markdown("<br/>", unsafe_allow_html=True)
    section_header("Detailed Breakdown")

    for s in STRATEGIES:
        with st.expander(
            f"Strategy {s['key']} - {s['title']}", expanded=(s['key'] == 'A')
        ):
            d1, d2 = st.columns([2, 1], gap="large")

            with d1:
                st.markdown(f"""
                <div style="font-size:0.82rem;color:#e4e6f0;
                            line-height:1.7;margin-bottom:0.8rem;">
                    {s['what']}
                </div>""", unsafe_allow_html=True)

                st.markdown("""
                <div style="font-size:0.7rem;letter-spacing:0.08em;
                            text-transform:uppercase;color:#8890a8;
                            margin-bottom:0.4rem;">Strengths</div>""",
                            unsafe_allow_html=True)
                for p in s["pros"]:
                    st.markdown(
                        f'<div style="font-size:0.8rem;color:#38c96e;'
                        f'margin:0.2rem 0;">✓ {p}</div>',
                        unsafe_allow_html=True)

                st.markdown("""
                <div style="font-size:0.7rem;letter-spacing:0.08em;
                            text-transform:uppercase;color:#8890a8;
                            margin-top:0.6rem;margin-bottom:0.4rem;">Limitations</div>""",
                            unsafe_allow_html=True)
                for c in s["cons"]:
                    st.markdown(
                        f'<div style="font-size:0.8rem;color:#e06c75;'
                        f'margin:0.2rem 0;">✗ {c}</div>',
                        unsafe_allow_html=True)

                st.markdown(f"""
                <div style="margin-top:0.8rem;font-size:0.75rem;color:#8890a8;">
                    <span style="color:#e4e6f0;">Index:</span> {s['index']}
                    &nbsp;·&nbsp;
                    <span style="color:#e4e6f0;">Filter:</span> {s['filter']}
                </div>""", unsafe_allow_html=True)

            with d2:
                st.markdown("""
                <div style="font-size:0.7rem;letter-spacing:0.08em;
                            text-transform:uppercase;color:#8890a8;
                            margin-bottom:0.5rem;">Execution Flow</div>""",
                            unsafe_allow_html=True)
                _flow_diagram(s["flow"])

    # ── Key design decisions ─────────────────────────────────
    st.markdown("<br/>", unsafe_allow_html=True)
    section_header("Design Decisions")

    r1, r2 = st.columns(2, gap="large")
    with r1:
        st.markdown("""
        <div style="background:#1a1d27;border:1px solid #2a2d3e;
                    border-radius:6px;padding:1rem;">
            <div style="font-size:0.72rem;letter-spacing:0.08em;
                        text-transform:uppercase;color:#4f8ef7;
                        margin-bottom:0.5rem;">POST_FILTER_MULTIPLIER = 20</div>
            <div style="font-size:0.8rem;color:#8890a8;line-height:1.6;">
                Strategies B and B′ fetch <code>top_k × 20</code> candidates before
                filtering. This ensures enough relevant results survive the filter
                even when the filter is highly selective (e.g., a rare building type).
                Tunable in <code>config.py</code>.
            </div>
        </div>""", unsafe_allow_html=True)

    with r2:
        st.markdown("""
        <div style="background:#1a1d27;border:1px solid #2a2d3e;
                    border-radius:6px;padding:1rem;">
            <div style="font-size:0.72rem;letter-spacing:0.08em;
                        text-transform:uppercase;color:#e06c75;
                        margin-bottom:0.5rem;">BGE Query Prefix (MICE)</div>
            <div style="font-size:0.8rem;color:#8890a8;line-height:1.6;">
                Strategy C prepends <code>"work order description: "</code>
                to the query before embedding. This matches the MICE document
                template format, ensuring the query vector occupies the same
                region of embedding space as the indexed documents.
            </div>
        </div>""", unsafe_allow_html=True)

    # ── Query router ─────────────────────────────────────────
    st.markdown("<br/>", unsafe_allow_html=True)
    section_header("Query Router")
    st.markdown("""
    <div style="background:#1a1d27;border:1px solid #2a2d3e;
                border-radius:6px;padding:1.1rem;">
        <div style="font-size:0.8rem;color:#8890a8;line-height:1.8;">
            The <code>query_router.py</code> module automatically selects a strategy:
            <ul style="margin-top:0.4rem;padding-left:1.2rem;">
                <li>Equipment or type vocabulary detected → <strong style="color:#f7a94f;">B′</strong></li>
                <li>Building ID pattern detected → <strong style="color:#38c96e;">B</strong></li>
                <li>No metadata signals → <strong style="color:#4f8ef7;">A</strong></li>
                <li>Explicit override → <strong style="color:#e06c75;">C</strong></li>
            </ul>
            The router is zero-latency (pure regex + vocabulary lookup, no ML inference).
            Used in production; experiments always compare all four directly.
        </div>
    </div>""", unsafe_allow_html=True)
