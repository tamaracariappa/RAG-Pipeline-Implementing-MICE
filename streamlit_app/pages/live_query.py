"""
pages/live_query.py - Live retrieval demo against real FAISS indexes.

Calls the existing retrieval.py strategies directly.
Displays results side-by-side with latency and router decision.
"""

from __future__ import annotations

import streamlit as st

from assets.styles import section_header, result_row
from charts.plotly_charts import latency_bar, score_histogram
from loaders.retrieval_runner import classify_query, run_all_strategies

EXAMPLE_QUERIES = [
    "hvac cooling failure in research building",
    "plumbing leak repair in student dormitory",
    "fire protection system inspection in laboratory",
    "electrical panel fault teaching facility",
    "roof water damage repair",
    "elevator maintenance work order",
]

STRATEGY_META = {
    "A":       ("#4f8ef7", "Semantic Only",        "Vector search, no metadata filter"),
    "B":       ("#38c96e", "Post-Filter",           "Fetch wide, filter after"),
    "B_prime": ("#f7a94f", "Pre-Filter",            "Expanded pool, filter before top-k"),
    "C":       ("#e06c75", "MICE",                  "Metadata baked into the embedding"),
}


def _render_results(results: list, strategy: str) -> None:
    color, _, _ = STRATEGY_META[strategy]
    if not results:
        st.markdown(
            f'<div style="color:#8890a8;font-size:0.82rem;padding:0.5rem;">'
            f'No results returned.</div>',
            unsafe_allow_html=True,
        )
        return
    for i, r in enumerate(results[:10], 1):
        result_row(
            rank=i,
            woid=r.woid,
            score=r.score,
            desc=r.wo_description,
            btype=r.wo_type,
            equipment=r.equipment,
        )


def render():
    st.markdown('<div class="page-title">Live Query Testing</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">'
        'Enter a natural-language query and run all four retrieval strategies '
        'against the real FAISS indexes.'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Query input ──────────────────────────────────────────
    section_header("Query")
    q1, q2 = st.columns([3, 1])
    with q1:
        query = st.text_input(
            "Natural language query",
            placeholder="e.g. hvac cooling failure in research building",
            label_visibility="collapsed",
        )
    with q2:
        top_k = st.number_input("Top-k", min_value=1, max_value=20, value=10)

    # Example queries
    st.markdown(
        '<div style="font-size:0.7rem;color:#8890a8;margin-bottom:0.3rem;">'
        'Quick examples:</div>',
        unsafe_allow_html=True,
    )
    ex_cols = st.columns(len(EXAMPLE_QUERIES))
    for col, ex in zip(ex_cols, EXAMPLE_QUERIES):
        with col:
            if st.button(ex[:30] + "…" if len(ex) > 30 else ex,
                         key=f"ex_{ex[:15]}", use_container_width=True):
                query = ex

    # ── Metadata filter override ─────────────────────────────
    with st.expander("Manual metadata filters (optional — overrides router)"):
        mf1, mf2, mf3 = st.columns(3)
        with mf1:
            manual_bid   = st.text_input("Building ID", placeholder="e.g. A050")
        with mf2:
            manual_btype = st.text_input("Facility Type",
                                         placeholder="e.g. research")
        with mf3:
            manual_equip = st.text_input("Equipment",
                                         placeholder="e.g. hvac")

    if not query:
        st.info("Enter a query above to run retrieval.")
        return

    # ── Router classification ────────────────────────────────
    analysis = classify_query(query)

    if analysis:
        section_header("Router Decision")
        rd1, rd2, rd3, rd4 = st.columns(4)
        with rd1:
            st.metric("Recommended Strategy", analysis.recommended_strategy)
        with rd2:
            st.metric("Signal Type", analysis.signal.value if hasattr(analysis.signal, 'value') else str(analysis.signal))
        with rd3:
            st.metric("Equipment Detected",
                      analysis.extracted_equipment or "—")
        with rd4:
            st.metric("Type Detected",
                      analysis.extracted_type or "—")

    # ── Build FilterConfig ───────────────────────────────────
    try:
        from retrieval import FilterConfig
        fc = FilterConfig(
            building_id   = manual_bid   or (analysis.extracted_building_id if analysis else None),
            building_type = manual_btype or (analysis.extracted_type        if analysis else None),
            equipment     = manual_equip or (analysis.extracted_equipment   if analysis else None),
        )
    except Exception:
        fc = None

    # ── Run retrieval ────────────────────────────────────────
    section_header("Retrieval Results", "all 4 strategies")

    with st.spinner("Running retrieval…"):
        all_results = run_all_strategies(query, top_k=int(top_k), filter_config=fc)

    # ── Latency summary ──────────────────────────────────────
    latencies = {s: lat for s, (_, lat) in all_results.items()}
    lc1, lc2, lc3, lc4 = st.columns(4)
    for col, (s, lat) in zip([lc1, lc2, lc3, lc4], latencies.items()):
        color, label, _ = STRATEGY_META[s]
        with col:
            st.markdown(f"""
            <div style="background:#1a1d27;border:1px solid #2a2d3e;
                        border-top:2px solid {color};border-radius:4px;
                        padding:0.7rem;text-align:center;">
                <div style="font-family:'IBM Plex Mono',monospace;font-size:1.1rem;
                            color:{color};font-weight:600;">{s}</div>
                <div style="font-size:0.68rem;color:#8890a8;text-transform:uppercase;
                            letter-spacing:0.06em;">{label}</div>
                <div style="font-family:'IBM Plex Mono',monospace;font-size:0.85rem;
                            color:#38c96e;margin-top:0.3rem;">{lat:.1f} ms</div>
                <div style="font-size:0.7rem;color:#8890a8;">
                    {len(all_results[s][0])} results
                </div>
            </div>""", unsafe_allow_html=True)

    # ── Side-by-side tabs ────────────────────────────────────
    st.markdown("<br/>", unsafe_allow_html=True)
    tabs = st.tabs(["Strategy A", "Strategy B", "Strategy B′", "Strategy C",
                    "Score Distribution", "Latency"])

    for tab, s in zip(tabs[:4], ["A", "B", "B_prime", "C"]):
        with tab:
            results, lat = all_results[s]
            color, label, desc = STRATEGY_META[s]

            st.markdown(f"""
            <div style="background:#1a1d27;border:1px solid #2a2d3e;
                        border-radius:4px;padding:0.7rem 1rem;
                        margin-bottom:0.8rem;">
                <span style="font-family:'IBM Plex Mono',monospace;
                            font-size:0.9rem;font-weight:600;color:{color};">
                    Strategy {s}</span>
                <span style="font-size:0.75rem;color:#8890a8;margin-left:0.8rem;">
                    {label} · {desc}</span>
                <span style="font-family:'IBM Plex Mono',monospace;
                            font-size:0.75rem;color:#38c96e;float:right;">
                    {lat:.1f} ms · {len(results)} results</span>
            </div>""", unsafe_allow_html=True)

            _render_results(results, s)

            # Filter info for B / B_prime
            if s in ("B", "B_prime") and fc:
                active_filters = {
                    k: v for k, v in [
                        ("building_id", fc.building_id),
                        ("type", fc.building_type),
                        ("equipment", fc.equipment),
                    ] if v
                }
                if active_filters:
                    filter_str = " · ".join(
                        f'{k}=<b>{v}</b>' for k, v in active_filters.items())
                    st.markdown(
                        f'<div style="font-size:0.72rem;color:#f7a94f;'
                        f'margin-top:0.5rem;">Active filters: {filter_str}</div>',
                        unsafe_allow_html=True,
                    )

    with tabs[4]:
        scores_by_strategy = {}
        for s, (results, _) in all_results.items():
            if results:
                scores_by_strategy[s] = [r.score for r in results]
        if scores_by_strategy:
            fig = score_histogram(
                scores_by_strategy, "Score Distribution by Strategy")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("""
            <div style="font-size:0.78rem;color:#8890a8;">
                Higher scores = more similar to the query vector.
                Strategy C (MICE) often shows a different distribution
                because queries are prefixed with "work order description: "
                to match the MICE document format.
            </div>""", unsafe_allow_html=True)

    with tabs[5]:
        fig_lat = latency_bar(latencies)
        st.plotly_chart(fig_lat, use_container_width=True)
        st.markdown("""
        <div style="font-size:0.78rem;color:#8890a8;line-height:1.6;">
            Strategy A is typically fastest (single FAISS search).
            B and B′ are slower due to the expanded candidate pool and Python-level filtering.
            C searches the MICE index with a prefixed query—latency is similar to A
            but the index contains metadata-infused vectors.
        </div>""", unsafe_allow_html=True)

    # ── WOID comparison table ────────────────────────────────
    st.markdown("<br/>", unsafe_allow_html=True)
    section_header("Overlap Analysis")
    woid_sets = {s: {r.woid for r in res} for s, (res, _) in all_results.items()}
    if any(woid_sets.values()):
        import pandas as pd
        all_woids = sorted(set().union(*woid_sets.values()))
        overlap_data = [
            {
                "WOID": w,
                "A":  "✓" if w in woid_sets.get("A", set()) else "",
                "B":  "✓" if w in woid_sets.get("B", set()) else "",
                "B′": "✓" if w in woid_sets.get("B_prime", set()) else "",
                "C":  "✓" if w in woid_sets.get("C", set()) else "",
            }
            for w in all_woids
        ]
        overlap_df = pd.DataFrame(overlap_data)
        st.dataframe(overlap_df, use_container_width=True, height=300)
        st.caption(
            f"Total unique WOIDs across all strategies: {len(all_woids)}. "
            "Divergence shows where metadata filtering changes the result set."
        )
