"""
pages/live_query.py - Query Investigation Workbench

Backward-compatible upgrade of the Live Query page.
All existing retrieval logic, APIs, and contracts are unchanged.
This file contains presentation-layer improvements only.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd

from assets.styles import section_header, result_row
from charts.plotly_charts import latency_bar, score_histogram
from loaders.retrieval_runner import classify_query, run_all_strategies

# ─────────────────────────────────────────────────────────────
# Constants — unchanged from original
# ─────────────────────────────────────────────────────────────

EXAMPLE_QUERIES = [
    "hvac cooling failure in research building",
    "plumbing leak repair in student dormitory",
    "fire protection system inspection in laboratory",
    "electrical panel fault teaching facility",
    "roof water damage repair",
    "elevator maintenance work order",
]

EXAMPLE_CATEGORIES = {
    "Semantic Query": [
        "roof water damage repair",
        "elevator maintenance work order",
    ],
    "Constrained Query": [
        "hvac fault in building A050",
        "plumbing issue B120",
    ],
    "Metadata-Heavy Query": [
        "hvac cooling failure in research building",
        "fire protection system inspection in laboratory",
        "electrical panel fault teaching facility",
    ],
}

STRATEGY_META = {
    "A":       ("#2D336B", "Semantic Only",  "Vector search, no metadata filter"),
    "B":       ("#7886C7", "Post-Filter",    "Fetch wide, filter after"),
    "B_prime": ("#A9B5DF", "Pre-Filter",     "Expanded pool, filter before top-k"),
    "C":       ("#5C6BC0", "MICE",           "Metadata baked into the embedding"),
}

STRATEGY_REASONING = {
    "A":       "No metadata signals detected — pure semantic vector search is appropriate.",
    "B":       "Building ID detected — post-filter safely narrows results by known identifier.",
    "B_prime": "Equipment/type signals detected — pre-filter expands pool then prunes efficiently.",
    "C":       "MICE index selected — metadata is embedded directly into the vector representation.",
}

# ─────────────────────────────────────────────────────────────
# Helper: improved result card with chips
# ─────────────────────────────────────────────────────────────

def _chip(label: str, value: str, color: str = "#2D336B") -> str:
    if not value or value == "nan":
        return ""
    return (
        f'<span style="display:inline-block;font-family:\'IBM Plex Mono\',monospace;'
        f'font-size:0.62rem;background:#F0F2FA;color:{color};border:1px solid #A9B5DF;'
        f'padding:0.1rem 0.5rem;border-radius:12px;margin-right:0.3rem;">'
        f'<span style="opacity:0.6;">{label}</span> {value}</span>'
    )


def _render_result_card(rank: int, r, color: str) -> None:
    is_top = rank == 1
    border_style = f"border-left:3px solid {color};" if is_top else "border-left:1px solid #A9B5DF;"
    bg = "#FAFBFF" if is_top else "#FFFFFF"
    rank_color = color if is_top else "#A9B5DF"

    btype    = str(r.wo_type)   if hasattr(r, "wo_type")    else ""
    equip    = str(r.equipment) if hasattr(r, "equipment")  else ""
    woid     = str(r.woid)      if hasattr(r, "woid")       else "—"
    score    = r.score          if hasattr(r, "score")      else 0.0
    desc     = str(r.wo_description) if hasattr(r, "wo_description") else ""
    building = str(r.building_id) if hasattr(r, "building_id") else ""

    chips = (
        _chip("building", building, "#2D336B") +
        _chip("type",     btype,    "#5C6BC0") +
        _chip("equip",    equip,    "#7886C7")
    )

    st.markdown(f"""
    <div style="background:{bg};{border_style}border-top:1px solid #A9B5DF;
                border-right:1px solid #A9B5DF;border-bottom:1px solid #A9B5DF;
                border-radius:4px;padding:0.6rem 0.85rem;margin-bottom:0.35rem;">
        <div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.25rem;">
            <span style="font-family:'IBM Plex Mono',monospace;font-size:0.7rem;
                         font-weight:600;color:{rank_color};min-width:1.6rem;">#{rank}</span>
            <span style="font-family:'IBM Plex Mono',monospace;font-size:0.72rem;
                         color:#5C6BC0;">{woid}</span>
            <span style="font-family:'IBM Plex Mono',monospace;font-size:0.72rem;
                         color:{color};font-weight:600;margin-left:auto;">{score:.4f}</span>
        </div>
        <div style="font-size:0.81rem;color:#2D336B;line-height:1.45;margin-bottom:0.3rem;">
            {desc[:200]}{'…' if len(desc) > 200 else ''}
        </div>
        <div style="margin-top:0.2rem;">{chips}</div>
    </div>""", unsafe_allow_html=True)


def _render_results(results: list, strategy: str) -> None:
    color, _, _ = STRATEGY_META[strategy]
    if not results:
        st.markdown(
            '<div style="color:#7886C7;font-size:0.82rem;padding:0.5rem;">'
            'No results returned.</div>',
            unsafe_allow_html=True,
        )
        return
    for i, r in enumerate(results[:10], 1):
        _render_result_card(i, r, color)


# ─────────────────────────────────────────────────────────────
# Section renderers
# ─────────────────────────────────────────────────────────────

def _render_pipeline_cards(query: str, analysis, top_k: int) -> None:
    """Section 1 — Query Analysis Pipeline (4 horizontal cards)."""
    import datetime

    section_header("Query Analysis Pipeline")

    c1, c2, c3, c4 = st.columns(4)

    # ── Card 1: Query Summary ──
    with c1:
        st.markdown(f"""
        <div style="background:#FFFFFF;border:1px solid #A9B5DF;border-top:3px solid #2D336B;
                    border-radius:4px;padding:0.9rem 1rem;height:100%;">
            <div style="font-size:0.62rem;text-transform:uppercase;letter-spacing:0.08em;
                        color:#7886C7;margin-bottom:0.5rem;">① Query Summary</div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:0.78rem;
                        color:#2D336B;font-weight:600;line-height:1.4;
                        word-break:break-word;margin-bottom:0.5rem;">
                "{query[:90]}{'…' if len(query) > 90 else ''}"
            </div>
            <div style="font-size:0.7rem;color:#7886C7;">
                {len(query.split())} tokens · top-{top_k} results
            </div>
            <div style="font-size:0.65rem;color:#A9B5DF;margin-top:0.2rem;">
                {datetime.datetime.now().strftime("%H:%M:%S")}
            </div>
        </div>""", unsafe_allow_html=True)

    # ── Card 2: Router Analysis ──
    signal_val = "—"
    is_constrained = False
    if analysis:
        signal_val = analysis.signal.value if hasattr(analysis.signal, "value") else str(analysis.signal)
        is_constrained = signal_val != "none"

    query_class = "Constrained" if is_constrained else "Semantic"
    class_color = "#5C6BC0" if is_constrained else "#2D336B"

    with c2:
        st.markdown(f"""
        <div style="background:#FFFFFF;border:1px solid #A9B5DF;border-top:3px solid #7886C7;
                    border-radius:4px;padding:0.9rem 1rem;height:100%;">
            <div style="font-size:0.62rem;text-transform:uppercase;letter-spacing:0.08em;
                        color:#7886C7;margin-bottom:0.5rem;">② Router Analysis</div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:1.1rem;
                        font-weight:700;color:{class_color};margin-bottom:0.3rem;">
                {query_class}
            </div>
            <div style="font-size:0.72rem;color:#2D336B;margin-bottom:0.3rem;">
                Signal: <strong>{signal_val}</strong>
            </div>
            <div style="font-size:0.68rem;color:#7886C7;line-height:1.5;">
                {"Metadata tokens detected — constrained retrieval path." if is_constrained
                  else "No metadata tokens — pure semantic path."}
            </div>
        </div>""", unsafe_allow_html=True)

    # ── Card 3: Metadata Detection ──
    equip    = (analysis.extracted_equipment   or None) if analysis else None
    btype    = (analysis.extracted_type        or None) if analysis else None
    bid      = (analysis.extracted_building_id or None) if analysis else None

    def _det_row(label: str, value) -> str:
        if value:
            return (f'<div style="font-size:0.72rem;color:#2D336B;margin-bottom:0.2rem;">'
                    f'<span style="color:#5C6BC0;font-weight:600;">✓</span> '
                    f'<span style="color:#7886C7;">{label}:</span> <strong>{value}</strong></div>')
        return (f'<div style="font-size:0.72rem;color:#A9B5DF;margin-bottom:0.2rem;">'
                f'<span>–</span> <span style="color:#A9B5DF;">{label}: None Detected</span></div>')

    with c3:
        st.markdown(f"""
        <div style="background:#FFFFFF;border:1px solid #A9B5DF;border-top:3px solid #A9B5DF;
                    border-radius:4px;padding:0.9rem 1rem;height:100%;">
            <div style="font-size:0.62rem;text-transform:uppercase;letter-spacing:0.08em;
                        color:#7886C7;margin-bottom:0.5rem;">③ Metadata Detection</div>
            {_det_row("Equipment",   equip)}
            {_det_row("Facility Type", btype)}
            {_det_row("Building ID", bid)}
        </div>""", unsafe_allow_html=True)

    # ── Card 4: Strategy Recommendation ──
    rec  = analysis.recommended_strategy if analysis else "A"
    rec_color, rec_label, rec_desc = STRATEGY_META.get(rec, ("#2D336B", "—", "—"))
    reasoning = STRATEGY_REASONING.get(rec, "")
    display_name = rec.replace("_prime", "′")

    with c4:
        st.markdown(f"""
        <div style="background:#FFFFFF;border:1px solid #A9B5DF;border-top:3px solid #5C6BC0;
                    border-radius:4px;padding:0.9rem 1rem;height:100%;">
            <div style="font-size:0.62rem;text-transform:uppercase;letter-spacing:0.08em;
                        color:#7886C7;margin-bottom:0.5rem;">④ Strategy Recommendation</div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:1.4rem;
                        font-weight:700;color:{rec_color};margin-bottom:0.2rem;">
                Strategy {display_name}
            </div>
            <div style="font-size:0.72rem;color:#2D336B;font-weight:600;
                        margin-bottom:0.3rem;">{rec_label}</div>
            <div style="font-size:0.68rem;color:#7886C7;line-height:1.5;">{reasoning}</div>
        </div>""", unsafe_allow_html=True)


def _render_retrieval_race(all_results: dict) -> None:
    """Section 2 — Retrieval Race: compact strategy comparison table."""
    section_header("Retrieval Race", "this query")

    latencies = {s: lat for s, (_, lat) in all_results.items()}
    counts    = {s: len(res) for s, (res, _) in all_results.items()}

    if not latencies:
        return

    min_lat = min(latencies.values())
    max_lat = max(latencies.values())

    header_html = """
    <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;
                gap:0.5rem;margin-bottom:0.4rem;">"""

    for s in ["A", "B", "B_prime", "C"]:
        lat   = latencies.get(s, 0)
        cnt   = counts.get(s, 0)
        color, label, _ = STRATEGY_META[s]
        display_s = s.replace("_prime", "′")

        badge = ""
        if lat == min_lat and len(latencies) > 1:
            badge = ('<span style="font-size:0.6rem;background:#E8F5E9;color:#2E7D32;'
                     'border:1px solid #A5D6A7;border-radius:10px;padding:0.05rem 0.4rem;'
                     'margin-left:0.3rem;">fastest</span>')
        elif lat == max_lat and len(latencies) > 1:
            badge = ('<span style="font-size:0.6rem;background:#FFF3E0;color:#E65100;'
                     'border:1px solid #FFCC80;border-radius:10px;padding:0.05rem 0.4rem;'
                     'margin-left:0.3rem;">slowest</span>')

        header_html += f"""
        <div style="background:#FFFFFF;border:1px solid #A9B5DF;border-top:2px solid {color};
                    border-radius:4px;padding:0.6rem 0.8rem;">
            <div style="font-family:'IBM Plex Mono',monospace;font-size:0.9rem;
                        font-weight:700;color:{color};">Strategy {display_s}</div>
            <div style="font-size:0.68rem;color:#7886C7;margin-bottom:0.3rem;">{label}</div>
            <div style="display:flex;align-items:baseline;gap:0.4rem;">
                <span style="font-family:'IBM Plex Mono',monospace;font-size:0.85rem;
                             color:#2D336B;font-weight:600;">{lat:.1f} ms</span>
                {badge}
            </div>
            <div style="font-size:0.7rem;color:#7886C7;margin-top:0.15rem;">
                {cnt} results returned
            </div>
        </div>"""

    header_html += "</div>"
    st.markdown(header_html, unsafe_allow_html=True)

    st.markdown(
        '<div style="font-size:0.7rem;color:#A9B5DF;margin-top:0.2rem;">'
        'Latency reflects FAISS search + Python post-processing. '
        'Recall / MRR / NDCG are shown in the Evaluation Dashboard.</div>',
        unsafe_allow_html=True,
    )


def _render_result_explorer(all_results: dict, fc) -> None:
    """Section 3 — Result Explorer: improved strategy tabs."""
    section_header("Result Explorer", "strategy tabs")

    tabs = st.tabs(["Strategy A", "Strategy B", "Strategy B′", "Strategy C",
                    "Score Distribution", "Latency"])

    for tab, s in zip(tabs[:4], ["A", "B", "B_prime", "C"]):
        with tab:
            results, lat = all_results[s]
            color, label, desc = STRATEGY_META[s]
            display_s = s.replace("_prime", "′")

            st.markdown(f"""
            <div style="background:#FFFFFF;border:1px solid #A9B5DF;border-radius:4px;
                        padding:0.6rem 1rem;margin-bottom:0.7rem;
                        display:flex;align-items:center;justify-content:space-between;">
                <div>
                    <span style="font-family:'IBM Plex Mono',monospace;font-size:0.88rem;
                                font-weight:700;color:{color};">Strategy {display_s}</span>
                    <span style="font-size:0.72rem;color:#7886C7;margin-left:0.7rem;">
                        {label} · {desc}</span>
                </div>
                <span style="font-family:'IBM Plex Mono',monospace;font-size:0.75rem;
                             color:#7886C7;">{lat:.1f} ms · {len(results)} results</span>
            </div>""", unsafe_allow_html=True)

            _render_results(results, s)

            if s in ("B", "B_prime") and fc:
                active = {k: v for k, v in [
                    ("building_id", fc.building_id),
                    ("type", fc.building_type),
                    ("equipment", fc.equipment),
                ] if v}
                if active:
                    chips = "".join(_chip(k, v) for k, v in active.items())
                    st.markdown(
                        f'<div style="font-size:0.7rem;color:#7886C7;margin-top:0.4rem;">'
                        f'Active filters: {chips}</div>',
                        unsafe_allow_html=True,
                    )

    with tabs[4]:
        scores_by_strategy = {
            s: [r.score for r in res]
            for s, (res, _) in all_results.items() if res
        }
        if scores_by_strategy:
            fig = score_histogram(scores_by_strategy, "Score Distribution by Strategy")
            st.plotly_chart(fig, width='stretch')
            st.markdown(
                '<div style="font-size:0.78rem;color:#7886C7;">'
                'Higher scores = more similar to the query vector. '
                'Strategy C (MICE) often shows a different distribution '
                'because queries are prefixed with "work order description: " '
                'to match the MICE document format.</div>',
                unsafe_allow_html=True,
            )

    with tabs[5]:
        latencies = {s: lat for s, (_, lat) in all_results.items()}
        fig_lat = latency_bar(latencies)
        st.plotly_chart(fig_lat, width='stretch')
        st.markdown(
            '<div style="font-size:0.78rem;color:#7886C7;line-height:1.6;">'
            'Strategy A is typically fastest (single FAISS search). '
            'B and B′ are slower due to the expanded candidate pool and Python-level filtering. '
            'C searches the MICE index with a prefixed query — latency is similar to A '
            'but the index contains metadata-infused vectors.</div>',
            unsafe_allow_html=True,
        )


def _render_divergence_analysis(all_results: dict) -> None:
    """Section 4 — Strategy Divergence Analysis."""
    section_header("Strategy Divergence Analysis", "overlap & uniqueness")

    woid_sets = {s: {r.woid for r in res} for s, (res, _) in all_results.items()}
    if not any(woid_sets.values()):
        st.caption("No results to compare.")
        return

    all_woids = set().union(*woid_sets.values())
    total = len(all_woids)

    # Common to all strategies
    common_all = set.intersection(*[s for s in woid_sets.values() if s])
    common_pct = round(len(common_all) / total * 100) if total else 0

    # Per-strategy unique (not in any other)
    unique_per = {}
    for s, ws in woid_sets.items():
        others = set().union(*[v for k, v in woid_sets.items() if k != s])
        unique_per[s] = ws - others

    col_summary, col_detail = st.columns([1, 2])

    with col_summary:
        st.markdown(f"""
        <div style="background:#FFFFFF;border:1px solid #A9B5DF;border-radius:4px;
                    padding:0.9rem 1rem;margin-bottom:0.5rem;">
            <div style="font-size:0.62rem;text-transform:uppercase;letter-spacing:0.08em;
                        color:#7886C7;margin-bottom:0.6rem;">Overlap Summary</div>
            <div style="display:flex;flex-direction:column;gap:0.4rem;">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <span style="font-size:0.75rem;color:#7886C7;">Total unique WOIDs</span>
                    <span style="font-family:'IBM Plex Mono',monospace;font-size:0.9rem;
                                 color:#2D336B;font-weight:700;">{total}</span>
                </div>
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <span style="font-size:0.75rem;color:#7886C7;">Common to all strategies</span>
                    <span style="font-family:'IBM Plex Mono',monospace;font-size:0.9rem;
                                 color:#5C6BC0;font-weight:700;">{len(common_all)}</span>
                </div>
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <span style="font-size:0.75rem;color:#7886C7;">Overlap %</span>
                    <span style="font-family:'IBM Plex Mono',monospace;font-size:0.9rem;
                                 color:#2D336B;font-weight:700;">{common_pct}%</span>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

        # Per-strategy unique counts
        for s, uniq in unique_per.items():
            color, label, _ = STRATEGY_META[s]
            display_s = s.replace("_prime", "′")
            st.markdown(f"""
            <div style="background:#FFFFFF;border:1px solid #A9B5DF;
                        border-left:3px solid {color};border-radius:4px;
                        padding:0.45rem 0.75rem;margin-bottom:0.3rem;
                        display:flex;justify-content:space-between;align-items:center;">
                <span style="font-size:0.72rem;color:#2D336B;">
                    <strong style="color:{color};">Strategy {display_s}</strong>
                    <span style="color:#7886C7;"> · {label}</span>
                </span>
                <span style="font-family:'IBM Plex Mono',monospace;font-size:0.8rem;
                             color:{color};font-weight:600;">{len(uniq)} unique</span>
            </div>""", unsafe_allow_html=True)

    with col_detail:
        # Full overlap table
        overlap_data = [
            {
                "WOID": w,
                "A":    "✓" if w in woid_sets.get("A", set())       else "·",
                "B":    "✓" if w in woid_sets.get("B", set())       else "·",
                "B′":   "✓" if w in woid_sets.get("B_prime", set()) else "·",
                "C":    "✓" if w in woid_sets.get("C", set())       else "·",
            }
            for w in sorted(all_woids)
        ]
        df = pd.DataFrame(overlap_data)
        st.dataframe(df, width='stretch', height=260)
        st.caption(
            "✓ = returned by strategy · · = not returned  |  "
            "Divergence reveals where metadata filtering reshapes the result set."
        )


def _render_metadata_impact(analysis, all_results: dict) -> None:
    """Section 5 — Metadata Impact Insights (research-focused explanatory UI)."""
    section_header("Metadata Impact Insights")

    equip = (analysis.extracted_equipment   or None) if analysis else None
    btype = (analysis.extracted_type        or None) if analysis else None
    bid   = (analysis.extracted_building_id or None) if analysis else None

    detected = [(k, v) for k, v in [
        ("Equipment",    equip),
        ("Facility Type", btype),
        ("Building ID",  bid),
    ] if v]

    has_meta = bool(detected)

    # Counts
    counts = {s: len(res) for s, (res, _) in all_results.items()}
    a_count = counts.get("A", 0)
    b_count = counts.get("B", 0)
    bp_count = counts.get("B_prime", 0)

    left, right = st.columns([1, 1])

    with left:
        if has_meta:
            detected_html = "".join(
                f'<div style="font-size:0.75rem;color:#2D336B;margin-bottom:0.25rem;">'
                f'<span style="color:#5C6BC0;font-weight:700;">✓</span> '
                f'<span style="color:#7886C7;">{label}:</span> <strong>{val}</strong></div>'
                for label, val in detected
            )
            st.markdown(f"""
            <div style="background:#FFFFFF;border:1px solid #A9B5DF;border-radius:4px;
                        padding:0.9rem 1rem;">
                <div style="font-size:0.72rem;font-weight:600;color:#2D336B;margin-bottom:0.5rem;">
                    Metadata Signals Detected
                </div>
                {detected_html}
                <div style="margin-top:0.6rem;padding-top:0.5rem;border-top:1px solid #F0F2FA;
                            font-size:0.75rem;color:#5C6BC0;line-height:1.6;">
                    This query is likely to <strong>benefit from metadata-aware retrieval</strong>.
                    Strategies B and B′ will apply filters; Strategy A will not.
                </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:#FFFFFF;border:1px solid #A9B5DF;border-radius:4px;
                        padding:0.9rem 1rem;">
                <div style="font-size:0.72rem;font-weight:600;color:#2D336B;margin-bottom:0.5rem;">
                    No Metadata Signals Detected
                </div>
                <div style="font-size:0.75rem;color:#7886C7;line-height:1.6;">
                    – Equipment: None Detected<br>
                    – Facility Type: None Detected<br>
                    – Building ID: None Detected
                </div>
                <div style="margin-top:0.6rem;padding-top:0.5rem;border-top:1px solid #F0F2FA;
                            font-size:0.75rem;color:#7886C7;line-height:1.6;">
                    Semantic retrieval is expected to perform similarly across strategies.
                    All strategies fall back to vector similarity.
                </div>
            </div>""", unsafe_allow_html=True)

    with right:
        # Expected vs actual behavior note
        if has_meta:
            meta_effect = ""
            if b_count != a_count or bp_count != a_count:
                diff_b  = b_count  - a_count
                diff_bp = bp_count - a_count
                meta_effect = (
                    f"B returned <strong>{abs(diff_b)}</strong> {'more' if diff_b >= 0 else 'fewer'} "
                    f"results than A. B′ returned <strong>{abs(diff_bp)}</strong> "
                    f"{'more' if diff_bp >= 0 else 'fewer'} results than A."
                )
            else:
                meta_effect = "All strategies returned an equal number of results for this query."

            st.markdown(f"""
            <div style="background:#FFFFFF;border:1px solid #A9B5DF;border-radius:4px;
                        padding:0.9rem 1rem;">
                <div style="font-size:0.72rem;font-weight:600;color:#2D336B;margin-bottom:0.5rem;">
                    Observed Metadata Effect
                </div>
                <div style="font-size:0.75rem;color:#2D336B;line-height:1.6;">
                    {meta_effect}
                </div>
                <div style="margin-top:0.5rem;font-size:0.72rem;color:#7886C7;line-height:1.5;">
                    Metadata filtering narrows the candidate space, which can
                    improve precision at the cost of recall when filters are noisy.
                </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:#FFFFFF;border:1px solid #A9B5DF;border-radius:4px;
                        padding:0.9rem 1rem;">
                <div style="font-size:0.72rem;font-weight:600;color:#2D336B;margin-bottom:0.5rem;">
                    Expected Retrieval Behaviour
                </div>
                <div style="font-size:0.75rem;color:#7886C7;line-height:1.6;">
                    Without metadata constraints, all four strategies operate on the
                    full vector index. Result sets should be highly similar. Divergence
                    between strategies is likely due to MICE embedding format differences
                    rather than filtering behaviour.
                </div>
            </div>""", unsafe_allow_html=True)


def _render_sample_gallery(current_query: str) -> str | None:
    """Section 6 — Sample Query Gallery (collapsible). Returns chosen query or None."""
    chosen = None
    with st.expander("Try Example Queries", expanded=False):
        st.markdown(
            '<div style="font-size:0.75rem;color:#7886C7;margin-bottom:0.7rem;">'
            'Click any query to populate the input field, then press Run.</div>',
            unsafe_allow_html=True,
        )
        for category, examples in EXAMPLE_CATEGORIES.items():
            st.markdown(
                f'<div style="font-size:0.65rem;text-transform:uppercase;'
                f'letter-spacing:0.08em;color:#A9B5DF;margin-bottom:0.3rem;">'
                f'{category}</div>',
                unsafe_allow_html=True,
            )
            cols = st.columns(len(examples))
            for col, ex in zip(cols, examples):
                with col:
                    label = ex[:35] + "…" if len(ex) > 35 else ex
                    if st.button(label, key=f"gallery_{ex[:20]}", width='stretch'):
                        chosen = ex
            st.markdown("<div style='margin-bottom:0.4rem;'></div>", unsafe_allow_html=True)
    return chosen


def _render_observation(query: str, analysis, all_results: dict) -> None:
    """Section 7 — Research Observation Panel (deterministic narrative)."""
    section_header("Observation")

    woid_sets = {s: {r.woid for r in res} for s, (res, _) in all_results.items()}
    counts    = {s: len(res) for s, (res, _) in all_results.items()}

    all_woids   = set().union(*woid_sets.values()) if any(woid_sets.values()) else set()
    common_all  = set.intersection(*[s for s in woid_sets.values() if s]) if any(woid_sets.values()) else set()
    total       = len(all_woids)
    common_pct  = round(len(common_all) / total * 100) if total else 0

    equip = (analysis.extracted_equipment   or None) if analysis else None
    btype = (analysis.extracted_type        or None) if analysis else None
    bid   = (analysis.extracted_building_id or None) if analysis else None
    has_meta = bool(equip or btype or bid)

    observations = []

    # Observation 1: overlap
    if common_pct >= 80:
        observations.append(
            "All strategies produced highly similar retrieval sets for this query. "
            f"{common_pct}% of returned work orders were common across all four strategies."
        )
    elif common_pct >= 40:
        observations.append(
            f"Moderate strategy divergence observed — {common_pct}% of results were shared "
            "across all strategies. Metadata filtering reshaped the result set for constrained strategies."
        )
    else:
        observations.append(
            f"High strategy divergence: only {common_pct}% of results were shared across all strategies. "
            "Metadata filters significantly reduced the candidate search space."
        )

    # Observation 2: metadata influence
    if has_meta:
        signals = ", ".join(filter(None, [equip, btype, bid]))
        a_count  = counts.get("A", 0)
        bp_count = counts.get("B_prime", 0)
        if bp_count < a_count:
            observations.append(
                f"Metadata signals ({signals}) activated pre-filtering in Strategy B′, "
                f"which returned {a_count - bp_count} fewer results than Strategy A. "
                "This demonstrates how metadata reduces the candidate pool."
            )
        elif bp_count > a_count:
            observations.append(
                f"Despite metadata pre-filtering, Strategy B′ returned {bp_count - a_count} "
                "more results than Strategy A due to the expanded candidate pool."
            )
        else:
            observations.append(
                f"Metadata signals ({signals}) were detected but did not change result "
                "counts relative to Strategy A for this particular query."
            )
    else:
        observations.append(
            "No metadata signals were detected. All strategies operated on the full vector "
            "index. Any divergence between strategies is attributable to the MICE embedding "
            "format rather than metadata-driven filtering."
        )

    # Observation 3: unique strategy findings
    c_woids = woid_sets.get("C", set())
    a_woids = woid_sets.get("A", set())
    c_unique = c_woids - a_woids
    a_unique = a_woids - c_woids
    if c_unique:
        observations.append(
            f"Strategy C (MICE) returned {len(c_unique)} work order(s) not found by Strategy A, "
            "suggesting the metadata-infused embedding space captures different semantic clusters."
        )
    if a_unique:
        observations.append(
            f"Strategy A returned {len(a_unique)} work order(s) not found by Strategy C, "
            "consistent with MICE embedding changing the effective similarity landscape."
        )

    # Render
    obs_html = '<div style="background:#FFFFFF;border:1px solid #A9B5DF;border-radius:4px;padding:0.9rem 1.2rem;">'
    for i, obs in enumerate(observations, 1):
        obs_html += f"""
        <div style="display:flex;gap:0.6rem;margin-bottom:0.55rem;
                    {'border-top:1px solid #F0F2FA;padding-top:0.55rem;' if i > 1 else ''}">
            <span style="font-family:'IBM Plex Mono',monospace;font-size:0.68rem;
                         color:#A9B5DF;min-width:1.2rem;padding-top:0.05rem;">{i}.</span>
            <span style="font-size:0.78rem;color:#2D336B;line-height:1.6;">{obs}</span>
        </div>"""
    obs_html += "</div>"
    st.markdown(obs_html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Main render
# ─────────────────────────────────────────────────────────────

def render():
    st.markdown(
        '<div class="page-title">Query Investigation Workbench</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="page-subtitle">'
        'Enter a natural-language query to observe how the routing system classifies it, '
        'how metadata signals influence retrieval, and how all four strategies behave '
        'against the real FAISS indexes.'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Query input ──────────────────────────────────────────
    section_header("Query")
    q1, q2 = st.columns([4, 1])
    with q1:
        query = st.text_input(
            "Natural language query",
            placeholder="e.g. hvac cooling failure in research building",
            label_visibility="collapsed",
            key="workbench_query",
        )
    with q2:
        top_k = st.number_input("Top-k", min_value=1, max_value=20, value=10)

    # Quick example buttons (original behavior preserved)
    st.markdown(
        '<div style="font-size:0.7rem;color:#7886C7;margin-bottom:0.3rem;">'
        'Quick examples:</div>',
        unsafe_allow_html=True,
    )
    ex_cols = st.columns(len(EXAMPLE_QUERIES))
    for col, ex in zip(ex_cols, EXAMPLE_QUERIES):
        with col:
            label = (ex[:30] + "…") if len(ex) > 30 else ex
            if st.button(label, key=f"ex_{ex[:15]}", width='stretch'):
                query = ex

    # ── Sample Query Gallery (Section 6) ────────────────────
    gallery_choice = _render_sample_gallery(query)
    if gallery_choice:
        query = gallery_choice

    # ── Manual metadata filter override (preserved) ─────────
    with st.expander("Manual metadata filters (optional — overrides router)"):
        mf1, mf2, mf3 = st.columns(3)
        with mf1:
            manual_bid   = st.text_input("Building ID",   placeholder="e.g. A050")
        with mf2:
            manual_btype = st.text_input("Facility Type", placeholder="e.g. research")
        with mf3:
            manual_equip = st.text_input("Equipment",     placeholder="e.g. hvac")

    if not query:
        st.info("Enter a query above to run the investigation.")
        return

    # ── Router classification (unchanged API call) ───────────
    analysis = classify_query(query)

    # ── Build FilterConfig (unchanged) ──────────────────────
    try:
        from retrieval import FilterConfig
        fc = FilterConfig(
            building_id   = manual_bid   or (analysis.extracted_building_id if analysis else None),
            building_type = manual_btype or (analysis.extracted_type        if analysis else None),
            equipment     = manual_equip or (analysis.extracted_equipment   if analysis else None),
        )
    except Exception:
        fc = None

    # ── Section 1: Query Analysis Pipeline ──────────────────
    _render_pipeline_cards(query, analysis, int(top_k))

    # ── Run retrieval (unchanged API call) ───────────────────
    st.markdown("<div style='margin-top:0.5rem;'></div>", unsafe_allow_html=True)
    with st.spinner("Running retrieval across all strategies…"):
        all_results = run_all_strategies(query, top_k=int(top_k), filter_config=fc)

    # ── Section 2: Retrieval Race ────────────────────────────
    _render_retrieval_race(all_results)

    # ── Section 3: Result Explorer ───────────────────────────
    _render_result_explorer(all_results, fc)

    # ── Sections 4 & 5: two-column layout ───────────────────
    st.markdown("<div style='margin-top:0.2rem;'></div>", unsafe_allow_html=True)
    div4, div5 = st.columns([1, 1])

    with div4:
        _render_divergence_analysis(all_results)

    with div5:
        _render_metadata_impact(analysis, all_results)

    # ── Section 7: Observation ───────────────────────────────
    _render_observation(query, analysis, all_results)