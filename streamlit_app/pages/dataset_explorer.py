"""
pages/dataset_explorer.py - Interactive Dataset & Representation Explorer.

Redesigned for university evaluators and research reviewers.
All existing loaders, helpers, and session state behaviour preserved.
"""

import streamlit as st
import pandas as pd

from assets.styles import repr_block
from loaders.data_loader import load_cleaned_df


# ── Representation builders (unchanged) ─────────────────────

def _build_text_repr(row: dict) -> str:
    parts = [
        str(row.get("BuildingName", "")).strip(),
        str(row.get("Type", "")).strip(),
        str(row.get("WODescription", "")).strip(),
    ]
    return " | ".join(p for p in parts if p)


def _build_mice_repr(row: dict) -> str:
    def v(key, *bad):
        val = str(row.get(key, "")).strip().lower()
        if not val or val in ("nan", "none", "unknown_building", "unknown_type", *bad):
            return "unknown"
        return val
    return (
        f"building id: {v('BuildingID')}.\n"
        f"building name: {v('BuildingName')}.\n"
        f"facility type: {v('Type')}.\n"
        f"equipment system: {v('equipment')}.\n"
        f"work period: {v('WOStartDate')} to {v('WOEndDate')}.\n"
        f"work order description: {v('WODescription')}."
    )


# ── Quality cards (static — educational) ────────────────────

QUALITY_CARDS = [
    ("\u2713", "Duplicate WOIDs Removed",       "Exact WOID duplicates identified and dropped before indexing."),
    ("\u2713", "Missing Building IDs Grouped",   "Records with no BuildingID consolidated under a shared placeholder."),
    ("\u2713", "Schema Standardised",            "Column names normalised; date fields parsed to consistent ISO format."),
    ("\u2713", "Weak Descriptions Removed",      "Work orders with empty or single-token descriptions excluded."),
    ("\u2713", "Embedding-Ready Dataset",        "All retained records have non-null WODescription, BuildingName, and Type."),
]


def render():
    # ── Page-scoped CSS ──────────────────────────────────────
    st.markdown("""
    <style>
    .main .block-container {
        max-width: 1600px !important;
        padding: 2.5rem 3rem 5rem !important;
    }

    /* ── Hero ── */
    .de-hero {
        padding: 2rem 0 1.8rem;
    }
    .de-eyebrow {
        font-family: var(--mono);
        font-size: 0.72rem;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: var(--accent2);
        margin-bottom: 0.6rem;
    }
    .de-title {
        font-size: 2.6rem;
        font-weight: 700;
        color: var(--accent);
        letter-spacing: -0.02em;
        line-height: 1.1;
        margin-bottom: 0.7rem;
    }
    .de-subtitle {
        font-size: 1.05rem;
        color: #4A5490;
        line-height: 1.6;
        margin: 0;
    }

    /* ── Stat cards grid ── */
    .de-stat-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
        margin-top: 1.6rem;
    }
    .de-stat-card {
        background: #FFFFFF;
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 1.4rem 1.6rem;
        display: flex;
        flex-direction: column;
        gap: 0.3rem;
        transition: border-color 0.18s, box-shadow 0.18s;
    }
    .de-stat-card:hover {
        border-color: #2D336B;
        box-shadow: 0 2px 10px rgba(45,51,107,0.09);
    }
    .de-stat-label {
        font-family: var(--mono);
        font-size: 0.65rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--muted);
    }
    .de-stat-value {
        font-family: var(--mono);
        font-size: 2rem;
        font-weight: 700;
        color: var(--accent);
        line-height: 1.1;
    }
    .de-stat-sub {
        font-size: 0.85rem;
        color: var(--accent2);
    }

    /* ── Section label + divider ── */
    .de-sec-div {
        margin: 2rem 0 1.4rem;
        border: none;
        border-top: 1px solid var(--border);
    }
    .de-sec-label {
        font-family: var(--mono);
        font-size: 0.68rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: var(--accent2);
        margin-bottom: 0.9rem;
    }
    .de-sec-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: var(--accent);
        margin-bottom: 1rem;
    }

    /* ── Quality cards ── */
    .de-quality-grid {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 0.85rem;
    }
    .de-quality-card {
        background: #FFFFFF;
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 1.2rem 1.3rem;
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
    }
    .de-quality-check {
        font-size: 1.1rem;
        font-weight: 700;
        color: #2D336B;
        background: #F0F3FF;
        width: 2rem;
        height: 2rem;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .de-quality-name {
        font-size: 0.93rem;
        font-weight: 600;
        color: var(--accent);
        line-height: 1.3;
    }
    .de-quality-desc {
        font-size: 0.85rem;
        color: #4A5490;
        line-height: 1.5;
    }

    /* ── Filter panel ── */
    .de-filter-title {
        font-family: var(--mono);
        font-size: 0.68rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: var(--accent2);
        margin-bottom: 1rem;
    }
    .de-record-count {
        font-size: 0.92rem;
        color: var(--accent2);
        margin-top: 0.8rem;
        font-family: var(--mono);
    }

    /* ── Pipeline flow ── */
    .de-pipeline {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 0;
        margin: 0.5rem 0 1.5rem;
    }
    .de-pipe-step {
        background: #FFFFFF;
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 1.2rem 1.4rem;
        width: 280px;
        text-align: center;
        font-size: 0.9rem;
        font-weight: 600;
        color: var(--accent);
    }
    .de-pipe-step.highlight {
        background: #F0F3FF;
        border-color: #2D336B;
    }
    .de-pipe-arrow-v {
        font-size: 1rem;
        color: var(--accent2);
        line-height: 1.8;
    }
    .de-pipe-fork {
        display: flex;
        gap: 1rem;
        align-items: flex-start;
        justify-content: center;
        margin-top: 0;
    }
    .de-pipe-fork-card {
        flex: 1;
        max-width: 260px;
        background: #FFFFFF;
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 1.2rem 1rem;
        text-align: center;
    }
    .de-pipe-fork-label {
        font-family: var(--mono);
        font-size: 0.65rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--muted);
        margin-bottom: 0.3rem;
    }
    .de-pipe-fork-name {
        font-size: 0.9rem;
        font-weight: 600;
        color: var(--accent);
    }
    .de-pipe-fork-strats {
        font-family: var(--mono);
        font-size: 0.72rem;
        color: var(--accent2);
        margin-top: 0.2rem;
    }

    /* ── Raw field grid ── */
    .de-field-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.5rem 2rem;
        padding: 1.2rem 1.4rem;
        background: #FFFFFF;
        border: 1px solid var(--border);
        border-radius: 8px;
    }
    .de-field-item {
        display: flex;
        flex-direction: column;
        gap: 0.15rem;
        padding: 0.5rem 0;
        border-bottom: 1px solid #F0F3FF;
    }
    .de-field-key {
        font-family: var(--mono);
        font-size: 0.65rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--muted);
    }
    .de-field-val {
        font-size: 0.95rem;
        color: var(--accent);
        word-break: break-word;
    }

    /* ── Representation cards ── */
    .de-repr-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1.2rem;
        margin-top: 1rem;
    }
    .de-repr-card {
        background: #FFFFFF;
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 1.5rem 1.6rem;
        display: flex;
        flex-direction: column;
        gap: 1rem;
    }
    .de-repr-track-label {
        font-family: var(--mono);
        font-size: 0.65rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: var(--muted);
        margin-bottom: 0.2rem;
    }
    .de-repr-track-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: var(--accent);
        margin-bottom: 0.3rem;
    }
    .de-repr-badges {
        display: flex;
        gap: 0.35rem;
        flex-wrap: wrap;
        margin-bottom: 0.2rem;
    }
    .de-repr-badge {
        font-family: var(--mono);
        font-size: 0.75rem;
        font-weight: 600;
        background: #2D336B;
        color: #FFF2F2;
        padding: 0.18rem 0.5rem;
        border-radius: 4px;
    }
    .de-repr-badge.light {
        background: #A9B5DF;
        color: #2D336B;
    }
    .de-repr-desc {
        font-size: 0.93rem;
        color: #4A5490;
        line-height: 1.55;
        margin-bottom: 0.3rem;
    }
    .de-repr-block {
        background: #FFF2F2;
        border: 1px solid var(--border);
        border-radius: 4px;
        padding: 1rem 1.1rem;
        font-family: var(--mono);
        font-size: 0.9rem;
        color: #2D336B;
        white-space: pre-wrap;
        word-break: break-word;
        line-height: 1.7;
    }
    .de-repr-traits {
        display: flex;
        flex-direction: column;
        gap: 0.3rem;
    }
    .de-repr-trait {
        font-size: 0.9rem;
        color: #4A5490;
    }
    .de-repr-trait::before {
        content: "\u2713  ";
        font-weight: 700;
        color: #2D336B;
    }

    /* ── Charts ── */
    .de-chart-title {
        font-size: 1rem;
        font-weight: 600;
        color: var(--accent);
        margin-bottom: 0.5rem;
        font-family: var(--mono);
        letter-spacing: 0.04em;
    }
    </style>
    """, unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════
    # LOAD DATA
    # ════════════════════════════════════════════════════════
    with st.spinner("Loading dataset sample\u2026"):
        df = load_cleaned_df(nrows=5000)

    # ════════════════════════════════════════════════════════
    # SECTION 1 — HERO
    # ════════════════════════════════════════════════════════
    if df is not None:
        n_records   = len(df)
        n_buildings = df["BuildingID"].nunique()
        n_types     = df["Type"].nunique()
        n_equip     = df["equipment"].nunique()
    else:
        n_records = n_buildings = n_types = n_equip = 0

    st.markdown(f"""
    <div class="de-hero">
        <div class="de-eyebrow">FM-RAG Research Platform \u00b7 Dataset Explorer</div>
        <div class="de-title">Dataset Explorer</div>
        <div class="de-subtitle">
            Explore Facility Management work orders, metadata distributions,
            and retrieval representations.
        </div>
        <div class="de-stat-grid">
            <div class="de-stat-card">
                <div class="de-stat-label">Loaded Records</div>
                <div class="de-stat-value">{n_records:,}</div>
                <div class="de-stat-sub">From preprocessed_clean.csv</div>
            </div>
            <div class="de-stat-card">
                <div class="de-stat-label">Unique Buildings</div>
                <div class="de-stat-value">{n_buildings:,}</div>
                <div class="de-stat-sub">Distinct facility identifiers</div>
            </div>
            <div class="de-stat-card">
                <div class="de-stat-label">Facility Types</div>
                <div class="de-stat-value">{n_types:,}</div>
                <div class="de-stat-sub">Building classification categories</div>
            </div>
            <div class="de-stat-card">
                <div class="de-stat-label">Equipment Categories</div>
                <div class="de-stat-value">{n_equip:,}</div>
                <div class="de-stat-sub">Distinct equipment systems</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if df is None:
        st.warning(
            "\u26a0\ufe0f Cleaned dataset not found at `data/preprocessed_clean.csv`. "
            "Run the pipeline first (`python main.py`)."
        )
        return

    # ════════════════════════════════════════════════════════
    # SECTION 2 — DATASET QUALITY SUMMARY
    # ════════════════════════════════════════════════════════
    st.markdown('<hr class="de-sec-div">', unsafe_allow_html=True)
    st.markdown('<div class="de-sec-label">Dataset Quality Summary</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="de-sec-title">Preprocessing & Cleaning Pipeline</div>',
        unsafe_allow_html=True,
    )

    quality_html = '<div class="de-quality-grid">'
    for icon, name, desc in QUALITY_CARDS:
        quality_html += (
            f'<div class="de-quality-card">'
            f'  <div class="de-quality-check">{icon}</div>'
            f'  <div class="de-quality-name">{name}</div>'
            f'  <div class="de-quality-desc">{desc}</div>'
            f'</div>'
        )
    quality_html += "</div>"
    st.markdown(quality_html, unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════
    # SECTION 3 — FILTER PANEL
    # ════════════════════════════════════════════════════════
    st.markdown('<hr class="de-sec-div">', unsafe_allow_html=True)
    st.markdown('<div class="de-filter-title">Filter Work Orders</div>', unsafe_allow_html=True)

    fc1, fc2, fc3 = st.columns(3, gap="large")
    with fc1:
        types = ["All"] + sorted(df["Type"].unique().tolist())
        sel_type = st.selectbox("Facility Type", types, key="de_type")
    with fc2:
        equip = ["All"] + sorted(df["equipment"].unique().tolist())
        sel_equip = st.selectbox("Equipment System", equip, key="de_equip")
    with fc3:
        search_text = st.text_input(
            "Search description", placeholder="e.g. cooling failure", key="de_search"
        )

    # Apply filters
    filtered = df.copy()
    if sel_type != "All":
        filtered = filtered[filtered["Type"] == sel_type]
    if sel_equip != "All":
        filtered = filtered[filtered["equipment"] == sel_equip]
    if search_text:
        mask = filtered["WODescription"].str.contains(search_text, case=False, na=False)
        filtered = filtered[mask]

    st.markdown(
        f'<div class="de-record-count">{len(filtered):,} matching records'
        f' \u00b7 showing first 5,000 loaded</div>',
        unsafe_allow_html=True,
    )

    # ════════════════════════════════════════════════════════
    # SECTION 4 — WORK ORDER BROWSER
    # ════════════════════════════════════════════════════════
    st.markdown('<div class="de-sec-label">Work Order Records</div>', unsafe_allow_html=True)

    display_cols = ["WOID", "BuildingID", "BuildingName", "Type", "equipment", "WODescription"]
    show_cols = [c for c in display_cols if c in filtered.columns]
    table_df  = filtered[show_cols].head(200).reset_index(drop=True)

    # ── Row selection: try modern API, fall back to selectbox ──
    selected_woid = None
    use_new_api   = False

    try:
        sel_event = st.dataframe(
            table_df,
            width='stretch',
            height=600,
            column_config={
                "WODescription": st.column_config.TextColumn("Description", width="large"),
                "WOID":          st.column_config.TextColumn("WOID",        width="small"),
                "BuildingID":    st.column_config.TextColumn("Building ID",  width="small"),
                "BuildingName":  st.column_config.TextColumn("Building",     width="medium"),
                "Type":          st.column_config.TextColumn("Facility Type",width="medium"),
                "equipment":     st.column_config.TextColumn("Equipment",    width="medium"),
            },
            on_select="rerun",
            selection_mode="single-row",
            key="de_table_sel",
        )
        rows = sel_event.selection.rows if sel_event and hasattr(sel_event, "selection") else []
        if rows:
            selected_woid = table_df.iloc[rows[0]]["WOID"]
            use_new_api   = True
    except Exception:
        st.dataframe(
            table_df,
            width='stretch',
            height=600,
            column_config={
                "WODescription": st.column_config.TextColumn("Description", width="large"),
                "WOID":          st.column_config.TextColumn("WOID",        width="small"),
            },
        )

    # ════════════════════════════════════════════════════════
    # SECTION 5 — ROW SELECTION
    # ════════════════════════════════════════════════════════
    st.markdown('<hr class="de-sec-div">', unsafe_allow_html=True)

    woids = filtered["WOID"].tolist()
    if not woids:
        st.info("No records match the current filters.")
        return

    # Default to first record if no row has been clicked yet
    if selected_woid is None:
        selected_woid = woids[0]

    row_data = filtered[filtered["WOID"] == selected_woid].iloc[0].to_dict()

    # ════════════════════════════════════════════════════════
    # SECTION 6 — TRANSFORMATION PIPELINE
    # ════════════════════════════════════════════════════════
    pipe_col, repr_col = st.columns([1, 3], gap="large")

    with pipe_col:
        st.markdown("""
        <div style="margin-bottom:0.8rem;">
            <div class="de-sec-label" style="margin-bottom:0.5rem;">Transformation Pipeline</div>
            <div style="font-size:0.9rem;color:#4A5490;line-height:1.6;margin-bottom:1rem;">
                How a raw work order becomes a retrieval document.
            </div>
        </div>
        <div class="de-pipeline">
            <div class="de-pipe-step highlight">Selected Work Order</div>
            <div class="de-pipe-arrow-v">\u2193</div>
            <div class="de-pipe-step">Raw Record Fields</div>
            <div class="de-pipe-arrow-v">\u2193</div>
            <div class="de-pipe-step">Representation Construction</div>
            <div class="de-pipe-arrow-v">\u2193</div>
            <div class="de-pipe-fork">
                <div class="de-pipe-fork-card">
                    <div class="de-pipe-fork-label">Track 1</div>
                    <div class="de-pipe-fork-name">TEXT Representation</div>
                    <div class="de-pipe-fork-strats">A \u00b7 B \u00b7 B\u2032</div>
                </div>
                <div class="de-pipe-fork-card">
                    <div class="de-pipe-fork-label">Track 2</div>
                    <div class="de-pipe-fork-name">MICE Representation</div>
                    <div class="de-pipe-fork-strats">C</div>
                </div>
            </div>
            <div class="de-pipe-arrow-v">\u2193</div>
            <div class="de-pipe-step">BGE Embedding (768-dim)</div>
            <div class="de-pipe-arrow-v">\u2193</div>
            <div class="de-pipe-step">FAISS Index</div>
        </div>
        """, unsafe_allow_html=True)

    with repr_col:
        # Raw fields
        with st.expander("Raw Record Fields", expanded=False):
            fields      = list(row_data.items())
            mid         = len(fields) // 2
            left_fields = fields[:mid]
            right_fields= fields[mid:]

            field_html = '<div class="de-field-grid">'
            for k, v in fields:
                display_v = str(v)[:200] if v else "\u2014"
                field_html += (
                    f'<div class="de-field-item">'
                    f'  <div class="de-field-key">{k}</div>'
                    f'  <div class="de-field-val">{display_v}</div>'
                    f'</div>'
                )
            field_html += "</div>"
            st.markdown(field_html, unsafe_allow_html=True)

        st.markdown("<div style='height:1.2rem;'></div>", unsafe_allow_html=True)

        # ════════════════════════════════════════════════════
        # SECTION 7 — REPRESENTATION COMPARISON
        # ════════════════════════════════════════════════════
        st.markdown(
            '<div class="de-sec-label" style="margin-bottom:0.8rem;">Representation Comparison</div>',
            unsafe_allow_html=True,
        )

        text_repr = _build_text_repr(row_data)
        mice_repr = _build_mice_repr(row_data)

        st.markdown(f"""
        <div class="de-repr-grid">
            <div class="de-repr-card">
                <div>
                    <div class="de-repr-track-label">Embedding Track 1</div>
                    <div class="de-repr-track-title">TEXT Representation</div>
                    <div class="de-repr-badges">
                        <span class="de-repr-badge">A</span>
                        <span class="de-repr-badge">B</span>
                        <span class="de-repr-badge">B\u2032</span>
                    </div>
                    <div class="de-repr-desc">
                        Compact semantic representation. Pipe-delimited fields
                        encoded for dense retrieval. Metadata applied after search.
                    </div>
                </div>
                <div class="de-repr-block">{text_repr}</div>
                <div class="de-repr-traits">
                    <div class="de-repr-trait">Compact, focused representation</div>
                    <div class="de-repr-trait">Strong semantic signal</div>
                    <div class="de-repr-trait">Metadata filtered post-retrieval</div>
                </div>
            </div>
            <div class="de-repr-card">
                <div>
                    <div class="de-repr-track-label">Embedding Track 2</div>
                    <div class="de-repr-track-title">MICE Representation</div>
                    <div class="de-repr-badges">
                        <span class="de-repr-badge light">C</span>
                    </div>
                    <div class="de-repr-desc">
                        Metadata-Infused Contextual Embedding. All facility
                        context encoded directly into the vector. No post-filtering.
                    </div>
                </div>
                <div class="de-repr-block">{mice_repr}</div>
                <div class="de-repr-traits">
                    <div class="de-repr-trait">Rich contextual metadata</div>
                    <div class="de-repr-trait">Metadata embedded in vector</div>
                    <div class="de-repr-trait">No post-retrieval filtering needed</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════
    # SECTION 8 — DATASET COMPOSITION
    # ════════════════════════════════════════════════════════
    st.markdown('<hr class="de-sec-div">', unsafe_allow_html=True)
    st.markdown('<div class="de-sec-label">Dataset Composition</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="de-sec-title">Distribution by Facility Type & Equipment System</div>',
        unsafe_allow_html=True,
    )

    try:
        import plotly.graph_objects as go
        from charts.plotly_charts import THEME

        dc1, dc2 = st.columns(2, gap="large")

        with dc1:
            st.markdown('<div class="de-chart-title">Top 10 Facility Types</div>',
                        unsafe_allow_html=True)
            type_counts = df["Type"].value_counts().head(10)
            if not type_counts.empty:
                fig = go.Figure(go.Bar(
                    x=type_counts.values,
                    y=type_counts.index,
                    orientation="h",
                    marker_color="#2D336B",
                    marker_line_width=0,
                ))
                fig.update_layout(
                    **THEME,
                    xaxis=dict(gridcolor="#D8DCF0", title="Work Orders"),
                    yaxis=dict(showgrid=False, automargin=True),
                    margin=dict(l=10, r=20, t=10, b=30),
                    height=380,
                    showlegend=False,
                )
                st.plotly_chart(fig, width='stretch')

        with dc2:
            st.markdown('<div class="de-chart-title">Top 10 Equipment Systems</div>',
                        unsafe_allow_html=True)
            eq_counts = df["equipment"].value_counts().head(10)
            if not eq_counts.empty:
                fig2 = go.Figure(go.Bar(
                    x=eq_counts.values,
                    y=eq_counts.index,
                    orientation="h",
                    marker_color="#7886C7",
                    marker_line_width=0,
                ))
                fig2.update_layout(
                    **THEME,
                    xaxis=dict(gridcolor="#D8DCF0", title="Work Orders"),
                    yaxis=dict(showgrid=False, automargin=True),
                    margin=dict(l=10, r=20, t=10, b=30),
                    height=380,
                    showlegend=False,
                )
                st.plotly_chart(fig2, width='stretch')

    except Exception:
        st.info("Charts unavailable \u2014 install plotly to enable distribution visualisations.")