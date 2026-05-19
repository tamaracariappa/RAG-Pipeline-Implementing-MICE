"""
pages/dataset_explorer.py - Inspect work orders and their representations.
"""

import streamlit as st
import pandas as pd

from assets.styles import section_header, repr_block
from loaders.data_loader import load_cleaned_df


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


def render():
    st.markdown('<div class="page-title">Dataset Explorer</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">'
        'Inspect raw work orders and compare their embedding representations.'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Load data ────────────────────────────────────────────
    with st.spinner("Loading dataset sample…"):
        df = load_cleaned_df(nrows=5000)

    if df is None:
        st.warning(
            "⚠️ Cleaned dataset not found at `data/preprocessed_clean.csv`. "
            "Run the pipeline first (`python main.py`)."
        )
        return

    # ── Filters ──────────────────────────────────────────────
    section_header("Filters")
    fc1, fc2, fc3 = st.columns(3)

    with fc1:
        types = ["All"] + sorted(df["Type"].unique().tolist())
        sel_type = st.selectbox("Facility Type", types)

    with fc2:
        equip = ["All"] + sorted(df["equipment"].unique().tolist())
        sel_equip = st.selectbox("Equipment System", equip)

    with fc3:
        search_text = st.text_input("Search description", placeholder="e.g. cooling failure")

    # Apply filters
    filtered = df.copy()
    if sel_type != "All":
        filtered = filtered[filtered["Type"] == sel_type]
    if sel_equip != "All":
        filtered = filtered[filtered["equipment"] == sel_equip]
    if search_text:
        mask = filtered["WODescription"].str.contains(
            search_text, case=False, na=False)
        filtered = filtered[mask]

    st.caption(f"{len(filtered):,} matching records (showing first 5,000 loaded)")

    # ── Table view ───────────────────────────────────────────
    section_header("Work Order Records")

    display_cols = ["WOID", "BuildingID", "BuildingName",
                    "Type", "equipment", "WODescription"]
    show_cols = [c for c in display_cols if c in filtered.columns]

    table_df = filtered[show_cols].head(200).reset_index(drop=True)
    st.dataframe(
        table_df,
        use_container_width=True,
        height=300,
        column_config={
            "WODescription": st.column_config.TextColumn(
                "Description", width="large"),
            "WOID": st.column_config.TextColumn("WOID", width="small"),
        },
    )

    # ── Row inspector ────────────────────────────────────────
    section_header("Row Inspector · Representation Comparison")

    woids = filtered["WOID"].tolist()
    if not woids:
        st.info("No records match the current filters.")
        return

    selected_woid = st.selectbox("Select a Work Order", woids[:200],
                                 key="row_selector")
    row_data = filtered[filtered["WOID"] == selected_woid].iloc[0].to_dict()

    # Raw fields
    with st.expander("📋 Raw Fields", expanded=True):
        field_cols = st.columns(2)
        fields = list(row_data.items())
        mid = len(fields) // 2
        for i, (k, v) in enumerate(fields):
            col = field_cols[0] if i < mid else field_cols[1]
            with col:
                st.markdown(
                    f'<div style="margin-bottom:0.4rem;">'
                    f'<span style="font-size:0.68rem;letter-spacing:0.08em;'
                    f'text-transform:uppercase;color:#8890a8;">{k}</span><br/>'
                    f'<span style="font-size:0.84rem;color:#e4e6f0;">'
                    f'{str(v)[:200] or "—"}</span></div>',
                    unsafe_allow_html=True,
                )

    # Side-by-side representation comparison
    st.markdown("<br/>", unsafe_allow_html=True)
    rc1, rc2 = st.columns(2, gap="large")

    with rc1:
        st.markdown("""
        <div style="font-size:0.7rem;letter-spacing:0.1em;text-transform:uppercase;
                    color:#4f8ef7;margin-bottom:0.5rem;font-weight:600;">
            TEXT Representation · Strategies A, B, B′
        </div>
        <div style="font-size:0.78rem;color:#8890a8;margin-bottom:0.6rem;">
            Semantic signal only. Compact pipe-delimited format.
        </div>""", unsafe_allow_html=True)
        repr_block(_build_text_repr(row_data))

        st.markdown("""
        <div style="font-size:0.75rem;color:#8890a8;margin-top:0.5rem;line-height:1.6;">
            This string is encoded by BAAI/bge-base-en-v1.5 into a 768-dimensional vector.
            Metadata (building type, equipment) is NOT part of the vector—it is applied
            as a Python filter after retrieval.
        </div>""", unsafe_allow_html=True)

    with rc2:
        st.markdown("""
        <div style="font-size:0.7rem;letter-spacing:0.1em;text-transform:uppercase;
                    color:#e06c75;margin-bottom:0.5rem;font-weight:600;">
            MICE Representation · Strategy C
        </div>
        <div style="font-size:0.78rem;color:#8890a8;margin-bottom:0.6rem;">
            Metadata-Infused Contextual Embedding. All fields encoded together.
        </div>""", unsafe_allow_html=True)
        repr_block(_build_mice_repr(row_data))

        st.markdown("""
        <div style="font-size:0.75rem;color:#8890a8;margin-top:0.5rem;line-height:1.6;">
            Each field is labelled ("facility type: …") so the model learns stable
            associations between label and value. Metadata is part of the vector itself—
            no post-retrieval filtering needed.
        </div>""", unsafe_allow_html=True)

    # ── Dataset statistics ───────────────────────────────────
    st.markdown("<br/>", unsafe_allow_html=True)
    section_header("Dataset Statistics")

    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.metric("Total Records", f"{len(df):,}")
    with s2:
        st.metric("Unique Buildings", f"{df['BuildingID'].nunique():,}")
    with s3:
        st.metric("Facility Types", f"{df['Type'].nunique():,}")
    with s4:
        st.metric("Equipment Categories", f"{df['equipment'].nunique():,}")

    # Distribution charts
    dc1, dc2 = st.columns(2)
    with dc1:
        type_counts = df["Type"].value_counts().head(10)
        if not type_counts.empty:
            import plotly.graph_objects as go
            from charts.plotly_charts import THEME
            fig = go.Figure(go.Bar(
                x=type_counts.values,
                y=type_counts.index,
                orientation="h",
                marker_color="#4f8ef7",
            ))
            fig.update_layout(
                **THEME,
                title=dict(text="Facility Types", font_size=12),
                xaxis=dict(gridcolor="#2a2d3e"),
                yaxis=dict(showgrid=False),
                margin=dict(l=10, r=20, t=40, b=30),
                height=280,
            )
            st.plotly_chart(fig, use_container_width=True)

    with dc2:
        eq_counts = df["equipment"].value_counts().head(10)
        if not eq_counts.empty:
            import plotly.graph_objects as go
            from charts.plotly_charts import THEME
            fig2 = go.Figure(go.Bar(
                x=eq_counts.values,
                y=eq_counts.index,
                orientation="h",
                marker_color="#38c96e",
            ))
            fig2.update_layout(
                **THEME,
                title=dict(text="Equipment Systems", font_size=12),
                xaxis=dict(gridcolor="#2a2d3e"),
                yaxis=dict(showgrid=False),
                margin=dict(l=10, r=20, t=40, b=30),
                height=280,
            )
            st.plotly_chart(fig2, use_container_width=True)
