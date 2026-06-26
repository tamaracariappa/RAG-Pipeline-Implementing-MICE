"""
pages/vector_storage.py  ·  Vector Storage & Semantic Representation

Embedding Laboratory — select a real FMUCD record from the table,
see exactly how it becomes two embedding vectors.

UX redesign only. All FAISS logic, loaders, and reconstruction
functions are preserved exactly. Presentation-layer changes only.

Sections (final):
  1. Hero
  2. Vector Database at a Glance
  3. How FAISS Stores & Retrieves Vectors
  4. Embedding Inspector  ← main interaction
  5. Cosine Similarity
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from assets.styles import section_header, repr_block
from charts.plotly_charts import THEME
from loaders.data_loader import (
    get_index_stats,
    load_config,
    load_cleaned_df,
    load_text_metadata,
    load_mice_metadata,
)


# ── Representation builders (identical to dataset_explorer.py) ──────────────
def _build_text_repr(row: dict) -> str:
    parts = [
        str(row.get("BuildingName", "")).strip(),
        str(row.get("Type", "")).strip(),
        str(row.get("WODescription", "")).strip(),
    ]
    return " | ".join(p for p in parts if p)


def _build_mice_repr(row: dict) -> str:
    def v(key):
        val = str(row.get(key, "")).strip().lower()
        if not val or val in ("nan", "none", "unknown_building", "unknown_type"):
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


# ── Reconstruct stored vector by WOID ───────────────────────────────────────
def _get_vector_for_woid(woid: str, track: str) -> np.ndarray | None:
    """
    Look up `woid` in the FAISS metadata list, reconstruct its stored vector.
    track: 'text' | 'mice'
    Returns None on any failure (FAISS not loaded, WOID not found, etc.)
    """
    try:
        import faiss_store as fs
        from loaders.data_loader import _load_faiss_indexes
        _load_faiss_indexes()

        meta  = fs.text_metadata if track == "text" else fs.mice_metadata
        index = fs.text_index    if track == "text" else fs.mice_index

        if meta is None or index is None:
            return None

        pos = next(
            (i for i, m in enumerate(meta)
             if str(m.get("WOID", "")).strip() == str(woid).strip()),
            None,
        )
        if pos is None:
            return None

        vec = np.zeros(index.d, dtype=np.float32)
        index.reconstruct(pos, vec)
        return vec
    except Exception:
        return None


# ── Full-vector heatmap (768 dims reshaped 24 × 32) ─────────────────────────
def _vec_heatmap(vec: np.ndarray, title: str) -> go.Figure:
    ncols   = 32
    nrows   = vec.size // ncols
    mat     = vec[: nrows * ncols].reshape(nrows, ncols)
    custom  = np.arange(nrows * ncols).reshape(nrows, ncols)

    fig = go.Figure(go.Heatmap(
        z=mat,
        customdata=custom,
        colorscale="RdBu",
        zmid=0,
        showscale=True,
        colorbar=dict(thickness=12, tickfont=dict(size=9)),
        hovertemplate=(
            "Row %{y} · Col %{x}<br>"
            "Dimension %{customdata}<br>"
            "Value: %{z:.4f}<extra></extra>"
        ),
    ))
    fig.update_layout(
        **THEME,
        title=dict(text=title, font_size=13),
        xaxis=dict(title="Dimension within row (0–31)",
                   showgrid=False, tickfont=dict(size=9)),
        yaxis=dict(title="Row block (32 dims each)",
                   showgrid=False, tickfont=dict(size=9)),
        margin=dict(l=55, r=20, t=46, b=46),
        height=340,
    )
    return fig


# ── Raw vector display — horizontal scrolling monospace block ───────────────
def _raw_vector_df(vec: np.ndarray, accent: str) -> None:
    """
    Display first 32 dimensions as a single horizontally scrollable row.
    Format:  dim_0: +0.0412   dim_1: -0.1183   …   dim_31: +0.0198
    """
    entries = "   ".join(
        f'<span style="color:{accent};">{vec[i]:+.4f}</span>'
        for i in range(32)
    )
    st.markdown(
        f'<div style="background:#FFF2F2;border:1px solid #A9B5DF;'
        f'border-radius:4px;padding:0.75rem 1rem;'
        f'font-family:\'IBM Plex Mono\',monospace;font-size:0.8rem;'
        f'color:#2D336B;white-space:nowrap;overflow-x:auto;line-height:1.8;">'
        f'{entries}'
        f'</div>',
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════
# SECTION 1 · HERO
# ════════════════════════════════════════════════════════════
def _hero() -> None:
    st.markdown("""
    <div style="padding:2rem 0 1.6rem;">
        <div style="font-size:2.6rem;font-weight:700;color:#2D336B;
                    letter-spacing:-0.02em;line-height:1.2;margin-bottom:0.55rem;">
            Vector Storage &amp; Semantic Representation
        </div>
        <div style="font-size:1.15rem;color:#7886C7;
                    max-width:820px;line-height:1.65;">
            How facility management records are transformed into
            high-dimensional vectors and stored for semantic retrieval.
        </div>
    </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# SECTION 2 · VECTOR DATABASE AT A GLANCE
# ════════════════════════════════════════════════════════════
def _scale_cards(stats: dict, config: dict) -> None:
    section_header("Vector Database at a Glance")

    dim    = stats.get("dim") or config.get("embedding_dim", 768)
    text_n = stats.get("text_total", 0)
    mice_n = stats.get("mice_total", 0)
    model  = config.get("embedding_model", "BAAI/bge-base-en-v1.5")

    cards = [
        ("#2D336B", "TEXT Vectors",    f"{text_n:,}" if text_n else "—",
         "Strategies A · B · B′"),
        ("#5C6BC0", "MICE Vectors",    f"{mice_n:,}" if mice_n else "—",
         "Strategy C only"),
        ("#7886C7", "Dimensions",      str(dim),       "per vector"),
        ("#7886C7", "Index Type",      "FlatIP",       "Exact cosine search"),
        ("#2D336B", "Embedding Model", "bge-base-en",  model),
    ]

    for col, (accent, label, value, sub) in zip(st.columns(5, gap="small"), cards):
        with col:
            st.markdown(f"""
            <div style="background:#FFFFFF;border-top:4px solid {accent};
                        border-radius:8px;padding:1.5rem 1.3rem 1.3rem;
                        box-shadow:0 2px 8px rgba(45,51,107,0.07);
                        min-height:130px;box-sizing:border-box;">
                <div style="font-size:0.78rem;text-transform:uppercase;
                            letter-spacing:0.1em;color:#A9B5DF;
                            margin-bottom:0.55rem;">{label}</div>
                <div style="font-family:'IBM Plex Mono',monospace;
                            font-size:2.1rem;font-weight:700;
                            color:#2D336B;line-height:1.1;
                            margin-bottom:0.4rem;">{value}</div>
                <div style="font-size:0.88rem;color:#7886C7;">{sub}</div>
            </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# SECTION 3 · HOW FAISS STORES & RETRIEVES VECTORS
# ════════════════════════════════════════════════════════════
def _faiss_flow() -> None:
    section_header("How FAISS Stores & Retrieves Vectors")

    steps = [
        ("📄", "Document",        "Raw work order\nwith metadata"),
        ("🔤", "Representation",  "TEXT or MICE\nformatted string"),
        ("🧠", "Embedding\nModel","bge-base-en-v1.5\n768-dim encoder"),
        ("📐", "768-D\nVector",   "L2-normalised\nfloat32 array"),
        ("🗄️", "FAISS\nIndex",   "IndexFlatIP\nexact inner-product"),
        ("🎯", "Top-K\nResults",  "Nearest-neighbour\ncosine similarity"),
    ]

    n      = len(steps)
    widths = [1 if i % 2 == 0 else 0.18 for i in range(n * 2 - 1)]
    cols   = st.columns(widths)

    for i, (icon, title, desc) in enumerate(steps):
        with cols[i * 2]:
            border = "#2D336B" if i in (0, n - 1) else "#7886C7"
            bg     = "#F8F9FD" if i in (0, n - 1) else "#FFFFFF"
            st.markdown(f"""
            <div style="background:{bg};border:1px solid #A9B5DF;
                        border-top:4px solid {border};border-radius:10px;
                        padding:1.4rem 0.8rem 1.2rem;text-align:center;">
                <div style="font-size:2rem;margin-bottom:0.5rem;">{icon}</div>
                <div style="font-size:1rem;font-weight:700;color:#2D336B;
                            white-space:pre-line;line-height:1.3;
                            margin-bottom:0.4rem;">{title}</div>
                <div style="font-size:0.82rem;color:#7886C7;
                            white-space:pre-line;line-height:1.5;">{desc}</div>
            </div>""", unsafe_allow_html=True)
        if i < n - 1:
            with cols[i * 2 + 1]:
                st.markdown(
                    '<div style="display:flex;align-items:center;'
                    'justify-content:center;height:100%;">'
                    '<span style="font-size:1.6rem;color:#A9B5DF;">→</span>'
                    '</div>',
                    unsafe_allow_html=True,
                )


# ════════════════════════════════════════════════════════════
# SECTION 4 · EMBEDDING INSPECTOR
# ════════════════════════════════════════════════════════════
def _embedding_inspector() -> None:
    section_header("Embedding Inspector",
                   "select a record · trace it to its vectors")

    # ── load dataset ─────────────────────────────────────────
    with st.spinner("Loading dataset…"):
        df = load_cleaned_df(nrows=5000)

    if df is None:
        st.warning("Dataset not available. Run `python main.py` first.")
        return

    # ── filters ──────────────────────────────────────────────
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        sel_type  = st.selectbox(
            "Facility Type",
            ["All"] + sorted(df["Type"].unique().tolist()),
            key="ei_type",
        )
    with fc2:
        sel_equip = st.selectbox(
            "Equipment System",
            ["All"] + sorted(df["equipment"].unique().tolist()),
            key="ei_equip",
        )
    with fc3:
        search = st.text_input(
            "Search description",
            placeholder="e.g. hvac cooling failure",
            key="ei_search",
        )

    filtered = df.copy()
    if sel_type  != "All":
        filtered = filtered[filtered["Type"]      == sel_type]
    if sel_equip != "All":
        filtered = filtered[filtered["equipment"] == sel_equip]
    if search:
        filtered = filtered[
            filtered["WODescription"].str.contains(search, case=False, na=False)
        ]

    st.caption(f"{len(filtered):,} matching records · click any row to inspect its embedding")

    # ── browsable table with row selection ───────────────────
    display_cols = [c for c in
                    ["WOID", "BuildingName", "Type", "equipment", "WODescription"]
                    if c in filtered.columns]
    table_df = filtered[display_cols].head(50).reset_index(drop=True)

    event = st.dataframe(
        table_df,
        use_container_width=True,
        height=280,
        on_select="rerun",
        selection_mode="single-row",
        column_config={
            "WODescription": st.column_config.TextColumn("Description", width="large"),
            "WOID":          st.column_config.TextColumn("WOID",        width="small"),
            "BuildingName":  st.column_config.TextColumn("Building",    width="medium"),
            "Type":          st.column_config.TextColumn("Type",        width="small"),
            "equipment":     st.column_config.TextColumn("Equipment",   width="small"),
        },
        key="ei_table",
    )

    # ── resolve selected row ──────────────────────────────────
    sel_rows = event.selection.get("rows", []) if event and event.selection else []

    if not sel_rows:
        st.markdown("""
        <div style="margin-top:1.2rem;background:#F8F9FD;border:1px dashed #A9B5DF;
                    border-radius:10px;padding:1.8rem;text-align:center;">
            <div style="font-size:1.05rem;color:#A9B5DF;">
                ↑ Click any row in the table above to inspect its embedding
            </div>
        </div>""", unsafe_allow_html=True)
        return

    row_idx = sel_rows[0]
    row     = table_df.iloc[row_idx].to_dict()

    # look up full record (table_df may have fewer columns)
    woid        = str(row.get("WOID", "")).strip()
    full_rows   = filtered[filtered["WOID"] == woid]
    if full_rows.empty:
        st.warning("Selected record not found in filtered dataset.")
        return
    full_row = full_rows.iloc[0].to_dict()

    # ── WORK ORDER CARD ───────────────────────────────────────
    st.markdown("<div style='margin-top:1.2rem;'></div>", unsafe_allow_html=True)

    woid_val  = full_row.get("WOID", "—")
    bld_val   = full_row.get("BuildingName", "—")
    type_val  = full_row.get("Type", "—")
    equip_val = full_row.get("equipment", "—")
    desc_val  = full_row.get("WODescription", "—")

    def _chip(label, value):
        return (
            '<div style="background:#FFFFFF;border:1px solid #A9B5DF;' +
            'border-radius:6px;padding:0.55rem 0.9rem;">' +
            '<div style="font-size:0.68rem;text-transform:uppercase;' +
            'letter-spacing:0.09em;color:#A9B5DF;margin-bottom:0.25rem;">' + label + '</div>' +
            '<div style="font-size:0.95rem;font-weight:600;color:#2D336B;' +
            'line-height:1.3;">' + str(value) + '</div>' +
            '</div>'
        )

    chips = (
        _chip("Work Order ID",  woid_val) +
        _chip("Building",       bld_val) +
        _chip("Facility Type",  type_val) +
        _chip("Equipment",      equip_val)
    )

    st.markdown(
        '<div style="background:#F0F3FB;border:1px solid #A9B5DF;border-radius:10px;' +
        'padding:1.3rem 1.6rem;margin-bottom:1.4rem;">' +
        '<div style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.1em;' +
        'color:#7886C7;margin-bottom:0.9rem;font-weight:700;">Selected Work Order</div>' +
        '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:0.6rem;' +
        'margin-bottom:1rem;">' + chips + '</div>' +
        '<div style="border-top:1px solid #A9B5DF;padding-top:0.85rem;">' +
        '<div style="font-size:0.68rem;text-transform:uppercase;letter-spacing:0.09em;' +
        'color:#A9B5DF;margin-bottom:0.35rem;">Description</div>' +
        '<div style="font-size:1rem;color:#2D336B;line-height:1.7;">' + desc_val + '</div>' +
        '</div></div>',
        unsafe_allow_html=True,
    )

    # ── REPRESENTATION CONSTRUCTION ───────────────────────────
    st.markdown("""
    <div style="font-size:0.85rem;font-weight:700;color:#2D336B;
                text-transform:uppercase;letter-spacing:0.07em;
                margin-bottom:0.7rem;">Representation Construction</div>""",
                unsafe_allow_html=True)

    text_repr = _build_text_repr(full_row)
    mice_repr = _build_mice_repr(full_row)

    rl, rr = st.columns(2, gap="large")

    with rl:
        st.markdown("""
        <div style="display:flex;align-items:center;gap:0.7rem;margin-bottom:0.6rem;">
            <div style="width:4px;height:1.4rem;background:#2D336B;border-radius:2px;"></div>
            <span style="font-size:1rem;font-weight:700;color:#2D336B;">
                TEXT Representation</span>
            <span style="font-size:0.78rem;color:#A9B5DF;margin-left:0.2rem;">
                → text.index · A B B′</span>
        </div>""", unsafe_allow_html=True)
        repr_block(text_repr)
        st.markdown("""
        <div style="font-size:0.88rem;color:#7886C7;line-height:1.6;margin-top:0.5rem;">
            Plain concatenation. Metadata is present but unlabelled —
            field boundaries are invisible to the encoder.
        </div>""", unsafe_allow_html=True)

    with rr:
        st.markdown("""
        <div style="display:flex;align-items:center;gap:0.7rem;margin-bottom:0.6rem;">
            <div style="width:4px;height:1.4rem;background:#5C6BC0;border-radius:2px;"></div>
            <span style="font-size:1rem;font-weight:700;color:#5C6BC0;">
                MICE Representation</span>
            <span style="font-size:0.78rem;color:#A9B5DF;margin-left:0.2rem;">
                → mice.index · C</span>
        </div>""", unsafe_allow_html=True)

        mice_rendered = mice_repr
        for lbl in ["building id:", "building name:", "facility type:",
                    "equipment system:", "work period:", "work order description:"]:
            mice_rendered = mice_rendered.replace(
                lbl,
                f'<span style="color:#5C6BC0;font-weight:700;">{lbl}</span>',
            )
        st.markdown(
            f'<div class="repr-block" style="white-space:pre-wrap;">'
            f'{mice_rendered}</div>',
            unsafe_allow_html=True,
        )
        st.markdown("""
        <div style="font-size:0.88rem;color:#7886C7;line-height:1.6;margin-top:0.5rem;">
            <span style="color:#5C6BC0;font-weight:600;">Highlighted labels</span>
            are injected metadata fields. The encoder learns stable
            associations between each label and its value.
        </div>""", unsafe_allow_html=True)

    # ── VECTOR RECONSTRUCTION ─────────────────────────────────
    st.markdown("<div style='margin-top:1.4rem;'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.85rem;font-weight:700;color:#2D336B;
                text-transform:uppercase;letter-spacing:0.07em;
                margin-bottom:0.4rem;">Generated Vector Embeddings</div>
    <div style="font-size:0.92rem;color:#7886C7;margin-bottom:0.9rem;
                max-width:720px;line-height:1.6;">
        Real stored embeddings reconstructed from FAISS for this work order.
        All 768 dimensions are shown — reshaped into a 24 × 32 grid.
        Hover any cell for the exact dimension index and value.
    </div>""", unsafe_allow_html=True)

    with st.spinner("Reconstructing vectors from FAISS…"):
        text_vec = _get_vector_for_woid(woid, "text")
        mice_vec = _get_vector_for_woid(woid, "mice")

    faiss_ok = text_vec is not None or mice_vec is not None

    if not faiss_ok:
        st.warning(
            "FAISS indexes are not loaded in this environment — "
            "illustrative vectors are shown below. "
            "On the full-spec machine run `python main.py` first."
        )
        rng      = np.random.default_rng(abs(hash(woid)) % (2 ** 32))
        text_vec = rng.standard_normal(768).astype(np.float32)
        text_vec /= np.linalg.norm(text_vec)
        mice_vec  = rng.standard_normal(768).astype(np.float32) * 1.15
        mice_vec /= np.linalg.norm(mice_vec)
        vec_label = "illustrative"
    else:
        vec_label = "real stored vector"

    # ── side-by-side: heatmap then raw table ──────────────────
    vl, vr = st.columns(2, gap="large")

    with vl:
        # header
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.5rem;">
            <div style="width:4px;height:1.3rem;background:#2D336B;border-radius:2px;"></div>
            <span style="font-size:0.95rem;font-weight:700;color:#2D336B;">
                TEXT Embedding</span>
            <span style="font-size:0.75rem;color:#A9B5DF;">{vec_label}</span>
        </div>""", unsafe_allow_html=True)

        # heatmap
        if text_vec is not None:
            fig_t = _vec_heatmap(text_vec,
                                 "TEXT · 768 dims · 24 rows × 32 cols")
            st.plotly_chart(fig_t, use_container_width=True, key="insp_text_heat")

        # connector label
        st.markdown("""
        <div style="text-align:center;font-size:0.78rem;color:#A9B5DF;
                    margin:0.2rem 0 0.4rem;">↓ &nbsp; underlying values</div>""",
                    unsafe_allow_html=True)

        # raw vector table
        if text_vec is not None:
            st.markdown("""
            <div style="font-size:0.78rem;font-weight:600;color:#2D336B;
                        letter-spacing:0.06em;text-transform:uppercase;
                        margin-bottom:0.3rem;">Raw Vector · first 32 dimensions</div>""",
                        unsafe_allow_html=True)
            _raw_vector_df(text_vec, "#2D336B")

    with vr:
        # header
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.5rem;">
            <div style="width:4px;height:1.3rem;background:#5C6BC0;border-radius:2px;"></div>
            <span style="font-size:0.95rem;font-weight:700;color:#5C6BC0;">
                MICE Embedding</span>
            <span style="font-size:0.75rem;color:#A9B5DF;">{vec_label}</span>
        </div>""", unsafe_allow_html=True)

        # heatmap
        if mice_vec is not None:
            fig_m = _vec_heatmap(mice_vec,
                                 "MICE · 768 dims · 24 rows × 32 cols")
            st.plotly_chart(fig_m, use_container_width=True, key="insp_mice_heat")

        # connector label
        st.markdown("""
        <div style="text-align:center;font-size:0.78rem;color:#A9B5DF;
                    margin:0.2rem 0 0.4rem;">↓ &nbsp; underlying values</div>""",
                    unsafe_allow_html=True)

        # raw vector table
        if mice_vec is not None:
            st.markdown("""
            <div style="font-size:0.78rem;font-weight:600;color:#5C6BC0;
                        letter-spacing:0.06em;text-transform:uppercase;
                        margin-bottom:0.3rem;">Raw Vector · first 32 dimensions</div>""",
                        unsafe_allow_html=True)
            _raw_vector_df(mice_vec, "#5C6BC0")

    # ── key insight ───────────────────────────────────────────
    st.markdown("""
    <div style="margin-top:1.2rem;background:#FFFFFF;border:1px solid #A9B5DF;
                border-left:5px solid #5C6BC0;border-radius:0 8px 8px 0;
                padding:1rem 1.4rem;">
        <span style="font-size:1rem;color:#2D336B;line-height:1.7;">
            <strong>Key insight:</strong> this exact work order produces
            <strong>two different 768-dimensional vectors</strong>.
            Strategy C searches the MICE index directly —
            metadata is part of the similarity calculation itself,
            not a filter applied afterwards.
        </span>
    </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# SECTION 5 · COSINE SIMILARITY
# ════════════════════════════════════════════════════════════
def _cosine_similarity() -> None:
    section_header("Cosine Similarity")

    H = "180px"

    html = (
        '<div style="display:grid;grid-template-columns:2.2fr 1fr 1fr 1fr;gap:1rem;">' +

        # formula card
        '<div style="background:#FFFFFF;border:1px solid #A9B5DF;border-radius:10px;' +
        'padding:1.4rem 1.6rem;height:' + H + ';box-sizing:border-box;' +
        'display:flex;flex-direction:column;justify-content:center;">' +
            '<div style="font-family:IBM Plex Mono,monospace;font-size:1.1rem;' +
            'font-weight:700;color:#2D336B;text-align:center;margin-bottom:0.3rem;">' +
                'sim(Q,&thinsp;D) &nbsp;=&nbsp;' +
                '<span style="display:inline-flex;flex-direction:column;' +
                'text-align:center;line-height:1.3;vertical-align:middle;">' +
                    '<span style="border-bottom:1px solid #2D336B;' +
                    'padding-bottom:0.15rem;">Q &middot; D</span>' +
                    '<span style="padding-top:0.15rem;">' +
                    '&#x2016;Q&#x2016;&thinsp;&#x2016;D&#x2016;</span>' +
                '</span>' +
            '</div>' +
            '<div style="font-size:0.78rem;color:#A9B5DF;text-align:center;' +
            'margin-bottom:0.7rem;">reduces to Q&#x22C5;D when L2-normalised</div>' +
            '<div style="font-size:0.88rem;color:#7886C7;line-height:1.65;' +
            'text-align:center;">All vectors stored in <code>IndexFlatIP</code> ' +
            'are unit-normalised, so inner-product search equals cosine similarity.</div>' +
        '</div>'
    )

    for score, label, border in [
        ("≈ 1.0", "Near-identical<br>work orders",       "#2D336B"),
        ("≈ 0.7", "Related topic,<br>different context", "#7886C7"),
        ("≈ 0.3", "Unrelated<br>work orders",            "#A9B5DF"),
    ]:
        html += (
            '<div style="background:#FFFFFF;border:1px solid #A9B5DF;' +
            'border-top:3px solid ' + border + ';border-radius:10px;' +
            'padding:1.4rem 1rem;height:' + H + ';box-sizing:border-box;' +
            'display:flex;flex-direction:column;' +
            'align-items:center;justify-content:center;gap:0.4rem;text-align:center;">' +
                '<div style="font-family:IBM Plex Mono,monospace;font-size:2.2rem;' +
                'font-weight:700;color:#2D336B;line-height:1;">' + score + '</div>' +
                '<div style="font-size:0.92rem;color:#2D336B;font-weight:500;' +
                'line-height:1.55;">' + label + '</div>' +
                '<div style="font-size:0.72rem;color:#A9B5DF;">cosine similarity</div>' +
            '</div>'
        )

    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# PCA HELPERS  (preserved from original embedding_viz.py)
# ════════════════════════════════════════════════════════════

def _run_pca(vecs: np.ndarray):
    from sklearn.decomposition import PCA
    pca    = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(vecs)
    return coords, pca.explained_variance_ratio_.tolist()


def _build_hover(meta: list) -> list:
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
            f"<i>{desc}…</i>"
        )
    return texts


@st.cache_data(show_spinner=False, ttl=1800)
def _cached_pca(track: str, n_samples: int, color_by: str):
    from loaders.data_loader import sample_text_vectors, sample_mice_vectors
    if track == "Text (Strategies A, B, B′)":
        vecs, meta = sample_text_vectors(n_samples)
    else:
        vecs, meta = sample_mice_vectors(n_samples)
    if vecs is None or meta is None or len(vecs) == 0:
        return None
    coords, expl = _run_pca(vecs)
    labels = [m.get(color_by, "unknown") or "unknown" for m in meta]
    hover  = _build_hover(meta)
    return coords, labels, hover, expl


# ════════════════════════════════════════════════════════════
# SECTION · TEXT EMBEDDING HEATMAPS  (own slider, up to 50 rows)
# ════════════════════════════════════════════════════════════
def _heatmap_section() -> None:
    from charts.plotly_charts import vector_heatmap
    from loaders.data_loader import sample_text_vectors, sample_mice_vectors

    section_header("Sample Embedding Heatmaps", "TEXT vs MICE — raw vector values")

    n = st.slider("Rows to display", 5, 50, 20, step=5, key="heat_n")

    with st.spinner("Sampling vectors…"):
        text_vecs, _ = sample_text_vectors(n)
        mice_vecs, _ = sample_mice_vectors(n)

    if text_vecs is None and mice_vecs is None:
        st.warning("FAISS indexes not available. Run `python main.py` to build them.")
        return

    cl, cr = st.columns(2, gap="large")

    with cl:
        st.markdown(
            '<div style="font-size:0.85rem;font-weight:700;color:#2D336B;'
            'text-transform:uppercase;letter-spacing:0.07em;margin-bottom:0.5rem;">'
            'TEXT Index</div>',
            unsafe_allow_html=True,
        )
        if text_vecs is not None:
            fig = vector_heatmap(
                text_vecs[:n], n_dims=20,
                title=f"TEXT · {n} rows · first 20 of 768 dims",
            )
            fig.update_layout(height=360)
            st.plotly_chart(fig, use_container_width=True, key="heat_text")
            st.caption("Each row = one work order. Each column = one dimension. "
                       "Strategies A · B · B′ search this index.")
        else:
            st.info("TEXT index unavailable.")

    with cr:
        st.markdown(
            '<div style="font-size:0.85rem;font-weight:700;color:#5C6BC0;'
            'text-transform:uppercase;letter-spacing:0.07em;margin-bottom:0.5rem;">'
            'MICE Index</div>',
            unsafe_allow_html=True,
        )
        if mice_vecs is not None:
            fig = vector_heatmap(
                mice_vecs[:n], n_dims=20,
                title=f"MICE · {n} rows · first 20 of 768 dims",
            )
            fig.update_layout(height=360)
            st.plotly_chart(fig, use_container_width=True, key="heat_mice")
            st.caption("Strategy C searches this index. "
                       "Compare colour patterns to see how metadata injection shifts values.")
        else:
            st.info("MICE index unavailable.")


# ════════════════════════════════════════════════════════════
# SECTION · PCA ANALYSIS  (own slider, own colour-by)
# ════════════════════════════════════════════════════════════
def _pca_section() -> None:
    from charts.plotly_charts import pca_scatter
    from loaders.data_loader import sample_text_vectors, sample_mice_vectors

    section_header("PCA Embedding Analysis", "2D projection — TEXT vs MICE")

    st.markdown(
        '<div style="font-size:0.95rem;color:#46538C;margin-bottom:1rem;'
        'max-width:820px;line-height:1.65;">'
        'PCA projects the 768-dimensional embedding space into 2D. '
        'Each point is one work order. Points that cluster together are '
        'semantically similar and will be ranked closely by the retrieval engine.'
        '</div>',
        unsafe_allow_html=True,
    )

    ctrl1, ctrl2 = st.columns([3, 1])
    with ctrl1:
        n = st.slider("Sample size", 100, 2000, 500, step=100, key="pca_n")
    with ctrl2:
        color_by = st.selectbox("Colour by", ["equipment", "Type"], key="pca_color")

    with st.spinner("Computing PCA for both indexes…"):
        t_pca = _cached_pca("Text (Strategies A, B, B′)", n, color_by)
        m_pca = _cached_pca("MICE (Strategy C)",           n, color_by)

    if t_pca is None and m_pca is None:
        st.warning("FAISS indexes not available. Run `python main.py` to build them.")
        return

    cl, cr = st.columns(2, gap="large")

    with cl:
        st.markdown(
            '<div style="font-size:0.85rem;font-weight:700;color:#2D336B;'
            'text-transform:uppercase;letter-spacing:0.07em;margin-bottom:0.5rem;">'
            'TEXT Embedding Space</div>',
            unsafe_allow_html=True,
        )
        if t_pca:
            tc, tl, th, te = t_pca
            fig_t = pca_scatter(tc, tl, th,
                                title=f"TEXT · coloured by {color_by}",
                                explained_var=te)
            fig_t.update_layout(height=440)
            st.plotly_chart(fig_t, use_container_width=True, key="pca_text")
            st.caption(f"Strategies A · B · B′  ·  "
                       f"PC1 {te[0]:.1%} · PC2 {te[1]:.1%} variance explained.")
        else:
            st.info("TEXT index unavailable.")

    with cr:
        st.markdown(
            '<div style="font-size:0.85rem;font-weight:700;color:#5C6BC0;'
            'text-transform:uppercase;letter-spacing:0.07em;margin-bottom:0.5rem;">'
            'MICE Embedding Space</div>',
            unsafe_allow_html=True,
        )
        if m_pca:
            mc, ml, mh, me = m_pca
            fig_m = pca_scatter(mc, ml, mh,
                                title=f"MICE · coloured by {color_by}",
                                explained_var=me)
            fig_m.update_layout(height=440)
            st.plotly_chart(fig_m, use_container_width=True, key="pca_mice")
            st.caption(f"Strategy C  ·  "
                       f"PC1 {me[0]:.1%} · PC2 {me[1]:.1%} variance explained.")
        else:
            st.info("MICE index unavailable.")

    # Shared interpretation panel
    st.markdown(
        '<div style="background:#FFFFFF;border:1px solid #A9B5DF;border-radius:8px;'
        'padding:1rem 1.3rem;margin-top:0.6rem;">'
        '<div style="display:grid;grid-template-columns:1fr 1fr;gap:1.2rem;">'
        '<div>'
        '<div style="font-size:0.75rem;text-transform:uppercase;letter-spacing:0.09em;'
        'color:#2D336B;font-weight:700;margin-bottom:0.4rem;">Reading these charts</div>'
        '<div style="font-size:0.88rem;color:#46538C;line-height:1.7;">'
        'Each point is one work order projected to 2D via PCA. '
        'Nearby points are semantically similar — the retrieval engine ranks them closely. '
        'Tight, well-separated clusters by colour indicate the model has learned '
        'meaningful representations for that metadata category.'
        '</div>'
        '</div>'
        '<div style="border-left:1px solid #A9B5DF;padding-left:1.2rem;">'
        '<div style="font-size:0.75rem;text-transform:uppercase;letter-spacing:0.09em;'
        'color:#2D336B;font-weight:700;margin-bottom:0.4rem;">TEXT vs MICE</div>'
        '<div style="font-size:0.88rem;color:#46538C;line-height:1.7;">'
        'MICE embeddings tend to form tighter, more separable clusters because '
        'labelled metadata fields ("equipment system: hvac") give the model a '
        'stable anchor. TEXT captures richer free-text semantics but metadata '
        'boundaries may overlap across clusters.'
        '</div>'
        '</div>'
        '</div></div>',
        unsafe_allow_html=True,
    )



# ════════════════════════════════════════════════════════════
# MAIN RENDER
# ════════════════════════════════════════════════════════════
def render() -> None:
    stats  = get_index_stats()
    config = load_config()

    _hero()

    st.markdown("<div style='margin-top:0.4rem;'></div>", unsafe_allow_html=True)
    _scale_cards(stats, config)

    st.markdown("<div style='margin-top:0.8rem;'></div>", unsafe_allow_html=True)
    _faiss_flow()

    st.markdown("<div style='margin-top:0.8rem;'></div>", unsafe_allow_html=True)
    _embedding_inspector()

    st.markdown("<div style='margin-top:0.8rem;'></div>", unsafe_allow_html=True)
    _heatmap_section()

    st.markdown("<div style='margin-top:0.8rem;'></div>", unsafe_allow_html=True)
    _pca_section()

    st.markdown("<div style='margin-top:0.8rem;'></div>", unsafe_allow_html=True)
    _cosine_similarity()