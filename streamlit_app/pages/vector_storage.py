"""
pages/vector_storage.py – Visual explanation of how FAISS stores embeddings.
"""

import numpy as np
import streamlit as st

from assets.styles import section_header, repr_block
from charts.plotly_charts import vector_heatmap
from loaders.data_loader import (
    get_index_stats,
    load_config,
    sample_text_vectors,
    sample_mice_vectors,
)


def render():
    st.markdown('<div class="page-title">Vector Storage</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">'
        'How work orders are encoded as vectors and stored in FAISS flat indexes.'
        '</div>',
        unsafe_allow_html=True,
    )

    stats  = get_index_stats()
    config = load_config()

    # ── Index summary cards ──────────────────────────────────
    section_header("FAISS Index Summary")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("TEXT Index Vectors",
                  f"{stats.get('text_total', 0):,}" or "—")
    with c2:
        st.metric("MICE Index Vectors",
                  f"{stats.get('mice_total', 0):,}" or "—")
    with c3:
        st.metric("Vector Dimension",
                  str(stats.get("dim") or config.get("embedding_dim", 768)))
    with c4:
        st.metric("Index Type", "IndexFlatIP",
                  help="Flat inner-product index (exact search, cosine similarity)")

    # ── Architecture diagram ─────────────────────────────────
    section_header("Ingestion Pipeline")

    st.markdown("""
    <div style="display:flex;gap:0.5rem;align-items:center;
                flex-wrap:wrap;margin-bottom:1rem;">
        {steps}
    </div>
    """.format(steps="".join([
        f"""
        <div style="background:#1a1d27;border:1px solid #2a2d3e;
                    border-radius:4px;padding:0.5rem 0.85rem;
                    font-size:0.75rem;color:#e4e6f0;white-space:nowrap;">
            {s}
        </div>
        {"<div style='color:#4f8ef7;font-size:0.9rem;'>→</div>" if i < 5 else ""}
        """
        for i, s in enumerate([
            "Raw Work Order",
            "Text / MICE Repr",
            "bge-base-en-v1.5",
            "768-dim float32",
            "IndexFlatIP.add()",
            "FAISS on disk",
        ])
    ])), unsafe_allow_html=True)

    left, right = st.columns(2, gap="large")

    with left:
        st.markdown("""
        <div style="background:#1a1d27;border:1px solid #2a2d3e;
                    border-radius:6px;padding:1.1rem;">
            <div style="font-size:0.7rem;letter-spacing:0.1em;
                        text-transform:uppercase;color:#4f8ef7;
                        margin-bottom:0.8rem;">TEXT Index · text.index</div>
            <ul style="font-size:0.8rem;color:#8890a8;
                       padding-left:1.1rem;line-height:2.0;">
                <li>One vector per work order</li>
                <li>Input: BuildingName | Type | WODescription</li>
                <li>Used by strategies A, B, B′</li>
                <li>Metadata applied as Python post-filter</li>
            </ul>
        </div>""", unsafe_allow_html=True)

    with right:
        st.markdown("""
        <div style="background:#1a1d27;border:1px solid #2a2d3e;
                    border-radius:6px;padding:1.1rem;">
            <div style="font-size:0.7rem;letter-spacing:0.1em;
                        text-transform:uppercase;color:#e06c75;
                        margin-bottom:0.8rem;">MICE Index · mice.index</div>
            <ul style="font-size:0.8rem;color:#8890a8;
                       padding-left:1.1rem;line-height:2.0;">
                <li>One vector per work order</li>
                <li>Input: labelled sentence template (all fields)</li>
                <li>Used by strategy C only</li>
                <li>Metadata encoded inside the vector</li>
            </ul>
        </div>""", unsafe_allow_html=True)

    # ── Encoding example ─────────────────────────────────────
    st.markdown("<br/>", unsafe_allow_html=True)
    section_header("Encoding Example")

    ec1, ec2 = st.columns(2, gap="large")
    with ec1:
        st.markdown("""
        <div style="font-size:0.72rem;letter-spacing:0.08em;
                    text-transform:uppercase;color:#8890a8;
                    margin-bottom:0.4rem;">Input text</div>""",
                    unsafe_allow_html=True)
        repr_block(
            "engineering block | research | "
            "hvac cooling failure in laboratory. "
            "system offline since monday."
        )
    with ec2:
        st.markdown("""
        <div style="font-size:0.72rem;letter-spacing:0.08em;
                    text-transform:uppercase;color:#8890a8;
                    margin-bottom:0.4rem;">Output vector (768 dims · truncated)</div>""",
                    unsafe_allow_html=True)
        example_vec = np.array([
            0.0412, -0.1183, 0.0891, 0.2204, -0.0567,
            0.1337, -0.0244, 0.1902, -0.1541, 0.0773,
            0.2011, -0.0990, 0.0334, 0.1750, -0.2203,
            0.0611, 0.1424, -0.0882, 0.0198, 0.1660,
        ])
        repr_block("[ " + "  ".join(f"{v:+.4f}" for v in example_vec) + "  … (×748) ]")

    st.markdown("""
    <div style="font-size:0.78rem;color:#8890a8;margin-top:0.5rem;line-height:1.6;">
        The full 768-dimensional vector is L2-normalised before storage,
        so inner-product search equals cosine similarity.
        A query vector is computed the same way and the top-k closest
        document vectors are returned.
    </div>""", unsafe_allow_html=True)

    # ── Cosine similarity explainer ──────────────────────────
    st.markdown("<br/>", unsafe_allow_html=True)
    section_header("Cosine Similarity")

    st.markdown("""
    <div style="background:#1a1d27;border:1px solid #2a2d3e;
                border-radius:6px;padding:1.1rem;">
        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.9rem;
                    color:#98c4fb;text-align:center;margin-bottom:0.8rem;">
            sim(Q, D) = Q · D  &nbsp;|&nbsp;  (after L2 normalisation)
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:0.8rem;">
            <div style="text-align:center;">
                <div style="font-family:'IBM Plex Mono',monospace;font-size:1.2rem;
                            color:#38c96e;">≈ 1.0</div>
                <div style="font-size:0.74rem;color:#8890a8;margin-top:0.2rem;">
                    Near-identical work orders</div>
            </div>
            <div style="text-align:center;">
                <div style="font-family:'IBM Plex Mono',monospace;font-size:1.2rem;
                            color:#f7a94f;">≈ 0.7</div>
                <div style="font-size:0.74rem;color:#8890a8;margin-top:0.2rem;">
                    Related topic, different context</div>
            </div>
            <div style="text-align:center;">
                <div style="font-family:'IBM Plex Mono',monospace;font-size:1.2rem;
                            color:#e06c75;">≈ 0.3</div>
                <div style="font-size:0.74rem;color:#8890a8;margin-top:0.2rem;">
                    Unrelated work orders</div>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    # ── Live vector sample heatmap ───────────────────────────
    st.markdown("<br/>", unsafe_allow_html=True)
    section_header("Live Sample Vectors")

    track = st.radio("Index", ["TEXT", "MICE"], horizontal=True)
    n_vis = st.slider("Rows to display", 5, 30, 15)

    with st.spinner("Sampling vectors…"):
        vecs, meta = (sample_text_vectors(n_vis)
                      if track == "TEXT" else sample_mice_vectors(n_vis))

    if vecs is None:
        st.info("Build FAISS indexes first (`python main.py`).")
        return

    fig = vector_heatmap(
        vecs[:n_vis], n_dims=20,
        title=f"{track} Embedding Vectors · first 20 dims · {n_vis} samples"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Show WOIDs for sampled rows
    if meta:
        import pandas as pd
        sample_tbl = pd.DataFrame([{
            "WOID":        m.get("WOID", ""),
            "Type":        m.get("Type", ""),
            "Equipment":   m.get("equipment", ""),
            "Description": m.get("WODescription", "")[:80],
        } for m in meta[:n_vis]])
        st.dataframe(sample_tbl, use_container_width=True, height=220)

    # ── Checkpointing note ───────────────────────────────────
    st.markdown("<br/>", unsafe_allow_html=True)
    section_header("Checkpointing")
    st.markdown("""
    <div style="background:#1a1d27;border:1px solid #2a2d3e;
                border-radius:6px;padding:1rem;">
        <div style="font-size:0.8rem;color:#8890a8;line-height:1.8;">
            Ingestion is <strong style="color:#e4e6f0;">chunk-based and atomic</strong>.
            Each CSV chunk (~10,000 rows) is fully embedded, inserted, and persisted
            before the chunk is marked complete in <code>ingestion_progress.json</code>.
            A power cut can only interrupt between chunks — no partial or duplicate
            vectors are written. Delete the progress file to restart from scratch.
        </div>
    </div>""", unsafe_allow_html=True)
