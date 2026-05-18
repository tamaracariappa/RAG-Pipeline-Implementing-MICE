"""
components/sidebar.py – Navigation sidebar for FM-RAG platform.
"""

import streamlit as st


PAGES = [
    ("🏠", "Overview",               "Pipeline walkthrough"),
    ("📂", "Dataset Explorer",        "Inspect work orders & representations"),
    ("🔬", "Embedding Visualization", "PCA scatter of vector space"),
    ("⚡", "Live Query Testing",      "Run real retrieval across strategies"),
    ("🗄️", "Vector Storage",          "How FAISS stores embeddings"),
    ("🔀", "Retrieval Strategies",    "A / B / B′ / C explained"),
    ("📊", "Evaluation Dashboard",    "Recall, MRR, latency metrics"),
    ("🏗️", "System Architecture",     "Technical deep-dive"),
]


def render_sidebar() -> str:
    with st.sidebar:
        st.markdown("""
        <div style="padding:0.6rem 0 1.4rem;">
            <div style="font-family:'IBM Plex Mono',monospace;font-size:0.7rem;
                        letter-spacing:0.12em;text-transform:uppercase;
                        color:#4f8ef7;margin-bottom:0.3rem;">FM-RAG</div>
            <div style="font-size:1.0rem;font-weight:600;color:#e4e6f0;
                        line-height:1.3;">Research<br>Visualization Platform</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(
            '<div style="font-size:0.68rem;letter-spacing:0.08em;'
            'text-transform:uppercase;color:#8890a8;'
            'margin-bottom:0.5rem;">Navigation</div>',
            unsafe_allow_html=True,
        )

        # Track selection in session state
        if "active_page" not in st.session_state:
            st.session_state.active_page = "Overview"

        for icon, name, hint in PAGES:
            active = st.session_state.active_page == name
            btn_style = (
                "background:#1e2235;border:1px solid #4f8ef7;color:#4f8ef7;"
                if active else
                "background:transparent;border:1px solid transparent;color:#8890a8;"
            )
            if st.button(
                f"{icon}  {name}",
                key=f"nav_{name}",
                help=hint,
                use_container_width=True,
            ):
                st.session_state.active_page = name
                st.rerun()

        # System status footer
        st.markdown("---")
        st.markdown("""
        <div style="font-size:0.68rem;color:#8890a8;line-height:1.8;">
            <div><span style="color:#38c96e;">●</span> FAISS indexes</div>
            <div><span style="color:#38c96e;">●</span> BGE-base-en-v1.5</div>
            <div><span style="color:#38c96e;">●</span> Retrieval engine</div>
        </div>
        """, unsafe_allow_html=True)

    return st.session_state.active_page
