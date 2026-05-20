"""
app.py - FM-RAG Visualization & Demonstration Platform

Entry point for the Streamlit application.
Run with: streamlit run app.py
"""

import sys
import os

# ── Path setup ──────────────────────────────────────────────
# Allow importing the existing project modules from the parent directory.
# Adjust PROJECT_ROOT to point to the folder that contains retrieval.py,
# faiss_store.py, config.py, etc.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st

# ── Page config (must be first Streamlit call) ───────────────
st.set_page_config(
    page_title="FM-RAG Research Platform",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Hide Streamlit default multipage navigation
st.markdown("""
<style>

[data-testid="stSidebarNav"] {
    display: none;
}

section[data-testid="stSidebar"] {
    padding-top: 0rem !important;
}

section[data-testid="stSidebar"] > div {
    padding-top: 0rem !important;
}

[data-testid="stSidebarContent"] {
    padding-top: 0rem !important;
}

</style>
""", unsafe_allow_html=True)

# ── Inject global CSS ────────────────────────────────────────
from assets.styles import inject_global_css
inject_global_css()

# ── Sidebar navigation ───────────────────────────────────────
from components.sidebar import render_sidebar
page = render_sidebar()

# ── Route to pages ───────────────────────────────────────────
if page == "Overview":
    from pages.overview import render
elif page == "Dataset Explorer":
    from pages.dataset_explorer import render
elif page == "Embedding Visualization":
    from pages.embedding_viz import render
elif page == "Live Query Testing":
    from pages.live_query import render
elif page == "Vector Storage":
    from pages.vector_storage import render
elif page == "Retrieval Strategies":
    from pages.strategy_explainer import render
elif page == "Evaluation Dashboard":
    from pages.evaluation_dashboard import render
elif page == "System Architecture":
    from pages.architecture import render
else:
    from pages.overview import render

render()
