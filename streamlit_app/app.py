"""
app.py - FM-RAG Visualization & Demonstration Platform

Entry point for the Streamlit application.
Run with: streamlit run app.py

Navigation (updated):
  App Overview
  Dataset Explorer
  Vector Storage
  System & Retrieval Architecture  ← merged from architecture + strategy pages
  Evaluation Dashboard
"""

import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st

st.set_page_config(
    page_title="FM-RAG Research Platform",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stSidebarNav"]           { display: none; }
section[data-testid="stSidebar"]       { padding-top: 0rem !important; }
section[data-testid="stSidebar"] > div { padding-top: 0rem !important; }
[data-testid="stSidebarContent"]       { padding-top: 0rem !important; }
</style>
""", unsafe_allow_html=True)

from assets.styles import inject_global_css
inject_global_css()

from components.sidebar import render_sidebar
page = render_sidebar()

if page == "App Overview":
    from pages.overview import render
elif page == "Dataset Explorer":
    from pages.dataset_explorer import render
elif page == "Vector Storage":
    from pages.vector_storage import render
elif page == "System & Retrieval Architecture":
    from pages.architecture import render
elif page == "Live Query":
    from pages.live_query import render
elif page == "Evaluation Dashboard":
    from pages.evaluation_dashboard import render
else:
    from pages.overview import render

render()