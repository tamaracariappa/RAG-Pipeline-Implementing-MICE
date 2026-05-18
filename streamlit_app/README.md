# FM-RAG Visualization Platform

Streamlit-based visualization and demonstration system for the FM-RAG research pipeline.

## Structure

```
streamlit_app/
├── app.py                    ← Entry point
├── requirements_app.txt      ← App-specific deps
├── assets/
│   └── styles.py             ← Global CSS, helper renderers
├── components/
│   └── sidebar.py            ← Navigation sidebar
├── pages/
│   ├── overview.py           ← Landing page / pipeline overview
│   ├── dataset_explorer.py   ← Work order inspector + repr comparison
│   ├── embedding_viz.py      ← PCA scatter of FAISS vector spaces
│   ├── live_query.py         ← Live retrieval across all 4 strategies
│   ├── vector_storage.py     ← How FAISS stores embeddings
│   ├── strategy_explainer.py ← A / B / B′ / C explained visually
│   ├── evaluation_dashboard.py ← Metrics from eval_results.json
│   └── architecture.py       ← Technical deep-dive
├── charts/
│   └── plotly_charts.py      ← Reusable Plotly figure builders
├── loaders/
│   ├── data_loader.py        ← Cached wrappers for CSV, FAISS, eval JSON
│   └── retrieval_runner.py   ← Calls existing retrieval.py strategies
└── utils/                    ← (reserved for future utilities)
```

## Setup

1. Install the main project dependencies first:
   ```bash
   pip install -r requirements.txt
   ```

2. Install Streamlit app dependencies:
   ```bash
   pip install -r streamlit_app/requirements_app.txt
   ```

3. Run the FM-RAG pipeline to build FAISS indexes:
   ```bash
   python main.py
   ```

4. Launch the Streamlit app from the **project root** (not inside `streamlit_app/`):
   ```bash
   streamlit run streamlit_app/app.py
   ```

## Important: Run from Project Root

The app must be launched from the directory that contains `retrieval.py`, `config.py`,
`faiss_store.py`, etc. The `PROJECT_ROOT` is set automatically in `app.py`.

## Graceful Degradation

All pages handle missing data gracefully:

- **FAISS indexes not built** → shows explanatory placeholder, no crash
- **eval_results.json missing** → shows demo/synthetic data with a warning banner
- **Cleaned CSV missing** → shows a "run the pipeline first" message

## Pages

| Page | What it shows |
|------|--------------|
| Overview | Pipeline flow, strategy summary, quick stats |
| Dataset Explorer | Work orders + TEXT vs MICE representation comparison |
| Embedding Visualization | PCA 2D scatter of real FAISS vectors |
| Live Query Testing | Run all 4 strategies, compare results + latency |
| Vector Storage | How FAISS stores and retrieves embeddings |
| Retrieval Strategies | A / B / B′ / C with flow diagrams |
| Evaluation Dashboard | Recall@k, MRR, NDCG, latency from real eval output |
| System Architecture | Module reference, config, checkpointing, future plans |

## Performance Notes

- FAISS vector sampling uses `IndexFlatIP.reconstruct()` per-vector — never loads all 2.5M vectors
- PCA results are cached via `@st.cache_data(ttl=1800)` — recomputed at most every 30 minutes
- Cleaned CSV loads only 5,000 rows for the dataset explorer
- All expensive operations are cached with `@st.cache_resource` or `@st.cache_data`
