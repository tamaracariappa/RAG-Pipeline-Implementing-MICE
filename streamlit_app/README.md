# MICE Evaluation Dashboard

Single-page Streamlit dashboard for the FM-RAG retrieval experiment. It compares
all three embedding models — `bge_base`, `bge_m3`, `jina_v3` — across retrieval
strategies A, B and C.

## Structure

```text
streamlit_app/
├── app.py                 Entry point. The whole dashboard.
├── test_app.py            Self-check for app.py (python test_app.py)
├── .streamlit/config.toml Theme
└── requirements_app.txt   App-only dependencies
```

## Setup

```bash
pip install -r requirements.txt                   # from the project root
pip install -r streamlit_app/requirements_app.txt
```

## Run

```bash
cd streamlit_app
streamlit run app.py
```

## Data it reads

Per model key, from `data/embeddings/<model_key>/`:

| File | Contents |
|---|---|
| `eval_results.json` | Aggregate metrics per strategy |
| `per_query_analysis.csv` | One row per (query, strategy) |
| `pca_cache.json` | Static 2-D projection, written offline |
| `vector_sample_50.json` | 50 rows with their raw vectors |

Produce the last two with:

```bash
python scripts/precompute_pca.py              # all three models
python scripts/precompute_pca.py --models bge_m3
```

The dashboard never opens a FAISS index. The projection is precomputed, so the
app starts instantly and stays inside a small memory budget.

## Tabs

| Tab | What it shows |
|---|---|
| BGE Base / BGE-M3 / Jina v3 | One tab per model: strategy metric cards and ECharts metric comparisons |
| Vector Inspector | Precomputed 2-D projection and vector statistics for a chosen model |
| Query Explorer | AgGrid over `per_query_analysis.csv` — filter by model, strategy and outcome |

## Strategies

| Key | Name | Definition |
|---|---|---|
| A | Baseline | Dense search over the text index. No metadata. |
| B | Post-filter | Strategy A over-fetched, then filtered on metadata. |
| C | MICE | Metadata injected into the embedded text, MICE index. |

Definitions mirror `retrieval.py` so the dashboard cannot drift from the code.

## Degradation

Every read is cached and guarded. A missing or malformed file degrades that
section to a warning instead of taking the app down — a model with no
`eval_results.json` shows an "unavailable" notice and the other tabs still render.

## Tests

```bash
python streamlit_app/test_app.py
```

Asserts the model registry, the brand palette, chart option construction and the
missing-file fallbacks against real payloads.
