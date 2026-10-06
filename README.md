# FM-RAG: Metadata-Aware Retrieval for Facility Management Work Orders

A research-oriented Retrieval-Augmented Generation (RAG) pipeline designed to evaluate the impact of structured metadata on dense retrieval performance for large-scale Facility Management (FM) work-order datasets.

This project compares multiple retrieval strategies across semantic-only and metadata-aware retrieval settings using reproducible evaluation protocols, FAISS vector indexing, and large-scale embedding pipelines.

---

# Research Objective

The primary goal of this project is to investigate:

> **Does structured metadata improve retrieval quality in facility management RAG systems?**

The project evaluates whether metadata-aware retrieval strategies outperform standard semantic retrieval when searching noisy, real-world FM work-order descriptions.

---

# Core Research Contributions

This project implements and evaluates:

- Pure semantic dense retrieval
- Metadata-aware filtering strategies
- Metadata-infused contextual embeddings (MICE)
- Query-aware routing
- Deterministic evaluation datasets
- Large-scale FAISS ingestion with atomic checkpointing
- Per-query retrieval analysis and strategy comparison

---

# Dataset

Dataset used:

- **Facility Management Unified Classification Database (FMUCD)**
- Source:
  https://data.mendeley.com/datasets/cb8d2nsjss/1

Dataset size after processing:

- ~2.5 million work orders
- Multiple building types
- Multiple equipment/system categories
- Real-world noisy maintenance records

---

# Project Architecture

```text
Raw Dataset
    ↓
Preprocessing
    ↓
Cleaning & Metadata Standardization
    ↓
Representation Construction
    ├── Text-Only Representation
    └── MICE Representation
    ↓
Embedding Generation (BGE)
    ↓
FAISS Vector Stores
    ├── Text Index
    └── MICE Index
    ↓
Retrieval Strategies
    ├── Strategy A
    ├── Strategy B
    ├── Strategy B'
    └── Strategy C
    ↓
Evaluation & Analysis
```

---

# Retrieval Strategies

| Strategy | Description |
|---|---|
| A | Pure semantic dense retrieval |
| B | Metadata post-filter retrieval |
| B' | Metadata-aware retrieval (experimental pre-filter design) |
| C | Metadata-Infused Contextual Embeddings (MICE) |

---

# Embedding Models

The pipeline is a multi-model experiment. Three models are built and evaluated
independently, each with its own FAISS indexes and evaluation output under
`data/embeddings/<model_key>/`.

| Key | Model | Dim | Encoding procedure |
|---|---|---|---|
| `bge_base` | `BAAI/bge-base-en-v1.5` | 768 | Query-side instruction prefix, documents raw |
| `bge_m3` | `BAAI/bge-m3` | 1024 | No instruction on either side |
| `jina_v3` | `jinaai/jina-embeddings-v3` | 1024 | Task LoRA adapters: `retrieval.query` / `retrieval.passage` |

Each procedure follows that model's official model card. The registry lives in
`config.py`; select one for a pipeline run with the `MICE_EMBEDDING_MODEL`
environment variable (default: `bge_base`).

Shared characteristics:

- L2-normalized vectors
- GPU-compatible batching
- FAISS cosine similarity retrieval

---

# MICE Representation Design

The project introduces a structured metadata-infused embedding representation:

```text
building id: {id}.
building name: {name}.
facility type: {type}.
equipment system: {equipment}.
work period: {start} to {end}.
work order description: {description}.
```

Design goals:

- Preserve semantic structure
- Improve metadata awareness
- Maintain consistent positional encoding
- Reduce token ambiguity
- Align with BGE retrieval behavior

---

# Current Experimental Results

## Evaluation Dataset

| Metric | Value |
|---|---|
| Total queries | 298 |
| Semantic queries | 120 |
| Constrained queries | 178 |
| Relevant set size | Multi-document grouped relevance |
| Evaluation metrics | Recall@K, NDCG@K, MRR, Latency |

---

## Overall Retrieval Performance

| Strategy | MRR | Recall@10 | NDCG@10 | Avg Latency |
|---|---|---|---|---|
| A | 0.4620 | 0.0387 | 0.4386 | 541.8 ms |
| B | 0.6208 | 0.0676 | 0.6766 | 328.1 ms |
| B' | 0.6208 | 0.0676 | 0.6766 | 326.4 ms |
| C | 0.4280 | 0.0343 | 0.4039 | 515.6 ms |

---

# Key Findings

## Metadata Filtering Improves Retrieval

Metadata-aware filtering strategies significantly outperform baseline semantic retrieval on constrained retrieval tasks.

Observed improvements:

- Higher MRR
- Higher Recall@10
- Better ranking quality
- Lower latency

---

## Metadata Embedding Does NOT Always Help

An important finding from the experiments:

> Directly embedding metadata into dense representations (MICE) may dilute semantic retrieval quality.

Strategy C underperformed compared to metadata filtering approaches.

This suggests:

- Metadata filtering may be superior to metadata embedding
- Embedding overload can reduce semantic precision
- Structured metadata should be used selectively

---

# Query Routing System

The project includes a deterministic rule-based query router that:

- Detects equipment/system keywords
- Detects facility types
- Detects building IDs
- Routes queries to optimal retrieval strategies

Routing is fully reproducible and does not rely on external LLM inference.

---

# Evaluation Framework

The evaluation system includes:

- Deterministic query generation
- Multi-document relevance modeling
- Query-type separation
- Latency benchmarking
- Per-query comparisons
- Pairwise strategy analysis
- Metadata impact analysis

Metrics implemented:

- Recall@K
- NDCG@K
- Mean Reciprocal Rank (MRR)
- Latency mean/std/p95

---

# Atomic Checkpoint-Based Ingestion

The ingestion pipeline supports:

- Power-failure recovery
- Duplicate-safe chunk processing
- Persistent ingestion checkpoints
- Large-scale resumable indexing

Features:

- Chunk-level atomic commits
- Progress persistence
- Safe FAISS persistence
- Resume-from-failure support

---

# Repository Structure

```text
.
├── analysis.py
├── cleaning.py
├── config.py
├── dataset_quality_check.py
├── embedder.py
├── embedding_builder.py
├── embedding_quality.py
├── evaluation.py
├── faiss_store.py
├── faiss_test.py
├── main.py
├── mini_pipeline_test.py
├── preprocessing.py
├── query_router.py
├── retrieval.py
├── requirements.txt
├── README.md
├── scripts/
│   └── precompute_pca.py
└── streamlit_app/
    ├── app.py
    ├── test_app.py
    ├── requirements_app.txt
    └── .streamlit/config.toml
```

---

# File Descriptions

| File | Purpose |
|---|---|
| `main.py` | Main orchestration pipeline |
| `preprocessing.py` | Dataset normalization |
| `cleaning.py` | Metadata cleaning and schema enforcement |
| `embedder.py` | BGE embedding model loader |
| `embedding_builder.py` | Representation builders |
| `faiss_store.py` | FAISS indexing and persistence |
| `retrieval.py` | Retrieval strategies A/B/B'/C |
| `query_router.py` | Rule-based query routing |
| `evaluation.py` | Research evaluation framework |
| `analysis.py` | Per-query retrieval analysis |
| `dataset_quality_check.py` | Dataset diagnostics |
| `embedding_quality.py` | Embedding sanity checks |
| `scripts/precompute_pca.py` | Writes the dashboard's offline 2-D projection |
| `streamlit_app/app.py` | Evaluation dashboard (all three models) |

---

# Current Limitations

Current limitations identified during experimentation:

- Dense-only retrieval pipeline
- No lexical retrieval component
- No reranking stage
- No statistical significance testing
- Limited manually annotated relevance judgments
- Experimental B' strategy not yet true pre-filtering

---

# Planned Improvements

## 1. Hybrid Retrieval

Planned integration:

- BM25 retrieval
- Reciprocal Rank Fusion (RRF)
- Dense + lexical hybrid retrieval

Goal:

- Improve Recall@10
- Improve noisy query robustness
- Improve abbreviation handling

---

## 2. True Metadata Pre-Filtering

Future B' implementation will include:

- Metadata-partitioned FAISS indices
- Candidate pruning before ANN search
- Efficient metadata-aware retrieval

---

## 3. Cross-Encoder Reranking

Planned rerankers:

- `bge-reranker-base`
- `ms-marco-MiniLM`

Pipeline:

```text
Dense Retrieval → Top-100 → Cross-Encoder Rerank → Final Top-K
```

---

## 4. Statistical Significance Testing

Future analysis will include:

- Paired t-tests
- Wilcoxon signed-rank tests
- Bootstrap confidence intervals
- Effect size measurements

---

## 5. Human-Annotated Evaluation

Planned additions:

- Expert-labeled FM queries
- Real retrieval task benchmarking
- Ground-truth relevance judgments

---

## 6. Error Taxonomy Analysis

Future work includes categorizing:

- Lexical misses
- Abbreviation failures
- Semantic drift
- Metadata ambiguity
- False positives
- Retrieval collapse cases

---

# Installation

## Clone Repository

```bash
git clone <repository-url>
cd fm-rag
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Pipeline

Place the dataset inside:

```text
data/Facility Management Unified Classification Database (FMUCD).csv
```

Then run:

```bash
python main.py
```

---

# User Interface

A single-page Streamlit dashboard compares all three embedding models side by
side. It is read-only: it opens no FAISS index and runs no retrieval, reading
only the artifacts a pipeline run leaves behind.

```bash
pip install -r streamlit_app/requirements_app.txt
python scripts/precompute_pca.py          # one-off, writes the 2-D projection
cd streamlit_app && streamlit run app.py
```

| Tab | What it shows |
|---|---|
| BGE Base / BGE-M3 / Jina v3 | One tab per model: strategy metric cards and metric comparisons |
| Vector Inspector | Precomputed 2-D projection and vector statistics |
| Query Explorer | Per-query grid, filterable by model, strategy and outcome |

Reads per model from `data/embeddings/<model_key>/`: `eval_results.json`,
`per_query_analysis.csv`, `pca_cache.json`, `vector_sample_50.json`. A missing
file degrades that section to a warning rather than taking the app down, so the
dashboard is usable when only some models have been built.

Details and the self-check: [`streamlit_app/README.md`](streamlit_app/README.md).

---

# Requirements

Core dependencies:

```text
sentence-transformers
torch
faiss-cpu
pandas
numpy
tqdm
scikit-learn
```

---

# Future Research Direction

The long-term objective of this project is to develop:

> A scalable, metadata-aware retrieval framework for enterprise facility management systems capable of handling noisy real-world maintenance corpora.

Potential future extensions:

- LLM-integrated FM assistants
- Temporal-aware retrieval
- Graph-enhanced metadata retrieval
- Adaptive retrieval routing
- Retrieval-agent systems
- Enterprise-scale semantic search

---

# Author

Research project focused on:

- Retrieval-Augmented Generation (RAG)
- Information Retrieval (IR)
- Metadata-Aware Retrieval
- Facility Management AI Systems
- Large-Scale Vector Search

---
