# RAG-Pipeline-Implementing-MICE

## Project Overview

This repository contains the initial data preparation and cleaning pipeline for a Retrieval-Augmented Generation (RAG) comparison experiment.

The main objective is to compare two RAG approaches:

- RAG trained with structured metadata injection
- RAG trained without metadata

The goal is to demonstrate in which cases metadata-enhanced retrieval improves response quality and when a metadata-free baseline may still be sufficient.

## Current Status

### Implemented

- `main.py`
  - verifies the dataset exists
  - loads raw CSV data
  - triggers preprocessing and cleaning
  - saves intermediate outputs
- `preprocessing.py`
  - loads the raw dataset
  - normalizes text fields
  - prepares the following schema:
    - `BuildingID`
    - `BuildingName`
    - `Type`
    - `WOID`
    - `WODescription`
    - `WOStartDate`
    - `WOEndDate`
    - `equipment`
- `cleaning.py`
  - loads the preprocessed CSV
  - infers missing metadata using simple NLP pattern matching
  - fills missing BuildingID/BuildingName/Type values
  - enforces a fixed output schema
  - writes cleaned output as CSV

### Progress Summary

- Data ingestion and preparation pipeline is working
- Text normalization safeguards against CSV corruption and inconsistent values
- Metadata inference is implemented to support later comparison of metadata-rich vs metadata-free RAG
- Output files produced by the pipeline:
  - `data/preprocessed.csv`
  - `data/preprocessed_clean.csv`

### Pending Work

- implement vector index creation (Milvus, FAISS, or similar)
- add RAG retrieval and generation pipeline
- define the exact metadata injection strategy for comparison
- evaluate retrieval quality and generation accuracy
- run experiments that compare:
  - RAG with metadata-enhanced vectors
  - RAG without metadata

## How to Run

1. Place the dataset under `data/Facility Management Unified Classification Database (FMUCD).csv`
2. Run:

```bash
python main.py
```

3. If `data/preprocessed.csv` or `data/preprocessed_clean.csv` already exist, the pipeline will skip those steps.

## Dataset

This project currently uses the following dataset source:

- Dataset link: https://data.mendeley.com/datasets/cb8d2nsjss/1

> Update the dataset link above if a new dataset version or alternative source is adopted.

## Next Steps for the RAG Comparison Study

1. build the vector store and retrieval layer
2. create two dataset versions:
   - one containing metadata fields for retrieval
   - one using text-only documents without metadata
3. run retrieval experiments for both versions
4. measure performance with metrics such as:
   - retrieval precision / recall
   - answer relevance
   - query latency
5. document cases where metadata helps and cases where it does not

## Notes

This README is intentionally structured for the next phase of the project: comparing RAG with and without metadata. Update the dataset link and experiment details as the project progresses.
