"""
embedding_builder.py - Text representation factories for the two embedding tracks.

TEXT representation  (Strategies A, B, B')
  Semantic fields only: BuildingName | Type | WODescription
  Compact, high-signal text for natural-language queries.

MICE representation  (Strategy C)
  Metadata-Infused Contextual Embedding.
  Uses a consistent, labelled sentence template so the model can learn
  stable associations between field labels and their values.

  Design rationale for the template format:
    - Lowercase "field: value." sentences match BGE's MSMARCO training distribution.
    - Explicit field labels ("facility type:", "equipment system:") make each
      metadata dimension independently decodable during attention.
    - Field ordering: categorical → temporal → free-text puts high-selectivity
      fields early in the token sequence where positional attention is strongest.
    - No pipe separators (avoid token fragmentation on rare Unicode chars).
"""

from __future__ import annotations

from typing import Iterator

import pandas as pd

from config import CLEANED_PATH, CSV_CHUNK_SIZE


# ─────────────────────────────────────────────────────────────
# Representation builders
# ─────────────────────────────────────────────────────────────

def build_text_representation(row: pd.Series) -> str:
    """
    TEXT representation: BuildingName | Type | WODescription.
    Used by Strategies A, B, B'.
    """
    parts = [
        str(row.get("BuildingName",  "")).strip(),
        str(row.get("Type",          "")).strip(),
        str(row.get("WODescription", "")).strip(),
    ]
    return " | ".join(p for p in parts if p)


def build_mice_representation(row: pd.Series) -> str:
    """
    MICE representation: structured sentence encoding all preprocessed columns.

    Template (all lowercase for BGE alignment):
      "building id: {id}. building name: {name}. facility type: {type}.
       equipment system: {equipment}. work period: {start} to {end}.
       work order description: {description}."

    Empty / placeholder values are substituted with 'unknown' so every
    document has the same number of tokens in each field position,
    preserving positional consistency across the corpus.
    """
    def _val(key: str, fallbacks: tuple = ()) -> str:
        v = str(row.get(key, "")).strip().lower()
        if not v or v in ("nan", "none", "unknown_building", "unknown_type"):
            return "unknown"
        for f in fallbacks:
            if v == f.lower():
                return "unknown"
        return v

    building_id   = _val("BuildingID")
    building_name = _val("BuildingName")
    btype         = _val("Type")
    equipment     = _val("equipment")
    start         = _val("WOStartDate")
    end           = _val("WOEndDate")
    desc          = _val("WODescription")

    return (
        f"building id: {building_id}. "
        f"building name: {building_name}. "
        f"facility type: {btype}. "
        f"equipment system: {equipment}. "
        f"work period: {start} to {end}. "
        f"work order description: {desc}."
    )


# ─────────────────────────────────────────────────────────────
# Chunked CSV utilities
# ─────────────────────────────────────────────────────────────

def iter_cleaned_chunks(path: str = CLEANED_PATH) -> Iterator[pd.DataFrame]:
    """
    Yield successive DataFrames from the cleaned CSV.
    Only CSV_CHUNK_SIZE rows are resident in memory at any time.
    """
    for chunk in pd.read_csv(path, dtype=str, chunksize=CSV_CHUNK_SIZE):
        yield chunk.fillna("")


def add_text_columns(chunk: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of *chunk* with 'text_repr' and 'mice_repr' columns.
    Both are ready to pass directly to embedder.embed_texts().
    """
    chunk = chunk.copy()
    chunk["text_repr"] = chunk.apply(build_text_representation, axis=1)
    chunk["mice_repr"] = chunk.apply(build_mice_representation,  axis=1)
    return chunk
