""" 
    Data cleaning module Handles: 
    1. filling missing BuildingID, BuildingName, Type 
    2. ensuring schema consistency 
    3. safe CSV writing 
"""

from tqdm import tqdm
import pandas as pd
import numpy as np
import csv

EXPECTED_COLUMNS = [
    "BuildingID",
    "BuildingName",
    "Type",
    "WOID",
    "WODescription",
    "WOStartDate",
    "WOEndDate",
    "equipment"
]

def load_preprocessed_csv(path):

    df = pd.read_csv(
        path,
        dtype=str,
        engine="python",
        on_bad_lines="skip"
    )

    return df.fillna("")

def build_building_reference(df):

    building_reference = (
        df[["BuildingID", "BuildingName", "Type"]]
        .replace("", pd.NA)
        .dropna()
        .drop_duplicates()
    )

    return building_reference

def fill_missing_building_values(df, valid_rows):

    missing_mask = (df["BuildingID"] == "") | (df["BuildingID"].isna())

    missing_indices = df.index[missing_mask]

    total_missing = len(missing_indices)

    tqdm.write(f"Filling {total_missing:,} missing BuildingID values...")

    if total_missing == 0:
        return df

    ref_df = pd.DataFrame(valid_rows)

    sampled = ref_df.sample(n=total_missing, replace=True).reset_index(drop=True)

    for i, idx in enumerate(
        tqdm(missing_indices, desc="Filling building metadata", unit="rows")
    ):

        df.at[idx, "BuildingID"] = sampled.at[i, "BuildingID"]
        df.at[idx, "BuildingName"] = sampled.at[i, "BuildingName"]
        df.at[idx, "Type"] = sampled.at[i, "Type"]

    return df

def enforce_schema(df):

    for col in EXPECTED_COLUMNS:

        if col not in df.columns:
            df[col] = ""

    return df[EXPECTED_COLUMNS]

def save_clean_csv(df, output_path):

    tqdm.write("Saving CSV...")

    df.to_csv(
        output_path,
        index=False,
        quoting=csv.QUOTE_ALL
    )

    tqdm.write(f"Saved → {output_path}")

def clean_preprocessed_dataset(input_path, output_path):

    print("\nCleaning dataset...")

    df = load_preprocessed_csv(input_path)

    valid_rows = build_building_reference(df)

    df = fill_missing_building_values(df, valid_rows)

    df = enforce_schema(df)

    save_clean_csv(df, output_path)

    print("Cleaning complete.")

    return df