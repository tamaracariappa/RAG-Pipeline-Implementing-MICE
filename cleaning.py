""" 
    Data cleaning module Handles: 
    1. Filling in missing BuildingID, BuildingName, Type 
    2. Ensuring schema consistency 
    3. CSV writing 
"""

from tqdm import tqdm
import re
import pandas as pd
import numpy as np
import csv

tqdm.pandas()

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

def group_missing_building_ids(df, num_groups=20):
    """
    Replace missing BuildingIDs with consistent UNK groups.
    Ensures clustering without introducing false relationships.
    """

    tqdm.write("Grouping missing BuildingIDs into UNK buckets...")

    missing_mask = (
        df["BuildingID"].isna() |
        (df["BuildingID"].str.strip() == "") |
        (df["BuildingID"].str.lower().isin(["nan", "none"]))
    )

    total_missing = missing_mask.sum()

    if total_missing == 0:
        tqdm.write("No missing BuildingIDs found.")
        return df

    tqdm.write(f"Total missing BuildingIDs: {total_missing:,}")

    # Create repeating group labels
    group_ids = (
        np.arange(total_missing) % num_groups
    )

    df.loc[missing_mask, "BuildingID"] = (
        pd.Series(group_ids, index=df.index[missing_mask])
        .map(lambda x: f"UNK_{x}")
    )

    # OPTIONAL: keep metadata consistent
    df.loc[missing_mask & (df["BuildingName"] == ""), "BuildingName"] = "UNKNOWN_BUILDING"
    df.loc[missing_mask & (df["Type"] == ""), "Type"] = "UNKNOWN_TYPE"

    return df

def load_preprocessed_csv(path):

    df = pd.read_csv(
        path,
        dtype=str,
        engine="python",
        on_bad_lines="skip"
    )

    return df.fillna("")

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

    # ----------------------------------
    # STEP 1: Standardize BuildingID
    # ----------------------------------
    df["BuildingID"] = df["BuildingID"].astype(str).str.strip()
    df["BuildingID"] = df["BuildingID"].replace(["nan", "None"], "")

    # ----------------------------------
    # STEP 2: FIX WOID (REMOVE EMPTY)
    # ----------------------------------
    before = len(df)
    df["WOID"] = df["WOID"].replace("", np.nan)
    df = df.dropna(subset=["WOID"])
    print(f"Removed {before - len(df):,} rows with missing WOID")

    # ----------------------------------
    # STEP 3: REMOVE DUPLICATE WOIDs
    # ----------------------------------
    before = len(df)
    df = df.drop_duplicates(subset=["WOID"], keep="first")
    print(f"Removed {before - len(df):,} duplicate WOIDs")

    # ----------------------------------
    # STEP 4: REMOVE WEAK DESCRIPTIONS
    # ----------------------------------
    before = len(df)
    df = df[df["WODescription"].str.len() >= 20]
    print(f"Removed {before - len(df):,} short descriptions")

    # ----------------------------------
    # STEP 5: GROUP MISSING BUILDING IDs
    # ----------------------------------
    df = group_missing_building_ids(df, num_groups=20)

    # ----------------------------------
    # STEP 6: LIGHT NORMALIZATION
    # ----------------------------------
    df["BuildingName"] = df["BuildingName"].replace("", "UNKNOWN_BUILDING")
    df["Type"] = df["Type"].replace("", "UNKNOWN_TYPE")
    df["WOEndDate"] = df["WOEndDate"].replace("", "ONGOING")

    # ----------------------------------
    # STEP 7: SCHEMA ENFORCEMENT
    # ----------------------------------
    df = enforce_schema(df)

    # ----------------------------------
    # STEP 8: SAVE
    # ----------------------------------
    save_clean_csv(df, output_path)

    print("Cleaning complete.")

    return df

def normalize_text(text):

    if not text:
        return ""

    text = text.lower()

    text = re.sub(r"[^a-z0-9\s\-]", " ", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text
