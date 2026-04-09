""" 
    Data cleaning module Handles: 
    1. Filling in missing BuildingID, BuildingName, Type 
    2. Ensuring schema consistency 
    3. CSV writing 
"""

from collections import Counter
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

    # NEW STEP
    df = infer_metadata_with_nlp(df)

    valid_rows = build_building_reference(df)

    df = fill_missing_building_values(df, valid_rows)

    df = enforce_schema(df)

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

def extract_candidate_phrases(series):

    patterns = [

        r"-\s*([a-z0-9 ]+)",
        r"([a-z]+)\s*st",
        r"room\s*[a-z0-9]+",
        r"zone\s*[a-z0-9]+",
        r"lab\s*[a-z0-9]+",
        r"[a-z]+\s*hall",
        r"[a-z]+\s*gym",
        r"[a-z]+\s*office"

    ]

    found = []

    for text in series:

        text = normalize_text(text)

        for p in patterns:

            matches = re.findall(p, text)

            found.extend(matches)

    return found

def build_vocabulary(df, min_freq=25):

    phrases = extract_candidate_phrases(df["WODescription"])

    freq = Counter(phrases)

    vocab = {

        phrase:count

        for phrase,count in freq.items()

        if count >= min_freq

    }

    return vocab

TYPE_SEEDS = {

    "research":[
        "lab", "chemical", "experiment"
    ],

    "teaching":[
        "classroom", "lecture"
    ],

    "student experience":[
        "gym", "hall", "union", "locker"
    ],

    "office":[
        "office", "desk"
    ],

    "infrastructure":[
        "hvac", "steam", "fan", "valve", "pump"
    ]
}

def infer_type(desc, vocab):

    desc = normalize_text(desc)

    for t, seeds in TYPE_SEEDS.items():

        for s in seeds:

            if s in desc:
                return t

    for term in vocab:

        if term in desc:

            for t,seeds in TYPE_SEEDS.items():

                if any(seed in term for seed in seeds):
                    return t

    return ""

def infer_building(desc, vocab):

    desc = normalize_text(desc)

    for term in vocab:

        if term in desc:

            return term

    return ""

def infer_metadata_with_nlp(df):

    tqdm.write("Building domain vocabulary...")

    vocab = build_vocabulary(df)

    tqdm.write(f"Discovered {len(vocab)} semantic patterns")

    tqdm.write("Inferring missing metadata...")

    missing_type_mask = df["Type"] == ""

    df.loc[missing_type_mask, "Type"] = (

        df.loc[missing_type_mask, "WODescription"]

        .progress_apply(lambda x: infer_type(x, vocab))

    )


    missing_building_mask = df["BuildingName"] == ""

    df.loc[missing_building_mask, "BuildingName"] = (

        df.loc[missing_building_mask, "WODescription"]

        .progress_apply(lambda x: infer_building(x, vocab))

    )


    df["Type"] = df["Type"].replace("", "UNKNOWN_TYPE")

    df["BuildingName"] = df["BuildingName"].replace("", "UNKNOWN_BUILDING")

    df["WOEndDate"] = df["WOEndDate"].replace("", "ONGOING")

    df.loc[df["equipment"] == "", "equipment"] = df["Type"]

    return df

