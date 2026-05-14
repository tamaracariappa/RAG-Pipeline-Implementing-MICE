import pandas as pd
import re

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

def load_data(path):
    df = pd.read_csv(path, dtype=str)
    return df.fillna("")

# -----------------------------
# 1. Schema Check
# -----------------------------
def check_schema(df):
    print("\n=== SCHEMA CHECK ===")
    missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]

    if not missing_cols:
        print("✔ Schema is correct")
    else:
        print(f"❌ Missing columns: {missing_cols}")

# -----------------------------
# 2. Missing Values
# -----------------------------
def check_missing_values(df):
    print("\n=== MISSING VALUES ===")
    total = len(df)

    for col in EXPECTED_COLUMNS:
        missing = (df[col] == "").sum()
        print(f"{col}: {missing:,} missing ({missing/total:.2%})")

# -----------------------------
# 3. BuildingID Validation
# -----------------------------
def check_building_id(df):
    print("\n=== BUILDING ID QUALITY ===")

    valid = ~df["BuildingID"].str.startswith("UNK")
    invalid_count = (~valid).sum()

    print(f"Invalid BuildingIDs: {invalid_count:,} ({invalid_count/len(df):.2%})")

# -----------------------------
# 4. Text Quality (VERY IMPORTANT)
# -----------------------------
def check_text_quality(df):
    print("\n=== TEXT QUALITY ===")

    desc = df["WODescription"]

    short = (desc.str.len() < 20).sum()
    empty = (desc == "").sum()

    print(f"Empty descriptions: {empty:,}")
    print(f"Too short (<20 chars): {short:,}")

# -----------------------------
# 5. Duplicate Work Orders
# -----------------------------
def check_duplicates(df):
    print("\n=== DUPLICATES ===")

    dup_woid = df["WOID"].duplicated().sum()
    dup_rows = df.duplicated().sum()

    print(f"Duplicate WOIDs: {dup_woid:,}")
    print(f"Duplicate rows: {dup_rows:,}")

# -----------------------------
# 6. Cardinality (Important for Metadata RAG)
# -----------------------------
def check_cardinality(df):
    print("\n=== CARDINALITY ===")

    print(f"Unique Buildings: {df['BuildingID'].nunique()}")
    print(f"Unique Types: {df['Type'].nunique()}")

# -----------------------------
# 7. Embedding Readiness Score
# -----------------------------
def embedding_readiness(df):
    print("\n=== EMBEDDING READINESS ===")

    total = len(df)

    good_desc = df["WODescription"].str.len() > 30
    valid_id = df["BuildingID"] != ""
    has_type = df["Type"] != ""

    good_rows = (good_desc & valid_id & has_type).sum()

    score = good_rows / total

    print(f"Good rows for embedding: {good_rows:,}/{total:,}")
    print(f"Readiness score: {score:.2%}")

    if score > 0.85:
        print("✅ READY for RAG")
    elif score > 0.6:
        print("⚠️ Usable but needs improvement")
    else:
        print("❌ Not ready for embedding")

# -----------------------------
# MAIN
# -----------------------------
def run_quality_check(path):
    print("=" * 60)
    print("DATASET QUALITY REPORT")
    print("=" * 60)

    df = load_data(path)

    check_schema(df)
    check_missing_values(df)
    check_building_id(df)
    check_text_quality(df)
    check_duplicates(df)
    check_cardinality(df)
    embedding_readiness(df)

    print("\n=== DONE ===")

# -----------------------------
if __name__ == "__main__":
    path = "data/preprocessed_clean.csv"  # adjust if needed
    run_quality_check(path)