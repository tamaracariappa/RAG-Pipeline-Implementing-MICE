import os
import csv
from tqdm import tqdm


import preprocessing

def main():
    "Main RAG pipeline will be executed here"


    print("=" * 60)
    print("PROGRAM EXECUTION STARTED")
    print("=" * 60)

    # ====================================================================
    # STEP 1: Load dataset
    # ====================================================================
    print("\n Step 1 - Loading dataset...")
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    dataset_path = os.path.join(data_dir, 'Facility Management Unified Classification Database (FMUCD).csv')

    if not os.path.exists(dataset_path):
        print(f"ERROR: Dataset not found at {dataset_path}")
        print("Please ensure Facility Management Unified Classification Database (FMUCD).csv is in the data directory")
        return

    df = preprocessing.load_dataset(dataset_path)
    print(f"Loaded {len(df)} records from dataset")

    

if __name__ == "__main":
    main