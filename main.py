import os
import pandas as pd
from tqdm import tqdm

import preprocessing
import cleaning

def RAG_Pipeline():
    print("DEBUG: main() function called")
    "Main RAG pipeline will be executed here"

    print("=" * 60)
    print("PROGRAM EXECUTION STARTED")
    print("=" * 60)
    
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
        
    # STEP 1: Verify dataset exists

    dataset_path = os.path.join(
        data_dir,
        'Facility Management Unified Classification Database (FMUCD).csv'
    )

    if not os.path.exists(dataset_path):

        print(f"ERROR: Dataset not found at {dataset_path}")
        print("Dataset is required to run this script.")
        print("See README.md for instructions or download directly:")
        print("https://data.mendeley.com/datasets/cb8d2nsjss/1")

        return


    # STEP 2: Load dataset only if preprocessing needed

    preprocessed_path = os.path.join(data_dir, 'preprocessed.csv')

    if os.path.exists(preprocessed_path):

        print("\nPreprocessed dataset already exists. Skipping preprocessing step.")

    else:

        print("\nStep 2 - Loading dataset...")

        df = preprocessing.load_dataset(dataset_path)

        print(f"Loaded {len(df)} records from dataset")

        print("\nPreprocessing rows...")

        records = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing"):

            preprocessed = preprocessing.preprocess_row(row)

            records.append(preprocessed)

        processed_df = pd.DataFrame(records)

        processed_df.to_csv(preprocessed_path, index=False)

        print("Preprocessed dataset saved to preprocessed.csv")

    # STEP 3: Cleaning dataset

    cleaned_path = os.path.join(data_dir, "preprocessed_clean.csv")
    if os.path.exists(cleaned_path):
        print("\nClean dataset already exists. Skipping cleaning step.")
    else:
        print("\nStep 3 - Cleaning dataset...")
        preprocessing_path = os.path.join(data_dir, "preprocessed.csv")
        cleaning.clean_preprocessed_dataset(preprocessing_path, cleaned_path)
        print("Cleaned dataset saved to preprocessed_clean.csv")

if __name__ == "__main__":  
    RAG_Pipeline()
