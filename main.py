import os
import pandas as pd
from tqdm import tqdm

import preprocessing

def RAG_Pipeline():
    print("DEBUG: main() function called")
    "Main RAG pipeline will be executed here"

    print("=" * 60)
    print("PROGRAM EXECUTION STARTED")
    print("=" * 60)
    
    # STEP 1: Load dataset
    
    print("\n Step 1 - Loading dataset...")
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    dataset_path = os.path.join(data_dir, 'Facility Management Unified Classification Database (FMUCD).csv')

    if not os.path.exists(dataset_path):
        print(f"ERROR: Dataset not found at {dataset_path}")
        print("Please ensure Facility Management Unified Classification Database (FMUCD).csv is in the data directory")
        return

    df = preprocessing.load_dataset(dataset_path)
    print(f"Loaded {len(df)} records from dataset")
    
    # STEP 2: Preprocess rows and save the new preprocessed csv if not done ignore

    if( os.path.exists(os.path.join(data_dir, 'preprocessed.csv'))):
        print("\nPreprocessed dataset already exists. Skipping preprocessing step.")
    else:
        print("\nStep 2 - Preprocessing rows...")

        records = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing"):
            preprocessed = preprocessing.preprocess_row(row)
            records.append(preprocessed)

        print(f"Preprocessed {len(records)} records")

        print("\nSaving preprocessed dataset...")

        processed_df = pd.DataFrame(records)
        
        output_path = os.path.join(os.path.dirname(dataset_path), 'preprocessed.csv')
        processed_df.to_csv(output_path, index=False)

        print("Preprocessed dataset saved to preprocessed.csv")

if __name__ == "__main__":  
    RAG_Pipeline()
