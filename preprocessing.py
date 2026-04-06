'''
Data preprocessing module
handles the dataset loading, text normalization and row preprocessing
'''

import pandas as pd     #for dataset 
import re               #regular expression for patterns

def load_dataset(path):
    """
    returns a dataframe from the given csv file (path)
    """

    df = pd.read_csv(path, low_memory=False)
    return df

def normalize_text(text):
    """
    Normalize text for RAG dataset while preventing CSV corruption.
    
    Handles:
    - NaN values
    - newline characters
    - quotes
    - commas that break CSV structure
    - unusual punctuation anomalies
    - extra whitespace
    """

    if text is None or pd.isna(text):
        return ""

    text = str(text)

    # remove newline characters that break CSV rows
    text = text.replace("\n", " ").replace("\r", " ")

    # remove quotes (major cause of EOF parsing error)
    text = text.replace('"', '')
    text = text.replace("'", "")

    # replace commas with space (prevents extra columns forming)
    text = text.replace(",", " ")

    # remove other problematic separators sometimes found in datasets
    text = text.replace(";", " ")
    text = text.replace("|", " ")
    text = text.replace("/", " ")

    # convert to lowercase and strip whitespace
    text = text.lower().strip()

    # remove multiple whitespaces
    text = re.sub(r'\s+', ' ', text)

    # keep only safe characters for CSV + RAG search
    # allows: letters, numbers, spaces, dash, period
    text = re.sub(r'[^\w\s\-.]', '', text)

    return text

def preprocess_row(row):
    """
    Preprocess only required columns for RAG dataset.
    Returns cleaned values ready to be saved to CSV.
    """

    preprocessed = {
        "BuildingID": str(row.get("BuildingID", "")),

        "BuildingName": normalize_text(row.get("BuildingName", "")),

        "Type": normalize_text(row.get("Type", "")),

        "WOID": str(row.get("WOID", "")),

        "WODescription": normalize_text(row.get("WODescription", "")),

        "WOStartDate": str(row.get("WOStartDate", "")),

        "WOEndDate": str(row.get("WOEndDate", "")),

        "equipment": normalize_text(row.get("SystemDescription", ""))
    }

    # ensure no None / NaN
    for key in preprocessed:
        if preprocessed[key] is None or pd.isna(preprocessed[key]):
            preprocessed[key] = ""

    return preprocessed