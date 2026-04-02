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
    normalizes the text and removes all lowecase, strips the whitespace, removes special characters
    """

    if( text is None or pd.isna(text)):
        return ""
    
    text = str(text).lower().strip()

    #removes many whitespeaces and replaces with one whitespace
    text = re.sub(r'\s+',' ',text)
    #removes special characters but keeps alphanumeric and basic punctuation
    text = re.sub(r'[^\w\s\-.,]', '', text)

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