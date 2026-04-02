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

    df = pd.read_csv("D:\Facility Management Unified Classification Database (FMUCD).csv")
    return df

