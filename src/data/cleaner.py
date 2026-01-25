import numpy as np
import pandas as pd


def validate_dataset(df: pd.DataFrame, target_col: str):
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")
    if df.shape[1] < 2:
        raise ValueError("Dataset must contain at least one feature column.")


def clean_dataset(df: pd.DataFrame):
    # fills the null row with mean for numneric
    # and mode for categorical.
    df = df.copy()
    for col in df.columns:
        if df[col].dtype.kind in "biufc":
            df[col] = df[col].fillna(df[col].mean())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])
    return df
