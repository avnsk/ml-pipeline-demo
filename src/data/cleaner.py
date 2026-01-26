import numpy as np
import pandas as pd


def validate_dataset(df: pd.DataFrame, target_col: str):
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")
    if df.shape[1] < 2:
        raise ValueError("Dataset must contain at least one feature column.")


def fill_missing(df: pd.DataFrame, numeric_strategy="mean"):
    df = df.copy()
    for col in df.columns:
        if df[col].dtype.kind in "biufc":
            if numeric_strategy == "mean":
                df[col] = df[col].fillna(df[col].mean())
            elif numeric_strategy == "median":
                df[col] = df[col].fillna(df[col].median())
            elif numeric_strategy == "zero":
                df[col] = df[col].fillna(0)
            else:
                raise ValueError("Unknown numeric missing value strategy")
        else:
            # categorical → mode
            df[col] = df[col].fillna(df[col].mode()[0])
    return df
