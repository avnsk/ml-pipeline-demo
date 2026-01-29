import numpy as np
import pandas as pd


def log_transform(df: pd.DataFrame, columns):
    df = df.copy()
    for col in columns:
        df[col] = np.log1p(df[col])  # log(1 + x) for safety
    return df


def square_transform(df: pd.DataFrame, columns):
    df = df.copy()
    for col in columns:
        df[col] = df[col] ** 2
    return df
