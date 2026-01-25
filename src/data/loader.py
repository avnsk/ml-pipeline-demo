import pandas as pd


def load_csv(path: str):
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("Loaded dataset is empty.")
    return df
