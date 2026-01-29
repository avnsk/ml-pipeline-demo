import pandas as pd


def one_hot_encode(df: pd.DataFrame, exclude_cols=None):

    if exclude_cols is None:
        exclude_cols = []

    categorical_cols = df.select_dtypes(include=["object", "category"]).columns
    categorical_cols = [col for col in categorical_cols if col not in exclude_cols]
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    return df_encoded.astype(
        {col: int for col in df_encoded.select_dtypes("bool").columns}
    )
