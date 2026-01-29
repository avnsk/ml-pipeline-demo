from data.loader import load_csv
from data.cleaner import validate_dataset, fill_missing
from data.split import train_test_split
from data.split import train_test_split
from features.scaler import fit_transform
from features.encoder import one_hot_encode


class Pipeline:
    def __init__(self, data_path: str, target_col: str):
        self.data_path = data_path
        self.target_col = target_col

    def run(self, test_size=0.2):
        # Load
        df = load_csv(self.data_path)
        validate_dataset(df, self.target_col)

        # Clean
        df = fill_missing(df)
        df = one_hot_encode(df, exclude_cols=[self.target_col])
        print(df.head())

        X = df.drop(self.target_col, axis=1).to_numpy()
        y = df[self.target_col].values.reshape(-1, 1)
        X_scaled, means, stds = fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size
        )
        print("Pipeline run successful!")
        print(f"Train shape: X={X_train.shape}, y={y_train.shape}")
        print(f"Test shape:  X={X_test.shape}, y={y_test.shape}")

        return X_train, X_test, y_train, y_test, means, stds
