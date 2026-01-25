from data.loader import load_csv
from data.cleaner import validate_dataset, clean_dataset
from data.split import train_test_split


class Pipeline:
    def __init__(self, data_path: str, target_col: str):
        self.data_path = data_path
        self.target_col = target_col

    def run(self):
        # Load
        df = load_csv(self.data_path)
        validate_dataset(df, self.target_col)

        # Clean
        df = clean_dataset(df)

        # Split
        X = df.drop(self.target_col, axis=1).values
        y = df[self.target_col].values.reshape(-1, 1)
        X_train, X_test, y_train, y_test = train_test_split(X, y)

        print("Pipeline run successful!")
        print(f"Train shape: X={X_train.shape}, y={y_train.shape}")
        print(f"Test shape:  X={X_test.shape}, y={y_test.shape}")

        return X_train, X_test, y_train, y_test
