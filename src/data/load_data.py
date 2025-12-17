import os
import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split

def load_creditcard_data(
        data_dir: str = "data/raw",
        filename: str = "creditcard.csv",
        test_size: float = 0.2,
        random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:

    """Load the credit card fraud detection dataset and split into train/test/

    return:
        X_train, y_train, X_test, y_test
        """

    filepath = os.path.join(data_dir, filename)
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")

    df = pd.read_csv(filepath)

    # the dataset uses 'Class as target: 1 = fraud, 0 = normal
    if "Class" not in df.columns:
        raise ValueError("Expected column 'Class' in dataset as target label")

    X = df.drop(columns=["Class"])
    y = df["Class"]

    # splitting the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, y_train, X_test, y_test

