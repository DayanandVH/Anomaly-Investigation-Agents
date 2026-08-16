"""
Data loading for the anomaly-investigation agent.

Primary source: Kaggle "Credit Card Fraud Detection" dataset
(https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Download creditcard.csv and place it at data/creditcard.csv
- Columns: Time, V1..V28 (PCA-anonymized features), Amount, Class (0=normal, 1=fraud)

Fallback: if no real CSV is present, we generate a synthetic dataset with the
same shape/column names so the rest of the pipeline can be built and tested
without the real data. Swap in the real CSV any time — nothing else changes.
"""

import os
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

REAL_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "creditcard.csv")


def _generate_synthetic(n_samples: int = 5000, n_features: int = 10, random_state: int = 42) -> pd.DataFrame:
    """Create a synthetic transactions dataset with the same shape/spirit as
    the real Kaggle data: mostly normal transactions, a small fraction of
    anomalies (fraud), and named numeric feature columns.
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=6,
        n_redundant=2,
        n_clusters_per_class=1,
        weights=[0.98, 0.02],   # ~2% anomalies, similar imbalance to real fraud data
        flip_y=0.001,
        class_sep=1.5,
        random_state=random_state,
    )

    columns = [f"V{i+1}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=columns)

    # Give it a realistic "Amount" column instead of a raw feature
    rng = np.random.RandomState(random_state)
    df["Amount"] = np.round(np.abs(rng.normal(loc=60, scale=80, size=n_samples)), 2)
    df["Class"] = y
    return df


def load_data() -> tuple[pd.DataFrame, str]:
    """Load the real dataset if present, otherwise fall back to synthetic data.

    Returns:
        (dataframe, source) where source is "real" or "synthetic"
    """
    if os.path.exists(REAL_DATA_PATH):
        df = pd.read_csv(REAL_DATA_PATH)
        return df, "real"

    df = _generate_synthetic()
    return df, "synthetic"


if __name__ == "__main__":
    df, source = load_data()
    print(f"Loaded {source} data: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Anomaly rate: {df['Class'].mean():.3%}")
    print(df.head())
