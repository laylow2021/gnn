"""Feature engineering and preprocessing steps."""

from typing import Tuple

import pandas as pd
from sklearn.preprocessing import StandardScaler


def build_features(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Split DataFrame into feature matrix and target vector."""
    if target not in df.columns:
        raise KeyError(f"Target column {target} not found.")
    y = df[target]
    X = df.drop(columns=[target])
    return X, y


def normalize_features(X: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler]:
    """Apply standard scaling to numeric columns; returns transformed data and scaler."""
    scaler = StandardScaler()
    num_cols = X.select_dtypes(include=["number"]).columns
    X_scaled = X.copy()
    X_scaled[num_cols] = scaler.fit_transform(X[num_cols])
    return X_scaled, scaler

