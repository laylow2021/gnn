"""Data loading and validation utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from gnn.config.schema import GraphConfig


def load_transactions(path: str | Path) -> pd.DataFrame:
    """Load transactions from CSV or Parquet into a DataFrame."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file extension for {path}")


def validate_graph_columns(df: pd.DataFrame, graph_cfg: GraphConfig) -> None:
    """Ensure the required columns exist for graph construction."""
    required = [graph_cfg.src_column, graph_cfg.dst_column]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for graph building: {missing}")


def split_dataframe(
    df: pd.DataFrame, splits: Dict[str, float], seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame into train/val/test according to provided ratios."""
    train_ratio = splits.get("train", 0.7)
    val_ratio = splits.get("val", splits.get("dev", 0.15))
    test_ratio = splits.get("test", 0.15)
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")

    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_df = shuffled.iloc[:n_train]
    val_df = shuffled.iloc[n_train : n_train + n_val]
    test_df = shuffled.iloc[n_train + n_val :]
    return train_df, val_df, test_df


__all__ = ["load_transactions", "validate_graph_columns", "split_dataframe"]
