"""Feature computation utilities driven by declarative config."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd

from gnn.config.schema import FeatureDefinition, FeatureGroup, GraphConfig


def _ensure_numeric(series: pd.Series) -> pd.Series:
    """Convert categorical data to numeric codes so tensors can ingest them."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    codes, _ = pd.factorize(series.fillna("NA"))
    return pd.Series(codes.astype(float), index=series.index)


class FeatureCalculator:
    """Apply feature definitions to raw tabular data."""

    def __init__(self, graph_cfg: GraphConfig) -> None:
        self.graph_cfg = graph_cfg

    def _account_frame(self, df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
        frames: List[pd.DataFrame] = []
        for col, role in ((self.graph_cfg.src_column, "out"), (self.graph_cfg.dst_column, "in")):
            if col not in df.columns:
                continue
            subset = df[[col, *columns]].copy()
            subset.rename(columns={col: "account_id"}, inplace=True)
            subset["__role"] = role
            frames.append(subset)
        if not frames:
            raise KeyError("Source and destination columns not found in DataFrame.")
        return pd.concat(frames, ignore_index=True)

    def _named_columns(self, definition: FeatureDefinition, base_columns: Sequence[str]) -> List[str]:
        if len(base_columns) == 1:
            return [definition.name]
        return [f"{definition.name}__{col}" for col in base_columns]

    def compute_node_feature(
        self,
        df: pd.DataFrame,
        accounts: Sequence[str],
        definition: FeatureDefinition,
        group: FeatureGroup,
    ) -> pd.DataFrame:
        """Aggregate transaction data into account-level features."""
        accounts_index = pd.Index(accounts, name="account_id")
        method = definition.method.lower()

        if method == "degree_in":
            counts = df[self.graph_cfg.dst_column].value_counts()
            series = counts.reindex(accounts_index, fill_value=0).astype(float)
            return series.to_frame(definition.name)
        if method == "degree_out":
            counts = df[self.graph_cfg.src_column].value_counts()
            series = counts.reindex(accounts_index, fill_value=0).astype(float)
            return series.to_frame(definition.name)
        if method == "degree_total":
            incoming = df[self.graph_cfg.dst_column].value_counts()
            outgoing = df[self.graph_cfg.src_column].value_counts()
            total = incoming.add(outgoing, fill_value=0)
            series = total.reindex(accounts_index, fill_value=0).astype(float)
            return series.to_frame(definition.name)

        account_df = self._account_frame(df, definition.columns or [])
        groupers = account_df.groupby("account_id", observed=True)

        if method in {"mean", "sum", "max", "min", "median", "std"}:
            aggregated = groupers[definition.columns].agg(method)
        elif method in {"count", "size"}:
            aggregated = groupers.size().to_frame(definition.name)
        elif method == "nunique":
            aggregated = groupers[definition.columns].nunique()
        elif method == "most_recent":
            ts_col = definition.columns[0] if definition.columns else self.graph_cfg.timestamp_column
            if ts_col is None or ts_col not in df.columns:
                raise KeyError("timestamp column required for most_recent features")
            sorted_df = account_df.sort_values(ts_col)
            aggregated = (
                sorted_df.groupby("account_id").tail(1).set_index("account_id")[definition.columns]
            )
        elif method in {"recency_days", "age_days"}:
            ts_col = definition.columns[0] if definition.columns else self.graph_cfg.timestamp_column
            if ts_col is None or ts_col not in df.columns:
                raise KeyError("timestamp column required for recency features")
            account_df["__ts"] = pd.to_datetime(account_df[ts_col])
            newest = account_df["__ts"].max()
            delta_days = (newest - account_df["__ts"]).dt.total_seconds() / 86400.0
            aggregated = delta_days.groupby(account_df["account_id"]).mean().to_frame(definition.name)
        elif method == "rolling_sum":
            ts_col = definition.columns[0] if definition.columns else self.graph_cfg.timestamp_column
            value_cols = definition.columns[1:] if len(definition.columns) > 1 else definition.columns
            if ts_col is None or ts_col not in df.columns:
                raise KeyError("timestamp column required for rolling_sum features")
            if not value_cols:
                raise KeyError("value column required for rolling_sum features")
            window_days = float(definition.params.get("window_days", 7))
            account_df["__ts"] = pd.to_datetime(account_df[ts_col])
            account_df = account_df.sort_values(["account_id", "__ts"])
            frames: List[pd.DataFrame] = []
            for account, group_df in account_df.groupby("account_id"):
                rolled = (
                    group_df.set_index("__ts")[value_cols]
                    .rolling(f"{window_days}D")
                    .sum()
                    .tail(1)
                )
                rolled.index = [account]
                frames.append(rolled)
            aggregated = pd.concat(frames, axis=0)
        elif method in {"category_one_hot", "mode_category"}:
            if not definition.columns:
                raise KeyError("column required for category encoding")
            cat_col = definition.columns[0]
            mode = groupers[cat_col].agg(lambda x: x.value_counts().index[0] if not x.empty else None)
            mode = mode.reindex(accounts_index)
            if method == "mode_category":
                mode_filled = mode.fillna("NA")
                codes, uniques = pd.factorize(mode_filled)
                aggregated = pd.Series(codes.astype(float), index=accounts_index).to_frame(definition.name)
            else:
                mode_filled = mode.fillna("NA")
                dummies = pd.get_dummies(mode_filled)
                dummies.index = accounts_index
                aggregated = dummies
        else:
            aggregated = groupers[definition.columns].mean()

        aggregated = aggregated.reindex(accounts_index)
        aggregated = aggregated.fillna(0)
        aggregated.columns = self._named_columns(definition, aggregated.columns)
        aggregated = aggregated.apply(_ensure_numeric, axis=0)
        return aggregated

    def compute_edge_feature(
        self,
        df: pd.DataFrame,
        definition: FeatureDefinition,
        group: FeatureGroup,
    ) -> pd.DataFrame:
        """Apply edge-level feature definition to each transaction/edge."""
        method = definition.method.lower()

        if method in {"identity", "copy"}:
            base = df[definition.columns].copy()
        elif method == "age_days":
            ts_col = definition.columns[0] if definition.columns else self.graph_cfg.timestamp_column
            if ts_col is None or ts_col not in df.columns:
                raise KeyError("timestamp column required for age_days edge features")
            ts = pd.to_datetime(df[ts_col])
            newest = ts.max()
            base = (newest - ts).dt.total_seconds().to_frame(definition.name)
            base[definition.name] = base[definition.name] / 86400.0
        elif method == "normalized_amount":
            amount_col = definition.columns[0] if definition.columns else self.graph_cfg.amount_column
            if amount_col is None or amount_col not in df.columns:
                raise KeyError("amount column required for normalized_amount edge features")
            series = df[amount_col].astype(float)
            max_val = max(series.max(), 1e-9)
            base = (series / max_val).to_frame(definition.name)
        elif method in {"boolean", "flag"}:
            col = definition.columns[0]
            base = df[col].astype(bool).astype(float).to_frame(definition.name)
        elif method == "category_one_hot":
            if not definition.columns:
                raise KeyError("column required for category_one_hot edge feature")
            col = definition.columns[0]
            series = df[col].fillna("NA")
            dummies = pd.get_dummies(series)
            base = dummies
        elif method == "category_code":
            if not definition.columns:
                raise KeyError("column required for category_code edge feature")
            col = definition.columns[0]
            codes, _ = pd.factorize(df[col].fillna("NA"))
            base = pd.Series(codes.astype(float), name=definition.name).to_frame()
        else:
            base = df[definition.columns].copy()

        base = base.fillna(0)
        base.columns = self._named_columns(definition, base.columns)
        for col in base.columns:
            base[col] = _ensure_numeric(base[col])
        return base
