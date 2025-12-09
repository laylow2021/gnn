"""Build transaction-level features and anomaly scores using graph embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from gnn.config import TransactionAnomalyConfig


@dataclass
class TransactionFeatures:
    features: pd.DataFrame
    feature_names: Sequence[str]
    id_column: str


class TransactionFeatureBuilder:
    """Construct transaction features combining raw fields and graph embeddings."""

    def __init__(self, config: TransactionAnomalyConfig) -> None:
        self.config = config

    def build(
        self,
        df: pd.DataFrame,
        embeddings: Dict[str, np.ndarray],
    ) -> TransactionFeatures:
        cfg = self.config
        rows = []
        amount_present = cfg.amount_column in df.columns if cfg.amount_column else False
        direction_present = cfg.direction_column in df.columns if cfg.direction_column else False
        ts_present = cfg.timestamp_column in df.columns if cfg.timestamp_column else False

        for _, row in df.iterrows():
            record: Dict[str, float] = {}
            if amount_present:
                record["amount"] = float(row[cfg.amount_column])
                record["log_amount"] = float(np.log1p(abs(record["amount"])))
            if direction_present:
                record["direction_flag"] = 1.0 if str(row[cfg.direction_column]).upper().startswith("C") else 0.0
            if ts_present:
                ts = pd.to_datetime(row[cfg.timestamp_column])
                record["hour_of_day"] = ts.hour
                record["day_of_week"] = ts.dayofweek
            for col in cfg.features:
                if col in df.columns:
                    try:
                        record[col] = float(row[col])
                    except Exception:
                        record[col] = 0.0

            cp = str(row[cfg.counterparty_account_column])
            emb = embeddings.get(cp)
            if emb is not None:
                for idx, val in enumerate(emb):
                    record[f"emb_{idx}"] = float(val)

            rows.append(record)

        features_df = pd.DataFrame(rows)
        features_df.fillna(0, inplace=True)
        feature_names = list(features_df.columns)
        return TransactionFeatures(features=features_df, feature_names=feature_names, id_column=cfg.counterparty_account_column)


class TransactionAnomalyScorer:
    """Run anomaly scoring on transaction feature vectors."""

    def __init__(self, method: str = "isolation_forest", contamination: float = 0.01, random_state: int = 42) -> None:
        self.method = method
        self.contamination = contamination
        self.random_state = random_state
        self.model = None

    def fit_score(self, features: pd.DataFrame) -> Tuple[pd.Series, object]:
        if self.method != "isolation_forest":
            raise ValueError(f"Unsupported transaction anomaly method: {self.method}")
        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
        )
        scores = self.model.fit_predict(features)
        # IsolationForest decision_function: higher is less anomalous; we invert for risk score
        anomaly_scores = -self.model.decision_function(features)
        return pd.Series(anomaly_scores, name="transaction_anomaly_score"), self.model
