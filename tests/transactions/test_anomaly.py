import numpy as np
import pandas as pd

from gnn.config import TransactionAnomalyConfig
from gnn.transactions import TransactionFeatureBuilder, TransactionAnomalyScorer


def test_transaction_features_include_embeddings_and_raw():
    df = pd.DataFrame(
        {
            "counterparty_acct_id": ["a", "b"],
            "amount": [10.0, 20.0],
            "direction": ["CREDIT", "DEBIT"],
            "ts": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        }
    )
    cfg = TransactionAnomalyConfig(
        enabled=True,
        counterparty_account_column="counterparty_acct_id",
        amount_column="amount",
        timestamp_column="ts",
        direction_column="direction",
        features=[],
    )
    embeddings = {"a": np.array([1.0, 0.5]), "b": np.array([0.2, 0.3])}
    builder = TransactionFeatureBuilder(cfg)

    txn_feats = builder.build(df, embeddings)

    assert "amount" in txn_feats.feature_names
    assert any(col.startswith("emb_") for col in txn_feats.feature_names)
    assert "hour_of_day" in txn_feats.feature_names


def test_transaction_anomaly_scorer_runs():
    features = pd.DataFrame({"amount": [1.0, 2.0, 100.0], "emb_0": [0.1, 0.2, 5.0]})
    scorer = TransactionAnomalyScorer(method="isolation_forest", contamination=0.2, random_state=0)

    scores, model = scorer.fit_score(features)

    assert len(scores) == len(features)
    assert model is not None


def test_transaction_features_handle_missing_optional_columns():
    df = pd.DataFrame(
        {
            "counterparty_acct_id": ["a"],
            "amount": [10.0],
        }
    )
    cfg = TransactionAnomalyConfig(
        enabled=True,
        counterparty_account_column="counterparty_acct_id",
        amount_column="amount",
        timestamp_column="missing_ts",
        direction_column="missing_dir",
        features=["extra_missing"],
    )
    embeddings = {"a": np.array([0.1, 0.2])}
    builder = TransactionFeatureBuilder(cfg)

    txn_feats = builder.build(df, embeddings)

    assert "amount" in txn_feats.feature_names
    assert "hour_of_day" not in txn_feats.feature_names
