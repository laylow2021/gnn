import pandas as pd

from gnn.config import (
    FeatureDefinition,
    FeatureGroupConfig,
    FeaturesConfig,
    GraphConfig,
    EdgeDerivationConfig,
    FlowEdgeRuleConfig,
    BatchEdgeRuleConfig,
    SimilarityEdgeRuleConfig,
)
from gnn.graph import GraphBuilder


def test_graph_builder_creates_features_from_config():
    df = pd.DataFrame(
        {
            "src": ["a", "b", "a"],
            "dst": ["b", "c", "c"],
            "amount": [1.0, 2.0, 3.0],
            "ts": pd.date_range("2024-01-01", periods=3, freq="D"),
        }
    )
    features_cfg = FeaturesConfig(
        node=FeatureGroupConfig(
            structural=[FeatureDefinition(name="in_degree", method="degree_in")],
            temporal=[FeatureDefinition(name="recency", columns=["ts"], method="recency_days")],
        ),
        edge=FeatureGroupConfig(
            structural=[FeatureDefinition(name="amount", columns=["amount"], method="identity")],
            temporal=[FeatureDefinition(name="age_days", columns=["ts"], method="age_days")],
        ),
    )
    graph_cfg = GraphConfig(
        src_column="src",
        dst_column="dst",
        amount_column="amount",
        timestamp_column="ts",
        metadata_label="unit-test",
    )
    builder = GraphBuilder(graph_cfg, features_cfg)

    artifacts = builder.build(df)

    assert artifacts.data.num_nodes == 3
    assert artifacts.data.edge_attr.shape[1] == 2  # amount + age_days
    assert artifacts.data.x.shape[1] == 2  # in_degree + recency
    assert artifacts.data.metadata["feature_schema"]["edge"] == ["amount", "age_days"]


def test_graph_builder_uses_edge_derivation_when_configured():
    txn_df = pd.DataFrame(
        {
            "fintech_acct_id": ["f1", "f1"],
            "counterparty_acct_id": ["a", "b"],
            "timestamp": pd.to_datetime(["2024-01-01T10:00:00", "2024-01-01T10:30:00"]),
            "amount": [100.0, 100.0],
            "direction": ["CREDIT", "DEBIT"],
        }
    )
    graph_cfg = GraphConfig(
        src_column="src",
        dst_column="dst",
        edge_derivation=EdgeDerivationConfig(
            flow=FlowEdgeRuleConfig(
                enabled=True,
                fintech_account_column="fintech_acct_id",
                counterparty_account_column="counterparty_acct_id",
                timestamp_column="timestamp",
                amount_column="amount",
                direction_column="direction",
                incoming_values=["CREDIT"],
                outgoing_values=["DEBIT"],
            ),
            batch=BatchEdgeRuleConfig(enabled=False),
            similarity=SimilarityEdgeRuleConfig(enabled=False),
        ),
    )
    features_cfg = FeaturesConfig()
    builder = GraphBuilder(graph_cfg, features_cfg)

    artifacts = builder.build(txn_df)

    assert artifacts.data.edge_index.shape[1] == 1
    assert artifacts.data.metadata["feature_schema"]["edge"]  # should have flow features
