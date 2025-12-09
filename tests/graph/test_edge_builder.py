import pandas as pd

from gnn.config import EdgeDerivationConfig, FlowEdgeRuleConfig, BatchEdgeRuleConfig, SimilarityEdgeRuleConfig
from gnn.graph.edges import EdgeBuilder


def test_flow_edges_are_constructed_with_time_and_amount_window():
    df = pd.DataFrame(
        {
            "fintech_acct_id": ["f1", "f1", "f1"],
            "counterparty_acct_id": ["a", "b", "c"],
            "timestamp": pd.to_datetime(["2024-01-01T10:00:00", "2024-01-01T10:30:00", "2024-01-01T13:00:00"]),
            "amount": [100.0, 105.0, 200.0],
            "direction": ["CREDIT", "DEBIT", "DEBIT"],
            "reference": ["x", "x", "y"],
        }
    )
    cfg = EdgeDerivationConfig(
        flow=FlowEdgeRuleConfig(enabled=True, time_window_minutes=120, amount_tolerance_pct=0.1, reference_column="reference"),
        batch=BatchEdgeRuleConfig(enabled=False),
        similarity=SimilarityEdgeRuleConfig(enabled=False),
    )
    builder = EdgeBuilder(cfg)

    derived = builder.build(df)

    assert not derived.edges.empty
    assert set(derived.edges[["src", "dst"]].itertuples(index=False, name=None)) == {("a", "b")}
    assert "flow_count" in derived.feature_names
    assert derived.edges["type_flow"].iloc[0] == 1.0


def test_batch_edges_connect_comembers():
    df = pd.DataFrame(
        {
            "batch_id": ["b1", "b1"],
            "counterparty_acct_id": ["a", "b"],
            "amount": [10.0, 12.0],
        }
    )
    cfg = EdgeDerivationConfig(
        flow=FlowEdgeRuleConfig(enabled=False),
        batch=BatchEdgeRuleConfig(enabled=True, batch_id_column="batch_id", counterparty_account_column="counterparty_acct_id", amount_column="amount"),
        similarity=SimilarityEdgeRuleConfig(enabled=False),
    )
    builder = EdgeBuilder(cfg)

    derived = builder.build(df)

    assert set(derived.edges[["src", "dst"]].itertuples(index=False, name=None)) == {("a", "b"), ("b", "a")}
    assert "shared_batch_count" in derived.feature_names
    assert derived.edges["type_batch"].max() == 1.0


def test_similarity_edges_use_fuzzy_matching():
    df = pd.DataFrame(
        {
            "counterparty_acct_id": ["a", "b"],
            "counterparty_name": ["ACME LTD", "Acme Limited"],
            "counterparty_city": ["nyc", "new york"],
        }
    )
    cfg = EdgeDerivationConfig(
        flow=FlowEdgeRuleConfig(enabled=False),
        batch=BatchEdgeRuleConfig(enabled=False),
        similarity=SimilarityEdgeRuleConfig(
            enabled=True,
            counterparty_account_column="counterparty_acct_id",
            name_column="counterparty_name",
            city_column="counterparty_city",
            name_similarity_threshold=0.8,
            city_similarity_threshold=0.5,
            token_similarity_threshold=0.5,
        ),
    )
    builder = EdgeBuilder(cfg)

    derived = builder.build(df)

    assert set(derived.edges[["src", "dst"]].itertuples(index=False, name=None)) == {("a", "b"), ("b", "a")}
    assert derived.edges["type_similarity"].max() == 1.0
