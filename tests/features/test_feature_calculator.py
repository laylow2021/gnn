import pandas as pd

from gnn.config import FeatureDefinition, FeatureGroupConfig, FeaturesConfig, GraphConfig
from gnn.graph import GraphBuilder


def test_rolling_sum_node_feature(tmp_path):
    df = pd.DataFrame(
        {
            "src": ["a", "a", "a", "b"],
            "dst": ["b", "b", "c", "a"],
            "amount": [1.0, 2.0, 3.0, 4.0],
            "ts": pd.to_datetime(["2024-01-01", "2024-01-05", "2024-01-08", "2024-01-02"]),
        }
    )
    features_cfg = FeaturesConfig(
        node=FeatureGroupConfig(
            structural=[FeatureDefinition(name="in_degree", method="degree_in")],
            temporal=[
                FeatureDefinition(
                    name="amt_rolling",
                    columns=["ts", "amount"],
                    method="rolling_sum",
                    params={"window_days": 5},
                )
            ],
        )
    )
    graph_cfg = GraphConfig(src_column="src", dst_column="dst", timestamp_column="ts")
    builder = GraphBuilder(graph_cfg, features_cfg)

    artifacts = builder.build(df)

    # For account a: last two events within 5 days window have amounts 2.0 and 3.0 => rolling sum 5.0
    rolling_col_index = artifacts.node_features.index("amt_rolling")
    a_index = artifacts.accounts.index("a")
    assert artifacts.data.x[a_index, rolling_col_index] == 5.0


def test_edge_category_one_hot_feature():
    df = pd.DataFrame(
        {
            "src": ["a", "b"],
            "dst": ["b", "c"],
            "bank": ["x", "y"],
        }
    )
    features_cfg = FeaturesConfig(
        edge=FeatureGroupConfig(
            structural=[FeatureDefinition(name="bank", columns=["bank"], method="category_one_hot")]
        )
    )
    graph_cfg = GraphConfig(src_column="src", dst_column="dst")
    builder = GraphBuilder(graph_cfg, features_cfg)

    artifacts = builder.build(df)

    assert artifacts.data.edge_attr.shape[1] == 2  # bank__x, bank__y
    assert set(artifacts.edge_features) == {"bank__x", "bank__y"}
