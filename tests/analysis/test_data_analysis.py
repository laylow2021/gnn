import pandas as pd

from gnn.analysis import DataAnalysis
from gnn.config import GraphConfig


def test_data_analysis_creates_artifacts(tmp_path):
    df = pd.DataFrame(
        {
            "src": ["a", "b", "a", "c"],
            "dst": ["b", "c", "c", "a"],
            "amount": [10.0, 20.0, 5.0, 7.5],
            "ts": pd.date_range("2024-01-01", periods=4, freq="D"),
        }
    )
    graph_cfg = GraphConfig(src_column="src", dst_column="dst", amount_column="amount", timestamp_column="ts")
    analysis = DataAnalysis(df, graph_cfg)

    artifacts = analysis.run_all(tmp_path)

    assert artifacts["transactions"]["metrics"].exists()
    assert artifacts["accounts"]["activity_table"].exists()
    assert artifacts["graph"]["metrics"].exists()
    assert artifacts["features"]["missingness"].exists()
