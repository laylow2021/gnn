from gnn.config import (
    FeaturesConfig,
    ModelTask,
    PipelineConfig,
)


def test_pipeline_config_validates_and_labels_groups():
    cfg = {
        "data": {"raw_path": "data/sample.csv", "target": "label", "splits": {"train": 0.6, "val": 0.2, "test": 0.2}},
        "graph": {"src_column": "from_account", "dst_column": "to_account", "amount_column": "amount"},
        "features": {
            "node": {"structural": [{"name": "in_degree", "method": "degree_in"}]},
            "edge": {"structural": [{"name": "amount", "columns": ["amount"], "method": "identity"}]},
        },
        "model": {"task": "link_prediction", "name": "lp-baseline", "params": {"hidden_dim": 32}},
        "analysis": {"include_plots": False},
    }

    config = PipelineConfig.model_validate(cfg)

    assert config.model.task == ModelTask.LINK_PREDICTION
    assert isinstance(config.features, FeaturesConfig)
    assert len(config.features.node.structural) == 1
    assert config.data.splits["train"] == 0.6


def test_split_sum_validation():
    bad_cfg = {
        "data": {"raw_path": "data/sample.csv", "target": "label", "splits": {"train": 0.5, "val": 0.3, "test": 0.3}},
        "graph": {"src_column": "u", "dst_column": "v"},
        "features": {},
        "model": {"task": "link_prediction"},
    }
    try:
        PipelineConfig.model_validate(bad_cfg)
        assert False, "Expected ValueError for split validation"
    except ValueError:
        assert True
