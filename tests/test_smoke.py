from gnn import GraphBuilder, ModelRegistry, PipelineConfig


def test_imports_available():
    cfg = PipelineConfig.model_validate(
        {
            "data": {"raw_path": None, "target": None, "splits": {"train": 0.7, "val": 0.2, "test": 0.1}},
            "graph": {"src_column": "u", "dst_column": "v"},
            "features": {},
            "model": {"task": "link_prediction"},
        }
    )
    assert cfg.graph.src_column == "u"
    assert isinstance(ModelRegistry(), ModelRegistry)
    # GraphBuilder is imported from package; detailed build tests live elsewhere.
    assert GraphBuilder
