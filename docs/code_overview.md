# Code Overview

This project is organized into four layers with config-driven wiring.

```
[config]
  PipelineConfig
     ├─ GraphConfig (edge_derivation rules)
     ├─ FeaturesConfig (node/edge FeatureDefinition)
     ├─ ModelConfig / TrainingConfig
     └─ TransactionAnomalyConfig

[data + analysis]
  gnn.data.loader           -> load_transactions, validate_graph_columns
  gnn.analysis.DataAnalysis -> transaction/account/graph/feature summaries

[graph + features]
  gnn.graph.GraphBuilder
     ├─ EdgeBuilder (flow/batch/similarity edges)
     └─ FeatureCalculator (node/edge feature tensors)

[modeling + anomaly]
  gnn.modeling.ModelRegistry -> builds task models
  gnn.modeling.training      -> train_link_prediction / train_autoencoder
  gnn.modeling.embeddings    -> extract_embeddings
  gnn.transactions           -> TransactionFeatureBuilder + TransactionAnomalyScorer
  gnn.graph.subgraph         -> extract_topk_subgraphs
```

### Main flows
- **Graph build:** `GraphBuilder(cfg.graph, cfg.features).build(df)` -> torch_geometric `Data` with node/edge features and metadata.
- **Edge derivation:** `EdgeBuilder` infers flow/co-batch/similarity edges; skip if you already have src/dst edges.
- **Features:** `FeatureCalculator` turns declarative `FeatureDefinition`s into tensors (structural/temporal/static/risk groups).
- **Model:** `ModelRegistry` creates the configured model; `training` modules fit and return metrics + trained model.
- **Embeddings:** `extract_embeddings(model, data)` returns node embeddings for downstream use.
- **Transaction view:** `TransactionFeatureBuilder` combines txn fields + embeddings; `TransactionAnomalyScorer` scores anomalies.
- **Subgraphs:** `extract_topk_subgraphs` builds ego subgraphs around high-scoring accounts for investigation.

### Key config knobs
- Column names live in `GraphConfig` and `EdgeDerivationConfig.flow/batch/similarity`; set them to match your data.
- Feature columns live in each `FeatureDefinition.columns`; only columns present in your DataFrame are used.
- Task selection lives in `ModelConfig.task` (`link_prediction`, `anomaly_autoencoder`, etc.).
- Transaction anomaly options live in `TransactionAnomalyConfig` (columns, method, contamination, subgraph hops/min size).
