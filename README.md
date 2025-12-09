# gnn

Config-driven scaffold for building graph analytics pipelines. The repository is organized into four layers:
- **Config**: `PipelineConfig` ties together data, graph, feature, analysis, and model choices.
- **Data & Analysis**: loaders plus `DataAnalysis` for transaction/account/graph/feature overviews that run without any model.
- **Graph & Features**: `GraphBuilder` consumes declarative feature groups (structural, temporal, static, risk) to assemble tensors, and can derive counterparty edges (flow, co-batch, similarity) from raw fintech transactions.
- **Modeling & Anomaly**: `ModelRegistry` instantiates plug-and-play tasks (link prediction today, autoencoders for anomaly detection) with task-specific training/eval loops.
- **Transaction View**: Build transaction-level features (raw txn fields + graph embeddings) and score anomalies independently from the graph.

## Getting Started
- Python: 3.11+. Create a venv: `python -m venv .venv && .\.venv\Scripts\Activate.ps1`.
- Install deps: `pip install -r requirements.txt -r requirements-dev.txt`.
- Run tests: `pytest` from the repo root.

## Running
- CLI end-to-end: `python scripts/run_pipeline.py -c config/example.yaml --sample 1000 --topk 10 --save-scored artifacts/scored_transactions.csv --save-subgraphs artifacts/subgraphs.csv --subgraph-format csv`.
- Notebook: open `notebooks/pipeline_demo.ipynb` (falls back to a small synthetic sample if data is missing). Adjust config paths/column names to match your data.
- Config is the single source of truth; set column names in `graph.edge_derivation.*`, feature columns in `features.*`, and transaction columns in `transaction_anomaly.*`. Only columns present in your DataFrame will be used; missing columns are pruned.

## Quick Usage
```python
from pathlib import Path
import yaml
from gnn import DataAnalysis, GraphBuilder, ModelRegistry, PipelineConfig
from gnn.data import load_transactions, validate_graph_columns
from gnn.modeling import train_autoencoder, train_link_prediction

cfg = PipelineConfig.model_validate(yaml.safe_load(Path("config/example.yaml").read_text()))
df = load_transactions(cfg.data.raw_path)
validate_graph_columns(df, cfg.graph)

builder = GraphBuilder(cfg.graph, cfg.features)
artifacts = builder.build(df)

analysis = DataAnalysis(df, cfg.graph, cfg.features)
analysis.run_all(Path("artifacts/analysis"))

model = ModelRegistry().build(cfg.model, in_channels=artifacts.data.x.shape[1])
if cfg.model.task.value == "link_prediction":
    # Choose encoder via config params: mlp (default), graphsage, gcn, gat (with gat_heads)
    metrics, model = train_link_prediction(model, artifacts.data, epochs=cfg.train.epochs, lr=cfg.train.lr)
else:
    metrics, model = train_autoencoder(model, artifacts.data, epochs=cfg.train.epochs, lr=cfg.train.lr)
print(metrics)
```

Transaction anomaly scoring (View B) example:
```python
from gnn.transactions import TransactionFeatureBuilder, TransactionAnomalyScorer

embeddings = {"a": artifacts.data.x[0].numpy()}  # normally model outputs
tfb = TransactionFeatureBuilder(cfg.transaction_anomaly)
txn_feats = tfb.build(df, embeddings)
scores, model = TransactionAnomalyScorer(
    method=cfg.transaction_anomaly.method,
    contamination=cfg.transaction_anomaly.contamination,
).fit_score(txn_feats.features)
df["transaction_anomaly_score"] = scores
```

Similarity edges can be tuned via thresholds (`name_similarity_threshold`, `address_similarity_threshold`, `token_similarity_threshold`) in `graph.edge_derivation.similarity`.

CLI example (build graph, run analysis, train, score txns, and save):
```sh
python scripts/run_pipeline.py -c config/example.yaml --sample 1000 --topk 10 --save-scored artifacts/scored_transactions.csv --save-subgraphs artifacts/subgraphs.csv --subgraph-format csv
```

Subgraphs: the CLI extracts ego subgraphs around the top-K flagged accounts to aid investigation (printed to stdout), with hops/min_size configurable under `transaction_anomaly` and exportable as CSV or JSON via `--save-subgraphs`/`--subgraph-format`.
## Feature & Model Configuration
- Node and edge features are grouped under `structural`, `temporal`, `static`, and `risk` in `config/example.yaml`.
- Swap tasks by editing the `model` block (`link_prediction`, `anomaly_autoencoder`, more to come) without code changes.

## Code Overview
- See `docs/code_overview.md` for an ASCII diagram of modules/classes.
- Major entry points:
  - `gnn.graph.GraphBuilder` to create graph + features.
  - `gnn.graph.EdgeBuilder` to derive edges from flow/batch/similarity rules.
  - `gnn.modeling.ModelRegistry` and `training` to build/train models.
  - `gnn.transactions` for transaction feature building and anomaly scoring.
  - `gnn.graph.subgraph.extract_topk_subgraphs` for investigation.
  - `gnn.vis` for learning curves, graph plots, and anomaly score visuals (with optional SHAP if installed).

## Layout
- `src/gnn/`: package code (config, data/analysis, graph/features, modeling/anomaly).
- `tests/`: pytest suites mirroring `src/`.
- `scripts/`: helper CLIs or maintenance scripts.
- `docs/`: design or architecture notes.
- `data/`: small, versioned sample data (keep large/raw data out of Git).
