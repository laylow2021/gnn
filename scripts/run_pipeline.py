"""Example CLI to build graphs, run analysis, and train a model from config."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
import json

from gnn import (
    DataAnalysis,
    GraphBuilder,
    ModelRegistry,
    PipelineConfig,
)
from gnn.data import load_transactions, validate_graph_columns
from gnn.modeling import extract_embeddings, train_autoencoder, train_link_prediction
from gnn.graph import extract_topk_subgraphs
from gnn.transactions import TransactionAnomalyScorer, TransactionFeatureBuilder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run graph pipeline from a YAML config.")
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=Path("config/example.yaml"),
        help="Path to pipeline YAML config.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Optional number of rows to sample for quick runs.",
    )
    parser.add_argument(
        "--save-scored",
        type=Path,
        default=None,
        help="Optional path to write transactions with anomaly scores (CSV).",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=5,
        help="Number of top anomalous transactions to display.",
    )
    parser.add_argument(
        "--save-subgraphs",
        type=Path,
        default=None,
        help="Optional path to write subgraph summaries (CSV).",
    )
    parser.add_argument(
        "--subgraph-format",
        type=str,
        choices=["csv", "json"],
        default="csv",
        help="Format for subgraph export (when --save-subgraphs is set).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = PipelineConfig.model_validate(yaml.safe_load(args.config.read_text()))

    df = load_transactions(cfg.data.raw_path)
    if args.sample:
        df = df.sample(n=args.sample, random_state=cfg.data.seed)
    validate_graph_columns(df, cfg.graph)

    builder = GraphBuilder(cfg.graph, cfg.features)
    artifacts = builder.build(df)
    print(f"Graph built: {artifacts.data.num_nodes} nodes, {artifacts.data.edge_index.shape[1]} edges")
    if cfg.graph.edge_derivation:
        print("Edge derivation enabled; edge features:", artifacts.edge_features)

    analysis = DataAnalysis(df, cfg.graph, cfg.features)
    analysis.run_all(cfg.analysis.output_dir, graph=artifacts.data)

    registry = ModelRegistry()
    model = registry.build(cfg.model, in_channels=artifacts.data.x.shape[1])

    if cfg.model.task.value == "link_prediction":
        metrics, _ = train_link_prediction(
            model,
            artifacts.data,
            epochs=cfg.train.epochs,
            lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay,
            device=cfg.train.device,
        )
    elif cfg.model.task.value == "anomaly_autoencoder":
        metrics, _ = train_autoencoder(
            model,
            artifacts.data,
            epochs=cfg.train.epochs,
            lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay,
            device=cfg.train.device,
        )
    else:
        raise ValueError(f"Training not implemented for task {cfg.model.task}")

    print("Training metrics:", metrics)

    if cfg.transaction_anomaly.enabled:
        z = extract_embeddings(model, artifacts.data)
        embeddings = {acc: z[idx].detach().cpu().numpy() for idx, acc in enumerate(artifacts.accounts)}
        tfb = TransactionFeatureBuilder(cfg.transaction_anomaly)
        txn_feats = tfb.build(df, embeddings)
        scorer = TransactionAnomalyScorer(
            method=cfg.transaction_anomaly.method,
            contamination=cfg.transaction_anomaly.contamination,
        )
        scores, _ = scorer.fit_score(txn_feats.features)
        df_scored = df.copy()
        df_scored["transaction_anomaly_score"] = scores.values
        top = df_scored.nlargest(args.topk, "transaction_anomaly_score")
        print("Top transaction anomalies (head):")
        display_cols = [cfg.transaction_anomaly.counterparty_account_column, "transaction_anomaly_score"]
        for col in ("amount_column", "timestamp_column"):
            col_name = getattr(cfg.transaction_anomaly, col)
            if col_name and col_name in df_scored.columns:
                display_cols.append(col_name)
        if "txn_id" in df_scored.columns:
            display_cols.insert(0, "txn_id")
        print(top[display_cols].head())
        if args.save_scored:
            args.save_scored.parent.mkdir(parents=True, exist_ok=True)
            df_scored.to_csv(args.save_scored, index=False)
            print(f"Saved scored transactions to {args.save_scored}")

        # Optional subgraph extraction around top accounts
        account_scores = scores.groupby(df[cfg.transaction_anomaly.counterparty_account_column]).mean()
        subgraphs = extract_topk_subgraphs(
            artifacts.data,
            account_scores,
            k=args.topk,
            hops=cfg.transaction_anomaly.subgraph_hops,
            min_size=cfg.transaction_anomaly.subgraph_min_size,
        )
        if subgraphs:
            print("Top subgraphs around flagged accounts:")
            for sg in subgraphs:
                print(f"Seed={sg.seed}, score={sg.score:.4f}, nodes={len(sg.nodes)}, edges={sg.edge_count}, types={sg.edge_type_proportions}")
            if args.save_subgraphs:
                args.save_subgraphs.parent.mkdir(parents=True, exist_ok=True)
                records = []
                for sg in subgraphs:
                    base = {
                        "seed": sg.seed,
                        "score": sg.score,
                        "node_count": sg.node_count,
                        "edge_count": sg.edge_count,
                        "nodes": list(map(str, sg.nodes)),
                    }
                    if sg.edge_type_proportions:
                        for k, v in sg.edge_type_proportions.items():
                            base[f"prop_{k}"] = v
                    records.append(base)
                if args.subgraph_format == "json":
                    args.save_subgraphs.write_text(json.dumps(records, indent=2))
                else:
                    import pandas as pd

                    df_sub = pd.DataFrame(records)
                    # Expand nodes list into comma-separated string for CSV readability
                    df_sub["nodes"] = df_sub["nodes"].apply(lambda lst: ",".join(lst))
                    df_sub.to_csv(args.save_subgraphs, index=False)
                print(f"Saved subgraph summary to {args.save_subgraphs}")


if __name__ == "__main__":
    main()
