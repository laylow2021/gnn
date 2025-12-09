"""Data and graph analysis that runs independently of models."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from torch_geometric.data import Data  # noqa: E402

from gnn.config.schema import FeaturesConfig, GraphConfig


class DataAnalysis:
    """Generate transaction, account, graph, and feature overviews."""

    def __init__(
        self,
        df: pd.DataFrame,
        graph_cfg: GraphConfig,
        features_cfg: Optional[FeaturesConfig] = None,
    ) -> None:
        self.df = df
        self.graph_cfg = graph_cfg
        self.features_cfg = features_cfg or FeaturesConfig()

    def _ensure_dir(self, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def _get_account_series(self) -> tuple[pd.Series, pd.Series]:
        """Return source/destination account series, falling back to counterparty column if needed."""
        src_col = self.graph_cfg.src_column
        dst_col = self.graph_cfg.dst_column
        if src_col in self.df.columns and dst_col in self.df.columns:
            return self.df[src_col], self.df[dst_col]

        cp_col = None
        if self.graph_cfg.edge_derivation and self.graph_cfg.edge_derivation.flow:
            cp_col = self.graph_cfg.edge_derivation.flow.counterparty_account_column
        if cp_col and cp_col in self.df.columns:
            series = self.df[cp_col]
            return series, series
        # Fallback: use index to avoid crashing
        dummy = pd.Series(range(len(self.df)))
        return dummy, dummy

    def transaction_overview(self, output_dir: Path) -> Dict[str, Path]:
        output_dir = self._ensure_dir(output_dir)
        outputs: Dict[str, Path] = {}

        src_series, dst_series = self._get_account_series()
        metrics = {
            "num_transactions": len(self.df),
            "num_accounts": int(pd.Index(pd.unique(pd.concat([src_series, dst_series], ignore_index=True))).nunique()),
        }

        if self.graph_cfg.amount_column and self.graph_cfg.amount_column in self.df.columns:
            amount_series = self.df[self.graph_cfg.amount_column].astype(float)
            metrics.update(
                {
                    "total_amount": float(amount_series.sum()),
                    "mean_amount": float(amount_series.mean()),
                    "median_amount": float(amount_series.median()),
                }
            )
            plt.figure()
            sns.histplot(amount_series, bins=30, kde=False)
            plt.title("Transaction amount distribution")
            plt.xlabel(self.graph_cfg.amount_column)
            amount_plot = output_dir / "transaction_amount_hist.png"
            plt.savefig(amount_plot, bbox_inches="tight")
            plt.close()
            outputs["amount_plot"] = amount_plot

        if self.graph_cfg.timestamp_column and self.graph_cfg.timestamp_column in self.df.columns:
            ts = pd.to_datetime(self.df[self.graph_cfg.timestamp_column])
            metrics["time_span_days"] = float((ts.max() - ts.min()).total_seconds() / 86400.0)

        metrics_df = pd.DataFrame([metrics])
        metrics_path = output_dir / "transaction_overview.csv"
        metrics_df.to_csv(metrics_path, index=False)
        outputs["metrics"] = metrics_path
        return outputs

    def account_overview(self, output_dir: Path) -> Dict[str, Path]:
        output_dir = self._ensure_dir(output_dir)
        outputs: Dict[str, Path] = {}

        src_series, dst_series = self._get_account_series()
        src_counts = src_series.value_counts()
        dst_counts = dst_series.value_counts()
        account_summary = pd.DataFrame({"outgoing": src_counts, "incoming": dst_counts}).fillna(0)
        account_summary["total"] = account_summary["outgoing"] + account_summary["incoming"]
        summary_path = output_dir / "account_activity.csv"
        account_summary.sort_values("total", ascending=False).to_csv(summary_path)
        outputs["activity_table"] = summary_path

        top_accounts = account_summary.sort_values("total", ascending=False).head(20)
        plt.figure(figsize=(10, 4))
        sns.barplot(x=top_accounts.index.astype(str), y=top_accounts["total"])
        plt.xticks(rotation=45, ha="right")
        plt.title("Top accounts by total activity")
        plt.ylabel("transactions")
        bar_path = output_dir / "top_accounts.png"
        plt.savefig(bar_path, bbox_inches="tight")
        plt.close()
        outputs["top_accounts_plot"] = bar_path
        return outputs

    def graph_overview(self, output_dir: Path, graph: Optional[Data] = None) -> Dict[str, Path]:
        output_dir = self._ensure_dir(output_dir)
        outputs: Dict[str, Path] = {}

        if graph is None:
            G = nx.DiGraph() if self.graph_cfg.directed else nx.Graph()
            src_series, dst_series = self._get_account_series()
            edges = list(zip(src_series.astype(str), dst_series.astype(str)))
            G.add_edges_from(edges)
        else:
            G = nx.DiGraph() if self.graph_cfg.directed else nx.Graph()
            src, dst = graph.edge_index.tolist()
            account_ids = getattr(graph, "account_ids", list(range(graph.num_nodes)))
            edges = [(account_ids[s], account_ids[d]) for s, d in zip(src, dst)]
            G.add_edges_from(edges)

        metrics = {
            "num_nodes": G.number_of_nodes(),
            "num_edges": G.number_of_edges(),
            "density": float(nx.density(G)),
            "average_degree": float(
                sum(dict(G.degree()).values()) / max(1, G.number_of_nodes())
            ),
        }
        metrics_path = output_dir / "graph_overview.csv"
        pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
        outputs["metrics"] = metrics_path

        degrees = [deg for _, deg in G.degree()]
        if degrees:
            plt.figure()
            sns.histplot(degrees, bins=min(30, max(degrees)))
            plt.xlabel("degree")
            plt.ylabel("count")
            plt.title("Graph degree distribution")
            deg_path = output_dir / "degree_distribution.png"
            plt.savefig(deg_path, bbox_inches="tight")
            plt.close()
            outputs["degree_plot"] = deg_path

        return outputs

    def feature_overview(self, output_dir: Path) -> Dict[str, Path]:
        output_dir = self._ensure_dir(output_dir)
        outputs: Dict[str, Path] = {}

        missing = self.df.isna().mean().to_frame("missing_ratio")
        missing_path = output_dir / "feature_missingness.csv"
        missing.to_csv(missing_path)
        outputs["missingness"] = missing_path

        numeric_cols = self.df.select_dtypes(include=["number"]).columns.tolist()
        if len(numeric_cols) >= 2:
            corr = self.df[numeric_cols].corr()
            corr_path = output_dir / "feature_correlation.csv"
            corr.to_csv(corr_path)
            outputs["correlation"] = corr_path

            plt.figure(figsize=(8, 6))
            sns.heatmap(corr, cmap="coolwarm", center=0)
            plt.title("Numeric feature correlation")
            heatmap_path = output_dir / "feature_correlation.png"
            plt.savefig(heatmap_path, bbox_inches="tight")
            plt.close()
            outputs["correlation_plot"] = heatmap_path

        return outputs

    def run_all(self, output_dir: Path, graph: Optional[Data] = None) -> Dict[str, Dict[str, Path]]:
        """Run every overview and return produced artifacts."""
        output_dir = self._ensure_dir(output_dir)
        artifacts = {
            "transactions": self.transaction_overview(output_dir / "transactions"),
            "accounts": self.account_overview(output_dir / "accounts"),
            "graph": self.graph_overview(output_dir / "graph", graph=graph),
            "features": self.feature_overview(output_dir / "features"),
        }
        return artifacts
