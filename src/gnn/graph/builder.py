"""Build graph objects and feature tensors from tabular transactions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from gnn.config.schema import FeaturesConfig, GraphConfig
from gnn.features import FeatureCalculator
from .edges import EdgeBuilder


@dataclass
class GraphArtifacts:
    """Artifacts produced when building a graph."""

    data: Data
    accounts: List[str]
    edge_features: List[str]
    node_features: List[str]


class GraphBuilder:
    """Convert raw transaction DataFrames into graph structures with features."""

    def __init__(self, graph_cfg: GraphConfig, features_cfg: FeaturesConfig) -> None:
        self.graph_cfg = graph_cfg
        self.features_cfg = features_cfg
        self._calculator = FeatureCalculator(graph_cfg)
        self._edge_builder = EdgeBuilder(graph_cfg.edge_derivation) if graph_cfg.edge_derivation else None

    def build(self, df: pd.DataFrame) -> GraphArtifacts:
        """Create a torch_geometric Data graph with node/edge features."""
        if self._edge_builder:
            derived = self._edge_builder.build(df)
            edges_df = derived.edges.rename(columns={"src": self.graph_cfg.src_column, "dst": self.graph_cfg.dst_column})
            accounts = self._collect_accounts(edges_df)
            edge_index = self._build_edge_index(edges_df, accounts)
            edge_attr, edge_feature_names = self._tensor_from_edge_frame(edges_df, derived.feature_names)
            base_df = df.copy()
            # Ensure node feature computations have src/dst columns; if missing, map to counterparty column used in edge derivation.
            counterparty_col = None
            if self.graph_cfg.edge_derivation:
                if self.graph_cfg.edge_derivation.flow and self.graph_cfg.edge_derivation.flow.counterparty_account_column:
                    counterparty_col = self.graph_cfg.edge_derivation.flow.counterparty_account_column
                elif self.graph_cfg.edge_derivation.batch and self.graph_cfg.edge_derivation.batch.counterparty_account_column:
                    counterparty_col = self.graph_cfg.edge_derivation.batch.counterparty_account_column
                elif self.graph_cfg.edge_derivation.similarity and self.graph_cfg.edge_derivation.similarity.counterparty_account_column:
                    counterparty_col = self.graph_cfg.edge_derivation.similarity.counterparty_account_column
            
            if self.graph_cfg.src_column not in base_df.columns and counterparty_col and counterparty_col in base_df.columns:
                base_df[self.graph_cfg.src_column] = base_df[counterparty_col]
            if self.graph_cfg.dst_column not in base_df.columns and counterparty_col and counterparty_col in base_df.columns:
                base_df[self.graph_cfg.dst_column] = base_df[counterparty_col]
            node_attr, node_feature_names = self._build_node_features(base_df, accounts)
        else:
            accounts = self._collect_accounts(df)
            edge_index = self._build_edge_index(df, accounts)
            edge_attr, edge_feature_names = self._build_edge_features(df)
            node_attr, node_feature_names = self._build_node_features(df, accounts)

        data = Data(
            x=node_attr,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )
        data.num_nodes = len(accounts)
        data.account_ids = accounts
        data.metadata = {
            "account_to_idx": {acc: idx for idx, acc in enumerate(accounts)},
            "feature_schema": {"node": node_feature_names, "edge": edge_feature_names},
            "label": self.graph_cfg.metadata_label,
        }
        if self.graph_cfg.target_column and self.graph_cfg.target_column in df.columns:
            data.y = torch.tensor(df[self.graph_cfg.target_column].to_numpy())

        return GraphArtifacts(
            data=data,
            accounts=accounts,
            edge_features=edge_feature_names,
            node_features=node_feature_names,
        )

    def _collect_accounts(self, df: pd.DataFrame) -> List[str]:
        if self.graph_cfg.src_column not in df.columns or self.graph_cfg.dst_column not in df.columns:
            raise KeyError("source/destination columns missing for graph construction")
        accounts = pd.Index(
            pd.unique(
                pd.concat(
                    [
                        df[self.graph_cfg.src_column],
                        df[self.graph_cfg.dst_column],
                    ],
                    ignore_index=True,
                )
            )
        ).astype(str)
        return accounts.tolist()

    def _build_edge_index(self, df: pd.DataFrame, accounts: Sequence[str]) -> torch.Tensor:
        account_to_idx = {acc: idx for idx, acc in enumerate(accounts)}
        src = df[self.graph_cfg.src_column].astype(str).map(account_to_idx).to_numpy()
        dst = df[self.graph_cfg.dst_column].astype(str).map(account_to_idx).to_numpy()
        edge_index = np.vstack([src, dst]).astype(np.int64)
        return torch.from_numpy(edge_index)

    def _tensor_from_edge_frame(self, df: pd.DataFrame, feature_names: List[str]) -> Tuple[torch.Tensor, List[str]]:
        if feature_names:
            tensor = torch.tensor(df[feature_names].to_numpy(), dtype=torch.float)
        else:
            tensor = torch.empty((len(df), 0), dtype=torch.float)
        return tensor, feature_names

    def _build_edge_features(self, df: pd.DataFrame) -> Tuple[torch.Tensor, List[str]]:
        frames: List[pd.DataFrame] = []
        names: List[str] = []

        for group, definition in self.features_cfg.edge.iter_definitions():
            feature_df = self._calculator.compute_edge_feature(df, definition, group)
            frames.append(feature_df)
            names.extend(feature_df.columns.tolist())

        if frames:
            concat = pd.concat(frames, axis=1)
            tensor = torch.tensor(concat.to_numpy(), dtype=torch.float)
        else:
            tensor = torch.empty((len(df), 0), dtype=torch.float)
        return tensor, names

    def _build_node_features(
        self, df: pd.DataFrame, accounts: Sequence[str]
    ) -> Tuple[torch.Tensor, List[str]]:
        frames: List[pd.DataFrame] = []
        names: List[str] = []

        for group, definition in self.features_cfg.node.iter_definitions():
            feature_df = self._calculator.compute_node_feature(df, accounts, definition, group)
            frames.append(feature_df)
            names.extend(feature_df.columns.tolist())

        if frames:
            concat = pd.concat(frames, axis=1)
        else:
            concat = pd.DataFrame(index=accounts)

        # Ensure ordering aligns with accounts list
        concat = concat.reindex(accounts).fillna(0)
        tensor = torch.tensor(concat.to_numpy(), dtype=torch.float)
        return tensor, names
