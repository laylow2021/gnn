"""Edge derivation for counterparty graphs."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from difflib import SequenceMatcher
from typing import List, Tuple

import pandas as pd

from gnn.config import EdgeDerivationConfig


@dataclass
class DerivedEdges:
    """Derived edges and feature names."""

    edges: pd.DataFrame
    feature_names: List[str]


class EdgeBuilder:
    """Derive flow, co-batch, and similarity edges from fintech transaction tables."""

    def __init__(self, config: EdgeDerivationConfig) -> None:
        self.config = config

    def build(self, df: pd.DataFrame) -> DerivedEdges:
        frames: List[pd.DataFrame] = []

        if self.config.flow.enabled:
            frames.append(self._build_flow_edges(df))
        if self.config.batch.enabled:
            frames.append(self._build_batch_edges(df))
        if self.config.similarity.enabled:
            frames.append(self._build_similarity_edges(df))

        if not frames:
            raise ValueError("No edge derivation rules enabled.")

        combined = pd.concat(frames, ignore_index=True)
        agg = (
            combined.groupby(["src", "dst"])
            .agg(
                {
                    col: "sum"
                    for col in combined.columns
                    if col not in {"src", "dst", "type_flow", "type_batch", "type_similarity"}
                }
            )
            .reset_index()
        )
        # Preserve type flags (max = any)
        for flag in ["type_flow", "type_batch", "type_similarity"]:
            if flag in combined.columns:
                agg_flag = combined.groupby(["src", "dst"])[flag].max().reset_index()[flag]
                agg[flag] = agg_flag

        feature_names = [c for c in agg.columns if c not in {"src", "dst"}]
        return DerivedEdges(edges=agg, feature_names=feature_names)

    def _build_flow_edges(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config.flow
        required = [
            cfg.fintech_account_column,
            cfg.counterparty_account_column,
            cfg.timestamp_column,
            cfg.amount_column,
            cfg.direction_column,
        ]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise KeyError(f"Missing columns for flow edges: {missing}")

        inbound = df[df[cfg.direction_column].isin(cfg.incoming_values)].copy()
        outbound = df[df[cfg.direction_column].isin(cfg.outgoing_values)].copy()
        if inbound.empty or outbound.empty:
            return pd.DataFrame(columns=["src", "dst", "flow_count", "flow_total_amount", "flow_avg_amount", "median_time_lag_minutes", "type_flow"])

        inbound[cfg.timestamp_column] = pd.to_datetime(inbound[cfg.timestamp_column])
        outbound[cfg.timestamp_column] = pd.to_datetime(outbound[cfg.timestamp_column])

        edges: List[Tuple] = []
        tol = cfg.amount_tolerance_pct
        window = pd.Timedelta(minutes=cfg.time_window_minutes)
        for row in inbound.itertuples(index=False):
            ts_in = getattr(row, cfg.timestamp_column)
            amt_in = float(getattr(row, cfg.amount_column))
            cp_in = getattr(row, cfg.counterparty_account_column)
            ref_in = getattr(row, cfg.reference_column) if cfg.reference_column and cfg.reference_column in inbound.columns else None

            candidates = outbound[
                (outbound[cfg.timestamp_column] >= ts_in)
                & (outbound[cfg.timestamp_column] <= ts_in + window)
            ]
            if cfg.reference_column and cfg.reference_column in outbound.columns:
                candidates = candidates[candidates[cfg.reference_column] == ref_in]

            amt_low, amt_high = amt_in * (1 - tol), amt_in * (1 + tol)
            candidates = candidates[
                (candidates[cfg.amount_column] >= amt_low)
                & (candidates[cfg.amount_column] <= amt_high)
            ]

            for out_row in candidates.itertuples(index=False):
                cp_out = getattr(out_row, cfg.counterparty_account_column)
                ts_out = getattr(out_row, cfg.timestamp_column)
                amt_out = float(getattr(out_row, cfg.amount_column))
                lag_min = (ts_out - ts_in).total_seconds() / 60.0
                edges.append((cp_in, cp_out, amt_in, amt_out, lag_min))

        if not edges:
            return pd.DataFrame(columns=["src", "dst", "flow_count", "flow_total_amount", "flow_avg_amount", "median_time_lag_minutes", "type_flow"])

        flow_df = pd.DataFrame(edges, columns=["src", "dst", "amt_in", "amt_out", "lag_min"])
        grouped = flow_df.groupby(["src", "dst"])
        agg = grouped.agg(
            flow_count=("lag_min", "count"),
            flow_total_amount=("amt_in", "sum"),
            flow_avg_amount=("amt_in", "mean"),
            median_time_lag_minutes=("lag_min", "median"),
        ).reset_index()
        agg["type_flow"] = 1.0
        return agg

    def _build_batch_edges(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config.batch
        if not cfg.batch_id_column or cfg.batch_id_column not in df.columns:
            return pd.DataFrame(columns=["src", "dst", "shared_batch_count", "type_batch"])

        groups = df.groupby(cfg.batch_id_column)
        rows: List[Tuple] = []
        for _, group in groups:
            accounts = group[cfg.counterparty_account_column].dropna().astype(str).unique().tolist()
            if len(accounts) < 2:
                continue
            amount_lookup = dict(zip(group[cfg.counterparty_account_column], group.get(cfg.amount_column, 0)))
            for a, b in combinations(accounts, 2):
                amt_a = float(amount_lookup.get(a, 0) or 0)
                amt_b = float(amount_lookup.get(b, 0) or 0)
                similarity = 1.0 - (abs(amt_a - amt_b) / max(max(amt_a, amt_b), 1.0)) if cfg.amount_column else 0.0
                rows.append((a, b, 1.0, similarity))
                rows.append((b, a, 1.0, similarity))

        if not rows:
            return pd.DataFrame(columns=["src", "dst", "shared_batch_count", "avg_amount_similarity", "type_batch"])

        batch_df = pd.DataFrame(rows, columns=["src", "dst", "shared_batch_count", "avg_amount_similarity"])
        agg = batch_df.groupby(["src", "dst"]).agg(
            shared_batch_count=("shared_batch_count", "sum"),
            avg_amount_similarity=("avg_amount_similarity", "mean"),
        ).reset_index()
        agg["type_batch"] = 1.0
        return agg

    def _build_similarity_edges(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config.similarity
        account_col = cfg.counterparty_account_column
        if account_col not in df.columns:
            return pd.DataFrame(columns=["src", "dst", "similarity_score", "name_match", "address_match", "bank_match", "type_similarity"])

        norm = lambda x: str(x).strip().lower() if pd.notna(x) else ""
        rows: List[Tuple] = []

        def token_similarity(a: str, b: str) -> float:
            tokens_a = set(a.split())
            tokens_b = set(b.split())
            if not tokens_a or not tokens_b:
                return 0.0
            intersection = len(tokens_a & tokens_b)
            union = len(tokens_a | tokens_b)
            return intersection / union

        def pairs_from_column(col: str, threshold: float = 1.0, token_thresh: float = 0.0):
            if not col or col not in df.columns:
                return []
            grouped = df[df[col].notna()].copy()
            grouped[col] = grouped[col].apply(norm)
            for _, sub in grouped.groupby(col):
                accounts = sub[account_col].astype(str).unique().tolist()
                for a, b in combinations(accounts, 2):
                    yield a, b
            # fuzzy matching within column
            unique_vals = grouped[col].unique().tolist()
            for i, val_i in enumerate(unique_vals):
                for val_j in unique_vals[i + 1 :]:
                    if not val_i or not val_j:
                        continue
                    ratio = SequenceMatcher(None, val_i, val_j).ratio()
                    token_ratio = token_similarity(val_i, val_j)
                    if ratio >= threshold or token_ratio >= token_thresh:
                        accounts_i = grouped[grouped[col] == val_i][account_col].astype(str).unique().tolist()
                        accounts_j = grouped[grouped[col] == val_j][account_col].astype(str).unique().tolist()
                        for a in accounts_i:
                            for b in accounts_j:
                                yield a, b

        name_pairs = (
            set(
                pairs_from_column(
                    cfg.name_column,
                    cfg.name_similarity_threshold,
                    cfg.token_similarity_threshold,
                )
            )
            if cfg.name_column
            else set()
        )
        address_pairs = (
            set(
                pairs_from_column(
                    cfg.address_column,
                    cfg.address_similarity_threshold,
                    cfg.token_similarity_threshold,
                )
            )
            if cfg.address_column
            else set()
        )
        city_pairs = (
            set(
                pairs_from_column(
                    cfg.city_column,
                    cfg.city_similarity_threshold,
                    cfg.token_similarity_threshold,
                )
            )
            if cfg.city_column
            else set()
        )
        bank_pairs = set(pairs_from_column(cfg.bank_id_column)) if cfg.bank_id_column else set()

        all_pairs = (
            set.union(name_pairs, address_pairs, city_pairs, bank_pairs)
            if (name_pairs or address_pairs or city_pairs or bank_pairs)
            else set()
        )
        for a, b in all_pairs:
            name_match = 1.0 if (a, b) in name_pairs else 0.0
            address_match = 1.0 if (a, b) in address_pairs else 0.0
            city_match = 1.0 if (a, b) in city_pairs else 0.0
            bank_match = 1.0 if (a, b) in bank_pairs else 0.0
            # Address gets higher weight; others equal
            weight_name = 1.0 if cfg.name_column else 0.0
            weight_address = 2.0 if cfg.address_column else 0.0
            weight_city = 1.0 if cfg.city_column else 0.0
            weight_bank = 1.0 if cfg.bank_id_column else 0.0
            total_weight = max(weight_name + weight_address + weight_city + weight_bank, 1.0)
            weighted_sum = (
                name_match * weight_name
                + address_match * weight_address
                + city_match * weight_city
                + bank_match * weight_bank
            )
            similarity_score = weighted_sum / total_weight
            rows.append((a, b, similarity_score, name_match, address_match, city_match, bank_match))
            rows.append((b, a, similarity_score, name_match, address_match, city_match, bank_match))

        if not rows:
            return pd.DataFrame(columns=["src", "dst", "similarity_score", "name_match", "address_match", "city_match", "bank_match", "type_similarity"])

        sim_df = pd.DataFrame(
            rows,
            columns=["src", "dst", "similarity_score", "name_match", "address_match", "city_match", "bank_match"],
        )
        agg = sim_df.groupby(["src", "dst"]).agg(
            similarity_score=("similarity_score", "mean"),
            name_match=("name_match", "max"),
            address_match=("address_match", "max"),
            city_match=("city_match", "max"),
            bank_match=("bank_match", "max"),
        ).reset_index()
        agg["type_similarity"] = 1.0
        return agg
