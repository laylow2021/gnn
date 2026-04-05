import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, Optional, List
from itertools import combinations

class GraphBuilder:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.mapping = config['data']['column_mapping']
        self.customer_map = {} # Maps original ID to 0..N-1
        self.inv_customer_map = {}

    def build_graph(self, tx_df: pd.DataFrame) -> Data:
        """Construct a homogeneous PyG graph with raw node/edge features (unscaled)."""
        c_id = self.mapping['customer_id']
        
        # 0. ID Mapping (Support for strings/non-contiguous IDs)
        unique_ids = sorted(tx_df[c_id].unique())
        self.customer_map = {original: i for i, original in enumerate(unique_ids)}
        self.inv_customer_map = {i: original for original, i in self.customer_map.items()}
        
        # Apply mapping to a copy for graph construction
        df = tx_df.copy()
        df[c_id] = df[c_id].map(self.customer_map)

        # 1. Min Transaction Amount Filter
        amt_col = self.mapping['amount']
        min_tx_amt = self.config['graph'].get('min_transaction_amount', 0)
        df = df[df[amt_col] >= min_tx_amt].copy()

        # Pre-process dates
        date_col = self.mapping['date']
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df[date_col] = pd.to_datetime(df[date_col].astype(str), format='%Y%m%d', errors='coerce')
        
        # Calculate totals per customer for ratio denominators
        customer_totals = self._calculate_customer_totals(df)
        
        node_features_df = self._calculate_node_features(df)
        edges_df = self._match_transactions(df)
        collapsed_edges = self._collapse_edges(edges_df, customer_totals)
        
        # 2. Min Edge Flow Ratio Filter
        min_ratio = self.config['graph'].get('min_edge_flow_ratio', 0)
        collapsed_edges = collapsed_edges[(collapsed_edges['inflow_ratio'] >= min_ratio) | 
                                          (collapsed_edges['outflow_ratio'] >= min_ratio)].copy()

        # 3. Frequency & Scarcity Filters (Noise Reduction)
        min_freq = self.config['graph'].get('min_edge_frequency', 1)
        min_scarcity = self.config['graph'].get('min_scarcity_threshold', 0.0)
        
        collapsed_edges = collapsed_edges[
            (collapsed_edges['edge_frequency'] >= min_freq) & 
            (collapsed_edges['mean_scarcity'] >= min_scarcity)
        ].copy()

        if collapsed_edges.empty:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, len(self.config['graph']['edge_features'])), dtype=torch.float)
        else:
            # Create edge_index using the integer-mapped columns
            edge_index = torch.tensor(collapsed_edges[['source', 'target']].values.T, dtype=torch.long)
            edge_attr_cols = self.config['graph']['edge_features']
            edge_attr = torch.tensor(collapsed_edges[edge_attr_cols].values, dtype=torch.float)
        
        # Final Node Features Construction
        x = torch.tensor(node_features_df.values, dtype=torch.float)
        
        # Explicitly check for NaNs/Infs and fail if found
        if torch.isnan(x).any() or torch.isnan(edge_attr).any():
            raise ValueError("Graph features contain NaNs. Imputation must be handled during preprocessing.")

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        
        # Add original IDs back to collapsed_edges for audit trails
        collapsed_edges['source_id'] = collapsed_edges['source'].map(self.inv_customer_map)
        collapsed_edges['target_id'] = collapsed_edges['target'].map(self.inv_customer_map)
        data.collapsed_edges_df = collapsed_edges
        
        return data

    def _calculate_customer_totals(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        c_id = self.mapping['customer_id']
        dir_col = self.mapping['direction']
        amt_col = self.mapping['amount']
        in_val = self.mapping['direction_in']
        out_val = self.mapping['direction_out']
        
        totals = df.groupby([c_id, dir_col])[amt_col].sum().unstack(fill_value=0.0)
        return {
            'in': totals.get(in_val, pd.Series(0.0, index=totals.index)),
            'out': totals.get(out_val, pd.Series(0.0, index=totals.index))
        }

    def _calculate_node_features(self, df: pd.DataFrame) -> pd.DataFrame:
        c_id = self.mapping['customer_id']
        node_feats = self.config['graph'].get('node_features', [])
        
        if not node_feats:
            # If no features defined, return a default identity or count if possible, 
            # but usually, node_features should be defined.
            features = df.groupby(c_id).size().to_frame('tx_count')
        else:
            # Aggregate all configured columns from the input data (mean by default)
            features = df.groupby(c_id)[node_feats].mean()

        # Reindex to cover all customers and fill with 0 to ensure continuous indices
        max_cust = df[c_id].max()
        all_customers = pd.DataFrame(index=range(max_cust + 1))
        return all_customers.join(features).fillna(0)

    def _match_transactions(self, df: pd.DataFrame) -> pd.DataFrame:
        strategy = self.config['graph'].get('matching_strategy', 'exact_1_to_1')
        
        # Track (transaction_id, counterparty_id) pairs to prevent duplicate edges 
        # between the same two people using the same transactions.
        used_pairs = set()
        
        if strategy == 'many_to_one':
            return self._match_many_to_one(df, used_pairs)
        elif strategy == 'one_to_many':
            return self._match_one_to_many(df, used_pairs)
        elif strategy == 'hybrid':
            m1 = self._match_many_to_one(df, used_pairs)
            m2 = self._match_one_to_many(df, used_pairs)
            return pd.concat([m1, m2])
        return self._match_one_to_one(df, used_pairs)

    def _match_one_to_one(self, df: pd.DataFrame, used_pairs: set) -> pd.DataFrame:
        c_id = self.mapping['customer_id']; dir_col = self.mapping['direction']; amt_col = self.mapping['amount']
        date_col = self.mapping['date']; tx_id_col = self.mapping['transaction_id']; in_val = self.mapping['direction_in']; out_val = self.mapping['direction_out']
        in_tx = df[df[dir_col] == in_val].copy(); out_tx = df[df[dir_col] == out_val].copy()
        matches = []; tol = self.config['graph']['amount_tolerance_pct']; window = self.config['graph']['time_window_days']
        
        for _, in_row in in_tx.iterrows():
            # Find candidate OUTs where this (IN_TX, OUT_Cust) hasn't been used (Inclusive temporal check)
            mask = ((out_tx[date_col] <= in_row[date_col]) & (out_tx[date_col] >= in_row[date_col] - pd.Timedelta(days=window)) &
                    (out_tx[amt_col] >= in_row[amt_col] * (1 - tol)) & (out_tx[amt_col] <= in_row[amt_col] * (1 + tol)))
            
            potential_outs = out_tx[mask]
            for _, out_row in potential_outs.iterrows():
                # Check uniqueness of the PAIR
                pair_in = (in_row[tx_id_col], out_row[c_id])
                pair_out = (out_row[tx_id_col], in_row[c_id])
                
                if pair_in not in used_pairs and pair_out not in used_pairs:
                    matches.append(self._create_match_dict(out_row, in_row, min(out_row[amt_col], in_row[amt_col]), 1.0, c_id, date_col, tx_id_col, amt_col))
                    used_pairs.add(pair_in)
                    used_pairs.add(pair_out)
                    break # One best match per IN transaction per strategy pass
        return pd.DataFrame(matches)

    def _match_many_to_one(self, df: pd.DataFrame, used_pairs: set) -> pd.DataFrame:
        """Finds multiple OUTs (mules) summing to one large IN (hub) with pair-level uniqueness."""
        c_id = self.mapping['customer_id']; dir_col = self.mapping['direction']; amt_col = self.mapping['amount']
        date_col = self.mapping['date']; tx_id_col = self.mapping['transaction_id']; in_val = self.mapping['direction_in']; out_val = self.mapping['direction_out']
        in_tx = df[df[dir_col] == in_val].copy(); out_tx = df[df[dir_col] == out_val].copy()
        matches = []; tol = self.config['graph']['amount_tolerance_pct']; window = self.config['graph']['time_window_days']
        max_depth = self.config['graph'].get('max_mule_depth', 4); min_tx = self.config['graph'].get('min_transaction_amount', 1)

        for _, in_row in in_tx.iterrows():
            target_amt = float(in_row[amt_col])
            potential_mask = ((out_tx[date_col] <= in_row[date_col]) & 
                              (out_tx[date_col] >= in_row[date_col] - pd.Timedelta(days=window)))
            candidates = out_tx[potential_mask]
            if candidates.empty: continue
            
            # 1:1 Match check first (with pair check)
            one_to_one = candidates[(candidates[amt_col] >= target_amt*(1-tol)) & (candidates[amt_col] <= target_amt*(1+tol))]
            if not one_to_one.empty:
                out_row = one_to_one.sort_values(date_col, ascending=False).iloc[0]
                pair_in = (in_row[tx_id_col], out_row[c_id]); pair_out = (out_row[tx_id_col], in_row[c_id])
                if pair_in not in used_pairs:
                    matches.append(self._create_match_dict(out_row, in_row, min(out_row[amt_col], target_amt), 1.0, c_id, date_col, tx_id_col, amt_col))
                    used_pairs.add(pair_in); used_pairs.add(pair_out)
                    continue

            # Optimized Many-to-One
            max_n = min(int(target_amt // min_tx), max_depth)
            for n in range(2, max_n + 1):
                part_amt = target_amt / n
                part_mask = (candidates[amt_col] >= part_amt*(1-tol)) & (candidates[amt_col] <= part_amt*(1+tol))
                parts = candidates[part_mask]
                
                # Check if enough parts haven't been used with this Hub Hub
                available_parts = []
                for _, p_row in parts.iterrows():
                    if (p_row[tx_id_col], in_row[c_id]) not in used_pairs:
                        available_parts.append(p_row)
                
                if len(available_parts) >= n:
                    for i in range(n):
                        out_row = available_parts[i]
                        matches.append(self._create_match_dict(out_row, in_row, out_row[amt_col], 1.0, c_id, date_col, tx_id_col, amt_col))
                        used_pairs.add((out_row[tx_id_col], in_row[c_id]))
                        used_pairs.add((in_row[tx_id_col], out_row[c_id]))
                    break 
        return pd.DataFrame(matches)

    def _match_one_to_many(self, df: pd.DataFrame, used_pairs: set) -> pd.DataFrame:
        """Finds one large OUT (hub) distributed into multiple smaller INs (mules) with pair-level uniqueness."""
        c_id = self.mapping['customer_id']; dir_col = self.mapping['direction']; amt_col = self.mapping['amount']
        date_col = self.mapping['date']; tx_id_col = self.mapping['transaction_id']; in_val = self.mapping['direction_in']; out_val = self.mapping['direction_out']
        in_tx = df[df[dir_col] == in_val].copy(); out_tx = df[df[dir_col] == out_val].copy()
        matches = []; tol = self.config['graph']['amount_tolerance_pct']; window = self.config['graph']['time_window_days']; max_depth = self.config['graph'].get('max_mule_depth', 4); min_tx = self.config['graph'].get('min_transaction_amount', 1)

        for _, out_row in out_tx.iterrows():
            source_amt = float(out_row[amt_col])
            potential_mask = ((in_tx[date_col] >= out_row[date_col]) & 
                              (in_tx[date_col] <= out_row[date_col] + pd.Timedelta(days=window)))
            candidates = in_tx[potential_mask]
            if candidates.empty: continue

            # 1:1 Match check first (with pair check)
            one_to_one = candidates[(candidates[amt_col] >= source_amt*(1-tol)) & (candidates[amt_col] <= source_amt*(1+tol))]
            if not one_to_one.empty:
                in_row = one_to_one.sort_values(date_col, ascending=True).iloc[0]
                pair_in = (in_row[tx_id_col], out_row[c_id]); pair_out = (out_row[tx_id_col], in_row[c_id])
                if pair_out not in used_pairs:
                    matches.append(self._create_match_dict(out_row, in_row, min(source_amt, in_row[amt_col]), 1.0, c_id, date_col, tx_id_col, amt_col))
                    used_pairs.add(pair_out); used_pairs.add(pair_in)
                    continue

            # Optimized One-to-Many
            max_n = min(int(source_amt // min_tx), max_depth)
            for n in range(2, max_n + 1):
                part_amt = source_amt / n
                part_mask = (candidates[amt_col] >= part_amt*(1-tol)) & (candidates[amt_col] <= part_amt*(1+tol))
                parts = candidates[part_mask]

                # Check if enough parts haven't been used with this Hub
                available_parts = []
                for _, p_row in parts.iterrows():
                    if (p_row[tx_id_col], out_row[c_id]) not in used_pairs:
                        available_parts.append(p_row)

                if len(available_parts) >= n:
                    for i in range(n):
                        in_row = available_parts[i]
                        matches.append(self._create_match_dict(out_row, in_row, in_row[amt_col], 1.0, c_id, date_col, tx_id_col, amt_col))
                        used_pairs.add((in_row[tx_id_col], out_row[c_id]))
                        used_pairs.add((out_row[tx_id_col], in_row[c_id]))
                    break 
        return pd.DataFrame(matches)
def _create_match_dict(self, out_row, in_row, amt, scarcity_multiplier, c_id, date_col, tx_id_col, amt_col):
    # Use provided inverse frequency from data if mapped, else default to 1.0
    sc_col = self.mapping.get('amount_scarcity')
    base_scarcity = out_row[sc_col] if sc_col and sc_col in out_row else 1.0
    scarcity = base_scarcity * scarcity_multiplier

    return {
        'source': out_row[c_id], 'target': in_row[c_id], 'inferred_amount': amt,
...
            'time_delta': (in_row[date_col] - out_row[date_col]).total_seconds() / 3600,
            'scarcity_score': scarcity, 'out_tx_id': out_row[tx_id_col],
            'out_date': out_row[date_col], 'out_amount_raw': out_row[amt_col],
            'in_tx_id': in_row[tx_id_col], 'in_date': in_row[date_col], 'in_amount_raw': in_row[amt_col]
        }

    def _collapse_edges(self, matches_df: pd.DataFrame, customer_totals: Dict[str, pd.Series]) -> pd.DataFrame:
        if matches_df.empty: return pd.DataFrame(columns=['source', 'target'] + self.config['graph']['edge_features'])
        def join_metadata(x): return list(x)
        collapsed = matches_df.groupby(['source', 'target']).agg(
            total_inferred_amt=('inferred_amount', 'sum'), edge_frequency=('inferred_amount', 'count'),
            mean_scarcity=('scarcity_score', 'mean'), min_time_delta=('time_delta', 'min'),
            out_tx_ids=('out_tx_id', join_metadata), out_amounts=('out_amount_raw', join_metadata),
            out_dates=('out_date', join_metadata), in_tx_ids=('in_tx_id', join_metadata),
            in_amounts=('in_amount_raw', join_metadata), in_dates=('in_date', join_metadata)
        ).reset_index()
        source_total_out = customer_totals['out'].reindex(collapsed['source']).values
        collapsed['outflow_ratio'] = collapsed['total_inferred_amt'] / (source_total_out + 1e-9)
        target_total_in = customer_totals['in'].reindex(collapsed['target']).values
        collapsed['inflow_ratio'] = collapsed['total_inferred_amt'] / (target_total_in + 1e-9)
        collapsed['log_total_amount'] = np.log1p(collapsed['total_inferred_amt'])
        return collapsed