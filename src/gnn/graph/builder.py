import torch
from torch_geometric.data import HeteroData
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional
from src.gnn.config.schema import Schema

class GraphBuilder:
    """
    Constructs a PyG HeteroData object from the transactions DataFrame.
    Focuses on ID mapping and graph connectivity.
    """

    def __init__(self):
        # Mappings: Original ID -> Contiguous Integer
        self.customer_to_idx: Dict[str, int] = {}
        self.idx_to_customer: Dict[int, str] = {}
        
        # Account ID is a tuple of (bank_name, bank_account)
        self.account_to_idx: Dict[Tuple[str, str], int] = {}
        self.idx_to_account: Dict[int, Tuple[str, str]] = {}
        
        self.txn_to_idx: Dict[str, int] = {}
        self.idx_to_txn: Dict[int, str] = {}

    def _get_or_create_idx(self, key, mapping, reverse_mapping) -> int:
        if key not in mapping:
            idx = len(mapping)
            mapping[key] = idx
            reverse_mapping[idx] = key
        return mapping[key]

    def build(self, df: pd.DataFrame) -> HeteroData:
        """
        Builds the HeteroData object.
        
        Args:
            df: Preprocessed DataFrame containing transactions.
                Must contain columns defined in Schema.
        """
        data = HeteroData()

        # ---------------------------------------------------------
        # 1. ID Mapping & Node Extraction
        # ---------------------------------------------------------
        # We need to iterate efficiently. 
        # For large dataframes, vectorization is preferred, but for ID mapping 
        # ensuring uniqueness and consistency might require factorize or unique.
        
        # Customer IDs
        unique_customers = df[Schema.CUSTOMER_NAME].unique()
        # Update mappings
        for cust in unique_customers:
            self._get_or_create_idx(cust, self.customer_to_idx, self.idx_to_customer)
        
        # Account IDs (Composite key: Bank Name + Account Number)
        # Optimization: Use drop_duplicates on the DataFrame slice to avoid 
        # creating a massive list of tuples for the set() constructor.
        unique_accounts_df = df[[Schema.BANK_NAME, Schema.BANK_ACCOUNT]].drop_duplicates()
        
        # itertuples(index=False, name=None) yields standard tuples
        for acc_tuple in unique_accounts_df.itertuples(index=False, name=None):
            self._get_or_create_idx(acc_tuple, self.account_to_idx, self.idx_to_account)

        # Transaction IDs
        unique_txns = df[Schema.TRANSACTION_ID].unique()
        for txn in unique_txns:
            self._get_or_create_idx(txn, self.txn_to_idx, self.idx_to_txn)

        # ---------------------------------------------------------
        # 2. Edge Index Construction
        # ---------------------------------------------------------
        # Map DataFrame columns to integer IDs
        
        # We can use map() but it might be slow for very large DFs if dict is huge.
        # For 16M records, we might want to optimize this, but for now standard map is safe.
        
        # Customer Indices
        customer_indices = df[Schema.CUSTOMER_NAME].map(self.customer_to_idx).values
        
        # Account Indices
        # Create a temporary series of tuples to map
        # Note: This creation can be memory intensive. 
        # Alternative: Multi-index map or apply.
        acc_series = pd.Series(list(zip(df[Schema.BANK_NAME], df[Schema.BANK_ACCOUNT])))
        account_indices = acc_series.map(self.account_to_idx).values
        
        # Transaction Indices
        txn_indices = df[Schema.TRANSACTION_ID].map(self.txn_to_idx).values

        # Define Edges:
        # (customer, owns, account)
        # One customer can execute a transaction via an account.
        # Ideally, data structure implies: Customer -> Account
        # The dataframe links Customer and Account on each row.
        # We need unique (customer, account) pairs for the 'owns' edge.
        
        # Extract unique (customer_idx, account_idx) pairs
        cust_acc_pairs = np.unique(np.vstack([customer_indices, account_indices]), axis=1)
        
        # Edge: Customer -> Account
        data[Schema.NODE_CUSTOMER, Schema.EDGE_OWNS, Schema.NODE_ACCOUNT].edge_index = \
            torch.tensor(cust_acc_pairs, dtype=torch.long)

        # Edge: Account -> Transaction
        # Each row is a transaction executed by the account on that row.
        # So it's a 1-to-1 mapping from the dataframe row (Account ID -> Transaction ID)
        acc_txn_pairs = np.vstack([account_indices, txn_indices])
        
        data[Schema.NODE_ACCOUNT, Schema.EDGE_EXECUTED, Schema.NODE_TRANSACTION].edge_index = \
            torch.tensor(acc_txn_pairs, dtype=torch.long)

        # ---------------------------------------------------------
        # 3. Node Features (Placeholders/Calculated)
        # ---------------------------------------------------------
        
        # -- Customer Features --
        # [rarity_score, total_volume, txn_count]
        # We need to aggregate at customer level.
        # Assuming rarity_score is per customer (constant for a customer).
        # We'll groupby customer_idx.
        
        # Create a temporary DF with mapped indices to facilitate grouping
        df_mapped = df.copy()
        df_mapped['cust_idx'] = customer_indices
        df_mapped['acc_idx'] = account_indices
        
        # Ensure we cover all customers (0 to N-1)
        num_customers = len(self.customer_to_idx)
        
        # Aggregations
        cust_group = df_mapped.groupby('cust_idx')
        
        # Total Volume
        cust_vol = cust_group[Schema.VOLUME].sum().reindex(range(num_customers), fill_value=0)
        
        # Txn Count
        cust_count = cust_group[Schema.VOLUME].count().reindex(range(num_customers), fill_value=0)
        
        # Rarity Score (Check if exists, else 0)
        if Schema.RARITY_SCORE in df.columns:
            # Take first value as it should be constant per customer
            cust_rarity = cust_group[Schema.RARITY_SCORE].first().reindex(range(num_customers), fill_value=0)
        else:
            cust_rarity = pd.Series(0, index=range(num_customers))

        cust_features = torch.tensor(np.stack([
            cust_rarity.values, 
            cust_vol.values, 
            cust_count.values
        ], axis=1), dtype=torch.float)
        
        data[Schema.NODE_CUSTOMER].x = cust_features
        data[Schema.NODE_CUSTOMER].num_nodes = num_customers

        # -- Account Features --
        # [bank_name_embedding, is_domestic]
        # For now, we will use placeholders or simple encodings.
        num_accounts = len(self.account_to_idx)
        
        # For bank_name_embedding, we'd need a model. 
        # Let's just use a random vector or integer encoding for now?
        # The prompt says: [bank_name_embedding, is_domestic]
        # We'll use a dummy embedding of size 8 for bank name + 1 for domestic
        # This is a placeholder.
        
        # In a real scenario, we'd map bank names to embeddings.
        # Here we just initialize zeros.
        acc_features = torch.zeros((num_accounts, 9), dtype=torch.float) 
        data[Schema.NODE_ACCOUNT].x = acc_features
        data[Schema.NODE_ACCOUNT].num_nodes = num_accounts

        # -- Transaction Features --
        # [amount, timestamp_norm, direction]
        # Sort by txn_idx to ensure alignment (since txn_idx was assigned based on unique order, 
        # but map returned values in DF order. Wait.
        # The dataframe rows are transactions. If txn IDs are unique per row, 
        # we can just reorder the DF by txn_idx.
        
        # Create a DF indexed by txn_idx
        df_txns = df_mapped.set_index('txn_to_idx' if 'txn_to_idx' in df_mapped else df_mapped.index) 
        # Actually we have 'txn_indices' array corresponding to DF rows.
        # We need features for Transaction 0, Transaction 1, ...
        # Since we assigned IDs based on 'unique_txns', we can construct the feature matrix in that order.
        
        # Faster way: 
        # We have the raw DF. We can drop duplicates on transaction_id (if any, though they should be unique)
        # and then map to ID, sort by ID.
        
        txn_feat_df = df[[Schema.TRANSACTION_ID, Schema.VOLUME, Schema.DIRECTION]].drop_duplicates(subset=Schema.TRANSACTION_ID)
        
        if Schema.TIMESTAMP_NORM in df.columns:
             txn_feat_df[Schema.TIMESTAMP_NORM] = df[Schema.TIMESTAMP_NORM]
        else:
             txn_feat_df[Schema.TIMESTAMP_NORM] = 0.0

        txn_feat_df['idx'] = txn_feat_df[Schema.TRANSACTION_ID].map(self.txn_to_idx)
        txn_feat_df = txn_feat_df.sort_values('idx')
        
        # Handle Direction (String -> Int)
        # Assuming 'Inbound'/'Outbound'.
        direction_map = {'Inbound': 1, 'Outbound': -1} # Simple mapping
        txn_feat_df['dir_val'] = txn_feat_df[Schema.DIRECTION].map(direction_map).fillna(0)
        
        txn_features = torch.tensor(np.stack([
            txn_feat_df[Schema.VOLUME].values,
            txn_feat_df[Schema.TIMESTAMP_NORM].values,
            txn_feat_df['dir_val'].values
        ], axis=1), dtype=torch.float)
        
        data[Schema.NODE_TRANSACTION].x = txn_features
        data[Schema.NODE_TRANSACTION].num_nodes = len(self.txn_to_idx)

        return data
