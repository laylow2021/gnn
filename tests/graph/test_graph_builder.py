import pytest
import pandas as pd
import torch
from src.gnn.graph.builder import GraphBuilder
from src.gnn.config.schema import Schema

@pytest.fixture
def mock_df():
    data = {
        Schema.TRANSACTION_ID: ['t1', 't2', 't3', 't4'],
        Schema.DATE: pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-01', '2023-01-03']),
        Schema.VOLUME: [100.0, 200.0, 150.0, 300.0],
        Schema.DIRECTION: ['Inbound', 'Outbound', 'Inbound', 'Outbound'],
        Schema.CUSTOMER_NAME: ['Alice', 'Bob', 'Alice', 'Charlie'],
        Schema.BANK_NAME: ['BankA', 'BankB', 'BankA', 'BankC'],
        Schema.BANK_ACCOUNT: ['Acc1', 'Acc2', 'Acc1', 'Acc3'],
        # Optional columns that might be pre-calculated
        Schema.RARITY_SCORE: [0.1, 0.2, 0.1, 0.3],
        Schema.TIMESTAMP_NORM: [0.0, 0.1, 0.0, 0.2]
    }
    return pd.DataFrame(data)

def test_graph_builder_structure(mock_df):
    builder = GraphBuilder()
    data = builder.build(mock_df)
    
    # Check Node Types
    assert Schema.NODE_CUSTOMER in data.node_types
    assert Schema.NODE_ACCOUNT in data.node_types
    assert Schema.NODE_TRANSACTION in data.node_types
    
    # Check Edge Types
    edge_types = data.edge_types
    assert (Schema.NODE_CUSTOMER, Schema.EDGE_OWNS, Schema.NODE_ACCOUNT) in edge_types
    assert (Schema.NODE_ACCOUNT, Schema.EDGE_EXECUTED, Schema.NODE_TRANSACTION) in edge_types

    # Check Node Counts
    # Customers: Alice, Bob, Charlie -> 3
    assert data[Schema.NODE_CUSTOMER].num_nodes == 3
    # Accounts: (BankA, Acc1), (BankB, Acc2), (BankC, Acc3) -> 3
    assert data[Schema.NODE_ACCOUNT].num_nodes == 3
    # Transactions: t1, t2, t3, t4 -> 4
    assert data[Schema.NODE_TRANSACTION].num_nodes == 4

def test_id_mapping_correctness(mock_df):
    builder = GraphBuilder()
    data = builder.build(mock_df)
    
    # Check mappings
    alice_id = builder.customer_to_idx['Alice']
    bob_id = builder.customer_to_idx['Bob']
    
    assert alice_id is not None
    assert bob_id is not None
    assert alice_id != bob_id
    
    # Check Account Mapping
    acc1_id = builder.account_to_idx[('BankA', 'Acc1')]
    assert acc1_id is not None

def test_features_shape(mock_df):
    builder = GraphBuilder()
    data = builder.build(mock_df)
    
    # Customer Features: [rarity, volume, count] -> 3 dims
    assert data[Schema.NODE_CUSTOMER].x.shape == (3, 3)
    
    # Transaction Features: [volume, timestamp, direction] -> 3 dims
    assert data[Schema.NODE_TRANSACTION].x.shape == (4, 3)

    # Account Features: Placeholder -> 9 dims (as per code)
    assert data[Schema.NODE_ACCOUNT].x.shape == (3, 9)
