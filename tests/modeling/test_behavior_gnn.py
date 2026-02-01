import pytest
import pandas as pd
import numpy as np
import torch
from src.gnn.modeling.behavior_gnn import BehaviorGraphBuilder, BehavioralGNN, train_behavior_gnn, detect_behavioral_anomalies

@pytest.fixture
def mock_behavior_df():
    # Create synthetic data with structure
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        'cust_name': [f'User_{i}' for i in range(n)],
        'txn_amt': np.random.uniform(10, 10000, n),
        'txn_date': pd.date_range('2024-01-01', periods=n, freq='D'), 
        'bank_name': np.random.choice(['BankA', 'BankB'], n),
        'direction': np.random.choice(['IN', 'OUT'], n)
    })
    return df

def test_builder_full_config(mock_behavior_df):
    config = {
        'customer': 'cust_name',
        'amount': 'txn_amt',
        'date': 'txn_date',
        'bank': 'bank_name',
        'direction': 'direction',
        'amount_bins': [-1, 5000, 100000]
    }
    
    builder = BehaviorGraphBuilder(config)
    data = builder.build(mock_behavior_df)
    
    # Check node types
    assert 'customer' in data.node_types
    assert 'amount_bin' in data.node_types
    assert 'day_of_month' in data.node_types
    assert 'bank' in data.node_types
    
    # Check edges exist
    assert ('customer', 'transacts_vol', 'amount_bin') in data.edge_types
    assert ('customer', 'uses_bank', 'bank') in data.edge_types
    assert data['amount_bin'].num_nodes <= 4 

def test_date_window_logic(mock_behavior_df):
    # Test Sliding Window
    config = {
        'customer': 'cust_name',
        'date': 'txn_date',
        'date_window': 1 
    }
    builder = BehaviorGraphBuilder(config)
    data = builder.build(mock_behavior_df)
    
    num_txns = len(mock_behavior_df)
    num_edges = data['customer', 'active_on_date', 'date'].edge_index.size(1)
    
    # Expect roughly 3 edges per txn (T-1, T, T+1)
    assert num_edges > num_txns * 2 
    assert num_edges <= num_txns * 3

def test_date_granularity(mock_behavior_df):
    # Test Coarsening
    config = {
        'customer': 'cust_name',
        'date': 'txn_date',
        'date_granularity': 'month'
    }
    builder = BehaviorGraphBuilder(config)
    data = builder.build(mock_behavior_df)
    
    # 100 days starting Jan 1st covers Jan, Feb, March, April.
    # Should result in ~4 date nodes.
    assert data['date'].num_nodes <= 5
    
    # Edge count should be exactly 1 per txn (no sliding window for monthly)
    num_edges = data['customer', 'active_on_date', 'date'].edge_index.size(1)
    assert num_edges == len(mock_behavior_df)

def test_training_pipeline(mock_behavior_df):
    config = {
        'customer': 'cust_name',
        'amount': 'txn_amt',
        'date': 'txn_date'
    }
    builder = BehaviorGraphBuilder(config)
    data = builder.build(mock_behavior_df)
    
    model = BehavioralGNN(data.metadata(), hidden_channels=16, out_channels=16)
    
    # Run one epoch of training
    model = train_behavior_gnn(model, data, epochs=1, lr=0.01)
    
    # Check output shape
    out = model(data.x_dict, data.edge_index_dict)
    assert out['customer'].shape == (100, 16)

def test_anomaly_detection(mock_behavior_df):
    config = {'customer': 'cust_name', 'amount': 'txn_amt'}
    builder = BehaviorGraphBuilder(config)
    data = builder.build(mock_behavior_df)
    model = BehavioralGNN(data.metadata(), hidden_channels=8, out_channels=8)
    
    # Mock inference
    results = detect_behavioral_anomalies(model, data, builder.cust_map)
    
    assert 'risk_score' in results.columns
    assert 'is_anomaly' in results.columns
    assert len(results) == 100
