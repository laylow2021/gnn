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
        'txn_date': pd.date_range('2024-01-01', periods=n, freq='H'),
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
        'direction': 'direction'
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

def test_builder_reduced_scope(mock_behavior_df):
    # Test with minimal config (just customer and amount)
    config = {
        'customer': 'cust_name',
        'amount': 'txn_amt'
    }
    
    builder = BehaviorGraphBuilder(config)
    data = builder.build(mock_behavior_df)
    
    assert 'customer' in data.node_types
    assert 'amount_bin' in data.node_types
    assert 'date' not in data.node_types
    assert 'bank' not in data.node_types

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
