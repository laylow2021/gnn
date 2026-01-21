import pytest
import pandas as pd
import numpy as np
from src.gnn.analysis.statistical_aml import StatisticalAML
from src.gnn.config.schema import Schema

@pytest.fixture
def mock_df():
    # Create a mixed dataset
    # Dense User: Alice (6 txns, 3 accounts)
    # Sparse User: Bob (1 txn, 1 account)
    data = []
    
    # Alice
    for i in range(6):
        data.append({
            Schema.CUSTOMER_NAME: 'Alice',
            Schema.TRANSACTION_ID: f't_a_{i}',
            Schema.BANK_ACCOUNT: f'acc_{i%3}', # 0, 1, 2 (3 unique)
            Schema.BANK_NAME: 'BankA',
            Schema.VOLUME: 500.0, # Low entropy if all same
            Schema.DATE: pd.Timestamp('2023-01-01') + pd.Timedelta(days=i)
        })
        
    # Bob
    data.append({
        Schema.CUSTOMER_NAME: 'Bob',
        Schema.TRANSACTION_ID: 't_b_1',
        Schema.BANK_ACCOUNT: 'acc_b_1',
        Schema.BANK_NAME: 'BankB',
        Schema.VOLUME: 10000.0, # Outlier volume?
        Schema.DATE: pd.Timestamp('2023-01-01')
    })
    
    # Charlie (Sparse but Normal)
    data.append({
        Schema.CUSTOMER_NAME: 'Charlie',
        Schema.TRANSACTION_ID: 't_c_1',
        Schema.BANK_ACCOUNT: 'acc_c_1',
        Schema.BANK_NAME: 'BankB',
        Schema.VOLUME: 100.0,
        Schema.DATE: pd.Timestamp('2023-01-01')
    })
    
    return pd.DataFrame(data)

def test_split_population(mock_df):
    aml = StatisticalAML()
    dense, sparse = aml.split_population(mock_df)
    
    assert 'Alice' in dense[Schema.CUSTOMER_NAME].values
    assert 'Bob' not in dense[Schema.CUSTOMER_NAME].values
    
    assert 'Bob' in sparse[Schema.CUSTOMER_NAME].values
    assert 'Charlie' in sparse[Schema.CUSTOMER_NAME].values
    assert 'Alice' not in sparse[Schema.CUSTOMER_NAME].values

def test_track_a_pipeline(mock_df):
    aml = StatisticalAML()
    dense, _ = aml.split_population(mock_df)
    
    # Feature Engineering
    features = aml.engineer_features_track_a(dense)
    assert 'entropy' in features.columns
    assert 'Alice' in features.index
    # Entropy of constant 500.0 should be 0
    assert features.loc['Alice', 'entropy'] == 0.0
    
    # Run Model
    results = aml.run_track_a(features, eps=0.5, min_samples=1) # min_samples=1 to allow single cluster
    assert 'risk_score' in results.columns
    assert 'cluster' in results.columns

def test_track_b_pipeline(mock_df):
    aml = StatisticalAML()
    _, sparse = aml.split_population(mock_df)
    
    # Feature Engineering
    # Need enough data for bank stats usually, but code handles small data
    features = aml.engineer_features_track_b(sparse)
    assert 'z_score' in features.columns
    
    # Run Model
    results = aml.run_track_b(features, contamination=0.5) # High contam to force outlier
    assert 'risk_score' in results.columns
    
def test_full_flow(mock_df):
    aml = StatisticalAML()
    dense, sparse = aml.split_population(mock_df)
    
    f_a = aml.engineer_features_track_a(dense)
    res_a = aml.run_track_a(f_a, min_samples=1)
    
    f_b = aml.engineer_features_track_b(sparse)
    res_b = aml.run_track_b(f_b)
    
    final = aml.merge_results(res_a, res_b)
    
    assert len(final) == 3 # Alice, Bob, Charlie
    assert set(final[Schema.CUSTOMER_NAME]) == {'Alice', 'Bob', 'Charlie'}
    assert 'risk_score' in final.columns
    assert 'source_model' in final.columns
