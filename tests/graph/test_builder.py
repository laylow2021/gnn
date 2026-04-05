import pytest
import pandas as pd
import numpy as np
from src.graph.builder import GraphBuilder

@pytest.fixture
def config():
    return {
        'graph': {
            'time_window_days': 2,
            'amount_tolerance_pct': 0.1,
            'edge_features': ['log_total_amount', 'edge_frequency', 'mean_scarcity', 'min_time_delta', 'outflow_ratio', 'inflow_ratio']
        },
        'data': {
            'column_mapping': {
                'customer_id': 'cust_id',
                'transaction_id': 'transaction_id',
                'amount': 'amt',
                'date': 'dt',
                'direction': 'dir',
                'direction_in': 'IN',
                'direction_out': 'OUT'
            }
        }
    }

def test_min_amount_matching(config):
    # Setup scenario: OUT 100, IN 110. Inferred should be 100.
    df = pd.DataFrame([
        {'transaction_id': 1, 'cust_id': 0, 'dt': pd.to_datetime('2026-01-01'), 'amt': 100.0, 'dir': 'OUT'},
        {'transaction_id': 2, 'cust_id': 1, 'dt': pd.to_datetime('2026-01-01 12:00:00'), 'amt': 110.0, 'dir': 'IN'}
    ])
    
    builder = GraphBuilder(config)
    data = builder.build_graph(df)
    
    collapsed = data.collapsed_edges_df
    assert len(collapsed) == 1
    assert collapsed.iloc[0]['total_inferred_amt'] == 100.0
    # outflow_ratio: 100 / 100 = 1.0
    assert pytest.approx(collapsed.iloc[0]['outflow_ratio'], 0.001) == 1.0
    # inflow_ratio: 100 / 110 = 0.909
    assert pytest.approx(collapsed.iloc[0]['inflow_ratio'], 0.001) == 0.909

def test_ratio_calculations_multiple_tx(config):
    # Source 0 has total OUT 500. 200 of it goes to Target 1.
    # Target 1 has total IN 1000. 200 of it comes from Source 0.
    df = pd.DataFrame([
        {'transaction_id': 1, 'cust_id': 0, 'dt': pd.to_datetime('2026-01-01'), 'amt': 200.0, 'dir': 'OUT'},
        {'transaction_id': 2, 'cust_id': 0, 'dt': pd.to_datetime('2026-01-01'), 'amt': 300.0, 'dir': 'OUT'}, # Noise
        {'transaction_id': 3, 'cust_id': 1, 'dt': pd.to_datetime('2026-01-01 01:00:00'), 'amt': 200.0, 'dir': 'IN'},
        {'transaction_id': 4, 'cust_id': 1, 'dt': pd.to_datetime('2026-01-01'), 'amt': 800.0, 'dir': 'IN'}  # Noise
    ])
    
    builder = GraphBuilder(config)
    data = builder.build_graph(df)
    
    collapsed = data.collapsed_edges_df
    edge_0_1 = collapsed[(collapsed['source'] == 0) & (collapsed['target'] == 1)]
    
    assert len(edge_0_1) == 1
    # total_inferred = 200
    # outflow_ratio = 200 / (200 + 300) = 0.4
    assert pytest.approx(edge_0_1.iloc[0]['outflow_ratio'], 0.001) == 0.4
    # inflow_ratio = 200 / (200 + 800) = 0.2
    assert pytest.approx(edge_0_1.iloc[0]['inflow_ratio'], 0.001) == 0.2

def test_precise_min_flow(config):
    # Scenario where IN is smaller than OUT
    # OUT 1000, IN 950. Inferred = 950.
    df = pd.DataFrame([
        {'transaction_id': 1, 'cust_id': 10, 'dt': pd.to_datetime('2026-01-05'), 'amt': 1000.0, 'dir': 'OUT'},
        {'transaction_id': 2, 'cust_id': 20, 'dt': pd.to_datetime('2026-01-05 02:00:00'), 'amt': 950.0, 'dir': 'IN'}
    ])
    
    builder = GraphBuilder(config)
    data = builder.build_graph(df)
    
    collapsed = data.collapsed_edges_df
    assert collapsed.iloc[0]['total_inferred_amt'] == 950.0

def test_self_loop_detection(config):
    # Setup scenario: Source 0 deposits and then withdraws same amount
    df = pd.DataFrame([
        {'transaction_id': 1, 'cust_id': 0, 'dt': pd.to_datetime('2026-01-01'), 'amt': 500.0, 'dir': 'OUT'},
        {'transaction_id': 2, 'cust_id': 0, 'dt': pd.to_datetime('2026-01-01 05:00:00'), 'amt': 500.0, 'dir': 'IN'}
    ])
    
    builder = GraphBuilder(config)
    data = builder.build_graph(df)
    
    collapsed = data.collapsed_edges_df
    assert len(collapsed) == 1
    assert collapsed.iloc[0]['source'] == 0
    assert collapsed.iloc[0]['target'] == 0
    assert collapsed.iloc[0]['total_inferred_amt'] == 500.0
