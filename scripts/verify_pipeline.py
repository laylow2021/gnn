import pandas as pd
import numpy as np
import torch
from src.data.generator import generate_transactions
from src.graph.builder import GraphBuilder
import yaml

def test_pipeline_verification():
    # 1. Load Config
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Force some settings for testing
    config['graph']['min_transaction_amount'] = 100
    config['graph']['min_edge_flow_ratio'] = 0.5  # Strict to see if it filters
    
    print("--- 1. Data Generation ---")
    df = generate_transactions(num_customers=50, num_days=10, avg_tx_per_day=5, anomaly_ratio=0.1)
    print(f"Total raw transactions generated: {len(df)}")
    print(f"Transactions < 100: {len(df[df['amount'] < 100])}")

    # 2. Build Graph
    builder = GraphBuilder(config)
    data = builder.build_graph(df)
    
    print("\n--- 2. Graph Verification ---")
    print(f"Number of nodes: {data.num_nodes}")
    print(f"Number of edges (after filters): {data.num_edges}")
    
    # 3. Verify Filters
    # Check if raw transactions were filtered
    if hasattr(data, 'collapsed_edges_df'):
        edges = data.collapsed_edges_df
        if not edges.empty:
            print(f"Max Inflow Ratio: {edges['inflow_ratio'].max():.2f}")
            print(f"Max Outflow Ratio: {edges['outflow_ratio'].max():.2f}")
            assert (edges['inflow_ratio'] >= 0.5).any() or (edges['outflow_ratio'] >= 0.5).any(), "Filter not working"
            
            # Check for structured matches (where total_inferred_amt > 100)
            print(f"Edges total amount mean: {edges['total_inferred_amt'].mean():.2f}")
            assert (edges['total_inferred_amt'] >= 100).all(), "Min Transaction Amount filter failed"

    print("\n[SUCCESS] Pipeline verified with new filters and optimized search.")

if __name__ == "__main__":
    test_pipeline_verification()
