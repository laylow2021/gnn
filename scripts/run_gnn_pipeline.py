import sys
import os
import argparse
import pandas as pd
import torch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.gnn.graph.builder import GraphBuilder
from src.gnn.modeling.training import train_model
from src.gnn.analysis.inference import detect_anomalies
from src.gnn.config.schema import Schema

def main():
    parser = argparse.ArgumentParser(description="Run End-to-End GNN AML Pipeline")
    parser.add_argument('--input', type=str, help='Path to input CSV', default='data/transactions.csv')
    parser.add_argument('--epochs', type=int, default=10, help='Training epochs')
    parser.add_argument('--output', type=str, default='output_risks.csv', help='Output path')
    
    args = parser.parse_args()
    
    # 1. Load Data
    # For demo purposes, if file doesn't exist, we generate mock data
    if not os.path.exists(args.input):
        print(f"Input file {args.input} not found. Generating mock data...")
        from tests.graph.test_graph_builder import mock_df
        # We need a bigger mock df for training to work well
        import numpy as np
        data = {
            Schema.TRANSACTION_ID: [f't{i}' for i in range(100)],
            Schema.DATE: pd.date_range('2023-01-01', periods=100),
            Schema.VOLUME: np.random.rand(100) * 1000,
            Schema.DIRECTION: np.random.choice(['Inbound', 'Outbound'], 100),
            Schema.CUSTOMER_NAME: np.random.choice(['CustA', 'CustB', 'CustC', 'CustD', 'Mule'], 100),
            Schema.BANK_NAME: ['BankX'] * 100,
            Schema.BANK_ACCOUNT: [f'Acc{i%10}' for i in range(100)],
            Schema.RARITY_SCORE: np.random.rand(100),
            Schema.TIMESTAMP_NORM: np.random.rand(100)
        }
        df = pd.DataFrame(data)
    else:
        df = pd.read_csv(args.input)
        
    print(f"Loaded {len(df)} transactions.")
    
    # 2. Build Graph
    print("Building Graph...")
    builder = GraphBuilder()
    graph = builder.build(df)
    print(graph)
    
    # 3. Train Model
    print("Training FraudGAT...")
    model, embeddings = train_model(graph, epochs=args.epochs)
    
    # 4. Inference
    print("Running Anomaly Detection...")
    results = detect_anomalies(embeddings, builder.idx_to_customer)
    
    # 5. Output
    print(f"Saving results to {args.output}")
    results.to_csv(args.output, index=False)
    
    print("\n--- Top Anomalies ---")
    print(results.head())

if __name__ == "__main__":
    main()
