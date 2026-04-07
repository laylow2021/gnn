import pandas as pd
import yaml
import os
import torch
from src.pipeline import AMLPipeline
from src.data.generator import generate_transactions
from src.models.gnn import AMLGraphAutoencoder, compute_combined_loss

def verify_full_pipeline():
    # 1. Setup
    config_path = 'config/config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Minimize data for fast testing
    config['data']['num_customers'] = 20
    
    pipeline = AMLPipeline(config)
    train_df = generate_transactions(num_customers=20, num_days=5, avg_tx_per_day=2)
    oot_df = generate_transactions(num_customers=20, num_days=2, avg_tx_per_day=2)
    
    print("--- 1. Testing Training Bypass ---")
    # Manually initialize model for training test since train() has a pass for training logic
    # We test if the graph building and model initialization works within the pipeline context
    data = pipeline.graph_builder.build_graph(train_df)
    in_channels = data.num_node_features
    edge_in_channels = data.num_edge_features
    
    pipeline.model = AMLGraphAutoencoder(in_channels, edge_in_channels, config).to(pipeline.device)
    print("Graph built and model initialized successfully.")

    # 2. Testing Save/Load
    print("\n--- 2. Testing Save/Load ---")
    checkpoint_path = 'artifacts/test_v2'
    pipeline.save(checkpoint_path)
    
    new_pipeline = AMLPipeline(config)
    new_pipeline.load_from_checkpoint(checkpoint_path)
    print("Pipeline saved and reloaded successfully.")

    # 3. Testing Scoring & Automated Reporting
    print("\n--- 3. Testing Scoring & Automated Reporting ---")
    pdf_out = 'output/test_report.pdf'
    excel_out = 'output/test_anomalies.xlsx'
    os.makedirs('output', exist_ok=True)
    
    # Test with automated neighborhood plotting (empty list)
    node_mse, edge_mse = new_pipeline.score(
        oot_df, 
        pdf_path=pdf_out, 
        excel_path=excel_out,
        local_neighborhood_node_ids=[] # Automated
    )
    
    # 4. Final Validations
    print("\n--- 4. Final Validations ---")
    assert os.path.exists(pdf_out), "PDF Report missing!"
    assert os.path.exists(excel_out), "Excel Report missing!"
    
    with pd.ExcelFile(excel_out) as xls:
        sheets = xls.sheet_names
        print(f"Generated Sheets: {sheets}")
        assert 'Top Anomaly Raw TX' in sheets
        assert 'Top Node Anomalies' in sheets
        
        # Check for feature MSE columns
        df_nodes = pd.read_excel(xls, 'Top Node Anomalies')
        mse_cols = [c for c in df_nodes.columns if c.startswith('mse_')]
        print(f"Feature MSE columns found: {mse_cols}")
        assert len(mse_cols) > 0, "Feature MSE decomposition missing from Excel!"

    print("\n[SUCCESS] Full pipeline (Train bypass, Save, Load, Score, Report) verified.")

if __name__ == "__main__":
    verify_full_pipeline()
