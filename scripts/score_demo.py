import pandas as pd
import yaml
import os
from src.pipeline import AMLPipeline
from src.data.generator import generate_transactions

def run_score_demo():
    # 1. Load Config
    config_path = 'config/config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 2. Initialize Pipeline
    pipeline = AMLPipeline(config)

    # 3. Load Model from Artifacts
    model_path = 'artifacts/model_v1'
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found. Please ensure artifacts are present.")
        # Create a dummy model and save it if not exists, for demonstration purposes
        # But in a real scenario, the user would have the model.
        # I'll try to load it first.
        return

    pipeline.load_from_checkpoint(model_path)

    # 4. Generate some test data
    print("Generating test transactions...")
    test_df = generate_transactions(num_customers=50, num_days=5, avg_tx_per_day=3, anomaly_ratio=0.05)

    # 5. Define output paths
    pdf_output = 'output/scoring_results.pdf'
    excel_output = 'output/scoring_results.xlsx'
    os.makedirs('output', exist_ok=True)

    # 6. Run Scoring
    # Pick a random node for local neighborhood visualization if nodes exist
    local_node = test_df['customer_id'].iloc[0] if not test_df.empty else None
    
    node_mse, edge_mse = pipeline.score(
        test_df, 
        pdf_path=pdf_output, 
        excel_path=excel_output,
        local_neighborhood_node_id=local_node
    )

    print(f"\nDemo Complete!")
    print(f"PDF Results: {pdf_output}")
    print(f"Excel Results: {excel_output}")

    # Verify Excel sheets
    if os.path.exists(excel_output):
        with pd.ExcelFile(excel_output) as xls:
            print(f"Excel Sheets: {xls.sheet_names}")
            assert 'Top Anomaly Raw TX' in xls.sheet_names
            assert 'Top Node Anomalies' in xls.sheet_names
            assert 'Top Edge Anomalies' in xls.sheet_names

if __name__ == "__main__":
    run_score_demo()
