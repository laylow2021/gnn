import pandas as pd
import numpy as np
import torch
from sklearn.ensemble import IsolationForest
from src.gnn.config.schema import Schema

def detect_anomalies(z_dict, idx_to_customer, contamination=0.01):
    """
    Runs Isolation Forest on the Customer embeddings to detect anomalies.
    """
    # Extract Customer Embeddings
    # Tensor to Numpy
    cust_embeddings = z_dict[Schema.NODE_CUSTOMER].detach().numpy()
    
    # Train Isolation Forest
    clf = IsolationForest(contamination=contamination, random_state=42)
    preds = clf.fit_predict(cust_embeddings) # 1 normal, -1 anomaly
    scores = clf.decision_function(cust_embeddings) # Lower is more anomalous
    
    # Map back to Customer Names
    results = []
    for i, score in enumerate(scores):
        cust_name = idx_to_customer.get(i, f"Unknown_{i}")
        is_anomaly = (preds[i] == -1)
        results.append({
            Schema.CUSTOMER_NAME: cust_name,
            "gnn_anomaly_score": score, # Raw score
            "is_anomaly": is_anomaly,
            "risk_score": 100 if is_anomaly else 0 # Simple mapping for now
        })
        
    df_results = pd.DataFrame(results)
    return df_results.sort_values("gnn_anomaly_score") # Lowest scores first (most anomalous)
