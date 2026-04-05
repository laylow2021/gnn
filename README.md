# AML Graph Autoencoder for FinTech Fraud Detection

An enterprise-grade, modular GNN pipeline for detecting money laundering (AML) anomalies using a homogeneous virtual-edge graph architecture.

## 🚀 Key Features
- **Heuristic Virtual Edges:** Infers latent connections between customers based on transaction timing and amount symmetry.
  - **Many-to-One (Fan-In):** Multiple Withdrawals (OUT) $\to$ One Deposit (IN).
  - **One-to-Many (Fan-Out):** One Withdrawal (OUT) $\to$ Multiple Deposits (IN).
- **Edge Collapsing:** Prevents multigraph swamping by aggregating raw matched transactions into single weighted edges (velocity, scarcity, time-delta).

- **GATv2 Autoencoder:** Uses Graph Attention Networks to learn structural and feature-based normal behavior.
- **Out-of-Time (OOT) Validation:** Built-in support for training on historical periods and testing on future data with persistent scalers and models.
- **Deep Explainability:** Global and Local feature importance for both node and edge anomalies, decomposing reconstruction error into actionable insights.

## 📂 Project Structure
```text
├── config/
│   └── config.yaml          # Hyperparameters & Column Mappings
├── artifacts/
│   └── model_v1/            # Saved Model, Scaler, and Config
├── notebooks/
│   └── main_pipeline.ipynb  # End-to-End OOT Training & Inference
├── src/
│   ├── data/
│   │   └── generator.py     # AML Typology Simulation (Pass-Through, Fan-In)
│   ├── graph/
│   │   └── builder.py       # Heuristic Matching & Edge Collapsing
│   ├── models/
│   │   └── gnn.py           # GATv2 Autoencoder & Combined MSE Loss
│   ├── explain/
│   │   └── interpreter.py   # Global/Local Feature Importance & Visualization
│   └── utils/
│       └── reproducibility.py # Seeding & Checkpoint Persistence
└── requirements.txt         # Core Dependencies
```

## 🛠️ Getting Started
1. **Environment Setup:**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install -r requirements.txt
   ```
2. **Run Pipeline:**
   Open `notebooks/main_pipeline.ipynb` in Jupyter and execute all cells.

## ⚙️ Configuration (`config/config.yaml`)
The project is fully configuration-driven. You can adapt it to any dataset by updating the `column_mapping`:
```yaml
data:
  column_mapping:
    customer_id: "your_customer_id"
    amount: "tx_amount"
    date: "tx_timestamp"
    direction: "tx_type"
    direction_in: "DEPOSIT"
    direction_out: "WITHDRAWAL"
```

## 🧠 Model Logic
### 1. Graph Construction
- **Node Features:** `directional_ratio`, `funding_deficit`, and `temporal_z_score`.
- **Edge Matching:** Heuristic 1:1 or Many:1 linking within `time_window_days` and `amount_tolerance_pct`.
- **Edge Features:** `log_total_amount`, `edge_frequency`, `mean_scarcity`, and `min_time_delta`.

### 2. Loss Function
$$Loss_{Total} = 0.7(\text{Node MSE}) + 0.3(\text{Edge Attr MSE})$$
- We avoid Link Prediction (BCE) to prevent circular feedback loops on virtual edges.

### 3. Anomaly Explanation
- **Nodes:** High reconstruction error in specific features (e.g., a massive `funding_deficit`).
- **Edges:** "Surprising" connections where the aggregate velocity or amount doesn't fit the learned pattern.