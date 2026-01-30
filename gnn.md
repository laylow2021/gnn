# PROJECT CONTEXT: Unsupervised AML Transaction Monitoring (GNN + GAT)

## 1. Project Overview
**Goal:** Build an unsupervised anomaly detection system for PayPal transactions using a Graph Neural Network (GNN) approach.
**Data Scale:** 16 Million records, 40-day window.
**Problem Type:** Cold Start / Unsupervised (No SARs/Labels available).
**Key Challenge:** The "Supernode" skew—most users have 1 transaction, but some have 100+ accounts.

## 2. Technical Stack Constraints
* **Language:** Python 3.11+
* **Deep Learning:** PyTorch, PyTorch Geometric (PyG).
* **Data Manipulation:** Pandas, Scikit-Learn.
* **Graph Utilities:** NetworkX (for initial checks), PyG `HeteroData` for modeling.

## 3. Data Schema (Inferred)
The input dataframe (`df`) contains:
* `transaction_id`: Unique ID.
* `date`: Timestamp.
* `volume`: Transaction amount.
* `direction`: Flow (Inbound/Outbound).
* `customer_name`: Name of the external party.
* `bank_name`: External bank name.
* `bank_account`: External account number.

---

## 4. Implementation Plan

### Phase 1: Feature Engineering & Pre-processing
**Objective:** Prepare node features and handle the "Supernode" identifier.
1.  **Name Rarity Score:**
    * Calculate frequency of every `customer_name`.
    * Create feature: `rarity_score = 1 - (count / total_unique_names)`.
    * *Logic:* Differentiates common names (John Smith) from unique entities.
2.  **Entity Resolution:**
    * Treat `(customer_name, bank_account)` tuples as unique endpoints where necessary, but node aggregation should happen at the `customer_name` level for the graph.

### Phase 2: Heterogeneous Graph Construction (PyG)
**Objective:** Convert tabular data into a `HeteroData` object.
1.  **Node Types:**
    * `customer` (Features: `[rarity_score, total_volume, txn_count]`)
    * `account` (Features: `[bank_name_embedding, is_domestic]`)
    * `transaction` (Features: `[amount, timestamp_norm, direction]`)
2.  **Edge Types:**
    * (`customer`, `owns`, `account`)
    * (`account`, `executed`, `transaction`)
    * *Note:* Create reverse edges for message passing if using undirected logic.
3.  **Indexing:**
    * Map string IDs (`customer_name`) to contiguous integers `0...N` using `torch.unique` or `LabelEncoder`.
    * Store mappings to retrieve original IDs later.

### Phase 3: Model Architecture (GAT)
**Objective:** Handle the "Supernode" skew using Attention mechanisms.
1.  **Model Class:** `FraudGAT(torch.nn.Module)`
2.  **Layers:** Use `HeteroConv` wrapping `GATConv`.
    * *Why GAT?* To learn attention weights for neighbors. It prevents a customer with 100 accounts from having their signal "washed out" by averaging. The model learns to attend to the specific "risky" account among the 100.
3.  **Forward Pass:**
    * Layer 1: `GATConv` (Relu activation).
    * Layer 2: `GATConv` (Linear output for embeddings).

### Phase 4: Training (Self-Supervised Link Prediction)
**Objective:** Train the model to understand structural legitimacy without labels.
1.  **Strategy:** Negative Sampling.
    * **Positive Edges:** Real (`customer`, `owns`, `account`) links.
    * **Negative Edges:** Randomly paired (`customer`, `random_account`).
2.  **Loss Function:** Binary Cross Entropy or Max Margin Loss.
    * Maximize dot product similarity for real edges.
    * Minimize similarity for negative edges.

### Phase 5: Anomaly Detection (The Output)
1.  **Inference:** Run the trained model to generate `customer_embeddings` ($Z$).
2.  **Outlier Detection:**
    * Feed $Z$ into `sklearn.ensemble.IsolationForest`.
    * *Hypothesis:* Customers with complex, mule-like structures will have embeddings that cluster differently from normal users.
3.  **Output:** A list of `customer_name` sorted by the Isolation Forest anomaly score.

### Phase 6: Bipartite AML GNN Script

**Objective:** Provide a standalone, flexible script for bipartite financial network analysis.

1.  **Script Location:** `scripts/bipartite_aml_gnn.py`
2.  **Key Features:**
    *   **Unified Architecture:** Supports `GraphSAGE`, `GAT`, and `R-GCN` via a single configuration.
    *   **Bipartite Structure:** Explicitly models `customer` <-> `bank_account` relationships with bidirectional edges (`funds`, `pays`) to ensure information flow.
    *   **Scalability:** Implements `NeighborLoader` for handling large graphs by sampling neighborhoods during training and inference.
    *   **Flexibility:** User-defined column mapping allowing easy adaptation to different datasets without code changes.
    *   **Task:** Supports supervised learning (Binary Classification) with `BCEWithLogitsLoss`.

---

## 5. Coding Guidelines for the Assistant
* **Type Hinting:** Use strict Python type hints (e.g., `def train(data: HeteroData) -> float:`).
* **Modularity:** Keep the graph construction separate from the model definition.
* **Memory Efficiency:** The dataset is 16M rows. Use `float32` instead of `float64`. Use generators or batched processing (`NeighborLoader`) if the graph exceeds GPU memory.