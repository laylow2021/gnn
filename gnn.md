\# PROJECT CONTEXT: Unsupervised AML Transaction Monitoring (GNN + GAT)



\## 1. Project Overview

\*\*Goal:\*\* Build an unsupervised anomaly detection system for PayPal transactions using a Graph Neural Network (GNN) approach.

\*\*Data Scale:\*\* 16 Million records, 40-day window.

\*\*Problem Type:\*\* Cold Start / Unsupervised (No SARs/Labels available).

\*\*Key Challenge:\*\* The "Supernode" skew—most users have 1 transaction, but some have 100+ accounts.



\## 2. Technical Stack Constraints

\* \*\*Language:\*\* Python 3.11+

\* \*\*Deep Learning:\*\* PyTorch, PyTorch Geometric (PyG).

\* \*\*Data Manipulation:\*\* Pandas, Scikit-Learn.

\* \*\*Graph Utilities:\*\* NetworkX (for initial checks), PyG `HeteroData` for modeling.



\## 3. Data Schema (Inferred)

The input dataframe (`df`) contains:

\* `transaction\_id`: Unique ID.

\* `date`: Timestamp.

\* `volume`: Transaction amount.

\* `direction`: Flow (Inbound/Outbound).

\* `customer\_name`: Name of the external party.

\* `bank\_name`: External bank name.

\* `bank\_account`: External account number.



---



\## 4. Implementation Plan



\### Phase 1: Feature Engineering \& Pre-processing

\*\*Objective:\*\* Prepare node features and handle the "Supernode" identifier.

1\.  \*\*Name Rarity Score:\*\*

&nbsp;   \* Calculate frequency of every `customer\_name`.

&nbsp;   \* Create feature: `rarity\_score = 1 - (count / total\_unique\_names)`.

&nbsp;   \* \*Logic:\* Differentiates common names (John Smith) from unique entities.

2\.  \*\*Entity Resolution:\*\*

&nbsp;   \* Treat `(customer\_name, bank\_account)` tuples as unique endpoints where necessary, but node aggregation should happen at the `customer\_name` level for the graph.



\### Phase 2: Heterogeneous Graph Construction (PyG)

\*\*Objective:\*\* Convert tabular data into a `HeteroData` object.

1\.  \*\*Node Types:\*\*

&nbsp;   \* `customer` (Features: `\[rarity\_score, total\_volume, txn\_count]`)

&nbsp;   \* `account` (Features: `\[bank\_name\_embedding, is\_domestic]`)

&nbsp;   \* `transaction` (Features: `\[amount, timestamp\_norm, direction]`)

2\.  \*\*Edge Types:\*\*

&nbsp;   \* (`customer`, `owns`, `account`)

&nbsp;   \* (`account`, `executed`, `transaction`)

&nbsp;   \* \*Note:\* Create reverse edges for message passing if using undirected logic.

3\.  \*\*Indexing:\*\*

&nbsp;   \* Map string IDs (`customer\_name`) to contiguous integers `0...N` using `torch.unique` or `LabelEncoder`.

&nbsp;   \* Store mappings to retrieve original IDs later.



\### Phase 3: Model Architecture (GAT)

\*\*Objective:\*\* Handle the "Supernode" skew using Attention mechanisms.

1\.  \*\*Model Class:\*\* `FraudGAT(torch.nn.Module)`

2\.  \*\*Layers:\*\* Use `HeteroConv` wrapping `GATConv`.

&nbsp;   \* \*Why GAT?\* To learn attention weights for neighbors. It prevents a customer with 100 accounts from having their signal "washed out" by averaging. The model learns to attend to the specific "risky" account among the 100.

3\.  \*\*Forward Pass:\*\*

&nbsp;   \* Layer 1: `GATConv` (Relu activation).

&nbsp;   \* Layer 2: `GATConv` (Linear output for embeddings).



\### Phase 4: Training (Self-Supervised Link Prediction)

\*\*Objective:\*\* Train the model to understand structural legitimacy without labels.

1\.  \*\*Strategy:\*\* Negative Sampling.

&nbsp;   \* \*\*Positive Edges:\*\* Real (`customer`, `owns`, `account`) links.

&nbsp;   \* \*\*Negative Edges:\*\* Randomly paired (`customer`, `random\_account`).

2\.  \*\*Loss Function:\*\* Binary Cross Entropy or Max Margin Loss.

&nbsp;   \* Maximize dot product similarity for real edges.

&nbsp;   \* Minimize similarity for negative edges.



\### Phase 5: Anomaly Detection (The Output)

1\.  \*\*Inference:\*\* Run the trained model to generate `customer\_embeddings` ($Z$).

2\.  \*\*Outlier Detection:\*\*

&nbsp;   \* Feed $Z$ into `sklearn.ensemble.IsolationForest`.

&nbsp;   \* \*Hypothesis:\* Customers with complex, mule-like structures will have embeddings that cluster differently from normal users.

3\.  \*\*Output:\*\* A list of `customer\_name` sorted by the Isolation Forest anomaly score.



---



\## 5. Coding Guidelines for the Assistant

\* \*\*Type Hinting:\*\* Use strict Python type hints (e.g., `def train(data: HeteroData) -> float:`).

\* \*\*Modularity:\*\* Keep the graph construction separate from the model definition.

\* \*\*Memory Efficiency:\*\* The dataset is 16M rows. Use `float32` instead of `float64`. Use generators or batched processing (`NeighborLoader`) if the graph exceeds GPU memory.

