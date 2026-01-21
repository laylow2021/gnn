# PROJECT CONTEXT: Statistical & Clustering AML Strategy (Non-GNN)

## 1. Project Overview
**Goal:** Implement a baseline unsupervised anomaly detection system for PayPal transactions, specifically handling the extreme data skew between "Sparse" users (1-2 transactions) and "Dense" users (Supernodes/100+ accounts).
**Data Scale:** 16 Million records, 40-day window.
**Methodology:** Split population into two tracks and apply distinct statistical models to each.

## 2. Execution Pipeline

### Phase 1: Population Splitting
**Objective:** Bifurcate the dataset to avoid model noise.
1.  **Aggregation:** Group by `customer_name`.
2.  **Metric:** Count unique `bank_account_number` and total `transaction_count`.
3.  **Logic:**
    * **DataFrame A (Dense/Supernodes):** Users with `unique_accounts > 2` OR `transaction_count > 5`.
    * **DataFrame B (Sparse/One-Shots):** Users with `unique_accounts <= 2` AND `transaction_count <= 5`.

---

### Phase 2: Track A - The "Supernode" Model (Clustering)
**Target:** Detect structural money laundering (Layering, Mule Rings) in high-activity users.
**Model:** DBSCAN (Density-Based Spatial Clustering of Applications with Noise).

#### Step 2.1: Advanced Feature Engineering
Generate a single vector per customer in DataFrame A.
1.  **Shannon Entropy of Amounts:**
    * Calculate probability distribution of transaction amounts for the user.
    * Compute Entropy: $-\sum p(x) \log p(x)$.
    * *Hypothesis:* Low entropy = Machine-like/Structuring (always $500). High entropy = Organic.
2.  **Time Dispersion (Velocity Variance):**
    * Calculate standard deviation of time deltas between transactions.
3.  **Bank Diversity Ratio:**
    * `unique_banks_count` / `unique_accounts_count`.
    * *Hypothesis:* Ratio ~ 1.0 implies mule network (1 account per bank). Ratio Low implies internal transfers.
4.  **Name Rarity Score:** (Reuse from general plan).

#### Step 2.2: Modeling (DBSCAN)
1.  **Preprocessing:** Scale features using `StandardScaler` (Critical for DBSCAN).
2.  **Hyperparameters:**
    * `eps`: Determine via k-distance graph (knee method).
    * `min_samples`: Set to roughly `ln(n)`.
3.  **Output:**
    * Cluster `-1`: Outliers (The primary AML suspects).
    * Cluster `0, 1...`: Normal behavior groups (e.g., "Standard Payroll").

---

### Phase 3: Track B - The "Sparse" Model (Peer Profiling)
**Target:** Detect single-transaction anomalies (e.g., a massive wire transfer from a usually low-value bank).
**Model:** Isolation Forest.

#### Step 3.1: Contextual Feature Engineering
Since these users lack history, compare them to their "Peer Group" (The Bank).
1.  **Bank Risk Context:**
    * Calculate global stats for each `bank_name` (e.g., `Mean_Txn_Amt`, `Std_Txn_Amt`).
2.  **Z-Score Normalization:**
    * For every transaction, calculate: $Z = \frac{\text{Txn\_Amount} - \text{Bank\_Mean}}{\text{Bank\_Std}}$.
3.  **Input Vector:** `[Txn_Amount, Z_Score, Name_Rarity_Score]`.

#### Step 3.2: Modeling (Isolation Forest)
1.  **Algorithm:** `sklearn.ensemble.IsolationForest`.
2.  **Settings:**
    * `contamination`: Set strictly (e.g., `0.001` or `0.005`) to flag only extreme deviations.
    * `n_estimators`: 100.
3.  **Output:** Anomaly Score (Lower is more anomalous).

---

### Phase 4: Merging & Reporting
**Objective:** Combine results into a unified risk alert list.
1.  **Normalization:** Normalize scores from DBSCAN (distance to nearest cluster core) and Isolation Forest (decision function) to a 0-100 scale.
2.  **Final Output Schema:**
    * `customer_name`
    * `risk_score` (0-100)
    * `reason_code` (e.g., "High_Entropy_Supernode" or "Peer_Group_Outlier")
    * `source_model` ("Track_A" or "Track_B")

---

## 5. Coding Instructions for Assistant
* **Vectorization:** Do not use Python loops for entropy calculation. Use `scipy.stats.entropy` or `numpy` vectorization.
* **Handling Categoricals:** Do not One-Hot Encode `bank_name` for Track B (dimensionality curse). Use Target Encoding or Statistical Aggregates (Mean/Std) as described in Step 3.1.
* **DBSCAN Tuning:** Include a helper function to plot the k-distance graph to help the user find the optimal `eps` value.