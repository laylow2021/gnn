# ARCHITECTURE: Mirrored Entity Resolution & AML System

## 1. Executive Summary
**Objective:** Detect financial crime risks across 16M transactions by combining "Split-Track" Entity Resolution with "Mirrored" Vector Linking.
**Core Logic:**
1.  **Split:** Regular vs. Super Nodes.
2.  **Resolve:**
    * *Regular:* Aggregate by Name.
    * *Super Node:* **Multi-View Clustering** (Deterministic) OR **Graph Neural Network** (Probabilistic).
3.  **Detect:** Identify high-risk entities using "State Shift" and "Peak Features."
4.  **Link:** Connect seemingly unrelated high-risk entities by matching their "Mirrored" transaction vectors.

---

## 2. Advanced Feature Engineering

### A. The "Peak" Vector (Noise Filtering)
* **Goal:** Ignore the 90% of "coffee money" noise. Focus only on the laundering signal.
* **Logic:** Calculate features *only* on the **Top 10% Largest Transactions** for each entity.
* **Features:**
    * `Peak_Flow_Through`: Flow-through ratio of large txns.
    * `Peak_Roundness`: % of large txns ending in .00.
    * `Peak_Entropy`: Time consistency of large txns.

### B. The "State Shift" Score (Sleeper Detection)
* **Goal:** Catch "Bust-Out" accounts that were dormant and suddenly woke up.
* **Logic:**
    * `V_Historic`: Behavior over lifetime (excluding last 7 days).
    * `V_Burst`: Behavior over last 7 days.
    * **Feature:** `Shift_Score = Cosine_Distance(V_Historic, V_Burst)`.

---

## 3. Phase 1: The Fork (Resolution)

### Path A: Regular Nodes (Count <= 20)
* **Action:** Aggregate transactions to `Customer_Name`.
* **Output:** `Name_Vector` (Includes Peak Features + Shift Score).

### Path B: Super Nodes (Count > 20)
* **Target:** Common names (e.g., "John Smith") where simple aggregation fails.
* **Action:** Choose one of the two resolution strategies below.

#### **Option 1: Multi-View Hierarchical Clustering (Recommended MVP)**
* **Type:** Deterministic / Rule-Based.
* **Pros:** Highly explainable, precise for known typologies.
* **Logic:** Calculate a **Pairwise Similarity Matrix** based on 3 views:
    1.  **Flow View ($W=0.5$):** Matches if `Abs(Time_A - Time_B) < 3 mins` AND `Abs(Amount_A - Amount_B) < 1%`. (Captures "Ping-Pong" layering).
    2.  **Bank View ($W=0.3$):** Matches if `Bank_Name` is identical AND `Levenshtein(Acct_ID) < 2`. (Captures Household).
    3.  **Behavior View ($W=0.2$):** Matches if `Cosine_Sim(Peak_Vector) > 0.98`. (Captures Clones).
* **Algorithm:** Agglomerative Clustering (Threshold > 0.8).

#### **Option 2: Graph Neural Network (Advanced Upgrade)**
* **Type:** Probabilistic / Deep Learning.
* **Pros:** Can detect non-linear, fuzzy patterns and evasion (e.g., variable time delays).
* **Step 1: Inferred Graph Construction**
    * Since direct edges don't exist, build a **Similarity Graph** (k-NN) for the Super Node population.
    * **Nodes:** Accounts.
    * **Edges:** Create an edge if *any* similarity exists (Time, Amount, or Bank). This creates a "Noisy Graph."
* **Step 2: The Model (GATv2)**
    * **Architecture:** Graph Attention Network (GATv2) + Jumping Knowledge (JK).
    * **Task:** Link Prediction / Node Embedding.
    * **Mechanism:** The Attention mechanism learns to down-weight the "Noise Edges" (coincidences) and up-weight the "Signal Edges" (real hidden links).
* **Step 3: Clustering**
    * Run **DBSCAN** or **K-Means** on the output embeddings ($Z$) to find the true sub-clusters.

---

## 4. Phase 2: Anomaly Detection (The Filter)
**Input:** Unified list of Vectors (Regular + Resolved Super Node Clusters).
**Algorithm:** Isolation Forest.
**Action:** Select the top **5% High-Risk Candidates** based on `Peak_Features` and `Shift_Score`.
* *Crucial:* This step automatically captures the high-frequency "A->PayPal" and "PayPal->B" actors because their high velocity/entropy triggers the anomaly score.

---

## 5. Phase 3: "Mirrored" Linking (Cross-Entity)
**Goal:** Link "Sender A" to "Receiver B" by spotting identical (mirrored) anomaly patterns between different Resolved Entities.
**Input:** Only the **High-Risk Candidates** from Phase 2.

**The "Twin Vector" Logic:**
1.  **Construct Directional Vectors:**
    * For every Suspicious Entity, create two sub-vectors:
        * `V_Out`: Characteristics of Outgoing money.
        * `V_In`: Characteristics of Incoming money.
2.  **Comparison:**
    * Compare `Entity_A (V_Out)` vs `Entity_B (V_In)`.
3.  **Match Criteria:**
    * IF `Cos_Similarity(A_Out, B_In) > 0.98` AND `Abs(Vol_A - Vol_B) < 10%`.
4.  **Verdict:**
    * **LINK:** Merge into `Syndicate_ID`. A is the Source, B is the Sink.

---

## 6. Final Output Table

| Entity ID | Type | Risk Flag | Detection Method |
| :--- | :--- | :--- | :--- |
| **Name_X** | Regular | **Bust-Out** | **State Shift:** Sudden spike in `Peak_Entropy`. |
| **Entity_JohnSmith_01** | Super Node | **Layering** | **Path B (Option 1):** Resolved via Self-Transfer logic (Ping-Pong). |
| **Entity_JohnSmith_02** | Super Node | **Evasive Ring** | **Path B (Option 2):** GNN embedding clustered noisy temporal links. |
| **Syndicate_77** | **LINKED** | **Laundering Ring** | **Mirrored Match:** `Name_Alice` (Out-Vector) matches `Entity_Bob` (In-Vector). |