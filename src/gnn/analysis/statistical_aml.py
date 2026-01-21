import pandas as pd
import numpy as np
from scipy.stats import entropy
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
from src.gnn.config.schema import Schema

class StatisticalAML:
    """
    Implements the Statistical & Clustering AML Strategy (Non-GNN).
    Separates users into 'Dense' (Track A) and 'Sparse' (Track B) populations
    and applies distinct anomaly detection models.
    """

    def split_population(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Phase 1: Population Splitting.
        Splits data into Track A (Supernodes) and Track B (Sparse).
        
        Args:
            df: Input transaction DataFrame.
            
        Returns:
            df_dense (Track A), df_sparse (Track B)
        """
        # Aggregation by customer
        # We need unique accounts and total transactions
        # Optimizing: Calculate these per customer once
        
        # Check if features are already aggregated? No, input is raw transactions.
        
        stats = df.groupby(Schema.CUSTOMER_NAME).agg(
            unique_accounts=(Schema.BANK_ACCOUNT, 'nunique'),
            txn_count=(Schema.TRANSACTION_ID, 'count')
        ).reset_index()
        
        # Logic: 
        # A: unique_accounts > 2 OR txn_count > 5
        # B: unique_accounts <= 2 AND txn_count <= 5 (Implicitly the rest)
        
        is_dense = (stats['unique_accounts'] > 2) | (stats['txn_count'] > 5)
        
        dense_customers = stats.loc[is_dense, Schema.CUSTOMER_NAME]
        sparse_customers = stats.loc[~is_dense, Schema.CUSTOMER_NAME]
        
        df_dense = df[df[Schema.CUSTOMER_NAME].isin(dense_customers)].copy()
        df_sparse = df[df[Schema.CUSTOMER_NAME].isin(sparse_customers)].copy()
        
        return df_dense, df_sparse

    # -------------------------------------------------------------------------
    # Phase 2: Track A - The "Supernode" Model (DBSCAN)
    # -------------------------------------------------------------------------

    def _calculate_entropy(self, x):
        """Helper to calculate Shannon entropy of value counts."""
        # Binning might be needed for continuous variables if we want entropy of distribution.
        # However, 'probability distribution of transaction amounts' usually implies
        # treating amounts as categories or binning them. 
        # For simplicity and speed on amounts, let's use value_counts(normalize=True).
        # If amounts are highly unique, entropy is high. If structuring (many $500), entropy is low.
        counts = x.value_counts(normalize=True)
        return entropy(counts)

    def engineer_features_track_a(self, df_dense: pd.DataFrame) -> pd.DataFrame:
        """
        Step 2.1: Advanced Feature Engineering for Dense Users.
        Returns a DataFrame with one row per customer.
        """
        # 1. Name Rarity (Global) - Assuming it's already computed or we compute here.
        # If not present, we compute it simply.
        total_unique_names = df_dense[Schema.CUSTOMER_NAME].nunique()
        name_counts = df_dense[Schema.CUSTOMER_NAME].value_counts()
        rarity_map = 1 - (name_counts / total_unique_names)
        
        # Group by Customer
        grouped = df_dense.groupby(Schema.CUSTOMER_NAME)
        
        # 2. Shannon Entropy of Amounts
        # Vectorized approach: It's hard to fully vectorize entropy across variable-sized groups without loops or apply.
        # We'll use apply, but optimized.
        # Alternatively, for huge data, we could bin and use matrix operations, but apply is safer for correctness here.
        amount_entropy = grouped[Schema.VOLUME].apply(self._calculate_entropy)
        
        # 3. Time Dispersion (Velocity Variance)
        # We need datetime objects
        if not pd.api.types.is_datetime64_any_dtype(df_dense[Schema.DATE]):
            df_dense[Schema.DATE] = pd.to_datetime(df_dense[Schema.DATE])
            
        def calc_time_std(x):
            if len(x) < 2: return 0.0
            return x.diff().dt.total_seconds().std()
            
        # Optimization: Sort once
        df_sorted = df_dense.sort_values([Schema.CUSTOMER_NAME, Schema.DATE])
        grouped_sorted = df_sorted.groupby(Schema.CUSTOMER_NAME)
        time_dispersion = grouped_sorted[Schema.DATE].apply(calc_time_std).fillna(0)
        
        # 4. Bank Diversity Ratio
        # unique_banks / unique_accounts
        n_banks = grouped[Schema.BANK_NAME].nunique()
        n_accounts = grouped[Schema.BANK_ACCOUNT].nunique()
        bank_diversity = n_banks / n_accounts
        
        # Combine
        features = pd.DataFrame({
            'entropy': amount_entropy,
            'time_dispersion': time_dispersion,
            'bank_diversity': bank_diversity,
            'rarity_score': rarity_map
        }).fillna(0) # Fill NaNs (e.g. if std is nan)
        
        return features

    def plot_k_distance(self, X: np.ndarray, k: int = 5):
        """
        Helper to plot k-distance graph for finding DBSCAN eps.
        """
        neigh = NearestNeighbors(n_neighbors=k)
        nbrs = neigh.fit(X)
        distances, indices = nbrs.kneighbors(X)
        
        # Sort distance to k-th nearest neighbor (column k-1)
        distances = np.sort(distances[:, k-1], axis=0)
        
        plt.figure(figsize=(10, 6))
        plt.plot(distances)
        plt.title(f"K-Distance Graph (k={k})")
        plt.ylabel(f"{k}-th Nearest Neighbor Distance")
        plt.xlabel("Points sorted by distance")
        plt.grid(True)
        plt.show()

    def run_track_a(self, feature_df: pd.DataFrame, eps: float = 0.5, min_samples: int = 5) -> pd.DataFrame:
        """
        Step 2.2: Modeling (DBSCAN)
        """
        # Preprocessing
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(feature_df)
        
        # DBSCAN
        db = DBSCAN(eps=eps, min_samples=min_samples)
        labels = db.fit_predict(X_scaled)
        
        # Result DataFrame
        results = feature_df.copy()
        results['cluster'] = labels
        results['is_anomaly'] = labels == -1
        
        # Calculate 'score' for merging (Distance to nearest core point?)
        # DBSCAN doesn't give a score natively. 
        # We can map -1 to 100 (high risk) and others to 0-50 based on distance?
        # For simplicity per prompt: "Normalize scores... distance to nearest cluster core"
        # Since sklearn DBSCAN doesn't expose centroids easily, we'll assign:
        # Outliers (-1) = 100
        # Clustered = 0 (baseline)
        
        results['risk_score'] = np.where(labels == -1, 100, 0)
        results['reason_code'] = np.where(labels == -1, "Cluster_Outlier_Track_A", "Normal")
        
        return results

    # -------------------------------------------------------------------------
    # Phase 3: Track B - The "Sparse" Model (Peer Profiling)
    # -------------------------------------------------------------------------

    def engineer_features_track_b(self, df_sparse: pd.DataFrame) -> pd.DataFrame:
        """
        Step 3.1: Contextual Feature Engineering for Sparse Users.
        Returns a DataFrame on TRANSACTION level (since users have little history).
        """
        # 1. Bank Risk Context
        # Calculate global stats per bank (Mean, Std of Volume)
        bank_stats = df_sparse.groupby(Schema.BANK_NAME)[Schema.VOLUME].agg(['mean', 'std']).to_dict(orient='index')
        
        # Map back to dataframe
        # Handling NaNs in std (single transaction banks) -> default to 1.0 or 0.0 to avoid division by zero
        # Better: Global mean imputation for missing
        
        def get_z_score(row):
            b_name = row[Schema.BANK_NAME]
            vol = row[Schema.VOLUME]
            if b_name in bank_stats:
                stats = bank_stats[b_name]
                mu = stats['mean']
                sigma = stats['std']
                if pd.isna(sigma) or sigma == 0:
                    return 0.0 # Cannot compute Z-score
                return (vol - mu) / sigma
            return 0.0

        # Vectorized mapping is faster
        # Create temp DF for mapping
        stats_df = df_sparse.groupby(Schema.BANK_NAME)[Schema.VOLUME].agg(['mean', 'std'])
        stats_df['std'] = stats_df['std'].replace(0, 1e-9) # Avoid div/0
        
        merged = df_sparse.merge(stats_df, on=Schema.BANK_NAME, how='left')
        merged['z_score'] = (merged[Schema.VOLUME] - merged['mean']) / merged['std']
        merged['z_score'] = merged['z_score'].fillna(0)
        
        # Name Rarity (recalc for sparse set or reuse global if available)
        total_unique = merged[Schema.CUSTOMER_NAME].nunique()
        counts = merged[Schema.CUSTOMER_NAME].value_counts()
        rarity_map = 1 - (counts / total_unique)
        
        merged['rarity_score'] = merged[Schema.CUSTOMER_NAME].map(rarity_map)
        
        # Features: [Volume, Z_Score, Rarity]
        # We return the whole merged DF to keep IDs
        return merged

    def run_track_b(self, df_prepared: pd.DataFrame, contamination: float = 0.001) -> pd.DataFrame:
        """
        Step 3.2: Modeling (Isolation Forest)
        """
        features = df_prepared[[Schema.VOLUME, 'z_score', 'rarity_score']].fillna(0)
        
        iso = IsolationForest(contamination=contamination, n_estimators=100, random_state=42)
        preds = iso.fit_predict(features) # 1 for inlier, -1 for outlier
        scores = iso.decision_function(features) # Average anomaly score
        
        results = df_prepared.copy()
        results['anomaly_flag'] = preds
        results['raw_score'] = scores # Lower is more anomalous
        
        # Normalize Score to 0-100
        # Decision function: lower = more abnormal. Range roughly -0.5 to 0.5
        # We want High Score = High Risk.
        # Simple inversion and scaling:
        # normalized = (max - score) / (max - min) * 100 ? 
        # Or just ranking.
        # Let's use simple min-max scaling inverted.
        min_s = scores.min()
        max_s = scores.max()
        if max_s != min_s:
            norm_score = (max_s - scores) / (max_s - min_s) * 100
        else:
            norm_score = 0
            
        results['risk_score'] = norm_score
        results['reason_code'] = np.where(preds == -1, "Peer_Outlier_Track_B", "Normal")
        
        return results

    # -------------------------------------------------------------------------
    # Phase 4: Merging
    # -------------------------------------------------------------------------

    def merge_results(self, res_a: pd.DataFrame, res_b: pd.DataFrame) -> pd.DataFrame:
        """
        Combines results into a unified risk list.
        """
        # Track A is at Customer Level
        out_a = res_a.reset_index() # customer_name is index
        out_a = out_a[[Schema.CUSTOMER_NAME, 'risk_score', 'reason_code']]
        out_a['source_model'] = "Track_A"
        
        # Track B is at Transaction Level -> Aggregate to Customer
        # Take max risk score per customer?
        out_b = res_b.groupby(Schema.CUSTOMER_NAME).agg({
            'risk_score': 'max',
            'reason_code': lambda x: x.mode()[0] if not x.mode().empty else 'Normal' 
        }).reset_index()
        out_b['source_model'] = "Track_B"
        
        final_df = pd.concat([out_a, out_b], ignore_index=True)
        return final_df.sort_values('risk_score', ascending=False)
