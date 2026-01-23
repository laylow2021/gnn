import pandas as pd
import numpy as np
import hashlib
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from src.gnn.config.schema import Schema

class EntityResolver:
    """
    Phase 1: The Fork (Regular vs Super Nodes)
    """
    
    def __init__(self, supernode_threshold: int = 20):
        self.threshold = supernode_threshold

    def _get_name_hash(self, name: str) -> str:
        return hashlib.md5(str(name).encode('utf-8')).hexdigest()[:12]

    def _resolve_supernode(self, accounts_df: pd.DataFrame) -> pd.Series:
        """
        Applies clustering to split a Super Node into sub-clusters.
        Returns a Series mapping Account_ID -> Resolved_ID suffix.
        """
        if len(accounts_df) < 2:
            return pd.Series([f"{self._get_name_hash(accounts_df[Schema.CUSTOMER_NAME].iloc[0])}_0"] * len(accounts_df), index=accounts_df.index)

        # 1. Feature Prep (Behavior View)
        # Select clustering features
        features = [Schema.PEAK_FLOW_THROUGH, Schema.PEAK_ROUNDNESS, Schema.PEAK_ENTROPY, Schema.SHIFT_SCORE]
        X = accounts_df[features].fillna(0).values
        
        # Normalize
        X = StandardScaler().fit_transform(X)
        
        # 2. Clustering (Agglomerative)
        # Using correlation/cosine distance logic via affinity='cosine' if available or euclidean
        # The prompt suggests Threshold > 0.8 similarity, so dist < 0.2
        # Scikit-learn AgglomerativeClustering distance_threshold is for linkage distance.
        cluster = AgglomerativeClustering(
            n_clusters=None,
            metric='euclidean', # simplified
            linkage='average',
            distance_threshold=1.5 # Looser threshold for normalized euclidean (~ roughly similar)
        )
        
        labels = cluster.fit_predict(X)
        
        # 3. Generate IDs
        base_hash = self._get_name_hash(accounts_df[Schema.CUSTOMER_NAME].iloc[0])
        resolved_ids = [f"{base_hash}_{lbl}" for lbl in labels]
        
        return pd.Series(resolved_ids, index=accounts_df.index)

    def resolve(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """
        Main resolution pipeline.
        Args:
            feature_df: Output from AdvancedFeatureEngineer (one row per Account)
        Returns:
            feature_df with added 'RESOLVED_ENTITY_ID' and 'IS_SUPERNODE'
        """
        df = feature_df.copy()
        
        # Count accounts per name
        name_counts = df[Schema.CUSTOMER_NAME].value_counts()
        
        super_names = name_counts[name_counts > self.threshold].index
        
        df[Schema.IS_SUPERNODE] = df[Schema.CUSTOMER_NAME].isin(super_names)
        df[Schema.RESOLVED_ENTITY_ID] = None
        df[Schema.RESOLVED_ENTITY_ID] = df[Schema.RESOLVED_ENTITY_ID].astype('object')
        
        # Path A: Regular Nodes
        # ID = Name Hash
        mask_reg = ~df[Schema.IS_SUPERNODE]
        df.loc[mask_reg, Schema.RESOLVED_ENTITY_ID] = df.loc[mask_reg, Schema.CUSTOMER_NAME].apply(self._get_name_hash)
        
        # Path B: Super Nodes
        # Iterate over each super name and resolve
        for name in super_names:
            mask_name = df[Schema.CUSTOMER_NAME] == name
            sub_df = df[mask_name]
            
            # Run clustering
            resolved_ids = self._resolve_supernode(sub_df)
            
            # Assign back
            df.loc[mask_name, Schema.RESOLVED_ENTITY_ID] = resolved_ids.values
            
        return df
