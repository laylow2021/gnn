import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from src.gnn.config.schema import Schema

class AnomalyDetector:
    """
    Phase 2: Anomaly Detection (The Filter)
    """
    
    def __init__(self, contamination: float = 0.05):
        self.contamination = contamination
        self.model = IsolationForest(contamination=contamination, random_state=42)

    def detect(self, resolved_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregates to Entity Level and runs Isolation Forest.
        Returns DataFrame of Entities with Risk Scores.
        """
        # 1. Aggregate to Entity Level
        # We take the mean of the behavioral features.
        # For Shift Score, maybe Max is better? "If any account shifted, the entity is risky."
        feature_cols = [Schema.PEAK_FLOW_THROUGH, Schema.PEAK_ROUNDNESS, Schema.PEAK_ENTROPY, Schema.SHIFT_SCORE]
        
        entity_df = resolved_df.groupby(Schema.RESOLVED_ENTITY_ID)[feature_cols].mean()
        
        # 2. Train Model
        # Fill NaNs
        X = entity_df.fillna(0)
        
        self.model.fit(X)
        
        # 3. Score
        # decision_function: lower is more anomalous. 
        # We want a Risk Score where Higher = Riskier.
        # So we negate it or map it.
        raw_scores = self.model.decision_function(X)
        risk_scores = -raw_scores # Higher is more anomalous
        
        # Predictions (-1 = outlier/risky, 1 = inlier)
        preds = self.model.predict(X)
        
        entity_df[Schema.RISK_SCORE] = risk_scores
        entity_df['is_high_risk'] = (preds == -1)
        
        # Merge back high-level info (Name)
        # Just take the first name found for this entity
        name_map = resolved_df.groupby(Schema.RESOLVED_ENTITY_ID)[Schema.CUSTOMER_NAME].first()
        entity_df[Schema.CUSTOMER_NAME] = name_map
        
        return entity_df
