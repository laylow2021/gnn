import pandas as pd
import numpy as np
import hashlib
from typing import Tuple, List
from scipy.spatial.distance import cosine
from src.gnn.config.schema import Schema

class AdvancedFeatureEngineer:
    """
    Implements 'Peak Vector' and 'State Shift' feature engineering.
    """

    def __init__(self):
        pass

    def _generate_account_id(self, bank_name: str, account_number: str) -> str:
        raw_id = f"{str(bank_name).strip().upper()}_{str(account_number).strip()}"
        return hashlib.sha256(raw_id.encode('utf-8')).hexdigest()

    def _calculate_peak_features(self, group) -> pd.Series:
        """
        Calculates features on the Top 10% largest transactions.
        """
        # Sort by volume desc
        txns = group.sort_values(Schema.VOLUME, ascending=False)
        top_10_pct_n = max(1, int(len(txns) * 0.1))
        peak_txns = txns.head(top_10_pct_n)

        # 1. Peak Flow Through (Ratio of Outbound Volume / Total Volume in Peak)
        # If mainly inbound, ratio close to 0. Mainly outbound, close to 1. 
        # Or (In - Out) / (In + Out)?
        # Let's use Outbound / Total for simplicity of "Flow Through" concept.
        total_vol = peak_txns[Schema.VOLUME].sum()
        if total_vol == 0:
            flow_through = 0.0
        else:
            out_vol = peak_txns[peak_txns[Schema.DIRECTION] == 'Outbound'][Schema.VOLUME].sum()
            flow_through = out_vol / total_vol

        # 2. Peak Roundness (% of amounts ending in .00)
        # Checking if float is integer.
        is_round = (peak_txns[Schema.VOLUME] % 1 == 0)
        roundness = is_round.mean()

        # 3. Peak Entropy (Time consistency)
        # Variance of time deltas in seconds.
        if len(peak_txns) < 2:
            entropy = 0.0
        else:
            # Sort by date for delta calculation
            sorted_dates = peak_txns.sort_values(Schema.DATE)[Schema.DATE]
            deltas = sorted_dates.diff().dt.total_seconds().dropna()
            # Normalize variance? Or just log variance to handle scale.
            # Using Coefficient of Variation of deltas might be better, or just std dev.
            # Let's use Log(StdDev + 1)
            entropy = np.log1p(deltas.std()) if len(deltas) > 0 else 0.0

        return pd.Series({
            Schema.PEAK_FLOW_THROUGH: flow_through,
            Schema.PEAK_ROUNDNESS: roundness,
            Schema.PEAK_ENTROPY: entropy
        })

    def _calculate_shift_score(self, group, current_date=None) -> float:
        """
        Calculates Cosine Distance between Historic behavior and Burst behavior (last 7 days).
        """
        if current_date is None:
            current_date = group[Schema.DATE].max()
        
        cutoff_date = current_date - pd.Timedelta(days=7)
        
        burst = group[group[Schema.DATE] > cutoff_date]
        historic = group[group[Schema.DATE] <= cutoff_date]
        
        if len(historic) == 0 or len(burst) == 0:
            return 0.0 # No shift possible if no history or no burst
            
        # Define simple vector for comparison: [Mean Vol, Count, % Outbound]
        def get_mini_vec(df_subset):
            mean_vol = df_subset[Schema.VOLUME].mean()
            count = len(df_subset)
            pct_out = (df_subset[Schema.DIRECTION] == 'Outbound').mean()
            return np.array([mean_vol, count, pct_out])

        v_hist = get_mini_vec(historic)
        v_burst = get_mini_vec(burst)
        
        # Handle zero vectors / NaNs
        if np.all(v_hist == 0) or np.all(v_burst == 0):
            return 0.0
            
        # Cosine Distance = 1 - Cosine Similarity
        # Range [0, 2]. 0 = Identical. 
        # Use fillna for safety
        try:
            dist = cosine(v_hist, v_burst)
            if np.isnan(dist): return 0.0
            return dist
        except:
            return 0.0

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies feature engineering to raw transaction DataFrame.
        Aggregates by ACCOUNT initially.
        """
        df = df.copy()
        
        # Pre-calc Account ID
        df[Schema.ACCOUNT_ID] = df.apply(
            lambda x: self._generate_account_id(x[Schema.BANK_NAME], x[Schema.BANK_ACCOUNT]), 
            axis=1
        )
        
        # Ensure Date is datetime
        df[Schema.DATE] = pd.to_datetime(df[Schema.DATE])
        
        # Group by Account to calc features
        # We need Peak Features and Shift Score per Account.
        # Note: In Phase 1 we might aggregate by Name, but features are best calc'd at Account level first?
        # The prompt says: "Phase 1: ... Regular: Aggregate transactions to Customer_Name. Output: Name_Vector (Includes Peak Features...)"
        # So we should group by CUSTOMER_NAME?
        # But for Super Nodes, we need Account level granularity.
        # Strategy: Calculate at Account Level first, then aggregation logic in Resolution phase will handle the merge.
        
        # Actually, let's calc at Account level, and also pass raw info.
        # Wait, regular nodes aggregate Txns to Name.
        # So features should probably be calculated per "Entity Candidate".
        
        # Let's perform calculation at ACCOUNT level for now, as that's the base unit.
        
        results = []
        current_date = df[Schema.DATE].max()
        
        grouped = df.groupby(Schema.ACCOUNT_ID)
        
        for acc_id, group in grouped:
            peak_feats = self._calculate_peak_features(group)
            shift_score = self._calculate_shift_score(group, current_date)
            
            # Basic info
            name = group[Schema.CUSTOMER_NAME].mode()[0] # Most frequent name
            bank = group[Schema.BANK_NAME].iloc[0]
            acct = group[Schema.BANK_ACCOUNT].iloc[0]
            
            row = peak_feats.to_dict()
            row[Schema.SHIFT_SCORE] = shift_score
            row[Schema.ACCOUNT_ID] = acc_id
            row[Schema.CUSTOMER_NAME] = name
            row[Schema.BANK_NAME] = bank
            row[Schema.BANK_ACCOUNT] = acct
            row['txn_count'] = len(group)
            
            results.append(row)
            
        return pd.DataFrame(results)
