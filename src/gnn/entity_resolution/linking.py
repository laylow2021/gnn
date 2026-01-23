import pandas as pd
import numpy as np
import hashlib
from scipy.spatial.distance import cosine
from src.gnn.config.schema import Schema

class SyndicateLinker:
    """
    Phase 3: Mirrored Linking (Cross-Entity)
    """
    
    def __init__(self, similarity_threshold: float = 0.98, volume_tolerance: float = 0.1):
        self.sim_threshold = similarity_threshold
        self.vol_tolerance = volume_tolerance

    def _get_direction_vector(self, df_subset) -> np.array:
        """
        Calculates vector for a set of transactions (In or Out).
        Vector: [Mean_Log_Volume, Std_Log_Volume, Count_Log, Hour_Mean]
        """
        if len(df_subset) == 0:
            return None
            
        vols = np.log1p(df_subset[Schema.VOLUME])
        
        # Hour of day (cyclic? keep simple mean for now or sin/cos)
        # Using simple mean for MVP
        hours = df_subset[Schema.DATE].dt.hour
        
        vec = np.array([
            vols.mean(),
            vols.std() if len(vols) > 1 else 0,
            np.log1p(len(df_subset)),
            hours.mean()
        ])
        return np.nan_to_num(vec)

    def link(self, high_risk_entities: pd.DataFrame, raw_df: pd.DataFrame, account_map: pd.DataFrame) -> pd.DataFrame:
        """
        Links entities that have mirrored flow.
        Args:
            high_risk_entities: DF from AnomalyDetector (index=Entity_ID).
            raw_df: Raw transactions.
            account_map: DF mapping Account_ID -> Resolved_Entity_ID (from Phase 1).
        """
        # Filter raw_df to only high risk accounts
        # 1. Map raw_df to Entity ID
        # We need Account ID in raw_df first.
        # Assuming raw_df has been processed to have account_id? Or we generate it.
        # Ideally we pass a raw_df that already has 'account_id' from feature engineering step.
        # We will assume caller handles this or we re-generate.
        # Let's assume we re-generate or join.
        
        # Optimization: Filter Account Map to high risk entities
        hr_ids = high_risk_entities[high_risk_entities['is_high_risk']].index
        hr_accounts = account_map[account_map[Schema.RESOLVED_ENTITY_ID].isin(hr_ids)]
        
        if len(hr_accounts) == 0:
            return pd.DataFrame(columns=['Source', 'Target', 'Syndicate_ID'])

        # Join raw_df with hr_accounts
        # We need to match on Account ID.
        # If raw_df doesn't have it, we must generate it. 
        # But features.py generated it. We can assume raw_df passed here might be the one processed by features?
        # Let's assume raw_df has columns bank_name, bank_account.
        
        # Helper map (Bank, Account) -> EntityID
        # Vectorized lookup is tricky without composite key.
        # Let's assume the user passes a raw_df that has 'account_id' added (output of features.fit_transform's internal logic).
        # Actually, features.py returned a feature_df (one row per account).
        
        # We need to construct the vectors.
        # Let's group raw_df by (Bank, Account) and merge with hr_accounts.
        # Or simpler: Re-hash in loop? Slow.
        # Let's assume raw_df has 'account_id' or we construct it quickly.
        
        # For the demo, I will assume we can iterate entities and pull their data.
        
        # Build Entity Profiles (V_Out, V_In, Total_Vol)
        profiles = {}
        
        # Pre-process raw_df for speed
        # Add account_id if missing
        if Schema.ACCOUNT_ID not in raw_df.columns:
             # This is slow, but necessary if missing
             raw_df[Schema.ACCOUNT_ID] = raw_df.apply(lambda x: f"{str(x[Schema.BANK_NAME]).strip().upper()}_{str(x[Schema.BANK_ACCOUNT]).strip()}", axis=1).apply(lambda x: hashlib.sha256(x.encode()).hexdigest())

        # Filter raw to HR accounts
        merged = raw_df[raw_df[Schema.ACCOUNT_ID].isin(hr_accounts[Schema.ACCOUNT_ID])]
        merged = merged.merge(hr_accounts[[Schema.ACCOUNT_ID, Schema.RESOLVED_ENTITY_ID]], on=Schema.ACCOUNT_ID)
        
        grouped = merged.groupby(Schema.RESOLVED_ENTITY_ID)
        
        for ent_id, group in grouped:
            out_txns = group[group[Schema.DIRECTION] == 'Outbound']
            in_txns = group[group[Schema.DIRECTION] == 'Inbound']
            
            profiles[ent_id] = {
                'v_out': self._get_direction_vector(out_txns),
                'v_in': self._get_direction_vector(in_txns),
                'vol_out': out_txns[Schema.VOLUME].sum(),
                'vol_in': in_txns[Schema.VOLUME].sum()
            }
            
        # Pairwise Comparison
        links = []
        ent_ids = list(profiles.keys())
        
        for i, id_a in enumerate(ent_ids):
            prof_a = profiles[id_a]
            if prof_a['v_out'] is None: continue
            
            for j, id_b in enumerate(ent_ids):
                if i == j: continue
                prof_b = profiles[id_b]
                if prof_b['v_in'] is None: continue
                
                # Check 1: Volume Match
                # Abs(Vol_A_Out - Vol_B_In) < 10%
                vol_a = prof_a['vol_out']
                vol_b = prof_b['vol_in']
                
                if vol_a == 0 or vol_b == 0: continue
                
                diff = abs(vol_a - vol_b) / max(vol_a, vol_b)
                if diff > self.vol_tolerance:
                    continue
                    
                # Check 2: Cosine Similarity
                # Dist = cosine(u, v). Sim = 1 - Dist.
                # cosine returns distance [0, 2].
                dist = cosine(prof_a['v_out'], prof_b['v_in'])
                sim = 1.0 - dist
                
                if sim > self.sim_threshold:
                    links.append({
                        'Source': id_a,
                        'Target': id_b,
                        'Syndicate_ID': f"SYN_{min(id_a, id_b)[:6]}_{max(id_a, id_b)[:6]}",
                        'Similarity': sim,
                        'Vol_Diff_Pct': diff
                    })
                    
        return pd.DataFrame(links)
