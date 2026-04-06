import pandas as pd
import numpy as np
import datetime
from typing import List, Tuple

def generate_transactions(
    num_customers: int = 1000, 
    num_days: int = 30, 
    avg_tx_per_day: int = 5,
    anomaly_ratio: float = 0.05
) -> pd.DataFrame:
    """Generate a simulated daily transaction ledger mimicking a FinTech platform."""
    np.random.seed(42)
    start_date = datetime.date(2026, 1, 1)
    date_range = [start_date + datetime.timedelta(days=d) for d in range(num_days)]
    
    # Regular customers and random noise
    num_tx = num_customers * num_days * avg_tx_per_day
    customer_ids = np.random.randint(0, num_customers, num_tx)
    dates = np.random.choice(date_range, num_tx)
    amounts = np.abs(np.random.normal(500, 200, num_tx))
    directions = np.random.choice(['IN', 'OUT'], num_tx)
    
    tx_df = pd.DataFrame({
        'transaction_id': range(num_tx),
        'customer_id': customer_ids,
        'date': pd.to_datetime(dates),
        'amount': np.round(amounts, 2),
        'direction': directions,
        'label': 0  # 0: Normal
    })
    
    # Inject AML Typologies
    tx_df = _inject_pass_through(tx_df, int(num_customers * anomaly_ratio * 0.2), date_range, num_customers)
    tx_df = _inject_repeated_layering(tx_df, int(num_customers * anomaly_ratio * 0.2), date_range, num_customers)
    tx_df = _inject_persistent_link(tx_df, int(num_customers * anomaly_ratio * 0.1), date_range, num_customers)
    tx_df = _inject_daily_self_passthrough(tx_df, int(num_customers * anomaly_ratio * 0.1), date_range, num_customers)
    tx_df = _inject_fan_in(tx_df, int(num_customers * anomaly_ratio * 0.4), date_range, num_customers)
    
    return tx_df.sort_values('date').reset_index(drop=True)

def _inject_persistent_link(df: pd.DataFrame, num_typologies: int, date_range: List[datetime.date], num_customers: int) -> pd.DataFrame:
    """Inject a pair with 5 identical transactions over 5 consecutive days. Label: 5"""
    new_rows = []
    base_id = df['transaction_id'].max() + 1
    
    for _ in range(num_typologies):
        source_id = np.random.randint(0, num_customers)
        target_id = np.random.randint(0, num_customers)
        while target_id == source_id:
            target_id = np.random.randint(0, num_customers)
            
        # Constant amount for all 5 days
        amount = np.round(np.random.uniform(1000, 5000), 2)
        start_day_idx = np.random.randint(0, len(date_range) - 6)
        
        for d in range(5):
            current_date = date_range[start_day_idx + d]
            # OUT
            new_rows.append({'transaction_id': base_id, 'customer_id': source_id, 'date': pd.to_datetime(current_date), 'amount': amount, 'direction': 'OUT', 'label': 5})
            # IN (Same Day)
            new_rows.append({'transaction_id': base_id + 1, 'customer_id': target_id, 'date': pd.to_datetime(current_date), 'amount': amount, 'direction': 'IN', 'label': 5})
            base_id += 2
            
    return pd.concat([df, pd.DataFrame(new_rows)])

def _inject_daily_self_passthrough(df: pd.DataFrame, num_typologies: int, date_range: List[datetime.date], num_customers: int) -> pd.DataFrame:
    """Inject same-day self-loop passthroughs for consecutive days. Label: 4"""
    new_rows = []
    base_id = df['transaction_id'].max() + 1
    
    for _ in range(num_typologies):
        customer_id = np.random.randint(0, num_customers)
        amount = np.round(np.random.uniform(2000, 8000), 2)
        # Create same-day IN/OUT for 3-5 consecutive days
        num_days = np.random.randint(3, 6)
        start_day_idx = np.random.randint(0, len(date_range) - num_days)
        
        for d in range(num_days):
            current_date = date_range[start_day_idx + d]
            # 1. IN
            new_rows.append({'transaction_id': base_id, 'customer_id': customer_id, 'date': pd.to_datetime(current_date), 'amount': amount, 'direction': 'IN', 'label': 4})
            # 2. OUT (Same Day)
            new_rows.append({'transaction_id': base_id + 1, 'customer_id': customer_id, 'date': pd.to_datetime(current_date), 'amount': amount, 'direction': 'OUT', 'label': 4})
            base_id += 2
            
    return pd.concat([df, pd.DataFrame(new_rows)])

def _inject_repeated_layering(df: pd.DataFrame, num_typologies: int, date_range: List[datetime.date], num_customers: int) -> pd.DataFrame:
    """Inject recurring 1:1 matches between the same pair. Label: 3"""
    new_rows = []
    base_id = df['transaction_id'].max() + 1
    
    for _ in range(num_typologies):
        source_id = np.random.randint(0, num_customers)
        target_id = np.random.randint(0, num_customers)
        while target_id == source_id:
            target_id = np.random.randint(0, num_customers)
            
        amount = np.round(np.random.uniform(1000, 5000), 2)
        num_repeats = np.random.randint(3, 6)
        
        for i in range(num_repeats):
            day_idx = i * 5
            if day_idx >= len(date_range) - 2: break
            
            start_date = date_range[day_idx]
            new_rows.append({'transaction_id': base_id, 'customer_id': source_id, 'date': pd.to_datetime(start_date), 'amount': amount, 'direction': 'OUT', 'label': 3})
            next_date = start_date + datetime.timedelta(days=1)
            new_rows.append({'transaction_id': base_id + 1, 'customer_id': target_id, 'date': pd.to_datetime(next_date), 'amount': amount, 'direction': 'IN', 'label': 3})
            base_id += 2
            
    return pd.concat([df, pd.DataFrame(new_rows)])

def _inject_pass_through(df: pd.DataFrame, num_typologies: int, date_range: List[datetime.date], num_customers: int) -> pd.DataFrame:
    """Inject 1:1 matching OUT/IN within 1-2 days. Label: 1"""
    new_rows = []
    base_id = df['transaction_id'].max() + 1
    
    for _ in range(num_typologies):
        source_id = np.random.randint(0, num_customers)
        target_id = np.random.randint(0, num_customers)
        while target_id == source_id:
            target_id = np.random.randint(0, num_customers)
            
        start_date = np.random.choice(date_range[:-2])
        amount = np.round(np.random.uniform(1000, 5000), 2)
        
        new_rows.append({'transaction_id': base_id, 'customer_id': source_id, 'date': pd.to_datetime(start_date), 'amount': amount, 'direction': 'OUT', 'label': 1})
        next_date = start_date + datetime.timedelta(days=np.random.randint(1, 3))
        new_rows.append({'transaction_id': base_id + 1, 'customer_id': target_id, 'date': pd.to_datetime(next_date), 'amount': amount, 'direction': 'IN', 'label': 1})
        base_id += 2
        
    return pd.concat([df, pd.DataFrame(new_rows)])

def _inject_fan_in(df: pd.DataFrame, num_typologies: int, date_range: List[datetime.date], num_customers: int) -> pd.DataFrame:
    """Inject Many-to-One (Fan-In). Label: 2"""
    new_rows = []
    base_id = df['transaction_id'].max() + 1
    
    for _ in range(num_typologies):
        hub_id = np.random.randint(0, num_customers)
        num_mules = np.random.randint(3, 6)
        mule_ids = np.random.choice(range(num_customers), num_mules, replace=False)
        hub_date = np.random.choice(date_range[2:])
        
        total_amount = 0
        for m_id in mule_ids:
            mule_date = hub_date - datetime.timedelta(days=np.random.randint(1, 3))
            amount = np.round(np.random.uniform(100, 500), 2)
            total_amount += amount
            new_rows.append({'transaction_id': base_id, 'customer_id': m_id, 'date': pd.to_datetime(mule_date), 'amount': amount, 'direction': 'OUT', 'label': 2})
            base_id += 1
            
        new_rows.append({'transaction_id': base_id, 'customer_id': hub_id, 'date': pd.to_datetime(hub_date), 'amount': np.round(total_amount, 2), 'direction': 'IN', 'label': 2})
        base_id += 1
        
    return pd.concat([df, pd.DataFrame(new_rows)])
