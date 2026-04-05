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
        'direction': directions
    })
    
    # Inject AML Typologies
    tx_df = _inject_pass_through(tx_df, int(num_customers * anomaly_ratio * 0.4), date_range, num_customers)
    tx_df = _inject_fan_in(tx_df, int(num_customers * anomaly_ratio * 0.4), date_range, num_customers)
    
    return tx_df.sort_values('date').reset_index(drop=True)

def _inject_pass_through(df: pd.DataFrame, num_typologies: int, date_range: List[datetime.date], num_customers: int) -> pd.DataFrame:
    """Inject 1:1 matching OUT/IN within 1-2 days (Money flow: Source -> Target)."""
    new_rows = []
    base_id = df['transaction_id'].max() + 1
    
    for _ in range(num_typologies):
        # We need TWO different customers to form an actual edge
        source_id = np.random.randint(0, num_customers)
        target_id = np.random.randint(0, num_customers)
        while target_id == source_id:
            target_id = np.random.randint(0, num_customers)
            
        start_date = np.random.choice(date_range[:-2])
        amount = np.round(np.random.uniform(1000, 5000), 2)
        
        # 1. OUT transaction (Money leaves Source)
        new_rows.append({'transaction_id': base_id, 'customer_id': source_id, 'date': pd.to_datetime(start_date), 'amount': amount, 'direction': 'OUT'})
        
        # 2. IN transaction (Money enters Target 1-2 days later)
        next_date = start_date + datetime.timedelta(days=np.random.randint(1, 3))
        new_rows.append({'transaction_id': base_id + 1, 'customer_id': target_id, 'date': pd.to_datetime(next_date), 'amount': amount, 'direction': 'IN'})
        base_id += 2
        
    return pd.concat([df, pd.DataFrame(new_rows)])

def _inject_fan_in(df: pd.DataFrame, num_typologies: int, date_range: List[datetime.date], num_customers: int) -> pd.DataFrame:
    """Inject Many-to-One (Fan-In): 3-5 mules withdraw small amounts (OUT), 1 Hub deposits (IN) the aggregate total."""
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
            # Mules withdraw (OUT) to send money
            new_rows.append({'transaction_id': base_id, 'customer_id': m_id, 'date': pd.to_datetime(mule_date), 'amount': amount, 'direction': 'OUT'})
            base_id += 1
            
        # Hub receives the deposit (IN)
        new_rows.append({'transaction_id': base_id, 'customer_id': hub_id, 'date': pd.to_datetime(hub_date), 'amount': np.round(total_amount, 2), 'direction': 'IN'})
        base_id += 1
        
    return pd.concat([df, pd.DataFrame(new_rows)])
