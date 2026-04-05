import pandas as pd
import numpy as np
from typing import Dict, Any

class DataProcessor:
    """
    Handles data cleaning, imputation, and aggregation.
    Inject your EDA and cleaning logic here.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.mapping = config['data']['column_mapping']

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform basic cleaning (handling nulls, type conversion).
        """
        # TODO: Inject your cleaning logic from EDA notebook
        return df

    def aggregate_to_customer(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate transaction-level data to customer-level features
        for the clustering and Isolation Forest layers.
        """
        c_id = self.mapping['customer_id']
        # TODO: Inject your aggregation logic
        # Default: count transactions and mean amount
        cust_df = df.groupby(c_id).size().to_frame('tx_count')
        return cust_df
