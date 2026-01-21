from dataclasses import dataclass

@dataclass
class Schema:
    # Raw Columns
    TRANSACTION_ID = "transaction_id"
    DATE = "date"
    VOLUME = "volume"
    DIRECTION = "direction"
    CUSTOMER_NAME = "customer_name"
    BANK_NAME = "bank_name"
    BANK_ACCOUNT = "bank_account"

    # Generated/Feature Columns
    RARITY_SCORE = "rarity_score"
    TIMESTAMP_NORM = "timestamp_norm"
    DIRECTION_ENCODED = "direction_encoded"
    
    # Node Types
    NODE_CUSTOMER = "customer"
    NODE_ACCOUNT = "account"
    NODE_TRANSACTION = "transaction"

    # Edge Types
    EDGE_OWNS = "owns"
    EDGE_EXECUTED = "executed"
