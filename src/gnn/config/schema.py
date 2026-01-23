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
    
    # Advanced Features (Peak & Shift)
    PEAK_FLOW_THROUGH = "peak_flow_through"
    PEAK_ROUNDNESS = "peak_roundness"
    PEAK_ENTROPY = "peak_entropy"
    SHIFT_SCORE = "shift_score"
    
    # Identifiers
    ACCOUNT_ID = "account_id"
    RESOLVED_ENTITY_ID = "resolved_entity_id"
    SYNDICATE_ID = "syndicate_id"
    IS_SUPERNODE = "is_supernode"
    RISK_SCORE = "risk_score"
    
    # Node Types
    NODE_CUSTOMER = "customer"
    NODE_ACCOUNT = "account"
    NODE_TRANSACTION = "transaction"

    # Edge Types
    EDGE_OWNS = "owns"
    EDGE_EXECUTED = "executed"
    
    # Reverse Edge Types
    EDGE_OWNED_BY = "owned_by"
    EDGE_EXECUTED_BY = "executed_by"