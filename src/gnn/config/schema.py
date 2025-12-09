"""Configuration schemas for the modular graph framework."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from pydantic import BaseModel, Field, validator


class FeatureGroup(str, Enum):
    """Groups used to classify node and edge features."""

    STRUCTURAL = "structural"
    TEMPORAL = "temporal"
    STATIC = "static"
    RISK = "risk"


class FeatureDefinition(BaseModel):
    """Declarative feature specification driven by external config."""

    name: str
    columns: List[str] = Field(default_factory=list)
    method: str = Field(
        default="identity",
        description="How to compute the feature (e.g., identity, mean, sum, degree_in, degree_out, age_days).",
    )
    params: Dict[str, Any] = Field(default_factory=dict)
    description: Optional[str] = None


class FeatureGroupConfig(BaseModel):
    """Feature definitions grouped by semantic type."""

    structural: List[FeatureDefinition] = Field(default_factory=list)
    temporal: List[FeatureDefinition] = Field(default_factory=list)
    static: List[FeatureDefinition] = Field(default_factory=list)
    risk: List[FeatureDefinition] = Field(default_factory=list)

    def iter_definitions(self) -> Iterable[Tuple[FeatureGroup, FeatureDefinition]]:
        for group in FeatureGroup:
            for definition in getattr(self, group.value):
                yield group, definition


class FeaturesConfig(BaseModel):
    """Node and edge feature configuration."""

    node: FeatureGroupConfig = Field(default_factory=FeatureGroupConfig)
    edge: FeatureGroupConfig = Field(default_factory=FeatureGroupConfig)


class DataConfig(BaseModel):
    """Raw data and split configuration."""

    raw_path: Optional[Path] = None
    target: Optional[str] = None
    splits: Dict[str, float] = Field(
        default_factory=lambda: {"train": 0.7, "val": 0.15, "test": 0.15}
    )
    seed: int = 42

    @validator("splits")
    def _validate_splits(cls, v: Dict[str, float]) -> Dict[str, float]:
        total = sum(v.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError("Data splits must sum to 1.0")
        return v


class GraphConfig(BaseModel):
    """Describe how to map tabular transactions into a graph."""

    src_column: str = "src_account"
    dst_column: str = "dst_account"
    timestamp_column: Optional[str] = None
    amount_column: Optional[str] = None
    target_column: Optional[str] = None
    metadata_label: str = "dataset"
    directed: bool = True
    edge_derivation: Optional["EdgeDerivationConfig"] = None


class FlowEdgeRuleConfig(BaseModel):
    """Config for flow-based inferred edges (A -> fintech -> B)."""

    enabled: bool = False
    fintech_account_column: str = "fintech_acct_id"
    counterparty_account_column: str = "counterparty_acct_id"
    timestamp_column: str = "timestamp"
    amount_column: str = "amount"
    direction_column: str = "direction"
    incoming_values: List[str] = Field(default_factory=lambda: ["CREDIT", "IN"])
    outgoing_values: List[str] = Field(default_factory=lambda: ["DEBIT", "OUT"])
    reference_column: Optional[str] = "reference"
    time_window_minutes: int = 120
    amount_tolerance_pct: float = 0.1


class BatchEdgeRuleConfig(BaseModel):
    """Config for batch/co-usage edges."""

    enabled: bool = False
    batch_id_column: str = "batch_id"
    counterparty_account_column: str = "counterparty_acct_id"
    amount_column: Optional[str] = "amount"


class SimilarityEdgeRuleConfig(BaseModel):
    """Config for attribute-based similarity edges."""

    enabled: bool = False
    counterparty_account_column: str = "counterparty_acct_id"
    name_column: Optional[str] = "counterparty_name"
    address_column: Optional[str] = "counterparty_address"
    city_column: Optional[str] = "counterparty_city"
    bank_id_column: Optional[str] = "counterparty_bank_id"
    name_similarity_threshold: float = 0.9
    address_similarity_threshold: float = 0.9
    city_similarity_threshold: float = 0.9
    token_similarity_threshold: float = 0.8


class EdgeDerivationConfig(BaseModel):
    """High-level edge derivation controls."""

    flow: FlowEdgeRuleConfig = Field(default_factory=FlowEdgeRuleConfig)
    batch: BatchEdgeRuleConfig = Field(default_factory=BatchEdgeRuleConfig)
    similarity: SimilarityEdgeRuleConfig = Field(default_factory=SimilarityEdgeRuleConfig)


class TransactionAnomalyConfig(BaseModel):
    """Config for transaction-level anomaly scoring (View B)."""

    enabled: bool = False
    counterparty_account_column: str = "counterparty_acct_id"
    amount_column: str = "amount"
    timestamp_column: Optional[str] = None
    direction_column: Optional[str] = None
    features: List[str] = Field(default_factory=list)
    method: str = "isolation_forest"
    contamination: float = 0.01
    subgraph_hops: int = 1
    subgraph_min_size: int = 2


class AnalysisConfig(BaseModel):
    """Settings for data/graph analysis reports."""

    output_dir: Path = Path("artifacts/analysis")
    sample_size: Optional[int] = 5000
    include_plots: bool = True


class ModelTask(str, Enum):
    """Supported model task families."""

    LINK_PREDICTION = "link_prediction"
    ANOMALY_AUTOENCODER = "anomaly_autoencoder"
    NODE_CLASSIFICATION = "node_classification"


class ModelConfig(BaseModel):
    """Model selection and hyperparameters."""

    task: ModelTask = ModelTask.LINK_PREDICTION
    name: str = "baseline"
    params: Dict[str, Any] = Field(default_factory=dict)


class TrainingConfig(BaseModel):
    """Generic training hyperparameters."""

    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 1
    device: str = "cpu"


class PipelineConfig(BaseModel):
    """Top-level configuration tying every layer together."""

    data: DataConfig
    graph: GraphConfig
    features: FeaturesConfig
    model: ModelConfig
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    train: TrainingConfig = Field(default_factory=TrainingConfig)
    transaction_anomaly: TransactionAnomalyConfig = Field(default_factory=TransactionAnomalyConfig)

    @classmethod
    def model_validate(cls, obj) -> "PipelineConfig":
        # Compatibility shim with existing tests expecting model_validate
        return cls.parse_obj(obj)


# Resolve forward references for optional edge derivation
GraphConfig.update_forward_refs(EdgeDerivationConfig=EdgeDerivationConfig)
PipelineConfig.update_forward_refs(
    GraphConfig=GraphConfig,
    EdgeDerivationConfig=EdgeDerivationConfig,
    TransactionAnomalyConfig=TransactionAnomalyConfig,
)
