"""The output of compute_features()."""
from dataclasses import dataclass, field
from typing import Any


@dataclass
class FeatureVector:
    txn_id: str
    is_cold: bool
    values: dict[str, Any] = field(default_factory=dict)
