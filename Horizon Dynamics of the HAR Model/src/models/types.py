from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import torch

@dataclass
class ModelConfig:
    name: str = "HAR-RV"
    lookback_days: int = 30
    target_transform: str = "none"  # "log" or "none"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    features: List[str] = None

    def __post_init__(self):
        if self.features is None:
            self.features = ["rv_lag1", "rv_mean_5", "rv_mean_22"]

    @property
    def kind(self) -> str:
        return "har"


@dataclass
class TrainConfig:
    pass


@dataclass
class FitResult:
    loss_name: str
    train_loss: float
    val_loss: Optional[float]
    fitted_params: Dict[str, Any]
    extra: Dict[str, Any] = field(default_factory=dict)
