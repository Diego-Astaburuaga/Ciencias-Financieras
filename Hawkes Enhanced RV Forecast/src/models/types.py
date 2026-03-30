from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import torch

@dataclass
class ModelConfig:
    """
    Model-agnostic base configuration.
    
    Subclasses should extend this with model-specific parameters.
    """
    name: str = "Model"
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    lookback_days: int = 1  # Generic sequence length parameter

    def __post_init__(self) -> None:
        """Hook for subclasses to validate their fields."""
        pass


@dataclass
class HARConfig(ModelConfig):
    """
    Configuration for the HAR (Heterogeneous Autoregressive) model.
    
    Parameters
    ----------
    name : str
        Model identifier.
    device : str
        Device to use ("cuda" or "cpu").
    lookback_days : int
        Generic sequence lookback length (not used in HAR, kept for interface compatibility).
    features : List[str]
        Feature names for the HAR model (e.g., ["rv_lag1", "rv_mean_5", "rv_mean_22"]).
    """
    name: str = "HAR-RV"
    features: List[str] = field(default_factory=lambda: ["rv_lag1", "rv_mean_5", "rv_mean_22"])

    @property
    def kind(self) -> str:
        return "har"


@dataclass
class FitResult:
    """
    Result object returned by model.fit() method.
    
    Captures training summary including losses, fitted parameters, and metadata.
    
    Parameters
    ----------
    loss_name : str
        Name of the loss function used (e.g., "mse", "qlike").
    train_loss : float
        Final training loss value.
    val_loss : Optional[float]
        Final validation loss value, if available. None if no validation set was used.
    fitted_params : Dict[str, Any]
        Model-specific fitted parameters (e.g., coefficients for HAR, weights for LSTM).
    extra : Dict[str, Any]
        Additional metadata (e.g., epochs_run, early_stopped, n_train_samples).
    """
    loss_name: str
    train_loss: float
    val_loss: Optional[float] = None
    fitted_params: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)