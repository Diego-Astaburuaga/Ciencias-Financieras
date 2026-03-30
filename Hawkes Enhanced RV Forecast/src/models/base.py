from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
from dataclasses import asdict
import numpy as np
from .types import ModelConfig, FitResult


class BaseModel(ABC):
    """
    Common interface for all models in the experiment.

    Contract:
    - fit() must return a FitResult
    - predict() must return a 1D numpy array
    """

    _EPS = 1e-12
    _MAX_RATIO = 1e12

    def __init__(self, config: ModelConfig):
        self.config = config
        self.name = config.name
        self.is_fitted = False

    # ------------------------------------------------------------------
    # Abstract API
    # ------------------------------------------------------------------

    @abstractmethod
    def fit(self, *args, **kwargs) -> FitResult:
        """Train the model and return training summary."""
        pass

    @abstractmethod
    def predict(self, *args, **kwargs) -> np.ndarray:
        """Generate predictions."""
        pass

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_params(self) -> Dict[str, Any]:
        return asdict(self.config)

    def reset(self) -> None:
        """Reset fitted state (useful for rolling windows)."""
        self.is_fitted = False

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _as_1d_pair(y_true, y_pred) -> Tuple[np.ndarray, np.ndarray]:
        y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
        y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)

        if y_true.shape != y_pred.shape:
            raise ValueError("y_true and y_pred must have the same shape.")
        if not (np.all(np.isfinite(y_pred)) and np.all(np.isfinite(y_true))):
            raise ValueError("y_true and y_pred must be finite (no NaN/Inf).")

        return y_true, y_pred

    # ------------------------------------------------------------------
    # Loss functions
    # ------------------------------------------------------------------

    @staticmethod
    def mse(y_true, y_pred) -> float:
        y_true, y_pred = BaseModel._as_1d_pair(y_true, y_pred)
        return float(np.mean((y_pred - y_true) ** 2))

    @staticmethod
    def mae(y_true, y_pred) -> float:
        y_true, y_pred = BaseModel._as_1d_pair(y_true, y_pred)
        return float(np.mean(np.abs(y_pred - y_true)))

    @staticmethod
    def rmse(y_true, y_pred) -> float:
        return float(np.sqrt(BaseModel.mse(y_true, y_pred)))

    @staticmethod
    def rmae(y_true, y_pred) -> float:
        return float(np.sqrt(BaseModel.mae(y_true, y_pred)))

    @staticmethod
    def qlike(y_true, y_pred, eps=None, max_ratio=None) -> float:
        """
        QLIKE loss for volatility forecasting.

        Inputs are expected in sqrt(RV) space. They are squared to
        RV space before computing:
            QLIKE = RV_true / RV_pred - log(RV_true / RV_pred) - 1

        With ratio clipping for numerical stability.
        """

        y_true, y_pred = BaseModel._as_1d_pair(y_true, y_pred)

        eps = BaseModel._EPS if eps is None else eps
        max_ratio = BaseModel._MAX_RATIO if max_ratio is None else max_ratio

        # Convert sqrt(RV) → RV before computing QLIKE
        rv_true = np.clip(np.square(y_true), eps, None)
        rv_pred = np.clip(np.square(y_pred), eps, None)

        ratio = np.clip(rv_true / rv_pred, eps, max_ratio)
        return float(np.mean(ratio - np.log(ratio) - 1.0))

    @staticmethod
    def mse_loss_vector(y_true, y_pred) -> np.ndarray:
        y_true, y_pred = BaseModel._as_1d_pair(y_true, y_pred)
        return (y_pred - y_true) ** 2