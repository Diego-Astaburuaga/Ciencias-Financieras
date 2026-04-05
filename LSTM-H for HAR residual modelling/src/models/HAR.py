"""
Heterogeneous Autoregressive model for realized volatility.

References
----------
Corsi2009
    Corsi, F. (2009). A simple approximate long-memory model of realized
    volatility. Journal of Financial Econometrics, 7(2), 174-196.

Notes
-----
This implementation follows the HAR-RV specification introduced in Corsi2009.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np
from .metrics import qlike


@dataclass
class HARConfig:
    name: str = "HAR-RV"
    device: str = "cpu"
    fit_method: str = "ols"

    def __post_init__(self) -> None:
        method_key = self.fit_method.lower().strip()
        if method_key not in {"ols", "wls"}:
            raise ValueError("fit_method must be either 'ols' or 'wls'.")
        self.fit_method = method_key

    @property
    def kind(self) -> str:
        return "har"


class HARModel:
    """
    Heterogeneous Autoregressive (HAR) model.

    Expects:
    X : 2D array (n_samples, n_features)
        Example features: [RV_d, RV_w, RV_m]
    y : 1D array (n_samples,)
        Target: RV_{t+1}
    """

    def __init__(self, config: HARConfig, callback=None):
        self.config = config
        self.name = config.name
        self.is_fitted = False
        self.beta: Optional[np.ndarray] = None
        self.n_features: Optional[int] = None
        self.callback = callback
        self.fit_method: Optional[str] = config.fit_method
        self._rank: Optional[int] = None
        self.fit_state: Optional[dict[str, object]] = None

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_weights(weights, n_samples: int) -> np.ndarray:
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        if w.shape[0] != n_samples:
            raise ValueError("sample_weight must have the same length as X_train.")
        if not np.all(np.isfinite(w)):
            raise ValueError("sample_weight must contain only finite values.")
        if np.any(w <= 0):
            raise ValueError("sample_weight values must be strictly positive for WLS.")
        return w

    def fit(
        self,
        X_train,
        y_train,
        method: str = None,
        sample_weight=None,
        loss_fn=None,
        **kwargs,
    ) -> dict[str, object]:
    

        X = np.asarray(X_train, dtype=np.float64)
        y = np.asarray(y_train, dtype=np.float64).reshape(-1)

        if X.ndim != 2:
            raise ValueError("X_train must be 2D array.")

        if X.shape[0] != len(y):
            raise ValueError("X_train and y_train must have same number of samples.")

        self.n_features = X.shape[1]
        method_key = (method if method is not None else self.config.fit_method).lower().strip()
        if method_key not in {"ols", "wls"}:
            raise ValueError("method must be either 'ols' or 'wls'.")

        # Add intercept
        X_design = np.column_stack([np.ones(X.shape[0]), X])

        if method_key == "wls":
            if sample_weight is None:
                raise ValueError("sample_weight is required when method='wls'.")
            w = self._validate_weights(sample_weight, X.shape[0])
            sw = np.sqrt(w)
            X_fit = X_design * sw[:, None]
            y_fit = y * sw
        else:
            X_fit = X_design
            y_fit = y

        beta, residuals, rank, s = np.linalg.lstsq(X_fit, y_fit, rcond=None)

        if rank < X_design.shape[1]:
            print("Warning: Design matrix may be rank deficient.")

        self.beta = beta
        self.fit_method = method_key
        self._rank = int(rank)
        self.is_fitted = True

        if loss_fn is None:
            loss_fn = qlike

        # Training loss
        yhat_tr = self.predict(X_train)
        tr_loss = loss_fn(y_train, yhat_tr)

        if self.callback:
            self.callback(epoch=0, y_true=y_train, y_pred=yhat_tr, phase="train")


        self.fit_state = {
            "method": self.fit_method,
            "params": self.get_parameters_dict(),
            "mean_loss_per_obs": float(tr_loss),
        }
        return self.fit_state

    def get_parameters_dict(self) -> dict[str, float]:
        if not self.is_fitted or self.beta is None or self.n_features is None:
            raise RuntimeError("HARModel must be fitted before requesting parameters.")

        params: dict[str, float] = {"beta": float(self.beta[0])}

        if self.n_features >= 1:
            params["beta_d"] = float(self.beta[1])
        if self.n_features >= 2:
            params["beta_w"] = float(self.beta[2])
        if self.n_features >= 3:
            params["beta_m"] = float(self.beta[3])

        for coef_idx in range(4, self.n_features + 1):
            gamma_idx = coef_idx - 3
            params[f"gamma_{gamma_idx}"] = float(self.beta[coef_idx])

        return params

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, X, batched: bool | None = None, **kwargs):

        if not self.is_fitted:
            raise RuntimeError("HARModel must be fitted before prediction.")

        X = np.asarray(X, dtype=np.float64)
        is_single_input = X.ndim == 1

        if is_single_input:
            X = X.reshape(1, -1)
        elif X.ndim != 2:
            raise ValueError("X must be a 1D or 2D array.")

        if X.shape[1] != self.n_features:
            raise ValueError("Feature dimension mismatch.")

        X_design = np.column_stack([np.ones(X.shape[0]), X])

        yhat = (X_design @ self.beta).astype(np.float64)

        if batched is True:
            return yhat
        if batched is False:
            if yhat.shape[0] != 1:
                raise ValueError("batched=False requires exactly one input sample.")
            return float(yhat[0])

        return float(yhat[0]) if is_single_input else yhat