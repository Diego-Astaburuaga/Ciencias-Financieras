from typing import Optional
import numpy as np
from .types import ModelConfig, FitResult
from .base import BaseModel


class HARModel(BaseModel):
    """
    Heterogeneous Autoregressive (HAR) model.

    Expects:
    X : 2D array (n_samples, n_features)
        Example features: [RV_d, RV_w, RV_m]
    y : 1D array (n_samples,)
        Target: RV_{t+1}
    """

    def __init__(self, config: ModelConfig, callback=None):
        super().__init__(config)
        self.beta: Optional[np.ndarray] = None
        self.n_features: Optional[int] = None
        self.callback = callback

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        loss_fn=None,
        **kwargs
    ) -> FitResult:

        X = np.asarray(X_train, dtype=np.float64)
        y = np.asarray(y_train, dtype=np.float64).reshape(-1)

        if X.ndim != 2:
            raise ValueError("X_train must be 2D array.")

        if X.shape[0] != len(y):
            raise ValueError("X_train and y_train must have same number of samples.")

        self.n_features = X.shape[1]

        # Add intercept
        X_design = np.column_stack([np.ones(X.shape[0]), X])

        # OLS solution
        beta, residuals, rank, s = np.linalg.lstsq(X_design, y, rcond=None)

        if rank < X_design.shape[1]:
            print("Warning: Design matrix may be rank deficient.")

        self.beta = beta
        self.is_fitted = True

        if loss_fn is None:
            loss_fn = self.qlike

        # Training loss
        yhat_tr = self.predict(X_train)
        tr_loss = loss_fn(y_train, yhat_tr)

        if self.callback:
            self.callback(epoch=0, y_true=y_train, y_pred=yhat_tr, phase="train")

        # Validation
        val_loss = None

        if X_val is not None and y_val is not None:
            yhat_va = self.predict(X_val)
            val_loss = float(loss_fn(y_val, yhat_va))

            if self.callback:
                self.callback(epoch=0, y_true=y_val, y_pred=yhat_va, phase="val")

        if hasattr(loss_fn, "__name__"):
            loss_name = loss_fn.__name__
        else:
            loss_name = loss_fn.__class__.__name__

        return FitResult(
            loss_name=loss_name,
            train_loss=float(tr_loss),
            val_loss=val_loss,
            fitted_params={
                "beta": self.beta.copy(),
                "rank": int(rank)
            },
            extra={
                "n_features": int(self.n_features),
                "n_train_samples": int(X.shape[0])
            }
        )

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, X, **kwargs):

        if not self.is_fitted:
            raise RuntimeError("HARModel must be fitted before prediction.")

        X = np.asarray(X, dtype=np.float64)

        if X.ndim != 2:
            raise ValueError("X must be 2D array.")

        if X.shape[1] != self.n_features:
            raise ValueError("Feature dimension mismatch.")

        X_design = np.column_stack([np.ones(X.shape[0]), X])

        yhat = X_design @ self.beta

        return yhat.astype(np.float64)