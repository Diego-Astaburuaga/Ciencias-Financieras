"""
LSTM model for realized-volatility forecasting.

References
----------
Hochreiter1997
    Hochreiter, S., and Schmidhuber, J. (1997). Long short-term memory.
    Neural Computation, 9(8), 1735-1780.

Notes
-----
The recurrent architecture used in this module is the standard LSTM introduced
in Hochreiter1997.

Architecture
------------
Input  : (batch, lookback_days, n_features)
         Built internally from a 2-D (T, F) array by sliding a window of
         length ``lookback_days`` over the time axis.  The model is agnostic
         to the number of input features.
LSTM   : single layer, hidden size ``hidden_size``.
Head   : Linear(hidden_size → 1).
Output : controlled by ``output_activation`` — softplus, relu, tanh, gelu,
         or linear (raw linear).

Prediction modes
----------------
``predict(X, batched=False)``
    X is a 2-D array of shape ``(lookback_days, n_features)`` — a single
    context window.  Returns a scalar ``float``.

``predict(X, batched=True)``
    X is a 2-D array of shape ``(T, n_features)`` with T > lookback_days.
    Slides a window of length ``lookback_days`` over the time axis and returns
    an ``np.ndarray`` of shape ``(T − lookback_days,)`` — one prediction per
    valid window.

    For rolling-window evaluation prepend the last ``lookback_days`` rows of
    the training set so every test position has a full context window::

        X_eval = np.concatenate([X_train[-lookback:], X_test], axis=0)
        preds  = model.predict(X_eval, batched=True)  # shape (len(X_test),)

Training criteria
-----------------
mse   : mean squared error.
mae   : mean absolute error.
hmse  : heteroskedastic MSE — mean((ŷ − y)² / y²).
hmae  : heteroskedastic MAE — mean(|ŷ − y| / y).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .metrics import qlike


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class LSTMConfig:
    """
    Configuration for :class:`LSTMModel`.

    Parameters
    ----------
    name : str
        Model identifier used in results and comparisons.
    device : str
        PyTorch device string (``"cpu"`` or ``"cuda"``).
    lookback_days : int
        Number of past time steps fed to the LSTM as context (default 30).
    hidden_size : int
        Number of hidden units in the LSTM layer.
    representation : str
        Only ``"lstm_hidden"`` is currently supported — uses the final hidden
        state as the representation fed to the linear output head.
    output_activation : str
        Non-linearity applied to the scalar logit before returning predictions.
        ``"softplus"`` : F.softplus(x) + ε — smooth and strictly positive.
        ``"relu"``    : F.relu(x) + ε — linear for positive values.
        ``"tanh"``    : bounded in (−1, 1).
        ``"gelu"``    : smooth approximation of ReLU.
        ``"linear"``  : raw linear output (no positivity constraint).
    criterion : str
        Training loss.
        ``"mse"``  : mean squared error.
        ``"mae"``  : mean absolute error.
        ``"hmse"`` : heteroskedastic MSE — mean((ŷ − y)² / y²).
        ``"hmae"`` : heteroskedastic MAE — mean(|ŷ − y| / y).
    optimizer : str
        ``"adam"`` (default) or ``"adamw"`` (decoupled weight decay).
    lr : float
        Learning rate.
    epochs : int
        Maximum training epochs.
    batch_size : int
        Mini-batch size.
    patience : int
        Early-stopping patience (epochs without val improvement).
        Set to 0 to disable.
    min_delta : float
        Minimum validation loss decrease to count as an improvement.
    """

    name: str = "LSTM"
    device: str = "cpu"
    lookback_days: int = 30
    hidden_size: int = 64
    representation: str = "lstm_hidden"
    output_activation: str = "linear"
    criterion: str = "mse"
    optimizer: str = "adam"
    lr: float = 1e-3
    epochs: int = 300
    batch_size: int = 32
    patience: int = 30
    min_delta: float = 0.0

    _VALID_REPRESENTATION    = frozenset({"lstm_hidden"})
    _VALID_OUTPUT_ACTIVATION = frozenset({"softplus", "relu", "tanh", "gelu", "linear"})
    _VALID_CRITERION         = frozenset({"mse", "mae", "hmse", "hmae"})
    _VALID_OPTIMIZER         = frozenset({"adam", "adamw"})

    def __post_init__(self) -> None:
        if self.representation not in self._VALID_REPRESENTATION:
            raise ValueError(
                f"representation must be one of {set(self._VALID_REPRESENTATION)}."
            )
        if self.output_activation not in self._VALID_OUTPUT_ACTIVATION:
            raise ValueError(
                f"output_activation must be one of {set(self._VALID_OUTPUT_ACTIVATION)}."
            )
        if self.criterion not in self._VALID_CRITERION:
            raise ValueError(
                f"criterion must be one of {set(self._VALID_CRITERION)}."
            )
        if self.optimizer not in self._VALID_OPTIMIZER:
            raise ValueError(
                f"optimizer must be one of {set(self._VALID_OPTIMIZER)}."
            )
        if self.lookback_days <= 0:
            raise ValueError("lookback_days must be > 0.")
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be > 0.")
        if self.patience < 0:
            raise ValueError("patience must be >= 0.")
        if self.min_delta < 0:
            raise ValueError("min_delta must be >= 0.")

    @property
    def kind(self) -> str:
        return "lstm"


@dataclass
class FitResult:
    loss_name: str
    train_loss: float
    val_loss: float | None = None
    fitted_params: dict[str, object] = field(default_factory=dict)
    extra: dict[str, object] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────────────
# Training criteria (PyTorch)
# ──────────────────────────────────────────────────────────────────────────────

def _torch_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target)


def _torch_mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(pred, target)


def _torch_hmse(
    pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12
) -> torch.Tensor:
    t = target.clamp(min=eps)
    return ((pred - t) / t).square().mean()


def _torch_hmae(
    pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12
) -> torch.Tensor:
    t = target.clamp(min=eps)
    return ((pred - t).abs() / t).mean()


_CRITERION_FNS: dict[str, Callable] = {
    "mse":  _torch_mse,
    "mae":  _torch_mae,
    "hmse": _torch_hmse,
    "hmae": _torch_hmae,
}


# ──────────────────────────────────────────────────────────────────────────────
# Internal PyTorch module
# ──────────────────────────────────────────────────────────────────────────────

class _LSTMNet(nn.Module):
    """Single-layer LSTM with a linear output head."""

    def __init__(
        self,
        n_features: int,
        hidden_size: int,
        output_activation: str,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, 1)
        self._output_activation = output_activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape (batch, seq_len, n_features)

        Returns
        -------
        Tensor, shape (batch,)
        """
        _, (h_n, _) = self.lstm(x)       # h_n: (1, batch, hidden_size)
        rep = h_n.squeeze(0)              # (batch, hidden_size)
        out = self.head(rep).squeeze(-1)  # (batch,)
        act = self._output_activation
        if act == "softplus":
            return F.softplus(out) + 1e-8
        if act == "relu":
            return F.relu(out) + 1e-8
        if act == "tanh":
            return torch.tanh(out)
        if act == "gelu":
            return F.gelu(out)
        return out  # linear → identity


# ──────────────────────────────────────────────────────────────────────────────
# LSTMModel
# ──────────────────────────────────────────────────────────────────────────────

class LSTMModel:
    """
    LSTM-based realized-volatility forecasting model.

    Parameters
    ----------
    config : LSTMConfig
    callback : callable | None
        Optional epoch-level callback ``callback(epoch, y_true, y_pred, phase)``.
    """

    def __init__(
        self,
        config: LSTMConfig,
        callback: Optional[Callable] = None,
    ) -> None:
        self.config = config
        self.name = config.name
        self.is_fitted = False
        self.callback = callback

        self._net: Optional[_LSTMNet] = None
        self._n_features: Optional[int] = None
        self._device = torch.device(config.device)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _build_net(self, n_features: int) -> _LSTMNet:
        cfg = self.config
        return _LSTMNet(
            n_features=n_features,
            hidden_size=cfg.hidden_size,
            output_activation=cfg.output_activation,
        ).to(self._device)

    @staticmethod
    def _make_sequences(
        X: np.ndarray,
        y: np.ndarray,
        lookback: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Slide a window of length ``lookback`` over ``X``.

        Window i covers X[i : i+lookback] and predicts y[i+lookback].
        Returns (T − lookback) pairs.
        """
        T, F = X.shape
        n = T - lookback
        if n <= 0:
            raise ValueError(
                f"Not enough samples ({T}) for lookback_days={lookback}."
            )
        X_seq = np.lib.stride_tricks.sliding_window_view(X, (lookback, F))[
            :-1
        ].reshape(n, lookback, F)
        y_seq = y[lookback:]
        return X_seq.astype(np.float32), y_seq.astype(np.float32)

    @staticmethod
    def _slide_X(X: np.ndarray, lookback: int) -> np.ndarray:
        """
        Sliding windows without targets: returns (T − lookback, lookback, F).
        """
        T, F = X.shape
        n = T - lookback
        if n <= 0:
            raise ValueError(
                f"Not enough rows ({T}) for lookback_days={lookback}."
            )
        return np.lib.stride_tricks.sliding_window_view(X, (lookback, F))[
            :-1
        ].reshape(n, lookback, F).astype(np.float32)

    def _to_loader(
        self,
        X_seq: np.ndarray,
        y_seq: np.ndarray,
        shuffle: bool,
    ) -> DataLoader:
        X_t = torch.from_numpy(X_seq).to(self._device)
        y_t = torch.from_numpy(y_seq).to(self._device)
        ds = TensorDataset(X_t, y_t)
        return DataLoader(
            ds,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            drop_last=False,
        )

    # ── public API ────────────────────────────────────────────────────────────

    def fit(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        loss_fn: Optional[Callable] = None,
        warm_start: bool = False,
        **kwargs,
    ) -> FitResult:
        """
        Train the LSTM on ``(X_train, y_train)``.

        Parameters
        ----------
        X_train : array-like, shape (T, n_features)
        y_train : array-like, shape (T,)
        X_val, y_val : optional validation arrays for early stopping.
        loss_fn : numpy loss ``(y_true, y_pred) → float`` used only for
            reporting in ``FitResult``.  Defaults to QLIKE.

        Returns
        -------
        FitResult
        """
        X = np.asarray(X_train, dtype=np.float64)
        y = np.asarray(y_train, dtype=np.float64).ravel()

        if X.ndim != 2:
            raise ValueError("X_train must be a 2-D array (T, n_features).")
        if X.shape[0] != len(y):
            raise ValueError("X_train and y_train must have the same length.")

        cfg = self.config
        lookback = cfg.lookback_days
        self._n_features = X.shape[1]

        X_seq, y_seq = self._make_sequences(X, y, lookback)

        # Re-initialise weights unless warm-starting from previous fit state.
        reuse_existing = (
            warm_start
            and self._net is not None
            and self.is_fitted
            and self._n_features == X.shape[1]
        )
        if not reuse_existing:
            self._net = self._build_net(self._n_features)
        _optim_cls = (
            torch.optim.AdamW if cfg.optimizer == "adamw" else torch.optim.Adam
        )
        optim = _optim_cls(self._net.parameters(), lr=cfg.lr)
        criterion = _CRITERION_FNS[cfg.criterion]

        train_loader = self._to_loader(X_seq, y_seq, shuffle=True)

        has_val = X_val is not None and y_val is not None
        val_loader = None
        if has_val:
            X_v = np.asarray(X_val, dtype=np.float64)
            y_v = np.asarray(y_val, dtype=np.float64).ravel()
            X_seq_v, y_seq_v = self._make_sequences(X_v, y_v, lookback)
            val_loader = self._to_loader(X_seq_v, y_seq_v, shuffle=False)

        best_monitor = float("inf")
        best_state = None
        no_improve = 0
        epoch = 0

        for epoch in range(cfg.epochs):
            # ── training pass ──────────────────────────────────────────────
            self._net.train()
            train_accum, train_n = 0.0, 0
            for X_batch, y_batch in train_loader:
                optim.zero_grad()
                pred = self._net(X_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                optim.step()
                n = int(y_batch.shape[0])
                train_accum += float(loss.item()) * n
                train_n += n

            monitor = train_accum / max(train_n, 1)

            # ── callback & early stopping ───────────────────────────────────
            if self.callback or cfg.patience > 0:
                self._net.eval()
                with torch.no_grad():
                    if self.callback:
                        X_all = torch.from_numpy(X_seq).to(self._device)
                        p_tr = self._net(X_all).cpu().numpy().astype(np.float64)
                        self.callback(
                            epoch=epoch,
                            y_true=y_seq.astype(np.float64),
                            y_pred=p_tr,
                            phase="train",
                        )

                    if has_val and val_loader is not None:
                        vp_list, vt_list = [], []
                        for Xb, yb in val_loader:
                            vp_list.append(self._net(Xb))
                            vt_list.append(yb)
                        vp = torch.cat(vp_list)
                        vt = torch.cat(vt_list)
                        monitor = float(criterion(vp, vt).item())

                        if self.callback:
                            self.callback(
                                epoch=epoch,
                                y_true=vt.cpu().numpy().astype(np.float64),
                                y_pred=vp.cpu().numpy().astype(np.float64),
                                phase="val",
                            )

                    if cfg.patience > 0:
                        if monitor < best_monitor - cfg.min_delta:
                            best_monitor = monitor
                            best_state = {
                                k: v.clone()
                                for k, v in self._net.state_dict().items()
                            }
                            no_improve = 0
                        else:
                            no_improve += 1
                            if no_improve >= cfg.patience:
                                break

        if best_state is not None:
            self._net.load_state_dict(best_state)

        self.is_fitted = True

        # ── report losses via numpy loss_fn ────────────────────────────────
        if loss_fn is None:
            loss_fn = qlike

        self._net.eval()
        with torch.no_grad():
            X_all = torch.from_numpy(X_seq).to(self._device)
            y_hat_seq = self._net(X_all).cpu().numpy().astype(np.float64)
        tr_loss = float(loss_fn(y_seq.astype(np.float64), y_hat_seq))

        val_loss_report: Optional[float] = None
        if has_val:
            X_v_np = np.asarray(X_val, dtype=np.float64)
            y_v_np = np.asarray(y_val, dtype=np.float64).ravel()
            X_seq_v2, y_seq_v2 = self._make_sequences(X_v_np, y_v_np, lookback)
            with torch.no_grad():
                X_vt = torch.from_numpy(X_seq_v2).to(self._device)
                y_hat_v = self._net(X_vt).cpu().numpy().astype(np.float64)
            val_loss_report = float(loss_fn(y_seq_v2.astype(np.float64), y_hat_v))

        loss_name = getattr(loss_fn, "__name__", loss_fn.__class__.__name__)

        return FitResult(
            loss_name=loss_name,
            train_loss=tr_loss,
            val_loss=val_loss_report,
            fitted_params={
                "n_features": self._n_features,
                "lookback_days": lookback,
                "hidden_size": cfg.hidden_size,
                "output_activation": cfg.output_activation,
            },
            extra={
                "n_train_samples": int(X.shape[0]),
                "n_seq_train": int(X_seq.shape[0]),
                "criterion": cfg.criterion,
                "optimizer": cfg.optimizer,
                "min_delta": cfg.min_delta,
                "monitor": "val" if has_val else "train",
                "best_monitor_loss": best_monitor,
                "epochs_run": epoch + 1,
                "early_stopped": best_state is not None,
            },
        )

    def predict(self, X, batched: bool = True):
        """
        Generate predictions.

        Parameters
        ----------
        X : array-like
            ``batched=False``: shape ``(lookback_days, n_features)`` — one
            context window.  Returns a ``float``.

            ``batched=True`` : shape ``(T, n_features)`` with T > lookback_days.
            Slides a window of length ``lookback_days`` and returns an
            ``np.ndarray`` of shape ``(T − lookback_days,)``.

        batched : bool
            ``False`` → scalar prediction.
            ``True``  → array of predictions.
        """
        if not self.is_fitted or self._net is None:
            raise RuntimeError("LSTMModel must be fitted before prediction.")

        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError("X must be a 2-D array.")
        if X.shape[1] != self._n_features:
            raise ValueError(
                f"Feature dimension mismatch: expected {self._n_features}, "
                f"got {X.shape[1]}."
            )

        lookback = self.config.lookback_days
        self._net.eval()

        if not batched:
            if X.shape[0] != lookback:
                raise ValueError(
                    f"batched=False requires exactly {lookback} rows, "
                    f"got {X.shape[0]}."
                )
            X_t = torch.from_numpy(X[np.newaxis]).to(self._device)  # (1, lookback, F)
            with torch.no_grad():
                return float(self._net(X_t).item())

        # batched=True: (T, F) → (T − lookback,) predictions
        X_seq = self._slide_X(X, lookback)  # (T - lookback, lookback, F)
        X_t = torch.from_numpy(X_seq).to(self._device)
        with torch.no_grad():
            return self._net(X_t).cpu().numpy().astype(np.float64)

    def reset(self) -> None:
        """Discard trained weights."""
        self.is_fitted = False
        self._net = None
        self._n_features = None
