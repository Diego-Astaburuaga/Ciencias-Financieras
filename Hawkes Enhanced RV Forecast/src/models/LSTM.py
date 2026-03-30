"""
LSTM model for realized-volatility forecasting.

Inherits from BaseModel and follows the same fit/predict contract as HARModel.

Architecture
------------
Input  : (batch, lookback_days, n_features)
         built internally from a 2-D (T, F) array by sliding a window of
         length ``lookback_days`` over the time axis.
LSTM   : 1 or 2 stacked layers (``depth``), hidden size ``hidden_size``.
         Dropout applied *between* layers when ``depth=2``.
Representation : last hidden state from top LSTM layer
                 [``representation="last_hidden"``].
Head   :
    - ``linear``: Linear(hidden_size → 1)
    - ``mlp``   : Linear(hidden_size → head_hidden_size) + activation
                  (relu/gelu/tanh) + Linear(head_hidden_size → 1)
Output : strictly positive scalar forecast.

Sequence convention
-------------------
For a dataset of T rows, the model predicts position t using
X[t-lookback : t] as context.  Valid predictions start at t = lookback_days,
so ``predict()`` returns a length-T array where the first ``lookback_days``
entries are NaN.

Training
--------
The internal training criterion is controlled by ``LSTMConfig.criterion``
("mse" or "qlike").  The ``loss_fn`` argument accepted by ``fit()`` is used
only for *reporting* the final loss in ``FitResult`` (matching HARModel's API).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .base import BaseModel
from .types import ModelConfig, FitResult


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class LSTMConfig(ModelConfig):
    """
    Configuration for :class:`LSTMModel`.

    Parameters
    ----------
    name : str
        Model identifier used in results and comparisons.
    lookback_days : int
        Sequence length fed to the LSTM (inherited from ModelConfig, default 30).
    hidden_size : int
        Number of hidden units in each LSTM layer.
    depth : int
        Number of stacked LSTM layers.  Must be 1 or 2.
    representation : str
        Sequence representation fed to the prediction head.
        Only ``"last_hidden"`` is currently implemented.
    head : str
        Output head type.
        ``"linear"``: Linear on the selected representation.
        ``"mlp"``: Two-layer MLP head.
    head_hidden_size : int
        Hidden width for ``head="mlp"``.
    activation : str
        Activation used only by ``head="mlp"``.
        ``"relu"`` (default), ``"gelu"`` or ``"tanh"``.
    dropout : float
        Dropout probability applied *between* LSTM layers (only active when
        ``depth=2``).  Ignored for ``depth=1``.
    output_activation : str
        Activation applied to the scalar output to enforce positivity.
        ``"softplus"`` (default): F.softplus(x) + 1e-8 — differentiable and
        strictly positive.
        ``"relu"``    : F.relu(x) + 1e-8 — linear for positive values.
        ``"identity"``: raw linear output (no positivity guarantee).
    criterion : str
        Training loss: ``"mse"`` (mean squared error) or ``"qlike"``
        (QLIKE loss — recommended for RV forecasting).
    optimizer : str
        Gradient-based optimiser.  ``"adam"`` (default) or ``"adamw"``.
        AdamW applies decoupled weight decay and is preferable when
        regularisation matters.
    lr : float
        Learning rate for the optimiser.
    epochs : int
        Maximum number of training epochs.
    batch_size : int
        Mini-batch size.
    patience : int
        Early-stopping patience (epochs without val improvement).
        Set to 0 to disable early stopping.
    min_delta : float
        Minimum decrease in validation loss required to count as an
        improvement for early stopping.
    """

    name: str = "LSTM-RV"
    hidden_size: int = 64
    depth: int = 1
    representation: str = "last_hidden"
    head: str = "linear"
    head_hidden_size: int = 64
    activation: str = "relu"
    dropout: float = 0.0
    output_activation: str = "softplus"
    criterion: str = "mse"
    optimizer: str = "adam"
    lr: float = 1e-3
    epochs: int = 300
    batch_size: int = 32
    patience: int = 30
    min_delta: float = 0

    _VALID_REPRESENTATION = frozenset({"last_hidden"})
    _VALID_DEPTH = frozenset({1, 2})
    _VALID_HEAD = frozenset({"linear", "mlp"})
    _VALID_ACTIVATION = frozenset({"relu", "gelu", "tanh"})
    _VALID_OUTPUT_ACTIVATION = frozenset({"softplus", "relu", "identity"})
    _VALID_CRITERION = frozenset({"mse", "qlike"})
    _VALID_OPTIMIZER = frozenset({"adam", "adamw"})

    def __post_init__(self):
        super().__post_init__()
        if self.representation not in self._VALID_REPRESENTATION:
            raise ValueError(
                f"representation must be one of {set(self._VALID_REPRESENTATION)}."
            )
        if self.depth not in self._VALID_DEPTH:
            raise ValueError(f"depth must be one of {set(self._VALID_DEPTH)}.")
        if self.head not in self._VALID_HEAD:
            raise ValueError(f"head must be one of {set(self._VALID_HEAD)}.")
        if self.activation not in self._VALID_ACTIVATION:
            raise ValueError(f"activation must be one of {set(self._VALID_ACTIVATION)}.")
        if self.output_activation not in self._VALID_OUTPUT_ACTIVATION:
            raise ValueError(f"output_activation must be one of {set(self._VALID_OUTPUT_ACTIVATION)}.")
        if self.criterion not in self._VALID_CRITERION:
            raise ValueError(f"criterion must be one of {set(self._VALID_CRITERION)}.")
        if self.optimizer not in self._VALID_OPTIMIZER:
            raise ValueError(f"optimizer must be one of {set(self._VALID_OPTIMIZER)}.")
        if self.head_hidden_size <= 0:
            raise ValueError("head_hidden_size must be > 0.")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError("dropout must be in [0, 1).")
        if self.patience < 0:
            raise ValueError("patience must be >= 0.")
        if self.min_delta < 0:
            raise ValueError("min_delta must be >= 0.")

    @property
    def kind(self) -> str:
        return "lstm"


# ──────────────────────────────────────────────────────────────────────────────
# Internal PyTorch module
# ──────────────────────────────────────────────────────────────────────────────

class _LSTMNet(nn.Module):
    """
    Thin nn.Module wrapper.

    Parameters
    ----------
    n_features : int
        Number of input features per time step.
    hidden_size : int
    depth : int  (1 or 2)
    activation : str  ("exp" or "relu")
    dropout : float
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int,
        representation: str,
        head: str,
        head_hidden_size: int,
        depth: int,
        activation: str,
        output_activation: str,
        dropout: float,
    ) -> None:
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=depth,
            batch_first=True,
            dropout=float(dropout) if depth > 1 else 0.0,
        )
        self._representation = representation
        self._head_kind = head
        self._output_activation = output_activation
        if head == "linear":
            self.head = nn.Linear(hidden_size, 1)
        else:
            act_layer = {
                "relu": nn.ReLU,
                "gelu": nn.GELU,
                "tanh": nn.Tanh,
            }[activation]
            self.head = nn.Sequential(
                nn.Linear(hidden_size, head_hidden_size),
                act_layer(),
                nn.Linear(head_hidden_size, 1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape (batch, seq_len, n_features)

        Returns
        -------
        Tensor, shape (batch,)
        """
        _, (h_n, _) = self.lstm(x)       # h_n: (depth, batch, hidden_size)
        if self._representation == "last_hidden":
            rep = h_n[-1]  # (batch, hidden_size)
        else:
            raise RuntimeError(f"Unsupported representation: {self._representation}")

        out = self.head(rep).squeeze(-1)   # (batch,)
        if self._output_activation == "softplus":
            return F.softplus(out) + 1e-8
        if self._output_activation == "relu":
            return F.relu(out) + 1e-8
        # identity — raw linear output
        return out


# ──────────────────────────────────────────────────────────────────────────────
# QLIKE training criterion (PyTorch)
# ──────────────────────────────────────────────────────────────────────────────

def _torch_qlike(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """QLIKE on sqrt(RV) inputs: square to RV space, then E[RV_t/RV_p − log(RV_t/RV_p) − 1]."""
    # Convert sqrt(RV) → RV before computing QLIKE
    rv_pred = pred.square().clamp(min=eps)
    rv_target = target.square().clamp(min=eps)
    ratio = (rv_target / rv_pred).clamp(min=eps, max=1e12)
    return (ratio - torch.log(ratio) - 1.0).mean()


# ──────────────────────────────────────────────────────────────────────────────
# LSTMModel
# ──────────────────────────────────────────────────────────────────────────────

class LSTMModel(BaseModel):
    """
    LSTM-based realized-volatility forecasting model.

    Inherits :class:`~src.models.base.BaseModel` and satisfies the same
    ``fit() → FitResult`` / ``predict() → np.ndarray`` contract as
    :class:`~src.models.har.HARModel`.

    Parameters
    ----------
    config : LSTMConfig
        All hyper-parameters (architecture, training, device).
    callback : callable | None
        Optional epoch-level callback with signature
        ``callback(epoch, y_true, y_pred, phase)``.
        ``y_true`` and ``y_pred`` are numpy arrays on the CPU.

    Examples
    --------
    >>> cfg = LSTMConfig(lookback_days=22, depth=2, activation="exp", epochs=50)
    >>> model = LSTMModel(cfg)
    >>> result = model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    >>> preds = model.predict(X_test)   # shape (T,), first 22 entries are NaN
    """

    def __init__(
        self,
        config: LSTMConfig,
        callback: Optional[Callable] = None,
    ) -> None:
        super().__init__(config)
        self.config: LSTMConfig = config
        self.callback = callback

        self._net: Optional[_LSTMNet] = None
        self._n_features: Optional[int] = None
        self._device = torch.device(config.device)

    # ------------------------------------------------------------------ private

    def _build_net(self, n_features: int) -> _LSTMNet:
        cfg = self.config
        net = _LSTMNet(
            n_features=n_features,
            hidden_size=cfg.hidden_size,
            representation=cfg.representation,
            head=cfg.head,
            head_hidden_size=cfg.head_hidden_size,
            depth=cfg.depth,
            activation=cfg.activation,
            output_activation=cfg.output_activation,
            dropout=cfg.dropout,
        ).to(self._device)
        return net

    @staticmethod
    def _make_sequences(
        X: np.ndarray,
        y: np.ndarray,
        lookback: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Slide a window of length ``lookback`` over the time axis.

        X[t-lookback : t]  →  predict  y[t]
        Valid for t = lookback, …, T-1.

        Returns
        -------
        X_seq : (T - lookback, lookback, F)
        y_seq : (T - lookback,)
        """
        T, F = X.shape
        n = T - lookback
        if n <= 0:
            raise ValueError(
                f"Not enough samples ({T}) for lookback_days={lookback}."
            )
        X_seq = np.lib.stride_tricks.sliding_window_view(X, (lookback, F))[
            : -1 or None
        ].reshape(n, lookback, F)
        y_seq = y[lookback:]
        return X_seq.astype(np.float32), y_seq.astype(np.float32)

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

    def _criterion_fn(self) -> Callable:
        if self.config.criterion == "qlike":
            return _torch_qlike
        return F.mse_loss

    # ------------------------------------------------------------------ public

    def fit(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        loss_fn: Optional[Callable] = None,
        **kwargs,
    ) -> FitResult:
        """
        Train the LSTM on ``(X_train, y_train)``.

        Parameters
        ----------
        X_train : array-like, shape (T, n_features)
            Feature matrix in temporal order.
        y_train : array-like, shape (T,)
            Target (e.g. next-day RV).
        X_val : array-like | None
            Validation features.  Used for early stopping when ``patience > 0``.
        y_val : array-like | None
            Validation targets.
        loss_fn : callable | None
            Numpy-based loss function ``(y_true, y_pred) → float`` used only
            for *reporting* in ``FitResult`` (e.g. ``BaseModel.qlike``).
            Defaults to QLIKE.

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

        # Build sequences
        X_seq, y_seq = self._make_sequences(X, y, lookback)

        # Rebuild network (re-init weights on each fit call for rolling windows)
        self._net = self._build_net(self._n_features)
        _optim_cls = (
            torch.optim.AdamW if cfg.optimizer == "adamw" else torch.optim.Adam
        )
        optim = _optim_cls(self._net.parameters(), lr=cfg.lr)
        criterion = self._criterion_fn()

        train_loader = self._to_loader(X_seq, y_seq, shuffle=True)

        has_val = X_val is not None and y_val is not None
        val_loader = None
        if has_val:
            X_v = np.asarray(X_val, dtype=np.float64)
            y_v = np.asarray(y_val, dtype=np.float64).ravel()
            X_seq_v, y_seq_v = self._make_sequences(X_v, y_v, lookback)
            val_loader = self._to_loader(X_seq_v, y_seq_v, shuffle=False)

        best_monitor_loss = float("inf")
        best_state = None
        no_improve = 0

        for epoch in range(cfg.epochs):
            # ── training pass ──
            self._net.train()
            train_loss_accum = 0.0
            train_n = 0
            for X_batch, y_batch in train_loader:
                optim.zero_grad()
                pred = self._net(X_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                optim.step()
                batch_n = int(y_batch.shape[0])
                train_loss_accum += float(loss.item()) * batch_n
                train_n += batch_n

            epoch_train = train_loss_accum / max(train_n, 1)
            monitor_loss = epoch_train

            # ── epoch-level callback & early stopping ──
            if self.callback or (cfg.patience > 0):
                self._net.eval()
                with torch.no_grad():
                    # Collect train predictions for callback
                    if self.callback:
                        X_t_all = torch.from_numpy(X_seq).to(self._device)
                        pred_tr_np = self._net(X_t_all).cpu().numpy().astype(np.float64)
                        self.callback(
                            epoch=epoch,
                            y_true=y_seq.astype(np.float64),
                            y_pred=pred_tr_np,
                            phase="train",
                        )

                    if has_val and val_loader is not None:
                        val_preds, val_targets = [], []
                        for Xb, yb in val_loader:
                            val_preds.append(self._net(Xb))
                            val_targets.append(yb)
                        vp = torch.cat(val_preds)
                        vt = torch.cat(val_targets)
                        epoch_val = float(criterion(vp, vt).item())
                        monitor_loss = epoch_val

                        if self.callback:
                            self.callback(
                                epoch=epoch,
                                y_true=vt.cpu().numpy().astype(np.float64),
                                y_pred=vp.cpu().numpy().astype(np.float64),
                                phase="val",
                            )

                    if cfg.patience > 0:
                        if monitor_loss < best_monitor_loss - cfg.min_delta:
                            best_monitor_loss = monitor_loss
                            best_state = {
                                k: v.clone()
                                for k, v in self._net.state_dict().items()
                            }
                            no_improve = 0
                        else:
                            no_improve += 1
                            if no_improve >= cfg.patience:
                                break

        # Restore best weights if early stopping fired
        if best_state is not None:
            self._net.load_state_dict(best_state)

        self.is_fitted = True

        # ── Report losses using the numpy loss_fn (matching HARModel convention) ──
        if loss_fn is None:
            loss_fn = self.qlike

        y_hat_tr = self.predict(X)                # (T,) with NaN prefix
        valid = ~np.isnan(y_hat_tr)
        tr_loss = float(loss_fn(y[valid], y_hat_tr[valid])) if valid.any() else float("nan")

        val_loss_report: Optional[float] = None
        if has_val:
            y_v_full = np.asarray(y_val, dtype=np.float64).ravel()
            y_hat_v = self.predict(np.asarray(X_val, dtype=np.float64))
            valid_v = ~np.isnan(y_hat_v)
            if valid_v.any():
                val_loss_report = float(loss_fn(y_v_full[valid_v], y_hat_v[valid_v]))

        loss_name = getattr(loss_fn, "__name__", loss_fn.__class__.__name__)

        return FitResult(
            loss_name=loss_name,
            train_loss=tr_loss,
            val_loss=val_loss_report,
            fitted_params={
                "n_features": self._n_features,
                "lookback_days": lookback,
                "representation": cfg.representation,
                "depth": cfg.depth,
                "hidden_size": cfg.hidden_size,
                "head": cfg.head,
                "head_hidden_size": cfg.head_hidden_size,
                "activation": cfg.activation,
            },
            extra={
                "n_train_samples": int(X.shape[0]),
                "n_seq_train": int(X_seq.shape[0]),
                "criterion": cfg.criterion,
                "optimizer": cfg.optimizer,
                "min_delta": cfg.min_delta,
                "monitor": "val" if has_val else "train",
                "best_monitor_loss": best_monitor_loss,
                "epochs_run": epoch + 1,
                "early_stopped": best_state is not None,
            },
        )

    def predict(self, X, **kwargs) -> np.ndarray:
        """
        Generate predictions.

        Parameters
        ----------
        X : array-like, shape (T, n_features)
            Feature matrix in temporal order.

        Returns
        -------
        np.ndarray, shape (T,)
            Predicted values.  The first ``lookback_days`` entries are ``NaN``
            because no full context window is available for those positions.

            For rolling-window evaluation, prepend the last ``lookback_days``
            rows of the training set to ``X`` before calling ``predict()``
            so that every test-period position receives a full context window:

            .. code-block:: python

                X_eval = np.concatenate([X_train[-lookback:], X_test], axis=0)
                preds  = model.predict(X_eval)[lookback:]  # test-period only
        """
        if not self.is_fitted or self._net is None:
            raise RuntimeError("LSTMModel must be fitted before prediction.")

        X = np.asarray(X, dtype=np.float64)

        if X.ndim != 2:
            raise ValueError("X must be a 2-D array (T, n_features).")
        if X.shape[1] != self._n_features:
            raise ValueError(
                f"Feature dimension mismatch: expected {self._n_features}, "
                f"got {X.shape[1]}."
            )

        lookback = self.config.lookback_days
        T = X.shape[0]

        if T <= lookback:
            return np.full(T, np.nan, dtype=np.float64)

        # Build sequences without targets (dummy y not used)
        dummy_y = np.zeros(T, dtype=np.float64)
        X_seq, _ = self._make_sequences(X, dummy_y, lookback)
        # X_seq shape: (T - lookback, lookback, n_features)

        self._net.eval()
        with torch.no_grad():
            X_t = torch.from_numpy(X_seq.astype(np.float32)).to(self._device)
            preds = self._net(X_t).cpu().numpy().astype(np.float64)

        output = np.full(T, np.nan, dtype=np.float64)
        output[lookback:] = preds   # positions lookback … T-1

        return output

    def reset(self) -> None:
        """Discard trained weights (useful before a new rolling-window fit)."""
        super().reset()
        self._net = None
        self._n_features = None
