from __future__ import annotations

import numpy as np


_EPS = 1e-12
_MAX_RATIO = 1e12


def as_1d_pair(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    y_true_arr = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred_arr = np.asarray(y_pred, dtype=np.float64).reshape(-1)

    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    if not (np.all(np.isfinite(y_pred_arr)) and np.all(np.isfinite(y_true_arr))):
        raise ValueError("y_true and y_pred must be finite (no NaN/Inf).")

    return y_true_arr, y_pred_arr


def mse(y_true, y_pred) -> float:
    y_true_arr, y_pred_arr = as_1d_pair(y_true, y_pred)
    return float(np.mean((y_pred_arr - y_true_arr) ** 2))


def mae(y_true, y_pred) -> float:
    y_true_arr, y_pred_arr = as_1d_pair(y_true, y_pred)
    return float(np.mean(np.abs(y_pred_arr - y_true_arr)))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


def qlike(y_true, y_pred, eps: float | None = None, max_ratio: float | None = None) -> float:
    y_true_arr, y_pred_arr = as_1d_pair(y_true, y_pred)

    eps_val = _EPS if eps is None else eps
    max_ratio_val = _MAX_RATIO if max_ratio is None else max_ratio

    rv_true = np.clip(np.square(y_true_arr), eps_val, None)
    rv_pred = np.clip(np.square(y_pred_arr), eps_val, None)

    ratio = np.clip(rv_true / rv_pred, eps_val, max_ratio_val)
    return float(np.mean(ratio - np.log(ratio) - 1.0))
