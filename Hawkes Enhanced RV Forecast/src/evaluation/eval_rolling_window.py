from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.config.paths import INTERIM_DAILY_RV_DIR, RESULTS_DIR
from src.models.LSTM import LSTMConfig, LSTMModel
from src.models.base import BaseModel
from src.models.har import HARModel
from src.models.types import ModelConfig, HARConfig


_FEATURE_ALIASES: dict[str, str] = {
    "perc_event_down": "ratio_event",
    "lambda_hawkes": "lambda_t",
}


@dataclass
class RollingWindowConfig:
    train_window: int = 1200
    test_window: int = 300
    step: int = 300
    lstm_val_window: int = 300
    lstm_n_lags: int = 30
    weekly_window: int = 7
    monthly_window: int = 30
    feature_cols: str | list[str] | None = None
    verbose: bool = True

    def __post_init__(self) -> None:
        if self.train_window <= 0:
            raise ValueError("train_window must be > 0")
        if self.test_window <= 0:
            raise ValueError("test_window must be > 0")
        if self.step <= 0:
            raise ValueError("step must be > 0")
        if self.lstm_val_window <= 0:
            raise ValueError("lstm_val_window must be > 0")
        if self.lstm_val_window >= self.train_window:
            raise ValueError("lstm_val_window must be smaller than train_window")
        if self.lstm_n_lags <= 0:
            raise ValueError("lstm_n_lags must be > 0")
        if self.weekly_window <= 0 or self.monthly_window <= 0:
            raise ValueError("weekly_window and monthly_window must be > 0")

        if self.feature_cols is None:
            return

        requested = [self.feature_cols] if isinstance(self.feature_cols, str) else list(self.feature_cols)
        requested = [_FEATURE_ALIASES.get(name, name) for name in requested]
        self.feature_cols = requested[0] if isinstance(self.feature_cols, str) else requested
        allowed = {"ratio_event", "exceedance_down", "lambda_t"}
        invalid = [name for name in requested if name not in allowed]
        if invalid:
            raise ValueError(
                "feature_cols supports only 'ratio_event', 'exceedance_down', and/or 'lambda_t'. "
                f"Got invalid values: {invalid}"
            )

    @property
    def optional_cols(self) -> list[str]:
        """Validated, ordered list of selected optional covariates."""
        if self.feature_cols is None:
            return []
        requested = (
            [self.feature_cols]
            if isinstance(self.feature_cols, str)
            else list(self.feature_cols)
        )
        _ORDER = ["ratio_event", "exceedance_down", "lambda_t"]
        return [c for c in _ORDER if c in requested]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_EVENT_DEFAULTS: dict[str, float] = {
    "ratio_event": 0.0,
    "exceedance_down": 0.0,
    "lambda_t": 0.0,
}


def _apply_optional_defaults(df: pd.DataFrame) -> None:
    """In-place: resolve legacy column aliases and fill defaults for missing event columns."""
    if "ratio_event" not in df.columns and "perc_event_down" in df.columns:
        df["ratio_event"] = df["perc_event_down"]
    if "ratio_event" not in df.columns and "perc_event" in df.columns:
        df["ratio_event"] = df["perc_event"]
    if "exceedance_down" not in df.columns and "exceedance" in df.columns:
        df["exceedance_down"] = df["exceedance"]
    if "lambda_t" not in df.columns and "lambda_hawkes" in df.columns:
        df["lambda_t"] = df["lambda_hawkes"]
    if "lambda_t" not in df.columns and "lambda_daily" in df.columns:
        df["lambda_t"] = df["lambda_daily"]
    for col, default in _EVENT_DEFAULTS.items():
        if col not in df.columns:
            df[col] = default


# ---------------------------------------------------------------------------
# Pipelines
# ---------------------------------------------------------------------------


class HARPipeline:
    """
    Feature engineering and per-window execution for HAR models.

    Regressors at each time t:
        RV_d  = RV_t
        RV_w  = rolling mean of RV over the past weekly_window days
        RV_m  = rolling mean of RV over the past monthly_window days
        For each selected covariate z_t:
            z_d = z_t
            z_w = rolling mean of z over the past weekly_window days
            z_m = rolling mean of z over the past monthly_window days

    Target: sqrt(RV_{t+1})
    """

    def __init__(
        self,
        weekly_window: int,
        monthly_window: int,
        optional_cols: list[str],
    ) -> None:
        self.weekly_window = weekly_window
        self.monthly_window = monthly_window
        self.optional_cols = optional_cols

    @property
    def feature_cols(self) -> list[str]:
        cols = ["RV_d", "RV_w", "RV_m"]
        for cov in self.optional_cols:
            cols.extend([f"{cov}_d", f"{cov}_w", f"{cov}_m"])
        return cols

    @property
    def feature_tag(self) -> str:
        return "base" if not self.optional_cols else "base+" + "+".join(self.optional_cols)

    def build_features(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """
        Build HAR feature DataFrame from raw daily-RV data.

        Parameters
        ----------
        raw_df : pd.DataFrame
            Must contain column 'RV' plus any required optional covariates.

        Returns
        -------
        pd.DataFrame
            Columns: feature_cols + 'target', indexed by date.
        """
        out = raw_df.copy()
        rv_sqrt = np.sqrt(np.clip(pd.to_numeric(out["RV"], errors="coerce"), 1e-12, None))
        out["RV_d"] = rv_sqrt
        out["RV_w"] = rv_sqrt.rolling(
            window=self.weekly_window, min_periods=self.weekly_window
        ).mean()
        out["RV_m"] = rv_sqrt.rolling(
            window=self.monthly_window, min_periods=self.monthly_window
        ).mean()
        _apply_optional_defaults(out)

        # Build HAR-style aggregated covariate blocks (d/w/m) for each selected z.
        for cov in self.optional_cols:
            out[f"{cov}_d"] = out[cov]
            out[f"{cov}_w"] = out[cov].rolling(
                window=self.weekly_window, min_periods=self.weekly_window
            ).mean()
            out[f"{cov}_m"] = out[cov].rolling(
                window=self.monthly_window, min_periods=self.monthly_window
            ).mean()

        out["target"] = rv_sqrt.shift(-1)
        out.dropna(subset=self.feature_cols + ["target"], inplace=True)
        return out[self.feature_cols + ["target"]]

    def run_window(
        self,
        data: pd.DataFrame,
        start: int,
        train_window: int,
        test_window: int,
        config: HARConfig,
        window_id: int,
        coin: str,
    ) -> pd.DataFrame:
        train_end = start + train_window
        test_end = train_end + test_window

        window = data.iloc[start:test_end]
        train_df = window.iloc[:train_window]
        test_df = window.iloc[train_window:]

        model = HARModel(config)
        model.fit(
            train_df[self.feature_cols].to_numpy(dtype=np.float64),
            train_df["target"].to_numpy(dtype=np.float64),
        )
        y_pred = model.predict(test_df[self.feature_cols].to_numpy(dtype=np.float64))
        y_test = test_df["target"].to_numpy(dtype=np.float64)

        return pd.DataFrame(
            {
                "date": test_df.index,
                "coin": coin,
                "window_id": window_id,
                "window_start": train_df.index[0],
                "window_train_end": train_df.index[-1],
                "window_test_end": test_df.index[-1],
                "model_name": config.name,
                "model_kind": "HAR",
                "y_true": y_test,
                "y_pred": y_pred,
            }
        )


class LSTMPipeline:
    """
    Feature engineering and per-window execution for LSTM models.

    Each time-step vector contains [RV, cov_1, ..., cov_m].  The LSTM receives
    a sliding window of lookback_days such vectors as input, giving the sequence:

        input_t = ( RV_{t-i}, cov_{1,t-i}, ..., cov_{m,t-i} )_{i=1}^{lookback_days}

    Network input shape: (batch, lookback_days, 1 + len(optional_cols)).
    Target: sqrt(RV_{t+1})
    """

    def __init__(self, optional_cols: list[str]) -> None:
        self.optional_cols = optional_cols

    @property
    def feature_cols(self) -> list[str]:
        return ["RV_sqrt"] + self.optional_cols

    @property
    def feature_tag(self) -> str:
        return "base" if not self.optional_cols else "base+" + "+".join(self.optional_cols)

    def build_features(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """
        Build LSTM feature DataFrame from raw daily-RV data.

        Parameters
        ----------
        raw_df : pd.DataFrame
            Must contain column 'RV' plus any required optional covariates.

        Returns
        -------
        pd.DataFrame
            Columns: feature_cols + 'target', indexed by date.
        """
        out = raw_df.copy()
        out["RV_sqrt"] = np.sqrt(np.clip(pd.to_numeric(out["RV"], errors="coerce"), 1e-12, None))
        _apply_optional_defaults(out)
        out["target"] = out["RV_sqrt"].shift(-1)
        out.dropna(subset=self.feature_cols + ["target"], inplace=True)
        return out[self.feature_cols + ["target"]]

    def run_window(
        self,
        data: pd.DataFrame,
        start: int,
        train_window: int,
        test_window: int,
        val_window: int,
        config: LSTMConfig,
        window_id: int,
        coin: str,
    ) -> pd.DataFrame:
        train_end = start + train_window
        test_end = train_end + test_window

        window = data.iloc[start:test_end]
        train_full = window.iloc[:train_window]
        test_df = window.iloc[train_window:]

        split_idx = train_window - val_window
        train_df = train_full.iloc[:split_idx]
        val_df = train_full.iloc[split_idx:]

        lookback = config.lookback_days
        if train_full.shape[0] < lookback:
            raise ValueError(
                f"LSTM lookback_days={lookback} exceeds available train rows ({train_full.shape[0]})"
            )

        model = LSTMModel(config)
        model.fit(
            train_df[self.feature_cols].to_numpy(dtype=np.float64),
            train_df["target"].to_numpy(dtype=np.float64),
            X_val=val_df[self.feature_cols].to_numpy(dtype=np.float64),
            y_val=val_df["target"].to_numpy(dtype=np.float64),
        )

        X_train_full = train_full[self.feature_cols].to_numpy(dtype=np.float64)
        X_test = test_df[self.feature_cols].to_numpy(dtype=np.float64)
        y_test = test_df["target"].to_numpy(dtype=np.float64)

        # Prepend last lookback rows of training data so every test step has full context.
        X_eval = np.concatenate([X_train_full[-lookback:], X_test], axis=0)
        y_pred_full = model.predict(X_eval)
        y_pred = y_pred_full[lookback:]

        if len(y_pred) != len(y_test):
            raise RuntimeError("Unexpected prediction length mismatch in LSTM window")

        return pd.DataFrame(
            {
                "date": test_df.index,
                "coin": coin,
                "window_id": window_id,
                "window_start": train_full.index[0],
                "window_train_end": train_full.index[-1],
                "window_test_end": test_df.index[-1],
                "model_name": config.name,
                "model_kind": "LSTM",
                "y_true": y_test,
                "y_pred": y_pred,
            }
        )


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class RollingWindowEvaluator:
    """
    Evaluate HAR and LSTM models on identical rolling windows.

    Delegates feature engineering and per-window execution to HARPipeline and
    LSTMPipeline respectively.  Both pipelines receive the same raw daily-RV
    data; their feature DataFrames are aligned on the common date index before
    window iteration, ensuring identical evaluation periods.

    Window policy
    -------------
    - Full train window  : train_window rows
    - Test window        : test_window rows
    - Step               : step rows
    - LSTM val split     : last lstm_val_window rows of train used for early stopping
    """

    def __init__(
        self,
        coin: str,
        rolling_config: RollingWindowConfig,
        har_configs: Iterable[ModelConfig] | None = None,
        lstm_configs: Iterable[LSTMConfig] | None = None,
        daily_rv_dir: Path = INTERIM_DAILY_RV_DIR,
    ) -> None:
        self.coin = coin.upper()
        self.cfg = rolling_config
        self.daily_rv_dir = Path(daily_rv_dir)
        self.results_dir = RESULTS_DIR / f"coin={self.coin}"
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.har_configs = list(har_configs) if har_configs is not None else [
            HARConfig(name="HAR-baseline")
        ]
        self.lstm_configs = list(lstm_configs) if lstm_configs is not None else [
            LSTMConfig(name="LSTM-baseline", lookback_days=rolling_config.lstm_n_lags)
        ]

        optional = rolling_config.optional_cols
        self.har_pipeline = HARPipeline(
            weekly_window=rolling_config.weekly_window,
            monthly_window=rolling_config.monthly_window,
            optional_cols=optional,
        )
        self.lstm_pipeline = LSTMPipeline(optional_cols=optional)

    # ------------------------------------------------------------------ utils

    def _log(self, message: str) -> None:
        if self.cfg.verbose:
            print(message)

    @staticmethod
    def _slugify(value: str) -> str:
        cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)
        return cleaned.strip("_") or "unnamed"

    @staticmethod
    def _deduplicate_records(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        key_cols = ["window_id", "date", "model_kind", "model_name"]
        if any(col not in df.columns for col in key_cols):
            return df
        return (
            df.sort_values(key_cols)
            .drop_duplicates(subset=key_cols, keep="last")
            .reset_index(drop=True)
        )

    @staticmethod
    def _load_existing_predictions(path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        df = pd.read_parquet(path)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df

    @staticmethod
    def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        return {
            "mse": BaseModel.mse(y_true, y_pred),
            "rmse": BaseModel.rmse(y_true, y_pred),
            "mae": BaseModel.mae(y_true, y_pred),
            "qlike": BaseModel.qlike(y_true, y_pred),
        }

    def _artifact_paths(self, model_name: str, model_kind: str) -> tuple[Path, Path]:
        pipeline = self.har_pipeline if model_kind.upper() == "HAR" else self.lstm_pipeline
        model_slug = self._slugify(model_name)
        kind_slug = self._slugify(model_kind.upper())
        feature_slug = self._slugify(pipeline.feature_tag)
        prefix = f"kind={kind_slug}__model={model_slug}__features={feature_slug}"
        return (
            self.results_dir / f"predictions__{prefix}.parquet",
            self.results_dir / f"summary__{prefix}.csv",
        )

    # ------------------------------------------------------------------ data

    def _load_daily_rv(self) -> pd.DataFrame:
        rv_path = self.daily_rv_dir / f"{self.coin}_rv.parquet"
        if not rv_path.exists():
            raise FileNotFoundError(f"Daily RV file not found: {rv_path}")
        df = pd.read_parquet(rv_path)
        if "RV" not in df.columns:
            raise ValueError(f"Column 'RV' not found in {rv_path}")
        if pd.to_numeric(df["RV"], errors="coerce").lt(0).any():
            raise ValueError(
                "Detected negative values in column 'RV'. "
                "This pipeline now expects RV on variance scale (non-negative). "
                "Rebuild daily RV data with the updated builder before evaluation."
            )
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, errors="coerce")
        df = df[~df.index.isna()].sort_index().copy()
        if df.empty:
            raise ValueError(f"Daily RV file has no valid rows: {rv_path}")
        self._log(f"Loaded daily RV for {self.coin}: {len(df)} rows from {rv_path}")
        return df

    @staticmethod
    def _iter_window_starts(
        n_rows: int, train_window: int, test_window: int, step: int
    ) -> list[int]:
        span = train_window + test_window
        if n_rows < span:
            raise ValueError(
                f"Not enough rows for rolling evaluation. Need at least {span}, got {n_rows}."
            )
        return list(range(0, n_rows - span + 1, step))

    # ------------------------------------------------------------------ main

    def run(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run rolling-window forecasting comparison.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            - predictions_df : per-date predictions for every model and window.
            - summary_df     : per-model aggregated metrics over all windows.
        """
        raw_df = self._load_daily_rv()
        har_data = self.har_pipeline.build_features(raw_df)
        lstm_data = self.lstm_pipeline.build_features(raw_df)

        # Align on common dates so every rolling window covers identical periods.
        common_idx = har_data.index.intersection(lstm_data.index)
        har_data = har_data.loc[common_idx]
        lstm_data = lstm_data.loc[common_idx]

        starts = self._iter_window_starts(
            len(har_data),
            self.cfg.train_window,
            self.cfg.test_window,
            self.cfg.step,
        )
        self._log(
            f"Feature matrices aligned: {len(har_data)} rows | "
            f"train={self.cfg.train_window}, test={self.cfg.test_window}, "
            f"step={self.cfg.step}, n_windows={len(starts)}"
        )
        self._log(f"Results directory: {self.results_dir}")

        model_jobs: list[tuple[str, ModelConfig | LSTMConfig]] = [
            *[("HAR", c) for c in self.har_configs],
            *[("LSTM", c) for c in self.lstm_configs],
        ]

        all_predictions: list[pd.DataFrame] = []
        summary_rows: list[dict[str, object]] = []

        for model_kind, model_cfg in model_jobs:
            model_name = model_cfg.name
            pred_path, summary_path = self._artifact_paths(model_name, model_kind)
            existing = self._deduplicate_records(self._load_existing_predictions(pred_path))

            ready_windows: set[int] = set()
            if not existing.empty and "window_id" in existing.columns:
                ready_windows = set(existing["window_id"].dropna().astype(int).tolist())

            pipeline = self.har_pipeline if model_kind.upper() == "HAR" else self.lstm_pipeline
            data = har_data if model_kind.upper() == "HAR" else lstm_data

            self._log(
                f"[{model_kind}:{model_name}] feature_tag={pipeline.feature_tag} | "
                f"ready_windows={len(ready_windows)}/{len(starts)}"
            )

            new_parts: list[pd.DataFrame] = []

            for window_id, start in enumerate(starts):
                if window_id in ready_windows:
                    self._log(
                        f"  [{model_kind}:{model_name}] window {window_id}: already completed, skipping"
                    )
                    continue

                self._log(
                    f"  [{model_kind}:{model_name}] window {window_id}: running "
                    f"(start_idx={start}, train={self.cfg.train_window}, test={self.cfg.test_window})"
                )

                if model_kind.upper() == "HAR":
                    window_df = self.har_pipeline.run_window(
                        data=data,
                        start=start,
                        train_window=self.cfg.train_window,
                        test_window=self.cfg.test_window,
                        config=model_cfg,  # type: ignore[arg-type]
                        window_id=window_id,
                        coin=self.coin,
                    )
                else:
                    window_df = self.lstm_pipeline.run_window(
                        data=data,
                        start=start,
                        train_window=self.cfg.train_window,
                        test_window=self.cfg.test_window,
                        val_window=self.cfg.lstm_val_window,
                        config=model_cfg,  # type: ignore[arg-type]
                        window_id=window_id,
                        coin=self.coin,
                    )

                new_parts.append(window_df)

                combined = self._deduplicate_records(
                    pd.concat([existing] + new_parts, ignore_index=True)
                )
                combined.to_parquet(pred_path, index=False)
                self._log(
                    f"  [{model_kind}:{model_name}] window {window_id}: "
                    f"saved {len(window_df)} rows -> {pred_path.name}"
                )

            final_preds = existing
            if new_parts:
                final_preds = self._deduplicate_records(
                    pd.concat([existing] + new_parts, ignore_index=True)
                )

            if final_preds.empty:
                self._log(f"[{model_kind}:{model_name}] no predictions available after processing")
                continue

            metrics = self._metrics(
                final_preds["y_true"].to_numpy(dtype=np.float64),
                final_preds["y_pred"].to_numpy(dtype=np.float64),
            )

            model_summary = {
                "coin": self.coin,
                "model_name": model_name,
                "model_kind": model_kind,
                "feature_tag": pipeline.feature_tag,
                "n_forecasts": int(len(final_preds)),
                "n_windows": int(final_preds["window_id"].nunique()),
                **metrics,
            }
            pd.DataFrame([model_summary]).to_csv(summary_path, index=False)
            self._log(f"[{model_kind}:{model_name}] summary saved -> {summary_path.name}")

            summary_rows.append(model_summary)
            all_predictions.append(final_preds)

        if not all_predictions:
            raise RuntimeError(
                "No predictions were produced. Verify inputs and model/window configuration."
            )

        predictions_df = self._deduplicate_records(
            pd.concat(all_predictions, ignore_index=True)
        )
        summary_df = pd.DataFrame(summary_rows).sort_values("qlike", ascending=True)
        self._log("Rolling-window evaluation completed")
        return predictions_df, summary_df


if __name__ == "__main__":
    rolling_cfg = RollingWindowConfig(
        train_window=1200,
        test_window=300,
        step=300,
        lstm_val_window=300,
    )
    evaluator = RollingWindowEvaluator(coin="BTC", rolling_config=rolling_cfg)
    preds, summary = evaluator.run()
    print(f"Generated {len(preds)} forecasts")
    print(summary)