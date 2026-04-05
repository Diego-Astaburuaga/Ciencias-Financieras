from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.config.paths import RESULTS_DIR, ROOT_DIR
from src.models import LSTMConfig, LSTMModel


DEFAULT_CONFIG: dict[str, object] = {
	"lstm_train_window_size": 1000,
	"lstm_prediction_window_size": 8,
	"lstm_val_ratio": 0.25,
	"lstm_lookback_days": 30,
	"lstm_hidden_size": 64,
	"lstm_output_activation": "linear",
	"lstm_criterion": "mse",
	"lstm_optimizer": "adam",
	"lstm_lr": 1e-3,
	"lstm_epochs": 300,
	"lstm_batch_size": 32,
	"lstm_patience": 30,
	"lstm_min_delta": 0.0,
	"lstm_device": "cpu",
}


def _parse_simple_yaml(path: Path) -> dict[str, object]:
	parsed: dict[str, object] = {}
	for raw_line in path.read_text(encoding="utf-8").splitlines():
		line = raw_line.split("#", 1)[0].strip()
		if not line or ":" not in line:
			continue
		key, value = line.split(":", 1)
		key = key.strip()
		value = value.strip().strip('"').strip("'")
		lowered = value.lower()
		if lowered in {"true", "false"}:
			parsed[key] = lowered == "true"
		else:
			try:
				parsed[key] = int(value)
			except ValueError:
				try:
					parsed[key] = float(value)
				except ValueError:
					parsed[key] = value
	return parsed


def load_config(config_path: Path) -> dict[str, object]:
	cfg = DEFAULT_CONFIG.copy()
	if config_path.exists() and config_path.stat().st_size > 0:
		cfg.update(_parse_simple_yaml(config_path))

	cfg["lstm_output_activation"] = str(cfg["lstm_output_activation"]).lower().strip()
	cfg["lstm_criterion"] = str(cfg["lstm_criterion"]).lower().strip()
	cfg["lstm_optimizer"] = str(cfg["lstm_optimizer"]).lower().strip()
	cfg["lstm_device"] = str(cfg["lstm_device"]).strip()

	for key in (
		"lstm_train_window_size",
		"lstm_prediction_window_size",
		"lstm_lookback_days",
		"lstm_hidden_size",
		"lstm_epochs",
		"lstm_batch_size",
		"lstm_patience",
	):
		cfg[key] = int(cfg[key])
		if cfg[key] <= 0:
			raise ValueError(f"{key} must be > 0.")

	cfg["lstm_lr"] = float(cfg["lstm_lr"])
	cfg["lstm_min_delta"] = float(cfg["lstm_min_delta"])
	cfg["lstm_val_ratio"] = float(cfg["lstm_val_ratio"])

	if not (0.0 < cfg["lstm_val_ratio"] < 1.0):
		raise ValueError("lstm_val_ratio must be in (0, 1).")

	if cfg["lstm_output_activation"] not in {"linear", "relu", "tanh", "gelu", "softplus"}:
		raise ValueError("lstm_output_activation must be one of: linear, relu, tanh, gelu, softplus.")
	if cfg["lstm_criterion"] not in {"mse", "mae", "hmse", "hmae"}:
		raise ValueError("lstm_criterion must be one of: mse, mae, hmse, hmae.")
	if cfg["lstm_optimizer"] not in {"adam", "adamw"}:
		raise ValueError("lstm_optimizer must be 'adam' or 'adamw'.")

	return cfg


def _load_har_forecast(coin: str) -> pd.DataFrame:
	coin = coin.upper()
	har_file = RESULTS_DIR / "har_forecast" / coin / "rolling_forecast.parquet"
	if not har_file.exists():
		raise FileNotFoundError(f"HAR rolling forecast not found: {har_file}")

	df = pd.read_parquet(har_file).copy()
	required = {"date", "y_true", "y_pred"}
	missing = required - set(df.columns)
	if missing:
		raise ValueError(f"Missing required HAR columns in {har_file}: {sorted(missing)}")

	df["date"] = pd.to_datetime(df["date"], errors="coerce")
	df = df.dropna(subset=["date"]).sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)
	df["y_t"] = pd.to_numeric(df["y_true"], errors="coerce")
	df["y_t^HAR"] = pd.to_numeric(df["y_pred"], errors="coerce")
	df["e_t"] = df["y_t^HAR"] - df["y_t"]
	df = df.dropna(subset=["y_t", "y_t^HAR", "e_t"]).reset_index(drop=True)
	return df[["date", "y_t", "y_t^HAR", "e_t"]]


def create_LSTM_data(coin: str, config: dict[str, object] | None = None) -> pd.DataFrame:
	"""
	Build supervised LSTM table in data/LSTM with columns:
	- y_{t+1} (target proxy from true series)
	- y_t, y_{t-1}, ..., y_{t-L}
	- e_t, e_{t-1}, ..., e_{t-L}
	"""
	coin = coin.upper()
	cfg = DEFAULT_CONFIG.copy() if config is None else config
	lookback = int(cfg["lstm_lookback_days"])

	base = _load_har_forecast(coin)
	if len(base) <= lookback + 1:
		raise ValueError(
			f"Not enough rows to build LSTM data with lookback={lookback}. Got {len(base)} rows."
		)

	rows: list[dict[str, float | pd.Timestamp]] = []
	for idx in range(lookback, len(base) - 1):
		row: dict[str, float | pd.Timestamp] = {
			"date": base.loc[idx + 1, "date"],
			"y_{t+1}": float(base.loc[idx + 1, "y_t"]),
			"e_{t+1}": float(base.loc[idx + 1, "e_t"]),
		}
		for lag in range(lookback + 1):
			row[f"y_{{t-{lag}}}"] = float(base.loc[idx - lag, "y_t"])
			row[f"e_{{t-{lag}}}"] = float(base.loc[idx - lag, "e_t"])
		rows.append(row)

	dataset = pd.DataFrame(rows)
	dataset = dataset.sort_values("date").reset_index(drop=True)

	out_dir = ROOT_DIR / "data" / "LSTM"
	out_dir.mkdir(parents=True, exist_ok=True)
	output_file = out_dir / f"{coin}_LSTM_dataset.parquet"
	dataset.to_parquet(output_file, index=False)

	print(f"LSTM dataset built for {coin}: {output_file}")
	print(f"Rows: {len(dataset)} | Range: {dataset['date'].min()} -> {dataset['date'].max()}")
	return dataset


def _split_train_val(
	train_df: pd.DataFrame,
	val_ratio: float,
	lookback: int,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
	n = len(train_df)
	n_val = max(1, int(round(n * val_ratio)))
	n_train = n - n_val

	# Keep enough rows in both splits for sequence creation.
	if n_train <= lookback:
		n_train = lookback + 1
		n_val = n - n_train

	if n_val <= lookback:
		return train_df, None

	train_part = train_df.iloc[:n_train].copy()
	val_part = train_df.iloc[n_train:].copy()
	return train_part, val_part


def run_lstm_rolling_forecast(coin: str, config: dict[str, object]) -> pd.DataFrame:
	coin = coin.upper()
	series = _load_har_forecast(coin)

	train_window = int(config["lstm_train_window_size"])
	pred_window = int(config["lstm_prediction_window_size"])
	step = pred_window
	lookback = int(config["lstm_lookback_days"])
	val_ratio = float(config["lstm_val_ratio"])

	if train_window <= lookback:
		raise ValueError("lstm_train_window_size must be > lstm_lookback_days.")
	if len(series) < train_window + pred_window:
		raise ValueError(
			f"Not enough rows for rolling LSTM. Need at least {train_window + pred_window}, got {len(series)}."
		)

	model_cfg = LSTMConfig(
		name=f"LSTM-{coin}",
		device=str(config["lstm_device"]),
		lookback_days=lookback,
		hidden_size=int(config["lstm_hidden_size"]),
		representation="lstm_hidden",
		output_activation=str(config["lstm_output_activation"]),
		criterion=str(config["lstm_criterion"]),
		optimizer=str(config["lstm_optimizer"]),
		lr=float(config["lstm_lr"]),
		epochs=int(config["lstm_epochs"]),
		batch_size=int(config["lstm_batch_size"]),
		patience=int(config["lstm_patience"]),
		min_delta=float(config["lstm_min_delta"]),
	)

	model = LSTMModel(config=model_cfg)
	preds: list[pd.DataFrame] = []
	starts = range(0, len(series) - (train_window + pred_window) + 1, step)
	total_windows = len(starts)
	print(f"Running LSTM rolling forecast for {coin}: 0/{total_windows}")

	for window_id, start in enumerate(starts):
		print(f"LSTM rolling window {window_id + 1}/{total_windows}", flush=True)
		train_end = start + train_window
		test_end = train_end + pred_window

		train_df = series.iloc[start:train_end].copy()
		test_df = series.iloc[train_end:test_end].copy()

		train_part, val_part = _split_train_val(train_df, val_ratio=val_ratio, lookback=lookback)

		X_train = train_part[["y_t", "e_t"]].to_numpy(dtype=np.float64)
		y_train = train_part["e_t"].to_numpy(dtype=np.float64)

		X_val = None
		y_val = None
		if val_part is not None and len(val_part) > lookback:
			X_val = val_part[["y_t", "e_t"]].to_numpy(dtype=np.float64)
			y_val = val_part["e_t"].to_numpy(dtype=np.float64)

		model.fit(
			X_train=X_train,
			y_train=y_train,
			X_val=X_val,
			y_val=y_val,
			warm_start=(window_id > 0),
		)

		# Recursive prediction in the test slice using predicted residuals in history.
		history_y = train_df["y_t"].to_list()
		history_e = train_df["e_t"].to_list()

		rows: list[dict[str, object]] = []
		for _, r in test_df.iterrows():
			seq_y = np.asarray(history_y[-lookback:], dtype=np.float64)
			seq_e = np.asarray(history_e[-lookback:], dtype=np.float64)
			X_single = np.column_stack([seq_y, seq_e])

			e_lstm = float(model.predict(X_single, batched=False))
			y_har = float(r["y_t^HAR"])
			y_true = float(r["y_t"])
			e_true = float(r["e_t"])
			y_lstm = y_har + e_lstm

			rows.append(
				{
					"date": r["date"],
					"y_t": y_true,
					"y_t^HAR": y_har,
					"e_t": e_true,
					"e_t^LSTM": e_lstm,
					"y_t^LSTM": y_lstm,
				}
			)

			history_y.append(y_true)
			history_e.append(e_lstm)

		preds.append(pd.DataFrame(rows))

	forecast = pd.concat(preds, ignore_index=True)
	out_dir = RESULTS_DIR / "lstm_forecast" / coin
	out_dir.mkdir(parents=True, exist_ok=True)
	output_file = out_dir / "rolling_forecast.parquet"
	forecast.to_parquet(output_file, index=False)

	print(f"Rolling LSTM forecast saved to: {output_file}")
	print(f"Forecast rows: {len(forecast)}")
	return forecast


def main() -> None:
	parser = argparse.ArgumentParser(description="Create LSTM data and run rolling LSTM residual forecasts.")
	parser.add_argument("coin", help="Coin ticker without USDT suffix, for example BTC")
	parser.add_argument(
		"--config",
		default=str(ROOT_DIR / "config.yaml"),
		help="Path to config YAML file.",
	)
	args = parser.parse_args()

	cfg = load_config(Path(args.config))
	create_LSTM_data(args.coin, config=cfg)
	run_lstm_rolling_forecast(args.coin, config=cfg)


if __name__ == "__main__":
	main()
