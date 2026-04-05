from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.config.paths import INTERIM_DAILY_RV_DIR, RESULTS_DIR, ROOT_DIR
from src.models import HARConfig, HARModel


DEFAULT_CONFIG: dict[str, object] = {
	"fitting_har_method": "ols",
	"har_train_window_size": 1000,
	"har_prediction_window_size": 8,
	"RV_transformation": "sqrt",
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

	cfg["fitting_har_method"] = str(cfg["fitting_har_method"]).lower().strip()
	cfg["RV_transformation"] = str(cfg["RV_transformation"]).lower().strip()

	if cfg["fitting_har_method"] not in {"ols", "wls"}:
		raise ValueError("fitting_har_method must be 'ols' or 'wls'.")
	if cfg["RV_transformation"] not in {"sqrt", "log"}:
		raise ValueError("RV_transformation must be 'sqrt' or 'log'.")

	for key in ("har_train_window_size", "har_prediction_window_size"):
		cfg[key] = int(cfg[key])
		if cfg[key] <= 0:
			raise ValueError(f"{key} must be > 0.")

	return cfg


def _transform_rv(rv: pd.Series, transformation: str) -> pd.Series:
	if transformation == "sqrt":
		return np.sqrt(np.clip(pd.to_numeric(rv, errors="coerce"), 1e-12, None))
	if transformation == "log":
		rv_num = pd.to_numeric(rv, errors="coerce")
		rv_num = rv_num.where(rv_num > 0)
		return np.log(rv_num)
	raise ValueError("RV_transformation must be 'sqrt' or 'log'.")


def _resolve_daily_rv_file(coin: str) -> Path:
	coin = coin.upper()
	new_layout = INTERIM_DAILY_RV_DIR / coin / "RV_data.parquet"
	legacy_layout = INTERIM_DAILY_RV_DIR / f"{coin}.parquet"
	if new_layout.exists():
		return new_layout
	if legacy_layout.exists():
		return legacy_layout
	raise FileNotFoundError(f"Daily RV input not found: {new_layout}")


def create_HAR_data(coin: str, config: dict[str, object] | None = None) -> pd.DataFrame:
	coin = coin.upper()
	cfg = DEFAULT_CONFIG.copy() if config is None else config
	rv_transformation = str(cfg["RV_transformation"])

	input_file = _resolve_daily_rv_file(coin)

	df = pd.read_parquet(input_file).copy()
	if "RV" not in df.columns:
		raise ValueError(f"Column 'RV' not found in {input_file}")

	if "date" in df.columns:
		df["date"] = pd.to_datetime(df["date"], errors="coerce")
		df = df.dropna(subset=["date"]).sort_values("date").drop_duplicates(subset=["date"]).set_index("date")
	elif not isinstance(df.index, pd.DatetimeIndex):
		df.index = pd.to_datetime(df.index, errors="coerce")

	df = df[~df.index.isna()].sort_index().copy()
	df["y_t"] = _transform_rv(df["RV"], rv_transformation)
	df["y_d"] = df["y_t"]
	df["y_w"] = df["y_t"].rolling(window=7, min_periods=7).mean()
	df["y_m"] = df["y_t"].rolling(window=30, min_periods=30).mean()
	df["target"] = df["y_t"].shift(-1)

	dataset = df[["y_d", "y_w", "y_m", "target"]].dropna().copy()
	dataset.index.name = "date"

	out_dir = ROOT_DIR / "data" / "HAR"
	out_dir.mkdir(parents=True, exist_ok=True)
	output_file = out_dir / f"{coin}_HAR_dataset.parquet"
	dataset.to_parquet(output_file)

	print(f"HAR dataset built for {coin}: {output_file}")
	print(f"Rows: {len(dataset)} | Range: {dataset.index.min()} -> {dataset.index.max()}")
	return dataset


def run_har_rolling_forecast(coin: str, config: dict[str, object]) -> pd.DataFrame:
	coin = coin.upper()
	data = create_HAR_data(coin, config=config)

	train_window = int(config["har_train_window_size"])
	pred_window = int(config["har_prediction_window_size"])
	step = pred_window

	if len(data) < train_window + pred_window:
		raise ValueError(
			f"Not enough rows for rolling HAR. Need at least {train_window + pred_window}, got {len(data)}."
		)

	model_cfg = HARConfig(
		name=f"HAR-{coin}",
		device="cpu",
		fit_method=str(config["fitting_har_method"]),
	)

	preds: list[pd.DataFrame] = []
	starts = range(0, len(data) - (train_window + pred_window) + 1, step)

	for window_id, start in enumerate(starts):
		train_end = start + train_window
		test_end = train_end + pred_window

		train_df = data.iloc[start:train_end]
		test_df = data.iloc[train_end:test_end]

		model = HARModel(config=model_cfg)
		fit_state = model.fit(
			X_train=train_df[["y_d", "y_w", "y_m"]].to_numpy(dtype=np.float64),
			y_train=train_df["target"].to_numpy(dtype=np.float64),
		)

		y_pred = model.predict(
			test_df[["y_d", "y_w", "y_m"]].to_numpy(dtype=np.float64),
			batched=True,
		)

		preds.append(
			pd.DataFrame(
				{
					"date": test_df.index,
					"coin": coin,
					"window_id": window_id,
					"fit_method": fit_state["method"],
					"y_true": test_df["target"].to_numpy(dtype=np.float64),
					"y_pred": y_pred,
				}
			)
		)

	forecast = pd.concat(preds, ignore_index=True)
	out_dir = RESULTS_DIR / "har_forecast" / coin
	out_dir.mkdir(parents=True, exist_ok=True)
	output_file = out_dir / "rolling_forecast.parquet"
	forecast.to_parquet(output_file, index=False)

	print(f"Rolling HAR forecast saved to: {output_file}")
	print(f"Forecast rows: {len(forecast)}")
	return forecast


def main() -> None:
	parser = argparse.ArgumentParser(description="Create HAR data and run rolling HAR forecasts.")
	parser.add_argument("coin", help="Coin ticker without USDT suffix, for example BTC")
	parser.add_argument(
		"--config",
		default=str(ROOT_DIR / "config.yaml"),
		help="Path to config YAML file.",
	)
	args = parser.parse_args()

	cfg = load_config(Path(args.config))
	run_har_rolling_forecast(args.coin, config=cfg)


if __name__ == "__main__":
	main()
