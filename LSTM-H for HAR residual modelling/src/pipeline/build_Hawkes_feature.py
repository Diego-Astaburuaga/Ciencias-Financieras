from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from src.config.paths import INTERIM_HAWKES_DAILY_DIR, INTERIM_DAILY_RV_JUMPS_DIR, ROOT_DIR
from src.models.Hawkes import ExponentialHawkesMLE, prepare_event_times


DEFAULT_CONFIG: dict[str, object] = {
	"hawkes_window_size": 1000,
	"hawkes_step_size": 8,
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
			continue

		try:
			parsed[key] = int(value)
			continue
		except ValueError:
			pass

		try:
			parsed[key] = float(value)
		except ValueError:
			parsed[key] = value

	return parsed


def load_config(config_path: Path | None = None) -> dict[str, object]:
	cfg = DEFAULT_CONFIG.copy()
	path = ROOT_DIR / "config.yaml" if config_path is None else Path(config_path)
	if path.exists() and path.stat().st_size > 0:
		cfg.update(_parse_simple_yaml(path))

	cfg["hawkes_window_size"] = int(cfg["hawkes_window_size"])
	cfg["hawkes_step_size"] = int(cfg["hawkes_step_size"])

	if cfg["hawkes_window_size"] <= 0:
		raise ValueError("hawkes_window_size must be > 0.")
	if cfg["hawkes_step_size"] <= 0:
		raise ValueError("hawkes_step_size must be > 0.")

	return cfg


def _resolve_jump_file(coin: str) -> Path:
	return INTERIM_DAILY_RV_JUMPS_DIR / coin.upper() / "RV_data.parquet"


def _resolve_output_file(coin: str) -> Path:
	output_dir = INTERIM_HAWKES_DAILY_DIR / coin.upper()
	output_dir.mkdir(parents=True, exist_ok=True)
	return output_dir / "hawkes_feature.parquet"


def _load_jump_series(coin: str) -> pd.DataFrame:
	input_file = _resolve_jump_file(coin)
	if not input_file.exists():
		raise FileNotFoundError(f"Jump data not found: {input_file}")

	jump_df = pd.read_parquet(input_file).copy()
	required_columns = {"date", "jump_indicator"}
	missing_columns = required_columns - set(jump_df.columns)
	if missing_columns:
		raise ValueError(f"Missing jump columns in {input_file}: {sorted(missing_columns)}")

	jump_df["date"] = pd.to_datetime(jump_df["date"], errors="coerce")
	jump_df["jump_indicator"] = pd.to_numeric(jump_df["jump_indicator"], errors="coerce")
	jump_df = jump_df.dropna(subset=["date", "jump_indicator"]).sort_values("date")
	jump_df = jump_df.drop_duplicates(subset=["date"]).reset_index(drop=True)
	jump_df["jump_indicator"] = jump_df["jump_indicator"].astype(int)
	return jump_df


def _build_window_event_times(window_df: pd.DataFrame) -> tuple[Optional[torch.Tensor], Optional[float], float, int, Optional[pd.Timestamp]]:
	event_dates = window_df.loc[window_df["jump_indicator"] == 1, "date"].reset_index(drop=True)
	n_events = int(len(event_dates))
	window_size = int(len(window_df))

	if n_events == 0:
		return None, None, 0.0, 0, None

	first_event_date = event_dates.iloc[0]
	relative_days = (event_dates - first_event_date).dt.days.astype(float)
	scale_factor = n_events / window_size
	transformed_times = prepare_event_times(relative_days.to_numpy() * scale_factor, device="cpu")

	window_end_day = float((window_df["date"].iloc[-1] - first_event_date).days + 1.0)
	transformed_horizon = window_end_day * scale_factor
	return transformed_times, transformed_horizon, scale_factor, n_events, first_event_date


def _initialize_model(previous_params: Optional[dict[str, float]]) -> ExponentialHawkesMLE:
	if previous_params is None:
		return ExponentialHawkesMLE(device="cpu", dtype=torch.float64)

	return ExponentialHawkesMLE(
		init_mu=previous_params["mu"],
		init_alpha=previous_params["alpha"],
		init_beta=previous_params["beta"],
		device="cpu",
		dtype=torch.float64,
	)


def _fit_window_model(
	window_df: pd.DataFrame,
	previous_params: Optional[dict[str, float]],
) -> dict[str, object]:
	event_times, horizon, scale_factor, n_events, first_event_date = _build_window_event_times(window_df)

	if event_times is None or horizon is None or first_event_date is None:
		return {
			"fit_status": "no_events",
			"params": None,
			"event_times": None,
			"scale_factor": scale_factor,
			"n_events": n_events,
			"first_event_date": None,
			"converged": False,
			"neg_loglik": np.nan,
		}

	model = _initialize_model(previous_params)
	fit_result = model.fit(
		event_times=event_times,
		T=horizon,
		lr=1e-2,
		max_iter=1500,
		optimizer_name="adam",
		penalty_stationarity=1e3,
		tol=1e-10,
		verbose=False,
	)

	params = {
		"mu": fit_result.mu,
		"alpha": fit_result.alpha,
		"beta": fit_result.beta,
		"branching_ratio": fit_result.branching_ratio,
	}
	return {
		"fit_status": "fitted",
		"params": params,
		"event_times": event_times.cpu().numpy(),
		"scale_factor": scale_factor,
		"n_events": n_events,
		"first_event_date": first_event_date,
		"converged": fit_result.converged,
		"neg_loglik": fit_result.neg_loglik,
	}


def _compute_future_intensity(
	params: dict[str, float],
	transformed_event_times: np.ndarray,
	transformed_eval_times: np.ndarray,
) -> np.ndarray:
	mu = float(params["mu"])
	alpha = float(params["alpha"])
	beta = float(params["beta"])

	deltas = transformed_eval_times[:, None] - transformed_event_times[None, :]
	excitation = np.exp(-beta * deltas).sum(axis=1)
	return mu + alpha * excitation


def _build_window_starts(n_rows: int, window_size: int, step_size: int) -> list[int]:
	return list(range(0, n_rows - window_size, step_size))


def _print_window_progress(
	window_number: int,
	total_windows: int,
	window_df: pd.DataFrame,
	future_df: pd.DataFrame,
	fit_status: str,
	n_events: int,
) -> None:
	window_start = pd.Timestamp(window_df["date"].iloc[0]).date()
	window_end = pd.Timestamp(window_df["date"].iloc[-1]).date()
	forecast_start = pd.Timestamp(future_df["date"].iloc[0]).date()
	forecast_end = pd.Timestamp(future_df["date"].iloc[-1]).date()
	print(
		f"Processing Hawkes window {window_number}/{total_windows} | "
		f"train=[{window_start}, {window_end}] | "
		f"forecast=[{forecast_start}, {forecast_end}] | "
		f"events={n_events} | status={fit_status}"
	)


def build_hawkes_feature(
	coin: str,
	config_path: Path | None = None,
) -> pd.DataFrame:
	coin = coin.upper()
	cfg = load_config(config_path)
	window_size = int(cfg["hawkes_window_size"])
	step_size = int(cfg["hawkes_step_size"])

	jump_df = _load_jump_series(coin)
	if len(jump_df) <= window_size:
		raise ValueError(
			f"Not enough rows for Hawkes features. Need more than {window_size}, got {len(jump_df)}."
		)

	rows: list[dict[str, object]] = []
	previous_params: Optional[dict[str, float]] = None
	window_starts = _build_window_starts(
		n_rows=len(jump_df),
		window_size=window_size,
		step_size=step_size,
	)
	total_windows = len(window_starts)

	if total_windows == 0:
		raise ValueError(
			"No Hawkes rolling windows could be created with the current window and step sizes."
		)

	print(
		f"Building Hawkes features for {coin}: "
		f"{total_windows} rolling windows, window_size={window_size}, step_size={step_size}"
	)

	for window_number, start in enumerate(window_starts, start=1):
		train_end = start + window_size
		forecast_end = min(train_end + step_size, len(jump_df))

		window_df = jump_df.iloc[start:train_end].copy()
		future_df = jump_df.iloc[train_end:forecast_end].copy()
		if future_df.empty:
			continue

		fit_state = _fit_window_model(window_df, previous_params)
		_print_window_progress(
			window_number=window_number,
			total_windows=total_windows,
			window_df=window_df,
			future_df=future_df,
			fit_status=str(fit_state["fit_status"]),
			n_events=int(fit_state["n_events"]),
		)

		if fit_state["params"] is not None:
			previous_params = fit_state["params"]

		if fit_state["fit_status"] == "no_events":
			for _, future_row in future_df.iterrows():
				rows.append(
					{
						"date": future_row["date"],
						"lambda_t": 0.0,
						"mu": 0.0,
						"alpha": 0.0,
						"beta": np.nan,
						"branching_ratio": 0.0,
						"window_start": window_df["date"].iloc[0],
						"window_end": window_df["date"].iloc[-1],
						"n_events_window": int(fit_state["n_events"]),
						"scale_factor": float(fit_state["scale_factor"]),
						"fit_status": "no_events",
						"converged": False,
						"neg_loglik": np.nan,
					}
				)
			continue

		first_event_date = fit_state["first_event_date"]
		scale_factor = float(fit_state["scale_factor"])
		transformed_event_times = np.asarray(fit_state["event_times"], dtype=np.float64)
		forecast_offsets = (future_df["date"] - first_event_date).dt.days.astype(float).to_numpy() + 1.0
		transformed_eval_times = forecast_offsets * scale_factor
		lambda_values = _compute_future_intensity(
			params=fit_state["params"],
			transformed_event_times=transformed_event_times,
			transformed_eval_times=transformed_eval_times,
		)

		for future_date, lambda_value in zip(future_df["date"], lambda_values):
			rows.append(
				{
					"date": future_date,
					"lambda_t": float(lambda_value),
					"mu": float(fit_state["params"]["mu"]),
					"alpha": float(fit_state["params"]["alpha"]),
					"beta": float(fit_state["params"]["beta"]),
					"branching_ratio": float(fit_state["params"]["branching_ratio"]),
					"window_start": window_df["date"].iloc[0],
					"window_end": window_df["date"].iloc[-1],
					"n_events_window": int(fit_state["n_events"]),
					"scale_factor": scale_factor,
					"fit_status": str(fit_state["fit_status"]),
					"converged": bool(fit_state["converged"]),
					"neg_loglik": float(fit_state["neg_loglik"]),
				}
			)

	feature_df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
	output_file = _resolve_output_file(coin)
	feature_df.to_parquet(output_file, index=False)

	print(f"Jump input used: {_resolve_jump_file(coin)}")
	print(f"Hawkes features saved to: {output_file}")
	print(f"Rows written: {len(feature_df)}")
	print(f"Completed Hawkes windows: {total_windows}/{total_windows}")

	return feature_df


def build_Hawkes_feature(
	coin: str,
	config_path: Path | None = None,
) -> pd.DataFrame:
	return build_hawkes_feature(coin=coin, config_path=config_path)


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Build rolling Hawkes intensity features from daily BNS jump indicators."
	)
	parser.add_argument("coin", help="Coin ticker without the USDT suffix, for example BTC")
	parser.add_argument(
		"--config",
		default=str(ROOT_DIR / "config.yaml"),
		help="Path to config YAML file.",
	)
	args = parser.parse_args()
	build_hawkes_feature(coin=args.coin, config_path=Path(args.config))


if __name__ == "__main__":
	main()
