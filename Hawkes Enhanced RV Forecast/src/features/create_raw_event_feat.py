from pathlib import Path

import numpy as np
import pandas as pd
from src.config.paths import INTERIM_DAILY_RV_DIR, INTERIM_EVENT_5_MIN_DIR


EVENT_DIR = INTERIM_EVENT_5_MIN_DIR
INTERIM_RV_DIR = INTERIM_DAILY_RV_DIR


def _normalize_daily_timestamps(values):
	"""Convert timestamps to normalized daily dates for reliable joins."""
	normalized = pd.to_datetime(values, errors="coerce", utc=True)

	if isinstance(normalized, pd.Series):
		return normalized.dt.floor("D").dt.tz_localize(None)

	return normalized.floor("D").tz_localize(None)


def build_daily_n_events_feature(
	coin: str,
	event_dir: Path = EVENT_DIR,
	rv_dir: Path = INTERIM_RV_DIR,
	intervals_per_day: int = 288,
	eps: float = 1e-12,
) -> pd.DataFrame:
	"""
	Aggregate 5-minute exceedance values into scaled daily features and attach
	them to the corresponding daily RV dataset.

	Each 5-minute mark_t entry is ReLU(y_true_down - q_down): positive when an
	extreme loss occurs, zero otherwise.

	Parameters
	----------
	coin : str
		Coin identifier, for example "BTC".
	event_dir : Path
		Directory containing coin-level 5-minute event parquet files.
	rv_dir : Path
		Directory containing coin-level daily RV parquet files.
	intervals_per_day : int
		Number of intraday intervals per day (default=288 for 5-minute bars).
	eps : float
		Small positive constant for numerical stability.

	Returns
	-------
	pd.DataFrame
		Updated daily RV dataframe with two additional features:
		``ratio_event``, ``exceedance_down``.
	"""
	event_path = Path(event_dir) / f"{coin}_5m_events.parquet"
	rv_path = Path(rv_dir) / f"{coin}_rv.parquet"

	if not event_path.exists():
		raise FileNotFoundError(f"Event file not found for {coin}: {event_path}")
	if not rv_path.exists():
		raise FileNotFoundError(f"Daily RV file not found for {coin}: {rv_path}")
	if intervals_per_day <= 0:
		raise ValueError("intervals_per_day must be > 0")
	if eps <= 0:
		raise ValueError("eps must be > 0")

	events_df = pd.read_parquet(event_path)
	rv_df = pd.read_parquet(rv_path)

	if "time" not in events_df.columns:
		raise ValueError(f"{event_path} is missing required columns: time")

	if "mark_t" in events_df.columns:
		event_marks = pd.to_numeric(events_df["mark_t"], errors="coerce").fillna(0.0)
	elif "Event_down" in events_df.columns:
		event_marks = pd.to_numeric(events_df["Event_down"], errors="coerce").fillna(0.0)
	elif "Event" in events_df.columns:
		event_marks = pd.to_numeric(events_df["Event"], errors="coerce").fillna(0.0)
	else:
		raise ValueError(f"{event_path} must contain 'mark_t' (or legacy 'Event_down'/'Event').")

	if rv_df.empty:
		for col in ("ratio_event", "exceedance_down"):
			rv_df[col] = pd.Series(dtype="float64")
		rv_df.to_parquet(rv_path)
		return rv_df

	def _daily_agg(events_series: pd.Series) -> tuple[pd.Series, pd.Series]:
		"""Return (daily_perc, daily_exceedance) for a given event series."""
		grouped = (
			events_df.assign(date=_normalize_daily_timestamps(events_df["time"]))
			.dropna(subset=["date"])
			.assign(_ev=events_series.values)
			.groupby("date")["_ev"]
		)
		daily_n = grouped.apply(lambda x: (x > 0).sum()).astype("float64")
		daily_perc = daily_n / float(intervals_per_day)
		daily_exc = grouped.apply(
			lambda x: float(np.sqrt(np.clip(np.square(np.asarray(x, dtype="float64")).sum(), eps, None)))
		)
		return daily_perc, daily_exc

	daily_perc_down, daily_exc_down = _daily_agg(event_marks)

	rv_dates = pd.Series(
		_normalize_daily_timestamps(rv_df.index),
		index=rv_df.index,
	)

	rv_df["ratio_event"] = rv_dates.map(daily_perc_down).fillna(0.0).astype("float64")
	rv_df["exceedance_down"] = rv_dates.map(daily_exc_down).fillna(0.0).astype("float64")
	rv_df.to_parquet(rv_path)

	print(f"{coin}: added 'ratio_event', 'exceedance_down' to {rv_path}")

	return rv_df


def build_all_daily_n_events_features(
	event_dir: Path = EVENT_DIR,
	rv_dir: Path = INTERIM_RV_DIR,
	intervals_per_day: int = 288,
	eps: float = 1e-12,
) -> dict[str, Path]:
	"""
Build and attach daily ratio_event and exceedance_down features for every
	coin that has an event parquet file available in the interim events directory.

	Returns
	-------
	dict[str, Path]
		Mapping from coin symbol to updated daily RV parquet path.
	"""
	event_dir = Path(event_dir)
	rv_dir = Path(rv_dir)

	event_files = sorted(event_dir.glob("*_5m_events.parquet"))
	if not event_files:
		raise FileNotFoundError(f"No event parquet files found in {event_dir}")

	updated_files = {}

	for event_file in event_files:
		coin = event_file.stem.replace("_5m_events", "")
		build_daily_n_events_feature(
			coin=coin,
			event_dir=event_dir,
			rv_dir=rv_dir,
			intervals_per_day=intervals_per_day,
			eps=eps,
		)
		updated_files[coin] = rv_dir / f"{coin}_rv.parquet"

	return updated_files


if __name__ == "__main__":
	updated = build_all_daily_n_events_features()
	print(f"Updated {len(updated)} daily RV files with ratio_event, exceedance_down.")
