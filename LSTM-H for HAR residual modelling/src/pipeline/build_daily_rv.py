from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.config.paths import INTERIM_DAILY_RV_DIR, RAW_DATA_DIR


EXPECTED_OBS_PER_DAY = 288
MIN_COVERAGE_RATIO = 0.90


def _coin_output_dir(base_dir: Path, coin: str) -> Path:
    coin_dir = base_dir / coin.upper()
    coin_dir.mkdir(parents=True, exist_ok=True)
    return coin_dir


def _output_paths(coin: str) -> tuple[Path, Path]:
    coin_dir = _coin_output_dir(INTERIM_DAILY_RV_DIR, coin)
    return coin_dir / "RV_data.parquet", coin_dir / "data_quality_summary.csv"


def _resolve_raw_file(coin: str) -> Path:
    coin = coin.upper()
    candidate_files = sorted(RAW_DATA_DIR.glob(f"{coin}USDT_binance_*.csv"))
    if not candidate_files:
        raise FileNotFoundError(f"No raw CSV found for coin {coin} in {RAW_DATA_DIR}")
    return candidate_files[-1]


def _parse_open_time(series: pd.Series) -> pd.Series:
    numeric_values = pd.to_numeric(series, errors="coerce")
    if numeric_values.notna().all():
        return pd.to_datetime(numeric_values, unit="ms", utc=True)
    return pd.to_datetime(series, utc=True, errors="coerce")


def _build_daily_stats(returns: pd.DataFrame) -> pd.DataFrame:
    all_days = pd.date_range(
        returns["date"].min(),
        returns["date"].max(),
        freq="D",
        tz=returns["date"].dt.tz,
    )

    daily_obs = (
        returns.groupby("date")["log_return"]
        .size()
        .reindex(all_days, fill_value=0)
        .rename("n_obs")
        .to_frame()
    )
    daily_obs.index.name = "date"

    daily_sum_sq = (
        returns.groupby("date")["log_return"]
        .apply(lambda values: float(np.square(values).sum()))
        .reindex(all_days)
        .rename("sum_sq")
        .to_frame()
    )
    daily_sum_sq.index.name = "date"

    daily_stats = daily_obs.join(daily_sum_sq, how="left")
    daily_stats["coverage_ratio"] = daily_stats["n_obs"] / EXPECTED_OBS_PER_DAY
    daily_stats["is_complete_missing"] = daily_stats["n_obs"] == 0
    daily_stats["is_partial_missing"] = (daily_stats["n_obs"] > 0) & (
        daily_stats["n_obs"] < EXPECTED_OBS_PER_DAY
    )
    daily_stats["rv_scaled_sum_sq"] = np.where(
        daily_stats["n_obs"] > 0,
        daily_stats["sum_sq"] * EXPECTED_OBS_PER_DAY / daily_stats["n_obs"],
        np.nan,
    )
    daily_stats["RV"] = np.where(
        daily_stats["coverage_ratio"] >= MIN_COVERAGE_RATIO,
        daily_stats["rv_scaled_sum_sq"],
        np.nan,
    )
    return daily_stats


def _trim_to_valid_interval(coin: str, daily_stats: pd.DataFrame) -> pd.DataFrame:
    last_valid_date = daily_stats["RV"].last_valid_index()
    if last_valid_date is None:
        raise RuntimeError(
            f"No valid RV observations for {coin} after applying {int(MIN_COVERAGE_RATIO * 100)}% coverage rule."
        )
    return daily_stats.loc[:last_valid_date].copy()


def _build_quality_summary(coin: str, daily_stats: pd.DataFrame) -> pd.DataFrame:
    total_days = int(len(daily_stats))
    n_partial_missing_days = int(daily_stats["is_partial_missing"].sum())
    n_complete_missing_days = int(daily_stats["is_complete_missing"].sum())
    pct_partial_missing_days = (
        100.0 * n_partial_missing_days / total_days if total_days else np.nan
    )
    pct_complete_missing_days = (
        100.0 * n_complete_missing_days / total_days if total_days else np.nan
    )
    missing_dates = daily_stats.index[daily_stats["RV"].isna()]
    n_missing_dates_between = int(len(missing_dates))
    missing_dates_between = ";".join(dt.strftime("%Y-%m-%d") for dt in missing_dates)
    rv_dates = daily_stats.index[daily_stats["RV"].notna()]

    return pd.DataFrame(
        {
            "coin": [coin.upper()],
            "expected_obs_per_day": [EXPECTED_OBS_PER_DAY],
            "coverage_threshold": [MIN_COVERAGE_RATIO],
            "init_date": [str(rv_dates.min()) if len(rv_dates) else ""],
            "end_date": [str(rv_dates.max()) if len(rv_dates) else ""],
            "valid_start": [str(daily_stats.index.min())],
            "valid_end": [str(daily_stats.index.max())],
            "total_days_in_valid_interval": [total_days],
            "days_with_missing_data": [n_partial_missing_days],
            "pct_days_with_missing_data": [pct_partial_missing_days],
            "days_completely_missed": [n_complete_missing_days],
            "pct_days_completely_missed": [pct_complete_missing_days],
            "n_missing_dates_between": [n_missing_dates_between],
            "missing_dates_between": [missing_dates_between],
            "rows_saved": [int(daily_stats["RV"].notna().sum())],
        }
    )


def build_daily_rv(coin: str) -> pd.DataFrame:
    coin = coin.upper()
    input_file = _resolve_raw_file(coin)

    df = pd.read_csv(input_file, usecols=["open_time", "open"])
    if df.empty:
        raise ValueError(f"Input file is empty: {input_file}")

    df["timestamp"] = _parse_open_time(df["open_time"])
    df["open"] = pd.to_numeric(df["open"], errors="coerce")
    df = df.dropna(subset=["timestamp", "open"]).copy()
    df = df.loc[df["open"] > 0].copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    if df.empty:
        raise ValueError(f"No valid rows available after cleaning {input_file}")

    df["log_return"] = np.log(df["open"]).diff()
    returns = df.loc[df["log_return"].notna(), ["timestamp", "log_return"]].copy()

    if returns.empty:
        raise ValueError(f"No valid log returns available after cleaning {input_file}")

    returns["date"] = returns["timestamp"].dt.floor("D")

    daily_stats = _build_daily_stats(returns)
    daily_stats = _trim_to_valid_interval(coin, daily_stats)
    daily_rv = daily_stats[["RV"]].dropna().reset_index().rename(columns={"index": "date"})

    output_file, quality_summary_file = _output_paths(coin)
    daily_rv.to_parquet(output_file, index=False)
    quality_summary = _build_quality_summary(coin, daily_stats)
    quality_summary.to_csv(quality_summary_file, index=False)

    print(f"Raw file used: {input_file}")
    print(f"Daily RV saved to: {output_file}")
    print(f"Data quality summary saved to: {quality_summary_file}")
    print(f"Valid interval for {coin}: {daily_rv['date'].min()} -> {daily_rv['date'].max()}")
    print(f"Rows written: {len(daily_rv)}")

    return daily_rv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build daily realized volatility from raw 5-minute Binance data."
    )
    parser.add_argument("coin", help="Coin ticker without the USDT suffix, for example BTC")
    args = parser.parse_args()
    build_daily_rv(args.coin)


if __name__ == "__main__":
    main()
