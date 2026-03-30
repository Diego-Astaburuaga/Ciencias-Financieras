import pandas as pd
import numpy as np
from src.config.paths import INTERIM_DAILY_RV_DIR, RAW_DATA_DIR, RESULTS_DIR


RAW_DIR = RAW_DATA_DIR
INTERIM_RV_DIR = INTERIM_DAILY_RV_DIR
DATA_QUALITY_DIR = RESULTS_DIR / "data_quality"

EXPECTED_OBS_PER_DAY = 288
MIN_COVERAGE_RATIO = 0.90

INTERIM_RV_DIR.mkdir(parents=True, exist_ok=True)
DATA_QUALITY_DIR.mkdir(parents=True, exist_ok=True)


def build_intraday_and_rv(coin: str) -> None:
    """
    Build daily Realized Volatility (RV) from intraday Binance data.

    The raw dataset is expected to contain at least these columns:
    open_time, open, high, low, close, volume

    RV is computed from intraday log-returns of the open price,
    where open_time is the timestamp of each observation.

    Output column ``RV`` is stored on the variance scale (not log scale).

    Parameters
    ----------
    coin : str
        Example: "BTC", "ETH"
    """

    candidate_files = sorted(RAW_DIR.glob(f"{coin}USDT_binance_*.csv"))
    if not candidate_files:
        raise FileNotFoundError(
            f"No raw CSV found for {coin} in {RAW_DIR}."
        )

    input_file = candidate_files[-1]
    output_rv = INTERIM_RV_DIR / f"{coin}_rv.parquet"
    coin_quality_dir = DATA_QUALITY_DIR / coin.upper()
    coin_quality_dir.mkdir(parents=True, exist_ok=True)
    quality_summary_file = coin_quality_dir / "data_quality_summary.csv"
    quality_daily_file = coin_quality_dir / "daily_coverage.csv"

    # -----------------------------
    # 1. Load raw data with the new schema
    # -----------------------------
    expected_cols = ["open_time", "open", "high", "low", "close", "volume"]
    df = pd.read_csv(input_file, usecols=expected_cols)

    if df.empty:
        raise ValueError(f"{input_file} is empty.")

    open_time_numeric = pd.to_numeric(df["open_time"], errors="coerce")
    if open_time_numeric.notna().all():
        df["open_time"] = pd.to_datetime(open_time_numeric, unit="ms", utc=True)
    else:
        df["open_time"] = pd.to_datetime(df["open_time"], utc=True, errors="coerce")

    df.dropna(subset=["open_time"], inplace=True)
    df.set_index("open_time", inplace=True)

    df["open"] = pd.to_numeric(df["open"], errors="coerce")
    df.dropna(subset=["open"], inplace=True)
    df = df[df["open"] > 0]

    df.sort_index(inplace=True)

    # -----------------------------
    # 2. Compute intraday log-returns from open price
    # -----------------------------
    df["log_return"] = np.log(df["open"] / df["open"].shift(1))
    df.dropna(subset=["log_return"], inplace=True)

    # -----------------------------
    # 3. Compute Daily Realized Volatility
    # -----------------------------
    df["date"] = df.index.floor("D")

    # Build a complete daily index to explicitly account for fully missing days.
    all_days = pd.date_range(df["date"].min(), df["date"].max(), freq="D", tz=df["date"].dt.tz)

    daily_obs = (
        df.groupby("date")["log_return"]
        .size()
        .reindex(all_days, fill_value=0)
        .rename("n_obs")
        .to_frame()
    )
    daily_obs.index.name = "date"

    daily_sum_sq = (
        df.groupby("date")["log_return"]
        .apply(lambda x: np.sum(x ** 2))
        .reindex(all_days)
        .rename("sum_sq")
        .to_frame()
    )
    daily_sum_sq.index.name = "date"

    daily_stats = daily_obs.join(daily_sum_sq, how="left")
    daily_stats["coverage_ratio"] = daily_stats["n_obs"] / EXPECTED_OBS_PER_DAY
    daily_stats["is_complete_missing"] = daily_stats["n_obs"] == 0
    daily_stats["is_partial_missing"] = (daily_stats["n_obs"] > 0) & (daily_stats["n_obs"] < EXPECTED_OBS_PER_DAY)

    # Keep RV on a comparable scale across days by annualizing to 288 intraday observations.
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

    daily_rv = daily_stats[["RV"]].copy()

    # If trailing observations are invalid, drop them and report final valid interval.
    last_valid_date = daily_rv["RV"].last_valid_index()
    if last_valid_date is None:
        raise RuntimeError(
            f"No valid RV observations for {coin} after applying {int(MIN_COVERAGE_RATIO * 100)}% coverage rule."
        )
    daily_rv = daily_rv.loc[:last_valid_date].copy()
    daily_stats = daily_stats.loc[daily_rv.index].copy()

    total_days = int(len(daily_stats))
    n_partial_missing_days = int(daily_stats["is_partial_missing"].sum())
    n_complete_missing_days = int(daily_stats["is_complete_missing"].sum())
    pct_partial_missing_days = 100.0 * n_partial_missing_days / total_days if total_days else np.nan
    pct_complete_missing_days = 100.0 * n_complete_missing_days / total_days if total_days else np.nan

    quality_summary = pd.DataFrame(
        {
            "coin": [coin.upper()],
            "expected_obs_per_day": [EXPECTED_OBS_PER_DAY],
            "coverage_threshold": [MIN_COVERAGE_RATIO],
            "valid_start": [str(daily_rv.index.min())],
            "valid_end": [str(daily_rv.index.max())],
            "total_days_in_valid_interval": [total_days],
            "days_with_missing_data": [n_partial_missing_days],
            "pct_days_with_missing_data": [pct_partial_missing_days],
            "days_completely_missed": [n_complete_missing_days],
            "pct_days_completely_missed": [pct_complete_missing_days],
        }
    )

    daily_stats_reset = daily_stats.reset_index().rename(columns={"index": "date"})
    daily_stats_reset["coin"] = coin.upper()

    daily_rv.to_parquet(output_rv)
    quality_summary.to_csv(quality_summary_file, index=False)
    daily_stats_reset.to_csv(quality_daily_file, index=False)

    print(f"Raw intraday file used: {input_file}")
    print(f"Daily RV saved to: {output_rv}")
    print(f"Data quality summary saved to: {quality_summary_file}")
    print(f"Daily coverage details saved to: {quality_daily_file}")
    print(
        f"Valid interval for {coin.upper()}: {daily_rv.index.min()} -> {daily_rv.index.max()}"
    )