import pandas as pd
import numpy as np
from pathlib import Path


RAW_DIR = Path("../data/raw")
INTERIM_RV_DIR = Path("../data/interim/daily_rv")

INTERIM_RV_DIR.mkdir(parents=True, exist_ok=True)


def build_intraday_and_rv(coin: str) -> None:
    """
    Build daily Realized Volatility (RV) from intraday Binance data.

    The raw dataset is expected to contain at least these columns:
    open_time, open, high, low, close, volume

    RV is computed from intraday log-returns of the open price,
    where open_time is the timestamp of each observation.

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

    daily_rv = (
        df
        .groupby("date")["log_return"]
        .apply(lambda x: np.sum(x**2))
        .to_frame(name="RV")
    )

    daily_rv.to_parquet(output_rv)

    print(f"Raw intraday file used: {input_file}")
    print(f"Daily RV saved to: {output_rv}")