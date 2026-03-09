import pandas as pd
from pathlib import Path


INTERIM_RV_DIR = Path("../data/interim/daily_rv")
PROCESSED_HAR_DIR = Path("../data/processed/HAR")

PROCESSED_HAR_DIR.mkdir(parents=True, exist_ok=True)


def build_har_dataset(
    coin: str,
    weekly_window: int = 7,
    monthly_window: int = 30
) -> None:
    """
    Build HAR dataset from daily Realized Volatility.

    Parameters
    ----------
    coin : str
        Example: "BTC", "ETH"
    weekly_window : int
        Weekly aggregation window (default=7 for crypto)
    monthly_window : int
        Monthly aggregation window (default=30 for crypto)
    """

    input_file = INTERIM_RV_DIR / f"{coin}_rv.parquet"
    output_file = PROCESSED_HAR_DIR / f"{coin}_har.parquet"

    if not input_file.exists():
        raise FileNotFoundError(f"{input_file} does not exist.")

    # -----------------------------
    # 1. Load daily RV
    # -----------------------------
    df = pd.read_parquet(input_file)

    df = df.sort_index()
    df = df.copy()

    # -----------------------------
    # 2. Construct HAR regressors
    # -----------------------------
    df["RV_d"] = df["RV"]

    df["RV_w"] = (
        df["RV"]
        .rolling(window=weekly_window, min_periods=weekly_window)
        .mean()
    )

    df["RV_m"] = (
        df["RV"]
        .rolling(window=monthly_window, min_periods=monthly_window)
        .mean()
    )

    # -----------------------------
    # 3. Construct target
    # -----------------------------
    df["target"] = df["RV"].shift(-1)

    # -----------------------------
    # 4. Drop incomplete rows
    # -----------------------------
    df.dropna(inplace=True)

    # Keep clean structure
    df = df[["RV_d", "RV_w", "RV_m", "target"]]

    # -----------------------------
    # 5. Save
    # -----------------------------
    df.to_parquet(output_file)

    print(f"HAR dataset saved to: {output_file}")