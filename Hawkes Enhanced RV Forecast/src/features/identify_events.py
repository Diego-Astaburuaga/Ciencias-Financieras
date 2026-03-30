#root/src/features/identify_events.py
# This module implements the quantile regression approach to identify extreme return events in cryptocurrency data.

# data path: root/data/raw/coinUSDT*_5m_2026.csv
from pathlib import Path

from src.config.paths import INTERIM_EVENT_5_MIN_DIR, RAW_DATA_DIR

def load_and_prepare_data(coin: str, raw_dir: str | Path | None = None) -> tuple:
    """
    Load cryptocurrency data and prepare features for quantile regression.

    Returns:
        X, y, regressors_and_targets, D
    """
    import numpy as np
    import pandas as pd

    raw_path = Path(raw_dir) if raw_dir is not None else RAW_DATA_DIR
    btc_file = next((p for p in raw_path.glob(f"{coin}USDT*_2026.csv")), None)
    if btc_file is None:
        raise FileNotFoundError(f"{coin}USDT file not found in {raw_path}")

    df = pd.read_csv(
        btc_file,
        usecols=["open_time", "open", "high", "low", "close", "volume"],
    )

    df["coin"] = f"{coin}USDT"
    df["current_time"] = pd.to_datetime(df["open_time"], errors="coerce")
    df["current_price"] = df["open"].astype("float64")
    df["neg_log_return"] = -np.log(df["current_price"]).diff()  # Positive for losses (downside)

    df["minutes"] = df["current_time"].dt.hour * 60 + df["current_time"].dt.minute
    df["d"] = (df["minutes"] // 5 + 1).astype("int16")
    D = int(df["d"].max())

    regressors_and_targets = df[["current_time", "neg_log_return", "d"]].copy()
    regressors_and_targets.rename(
        columns={"current_time": "time", "neg_log_return": "X_t"},
        inplace=True,
    )

    regressors_and_targets["X_t-1"] = regressors_and_targets["X_t"].shift(1).abs()
    regressors_and_targets["X_t-2"] = regressors_and_targets["X_t"].shift(2).abs()

    theta = 2 * np.pi * regressors_and_targets["d"].to_numpy() / D
    for j in (1, 2, 3):
        regressors_and_targets[f"cos_{j}"] = np.cos(j * theta)
        regressors_and_targets[f"sin_{j}"] = np.sin(j * theta)

    regressors_and_targets.dropna(inplace=True)

    feature_cols = (
        ["X_t-1", "X_t-2"]
        + [f"cos_{j}" for j in (1, 2, 3)]
        + [f"sin_{j}" for j in (1, 2, 3)]
    )

    X = regressors_and_targets[feature_cols]
    y_down = regressors_and_targets["X_t"]

    return X, y_down, regressors_and_targets, D


def fit_quantile_regression(
    X_train,
    y_train,
    X_test,
    y_test,
    tau: float = 0.95
):
    """
    Fit quantile regression on training data and predict on test data.

    Parameters
    ----------
    X_train : DataFrame or array-like
        Training features
    y_train : Series or array-like
        Training target (log returns)
    X_test : DataFrame or array-like
        Test features
    y_test : Series or array-like
        Test target (log returns)
    tau : float
        Quantile level (default=0.95)

    Returns
    -------
    predictions_df : DataFrame
        DataFrame with columns [y_true, y_pred, Event] and same index as y_test
    """
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import QuantileRegressor

    if not (0 < tau < 1):
        raise ValueError("tau must be in (0,1).")

    model = QuantileRegressor(
        quantile=tau,
        alpha=0.0,
        fit_intercept=True,
        solver="highs",
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    predictions_df = pd.DataFrame(
        {"y_true": y_test, "y_pred": y_pred},
        index=y_test.index
    )
    predictions_df["Event"] = np.maximum(0.0, predictions_df["y_true"] - predictions_df["y_pred"])

    return predictions_df


def _load_interim_data(coin: str, interim_dir: str | Path | None = None) -> tuple:
    """
    Load existing interim data for a coin if available.

    Parameters
    ----------
    coin : str
        Coin identifier (e.g., "BTC")
    interim_dir : str
        Path to interim directory

    Returns
    -------
    tuple
        (interim_df, last_time) if file exists, (None, None) otherwise.
        last_time is parsed from the 'time' column when available.
    """
    import pandas as pd

    interim_dir_path = Path(interim_dir) if interim_dir is not None else INTERIM_EVENT_5_MIN_DIR
    interim_path = interim_dir_path / f"{coin}_5m_events.parquet"

    if not interim_path.exists():
        return None, None

    try:
        interim_df = pd.read_parquet(interim_path)
        if len(interim_df) == 0:
            return interim_df, None

        if "time" in interim_df.columns:
            parsed_times = pd.to_datetime(interim_df["time"], errors="coerce")
            if parsed_times.notna().any():
                return interim_df, parsed_times.max()

        # Backward-compatible fallback for older files without a valid time column.
        last_index = interim_df.index[-1]
        return interim_df, last_index
    except Exception as e:
        print(f"Warning: Could not load interim file {interim_path}: {e}")
        return None, None


def _save_interim_chunk(
    coin: str,
    chunk_df,
    interim_dir: str | Path | None = None,
    append: bool = True
) -> None:
    """
    Save or append predictions to interim parquet file.

    Parameters
    ----------
    coin : str
        Coin identifier (e.g., "BTC")
    chunk_df : DataFrame
        DataFrame with predictions to save (columns: time, X_t, y_pred, Event)
    interim_dir : str
        Path to interim directory
    append : bool
        If True, append to existing file; if False, overwrite
    """
    import pandas as pd

    interim_dir_path = Path(interim_dir) if interim_dir is not None else INTERIM_EVENT_5_MIN_DIR
    interim_dir_path.mkdir(parents=True, exist_ok=True)

    interim_path = interim_dir_path / f"{coin}_5m_events.parquet"

    if append and interim_path.exists():
        try:
            existing_df = pd.read_parquet(interim_path)
            combined_df = pd.concat([existing_df, chunk_df]).drop_duplicates(
                subset=["time"], keep="last"
            ).sort_values("time")
            combined_df.to_parquet(interim_path, index=False)
        except Exception as e:
            print(f"Warning: Could not append to interim file. Overwriting: {e}")
            chunk_df.to_parquet(interim_path, index=False)
    else:
        chunk_df.to_parquet(interim_path, index=False)


def rolling_quantile_regression_window(
    coin: str,
    tau: float = 0.95,
    N: int = 7,
    P: int = 1,
    D: int = 288,
    raw_dir: str | Path | None = None,
    interim_dir: str | Path | None = None
) -> tuple:
    """
    Rolling-window quantile regression with checkpoint/resume capability.

    Performs rolling-window quantile regression on intraday data and saves
    predictions progressively to interim file, allowing interruption and resume.

    Parameters
    ----------
    coin : str
        Cryptocurrency identifier (e.g., "BTC", "ETH")
    tau : float
        Quantile level (default=0.95)
    N : int
        Training window in days (default=7)
    P : int
        Prediction horizon in days (default=1)
    D : int
        Number of 5-minute intervals in a day (default=288). Retained for
        backward compatibility; rolling splits are done by full calendar days.
    raw_dir : str
        Path to raw data directory
    interim_dir : str
        Path to interim directory for saving results

    Returns
    -------
    tuple
        (results_df, predictions_df, mean_pinball_loss, interim_path)
    """
    import numpy as np
    import pandas as pd
    from sklearn.metrics import mean_pinball_loss

    if not (0 < tau < 1):
        raise ValueError("tau must be in (0,1).")
    if N <= 0 or P <= 0:
        raise ValueError("N and P must be positive integers.")
    if D <= 0:
        raise ValueError("D must be positive.")

    # Load and prepare data
    print(f"Loading data for {coin}...")
    X, y_down, regressors_and_targets, D_actual = load_and_prepare_data(coin, raw_dir)

    if D_actual != D:
        print(f"Warning: Expected D={D}, but got D_actual={D_actual} from data")

    day_series = pd.to_datetime(regressors_and_targets["time"], errors="coerce").dt.floor("D")
    valid_day_mask = day_series.notna()
    if not valid_day_mask.all():
        X = X.loc[valid_day_mask]
        y_down = y_down.loc[valid_day_mask]
        regressors_and_targets = regressors_and_targets.loc[valid_day_mask].copy()
        day_series = day_series.loc[valid_day_mask]

    unique_days = pd.Index(day_series.sort_values().unique())
    if len(unique_days) <= N:
        raise ValueError("Not enough full days for the selected rolling window size.")

    max_start = len(unique_days) - N
    candidate_starts = list(range(0, max_start, P))
    if not candidate_starts:
        raise ValueError("No rolling windows available with the selected N and P.")

    # Check for existing checkpoint
    existing_df, last_time = _load_interim_data(coin, interim_dir)
    start_window_idx = 0

    if existing_df is not None and len(existing_df) > 0:
        print(f"Found existing interim data with {len(existing_df)} rows")
        if last_time is not None:
            last_day = pd.Timestamp(last_time).floor("D")
            for idx, start_day_idx in enumerate(candidate_starts):
                test_days = unique_days[start_day_idx + N:start_day_idx + N + P]
                if len(test_days) == 0:
                    continue
                if pd.Timestamp(test_days[-1]) > last_day:
                    start_window_idx = idx
                    break
            else:
                start_window_idx = len(candidate_starts)

            print(f"Resuming from rolling window {start_window_idx}")

    pred_chunks = []
    # Calculate total windows
    total_chunks = len(candidate_starts)

    print(f"Starting rolling window: tau={tau}, N={N} days, P={P} days")
    print(f"Total chunks to process: {total_chunks}")

    # Roll over day windows, starting from checkpoint
    for window_idx, start_day_idx in enumerate(candidate_starts):
        # Skip windows already processed
        if window_idx < start_window_idx:
            continue

        train_days = unique_days[start_day_idx:start_day_idx + N]
        test_days = unique_days[start_day_idx + N:start_day_idx + N + P]

        if len(test_days) == 0:
            continue

        train_mask = day_series.isin(train_days)
        test_mask = day_series.isin(test_days)

        train_indices = regressors_and_targets.index[train_mask]
        test_indices = regressors_and_targets.index[test_mask]

        if len(train_indices) == 0 or len(test_indices) == 0:
            continue

        X_train = X.loc[train_indices]
        y_train_down = y_down.loc[train_indices]
        y_test_down = y_down.loc[test_indices]
        X_test = X.loc[test_indices]

        # Fit and predict for downside (extreme losses)
        predictions_down = fit_quantile_regression(
            X_train, y_train_down, X_test, y_test_down, tau=tau
        )
        pred_chunks.append(predictions_down)

        # Prepare interim data (minimal columns)
        interim_chunk = regressors_and_targets.loc[test_indices, ["time", "X_t"]].copy()
        interim_chunk["y_pred_down"] = predictions_down["y_pred"].values
        interim_chunk["mark_t"] = predictions_down["Event"].values

        # Save incrementally
        _save_interim_chunk(coin, interim_chunk, interim_dir, append=True)

        progress = f"({window_idx + 1}/{total_chunks})"
        train_start_day = pd.Timestamp(train_days[0]).strftime("%d/%m/%Y")
        train_end_day = pd.Timestamp(train_days[-1]).strftime("%d/%m/%Y")
        test_start_day = pd.Timestamp(test_days[0]).strftime("%d/%m/%Y")
        test_end_day = pd.Timestamp(test_days[-1]).strftime("%d/%m/%Y")
        print(
            f"  Processed chunk {progress} | "
            f"train {train_start_day} to {train_end_day} | "
            f"test {test_start_day} to {test_end_day}"
        )

    # Aggregate results
    print("Aggregating results...")
    predictions_df = pd.concat(pred_chunks).sort_index() if pred_chunks else pd.DataFrame()

    if len(predictions_df) > 0:
        pinball = mean_pinball_loss(
            predictions_df["y_true"],
            predictions_df["y_pred"],
            alpha=tau
        )
    else:
        pinball = np.nan

    # Build complete results_df
    results_df = regressors_and_targets.copy()
    results_df["y_pred_down"] = np.nan
    results_df["mark_t"] = pd.Series(np.nan, index=results_df.index, dtype="float64")

    if len(predictions_df) > 0:
        results_df.loc[predictions_df.index, "y_pred_down"] = predictions_df["y_pred"]
        results_df.loc[predictions_df.index, "mark_t"] = predictions_df["Event"]

    interim_dir_path = Path(interim_dir) if interim_dir is not None else INTERIM_EVENT_5_MIN_DIR
    interim_path = interim_dir_path / f"{coin}_5m_events.parquet"

    print(f"Completed! Interim results saved to: {interim_path}")

    return results_df, predictions_df, pinball, str(interim_path)


# Usage
if __name__ == "__main__":
    # Example: Process BTC with rolling window and checkpoint/resume
    coin = "BTC"
    results_df, predictions_df, pinball, interim_path = rolling_quantile_regression_window(
        coin=coin,
        tau=0.95,
        N=7,
        P=1,
        D=288
    )

    print(f"\n{'='*60}")
    print(f"Results for {coin}")
    print(f"{'='*60}")
    print(f"Total rows: {len(results_df)}, Predicted rows: {len(predictions_df)}")
    print(f"tau=0.95 | rolling pinball loss={pinball:.8f}")
    print(f"Interim file: {interim_path}")
    print(f"\nLast 10 predictions:")
    print(results_df[["time", "X_t", "y_pred_down", "mark_t"]].tail(10))