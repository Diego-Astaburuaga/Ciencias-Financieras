import argparse
import importlib
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


SECONDS_PER_DAY = 86400.0


def fit_tick_expkern_fixed_beta(
    timestamps_scaled: np.ndarray,
    beta: float,
    gofit: str,
    max_iter: int,
) -> Tuple[float, float]:
    """
    Fit a univariate exponential-kernel Hawkes model with fixed decay beta.

    tick convention:
        phi(t) = alpha * beta * exp(-beta t) 1_{t>0}
        lambda(t) = mu + sum_{ti < t} alpha*beta*exp(-beta(t-ti))
    """
    try:
        hawkes_module = importlib.import_module("tick.hawkes")
    except ImportError as exc:
        raise ImportError(
            "The 'tick' package is required in the external Hawkes environment. "
            "Run this worker with the Python executable that has tick installed."
        ) from exc

    learner = hawkes_module.HawkesExpKern(
        decays=beta,
        gofit=gofit,
        max_iter=max_iter,
        verbose=False,
    )
    learner.fit([timestamps_scaled.astype(np.float64)])

    mu = float(learner.baseline[0])
    alpha = float(learner.adjacency[0, 0])
    return mu, alpha


def intensity_grid_exp_kernel(
    grid: np.ndarray,
    events: np.ndarray,
    mu: float,
    alpha: float,
    beta: float,
) -> np.ndarray:
    """Compute the Hawkes intensity on an ascending grid using O(n+m) recursion."""
    grid = np.asarray(grid, dtype=float)
    events = np.asarray(events, dtype=float)

    if grid.ndim != 1:
        raise ValueError("grid must be 1D")
    if events.ndim != 1:
        raise ValueError("events must be 1D")
    if grid.size == 0:
        return np.array([], dtype=float)

    out = np.empty(grid.size, dtype=float)
    state = 0.0
    event_idx = 0

    first_t = grid[0]
    while event_idx < events.size and events[event_idx] < first_t:
        state += np.exp(-beta * (first_t - events[event_idx]))
        event_idx += 1
    out[0] = mu + (alpha * beta) * state

    last_t = first_t
    for idx in range(1, grid.size):
        current_t = grid[idx]
        delta_t = current_t - last_t
        if delta_t < 0:
            raise ValueError("grid must be sorted ascending")

        state *= np.exp(-beta * delta_t)
        while event_idx < events.size and events[event_idx] < current_t:
            state += np.exp(-beta * (current_t - events[event_idx]))
            event_idx += 1

        out[idx] = mu + (alpha * beta) * state
        last_t = current_t

    return out


def compute_lambda_for_day(
    event_times_raw: np.ndarray,
    beta: float,
    gofit: str,
    grid_minutes: int,
    lookback_days: int = 30,
    max_iter: int = 2000,
) -> dict:
    """Fit Hawkes on the lookback-window events and return the mean daily intensity."""
    try:
        lookback_seconds = lookback_days * 24 * 60 * 60
        historical_events = event_times_raw[event_times_raw < lookback_seconds]
        if historical_events.size < 2:
            raise ValueError(f"Insufficient events ({historical_events.size}) in the window.")

        t0 = historical_events.min()
        span = historical_events.max() - t0
        if span <= 0:
            raise ValueError("Non-positive span of event times in the lookback window.")

        scale = historical_events.size / span
        events_scaled = (historical_events - t0) * scale
        mu, alpha = fit_tick_expkern_fixed_beta(
            events_scaled,
            beta=beta,
            gofit=gofit,
            max_iter=max_iter,
        )

        step = grid_minutes * 60.0
        grid_raw = np.arange(lookback_seconds, lookback_seconds + SECONDS_PER_DAY, step)
        grid_scaled = (grid_raw - t0) * scale
        intensity = intensity_grid_exp_kernel(
            grid=grid_scaled,
            events=events_scaled,
            mu=mu,
            alpha=alpha,
            beta=beta,
        )

        return {
            "lambda_daily": float(np.mean(intensity)) if intensity.size else np.nan,
            "n_events_window": int(historical_events.size),
            "mu": mu,
            "alpha": alpha,
            "beta": beta,
            "scale": scale,
            "T": span,
            "status": "success",
        }
    except Exception as exc:
        return {
            "lambda_daily": np.nan,
            "n_events_window": np.nan,
            "mu": np.nan,
            "alpha": np.nan,
            "beta": np.nan,
            "scale": np.nan,
            "T": np.nan,
            "status": f"error: {exc}",
        }


def load_event_input(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, low_memory=False)
    else:
        df = pd.read_parquet(path)

    time_col = "time" if "time" in df.columns else "open_time"
    if time_col not in df.columns:
        raise ValueError("Input must contain a 'time' or 'open_time' column.")

    out = pd.DataFrame()
    out["time"] = pd.to_datetime(df[time_col], utc=True, errors="coerce")

    if "is_extreme_event" in df.columns:
        out["is_extreme_event"] = df["is_extreme_event"].astype(bool)
    elif "mark_t" in df.columns:
        out["is_extreme_event"] = pd.to_numeric(df["mark_t"], errors="coerce").fillna(0.0).gt(0.0)
    elif "Event_down" in df.columns:
        out["is_extreme_event"] = pd.to_numeric(df["Event_down"], errors="coerce").fillna(0.0).gt(0.0)
    elif "Event" in df.columns:
        out["is_extreme_event"] = pd.to_numeric(df["Event"], errors="coerce").fillna(0.0).gt(0.0)
    else:
        raise ValueError("Input must contain 'is_extreme_event', 'mark_t', 'Event_down', or 'Event'.")

    out = out.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    if out.empty:
        raise ValueError("Input contains no valid timestamps after parsing.")
    return out


def process_all_days(
    df: pd.DataFrame,
    *,
    output_path: Path,
    beta: float,
    gofit: str,
    grid_minutes: int,
    lookback_days: int,
    max_iter: int,
    overwrite: bool,
) -> pd.DataFrame:
    required_columns = [
        "day",
        "lambda_t",
        "lambda_hawkes",
        "lambda_daily",
        "n_events_window",
        "mu",
        "alpha",
        "beta",
        "scale",
        "T",
        "status",
    ]
    min_day = df["time"].min().floor("D")
    max_day = df["time"].max().floor("D")
    candidate_days = pd.date_range(
        start=min_day + pd.Timedelta(days=lookback_days),
        end=max_day,
        freq="D",
    )

    if overwrite or not output_path.exists():
        existing = pd.DataFrame(columns=required_columns)
    else:
        if output_path.suffix.lower() == ".parquet":
            existing = pd.read_parquet(output_path)
        else:
            existing = pd.read_csv(output_path)
    if "lambda_hawkes" not in existing.columns and "lambda_daily" in existing.columns:
        existing["lambda_hawkes"] = existing["lambda_daily"]
    if "lambda_t" not in existing.columns and "lambda_hawkes" in existing.columns:
        existing["lambda_t"] = existing["lambda_hawkes"]
    if "lambda_t" not in existing.columns and "lambda_daily" in existing.columns:
        existing["lambda_t"] = existing["lambda_daily"]
    if "lambda_daily" not in existing.columns and "lambda_hawkes" in existing.columns:
        existing["lambda_daily"] = existing["lambda_hawkes"]
    for col in required_columns:
        if col not in existing.columns:
            existing[col] = np.nan
    existing = existing[required_columns]

    processed_days = set(
        pd.to_datetime(existing.get("day", pd.Series(dtype="object")), utc=True, errors="coerce")
        .dt.strftime("%Y-%m-%d")
        .dropna()
    )

    extreme_times = df.loc[df["is_extreme_event"], "time"]
    rows: list[dict[str, object]] = []
    for day in candidate_days:
        day_str = day.strftime("%Y-%m-%d")
        if day_str in processed_days:
            continue

        reference_time = day - pd.Timedelta(days=lookback_days)
        window_mask = (extreme_times >= reference_time) & (extreme_times < day)
        event_times_raw = (extreme_times.loc[window_mask] - reference_time).dt.total_seconds().to_numpy(dtype=float)
        event_times_raw = np.sort(event_times_raw[np.isfinite(event_times_raw)])

        result = compute_lambda_for_day(
            event_times_raw=event_times_raw,
            beta=beta,
            gofit=gofit,
            grid_minutes=grid_minutes,
            lookback_days=lookback_days,
            max_iter=max_iter,
        )
        result["lambda_t"] = result["lambda_daily"]
        result["lambda_hawkes"] = result["lambda_daily"]
        result["day"] = day_str
        rows.append(result)

    if rows:
        new_rows = pd.DataFrame(rows)[required_columns]
        existing = pd.concat([existing, new_rows], ignore_index=True)

    if not existing.empty:
        existing = existing.drop_duplicates(subset=["day"], keep="last").sort_values("day").reset_index(drop=True)

    if output_path.suffix.lower() == ".parquet":
        existing.to_parquet(output_path, index=False)
    else:
        existing.to_csv(output_path, index=False)

    return existing[required_columns] if not existing.empty else existing


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit Hawkes on downside-event times and compute daily mean intensity.")
    parser.add_argument("--input", required=True, help="Input file path (csv or parquet).")
    parser.add_argument("--output", required=True, help="Output path for daily lambda (.parquet recommended).")
    parser.add_argument("--beta", type=float, default=1.5, help="Decay parameter in scaled time.")
    parser.add_argument(
        "--gofit",
        type=str,
        default="likelihood",
        choices=["likelihood", "least-squares"],
        help="tick objective.",
    )
    parser.add_argument("--grid-minutes", type=int, default=5, help="Grid frequency in minutes for daily averaging.")
    parser.add_argument("--lookback-days", type=int, default=30, help="History window used to fit Hawkes before each day.")
    parser.add_argument("--max-iter", type=int, default=2000, help="tick max_iter.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite any existing output instead of resuming.")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    event_df = load_event_input(input_path)
    results = process_all_days(
        event_df,
        output_path=output_path,
        beta=float(args.beta),
        gofit=args.gofit,
        grid_minutes=int(args.grid_minutes),
        lookback_days=int(args.lookback_days),
        max_iter=int(args.max_iter),
        overwrite=bool(args.overwrite),
    )
    print(f"Saved {len(results)} Hawkes diagnostics rows to {output_path}.")


if __name__ == "__main__":
    main()
