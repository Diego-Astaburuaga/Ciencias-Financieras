from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Iterable, Literal

import pandas as pd

from src.config.paths import (
    INTERIM_DAILY_RV_DIR,
    INTERIM_EVENT_5_MIN_DIR,
    ROOT_DIR,
    TEMP_HAWKES_DIR,
)


EVENT_DIR = INTERIM_EVENT_5_MIN_DIR
INTERIM_RV_DIR = INTERIM_DAILY_RV_DIR
HAWKES_TEMP_DIR = TEMP_HAWKES_DIR
HAWKES_VALUE_COLUMNS = ("lambda_t", "lambda_hawkes", "lambda_daily")


def _event_parquet_path(coin: str, event_dir: Path) -> Path:
    return Path(event_dir) / f"{coin}_5m_events.parquet"


def _load_event_frame(coin: str, event_dir: Path) -> tuple[pd.DataFrame, Path]:
    event_path = _event_parquet_path(coin, event_dir)
    if not event_path.exists():
        raise FileNotFoundError(f"Event file not found for {coin}: {event_path}")

    df = pd.read_parquet(event_path)
    if "time" not in df.columns:
        raise ValueError(f"{event_path} must contain a 'time' column.")
    return df, event_path


def _normalize_daily_timestamps(values):
    normalized = pd.to_datetime(values, errors="coerce", utc=True)
    if isinstance(normalized, pd.Series):
        return normalized.dt.floor("D").dt.tz_localize(None)
    return normalized.floor("D").tz_localize(None)


def _resolve_hawkes_python(python_executable: str | Path | None) -> Path:
    if python_executable is not None:
        resolved = Path(python_executable).expanduser().resolve()
    elif os.environ.get("HAWKES_PYTHON"):
        resolved = Path(os.environ["HAWKES_PYTHON"]).expanduser().resolve()
    else:
        resolved = Path(sys.executable).resolve()

    if not resolved.exists():
        raise FileNotFoundError(f"Hawkes Python executable not found: {resolved}")
    return resolved


def _resolve_hawkes_command_prefix(
    python_executable: str | Path | None,
    hawkes_conda_env: str | None,
) -> list[str]:
    if python_executable is not None and hawkes_conda_env is not None:
        raise ValueError("Provide only one of python_executable or hawkes_conda_env, not both.")

    if python_executable is not None:
        python_path = _resolve_hawkes_python(python_executable)
        return [str(python_path)]

    selected_env = hawkes_conda_env or os.environ.get("HAWKES_CONDA_ENV") or "hawkes_env"
    conda_exe = os.environ.get("HAWKES_CONDA_EXE") or shutil.which("conda")
    if not conda_exe:
        raise FileNotFoundError(
            "Could not find 'conda' executable. Set HAWKES_CONDA_EXE or pass python_executable explicitly."
        )
    return [str(conda_exe), "run", "-n", selected_env, "python"]


def _load_event_input(coin: str, event_dir: Path) -> pd.DataFrame:
    df, event_path = _load_event_frame(coin, event_dir)

    if "mark_t" in df.columns:
        event_values = pd.to_numeric(df["mark_t"], errors="coerce").fillna(0.0)
    elif "Event_down" in df.columns:
        event_values = pd.to_numeric(df["Event_down"], errors="coerce").fillna(0.0)
    elif "Event" in df.columns:
        event_values = pd.to_numeric(df["Event"], errors="coerce").fillna(0.0)
    else:
        raise ValueError(f"{event_path} must contain 'mark_t' or legacy 'Event_down'/'Event'.")

    out = pd.DataFrame(
        {
            "time": pd.to_datetime(df["time"], utc=True, errors="coerce"),
            "is_extreme_event": event_values.gt(0.0),
        }
    )
    out = out.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    if out.empty:
        raise ValueError(f"{event_path} contains no valid timestamps.")
    return out


def _stage_hawkes_input(coin: str, events_df: pd.DataFrame, temp_dir: Path) -> Path:
    temp_dir.mkdir(parents=True, exist_ok=True)
    staged_path = Path(temp_dir) / f"{coin}_hawkes_events.csv"
    events_df.to_csv(staged_path, index=False)
    return staged_path


def _read_hawkes_diagnostics(diagnostics_path: Path) -> pd.DataFrame:
    if diagnostics_path.suffix.lower() == ".parquet":
        diagnostics_df = pd.read_parquet(diagnostics_path)
    else:
        diagnostics_df = pd.read_csv(diagnostics_path)

    if diagnostics_df.empty:
        return diagnostics_df

    diagnostics_df = diagnostics_df.copy()
    if "lambda_t" not in diagnostics_df.columns and "lambda_hawkes" in diagnostics_df.columns:
        diagnostics_df["lambda_t"] = diagnostics_df["lambda_hawkes"]
    if "lambda_t" not in diagnostics_df.columns and "lambda_daily" in diagnostics_df.columns:
        diagnostics_df["lambda_t"] = diagnostics_df["lambda_daily"]
    if "lambda_hawkes" not in diagnostics_df.columns and "lambda_daily" in diagnostics_df.columns:
        diagnostics_df["lambda_hawkes"] = diagnostics_df["lambda_daily"]
    if "lambda_daily" not in diagnostics_df.columns and "lambda_hawkes" in diagnostics_df.columns:
        diagnostics_df["lambda_daily"] = diagnostics_df["lambda_hawkes"]
    return diagnostics_df


def _build_daily_hawkes_series(diagnostics_path: Path) -> pd.Series:
    hawkes_df = _read_hawkes_diagnostics(diagnostics_path)
    hawkes_value_col = next((col for col in HAWKES_VALUE_COLUMNS if col in hawkes_df.columns), None)
    if hawkes_df.empty or "day" not in hawkes_df.columns or hawkes_value_col is None:
        raise ValueError(
            f"{diagnostics_path} must contain 'day' and one of {HAWKES_VALUE_COLUMNS} columns."
        )

    hawkes_df = hawkes_df.copy()
    hawkes_df["date"] = _normalize_daily_timestamps(hawkes_df["day"])
    hawkes_df[hawkes_value_col] = pd.to_numeric(hawkes_df[hawkes_value_col], errors="coerce")
    return (
        hawkes_df.dropna(subset=["date"])
        .drop_duplicates(subset=["date"], keep="last")
        .set_index("date")[hawkes_value_col]
    )


def _merge_hawkes_into_event_5_min(
    coin: str,
    event_dir: Path,
    hawkes_series: pd.Series,
) -> pd.DataFrame:
    events_df, event_path = _load_event_frame(coin, event_dir)

    event_times = pd.to_datetime(events_df["time"], utc=True, errors="coerce")
    event_dates = _normalize_daily_timestamps(event_times)
    events_df = events_df.copy()
    events_df["lambda_t"] = event_dates.map(hawkes_series).astype("float64")

    events_df.to_parquet(event_path, index=False)
    return events_df


def _merge_hawkes_into_daily_rv(
    coin: str,
    hawkes_series: pd.Series,
    rv_dir: Path,
) -> pd.DataFrame:
    rv_path = Path(rv_dir) / f"{coin}_rv.parquet"
    if not rv_path.exists():
        raise FileNotFoundError(f"Daily RV file not found for {coin}: {rv_path}")

    rv_df = pd.read_parquet(rv_path)
    if rv_df.empty:
        rv_df["lambda_t"] = pd.Series(dtype="float64")
        rv_df.to_parquet(rv_path)
        return rv_df

    if isinstance(rv_df.index, pd.DatetimeIndex):
        rv_dates = pd.Series(_normalize_daily_timestamps(rv_df.index), index=rv_df.index)
    elif "date" in rv_df.columns:
        rv_dates = pd.Series(_normalize_daily_timestamps(rv_df["date"]), index=rv_df.index)
    else:
        rv_dates = pd.Series(_normalize_daily_timestamps(rv_df.index), index=rv_df.index)

    rv_df["lambda_t"] = rv_dates.map(hawkes_series).astype("float64")
    rv_df.to_parquet(rv_path)
    return rv_df


def build_daily_hawkes_feature(
    coin: str,
    *,
    beta: float = 1.5,
    gofit: Literal["likelihood", "least-squares"] = "least-squares",
    grid_minutes: int = 5,
    lookback_days: int = 30,
    max_iter: int = 2000,
    python_executable: str | Path | None = None,
    hawkes_conda_env: str | None = None,
    event_dir: Path = EVENT_DIR,
    rv_dir: Path = INTERIM_RV_DIR,
    diagnostics_dir: Path | None = None,
    temp_dir: Path = HAWKES_TEMP_DIR,
    overwrite: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Build daily Hawkes intensity from event_5_min data and align it back into
    both event_5_min and daily RV parquet files.

    The worker script is executed in a potentially different Python environment so
    the current project does not need the tick dependency installed locally.
    Execution selection priority:
    1) explicit python_executable
    2) explicit hawkes_conda_env
    3) HAWKES_CONDA_ENV
    4) default conda env: hawkes_env
    """
    if lookback_days <= 0:
        raise ValueError("lookback_days must be > 0")
    if grid_minutes <= 0:
        raise ValueError("grid_minutes must be > 0")

    coin = coin.upper()
    events_df = _load_event_input(coin, Path(event_dir))
    staged_input = _stage_hawkes_input(coin, events_df, Path(temp_dir))

    diagnostics_path = Path(temp_dir) / f"{coin}_hawkes_diagnostics.parquet"
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)

    command_prefix = _resolve_hawkes_command_prefix(python_executable, hawkes_conda_env)
    script_path = Path(__file__).resolve().with_name("tick_env_hawkes_fit.py")
    if not script_path.exists():
        raise FileNotFoundError(f"Hawkes worker script not found: {script_path}")

    cmd = [
        *command_prefix,
        str(script_path),
        "--input", str(staged_input),
        "--output", str(diagnostics_path),
        "--beta", str(float(beta)),
        "--gofit", str(gofit),
        "--grid-minutes", str(int(grid_minutes)),
        "--lookback-days", str(int(lookback_days)),
        "--max-iter", str(int(max_iter)),
    ]
    if overwrite:
        cmd.append("--overwrite")

    res = subprocess.run(cmd, capture_output=True, text=True, check=False, cwd=str(ROOT_DIR))
    if res.returncode != 0:
        raise RuntimeError(
            "Hawkes worker failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"stdout:\n{res.stdout}\n"
            f"stderr:\n{res.stderr}"
        )
        
    if verbose and res.stdout.strip():
        print(res.stdout.strip())

    hawkes_series = _build_daily_hawkes_series(diagnostics_path)
    _merge_hawkes_into_event_5_min(coin, Path(event_dir), hawkes_series)
    updated_rv = _merge_hawkes_into_daily_rv(coin, hawkes_series, Path(rv_dir))
    if verbose:
        print(f"{coin}: aligned 'lambda_t' into {_event_parquet_path(coin, Path(event_dir))}")
        print(f"{coin}: added 'lambda_t' to {(Path(rv_dir) / f'{coin}_rv.parquet')}")
        if diagnostics_dir is not None:
            print("diagnostics_dir is deprecated and ignored; Hawkes alignment now uses event_5_min directly.")
    return updated_rv


def build_all_daily_hawkes_features(
    *,
    coins: Iterable[str] | None = None,
    beta: float = 1.5,
    gofit: Literal["likelihood", "least-squares"] = "least-squares",
    grid_minutes: int = 5,
    lookback_days: int = 30,
    max_iter: int = 2000,
    python_executable: str | Path | None = None,
    hawkes_conda_env: str | None = None,
    event_dir: Path = EVENT_DIR,
    rv_dir: Path = INTERIM_RV_DIR,
    diagnostics_dir: Path | None = None,
    temp_dir: Path = HAWKES_TEMP_DIR,
    overwrite: bool = False,
    verbose: bool = True,
) -> dict[str, Path]:
    event_dir = Path(event_dir)
    rv_dir = Path(rv_dir)

    if coins is None:
        event_files = sorted(event_dir.glob("*_5m_events.parquet"))
        if not event_files:
            raise FileNotFoundError(f"No event parquet files found in {event_dir}")
        selected_coins = [p.stem.replace("_5m_events", "") for p in event_files]
    else:
        selected_coins = [coin.upper() for coin in coins]

    updated_files: dict[str, Path] = {}
    for coin in selected_coins:
        build_daily_hawkes_feature(
            coin=coin,
            beta=beta,
            gofit=gofit,
            grid_minutes=grid_minutes,
            lookback_days=lookback_days,
            max_iter=max_iter,
            python_executable=python_executable,
            hawkes_conda_env=hawkes_conda_env,
            event_dir=event_dir,
            rv_dir=rv_dir,
            diagnostics_dir=diagnostics_dir,
            temp_dir=temp_dir,
            overwrite=overwrite,
            verbose=verbose,
        )
        updated_files[coin] = rv_dir / f"{coin}_rv.parquet"

    return updated_files


if __name__ == "__main__":
    updated = build_all_daily_hawkes_features()
    print(f"Updated {len(updated)} daily RV files with lambda_t.")
