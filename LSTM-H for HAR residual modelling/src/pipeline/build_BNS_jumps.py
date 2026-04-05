"""
Barndorff-Nielsen and Shephard jump detection on daily 5-minute return panels.

References
----------
BarndorffNielsen2006
    Barndorff-Nielsen, O. E., and Shephard, N. (2006). Econometrics of testing
    for jumps in financial economics using bipower variation. Journal of
    Financial Econometrics, 4(1), 1-30.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from src.config.paths import INTERIM_DAILY_RV_JUMPS_DIR, RAW_DATA_DIR, ROOT_DIR


DEFAULT_EXPECTED_OBS_PER_DAY = 288
DEFAULT_MIN_COVERAGE_RATIO = 0.90
DEFAULT_CONFIG: dict[str, object] = {
    "jump_test_alpha": 0.01,
}


@dataclass
class BNSJumpResult:
    rv: torch.Tensor
    bpv: torch.Tensor
    tpv: torch.Tensor
    jv: torch.Tensor
    z: torch.Tensor
    jump_indicator: torch.Tensor
    critical_value: float
    alpha: float


@dataclass
class JumpPipelineArtifacts:
    jump_data: pd.DataFrame
    quality_summary: pd.DataFrame
    output_dir: Path
    output_file: Path
    summary_file: Path


class ProjectJumpDataMapper:
    """
    Project-aware mapper from coin symbols to raw input files and output folders.

    This class isolates repository-specific path conventions from the generic
    jump-detection pipeline, which accepts timestamp and price series directly.
    """

    def __init__(
        self,
        coin: str,
        raw_data_dir: Path = RAW_DATA_DIR,
        output_root_dir: Path = INTERIM_DAILY_RV_JUMPS_DIR,
        time_column: str = "open_time",
        price_column: str = "open",
    ) -> None:
        self.coin = coin.upper()
        self.raw_data_dir = Path(raw_data_dir)
        self.output_root_dir = Path(output_root_dir)
        self.time_column = time_column
        self.price_column = price_column

    @property
    def output_dir(self) -> Path:
        return self.output_root_dir / self.coin

    @property
    def output_file(self) -> Path:
        return self.output_dir / "RV_data.parquet"

    @property
    def summary_file(self) -> Path:
        return self.output_dir / "data_quality_summary.csv"

    def resolve_input_file(self) -> Path:
        candidate_files = sorted(
            self.raw_data_dir.glob(f"{self.coin}USDT_binance_*.csv")
        )
        if not candidate_files:
            raise FileNotFoundError(
                f"No raw CSV found for coin {self.coin} in {self.raw_data_dir}"
            )
        return candidate_files[-1]

    def load_raw_frame(self) -> pd.DataFrame:
        input_file = self.resolve_input_file()
        frame = pd.read_csv(
            input_file,
            usecols=[self.time_column, self.price_column],
        )
        if frame.empty:
            raise ValueError(f"Input file is empty: {input_file}")
        return frame

    def load_timestamp_price_frame(self) -> pd.DataFrame:
        frame = self.load_raw_frame().copy()
        frame = frame.rename(
            columns={
                self.time_column: "timestamp",
                self.price_column: "price",
            }
        )
        frame["timestamp"] = parse_timestamp_series(frame["timestamp"])
        frame["price"] = pd.to_numeric(frame["price"], errors="coerce")
        return frame

    def load_timestamp_and_price(self) -> tuple[pd.Series, pd.Series]:
        frame = self.load_timestamp_price_frame()
        return frame["timestamp"], frame["price"]

    def ensure_output_dir(self) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self.output_dir

    def build(
        self,
        alpha: float | None = None,
        m_expected: int = DEFAULT_EXPECTED_OBS_PER_DAY,
        coverage_threshold: float = DEFAULT_MIN_COVERAGE_RATIO,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
    ) -> JumpPipelineArtifacts:
        timestamps, prices = self.load_timestamp_and_price()
        return build_daily_rv_jumps_from_series(
            timestamps=timestamps,
            prices=prices,
            output_dir=self.ensure_output_dir(),
            asset_name=self.coin,
            alpha=alpha,
            m_expected=m_expected,
            coverage_threshold=coverage_threshold,
            device=device,
            dtype=dtype,
        )


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


def load_config(config_path: Path | None = None) -> dict[str, object]:
    cfg = DEFAULT_CONFIG.copy()
    path = ROOT_DIR / "config.yaml" if config_path is None else Path(config_path)
    if path.exists() and path.stat().st_size > 0:
        cfg.update(_parse_simple_yaml(path))

    cfg["jump_test_alpha"] = float(cfg["jump_test_alpha"])
    if not (0.0 < cfg["jump_test_alpha"] < 1.0):
        raise ValueError("jump_test_alpha must be in (0, 1).")

    return cfg


def resolve_jump_test_alpha(alpha: float | None = None) -> float:
    if alpha is not None:
        if not (0.0 < float(alpha) < 1.0):
            raise ValueError("alpha must be in (0, 1).")
        return float(alpha)
    return float(load_config()["jump_test_alpha"])


def _get_device(device: Optional[str] = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _normal_abs_moment(p: float) -> float:
    """
    E|Z|^p for Z ~ N(0,1):
        E|Z|^p = 2^(p/2) * Gamma((p+1)/2) / sqrt(pi)
    """
    return (2.0 ** (p / 2.0)) * math.gamma((p + 1.0) / 2.0) / math.sqrt(math.pi)


def _infer_epoch_unit(values: pd.Series) -> str:
    """
    Infer whether numeric timestamps are expressed in seconds, milliseconds,
    microseconds, or nanoseconds based on their magnitude.
    """
    sample = values.dropna()
    if sample.empty:
        return "ms"

    scale = float(sample.abs().median())
    if scale < 1e11:
        return "s"
    if scale < 1e14:
        return "ms"
    if scale < 1e17:
        return "us"
    return "ns"


def parse_timestamp_series(values) -> pd.Series:
    series = pd.Series(values, copy=False)

    # If timestamps are already datetime-like, preserve them and standardize to UTC.
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, utc=True, errors="coerce")

    numeric_values = pd.to_numeric(series, errors="coerce")
    if numeric_values.notna().all():
        unit = _infer_epoch_unit(numeric_values)
        return pd.to_datetime(numeric_values, unit=unit, utc=True, errors="coerce")

    return pd.to_datetime(series, utc=True, errors="coerce")


def prepare_timestamp_price_frame(
    timestamps,
    prices,
    price_name: str = "price",
) -> pd.DataFrame:
    timestamp_series = pd.Series(timestamps, copy=False)
    price_series = pd.Series(prices, copy=False)

    if len(timestamp_series) != len(price_series):
        raise ValueError("timestamps and prices must have the same length.")
    if len(timestamp_series) < 2:
        raise ValueError("At least two observations are required.")

    frame = pd.DataFrame(
        {
            "timestamp": parse_timestamp_series(timestamp_series),
            price_name: pd.to_numeric(price_series, errors="coerce"),
        }
    )
    frame = frame.dropna(subset=["timestamp", price_name]).copy()
    frame = frame.loc[frame[price_name] > 0].copy()
    frame = (
        frame.sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"])
        .reset_index(drop=True)
    )

    if frame.empty:
        raise ValueError("No valid observations remain after cleaning timestamps and prices.")
    if len(frame) < 2:
        raise ValueError("At least two valid observations are required after cleaning.")

    return frame


def build_intraday_returns_from_prices(
    timestamps,
    prices,
    price_name: str = "price",
) -> pd.DataFrame:
    frame = prepare_timestamp_price_frame(
        timestamps=timestamps,
        prices=prices,
        price_name=price_name,
    )
    frame["log_return"] = np.log(frame[price_name]).diff()
    returns = frame.loc[frame["log_return"].notna(), ["timestamp", "log_return"]].copy()

    if returns.empty:
        raise ValueError("No valid log returns can be computed from the provided prices.")

    returns["date"] = returns["timestamp"].dt.floor("D")
    return returns


def build_daily_stats(
    returns: pd.DataFrame,
    expected_obs_per_day: int = DEFAULT_EXPECTED_OBS_PER_DAY,
    coverage_threshold: float = DEFAULT_MIN_COVERAGE_RATIO,
) -> pd.DataFrame:
    if returns.empty:
        raise ValueError("returns must contain at least one observation.")
    if expected_obs_per_day <= 0:
        raise ValueError("expected_obs_per_day must be > 0.")
    if not (0.0 < coverage_threshold <= 1.0):
        raise ValueError("coverage_threshold must be in (0, 1].")

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
    daily_stats["coverage_ratio"] = daily_stats["n_obs"] / expected_obs_per_day
    daily_stats["is_complete_missing"] = daily_stats["n_obs"] == 0
    daily_stats["is_partial_missing"] = (daily_stats["n_obs"] > 0) & (
        daily_stats["n_obs"] < expected_obs_per_day
    )
    daily_stats["rv_scaled_sum_sq"] = np.where(
        daily_stats["n_obs"] > 0,
        daily_stats["sum_sq"] * expected_obs_per_day / daily_stats["n_obs"],
        np.nan,
    )
    daily_stats["RV"] = np.where(
        daily_stats["coverage_ratio"] >= coverage_threshold,
        daily_stats["rv_scaled_sum_sq"],
        np.nan,
    )
    return daily_stats


def trim_to_valid_interval(
    daily_stats: pd.DataFrame,
    asset_name: str = "asset",
    coverage_threshold: float = DEFAULT_MIN_COVERAGE_RATIO,
) -> pd.DataFrame:
    last_valid_date = daily_stats["RV"].last_valid_index()
    if last_valid_date is None:
        raise RuntimeError(
            f"No valid RV observations for {asset_name} after applying {int(coverage_threshold * 100)}% coverage rule."
        )
    return daily_stats.loc[:last_valid_date].copy()


def compute_bns_jump_test(
    returns: torch.Tensor,
    alpha: float = 0.01,
    eps: float = 1e-12,
) -> BNSJumpResult:
    """
    Compute the Barndorff-Nielsen and Shephard (BNS) jump detection statistics.
    """

    if returns.ndim != 2:
        raise ValueError("`returns` must have shape (n_days, M).")

    _, m = returns.shape
    if m < 3:
        raise ValueError("BNS jump test requires at least M >= 3 intraday returns per day.")

    dtype = returns.dtype
    device = returns.device

    mu1 = math.sqrt(2.0 / math.pi)
    mu43 = _normal_abs_moment(4.0 / 3.0)
    c_bns = (mu1 ** -4) + 2.0 * (mu1 ** -2) - 5.0

    rv = torch.sum(returns ** 2, dim=1)
    abs_r = torch.abs(returns)
    bpv = (mu1 ** -2) * torch.sum(abs_r[:, 1:] * abs_r[:, :-1], dim=1)

    abs_r_43 = abs_r ** (4.0 / 3.0)
    tpv_sum = torch.sum(
        abs_r_43[:, 2:] * abs_r_43[:, 1:-1] * abs_r_43[:, :-2],
        dim=1,
    )
    tpv = m * (mu43 ** -3) * (m / (m - 2.0)) * tpv_sum

    jv = torch.clamp(rv - bpv, min=0.0)

    rv_safe = torch.clamp(rv, min=eps)
    bpv_safe = torch.clamp(bpv, min=eps)
    ratio_term = torch.maximum(
        torch.ones_like(tpv, dtype=dtype, device=device),
        tpv / (bpv_safe ** 2),
    )
    denom = torch.sqrt(c_bns * (1.0 / m) * ratio_term + eps)
    z = ((rv - bpv) / rv_safe) / denom

    normal = torch.distributions.Normal(
        loc=torch.tensor(0.0, device=device, dtype=dtype),
        scale=torch.tensor(1.0, device=device, dtype=dtype),
    )
    critical_value = float(
        normal.icdf(torch.tensor(1.0 - alpha, device=device, dtype=dtype)).item()
    )
    jump_indicator = (z > critical_value).to(dtype=torch.int64)

    return BNSJumpResult(
        rv=rv,
        bpv=bpv,
        tpv=tpv,
        jv=jv,
        z=z,
        jump_indicator=jump_indicator,
        critical_value=critical_value,
        alpha=alpha,
    )


def prepare_daily_return_matrix(
    df: pd.DataFrame,
    date_col: str,
    return_col: str,
    m_expected: Optional[int] = DEFAULT_EXPECTED_OBS_PER_DAY,
    device: Optional[str] = None,
    dtype: torch.dtype = torch.float32,
    drop_incomplete_days: bool = True,
) -> tuple[torch.Tensor, pd.Index]:
    """
    Convert a long DataFrame of intraday returns into a daily matrix of shape (n_days, M).
    """

    work = df[[date_col, return_col]].copy()
    grouped = work.groupby(date_col)[return_col].apply(list)

    if m_expected is not None:
        counts = grouped.apply(len)
        if drop_incomplete_days:
            grouped = grouped[counts == m_expected]
        else:
            bad_days = counts[counts != m_expected]
            if len(bad_days) > 0:
                raise ValueError(
                    f"Found incomplete days. Expected {m_expected} returns per day, but these days differ:\n{bad_days}"
                )

    valid_days = grouped.index
    matrix = torch.tensor(grouped.tolist(), dtype=dtype, device=_get_device(device))

    return matrix, valid_days


def compute_bns_from_dataframe(
    df: pd.DataFrame,
    date_col: str = "date",
    return_col: str = "r_5m",
    alpha: float = 0.01,
    m_expected: int = DEFAULT_EXPECTED_OBS_PER_DAY,
    device: Optional[str] = None,
    dtype: torch.dtype = torch.float32,
    drop_incomplete_days: bool = True,
) -> pd.DataFrame:
    """
    Compute daily BNS jump statistics from a long intraday return DataFrame.
    """

    returns_tensor, days = prepare_daily_return_matrix(
        df=df,
        date_col=date_col,
        return_col=return_col,
        m_expected=m_expected,
        device=device,
        dtype=dtype,
        drop_incomplete_days=drop_incomplete_days,
    )

    result = compute_bns_jump_test(returns=returns_tensor, alpha=alpha)

    out = pd.DataFrame(
        {
            date_col: days,
            "RV": result.rv.detach().cpu().numpy(),
            "BPV": result.bpv.detach().cpu().numpy(),
            "TPV": result.tpv.detach().cpu().numpy(),
            "JV": result.jv.detach().cpu().numpy(),
            "z": result.z.detach().cpu().numpy(),
            "jump_indicator": result.jump_indicator.detach().cpu().numpy(),
        }
    )
    out["alpha"] = alpha
    out["critical_value"] = result.critical_value
    return out


def build_jump_quality_summary(
    asset_name: str,
    daily_stats: pd.DataFrame,
    jump_results: pd.DataFrame,
    alpha: float,
    expected_obs_per_day: int,
    coverage_threshold: float,
) -> pd.DataFrame:
    total_days = int(len(daily_stats))
    n_partial_missing_days = int(daily_stats["is_partial_missing"].sum())
    n_complete_missing_days = int(daily_stats["is_complete_missing"].sum())
    pct_partial_missing_days = (
        100.0 * n_partial_missing_days / total_days if total_days else float("nan")
    )
    pct_complete_missing_days = (
        100.0 * n_complete_missing_days / total_days if total_days else float("nan")
    )

    saved_dates = pd.DatetimeIndex(pd.to_datetime(jump_results["date"], errors="coerce"))
    excluded_dates = daily_stats.index.difference(saved_dates)
    excluded_dates_str = ";".join(dt.strftime("%Y-%m-%d") for dt in excluded_dates)

    return pd.DataFrame(
        {
            "asset": [asset_name],
            "expected_obs_per_day": [expected_obs_per_day],
            "coverage_threshold": [coverage_threshold],
            "jump_test_alpha": [alpha],
            "init_date": [str(saved_dates.min()) if len(saved_dates) else ""],
            "end_date": [str(saved_dates.max()) if len(saved_dates) else ""],
            "valid_start": [str(daily_stats.index.min())],
            "valid_end": [str(daily_stats.index.max())],
            "total_days_in_valid_interval": [total_days],
            "days_with_missing_data": [n_partial_missing_days],
            "pct_days_with_missing_data": [pct_partial_missing_days],
            "days_completely_missed": [n_complete_missing_days],
            "pct_days_completely_missed": [pct_complete_missing_days],
            "n_missing_dates_between": [int(len(excluded_dates))],
            "missing_dates_between": [excluded_dates_str],
            "rows_saved": [int(len(jump_results))],
            "jump_days_detected": [int(jump_results["jump_indicator"].sum())],
        }
    )


def save_jump_pipeline_outputs(
    jump_data: pd.DataFrame,
    quality_summary: pd.DataFrame,
    output_dir: Path,
) -> JumpPipelineArtifacts:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "RV_data.parquet"
    summary_file = output_dir / "data_quality_summary.csv"

    jump_data.to_parquet(output_file, index=False)
    quality_summary.to_csv(summary_file, index=False)

    return JumpPipelineArtifacts(
        jump_data=jump_data,
        quality_summary=quality_summary,
        output_dir=output_dir,
        output_file=output_file,
        summary_file=summary_file,
    )


def build_daily_rv_jumps_from_series(
    timestamps,
    prices,
    output_dir: Path,
    asset_name: str = "asset",
    alpha: float | None = None,
    m_expected: int = DEFAULT_EXPECTED_OBS_PER_DAY,
    coverage_threshold: float = DEFAULT_MIN_COVERAGE_RATIO,
    device: Optional[str] = None,
    dtype: torch.dtype = torch.float32,
) -> JumpPipelineArtifacts:
    alpha_value = resolve_jump_test_alpha(alpha)
    returns = build_intraday_returns_from_prices(timestamps=timestamps, prices=prices)
    daily_stats = build_daily_stats(
        returns=returns,
        expected_obs_per_day=m_expected,
        coverage_threshold=coverage_threshold,
    )
    daily_stats = trim_to_valid_interval(
        daily_stats=daily_stats,
        asset_name=asset_name,
        coverage_threshold=coverage_threshold,
    )

    complete_days = daily_stats.index[daily_stats["n_obs"] == m_expected]
    if len(complete_days) == 0:
        raise RuntimeError(
            f"No complete {m_expected}-observation days available for jump detection for {asset_name}."
        )

    jump_input = returns.loc[returns["date"].isin(complete_days)].copy()
    jump_df = compute_bns_from_dataframe(
        df=jump_input,
        date_col="date",
        return_col="log_return",
        alpha=alpha_value,
        m_expected=m_expected,
        device=device,
        dtype=dtype,
        drop_incomplete_days=True,
    ).sort_values("date").reset_index(drop=True)

    quality_summary = build_jump_quality_summary(
        asset_name=asset_name,
        daily_stats=daily_stats,
        jump_results=jump_df,
        alpha=alpha_value,
        expected_obs_per_day=m_expected,
        coverage_threshold=coverage_threshold,
    )
    artifacts = save_jump_pipeline_outputs(
        jump_data=jump_df,
        quality_summary=quality_summary,
        output_dir=output_dir,
    )

    print(f"Daily RV with jumps saved to: {artifacts.output_file}")
    print(f"Data quality summary saved to: {artifacts.summary_file}")
    print(
        f"Valid interval for {asset_name}: {jump_df['date'].min()} -> {jump_df['date'].max()}"
    )
    print(f"Rows written: {len(jump_df)}")

    return artifacts


def build_BNS_jumps(
    coin: str,
    alpha: float | None = None,
    m_expected: int = DEFAULT_EXPECTED_OBS_PER_DAY,
    coverage_threshold: float = DEFAULT_MIN_COVERAGE_RATIO,
    device: Optional[str] = None,
    dtype: torch.dtype = torch.float32,
) -> pd.DataFrame:
    mapper = ProjectJumpDataMapper(coin=coin)
    input_file = mapper.resolve_input_file()
    artifacts = mapper.build(
        alpha=alpha,
        m_expected=m_expected,
        coverage_threshold=coverage_threshold,
        device=device,
        dtype=dtype,
    )

    print(f"Raw file used: {input_file}")
    return artifacts.jump_data


def build_daily_rv_jumps(
    coin: str,
    alpha: float | None = None,
    m_expected: int = DEFAULT_EXPECTED_OBS_PER_DAY,
    coverage_threshold: float = DEFAULT_MIN_COVERAGE_RATIO,
    device: Optional[str] = None,
    dtype: torch.dtype = torch.float32,
) -> pd.DataFrame:
    return build_BNS_jumps(
        coin=coin,
        alpha=alpha,
        m_expected=m_expected,
        coverage_threshold=coverage_threshold,
        device=device,
        dtype=dtype,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build daily RV, BPV, TPV, JV, and BNS jump indicators from raw 5-minute Binance data."
    )
    parser.add_argument("coin", help="Coin ticker without the USDT suffix, for example BTC")
    parser.add_argument("--alpha", type=float, default=None, help="Jump-test significance level. Defaults to jump_test_alpha from config.yaml.")
    args = parser.parse_args()
    build_BNS_jumps(args.coin, alpha=args.alpha)


if __name__ == "__main__":
    main()