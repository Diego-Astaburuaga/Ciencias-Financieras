import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Literal, Tuple

MCSStat = Literal["Tmax", "TR"]
ElimRule = Literal["worst_mean", "worst_tstat"]

# -----------------------------
# Alignment / panel builder
# -----------------------------
def _detect_time_key(df: pd.DataFrame) -> Optional[str]:
    candidates = ["window_id", "wid", "window", "w", "t", "val_end", "val_start", "train_end", "train_start"]
    for c in candidates:
        if c in df.columns:
            return c
    return None

def build_loss_panel(
    results_dir: Path,
    models: Dict[str, str],
    loss_col: str,
) -> Tuple[pd.DataFrame, str]:
    series_list = []
    used_keys = []

    for model_name, folder in models.items():
        df = pd.read_csv(results_dir / folder / "windows.csv")

        if loss_col not in df.columns:
            raise ValueError(f"[{model_name}] missing '{loss_col}'. Found: {list(df.columns)}")

        time_key = _detect_time_key(df)
        used_keys.append(time_key)

        if time_key is None:
            s = pd.Series(df[loss_col].to_numpy(dtype=float), name=model_name)
            s.index = pd.RangeIndex(len(s), name="row_index")
        else:
            tmp = df[[time_key, loss_col]].dropna()
            # numeric or date key
            if np.issubdtype(tmp[time_key].dtype, np.number):
                tmp[time_key] = pd.to_numeric(tmp[time_key], errors="coerce")
            else:
                tmp[time_key] = pd.to_datetime(tmp[time_key], errors="ignore")
            tmp = tmp.dropna(subset=[time_key]).drop_duplicates(subset=[time_key]).sort_values(time_key)
            s = pd.Series(tmp[loss_col].to_numpy(dtype=float), index=tmp[time_key], name=model_name)

        series_list.append(s)

    panel = pd.concat(series_list, axis=1).dropna(axis=0, how="any").sort_index()

    if any(k is None for k in used_keys):
        time_key_used = "row_index (fallback; add window_id to windows.csv for strict MCS validity)"
    else:
        time_key_used = max(set(used_keys), key=used_keys.count)

    if len(panel) < 30:
        raise ValueError(f"Too few aligned windows for MCS: T={len(panel)}")

    return panel, time_key_used


# -----------------------------
# Bootstrap utilities
# -----------------------------
def circular_block_bootstrap_indices(T: int, block_len: int, rng: np.random.Generator) -> np.ndarray:
    starts = rng.integers(0, T, size=int(np.ceil(T / block_len)))
    idx = []
    for s in starts:
        idx.extend([(s + k) % T for k in range(block_len)])
    return np.array(idx[:T], dtype=int)


# -----------------------------
# Result container
# -----------------------------
@dataclass
class MCSResult:
    loss_name: str
    alpha: float
    stat: str
    elim_rule: str
    included: List[str]
    eliminated_order: List[str]
    pvalues_elimination: pd.Series
    stats: pd.DataFrame


# =========================================================
#  ModelConfidenceSet class
# =========================================================
class ModelConfidenceSet:
    """
    General MCS runner.

    Input:
      - Either provide a loss panel L (DataFrame [T x M]) directly,
        or use from_results_dir(...) helper.

    Options:
      - stat: "Tmax" or "TR"
      - elim_rule:
          * "worst_mean": eliminate model with largest mean loss differential to average
          * "worst_tstat": eliminate model with largest |t_i| (standardized differential)
      - block bootstrap (circular) for dependence
    """

    def __init__(
        self,
        alpha: float = 0.10,
        B: int = 2000,
        block_len: Optional[int] = None,
        seed: int = 123,
        stat: MCSStat = "Tmax",
        elim_rule: ElimRule = "worst_mean",
    ):
        if not (0 < alpha < 1):
            raise ValueError("alpha must be in (0,1)")
        if B < 200:
            raise ValueError("B too small; use >= 1000 for stability.")
        self.alpha = float(alpha)
        self.B = int(B)
        self.block_len = block_len
        self.seed = int(seed)
        self.stat: MCSStat = stat
        self.elim_rule: ElimRule = elim_rule

    @staticmethod
    def from_results_dir(
        results_dir: Path,
        models: Dict[str, str],
        loss_col: str,
        **kwargs,
    ) -> Tuple["ModelConfidenceSet", pd.DataFrame, str]:
        panel, time_key_used = build_loss_panel(results_dir, models, loss_col)
        mcs = ModelConfidenceSet(**kwargs)
        return mcs, panel, time_key_used

    def run(self, L: pd.DataFrame, loss_name: str = "loss") -> MCSResult:
        rng = np.random.default_rng(self.seed)

        models_all = list(L.columns)
        X = L.to_numpy(dtype=float)  # [T, M]
        T, M = X.shape

        block_len = self.block_len
        if block_len is None:
            block_len = int(np.ceil(np.sqrt(T)))
        block_len = max(2, min(int(block_len), T))

        remaining = list(range(M))
        eliminated: List[str] = []
        pvals: List[float] = []
        step_rows: List[dict] = []

        def d_to_average(X_sub: np.ndarray) -> np.ndarray:
            # d_{i,t} = l_{i,t} - avg_j l_{j,t} over current set
            mean_t = X_sub.mean(axis=1, keepdims=True)
            return X_sub - mean_t

        def test_stat(dbar: np.ndarray, s: np.ndarray) -> float:
            t = dbar / s
            if self.stat == "Tmax":
                return float(np.max(t))
            elif self.stat == "TR":
                return float(np.max(t) - np.min(t))
            else:
                raise ValueError(f"Unknown stat: {self.stat}")

        def worst_index(dbar: np.ndarray, s: np.ndarray) -> int:
            if self.elim_rule == "worst_mean":
                return int(np.argmax(dbar))  # largest mean differential = worst
            elif self.elim_rule == "worst_tstat":
                t = dbar / s
                return int(np.argmax(np.abs(t)))  # largest standardized deviation
            else:
                raise ValueError(f"Unknown elim_rule: {self.elim_rule}")

        while len(remaining) > 1:
            Xr = X[:, remaining]           # [T, m]
            dr = d_to_average(Xr)          # [T, m]
            dbar = dr.mean(axis=0)         # [m]

            # Bootstrap dbar
            boot_dbar = np.empty((self.B, len(remaining)), dtype=float)
            for b in range(self.B):
                idx = circular_block_bootstrap_indices(T, block_len, rng)
                boot_dbar[b] = dr[idx].mean(axis=0)

            s = boot_dbar.std(axis=0, ddof=1)
            s = np.where(s < 1e-12, 1e-12, s)

            stat_obs = test_stat(dbar, s)

            # Bootstrap distribution of the same statistic
            boot_stats = np.empty(self.B, dtype=float)
            for b in range(self.B):
                boot_stats[b] = test_stat(boot_dbar[b], s)

            p_value = float(np.mean(boot_stats >= stat_obs))

            # choose model to drop if rejecting
            wloc = worst_index(dbar, s)
            wglob = remaining[wloc]
            wname = models_all[wglob]

            step_rows.append({
                "step": len(eliminated) + 1,
                "m_remaining": len(remaining),
                "stat": self.stat,
                "elim_rule": self.elim_rule,
                "test_value": stat_obs,
                "p_value": p_value,
                "worst_model": wname,
                "worst_dbar": float(dbar[wloc]),
                "worst_t": float((dbar[wloc] / s[wloc])),
                "block_len": block_len,
                "B": self.B,
            })

            # stop if fail to reject EPA
            if p_value > self.alpha:
                break

            eliminated.append(wname)
            pvals.append(p_value)
            remaining.pop(wloc)

        included = [models_all[i] for i in remaining]
        stats_df = pd.DataFrame(step_rows)
        pvalues_series = pd.Series(pvals, index=eliminated, name="p_value_at_elimination")

        return MCSResult(
            loss_name=loss_name,
            alpha=self.alpha,
            stat=self.stat,
            elim_rule=self.elim_rule,
            included=included,
            eliminated_order=eliminated,
            pvalues_elimination=pvalues_series,
            stats=stats_df,
        )


# =========================================================
#  Batch runner: losses x stats x elim_rules -> final table
# =========================================================
def run_mcs_grid(
    results_dir: Path,
    models: Dict[str, str],
    loss_cols: List[str],
    stats: List[MCSStat],
    elim_rules: List[ElimRule],
    alpha: float = 0.10,
    B: int = 2000,
    block_len: Optional[int] = None,
    seed: int = 123,
) -> pd.DataFrame:
    rows = []
    for loss_col in loss_cols:
        panel, _ = build_loss_panel(results_dir, models, loss_col)

        for st in stats:
            for er in elim_rules:
                mcs = ModelConfidenceSet(alpha=alpha, B=B, block_len=block_len, seed=seed, stat=st, elim_rule=er)
                res = mcs.run(panel, loss_name=loss_col)

                # pack a compact summary row
                mean_losses = panel.mean(axis=0)
                best_model = mean_losses.idxmin()
                rows.append({
                    "loss": loss_col,
                    "stat": st,
                    "elim_rule": er,
                    "alpha": alpha,
                    "T": len(panel),
                    "M": panel.shape[1],
                    "block_len": int(np.ceil(np.sqrt(len(panel)))) if block_len is None else int(block_len),
                    "B": B,
                    "best_by_mean": best_model,
                    "mcs_size": len(res.included),
                    "mcs_set": ", ".join(res.included),
                    "elimination_order": ", ".join(res.eliminated_order),
                    "last_p_value": (res.stats["p_value"].iloc[-1] if len(res.stats) else np.nan),
                })

    return pd.DataFrame(rows)


# =========================================================
# Example usage
# =========================================================
# losses_to_test = ["val_mse", "val_mae"]   # add "val_qlike" if you have it
# stats_to_test = ["Tmax", "TR"]
# elim_rules_to_test = ["worst_mean", "worst_tstat"]
#
# grid_table = run_mcs_grid(
#     RESULTS_DIR, MODELS,
#     loss_cols=losses_to_test,
#     stats=stats_to_test,
#     elim_rules=elim_rules_to_test,
#     alpha=0.10,
#     B=2000,
#     block_len=None,
#     seed=123,
# )
# display(grid_table.sort_values(["loss", "stat", "elim_rule"]))