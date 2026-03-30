# Extreme Events and Realized Volatility Forecasting

> **Information augmentation in linear and nonlinear models using intraday crypto data.**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Data Pipeline](#data-pipeline)
- [Notebooks](#notebooks)
- [Path Management](#path-management)
- [Setup](#setup)

---

## Overview

This project studies whether **intraday extreme-return information improves daily realized volatility (RV) forecasting** for crypto assets.

The central question is whether adding Hawkes-process-based and quantile-exceedance features to a baseline model — either linear (HAR) or nonlinear (LSTM) — yields better out-of-sample forecasts.

**Key design principle:** a strict fair-comparison protocol between HAR and LSTM:

| Aspect | HAR | LSTM |
|---|---|---|
| Representation | Aggregated (daily/weekly/monthly) | Raw sequential (30-day lookback) |
| Target | $RV_{t+1}$ | $RV_{t+1}$ |
| Rolling windows | Same | Same |
| Information set | Same daily features | Same daily features |

**Workflow:**

1. Build daily RV from 5-minute Binance OHLCV data.
2. Detect intraday extreme events via rolling quantile regression.
3. Aggregate event-intensity features to the daily frequency.
4. Fit Hawkes process intensity on downside event times.
5. Run rolling-window out-of-sample forecast comparison.

---

## Repository Structure

```text
.
├── data/
│   ├── raw/                    # Source 5-minute CSV files ({COIN}USDT)
│   ├── interim/
│   │   ├── daily_rv/           # {COIN}_rv.parquet — daily RV + features
│   │   ├── event_5_min/        # {COIN}_5m_events.parquet — intraday events
│   │   └── hawkes_daily/       # {COIN}_hawkes_daily.parquet — Hawkes diagnostics
│   └── temp/                   # Temporary experiment outputs
├── documents/                  # Project notes and supplementary material
├── notebooks/
│   ├── run_rolling_window.ipynb      # Main experiment runner
│   ├── data_handling.ipynb           # Data pipeline walkthrough
│   ├── data_quality.ipynb            # Coverage and quality checks
│   ├── results_outcome.ipynb         # Analysis and plots of results
│   └── visualization_rolling_window.ipynb
├── results/                    # Per-coin experiment outputs (coin={COIN}/)
├── src/
│   ├── config/
│   │   └── paths.py            # Centralized path definitions
│   ├── data/
│   │   └── build_intraday_and_rv.py
│   ├── features/
│   │   ├── identify_events.py
│   │   ├── create_raw_event_feat.py
│   │   ├── create_hawkes_feat.py
│   │   └── tick_env_hawkes_fit.py
│   ├── models/
│   │   ├── base.py
│   │   ├── har.py
│   │   ├── LSTM.py
│   │   └── types.py
│   └── evaluation/
│       └── eval_rolling_window.py
└── README.md
```

## Data Pipeline

### Step 1 — Raw intraday data → daily RV

**Script:** [`src/data/build_intraday_and_rv.py`](src/data/build_intraday_and_rv.py)

| | |
|---|---|
| **Input** | `data/raw/{COIN}USDT_binance_*.csv` |
| **Output** | `data/interim/daily_rv/{COIN}_rv.parquet` |

Let $p_{d,j}$ be the open price at day $d$, intraday slot $j$, with $M=288$ slots (5-minute bars).

$$
r_{d,j} = \log p_{d,j} - \log p_{d,j-1}, \qquad
RV_d = \sum_{j=1}^{M} r_{d,j}^2.
$$

---

### Step 2 — Intraday extreme events

**Script:** [`src/features/identify_events.py`](src/features/identify_events.py)

| | |
|---|---|
| **Input** | `data/raw/{COIN}USDT_binance_*.csv` |
| **Output** | `data/interim/event_5_min/{COIN}_5m_events.parquet` |

Define the loss-oriented return $x_t = -(\log p_t - \log p_{t-1})$. For quantile level $\tau$, fit a rolling conditional quantile $\hat{q}_t^{(\tau)}$ from lag/Fourier regressors. The downside exceedance is:

$$
E_t^{\downarrow} = \max\{0,\, x_t - \hat{q}_t^{(\tau)}\}.
$$

---

### Step 3 — Daily event features

**Script:** [`src/features/create_raw_event_feat.py`](src/features/create_raw_event_feat.py)

| | |
|---|---|
| **Input** | `event_5_min/{COIN}_5m_events.parquet`, `daily_rv/{COIN}_rv.parquet` |
| **Output** | Updated `daily_rv/{COIN}_rv.parquet` with two new columns |

| Column | Definition |
|---|---|
| `ratio_event` | $\frac{1}{M}\sum_{j}\mathbf{1}\{E_{d,j}^{\downarrow}>0\}$ — daily fraction of event intervals |
| `exceedance_down` | $\log\!\left(\sum_{j}(E_{d,j}^{\downarrow})^2\right)$ — log squared exceedance intensity |

---

### Step 4 — Hawkes process intensity

**Scripts:** [`src/features/create_hawkes_feat.py`](src/features/create_hawkes_feat.py), [`src/features/tick_env_hawkes_fit.py`](src/features/tick_env_hawkes_fit.py)

| | |
|---|---|
| **Input** | `event_5_min/{COIN}_5m_events.parquet`, `daily_rv/{COIN}_rv.parquet` |
| **Intermediate** | `data/interim/hawkes_daily/{COIN}_hawkes_daily.parquet` |
| **Output** | Updated `daily_rv/{COIN}_rv.parquet` with column `lambda_t` |

For each day $d$, a univariate Hawkes process with exponential kernel is fitted on the 30-day event lookback window:

$$
\lambda_d(t) = \mu_d + \alpha_d \beta \sum_{t_i < t} e^{-\beta (t - t_i)}, \qquad t \in [d,\, d+1).
$$

The daily feature is the mean intensity over the 5-minute grid:

$$
\lambda^{\mathrm{hawkes}}_d = \frac{1}{M}\sum_{j=1}^{M} \lambda_d(\tau_{d,j}).
$$

> **Conda environment for `tick`:** by default uses `conda run -n hawkes_env python`. Override with the `HAWKES_CONDA_ENV` environment variable, or pass `python_executable` directly to bypass conda.

---

### Step 5 — Rolling-window forecast evaluation

**Script:** [`src/evaluation/eval_rolling_window.py`](src/evaluation/eval_rolling_window.py)

| Setting | Value |
|---|---|
| Train window | 1 200 observations |
| Test window | 300 observations |
| Step | 300 observations |
| LSTM internal validation | last 300 of train window |
| LSTM lookback | 30 days |

Both HAR and LSTM share the same rolling date windows and the same target $y_t = RV_{t+1}$. They differ only in how they consume the common daily information set:

**HAR regressors** (aggregated):
$$
X_t^{HAR} = \big(RV_{d,t},\ RV_{w,t},\ RV_{m,t},\ z_{d,t},\ z_{w,t},\ z_{m,t}\big).
$$

**LSTM input** (raw sequence, lookback $L=30$):
$$
\mathbf{X}_t^{LSTM} = \left[(RV_{t-i},\, z_{1,t-i},\ldots,z_{m,t-i})\right]_{i=1}^{L}.
$$

Results are saved under `results/coin={COIN}/`. Completed windows are checkpointed and skipped on reruns.

---

## Notebooks

| Notebook | Purpose |
|---|---|
| [`run_rolling_window.ipynb`](notebooks/run_rolling_window.ipynb) | Runs all feature-combination experiments in order |
| [`data_handling.ipynb`](notebooks/data_handling.ipynb) | Step-by-step data pipeline walkthrough |
| [`data_quality.ipynb`](notebooks/data_quality.ipynb) | Daily coverage checks and quality summaries |
| [`results_outcome.ipynb`](notebooks/results_outcome.ipynb) | Loss delta plots, LaTeX tables, correlation heatmaps |
| [`visualization_rolling_window.ipynb`](notebooks/visualization_rolling_window.ipynb) | Rolling-window result visualizations |

---

## Path Management

All paths are centralized in [`src/config/paths.py`](src/config/paths.py). Scripts resolve locations via constants (`ROOT_DIR`, `RAW_DATA_DIR`, `INTERIM_DATA_DIR`, `RESULTS_DIR`, …) rather than fragile relative paths.

---

## Setup

```bash
# Clone and install in editable mode
pip install -e .
```

**Dependencies** (see `pyproject.toml`): `numpy`, `pandas`, `scikit-learn`, `scipy`, `pyarrow`, `torch`.