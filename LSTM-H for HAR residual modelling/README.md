# Enhancing HAR-RV Forecasting via Residual Modeling with LSTM and Event-Driven Features

This repository studies daily realized-volatility forecasting for crypto assets with a HAR baseline and LSTM-based residual modeling enriched with event-driven covariates.

## Current Structure

```text
.
|-- data/
|   |-- raw/                  # Raw Binance 5-minute data: {COIN}USDT_binance_*.csv
|   |-- interim/
|   |   `-- daily_rv/         # Daily realized volatility series: {COIN}.parquet
|   `-- temp/
|-- documents/
|-- notebooks/
|-- results/
|-- src/
|   |-- config/
|   |   `-- paths.py
|   |-- data/
|   |   `-- build_daily_rv.py
|   |-- evaluation/
|   |-- features/
|   `-- models/
`-- README.md
```

## Daily RV Builder

The first data product in this project is the daily realized-volatility series built from raw 5-minute Binance data.

Input:
- `data/raw/{COIN}USDT_binance_*.csv`

Output:
- `data/interim/daily_rv/{COIN}.parquet`

Method:
1. Read `open_time` and `open`.
2. Build the intraday series `(timestamp, log_return)` using the open price.
3. For each day, compute:

$$
r_t = \log(P_t) - \log(P_{t-1})
$$

$$
RV_d = \sum_{t \in d} r_t^2
$$

The output parquet stores the daily series with a `DatetimeIndex` named `date` and a single column `RV`.

Run:

```bash
python -m src.data.build_daily_rv BTC
```

## Setup

```bash
pip install -e .
```
