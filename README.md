# Trading CEX Data Processing

Time-series feature engineering for OHLCV data. Provides leakage-safe multi-timeframe features, lag features, and target generation.

## Structure

```
trading_cex_data_processing/
├── feature_engineering/      # Core library modules
│   ├── multi_timeframe_features.py  # 52+ feature families
│   ├── current_bar_features.py      # Current bar + lag features
│   ├── targets.py                   # Target generation (returns, MFE/MAE, barriers)
│   ├── utils.py                     # Lookback utilities, resampling
│   └── hmm_features.py              # HMM-based features
├── scripts/                  # Build scripts (run these)
│   ├── build_lookbacks.py
│   ├── build_current_bar_lag_features.py
│   ├── build_multi_timeframe_features.py
│   ├── build_targets.py
│   ├── build_hmm_v1_features_csv.py
│   └── build_hmm_v2_features_csv.py
└── tests/                    # Unit tests
```

## Sample Data

A sample dataset is included for testing:
- **File:** `data/BINANCE_BTCUSDT.P, 60.csv`
- **Period:** Feb 27, 2025 - Aug 12, 2025 (~5.5 months)
- **Rows:** 4000 (hourly OHLCV data)
- **Size:** 1.1MB

## Setup

```bash
cd /Users/noel/projects/trading_cex_data_processing
pip install -r requirements.txt
```

## Usage

### 1. Build lookbacks (once per dataset)

```bash
# Using the sample data
python scripts/build_lookbacks.py \
  --input "data/BINANCE_BTCUSDT.P, 60.csv" \
  --output "data/lookbacks" \
  --timeframes 1H 4H 12H 1D \
  --lookback 168
```

### 2. Build current-bar + lag features

```bash
# Using the sample data
python scripts/build_current_bar_lag_features.py \
  --dataset "BINANCE_BTCUSDT.P, 60" \
  --lookbacks-dir "data/lookbacks" \
  --ohlcv-1h "data/BINANCE_BTCUSDT.P, 60.csv" \
  --timeframes 1H 4H 12H 1D \
  --max-lag 3 \
  --output "data/features_with_lags.csv"
```

### 3. Build targets

```bash
# Using the sample data
python scripts/build_targets.py \
  --input "data/BINANCE_BTCUSDT.P, 60.csv" \
  --base-dir "data/targets" \
  --dataset "BINANCE_BTCUSDT.P, 60" \
  --output targets.csv \
  --freq 1H \
  --horizons 3 6 12 24
```

### 4. Use in other projects

Add to PYTHONPATH:
```bash
export PYTHONPATH="${PYTHONPATH}:/Users/noel/projects/trading_cex_data_processing"
```

Then import:
```python
from feature_engineering.multi_timeframe_features import compute_features_one
from feature_engineering.utils import generate_timeframe_lookbacks
from feature_engineering.targets import generate_targets_for_row
```

## Key Features

- **Leakage-safe**: Right-closed resampling, per-row lookbacks
- **Multi-timeframe**: Build features across 1H, 4H, 12H, 1D
- **52+ feature families**: TA, volatility, liquidity, statistical, entropy
- **Lag features**: Current-bar t and t-1, t-2, ..., t-k
- **Target engineering**: Log returns, MFE/MAE, triple-barrier labels

## Testing

```bash
python -m pytest tests/
```

## Data Conventions

- OHLCV columns: lowercase (`open`, `high`, `low`, `close`, `volume`)
- Timestamps: UTC-naive, typically hourly
- Timeframe suffixes: `_1H`, `_4H`, `_12H`, `_1D`
- Lag naming: `_lag_k` (e.g., `close_logret_lag_2_4H`)
