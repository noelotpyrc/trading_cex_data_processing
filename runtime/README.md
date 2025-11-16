# Runtime Utilities

Incremental feature generation utilities for runtime/production use.

## Purpose

These modules support **incremental** feature generation (processing only new/missing rows), as opposed to the batch `scripts/` which process entire datasets.

## Modules

### `data_loader.py`
- Load OHLCV data from DuckDB
- Validate hourly continuity
- Extract time windows

### `lookbacks_builder.py`  
- Build lookback windows for specific timestamps
- Trim to required base window
- Validate lookback integrity

### `features_builder.py`
- Compute multi-timeframe features from lookbacks
- Uses `feature_engineering` core modules
- Returns single-row feature DataFrames

## Usage Pattern

```python
from runtime.data_loader import load_ohlcv_duckdb
from runtime.lookbacks_builder import build_latest_lookbacks
from runtime.features_builder import compute_latest_features_from_lookbacks

# 1. Load OHLCV
df = load_ohlcv_duckdb("ohlcv.duckdb", table="ohlcv_1h", start="2025-01-01", end="2025-01-31")

# 2. Build lookbacks for latest timestamp
lookbacks = build_latest_lookbacks(df, window_hours=720, timeframes=["1H", "4H", "12H", "1D"])

# 3. Compute features
features_row = compute_latest_features_from_lookbacks(lookbacks)
```

## Dependencies

- **DuckDB** - For reading OHLCV data
- **feature_engineering** - Core feature computation modules (same repo)
- **scripts.build_multi_timeframe_features** - Imports `compute_features_one` function

## Use Cases

- Incremental feature backfilling (process only new timestamps)
- Real-time feature generation (compute for latest bar)
- Feature store updates (add features for new OHLCV rows)
