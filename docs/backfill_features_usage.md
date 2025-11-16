# Backfill Features Usage Guide

## Overview

The `backfill_features.py` script computes multi-timeframe features from OHLCV data in DuckDB and stores them in a features table.

## Quick Start

```bash
# Activate virtual environment
cd /Users/noel/projects/trading_cex_data_processing
source venv/bin/activate

# Backfill features with feature list filtering (recommended for production)
python scripts/backfill_features.py \
  --duckdb "/Volumes/Extreme SSD/trading_data/cex/db/binance_btcusdt_perp_ohlcv.duckdb" \
  --feat-duckdb "/Volumes/Extreme SSD/trading_data/cex/db/binance_btcusdt_perp_feature.duckdb" \
  --table ohlcv_btcusdt_1h \
  --feature-key "manual_backfill" \
  --mode last_from_features \
  --base-hours 720 \
  --feature-list "/Users/noel/projects/trading_cex/configs/feature_lists/binance_btcusdt_p60_default.json"
```

## Modes

### 1. Window Mode (Explicit Date Range)

Backfill features for a specific date range:

```bash
python scripts/backfill_features.py \
  --duckdb "<ohlcv.duckdb>" \
  --feat-duckdb "<features.duckdb>" \
  --table ohlcv_btcusdt_1h \
  --feature-key "manual_backfill" \
  --mode window \
  --start "2025-11-01 00:00:00" \
  --end "2025-11-30 23:00:00" \
  --base-hours 720 \
  --feature-list "/path/to/feature_list.json"
```

### 2. Incremental Mode (Continue from Last Feature)

Automatically continue from the last feature timestamp:

```bash
python scripts/backfill_features.py \
  --duckdb "<ohlcv.duckdb>" \
  --feat-duckdb "<features.duckdb>" \
  --table ohlcv_btcusdt_1h \
  --feature-key "manual_backfill" \
  --mode last_from_features \
  --base-hours 720 \
  --feature-list "/path/to/feature_list.json"
```

This mode:
- Finds the last timestamp in features table for the given feature_key
- Starts backfilling from next hour
- Processes up to latest closed bar (current hour - 1)

### 3. Explicit Timestamp List

Backfill specific timestamps:

```bash
python scripts/backfill_features.py \
  --duckdb "<ohlcv.duckdb>" \
  --feat-duckdb "<features.duckdb>" \
  --table ohlcv_btcusdt_1h \
  --feature-key "manual_backfill" \
  --mode ts_list \
  --ts "2025-11-13 12:00:00" "2025-11-13 15:00:00" \
  --base-hours 720 \
  --feature-list "/path/to/feature_list.json"
```

Or from a file:

```bash
cat > timestamps.txt << EOF
2025-11-13 12:00:00
2025-11-13 15:00:00
EOF

python scripts/backfill_features.py \
  --mode ts_list \
  --ts-file timestamps.txt \
  ... other args ...
```

## Common Options

### `--feature-list` (Recommended for Production)

Path to JSON file containing list of features to save.

**Usage:**
```bash
--feature-list "/Users/noel/projects/trading_cex/configs/feature_lists/binance_btcusdt_p60_default.json"
```

**Behavior:**
- **With `--feature-list`**: Saves only the 326 features in the JSON file
  - Matches production `manual_backfill` data
  - Reduces storage by 26% vs all features
  - Validates all required features are computed
- **Without `--feature-list`**: Saves all 440 computed features
  - Useful for exploratory analysis
  - Use when you don't have a feature list yet

**Feature list format:**
```json
[
  "close_open_diff_current_1H",
  "high_low_range_current_1H",
  "close_ema_12_12H",
  ...
]
```

**Example:**
```bash
# Production (326 features)
python scripts/backfill_features.py \
  --feature-list "/path/to/binance_btcusdt_p60_default.json" \
  ... other args ...
# Output: OK 2025-11-13 00:00:00: features upserted (326 features)

# Exploratory (all 440 features)
python scripts/backfill_features.py \
  ... other args ...
# Output: OK 2025-11-13 00:00:00: features upserted (440 features)
```

### `--feature-key`

Identifier for this feature snapshot. Use different keys for:
- Different feature engineering versions
- Different data sources
- Production vs testing

Examples: `prod_backfill`, `manual_backfill`, `test_v2`

### `--base-hours`

Base training window in hours (default: 720 = 30 days)

This determines how much historical OHLCV data is needed for feature computation. Common values:
- 168 (7 days)
- 720 (30 days)
- 1440 (60 days)

### `--buffer-hours`

Extra hours on top of base window (default: 0)

Used when you need extra historical data for certain features but want to trim the final lookback to base_hours.

### `--at-most`

Limit number of bars to process (useful for testing):

```bash
# Test with just 10 bars
python scripts/backfill_features.py \
  --mode window \
  --start "2025-11-01 00:00:00" \
  --end "2025-11-30 23:00:00" \
  --at-most 10 \
  ...
```

### `--overwrite`

Replace existing features (default: skip existing):

```bash
# Re-compute existing features
python scripts/backfill_features.py \
  --overwrite \
  ...
```

### `--dry-run`

Plan only, don't compute or write features:

```bash
python scripts/backfill_features.py \
  --dry-run \
  --mode window \
  --start "2025-11-01 00:00:00" \
  --end "2025-11-30 23:00:00" \
  ...
```

### `--timeframes`

Timeframes for multi-timeframe features (default: 1H 4H 12H 1D):

```bash
python scripts/backfill_features.py \
  --timeframes 1H 4H 8H 1D \
  ...
```

## Output

The script generates:
- **326 features per timestamp** (with `--feature-list`) or **440 features** (without)
- Features stored as JSON in DuckDB `features` table
- Schema: `(feature_key, ts, features, created_at)`

Example features (from binance_btcusdt_p60_default.json):
- `close_open_diff_current_1H`
- `high_low_range_current_1H`
- `close_zscore_20_4H`
- `close_ema_12_12H`
- `close_adl_1H`
- ...and 321 more

## Testing

Run the E2E test to verify everything works:

```bash
python scripts/test_backfill_e2e.py
```

This test:
1. Checks OHLCV data availability
2. Cleans up existing test features
3. Runs backfill for Nov 13-14, 2025
4. Verifies features were written correctly
5. Compares with reference features (if available)
6. Offers to clean up test data

## Production Usage Example

Equivalent to the legacy `backfill_inference_missing` command:

```bash
# Legacy command (for reference):
# python -m run.backfill_inference_missing \
#   --duckdb "/Volumes/Extreme SSD/trading_data/cex/db/binance_btcusdt_perp_ohlcv.duckdb" \
#   --feat-duckdb "/Volumes/Extreme SSD/trading_data/cex/db/binance_btcusdt_perp_feature.duckdb" \
#   --table ohlcv_btcusdt_1h \
#   --feature-key "manual_backfill" \
#   --mode last_from_predictions \
#   --write-features \
#   --base-hours 720

# New command (features only, no inference):
python scripts/backfill_features.py \
  --duckdb "/Volumes/Extreme SSD/trading_data/cex/db/binance_btcusdt_perp_ohlcv.duckdb" \
  --feat-duckdb "/Volumes/Extreme SSD/trading_data/cex/db/binance_btcusdt_perp_feature.duckdb" \
  --table ohlcv_btcusdt_1h \
  --feature-key "manual_backfill" \
  --mode last_from_features \
  --base-hours 720 \
  --feature-list "/Users/noel/projects/trading_cex/configs/feature_lists/binance_btcusdt_p60_default.json"
```

**Key differences:**
- ✅ Uses `--feature-list` to save only 326 features (matching production `manual_backfill`)
- ✅ Uses `last_from_features` mode (incremental from last feature timestamp)
- ✅ No prediction/inference logic (features only)
- ✅ Simpler command (no --dataset, --pred-duckdb needed)
- ✅ No lightgbm dependency required

## Notes

- **Closed bars only**: Script only processes bars where `timestamp <= current_hour - 1h`
- **Hourly continuity**: Requires uninterrupted hourly OHLCV data for the lookback window
- **NaN handling**: Some features may have NaN values (e.g., early in time series) - this is normal
- **Feature count**:
  - **With `--feature-list`**: 326 features (from config file)
  - **Without `--feature-list`**: 440 features (all computed features)
  - Production backfills should use `--feature-list` to match existing `manual_backfill` data
- **Same DB support**: `--feat-duckdb` defaults to `--duckdb` if not specified (features and OHLCV in same database)

## Troubleshooting

### "No target timestamps found"
- Check that OHLCV data exists for your date range
- Verify you're querying closed bars (not current/future hours)
- Check `--start` and `--end` are correct

### "SKIP: insufficient/irregular history window"
- Need at least `base_hours` of continuous OHLCV data before each target timestamp
- Check for gaps in OHLCV data
- Reduce `--base-hours` if you have limited history

### "SKIP: missing X required features"
- The computed features don't include all features in your feature list
- This usually means:
  - Feature engineering code has changed and removed features in the list
  - Feature list is from a different feature engineering version
- Solution: Update the feature list or use current feature engineering code

### "Feature count mismatch"
- **If you see 440 vs 326**: You forgot to add `--feature-list`
  - Solution: Use `--overwrite --feature-list` to replace with correct features
- **If comparing old vs new backfills**: Feature engineering may have evolved
  - Check that all reference features are present (should be 100% overlap)
  - Update reference if new features are intentional
