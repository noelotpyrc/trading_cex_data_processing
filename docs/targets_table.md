# Targets Table

## Overview

The `targets` table stores computed target labels (y_true) for training and calibrating trading strategies.

## Schema

```sql
CREATE TABLE targets (
    timestamp TIMESTAMP NOT NULL,
    dataset TEXT NOT NULL,
    target_key TEXT NOT NULL,
    target_value DOUBLE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (timestamp, dataset, target_key)
);
```

**Columns:**
- `timestamp`: Bar timestamp (e.g., '2025-11-16 10:00:00')
- `dataset`: Dataset identifier (e.g., 'binance_btcusdt_perp_1h')
- `target_key`: Target name (e.g., 'y_tp_before_sl_u0.04_d0.02_24h')
- `target_value`: Computed target (0.0 or 1.0 for binary, float for regression)
- `created_at`: When this target was computed

## Target Key Naming Convention

Format: `y_{type}_{params}_{horizon}`

Examples:
- `y_tp_before_sl_u0.04_d0.02_24h` - Binary: TP (+4%) before SL (-2%) within 24 hours
- `y_logret_168h` - Regression: Log return over 168 hours (7 days)
- `y_tb_label_u0.03_d0.015_12h` - Ternary: TP/SL/Natural with 12h horizon

## Usage

### Backfill Targets

```bash
python scripts/backfill_targets.py \
  --duckdb "/Volumes/Extreme SSD/trading_data/cex/db/binance_btcusdt_perp_ohlcv.duckdb" \
  --table ohlcv_btcusdt_1h \
  --dataset binance_btcusdt_perp_1h \
  --target-key y_tp_before_sl_u0.04_d0.02_24h \
  --horizon-bars 24 \
  --up-pct 0.04 \
  --down-pct 0.02 \
  --freq 1H \
  --mode window \
  --start "2020-01-01" \
  --end "2025-11-30"
```

### Query Targets

```sql
-- Get targets for a time range
SELECT timestamp, target_value
FROM targets
WHERE dataset = 'binance_btcusdt_perp_1h'
  AND target_key = 'y_tp_before_sl_u0.04_d0.02_24h'
  AND timestamp BETWEEN '2025-11-01' AND '2025-11-30'
ORDER BY timestamp;

-- Join with predictions for calibration
SELECT p.ts, p.y_pred, t.target_value AS y_true
FROM predictions p
JOIN targets t ON p.ts = t.timestamp
WHERE p.model_path = '...'
  AND t.dataset = 'binance_btcusdt_perp_1h'
  AND t.target_key = 'y_tp_before_sl_u0.04_d0.02_24h';

-- Check coverage
SELECT
    COUNT(*) AS total_rows,
    COUNT(DISTINCT timestamp) AS unique_timestamps,
    MIN(timestamp) AS first_ts,
    MAX(timestamp) AS last_ts
FROM targets
WHERE dataset = 'binance_btcusdt_perp_1h'
  AND target_key = 'y_tp_before_sl_u0.04_d0.02_24h';
```

## Modes

### 1. Window Mode (Initial Backfill)

Backfill for an explicit date range:

```bash
python scripts/backfill_targets.py \
  --duckdb "..." \
  --dataset binance_btcusdt_perp_1h \
  --target-key y_tp_before_sl_u0.04_d0.02_24h \
  --horizon-bars 24 \
  --up-pct 0.04 \
  --down-pct 0.02 \
  --mode window \
  --start "2025-01-01" \
  --end "2025-11-30"
```

### 2. Incremental Mode (Catch Up)

Continue from last computed target:

```bash
python scripts/backfill_targets.py \
  --duckdb "..." \
  --dataset binance_btcusdt_perp_1h \
  --target-key y_tp_before_sl_u0.04_d0.02_24h \
  --horizon-bars 24 \
  --up-pct 0.04 \
  --down-pct 0.02 \
  --mode last_from_targets
```

### 3. Explicit Timestamps

Backfill specific timestamps:

```bash
python scripts/backfill_targets.py \
  --duckdb "..." \
  --dataset binance_btcusdt_perp_1h \
  --target-key y_tp_before_sl_u0.04_d0.02_24h \
  --horizon-bars 24 \
  --up-pct 0.04 \
  --down-pct 0.02 \
  --mode ts_list \
  --ts "2025-11-01 00:00:00" "2025-11-01 01:00:00"
```

## Dry Run

Preview without writing to database:

```bash
python scripts/backfill_targets.py \
  --duckdb "..." \
  --dataset binance_btcusdt_perp_1h \
  --target-key y_tp_before_sl_u0.04_d0.02_24h \
  --horizon-bars 24 \
  --up-pct 0.04 \
  --down-pct 0.02 \
  --mode window \
  --start "2025-11-01" \
  --end "2025-11-02" \
  --dry-run
```

## Performance

- **Computation**: ~1-10 ms per timestamp (depends on horizon)
- **Throughput**: ~100-1000 rows/second
- **Storage**: ~16 bytes per target (very lightweight)

## Integration with Strategy Repo

The `trading_cex_strategy` repo expects targets to already exist in this table:

```python
# In trading_cex_strategy/strategies/ml_calibrated_bins/rebuild_bins.py
actuals_cal = load_actuals(ohlcv_duckdb, ohlcv_table, target_key, cal_start, cal_end)
```

The `load_actuals` function queries:
```sql
SELECT timestamp, target_value AS y_true
FROM targets
WHERE dataset = ? AND target_key = ?
  AND timestamp BETWEEN ? AND ?
```

## Maintenance

### Automated Daily Backfill

Set up a cron job to backfill new targets daily:

```bash
# Daily at 2 AM: backfill yesterday's targets
0 2 * * * cd /path/to/trading_cex_data_processing && \
  ./venv/bin/python scripts/backfill_targets.py \
  --duckdb "..." \
  --dataset binance_btcusdt_perp_1h \
  --target-key y_tp_before_sl_u0.04_d0.02_24h \
  --horizon-bars 24 \
  --up-pct 0.04 \
  --down-pct 0.02 \
  --mode last_from_targets
```

### Recompute Targets (Force Overwrite)

```bash
python scripts/backfill_targets.py \
  --duckdb "..." \
  --dataset binance_btcusdt_perp_1h \
  --target-key y_tp_before_sl_u0.04_d0.02_24h \
  --horizon-bars 24 \
  --up-pct 0.04 \
  --down-pct 0.02 \
  --mode window \
  --start "2025-11-01" \
  --end "2025-11-30" \
  --force  # Overwrite existing
```

## Troubleshooting

**ImportError: No module named 'feature_engineering'**

The script uses relative imports from the parent directory. Make sure you're running from the repo root or the import paths are correct.

**ValueError: No OHLCV data loaded**

Check that your OHLCV table has data for the requested time range, including the horizon buffer.

**Many targets fail to generate (None values)**

This is normal near the end of the dataset where there isn't enough forward data. For a 24h horizon, the last 24 bars will have NaN targets.
