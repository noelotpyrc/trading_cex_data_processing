#!/usr/bin/env python3
"""
Backfill target labels into DuckDB targets table for selected timestamps.

This script computes forward-window barrier targets (TP/SL) for each target
timestamp from OHLCV in DuckDB and persists them into a separate `targets` table.

Target table schema:
    CREATE TABLE targets (
        timestamp TIMESTAMP NOT NULL,
        dataset TEXT NOT NULL,
        target_key TEXT NOT NULL,
        target_value DOUBLE NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (timestamp, dataset, target_key)
    )

Selection modes for target timestamps:
  - window: explicit --start/--end
  - last_from_targets: continue from last targets.timestamp + 1h for the given dataset/target_key
  - ts_list: explicit timestamps via --ts

Example usage:
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
    --start "2025-01-01" \
    --end "2025-11-30"
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import duckdb
import numpy as np
import pandas as pd

from feature_engineering.targets import (
    TargetGenerationConfig,
    generate_targets_for_row,
    extract_forward_window,
)
from runtime.data_loader import load_ohlcv_duckdb


def _now_floor_utc() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(timezone.utc)).floor("h").tz_convert(None)


def _is_targets_table_present(con) -> bool:
    """Check if targets table exists."""
    try:
        res = con.execute(
            "SELECT 1 FROM information_schema.tables WHERE table_name = 'targets' LIMIT 1"
        ).fetchone()
        return bool(res)
    except Exception:
        return False


def _ensure_targets_table(con):
    """Create targets table if it doesn't exist."""
    if not _is_targets_table_present(con):
        print("Creating targets table...")
        con.execute("""
            CREATE TABLE targets (
                timestamp TIMESTAMP NOT NULL,
                dataset TEXT NOT NULL,
                target_key TEXT NOT NULL,
                target_value DOUBLE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (timestamp, dataset, target_key)
            )
        """)
        print("✓ Created targets table")


def _target_exists(con, dataset: str, target_key: str, ts: pd.Timestamp) -> bool:
    """Check if a specific target already exists."""
    try:
        row = con.execute(
            "SELECT 1 FROM targets WHERE dataset = ? AND target_key = ? AND timestamp = ? LIMIT 1",
            [str(dataset), str(target_key), pd.Timestamp(ts).to_pydatetime()],
        ).fetchone()
        return row is not None
    except Exception:
        return False


def _last_target_ts(con, dataset: str, target_key: str) -> Optional[pd.Timestamp]:
    """Get the last timestamp for a given dataset/target_key."""
    try:
        row = con.execute(
            "SELECT MAX(timestamp) FROM targets WHERE dataset = ? AND target_key = ?",
            [dataset, target_key]
        ).fetchone()
        if row and row[0] is not None:
            return pd.Timestamp(row[0])
    except Exception:
        pass
    return None


def _last_ohlcv_ts(con, table: str) -> Optional[pd.Timestamp]:
    """Get the last timestamp from OHLCV table."""
    try:
        row = con.execute(f"SELECT MAX(timestamp) FROM {table}").fetchone()
        if row and row[0] is not None:
            return pd.Timestamp(row[0])
    except Exception:
        pass
    return None


def _list_closed_bars_in_window(
    con: duckdb.DuckDBPyConnection, table: str, start: pd.Timestamp, end: pd.Timestamp
) -> List[pd.Timestamp]:
    """List all bars in the time window."""
    start = pd.to_datetime(start).tz_localize(None)
    end = pd.to_datetime(end).tz_localize(None)
    q = f"""
        SELECT timestamp FROM {table}
        WHERE timestamp BETWEEN ? AND ?
        ORDER BY timestamp
    """
    df = con.execute(q, [start.to_pydatetime(), end.to_pydatetime()]).fetch_df()
    return [pd.Timestamp(t) for t in df["timestamp"]] if not df.empty else []


def _parse_ts_list(ts_list: List[str]) -> List[pd.Timestamp]:
    """Parse a list of timestamp strings."""
    out: List[pd.Timestamp] = []
    for s in ts_list:
        try:
            out.append(pd.Timestamp(s).tz_localize(None))
        except Exception as e:
            print(f"Warning: Failed to parse timestamp '{s}': {e}", file=sys.stderr)
    return out


def _horizon_labels_for_freq(freq: str, horizons_bars: List[int]) -> dict[int, str]:
    """Generate horizon labels for a given frequency."""
    f = str(freq).upper()
    labels = {}
    if f.endswith("H"):
        for h in horizons_bars:
            labels[h] = f"{h}h"
    elif f in ("T", "MIN", "MINUTE") or f.endswith("T"):
        for h in horizons_bars:
            labels[h] = f"{h}min"
    elif f.endswith("D"):
        for h in horizons_bars:
            labels[h] = f"{h}d"
    else:
        for h in horizons_bars:
            labels[h] = f"{h}b"
    return labels


def generate_target_for_timestamp(
    df_ohlcv: pd.DataFrame,
    ts: pd.Timestamp,
    horizon_bars: int,
    up_pct: float,
    down_pct: float,
    target_key: str,
    tie_policy: str = 'conservative',
) -> Optional[float]:
    """
    Generate a single target value for a given timestamp.

    Args:
        df_ohlcv: OHLCV dataframe sorted by timestamp
        ts: Target timestamp
        horizon_bars: Horizon in bars
        up_pct: Upside barrier percentage (TP)
        down_pct: Downside barrier percentage (SL)
        target_key: Target key name (for looking up in results)
        tie_policy: Tie-breaking policy

    Returns:
        Target value (0.0 or 1.0) or None if cannot be computed
    """
    # Find index of timestamp
    try:
        idx = df_ohlcv.index.get_loc(ts)
    except KeyError:
        return None

    # Extract entry price
    entry_price = float(df_ohlcv['close'].iloc[idx])
    if not np.isfinite(entry_price):
        return None

    # Extract forward window
    ohlcv_core = df_ohlcv[['open', 'high', 'low', 'close', 'volume']].copy()
    fwd = extract_forward_window(ohlcv_core, idx, horizon_bars)

    if fwd.empty:
        return None

    # Generate target using existing feature_engineering logic
    h_labels = {horizon_bars: target_key.split('_')[-1]}  # Extract horizon label from target_key
    cfg = TargetGenerationConfig(
        horizons_bars=[horizon_bars],
        barrier_pairs=[(up_pct, down_pct)],
        tie_policy=tie_policy,
        horizon_labels=h_labels,
        include_returns=False,
        include_mfe_mae=False,
        include_barriers=True,
        log_returns=True,
    )

    res = generate_targets_for_row(fwd, entry_price, cfg)
    return res.get(target_key, None)


def backfill_targets(
    ohlcv_duckdb: str,
    table: str,
    dataset: str,
    target_key: str,
    horizon_bars: int,
    up_pct: float,
    down_pct: float,
    freq: str,
    timestamps: List[pd.Timestamp],
    *,
    tie_policy: str = 'conservative',
    buffer_hours: int = 0,
    skip_existing: bool = True,
    dry_run: bool = False,
) -> int:
    """
    Backfill targets for a list of timestamps.

    Args:
        ohlcv_duckdb: Path to OHLCV DuckDB
        table: OHLCV table name
        dataset: Dataset identifier (e.g., binance_btcusdt_perp_1h)
        target_key: Target key name (e.g., y_tp_before_sl_u0.04_d0.02_24h)
        horizon_bars: Horizon in bars
        up_pct: Upside barrier percentage
        down_pct: Downside barrier percentage
        freq: Data frequency (1H, 4H, 1D, etc.)
        timestamps: List of timestamps to backfill
        tie_policy: Tie-breaking policy
        buffer_hours: Hours of buffer data to load before first timestamp
        skip_existing: Skip timestamps that already have targets
        dry_run: If True, don't write to database

    Returns:
        Number of targets written
    """
    if not timestamps:
        print("No timestamps to process")
        return 0

    # Connect to DuckDB to check/create targets table
    con = duckdb.connect(ohlcv_duckdb, read_only=dry_run)

    # Ensure targets table exists (unless dry run)
    if not dry_run:
        _ensure_targets_table(con)

    # Close connection before load_ohlcv_duckdb opens its own
    con.close()

    # Load OHLCV with buffer (this opens its own connection)
    min_ts = min(timestamps)
    max_ts = max(timestamps)
    start_load = min_ts - pd.Timedelta(hours=buffer_hours)
    # Need horizon_bars of future data
    end_load = max_ts + pd.Timedelta(hours=horizon_bars + 1)

    print(f"Loading OHLCV from {start_load} to {end_load}...")
    df_ohlcv = load_ohlcv_duckdb(ohlcv_duckdb, table=table, start=start_load, end=end_load)

    if df_ohlcv.empty:
        print("ERROR: No OHLCV data loaded", file=sys.stderr)
        return 0

    # Set timestamp as index for fast lookup
    df_ohlcv['timestamp'] = pd.to_datetime(df_ohlcv['timestamp'])
    df_ohlcv = df_ohlcv.set_index('timestamp').sort_index()

    print(f"Loaded {len(df_ohlcv)} OHLCV rows")
    print(f"Processing {len(timestamps)} timestamps...")

    # Filter out existing targets (reopen connection for check)
    timestamps_to_process = []
    if skip_existing and not dry_run:
        con = duckdb.connect(ohlcv_duckdb, read_only=True)
        for ts in timestamps:
            if not _target_exists(con, dataset, target_key, ts):
                timestamps_to_process.append(ts)
        con.close()
        skipped = len(timestamps) - len(timestamps_to_process)
        if skipped > 0:
            print(f"Skipping {skipped} existing targets")
    else:
        timestamps_to_process = timestamps

    if not timestamps_to_process:
        print("All targets already exist")
        return 0

    # Generate targets
    targets_data = []
    success_count = 0
    fail_count = 0

    for i, ts in enumerate(timestamps_to_process):
        if (i + 1) % 100 == 0 or (i + 1) == len(timestamps_to_process):
            print(f"  Processing {i+1}/{len(timestamps_to_process)}...")

        target_value = generate_target_for_timestamp(
            df_ohlcv, ts, horizon_bars, up_pct, down_pct, target_key, tie_policy
        )

        # Check for both None and NaN
        if target_value is not None and np.isfinite(target_value):
            targets_data.append({
                'timestamp': ts,
                'dataset': dataset,
                'target_key': target_key,
                'target_value': float(target_value),
            })
            success_count += 1
        else:
            fail_count += 1

    print(f"Generated {success_count} targets ({fail_count} failed)")

    if dry_run:
        print("\n[DRY RUN] Would insert the following targets:")
        if targets_data:
            df_preview = pd.DataFrame(targets_data)
            print(f"\nShowing all {len(df_preview)} rows:")
            print(df_preview.to_string())
        return success_count

    # Insert into database (reopen connection for writing)
    if targets_data:
        con = duckdb.connect(ohlcv_duckdb, read_only=False)
        df_targets = pd.DataFrame(targets_data)

        # Final safety check: drop any rows with null target_value
        initial_len = len(df_targets)
        df_targets = df_targets.dropna(subset=['target_value'])
        if len(df_targets) < initial_len:
            print(f"Warning: Dropped {initial_len - len(df_targets)} rows with NULL target_value")

        if df_targets.empty:
            print("No valid targets to insert")
            con.close()
            return success_count

        print(f"Inserting {len(df_targets)} targets into database...")

        # Use INSERT OR REPLACE to handle duplicates
        con.execute("""
            INSERT OR REPLACE INTO targets (timestamp, dataset, target_key, target_value, created_at)
            SELECT timestamp, dataset, target_key, target_value, CURRENT_TIMESTAMP
            FROM df_targets
        """)

        print(f"✓ Inserted {len(df_targets)} targets")
        con.close()

    return success_count


def main():
    parser = argparse.ArgumentParser(description="Backfill target labels into DuckDB")

    # Database args
    parser.add_argument('--duckdb', required=True, help='Path to OHLCV DuckDB file')
    parser.add_argument('--table', default='ohlcv_btcusdt_1h', help='OHLCV table name')
    parser.add_argument('--dataset', required=True, help='Dataset identifier (e.g., binance_btcusdt_perp_1h)')

    # Target configuration
    parser.add_argument('--target-key', required=True, help='Target key name (e.g., y_tp_before_sl_u0.04_d0.02_24h)')
    parser.add_argument('--horizon-bars', type=int, required=True, help='Horizon in bars (e.g., 24)')
    parser.add_argument('--up-pct', type=float, required=True, help='Upside barrier percentage (e.g., 0.04 for 4%%)')
    parser.add_argument('--down-pct', type=float, required=True, help='Downside barrier percentage (e.g., 0.02 for 2%%)')
    parser.add_argument('--freq', default='1H', help='Data frequency (e.g., 1H, 4H, 1D)')
    parser.add_argument('--tie-policy', choices=['conservative', 'proximity_to_open'], default='conservative')

    # Timestamp selection modes
    parser.add_argument('--mode', choices=['window', 'last_from_targets', 'ts_list'], default='window',
                        help='Timestamp selection mode')
    parser.add_argument('--start', help='Start timestamp for window mode (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)')
    parser.add_argument('--end', help='End timestamp for window mode (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)')
    parser.add_argument('--ts', nargs='*', help='Explicit timestamps for ts_list mode')
    parser.add_argument('--ts-file', help='File containing timestamps (one per line) for ts_list mode')

    # Processing options
    parser.add_argument('--buffer-hours', type=int, default=720, help='Hours of buffer data to load (default: 720 = 30 days)')
    parser.add_argument('--skip-existing', action='store_true', default=True, help='Skip existing targets (default: True)')
    parser.add_argument('--force', action='store_true', help='Overwrite existing targets (sets skip-existing=False)')
    parser.add_argument('--dry-run', action='store_true', help='Preview without writing to database')
    parser.add_argument('--at-most', type=int, help='Process at most N timestamps')

    args = parser.parse_args()

    # Determine timestamps to process
    if args.mode == 'window':
        if not args.start or not args.end:
            parser.error("--start and --end required for window mode")
        start = pd.Timestamp(args.start)
        end = pd.Timestamp(args.end)

        con = duckdb.connect(args.duckdb, read_only=True)
        timestamps = _list_closed_bars_in_window(con, args.table, start, end)
        con.close()

        print(f"Window mode: {len(timestamps)} timestamps from {start} to {end}")

    elif args.mode == 'last_from_targets':
        con = duckdb.connect(args.duckdb, read_only=True)

        # Ensure targets table exists for querying
        if not _is_targets_table_present(con):
            print("ERROR: targets table does not exist. Use window mode for initial backfill.", file=sys.stderr)
            con.close()
            sys.exit(1)

        last_ts = _last_target_ts(con, args.dataset, args.target_key)
        if last_ts is None:
            print(f"No existing targets found for dataset={args.dataset}, target_key={args.target_key}")
            print("Use window mode for initial backfill.", file=sys.stderr)
            con.close()
            sys.exit(1)

        # Get OHLCV table's last timestamp
        ohlcv_max_ts = _last_ohlcv_ts(con, args.table)
        if ohlcv_max_ts is None:
            print(f"ERROR: No data in OHLCV table {args.table}", file=sys.stderr)
            con.close()
            sys.exit(1)

        # Continue from last + 1 hour
        start = last_ts + pd.Timedelta(hours=1)
        # End at OHLCV max timestamp minus horizon (to ensure we have enough forward data)
        end = ohlcv_max_ts - pd.Timedelta(hours=args.horizon_bars)

        if start > end:
            print(f"No new data to backfill. Last target: {last_ts}, OHLCV max: {ohlcv_max_ts}")
            con.close()
            return

        timestamps = _list_closed_bars_in_window(con, args.table, start, end)
        con.close()

        print(f"OHLCV data available up to: {ohlcv_max_ts}")
        print(f"Continuing from {start} to {end}: {len(timestamps)} new timestamps")

    elif args.mode == 'ts_list':
        ts_inputs = args.ts or []
        if args.ts_file:
            with open(args.ts_file) as f:
                ts_inputs.extend(line.strip() for line in f if line.strip())

        if not ts_inputs:
            parser.error("--ts or --ts-file required for ts_list mode")

        timestamps = _parse_ts_list(ts_inputs)
        print(f"Explicit timestamps: {len(timestamps)}")

    else:
        parser.error(f"Unknown mode: {args.mode}")

    # Apply at-most limit
    if args.at_most and len(timestamps) > args.at_most:
        print(f"Limiting to first {args.at_most} timestamps")
        timestamps = timestamps[:args.at_most]

    # Backfill
    skip_existing = not args.force if args.force else args.skip_existing

    count = backfill_targets(
        ohlcv_duckdb=args.duckdb,
        table=args.table,
        dataset=args.dataset,
        target_key=args.target_key,
        horizon_bars=args.horizon_bars,
        up_pct=args.up_pct,
        down_pct=args.down_pct,
        freq=args.freq,
        timestamps=timestamps,
        tie_policy=args.tie_policy,
        buffer_hours=args.buffer_hours,
        skip_existing=skip_existing,
        dry_run=args.dry_run,
    )

    if not args.dry_run:
        print(f"\n✓ Backfilled {count} targets for {args.dataset}/{args.target_key}")


if __name__ == '__main__':
    main()
