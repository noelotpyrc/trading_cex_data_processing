#!/usr/bin/env python3
"""
Backfill merged data from 5 DuckDB sources into a unified table.

This script merges data from multiple source databases (perp OHLCV, open interest,
long/short ratio, premium index, spot OHLCV) into two unified tables:
- unified_1h_training: with zero-fill handling for model training
- unified_1h_inference: raw data without handling for live inference

Selection modes for target timestamps:
  - window: explicit --start/--end
  - last_from_merged: continue from last unified_1h_training.timestamp + 1h
  - ts_list: explicit timestamps via --ts or --ts-file

Example:
  python scripts/backfill_merged_data.py \
    --db-dir "/Volumes/Extreme SSD/trading_data/cex/db" \
    --unified-db "/Volumes/Extreme SSD/trading_data/cex/db/binance_btcusdt_unified.duckdb" \
    --mode window --start "2024-01-01" --end "2024-12-31" \
    --at-most 1000 --dry-run
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import duckdb
import numpy as np
import pandas as pd


# === Data Source Configuration ===
DB_SOURCES = {
    'perp_ohlcv': {
        'file': 'binance_btcusdt_perp_ohlcv.duckdb',
        'table': 'ohlcv_btcusdt_1h',
        'columns': ['timestamp', 'open', 'high', 'low', 'close', 'volume'],
        'rename': {},
        'required': False,  # Will warn if missing, not error
    },
    'open_interest': {
        'file': 'binance_btcusdt_perp_open_interest.duckdb',
        'table': 'open_interest_btcusdt_1h',
        'columns': ['timestamp', 'sum_open_interest', 'sum_open_interest_value'],
        'rename': {},
        'required': False,
    },
    'long_short_ratio': {
        'file': 'binance_btcusdt_perp_long_short_ratio.duckdb',
        'table': 'long_short_ratio_btcusdt_1h',
        'columns': ['timestamp', 'long_short_ratio', 'long_account', 'short_account'],
        'rename': {},
        'required': False,
    },
    'premium_index': {
        'file': 'binance_btcusdt_perp_premium_index.duckdb',
        'table': 'premium_index_btcusdt_1h',
        'columns': ['timestamp', 'open', 'high', 'low', 'close'],
        'rename': {
            'open': 'premium_idx_open',
            'high': 'premium_idx_high',
            'low': 'premium_idx_low',
            'close': 'premium_idx_close',
        },
        'required': False,
    },
    'spot_ohlcv': {
        'file': 'binance_btcusdt_spot_ohlcv.duckdb',
        'table': 'ohlcv_btcusdt_1h',
        'columns': ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'num_trades', 'taker_buy_base_volume'],
        'rename': {
            'open': 'spot_open',
            'high': 'spot_high',
            'low': 'spot_low',
            'close': 'spot_close',
            'volume': 'spot_volume',
            'num_trades': 'spot_num_trades',
            'taker_buy_base_volume': 'spot_taker_buy_volume',
        },
        'required': False,
    },
}

# Columns to skip zero-fill (zeros are valid values for premium)
SKIP_ZERO_FILL = ['premium_idx_open', 'premium_idx_high', 'premium_idx_low', 'premium_idx_close']


def _now_floor_utc() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(timezone.utc)).floor("h").tz_convert(None)


def _generate_hourly_range(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Generate gap-free hourly timestamp base."""
    timestamps = pd.date_range(start=start, end=end, freq='h')
    return pd.DataFrame({'timestamp': timestamps})


def load_source_safe(source_name: str, config: dict, db_dir: Path,
                     start: pd.Timestamp, end: pd.Timestamp) -> Tuple[Optional[pd.DataFrame], List[str]]:
    """
    Load data from a DuckDB source with error handling.
    
    Returns:
        Tuple of (DataFrame or None, list of warnings)
    """
    warnings = []
    db_path = db_dir / config['file']
    
    if not db_path.exists():
        warnings.append(f"Source DB not found: {db_path}")
        return None, warnings
    
    try:
        conn = duckdb.connect(str(db_path), read_only=True)
        conn.execute("SET TimeZone='UTC';")
        
        cols = ', '.join(config['columns'])
        query = f"""
            SELECT {cols} FROM {config['table']}
            WHERE timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp
        """
        
        df = conn.execute(query, [start.to_pydatetime(), end.to_pydatetime()]).fetchdf()
        conn.close()
        
        # Rename columns if specified
        if config['rename']:
            df = df.rename(columns=config['rename'])
        
        # Ensure timestamp is datetime without timezone
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
        
        # Check for duplicates
        dup_count = df['timestamp'].duplicated().sum()
        if dup_count > 0:
            warnings.append(f"{source_name}: {dup_count} duplicate timestamps found")
            df = df.drop_duplicates(subset=['timestamp'], keep='last')
        
        return df, warnings
        
    except Exception as e:
        warnings.append(f"{source_name}: Failed to load - {e}")
        return None, warnings


def merge_sources(db_dir: Path, start: pd.Timestamp, end: pd.Timestamp) -> Tuple[pd.DataFrame, List[str]]:
    """
    Merge all sources into unified dataframe (raw, no ffill).
    
    Returns:
        Tuple of (merged_df, warnings)
    """
    all_warnings = []
    
    # Step 1: Generate gap-free timestamp base
    base = _generate_hourly_range(start, end)
    print(f"  Generated {len(base):,} hourly timestamps from {start} to {end}")
    
    # Step 2: Load and merge each source
    for source_name, config in DB_SOURCES.items():
        df_source, warnings = load_source_safe(source_name, config, db_dir, start, end)
        all_warnings.extend(warnings)
        
        if df_source is None:
            # Add NaN columns for missing source
            for col in config['columns']:
                if col == 'timestamp':
                    continue
                col_name = config['rename'].get(col, col)
                base[col_name] = np.nan
            print(f"  + {source_name}: MISSING - added NaN columns")
        else:
            # Get non-timestamp columns to merge
            merge_cols = ['timestamp'] + [c for c in df_source.columns if c != 'timestamp']
            
            # Left join on timestamp base
            base = base.merge(df_source[merge_cols], on='timestamp', how='left')
            
            # Count unmatched
            new_cols = [c for c in df_source.columns if c != 'timestamp']
            na_count = base[new_cols[0]].isna().sum() if new_cols else 0
            
            print(f"  + {source_name}: {len(df_source):,} rows, {na_count:,} unmatched ({100*na_count/len(base):.1f}%)")
    
    return base, all_warnings


def ensure_unified_table(con: duckdb.DuckDBPyConnection) -> None:
    """Create unified_1h table if it doesn't exist."""
    
    schema = """
        timestamp TIMESTAMP NOT NULL,
        open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, volume DOUBLE,
        sum_open_interest DOUBLE, sum_open_interest_value DOUBLE,
        long_short_ratio DOUBLE, long_account DOUBLE, short_account DOUBLE,
        premium_idx_open DOUBLE, premium_idx_high DOUBLE, 
        premium_idx_low DOUBLE, premium_idx_close DOUBLE,
        spot_open DOUBLE, spot_high DOUBLE, spot_low DOUBLE, spot_close DOUBLE,
        spot_volume DOUBLE, spot_num_trades DOUBLE, spot_taker_buy_volume DOUBLE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    """
    
    con.execute(f"CREATE TABLE IF NOT EXISTS unified_1h ({schema})")


def insert_rows(con: duckdb.DuckDBPyConnection, table: str, df: pd.DataFrame) -> int:
    """Insert rows into table. Returns count of inserted rows."""
    if df.empty:
        return 0
    
    # Add created_at column
    df = df.copy()
    df['created_at'] = datetime.now(timezone.utc)
    
    # Get column order from table
    cols = [c for c in df.columns]
    placeholders = ', '.join(['?' for _ in cols])
    col_names = ', '.join(cols)
    
    query = f"INSERT INTO {table} ({col_names}) VALUES ({placeholders})"
    
    count = 0
    for _, row in df.iterrows():
        try:
            con.execute(query, list(row))
            count += 1
        except Exception as e:
            print(f"  Warning: Failed to insert row for {row['timestamp']}: {e}")
    
    return count


def _last_merged_ts(con: duckdb.DuckDBPyConnection) -> Optional[pd.Timestamp]:
    """Get last timestamp in unified_1h table."""
    try:
        row = con.execute("SELECT MAX(timestamp) FROM unified_1h").fetchone()
        if row and row[0] is not None:
            return pd.Timestamp(row[0])
    except Exception:
        pass
    return None


def _parse_ts_list(ts_list: List[str]) -> List[pd.Timestamp]:
    """Parse list of timestamp strings."""
    out = []
    for s in ts_list:
        t = pd.to_datetime(s, errors="coerce")
        if pd.isna(t):
            continue
        out.append(pd.Timestamp(t).tz_localize(None))
    return sorted(set(out))


def _load_ts_file(path: Path) -> List[pd.Timestamp]:
    """Load timestamps from file."""
    if not path or not path.exists():
        return []
    lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    return _parse_ts_list(lines)


@dataclass
class BackfillMergedDataConfig:
    db_dir: Path
    unified_db_path: Path
    output_csv: Optional[Path]  # If set, output CSV instead of DB
    mode: str
    start: Optional[str]
    end: Optional[str]
    ts: List[str]
    ts_file: Optional[Path]
    at_most: Optional[int]
    dry_run: bool


def select_target_range(cfg: BackfillMergedDataConfig) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    Select start/end range based on mode.
    
    Returns:
        Tuple of (start_ts, end_ts)
    """
    now_floor = _now_floor_utc()
    cutoff = now_floor - pd.Timedelta(hours=1)  # Last closed bar
    
    if cfg.mode == "ts_list":
        ts_list = _parse_ts_list(cfg.ts)
        if cfg.ts_file:
            ts_list = sorted(set(ts_list) | set(_load_ts_file(cfg.ts_file)))
        if not ts_list:
            raise ValueError("No valid timestamps provided for ts_list mode")
        return min(ts_list), min(max(ts_list), cutoff)
    
    if cfg.mode == "last_from_merged":
        # Connect to unified DB to get last timestamp
        if cfg.unified_db_path.exists():
            con = duckdb.connect(str(cfg.unified_db_path), read_only=True)
            con.execute("SET TimeZone='UTC';")
            last_ts = _last_merged_ts(con)
            con.close()
        else:
            last_ts = None
        
        end_ts = pd.to_datetime(cfg.end).tz_localize(None) if cfg.end else cutoff
        start_ts = (last_ts + pd.Timedelta(hours=1)) if last_ts else (end_ts - pd.Timedelta(hours=48))
        return start_ts, end_ts
    
    # Default: window mode
    if not cfg.start and not cfg.end:
        end_ts = cutoff
        start_ts = end_ts - pd.Timedelta(hours=48)
    else:
        start_ts = pd.to_datetime(cfg.start).tz_localize(None) if cfg.start else None
        end_ts = pd.to_datetime(cfg.end).tz_localize(None) if cfg.end else cutoff
        if start_ts is None:
            start_ts = end_ts - pd.Timedelta(hours=48)
    
    return start_ts, end_ts


def backfill_merged_data(cfg: BackfillMergedDataConfig) -> int:
    """Main backfill function."""
    
    print("=" * 60)
    print("BACKFILL MERGED DATA")
    print(f"  Mode: {cfg.mode}")
    print(f"  DB dir: {cfg.db_dir}")
    print(f"  Output: {cfg.unified_db_path}")
    print("=" * 60)
    
    # Determine target range
    start_ts, end_ts = select_target_range(cfg)
    
    # Apply at_most limit
    if cfg.at_most is not None:
        max_end = start_ts + pd.Timedelta(hours=cfg.at_most - 1)
        end_ts = min(end_ts, max_end)
    
    print(f"\n  Target range: {start_ts} to {end_ts}")
    expected_rows = int((end_ts - start_ts) / pd.Timedelta(hours=1)) + 1
    print(f"  Expected rows: {expected_rows:,}")
    
    if cfg.dry_run:
        print("\n  [DRY RUN] Would merge sources and create unified tables")
        return 0
    
    # Step 1: Merge sources
    print("\n  Merging sources...")
    merged_df, warnings = merge_sources(cfg.db_dir, start_ts, end_ts)
    
    if warnings:
        print("\n  Warnings:")
        for w in warnings:
            print(f"    - {w}")
    
    print(f"\n  Merged {len(merged_df):,} rows")
    
    # Step 2: Save output
    if cfg.output_csv:
        # Save as CSV (for testing)
        print(f"\n  Saving to CSV: {cfg.output_csv}")
        merged_df.to_csv(cfg.output_csv, index=False)
        print(f"  ✓ Saved {len(merged_df):,} rows")
    else:
        # Save to DuckDB
        print(f"\n  Saving to DB: {cfg.unified_db_path}")
        con = duckdb.connect(str(cfg.unified_db_path))
        con.execute("SET TimeZone='UTC';")
        ensure_unified_table(con)
        row_count = insert_rows(con, "unified_1h", merged_df)
        con.close()
        print(f"\n  Inserted {row_count:,} rows into unified_1h")
    
    print("\n  Done!")
    
    return 0


def parse_args(argv: Optional[List[str]] = None) -> BackfillMergedDataConfig:
    p = argparse.ArgumentParser(description="Backfill merged data into unified DuckDB")
    p.add_argument("--db-dir", type=Path, required=True, help="Directory containing source DuckDB files")
    p.add_argument("--unified-db", type=Path, default=None, help="Output unified DuckDB path")
    p.add_argument("--output-csv", type=Path, default=None, help="Output CSV path (for testing, skips DB)")
    p.add_argument("--mode", choices=["window", "last_from_merged", "ts_list"], default="window")
    p.add_argument("--start", default=None, help="Start timestamp (inclusive)")
    p.add_argument("--end", default=None, help="End timestamp (inclusive)")
    p.add_argument("--ts", nargs="*", default=[], help="Explicit timestamps for ts_list mode")
    p.add_argument("--ts-file", type=Path, default=None, help="File with timestamps for ts_list mode")
    p.add_argument("--at-most", type=int, default=None, help="Cap number of hours to process")
    p.add_argument("--dry-run", action="store_true", help="Plan only, don't write")
    args = p.parse_args(argv)
    
    if not args.unified_db and not args.output_csv:
        p.error("Must specify --unified-db or --output-csv")
    
    return BackfillMergedDataConfig(
        db_dir=args.db_dir,
        unified_db_path=args.unified_db,
        output_csv=args.output_csv,
        mode=args.mode,
        start=args.start,
        end=args.end,
        ts=list(args.ts or []),
        ts_file=args.ts_file,
        at_most=args.at_most,
        dry_run=args.dry_run,
    )


def main(argv: Optional[List[str]] = None) -> int:
    cfg = parse_args(argv)
    try:
        return backfill_merged_data(cfg)
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
