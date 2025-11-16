#!/usr/bin/env python3
"""
Backfill feature snapshots into DuckDB for selected hourly bars.

This script computes full multi-timeframe features for each target timestamp
directly from OHLCV in DuckDB and persists them into a `features` table as a
JSON map, keyed by (feature_key, ts). It does not perform any model inference.

Selection modes for target timestamps (closed bars only):
  - window: explicit --start/--end (defaults to last 48h ending at latest closed)
  - last_from_features: continue from last features.ts + 1h for the given feature_key
  - ts_list: explicit timestamps via --ts or --ts-file

Example:
  python scripts/backfill_features.py \
    --duckdb "/Volumes/Extreme SSD/trading_data/cex/db/binance_btcusdt_perp_ohlcv.duckdb" \
    --feat-duckdb "/Volumes/Extreme SSD/trading_data/cex/db/binance_btcusdt_perp_feature.duckdb" \
    --table ohlcv_btcusdt_1h \
    --feature-key "prod_default" \
    --mode window --start "2025-08-01 00:00:00" --end "2025-08-07 23:00:00" \
    --buffer-hours 6 --at-most 200
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import duckdb  # type: ignore
import pandas as pd

from runtime.data_loader import load_ohlcv_duckdb, validate_hourly_continuity
from runtime.lookbacks_builder import build_latest_lookbacks, trim_lookbacks_to_base_window, validate_lookbacks_exact
from runtime.features_builder import compute_latest_features_from_lookbacks


DEFAULT_TABLE = "ohlcv_btcusdt_1h"


def _now_floor_utc() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(timezone.utc)).floor("h").tz_convert(None)


def _is_features_table_present(con) -> bool:
    try:
        res = con.execute(
            "SELECT 1 FROM information_schema.tables WHERE table_name = 'features' LIMIT 1"
        ).fetchone()
        return bool(res)
    except Exception:
        return False


def _feature_exists(con, feature_key: str, ts: pd.Timestamp) -> bool:
    try:
        row = con.execute(
            "SELECT 1 FROM features WHERE feature_key = ? AND ts = ? LIMIT 1",
            [str(feature_key), pd.Timestamp(ts).to_pydatetime()],
        ).fetchone()
        return row is not None
    except Exception:
        return False


def _list_closed_bars_in_window(
    con: duckdb.DuckDBPyConnection, table: str, start: pd.Timestamp, end: pd.Timestamp
) -> List[pd.Timestamp]:
    now_floor = _now_floor_utc()
    cutoff = now_floor - pd.Timedelta(hours=1)
    start = pd.to_datetime(start).tz_localize(None)
    end = pd.to_datetime(end).tz_localize(None)
    q = f"""
        SELECT timestamp FROM {table}
        WHERE timestamp BETWEEN ? AND ? AND timestamp <= ?
        ORDER BY timestamp
    """
    df = con.execute(q, [start.to_pydatetime(), end.to_pydatetime(), cutoff.to_pydatetime()]).fetch_df()
    return [pd.Timestamp(t) for t in df["timestamp"]] if not df.empty else []


def _last_features_ts(con, feature_key: str) -> Optional[pd.Timestamp]:
    try:
        row = con.execute("SELECT MAX(ts) FROM features WHERE feature_key = ?", [feature_key]).fetchone()
        if row and row[0] is not None:
            return pd.Timestamp(row[0])
    except Exception:
        pass
    return None


def _parse_ts_list(ts_list: Iterable[str]) -> List[pd.Timestamp]:
    out: List[pd.Timestamp] = []
    for s in ts_list:
        t = pd.to_datetime(s, errors="coerce", utc=True)
        if pd.isna(t):
            continue
        out.append(t.tz_convert("UTC").tz_localize(None))
    return sorted(set(out))


def _load_ts_file(path: Path) -> List[pd.Timestamp]:
    if not path or not path.exists():
        return []
    lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    return _parse_ts_list(lines)


def ensure_features_table(con: duckdb.DuckDBPyConnection) -> None:
    """Create features table if it doesn't exist."""
    con.execute("""
        CREATE TABLE IF NOT EXISTS features (
            feature_key TEXT NOT NULL,
            ts TIMESTAMP NOT NULL,
            features JSON NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (feature_key, ts)
        )
    """)


def upsert_feature_row(con: duckdb.DuckDBPyConnection, feature_key: str, ts: pd.Timestamp, features_dict: dict) -> None:
    """Insert or replace a feature row."""
    import json

    # Delete existing row if present
    con.execute("DELETE FROM features WHERE feature_key = ? AND ts = ?", [feature_key, ts.to_pydatetime()])

    # Insert new row (DuckDB JSON type handles dict automatically via json.dumps)
    con.execute(
        "INSERT INTO features (feature_key, ts, features) VALUES (?, ?, ?)",
        [feature_key, ts.to_pydatetime(), json.dumps(features_dict)]
    )


@dataclass
class BackfillFeaturesConfig:
    duckdb_path: Path  # OHLCV source DB
    feat_duckdb_path: Optional[Path]  # features target DB (defaults to duckdb_path)
    table: str
    feature_key: str
    mode: str
    start: Optional[str]
    end: Optional[str]
    ts: List[str]
    ts_file: Optional[Path]
    buffer_hours: int
    base_hours: int
    at_most: Optional[int]
    dry_run: bool
    overwrite: bool
    timeframes: List[str]
    feature_list: Optional[Path]  # Optional: JSON file with feature list to save


def _select_target_timestamps(cfg: BackfillFeaturesConfig) -> List[pd.Timestamp]:
    """Select closed-hour target timestamps using OHLCV DB and features DB when needed."""
    con_ohlcv = duckdb.connect(str(cfg.duckdb_path))
    try:
        con_ohlcv.execute("SET TimeZone='UTC';")
        now_floor = _now_floor_utc()
        cutoff = now_floor - pd.Timedelta(hours=1)

        if cfg.mode == "ts_list":
            ts_list = _parse_ts_list(cfg.ts)
            if cfg.ts_file:
                ts_list = sorted(set(ts_list) | set(_load_ts_file(cfg.ts_file)))
            if not ts_list:
                return []
            lo, hi = min(ts_list), max(ts_list)
            in_db = _list_closed_bars_in_window(con_ohlcv, cfg.table, lo, hi)
            ts = sorted(set(ts_list).intersection(set(in_db)))
            return [t for t in ts if t <= cutoff]

        if cfg.mode == "last_from_features":
            feat_db_path = cfg.feat_duckdb_path or cfg.duckdb_path
            try:
                con_feat = duckdb.connect(str(feat_db_path))
                con_feat.execute("SET TimeZone='UTC';")
                last_feat = _last_features_ts(con_feat, cfg.feature_key)
            finally:
                try:
                    con_feat.close()
                except Exception:
                    pass

            start_ts = (last_feat + pd.Timedelta(hours=1)) if last_feat is not None else None
            end_ts = pd.to_datetime(cfg.end).tz_localize(None) if cfg.end else cutoff
            if start_ts is None:
                start_ts = end_ts - pd.Timedelta(hours=48)
            return _list_closed_bars_in_window(con_ohlcv, cfg.table, start_ts, end_ts)

        # Default: window mode
        if not cfg.start and not cfg.end:
            end_ts = cutoff
            start_ts = end_ts - pd.Timedelta(hours=48)
        else:
            start_ts = pd.to_datetime(cfg.start).tz_localize(None) if cfg.start else None
            end_ts = pd.to_datetime(cfg.end).tz_localize(None) if cfg.end else cutoff
            if start_ts is None:
                start_ts = end_ts - pd.Timedelta(hours=48)
        return _list_closed_bars_in_window(con_ohlcv, cfg.table, start_ts, end_ts)
    finally:
        con_ohlcv.close()


def backfill_features(cfg: BackfillFeaturesConfig) -> int:
    # Determine target timestamps
    targets = _select_target_timestamps(cfg)
    if cfg.at_most is not None and cfg.at_most > 0:
        targets = targets[: cfg.at_most]

    if not targets:
        print("No target timestamps found to backfill features (after filtering/limits)")
        return 0

    # Load feature list if provided to filter features
    feature_filter = None
    if cfg.feature_list:
        try:
            import json
            with open(cfg.feature_list) as f:
                feature_filter = json.load(f)
            if not isinstance(feature_filter, list):
                print(f"[ERROR] Feature list must be a JSON array")
                return 2
            print(f"Loaded feature list: {len(feature_filter)} features from {cfg.feature_list}")
        except Exception as e:
            print(f"[ERROR] Failed to load feature list: {e}")
            return 2

    print(
        f"Backfill features plan: bars={len(targets)} "
        f"range=[{targets[0]} .. {targets[-1]}] base={cfg.base_hours}h buffer={cfg.buffer_hours}h key={cfg.feature_key}"
    )
    if cfg.dry_run:
        for t in targets:
            print("  -", t)
        return 0

    # Compute earliest-needed OHLCV and load in one shot
    required_hours = int(cfg.base_hours + max(0, cfg.buffer_hours))
    earliest_needed = targets[0] - pd.Timedelta(hours=required_hours - 1)
    df_all = load_ohlcv_duckdb(
        str(cfg.duckdb_path), table=cfg.table, start=earliest_needed, end=targets[-1]
    )
    if df_all.empty:
        print("[ERROR] No OHLCV rows loaded from DuckDB for required range")
        return 2

    # Prepare target DB connection
    feat_db_path = cfg.feat_duckdb_path or cfg.duckdb_path
    con_feat = duckdb.connect(str(feat_db_path))
    con_feat.execute("SET TimeZone='UTC';")
    ensure_features_table(con_feat)

    appended = 0
    try:
        for ts in targets:
            # Skip if exists and not overwriting
            if not cfg.overwrite and _is_features_table_present(con_feat) and _feature_exists(con_feat, cfg.feature_key, ts):
                print(f"SKIP {ts}: features already present for key={cfg.feature_key}")
                continue

            # Ensure sufficient history and continuity
            try:
                validate_hourly_continuity(df_all, end_ts=ts, required_hours=required_hours)
            except Exception as e:
                print(f"SKIP {ts}: insufficient/irregular history window ({e})")
                continue

            # Build lookbacks and compute features for this ts
            df_slice = df_all[df_all["timestamp"] <= ts].copy()
            lookbacks = build_latest_lookbacks(df_slice, window_hours=required_hours, timeframes=cfg.timeframes)
            lookbacks = trim_lookbacks_to_base_window(lookbacks, base_hours=cfg.base_hours)
            try:
                validate_lookbacks_exact(lookbacks, base_hours=cfg.base_hours, end_ts=ts)
            except Exception as e:
                print(f"SKIP {ts}: lookbacks invalid ({e})")
                continue

            features_row = compute_latest_features_from_lookbacks(lookbacks)

            # Convert features to dict (exclude timestamp column)
            features_dict = features_row.drop(columns=['timestamp']).iloc[0].to_dict()

            # Filter to specified features if feature list provided
            if feature_filter is not None:
                missing_features = [f for f in feature_filter if f not in features_dict]
                if missing_features:
                    print(f"SKIP {ts}: missing {len(missing_features)} required features (first 10): {missing_features[:10]}")
                    continue
                features_dict = {k: features_dict[k] for k in feature_filter}

            # Persist feature map
            try:
                upsert_feature_row(con_feat, cfg.feature_key, ts, features_dict)
                appended += 1
                print(f"OK {ts}: features upserted for key={cfg.feature_key} ({len(features_dict)} features)")
            except Exception as e:
                print(f"SKIP {ts}: cannot persist features ({e})")
                continue
    finally:
        try:
            con_feat.close()
        except Exception:
            pass

    print(f"Backfill features complete: wrote={appended} rows out of {len(targets)} planned")
    return 0


def parse_args(argv: Optional[List[str]] = None) -> BackfillFeaturesConfig:
    p = argparse.ArgumentParser(description="Backfill multi-timeframe features into a DuckDB features table")
    p.add_argument("--duckdb", type=Path, required=True, help="DuckDB database path for OHLCV source")
    p.add_argument("--feat-duckdb", type=Path, default=None, help="DuckDB path for features table (default: --duckdb)")
    p.add_argument("--table", type=str, default=DEFAULT_TABLE, help="OHLCV table name in DuckDB")
    p.add_argument("--feature-key", required=True, help="Feature snapshot key to associate with generated rows")
    p.add_argument("--mode", choices=["window", "last_from_features", "ts_list"], default="window")
    p.add_argument("--start", default=None, help="Start timestamp (inclusive) for window mode")
    p.add_argument("--end", default=None, help="End timestamp (inclusive) for window/last_from_features")
    p.add_argument("--ts", nargs="*", default=[], help="Explicit timestamps for ts_list mode")
    p.add_argument("--ts-file", type=Path, default=None, help="File with one timestamp per line for ts_list mode")
    p.add_argument("--buffer-hours", type=int, default=0, help="Extra hours on top of base window for lookbacks (default 0)")
    p.add_argument("--base-hours", type=int, default=30 * 24, help="Base training window in hours (default 720)")
    p.add_argument("--at-most", type=int, default=None, help="Cap the number of bars to process")
    p.add_argument("--overwrite", action="store_true", help="Replace existing feature rows for (feature_key, ts)")
    p.add_argument("--dry-run", action="store_true", help="Plan only; do not compute or write features")
    p.add_argument("--timeframes", nargs="+", default=["1H", "4H", "12H", "1D"], help="Timeframes to use for lookbacks")
    p.add_argument("--feature-list", type=Path, default=None, help="Optional: JSON file with feature list to save (filters computed features)")
    args = p.parse_args(argv)

    return BackfillFeaturesConfig(
        duckdb_path=args.duckdb,
        feat_duckdb_path=args.feat_duckdb,
        table=str(args.table),
        feature_key=str(args.feature_key),
        mode=str(args.mode),
        start=args.start,
        end=args.end,
        ts=list(args.ts or []),
        ts_file=args.ts_file,
        buffer_hours=int(args.buffer_hours),
        base_hours=int(args.base_hours),
        at_most=(int(args.at_most) if args.at_most is not None else None),
        dry_run=bool(args.dry_run),
        overwrite=bool(args.overwrite),
        timeframes=list(args.timeframes or ["1H", "4H", "12H", "1D"]),
        feature_list=args.feature_list,
    )


def main(argv: Optional[List[str]] = None) -> int:
    cfg = parse_args(argv)
    try:
        return backfill_features(cfg)
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
