#!/usr/bin/env python3
"""
Backfill derived features from unified data into DuckDB.

This script reads from unified_1h_training table and computes derived features
using the same logic as build_derived_features.py. Features are stored as JSON
blobs in the derived_features table of the perp feature database.

Selection modes for target timestamps:
  - window: explicit --start/--end
  - last_from_features: continue from last derived_features.ts + 1h
  - ts_list: explicit timestamps via --ts or --ts-file

Example:
  python scripts/backfill_derived_features.py \
    --unified-db "/Volumes/Extreme SSD/trading_data/cex/db/binance_btcusdt_unified.duckdb" \
    --feat-db "/Volumes/Extreme SSD/trading_data/cex/db/binance_btcusdt_perp_feature.duckdb" \
    --feature-key "derived_v1" \
    --mode window --start "2024-01-01" --end "2024-12-31" \
    --buffer-hours 10000 --at-most 1000 --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import duckdb
import numpy as np
import pandas as pd

# Import feature generation functions
from feature_engineering.primitives import (
    parkinson_volatility, historical_volatility, rsi, adx, zscore
)
from feature_engineering.derived_features import (
    # State Variables
    price_vwap_distance_zscore, price_ema_distance_zscore, price_roc_over_volatility,
    # Momentum - Price
    return_autocorr, variance_ratio,
    # Momentum - OI
    oi_price_accel_product, oi_price_momentum,
    # Momentum - Flow
    taker_imb_cvd_slope, taker_imb_zscore, relative_volume, trade_count_lead_price_corr,
    # Mean-Reversion - Price
    pullback_slope_ema, pullback_slope_vwap, mean_cross_rate_ema, mean_cross_rate_vwap,
    # Mean-Reversion - OI/Premium/Spot
    oi_zscore, oi_ema_distance_zscore, premium_zscore, long_short_ratio_zscore,
    spot_vol_zscore, avg_trade_size_zscore, taker_imb_price_corr, avg_trade_size_price_corr,
    # Regime
    efficiency_avg, vol_ratio, oi_volatility, oi_vol_ratio, cvar_var_ratio, tail_skewness,
    premium_vol_ratio, spot_dominance_zscore, spot_dom_vol_ratio,
    # Interactions
    displacement_speed_product, range_chop_interaction, range_stretch_interaction,
    scaled_acceleration, oi_price_ratio_spread, spot_vol_price_corr, oi_vol_price_corr,
    spot_dom_price_corr, spot_dom_oi_corr, trade_count_oi_corr, trade_count_spot_dom_corr,
    relative_amihud, oi_volume_efficiency, oi_volume_efficiency_signed
)


DEFAULT_FEATURE_LIST = Path(__file__).parent.parent / "configs" / "btcusdt_1h_features.json"

# Columns to skip zero-fill (zeros are valid values for premium)
SKIP_ZERO_FILL = ['premium_idx_open', 'premium_idx_high', 'premium_idx_low', 'premium_idx_close']


def _now_floor_utc() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(timezone.utc)).floor("h").tz_convert(None)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical time features."""
    ts = pd.to_datetime(df['timestamp'])
    
    hour = ts.dt.hour
    df['hour_of_day_sin'] = np.sin(2 * np.pi * hour / 24)
    df['hour_of_day_cos'] = np.cos(2 * np.pi * hour / 24)
    
    dow = ts.dt.dayofweek
    df['day_of_week_sin'] = np.sin(2 * np.pi * dow / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * dow / 7)
    
    wom = (ts.dt.day - 1) // 7 + 1
    df['week_of_month_sin'] = np.sin(2 * np.pi * wom / 5)
    df['week_of_month_cos'] = np.cos(2 * np.pi * wom / 5)
    
    moy = ts.dt.month
    df['month_of_year_sin'] = np.sin(2 * np.pi * moy / 12)
    df['month_of_year_cos'] = np.cos(2 * np.pi * moy / 12)
    
    return df


def apply_training_zero_fill(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply zero-fill logic for training mode.
    
    From build_derived_features.py:
    - Forward-fill zeros for all numeric columns EXCEPT premium
    - Restore original NaN positions after ffill
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in numeric_cols:
        if col in SKIP_ZERO_FILL:
            continue
            
        zero_count = (df[col] == 0).sum()
        if zero_count > 0:
            # Track original NaN positions
            original_nan_mask = df[col].isna()
            
            # Replace zeros with NaN, forward-fill
            df[col] = df[col].replace(0, np.nan).ffill()
            
            # Restore original NaN positions
            df.loc[original_nan_mask, col] = np.nan
    
    return df


def add_primitive_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add primitive features."""
    c, h, l = df['close'], df['high'], df['low']
    
    df['parkinson_volatility_24'] = parkinson_volatility(h, l, 24)
    df['parkinson_volatility_168'] = parkinson_volatility(h, l, 168)
    df['historical_volatility_24'] = historical_volatility(c, 24)
    df['historical_volatility_168'] = historical_volatility(c, 168)
    df['rsi_24'] = rsi(c, 24)
    df['rsi_168'] = rsi(c, 168)
    df['rsi_720'] = rsi(c, 720)
    df['adx_24'] = adx(h, l, c, 24)
    df['adx_168'] = adx(h, l, c, 168)
    df['adx_720'] = adx(h, l, c, 720)
    
    return df


def add_state_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add state variable features."""
    o, h, l, c, v = df['open'], df['high'], df['low'], df['close'], df['volume']
    
    df['price_vwap_distance_zscore_24_168'] = price_vwap_distance_zscore(o, h, l, c, v, 24, 168)
    df['price_vwap_distance_zscore_168_168'] = price_vwap_distance_zscore(o, h, l, c, v, 168, 168)
    df['price_vwap_distance_zscore_720_168'] = price_vwap_distance_zscore(o, h, l, c, v, 720, 168)
    df['price_ema_distance_zscore_24_168'] = price_ema_distance_zscore(c, 24, 168)
    df['price_ema_distance_zscore_24_720'] = price_ema_distance_zscore(c, 24, 720)
    df['price_roc_over_volatility_24_24'] = price_roc_over_volatility(c, h, l, 24, 24)
    df['price_roc_over_volatility_24_168'] = price_roc_over_volatility(c, h, l, 24, 168)
    df['price_roc_over_volatility_24_720'] = price_roc_over_volatility(c, h, l, 24, 720)
    
    return df


def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add momentum features."""
    c = df['close']
    oi = df['sum_open_interest']
    spot_vol = df['spot_volume']
    taker_buy = df['spot_taker_buy_volume']
    num_trades = df['spot_num_trades']
    vol = df['volume']
    ts = df['timestamp']
    
    df['return_autocorr_48'] = return_autocorr(c, 48)
    df['return_autocorr_168'] = return_autocorr(c, 168)
    df['variance_ratio_24_48'] = variance_ratio(c, 24, 48)
    df['variance_ratio_24_168'] = variance_ratio(c, 24, 168)
    df['variance_ratio_24_720'] = variance_ratio(c, 24, 720)
    df['oi_price_accel_product_168'] = oi_price_accel_product(oi, c, 168)
    df['oi_price_momentum_168'] = oi_price_momentum(oi, c, 168)
    df['taker_imb_cvd_slope_24'] = taker_imb_cvd_slope(taker_buy, spot_vol, 24)
    df['taker_imb_cvd_slope_168'] = taker_imb_cvd_slope(taker_buy, spot_vol, 168)
    df['taker_imb_zscore_168'] = taker_imb_zscore(taker_buy, spot_vol, 168)
    df['relative_volume_7'] = relative_volume(vol, ts, 7)
    df['relative_volume_14'] = relative_volume(vol, ts, 14)
    df['relative_volume_30'] = relative_volume(vol, ts, 30)
    df['trade_count_lead_price_corr_24_168'] = trade_count_lead_price_corr(num_trades, c, 24, 168)
    
    return df


def add_mean_reversion_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add mean-reversion features."""
    o, h, l, c, v = df['open'], df['high'], df['low'], df['close'], df['volume']
    oi = df['sum_open_interest']
    prem = df['premium_idx_close']
    ls = df['long_short_ratio']
    spot_vol = df['spot_volume']
    quote_vol = df['spot_volume'] * df['spot_close']
    num_trades = df['spot_num_trades']
    taker_buy = df['spot_taker_buy_volume']
    
    for ema_span in [24, 168, 720]:
        for window in [48, 168]:
            df[f'pullback_slope_ema_{ema_span}_{window}'] = pullback_slope_ema(c, ema_span, window)
    
    for vwap_window in [24, 168, 720]:
        for window in [48, 168]:
            df[f'pullback_slope_vwap_{vwap_window}_{window}'] = pullback_slope_vwap(o, h, l, c, v, vwap_window, window)
    
    for ema_span in [24, 168, 720]:
        for window in [48, 168]:
            df[f'mean_cross_rate_ema_{ema_span}_{window}'] = mean_cross_rate_ema(c, ema_span, window)
    
    for vwap_window in [24, 168, 720]:
        for window in [48, 168]:
            df[f'mean_cross_rate_vwap_{vwap_window}_{window}'] = mean_cross_rate_vwap(o, h, l, c, v, vwap_window, window)
    
    df['oi_zscore_168'] = oi_zscore(oi, 168)
    df['oi_ema_distance_zscore_24_168'] = oi_ema_distance_zscore(oi, 24, 168)
    df['oi_ema_distance_zscore_168_168'] = oi_ema_distance_zscore(oi, 168, 168)
    df['oi_ema_distance_zscore_720_168'] = oi_ema_distance_zscore(oi, 720, 168)
    df['premium_zscore_168'] = premium_zscore(prem, 168)
    df['long_short_ratio_zscore_48'] = long_short_ratio_zscore(ls, 48)
    df['long_short_ratio_zscore_168'] = long_short_ratio_zscore(ls, 168)
    df['spot_vol_zscore_168'] = spot_vol_zscore(spot_vol, 168)
    df['avg_trade_size_zscore_48'] = avg_trade_size_zscore(quote_vol, num_trades, 48)
    df['avg_trade_size_zscore_168'] = avg_trade_size_zscore(quote_vol, num_trades, 168)
    df['taker_imb_price_corr_168'] = taker_imb_price_corr(taker_buy, spot_vol, c, 168)
    df['avg_trade_size_price_corr_168'] = avg_trade_size_price_corr(quote_vol, num_trades, c, 168)
    
    return df


def add_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add regime features."""
    o, h, l, c = df['open'], df['high'], df['low'], df['close']
    oi = df['sum_open_interest']
    prem = df['premium_idx_close']
    spot_vol = df['spot_volume']
    perp_vol = df['volume']
    
    df['efficiency_avg_24'] = efficiency_avg(o, h, l, c, 24)
    df['efficiency_avg_168'] = efficiency_avg(o, h, l, c, 168)
    df['vol_ratio_24_168'] = vol_ratio(h, l, 24, 168)
    df['vol_ratio_24_720'] = vol_ratio(h, l, 24, 720)
    df['oi_volatility_168'] = oi_volatility(oi, 168)
    df['oi_vol_ratio_24_168'] = oi_vol_ratio(oi, 24, 168)
    df['cvar_var_ratio_168'] = cvar_var_ratio(c, 168)
    df['cvar_var_ratio_720'] = cvar_var_ratio(c, 720)
    df['tail_skewness_168'] = tail_skewness(c, 168)
    df['tail_skewness_720'] = tail_skewness(c, 720)
    df['premium_vol_ratio_24_48'] = premium_vol_ratio(prem, 24, 48)
    df['premium_vol_ratio_24_168'] = premium_vol_ratio(prem, 24, 168)
    df['spot_dominance_zscore_168'] = spot_dominance_zscore(spot_vol, perp_vol, 168)
    df['spot_dom_vol_ratio_24_168'] = spot_dom_vol_ratio(spot_vol, perp_vol, 24, 168)
    
    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add interaction features."""
    o, h, l, c, v = df['open'], df['high'], df['low'], df['close'], df['volume']
    oi = df['sum_open_interest']
    spot_vol = df['spot_volume']
    perp_vol = df['volume']
    num_trades = df['spot_num_trades']
    
    df['displacement_speed_product_168_24'] = displacement_speed_product(o, h, l, c, v, 168, 24)
    df['displacement_speed_product_168_48'] = displacement_speed_product(o, h, l, c, v, 168, 48)
    df['displacement_speed_product_720_24'] = displacement_speed_product(o, h, l, c, v, 720, 24)
    df['displacement_speed_product_720_48'] = displacement_speed_product(o, h, l, c, v, 720, 48)
    df['range_chop_interaction_24'] = range_chop_interaction(h, l, o, c, 24)
    df['range_chop_interaction_168'] = range_chop_interaction(h, l, o, c, 168)
    df['range_stretch_interaction_168_24'] = range_stretch_interaction(o, h, l, c, v, 168, 24)
    df['range_stretch_interaction_168_168'] = range_stretch_interaction(o, h, l, c, v, 168, 168)
    df['range_stretch_interaction_720_24'] = range_stretch_interaction(o, h, l, c, v, 720, 24)
    df['range_stretch_interaction_720_168'] = range_stretch_interaction(o, h, l, c, v, 720, 168)
    df['scaled_acceleration_24'] = scaled_acceleration(c, 24)
    df['scaled_acceleration_168'] = scaled_acceleration(c, 168)
    df['oi_price_ratio_spread_24_168'] = oi_price_ratio_spread(oi, c, 24, 168)
    df['spot_vol_price_corr_168'] = spot_vol_price_corr(spot_vol, c, 168)
    df['oi_vol_price_corr_168'] = oi_vol_price_corr(oi, c, 168)
    df['spot_dom_price_corr_24'] = spot_dom_price_corr(spot_vol, perp_vol, c, 24)
    df['spot_dom_price_corr_168'] = spot_dom_price_corr(spot_vol, perp_vol, c, 168)
    df['spot_dom_oi_corr_24'] = spot_dom_oi_corr(spot_vol, perp_vol, oi, 24)
    df['spot_dom_oi_corr_168'] = spot_dom_oi_corr(spot_vol, perp_vol, oi, 168)
    df['trade_count_oi_corr_168'] = trade_count_oi_corr(num_trades, oi, 168)
    df['trade_count_spot_dom_corr_168'] = trade_count_spot_dom_corr(num_trades, spot_vol, perp_vol, 168)
    df['relative_amihud_168'] = relative_amihud(c, v, 168)
    df['oi_volume_efficiency_24_168'] = oi_volume_efficiency(oi, v, 24, 168)
    
    eff_pos, eff_neg = oi_volume_efficiency_signed(oi, v, 48, 168)
    df['oi_volume_efficiency_signed_pos_48_168'] = eff_pos
    df['oi_volume_efficiency_signed_neg_48_168'] = eff_neg
    
    eff_pos2, eff_neg2 = oi_volume_efficiency_signed(oi, v, 168, 168)
    df['oi_volume_efficiency_signed_pos_168_168'] = eff_pos2
    df['oi_volume_efficiency_signed_neg_168_168'] = eff_neg2
    
    return df


def compute_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all derived features."""
    df = add_time_features(df)
    df = add_primitive_features(df)
    df = add_state_features(df)
    df = add_momentum_features(df)
    df = add_mean_reversion_features(df)
    df = add_regime_features(df)
    df = add_interaction_features(df)
    
    # Clean inf values
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    
    return df


def ensure_features_table(con: duckdb.DuckDBPyConnection, table_name: str) -> None:
    """Create features table if it doesn't exist."""
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            feature_key TEXT NOT NULL,
            ts TIMESTAMP NOT NULL,
            features JSON NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (feature_key, ts)
        )
    """)


def _feature_exists(con: duckdb.DuckDBPyConnection, table_name: str, feature_key: str, ts: pd.Timestamp) -> bool:
    """Check if feature row exists."""
    try:
        row = con.execute(
            f"SELECT 1 FROM {table_name} WHERE feature_key = ? AND ts = ? LIMIT 1",
            [feature_key, ts.to_pydatetime()]
        ).fetchone()
        return row is not None
    except Exception:
        return False


def _last_features_ts(con: duckdb.DuckDBPyConnection, table_name: str, feature_key: str) -> Optional[pd.Timestamp]:
    """Get last timestamp for feature_key."""
    try:
        row = con.execute(
            f"SELECT MAX(ts) FROM {table_name} WHERE feature_key = ?",
            [feature_key]
        ).fetchone()
        if row and row[0] is not None:
            return pd.Timestamp(row[0])
    except Exception:
        pass
    return None


def upsert_feature_row(con: duckdb.DuckDBPyConnection, table_name: str, feature_key: str, 
                        ts: pd.Timestamp, features_dict: Dict) -> None:
    """Insert or replace a feature row."""
    con.execute(
        f"DELETE FROM {table_name} WHERE feature_key = ? AND ts = ?",
        [feature_key, ts.to_pydatetime()]
    )
    con.execute(
        f"INSERT INTO {table_name} (feature_key, ts, features) VALUES (?, ?, ?)",
        [feature_key, ts.to_pydatetime(), json.dumps(features_dict)]
    )


def _parse_ts_list(ts_list: List[str]) -> List[pd.Timestamp]:
    out = []
    for s in ts_list:
        t = pd.to_datetime(s, errors="coerce")
        if pd.isna(t):
            continue
        out.append(pd.Timestamp(t).tz_localize(None))
    return sorted(set(out))


def _load_ts_file(path: Path) -> List[pd.Timestamp]:
    if not path or not path.exists():
        return []
    lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    return _parse_ts_list(lines)


@dataclass
class BackfillDerivedFeaturesConfig:
    unified_db_path: Path
    feat_db_path: Path
    table_name: str  # Output table name
    feature_key: str
    feature_list_path: Optional[Path]
    mode: str
    start: Optional[str]
    end: Optional[str]
    ts: List[str]
    ts_file: Optional[Path]
    buffer_hours: int
    at_most: Optional[int]
    dry_run: bool
    overwrite: bool
    for_training: bool  # If True, apply ffill before computing features


def select_target_timestamps(cfg: BackfillDerivedFeaturesConfig,
                              con_unified: duckdb.DuckDBPyConnection,
                              con_feat: duckdb.DuckDBPyConnection) -> List[pd.Timestamp]:
    """Select timestamps to process."""
    now_floor = _now_floor_utc()
    cutoff = now_floor - pd.Timedelta(hours=1)
    
    # Get available timestamps from unified table
    available = con_unified.execute(
        "SELECT timestamp FROM unified_1h ORDER BY timestamp"
    ).fetchdf()
    available_ts = set(pd.to_datetime(available['timestamp']).dt.tz_localize(None))
    
    if cfg.mode == "ts_list":
        ts_list = _parse_ts_list(cfg.ts)
        if cfg.ts_file:
            ts_list = sorted(set(ts_list) | set(_load_ts_file(cfg.ts_file)))
        return sorted([t for t in ts_list if t in available_ts and t <= cutoff])
    
    if cfg.mode == "last_from_features":
        last_ts = _last_features_ts(con_feat, cfg.table_name, cfg.feature_key)
        end_ts = pd.to_datetime(cfg.end).tz_localize(None) if cfg.end else cutoff
        start_ts = (last_ts + pd.Timedelta(hours=1)) if last_ts else (end_ts - pd.Timedelta(hours=48))
        return sorted([t for t in available_ts if start_ts <= t <= end_ts])
    
    # window mode
    if not cfg.start and not cfg.end:
        end_ts = cutoff
        start_ts = end_ts - pd.Timedelta(hours=48)
    else:
        start_ts = pd.to_datetime(cfg.start).tz_localize(None) if cfg.start else None
        end_ts = pd.to_datetime(cfg.end).tz_localize(None) if cfg.end else cutoff
        if start_ts is None:
            start_ts = end_ts - pd.Timedelta(hours=48)
    
    return sorted([t for t in available_ts if start_ts <= t <= end_ts])


def backfill_derived_features(cfg: BackfillDerivedFeaturesConfig) -> int:
    """Main backfill function."""
    
    print("=" * 60)
    print("BACKFILL DERIVED FEATURES")
    print(f"  Mode: {cfg.mode}")
    print(f"  Table: {cfg.table_name}")
    print(f"  Feature key: {cfg.feature_key}")
    print(f"  For training: {cfg.for_training} {'(with ffill)' if cfg.for_training else '(raw)'}")
    print(f"  Unified DB: {cfg.unified_db_path}")
    print(f"  Feature DB: {cfg.feat_db_path}")
    print("=" * 60)
    
    # Load feature list for validation
    feature_list = None
    if cfg.feature_list_path and cfg.feature_list_path.exists():
        with open(cfg.feature_list_path) as f:
            config = json.load(f)
            feature_list = config.get('features', [])
        print(f"\n  Feature list: {len(feature_list)} features from {cfg.feature_list_path}")
    
    # Connect to databases
    if not cfg.unified_db_path.exists():
        print(f"\n  [ERROR] Unified DB not found: {cfg.unified_db_path}")
        return 1
    
    con_unified = duckdb.connect(str(cfg.unified_db_path), read_only=True)
    con_unified.execute("SET TimeZone='UTC';")
    
    con_feat = duckdb.connect(str(cfg.feat_db_path))
    con_feat.execute("SET TimeZone='UTC';")
    ensure_features_table(con_feat, cfg.table_name)
    
    # Select target timestamps
    targets = select_target_timestamps(cfg, con_unified, con_feat)
    
    if cfg.at_most is not None:
        targets = targets[:cfg.at_most]
    
    if not targets:
        print("\n  No target timestamps found")
        con_unified.close()
        con_feat.close()
        return 0
    
    print(f"\n  Target timestamps: {len(targets)}")
    print(f"  Range: {targets[0]} to {targets[-1]}")
    print(f"  Buffer hours: {cfg.buffer_hours}")
    
    if cfg.dry_run:
        print("\n  [DRY RUN] Would process these timestamps:")
        for t in targets[:10]:
            print(f"    - {t}")
        if len(targets) > 10:
            print(f"    ... and {len(targets) - 10} more")
        con_unified.close()
        con_feat.close()
        return 0
    
    # Load all required data (targets + buffer)
    earliest = targets[0] - pd.Timedelta(hours=cfg.buffer_hours)
    latest = targets[-1]
    
    print(f"\n  Loading data from {earliest} to {latest}...")
    
    df_all = con_unified.execute(
        "SELECT * FROM unified_1h WHERE timestamp >= ? AND timestamp <= ? ORDER BY timestamp",
        [earliest.to_pydatetime(), latest.to_pydatetime()]
    ).fetchdf()
    df_all['timestamp'] = pd.to_datetime(df_all['timestamp']).dt.tz_localize(None)
    
    print(f"  Loaded {len(df_all):,} rows")
    
    if df_all.empty:
        print("\n  [ERROR] No data loaded")
        con_unified.close()
        con_feat.close()
        return 1
    
    # Apply ffill for training mode
    if cfg.for_training:
        print("  Applying training zero-fill...")
        df_all = apply_training_zero_fill(df_all)
    
    # Compute all features on full dataset
    print("\n  Computing features...")
    df_features = compute_all_features(df_all.copy())
    
    # Get feature columns (exclude raw data columns)
    raw_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume',
                'sum_open_interest', 'sum_open_interest_value',
                'long_short_ratio', 'long_account', 'short_account',
                'premium_idx_open', 'premium_idx_high', 'premium_idx_low', 'premium_idx_close',
                'spot_open', 'spot_high', 'spot_low', 'spot_close',
                'spot_volume', 'spot_num_trades', 'spot_taker_buy_volume', 'created_at']
    feature_cols = [c for c in df_features.columns if c not in raw_cols]
    
    print(f"  Generated {len(feature_cols)} feature columns")
    
    # Process each target timestamp
    processed = 0
    skipped = 0
    
    for ts in targets:
        # Skip if exists and not overwriting
        if not cfg.overwrite and _feature_exists(con_feat, cfg.table_name, cfg.feature_key, ts):
            skipped += 1
            continue
        
        # Get row for this timestamp
        row = df_features[df_features['timestamp'] == ts]
        if row.empty:
            print(f"  SKIP {ts}: not in computed features")
            continue
        
        # Extract features as dict
        features_dict = row[feature_cols].iloc[0].to_dict()
        
        # Filter to feature list if provided
        if feature_list:
            missing = [f for f in feature_list if f not in features_dict]
            if missing:
                print(f"  WARN {ts}: missing {len(missing)} features")
            features_dict = {k: features_dict[k] for k in feature_list if k in features_dict}
        
        # Convert NaN to None for JSON
        features_dict = {k: (None if pd.isna(v) else v) for k, v in features_dict.items()}
        
        # Persist
        try:
            upsert_feature_row(con_feat, cfg.table_name, cfg.feature_key, ts, features_dict)
            processed += 1
            if processed % 100 == 0:
                print(f"  Processed {processed}/{len(targets)}...")
        except Exception as e:
            print(f"  ERROR {ts}: {e}")
    
    con_unified.close()
    con_feat.close()
    
    print(f"\n  Done! Processed: {processed}, Skipped: {skipped}")
    return 0


def parse_args(argv: Optional[List[str]] = None) -> BackfillDerivedFeaturesConfig:
    p = argparse.ArgumentParser(description="Backfill derived features into DuckDB")
    p.add_argument("--unified-db", type=Path, required=True, help="Unified DuckDB path")
    p.add_argument("--feat-db", type=Path, required=True, help="Feature DuckDB path")
    p.add_argument("--feature-key", required=True, help="Feature key identifier")
    p.add_argument("--feature-list", type=Path, default=DEFAULT_FEATURE_LIST, help="Feature list JSON")
    p.add_argument("--mode", choices=["window", "last_from_features", "ts_list"], default="window")
    p.add_argument("--start", default=None, help="Start timestamp")
    p.add_argument("--end", default=None, help="End timestamp")
    p.add_argument("--ts", nargs="*", default=[], help="Timestamps for ts_list mode")
    p.add_argument("--ts-file", type=Path, default=None, help="File with timestamps")
    p.add_argument("--buffer-hours", type=int, default=10000, help="Buffer hours for rolling windows (10000h needed for 720-window EWM features)")
    p.add_argument("--at-most", type=int, default=None, help="Max timestamps to process")
    p.add_argument("--dry-run", action="store_true", help="Plan only")
    p.add_argument("--overwrite", action="store_true", help="Replace existing rows")
    p.add_argument("--for-training", action="store_true", help="Apply ffill before computing (for training)")
    p.add_argument("--table-name", default="derived_features", help="Output table name (default: derived_features)")
    args = p.parse_args(argv)
    
    return BackfillDerivedFeaturesConfig(
        unified_db_path=args.unified_db,
        feat_db_path=args.feat_db,
        table_name=args.table_name,
        feature_key=args.feature_key,
        feature_list_path=args.feature_list,
        mode=args.mode,
        start=args.start,
        end=args.end,
        ts=list(args.ts or []),
        ts_file=args.ts_file,
        buffer_hours=args.buffer_hours,
        at_most=args.at_most,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
        for_training=args.for_training,
    )


def main(argv: Optional[List[str]] = None) -> int:
    cfg = parse_args(argv)
    try:
        return backfill_derived_features(cfg)
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
