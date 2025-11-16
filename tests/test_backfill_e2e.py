#!/usr/bin/env python3
"""
End-to-end test for backfill_features.py using production databases.

Tests backfilling features for a small date range and verifies the results.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import duckdb
import pandas as pd


def run_e2e_test():
    """Run end-to-end test for feature backfilling."""

    # Test configuration
    OHLCV_DB = "/Volumes/Extreme SSD/trading_data/cex/db/binance_btcusdt_perp_ohlcv.duckdb"
    FEATURE_DB = "/Volumes/Extreme SSD/trading_data/cex/db/binance_btcusdt_perp_feature.duckdb"
    TABLE = "ohlcv_btcusdt_1h"
    FEATURE_KEY = "e2e_test_backfill"
    FEATURE_LIST = "/Users/noel/projects/trading_cex/configs/feature_lists/binance_btcusdt_p60_default.json"
    START = "2025-11-13 00:00:00"
    END = "2025-11-14 00:00:00"
    BASE_HOURS = 720

    print("=" * 80)
    print("E2E Test: backfill_features.py")
    print("=" * 80)
    print(f"OHLCV DB:     {OHLCV_DB}")
    print(f"Feature DB:   {FEATURE_DB}")
    print(f"Table:        {TABLE}")
    print(f"Feature key:  {FEATURE_KEY}")
    print(f"Feature list: {FEATURE_LIST}")
    print(f"Date range:   {START} to {END}")
    print(f"Base hours:   {BASE_HOURS}")
    print()

    # Step 1: Check OHLCV data availability
    print("Step 1: Checking OHLCV data availability...")
    try:
        con_ohlcv = duckdb.connect(OHLCV_DB, read_only=True)
        con_ohlcv.execute("SET TimeZone='UTC';")

        # Check if we have data for the target range
        result = con_ohlcv.execute(f"""
            SELECT COUNT(*), MIN(timestamp), MAX(timestamp)
            FROM {TABLE}
            WHERE timestamp BETWEEN ? AND ?
        """, [START, END]).fetchone()

        if result[0] == 0:
            print(f"❌ FAILED: No OHLCV data found in range {START} to {END}")
            con_ohlcv.close()
            return False

        print(f"✓ Found {result[0]} OHLCV rows from {result[1]} to {result[2]}")

        # Check if we have sufficient history for base_hours
        earliest_needed = pd.to_datetime(START) - pd.Timedelta(hours=BASE_HOURS - 1)
        result = con_ohlcv.execute(f"""
            SELECT MIN(timestamp), MAX(timestamp)
            FROM {TABLE}
        """).fetchone()

        if pd.to_datetime(result[0]) > earliest_needed:
            print(f"⚠️  WARNING: Earliest OHLCV data is {result[0]}, but need {earliest_needed} for {BASE_HOURS}h lookback")
            print(f"   Some features may have NaN values or script may skip early timestamps")
        else:
            print(f"✓ Sufficient OHLCV history available (earliest: {result[0]})")

        con_ohlcv.close()
    except Exception as e:
        print(f"❌ FAILED: Error checking OHLCV data: {e}")
        return False

    print()

    # Step 2: Clean up any existing test features
    print("Step 2: Cleaning up any existing test features...")
    try:
        con_feat = duckdb.connect(FEATURE_DB)
        con_feat.execute("SET TimeZone='UTC';")

        # Check if features table exists
        table_exists = con_feat.execute("""
            SELECT 1 FROM information_schema.tables
            WHERE table_name = 'features' LIMIT 1
        """).fetchone()

        if table_exists:
            count_before = con_feat.execute(
                "SELECT COUNT(*) FROM features WHERE feature_key = ?",
                [FEATURE_KEY]
            ).fetchone()[0]

            if count_before > 0:
                con_feat.execute(
                    "DELETE FROM features WHERE feature_key = ?",
                    [FEATURE_KEY]
                )
                print(f"✓ Deleted {count_before} existing test features")
            else:
                print("✓ No existing test features to clean up")
        else:
            print("✓ Features table doesn't exist yet (will be created)")

        con_feat.close()
    except Exception as e:
        print(f"❌ FAILED: Error cleaning up test features: {e}")
        return False

    print()

    # Step 3: Run backfill_features.py
    print("Step 3: Running backfill_features.py...")
    print()

    import subprocess
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "backfill_features.py"),
        "--duckdb", OHLCV_DB,
        "--feat-duckdb", FEATURE_DB,
        "--table", TABLE,
        "--feature-key", FEATURE_KEY,
        "--mode", "window",
        "--start", START,
        "--end", END,
        "--base-hours", str(BASE_HOURS),
        "--feature-list", FEATURE_LIST,
    ]

    print("Command:", " ".join(cmd))
    print()

    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    if result.returncode != 0:
        print(f"❌ FAILED: backfill_features.py exited with code {result.returncode}")
        return False

    print()

    # Step 4: Verify features were written
    print("Step 4: Verifying features were written...")
    try:
        con_feat = duckdb.connect(FEATURE_DB, read_only=True)
        con_feat.execute("SET TimeZone='UTC';")

        # Count features written
        result = con_feat.execute("""
            SELECT COUNT(*), MIN(ts), MAX(ts)
            FROM features
            WHERE feature_key = ?
        """, [FEATURE_KEY]).fetchone()

        if result[0] == 0:
            print(f"❌ FAILED: No features were written for key={FEATURE_KEY}")
            con_feat.close()
            return False

        print(f"✓ Found {result[0]} feature rows from {result[1]} to {result[2]}")

        # Sample one feature row to check structure
        sample = con_feat.execute("""
            SELECT ts, features
            FROM features
            WHERE feature_key = ?
            ORDER BY ts
            LIMIT 1
        """, [FEATURE_KEY]).fetchone()

        if sample:
            ts, features_json = sample
            features_dict = json.loads(features_json) if isinstance(features_json, str) else features_json
            print(f"✓ Sample timestamp: {ts}")
            print(f"✓ Features per row: {len(features_dict)}")
            print(f"✓ Sample feature names: {list(features_dict.keys())[:5]}")

            # Check for NaN values
            nan_count = sum(1 for v in features_dict.values() if v is None or (isinstance(v, float) and pd.isna(v)))
            if nan_count > 0:
                print(f"⚠️  WARNING: {nan_count}/{len(features_dict)} features have NaN values")
            else:
                print(f"✓ All features have non-NaN values")

        con_feat.close()
    except Exception as e:
        print(f"❌ FAILED: Error verifying features: {e}")
        return False

    print()

    # Step 5: Compare with reference features (if they exist)
    print("Step 5: Comparing with reference features (if available)...")
    try:
        con_feat = duckdb.connect(FEATURE_DB, read_only=True)
        con_feat.execute("SET TimeZone='UTC';")

        # Check what other feature keys exist for the same timestamps
        ref_keys = con_feat.execute("""
            SELECT DISTINCT feature_key
            FROM features
            WHERE ts BETWEEN ? AND ?
            AND feature_key != ?
            ORDER BY feature_key
        """, [START, END, FEATURE_KEY]).fetchall()

        if ref_keys:
            print(f"✓ Found {len(ref_keys)} other feature keys in same time range:")
            for (key,) in ref_keys[:5]:
                print(f"  - {key}")

            # Compare feature count with first reference key
            ref_key = ref_keys[0][0]
            ref_sample = con_feat.execute("""
                SELECT features
                FROM features
                WHERE feature_key = ?
                AND ts = ?
                LIMIT 1
            """, [ref_key, START]).fetchone()

            if ref_sample:
                ref_features = json.loads(ref_sample[0]) if isinstance(ref_sample[0], str) else ref_sample[0]
                test_sample = con_feat.execute("""
                    SELECT features
                    FROM features
                    WHERE feature_key = ?
                    AND ts = ?
                    LIMIT 1
                """, [FEATURE_KEY, START]).fetchone()

                if test_sample:
                    test_features = json.loads(test_sample[0]) if isinstance(test_sample[0], str) else test_sample[0]

                    if len(test_features) == len(ref_features):
                        print(f"✓ Feature count matches reference ({len(test_features)} features)")
                    else:
                        print(f"⚠️  WARNING: Feature count mismatch - test:{len(test_features)} vs ref:{len(ref_features)}")

                    # Check feature name overlap
                    test_keys = set(test_features.keys())
                    ref_keys_set = set(ref_features.keys())
                    overlap = len(test_keys & ref_keys_set)
                    print(f"✓ Feature name overlap: {overlap}/{len(ref_keys_set)} ({100*overlap/len(ref_keys_set):.1f}%)")

                    # Compare actual feature values
                    print("\nℹ️  Comparing feature values...")
                    value_mismatches = []
                    max_abs_diff = 0.0
                    max_rel_diff = 0.0

                    for feat_name in sorted(test_keys & ref_keys_set):
                        test_val = test_features[feat_name]
                        ref_val = ref_features[feat_name]

                        # Handle NaN/None
                        if (test_val is None or (isinstance(test_val, float) and pd.isna(test_val))) and \
                           (ref_val is None or (isinstance(ref_val, float) and pd.isna(ref_val))):
                            continue

                        if (test_val is None or (isinstance(test_val, float) and pd.isna(test_val))) != \
                           (ref_val is None or (isinstance(ref_val, float) and pd.isna(ref_val))):
                            value_mismatches.append(f"{feat_name}: test={test_val} vs ref={ref_val}")
                            continue

                        # Compare numeric values
                        try:
                            diff = abs(float(test_val) - float(ref_val))
                            max_abs_diff = max(max_abs_diff, diff)

                            if abs(float(ref_val)) > 1e-10:
                                rel_diff = diff / abs(float(ref_val))
                                max_rel_diff = max(max_rel_diff, rel_diff)

                                # Flag significant differences (>0.1% relative or >1e-6 absolute)
                                if rel_diff > 0.001 or diff > 1e-6:
                                    value_mismatches.append(f"{feat_name}: test={test_val:.8f} ref={ref_val:.8f} (rel_diff={rel_diff:.2e})")
                        except (TypeError, ValueError):
                            if test_val != ref_val:
                                value_mismatches.append(f"{feat_name}: test={test_val} vs ref={ref_val}")

                    if value_mismatches:
                        print(f"⚠️  WARNING: {len(value_mismatches)} features have value differences:")
                        for msg in value_mismatches[:10]:
                            print(f"    {msg}")
                        if len(value_mismatches) > 10:
                            print(f"    ... and {len(value_mismatches) - 10} more")
                    else:
                        print(f"✓ All feature values match (max_abs_diff={max_abs_diff:.2e}, max_rel_diff={max_rel_diff:.2e})")
        else:
            print("ℹ️  No reference features found in same time range")

        con_feat.close()
    except Exception as e:
        print(f"⚠️  WARNING: Error comparing with reference: {e}")

    print()

    # Step 6: Cleanup
    print("Step 6: Cleanup...")
    response = input(f"Delete test features with key='{FEATURE_KEY}'? [y/N]: ")
    if response.lower() == 'y':
        try:
            con_feat = duckdb.connect(FEATURE_DB)
            con_feat.execute("SET TimeZone='UTC';")
            deleted = con_feat.execute(
                "DELETE FROM features WHERE feature_key = ?",
                [FEATURE_KEY]
            )
            con_feat.close()
            print(f"✓ Deleted test features")
        except Exception as e:
            print(f"⚠️  WARNING: Error deleting test features: {e}")
    else:
        print(f"ℹ️  Test features kept (key='{FEATURE_KEY}')")

    print()
    print("=" * 80)
    print("✓ E2E Test PASSED")
    print("=" * 80)
    return True


if __name__ == "__main__":
    try:
        success = run_e2e_test()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n❌ FAILED: Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
