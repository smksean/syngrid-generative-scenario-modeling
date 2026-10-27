"""
prepare_opsd_subset.py
----------------------
Step 2 of the Syngrid data pipeline.

Loads a small subset of the Open Power System Data hourly time series dataset
for Germany (load + wind), cleans it, adds simple calendar features, and saves
to data/processed/.
"""

from pathlib import Path

import pandas as pd


RAW_PATH = Path("data/raw/time_series_60min_singleindex.csv")
OUT_PATH = Path("data/processed/opsd_de_wind_load_hourly.csv")
TRAIN_PATH = Path("data/processed/train.csv")
VAL_PATH = Path("data/processed/val.csv")
TEST_PATH = Path("data/processed/test.csv")

USECOLS = [
    "utc_timestamp",
    "DE_load_actual_entsoe_transparency",
    "DE_wind_generation_actual",
]

RENAME_MAP = {
    "utc_timestamp": "timestamp",
    "DE_load_actual_entsoe_transparency": "load",
    "DE_wind_generation_actual": "wind",
}


def main() -> int:
    print("Loading dataset (subset of columns)...")
    try:
        df = pd.read_csv(
            RAW_PATH,
            usecols=USECOLS,
            parse_dates=["utc_timestamp"],
            date_format="ISO8601",
        )
    except FileNotFoundError:
        print(f"ERROR: File not found at '{RAW_PATH}'.")
        print("Make sure you run this script from the project root (Syngrid/).")
        return 1
    except ValueError as e:
        # Typically raised if usecols don't exist.
        print(f"ERROR: Failed to load requested columns — {e}")
        return 1
    except Exception as e:
        print(f"ERROR: Failed to load CSV — {e}")
        return 1

    # Ensure timezone-aware UTC timestamps.
    df["utc_timestamp"] = pd.to_datetime(df["utc_timestamp"], utc=True, errors="coerce")

    df = df.rename(columns=RENAME_MAP).sort_values("timestamp", kind="mergesort")

    print(f"\n{'='*60}")
    print(f"Initial shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Date range : {df['timestamp'].min()}  →  {df['timestamp'].max()}")
    print("\nMissing values per column:")
    print(df.isna().sum().to_string())

    df = df.dropna(subset=["load", "wind"])

    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month

    df = df[
        [
            "timestamp",
            "load",
            "wind",
            "hour",
            "day_of_week",
            "month",
        ]
    ].reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    train_df.to_csv(TRAIN_PATH, index=False)
    val_df.to_csv(VAL_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)

    print(f"\n{'='*60}")
    print(f"Final shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Final range: {df['timestamp'].min()}  →  {df['timestamp'].max()}")

    print(f"\n{'='*60}")
    print("Chronological splits (by row):")
    print(f"Train shape: {train_df.shape[0]:,} rows × {train_df.shape[1]} columns")
    print(f"Train range: {train_df['timestamp'].min()}  →  {train_df['timestamp'].max()}")
    print(f"\nValidation shape: {val_df.shape[0]:,} rows × {val_df.shape[1]} columns")
    print(f"Validation range: {val_df['timestamp'].min()}  →  {val_df['timestamp'].max()}")
    print(f"\nTest shape: {test_df.shape[0]:,} rows × {test_df.shape[1]} columns")
    print(f"Test range: {test_df['timestamp'].min()}  →  {test_df['timestamp'].max()}")

    print("\nFirst 5 rows:")
    print(df.head(5).to_string(index=False))
    print(f"\nSaved: {OUT_PATH}")
    print(f"Saved: {TRAIN_PATH}")
    print(f"Saved: {VAL_PATH}")
    print(f"Saved: {TEST_PATH}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
