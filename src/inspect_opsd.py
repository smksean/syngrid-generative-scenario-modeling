"""
inspect_opsd.py
---------------
Step 1 of the Syngrid data pipeline.

Loads the Open Power System Data hourly time series CSV and prints
basic structural information: shape, column names, timestamp column,
sample rows, and candidate columns for wind generation and load/demand.
"""

import sys
import pandas as pd

DATA_PATH = "data/raw/time_series_60min_singleindex.csv"

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
print("Loading dataset...")
try:
    df = pd.read_csv(DATA_PATH, low_memory=False)
except FileNotFoundError:
    print(f"ERROR: File not found at '{DATA_PATH}'.")
    print("Make sure you run this script from the project root (Syngrid/).")
    sys.exit(1)
except Exception as e:
    print(f"ERROR: Failed to load CSV — {e}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print(f"Dataset shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

# ---------------------------------------------------------------------------
# First 30 column names
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print("First 30 column names:")
for i, col in enumerate(df.columns[:30], 1):
    print(f"  {i:>2}. {col}")

# ---------------------------------------------------------------------------
# Timestamp column detection
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print("Timestamp column detection:")

timestamp_col = None
for col in df.columns:
    if col.lower() in ("utc_timestamp", "timestamp", "date", "datetime", "time"):
        timestamp_col = col
        break

if timestamp_col:
    print(f"  Found timestamp column: '{timestamp_col}'")
    try:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True)
        print(f"  Parsed range : {df[timestamp_col].min()}  →  {df[timestamp_col].max()}")
        print(f"  Total periods: {df[timestamp_col].nunique():,}")
    except Exception as e:
        print(f"  WARNING: Could not parse as datetime — {e}")
else:
    print("  No obvious timestamp column found. Inspect column names manually.")

# ---------------------------------------------------------------------------
# First 5 rows (limited columns for readability)
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print("First 5 rows (first 6 columns):")
print(df.iloc[:5, :6].to_string(index=False))

# ---------------------------------------------------------------------------
# Candidate columns — wind generation
# ---------------------------------------------------------------------------
wind_keywords = ["wind", "wind_onshore", "wind_offshore", "entsoe_wind"]
wind_candidates = [
    col for col in df.columns
    if any(kw in col.lower() for kw in wind_keywords)
]

print(f"\n{'='*60}")
print(f"Wind generation candidate columns ({len(wind_candidates)} found):")
if wind_candidates:
    for col in wind_candidates:
        non_null = df[col].notna().sum()
        print(f"  • {col:<60}  ({non_null:,} non-null values)")
else:
    print("  None found — try broadening the keyword search.")

# ---------------------------------------------------------------------------
# Candidate columns — electricity load / demand
# ---------------------------------------------------------------------------
load_keywords = ["load", "demand", "consumption", "actual_load", "total_load"]
load_candidates = [
    col for col in df.columns
    if any(kw in col.lower() for kw in load_keywords)
]

print(f"\n{'='*60}")
print(f"Electricity load/demand candidate columns ({len(load_candidates)} found):")
if load_candidates:
    for col in load_candidates:
        non_null = df[col].notna().sum()
        print(f"  • {col:<60}  ({non_null:,} non-null values)")
else:
    print("  None found — try broadening the keyword search.")

print(f"\n{'='*60}")
print("Inspection complete.")
