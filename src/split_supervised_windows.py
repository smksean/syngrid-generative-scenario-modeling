"""
split_supervised_windows.py
---------------------------
Step 7 of the Syngrid data pipeline.

Assigns each supervised forecast window to train / val / test based on
the chronological split already locked in the processed CSV files.
Assignment is determined solely by the forecast start timestamp.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROC = Path("data/processed")

X_PATH      = PROC / "X.npy"
Y_LOAD_PATH = PROC / "y_load.npy"
Y_WIND_PATH = PROC / "y_wind.npy"
TS_PATH     = PROC / "forecast_timestamps.csv"

TRAIN_CSV = PROC / "train.csv"
VAL_CSV   = PROC / "val.csv"
TEST_CSV  = PROC / "test.csv"


def load_ts_range(path: Path) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return the (min, max) timestamp from a split CSV."""
    df = pd.read_csv(path, usecols=["timestamp"])
    ts = pd.to_datetime(df["timestamp"], utc=True)
    return ts.min(), ts.max()


def save_split(tag: str, mask: np.ndarray,
               X: np.ndarray, y_load: np.ndarray, y_wind: np.ndarray,
               timestamps: pd.DataFrame) -> None:
    np.save(PROC / f"X_{tag}.npy",      X[mask])
    np.save(PROC / f"y_load_{tag}.npy", y_load[mask])
    np.save(PROC / f"y_wind_{tag}.npy", y_wind[mask])
    timestamps[mask].reset_index(drop=True).to_csv(
        PROC / f"forecast_timestamps_{tag}.csv", index=False
    )


def print_split(tag: str, mask: np.ndarray,
                X: np.ndarray, y_load: np.ndarray, y_wind: np.ndarray,
                timestamps: pd.DataFrame) -> None:
    ts = timestamps["forecast_start"][mask]
    print(f"  X_{tag:<5}      : {X[mask].shape}")
    print(f"  y_load_{tag:<5} : {y_load[mask].shape}")
    print(f"  y_wind_{tag:<5} : {y_wind[mask].shape}")
    print(f"  range          : {ts.iloc[0]}  →  {ts.iloc[-1]}")


def main() -> int:
    for p in (X_PATH, Y_LOAD_PATH, Y_WIND_PATH, TS_PATH,
              TRAIN_CSV, VAL_CSV, TEST_CSV):
        if not p.exists():
            print(f"ERROR: File not found at '{p}'.")
            print("Run earlier pipeline steps first.")
            return 1

    X      = np.load(X_PATH)
    y_load = np.load(Y_LOAD_PATH)
    y_wind = np.load(Y_WIND_PATH)

    ts_df = pd.read_csv(TS_PATH)
    ts_df["forecast_start"] = pd.to_datetime(ts_df["forecast_start"], utc=True)
    forecast_ts = ts_df["forecast_start"]

    train_min, train_max = load_ts_range(TRAIN_CSV)
    val_min,   val_max   = load_ts_range(VAL_CSV)
    test_min,  test_max  = load_ts_range(TEST_CSV)

    train_mask = (forecast_ts >= train_min) & (forecast_ts <= train_max)
    val_mask   = (forecast_ts >= val_min)   & (forecast_ts <= val_max)
    test_mask  = (forecast_ts >= test_min)  & (forecast_ts <= test_max)

    train_mask = train_mask.to_numpy()
    val_mask   = val_mask.to_numpy()
    test_mask  = test_mask.to_numpy()

    # Sanity check: every window must belong to exactly one split.
    assigned = train_mask.astype(int) + val_mask.astype(int) + test_mask.astype(int)
    n_unassigned = int((assigned == 0).sum())
    n_overlap    = int((assigned  > 1).sum())

    if n_unassigned > 0:
        print(f"WARNING: {n_unassigned} window(s) were not assigned to any split.")
    if n_overlap > 0:
        print(f"WARNING: {n_overlap} window(s) overlap more than one split.")
    if n_unassigned == 0 and n_overlap == 0:
        print(f"Check passed: all {len(assigned):,} windows assigned to exactly one split.\n")

    for tag, mask in (("train", train_mask), ("val", val_mask), ("test", test_mask)):
        save_split(tag, mask, X, y_load, y_wind, ts_df)

    print(f"{'='*60}")
    print("Train split:")
    print_split("train", train_mask, X, y_load, y_wind, ts_df)
    print(f"\nValidation split:")
    print_split("val",   val_mask,   X, y_load, y_wind, ts_df)
    print(f"\nTest split:")
    print_split("test",  test_mask,  X, y_load, y_wind, ts_df)
    print(f"{'='*60}")
    print(f"\nSaved all splits to {PROC}/")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
