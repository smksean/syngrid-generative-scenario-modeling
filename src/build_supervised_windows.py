"""
build_supervised_windows.py
---------------------------
Step 5 of the Syngrid data pipeline.

Slides a fixed window over the cleaned hourly time series to produce
supervised learning arrays ready for model training.

Output shapes
  X             : (N, 48, 5)  — 48-hour look-back, 5 input features
  y_load        : (N, 24)     — next 24 hours of load
  y_wind        : (N, 24)     — next 24 hours of wind
  forecast_timestamps.csv : first timestamp of each 24-hour target window
"""

from pathlib import Path

import numpy as np
import pandas as pd


DATA_PATH = Path("data/processed/opsd_de_wind_load_hourly.csv")
OUT_DIR   = Path("data/processed")

FEATURES  = ["load", "wind", "hour", "day_of_week", "month"]
LOOKBACK  = 48   # hours of history
HORIZON   = 24   # hours to forecast


def sliding_windows(arr: np.ndarray, window: int) -> np.ndarray:
    """Return a (N, window, features) view using stride tricks — no copy."""
    n_windows = arr.shape[0] - window + 1
    shape   = (n_windows, window) + arr.shape[1:]
    strides = (arr.strides[0],)   + arr.strides
    return np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)


def main() -> int:
    if not DATA_PATH.exists():
        print(f"ERROR: File not found at '{DATA_PATH}'.")
        print("Run src/prepare_opsd_subset.py first.")
        return 1

    df = pd.read_csv(DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    n_total   = len(df)
    n_windows = n_total - LOOKBACK - HORIZON + 1

    if n_windows <= 0:
        print("ERROR: Dataset too short for the chosen LOOKBACK + HORIZON.")
        return 1

    feat = df[FEATURES].to_numpy(dtype=np.float32)

    # Build a (n_total - LOOKBACK + 1, LOOKBACK, 5) view, then keep only the
    # rows for which a full HORIZON exists after the window.
    all_windows = sliding_windows(feat, LOOKBACK)      # (n_total-47, 48, 5)
    X = np.array(all_windows[:n_windows])              # (N, 48, 5)  — force copy

    # Target sequences: rows LOOKBACK … end, shaped as (n_total-47, …)
    load_seq = df["load"].to_numpy(dtype=np.float32)
    wind_seq = df["wind"].to_numpy(dtype=np.float32)

    target_starts = sliding_windows(load_seq[LOOKBACK:].reshape(-1, 1), HORIZON)
    y_load = np.array(target_starts[:n_windows, :, 0])  # (N, 24)

    target_wind   = sliding_windows(wind_seq[LOOKBACK:].reshape(-1, 1), HORIZON)
    y_wind = np.array(target_wind[:n_windows, :, 0])     # (N, 24)

    forecast_starts = df["timestamp"].iloc[
        LOOKBACK : LOOKBACK + n_windows
    ].reset_index(drop=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / "X.npy",      X)
    np.save(OUT_DIR / "y_load.npy", y_load)
    np.save(OUT_DIR / "y_wind.npy", y_wind)
    forecast_starts.to_frame(name="forecast_start").to_csv(
        OUT_DIR / "forecast_timestamps.csv", index=False
    )

    print(f"X             shape : {X.shape}")
    print(f"y_load        shape : {y_load.shape}")
    print(f"y_wind        shape : {y_wind.shape}")
    print(f"First forecast      : {forecast_starts.iloc[0]}")
    print(f"Last  forecast      : {forecast_starts.iloc[-1]}")
    print(f"\nSaved to {OUT_DIR}/")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
