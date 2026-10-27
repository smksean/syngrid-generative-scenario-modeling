"""
baseline_persistence.py
-----------------------
Step 6 of the Syngrid data pipeline.

Naive persistence baseline for the 24-hour load and wind forecasting task.
Each forecast simply repeats the last observed value from the 48-hour
look-back window across all 24 forecast steps.

Feature index inside X:
  0 = load  |  1 = wind  |  2 = hour  |  3 = day_of_week  |  4 = month
"""

from pathlib import Path

import numpy as np
import pandas as pd


X_PATH          = Path("data/processed/X.npy")
Y_LOAD_PATH     = Path("data/processed/y_load.npy")
Y_WIND_PATH     = Path("data/processed/y_wind.npy")
TS_PATH         = Path("data/processed/forecast_timestamps.csv")
OUT_DIR         = Path("results")
OUT_CSV         = OUT_DIR / "baseline_persistence_metrics.csv"

LOAD_IDX = 0
WIND_IDX = 1
HORIZON  = 24


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def main() -> int:
    for p in (X_PATH, Y_LOAD_PATH, Y_WIND_PATH, TS_PATH):
        if not p.exists():
            print(f"ERROR: File not found at '{p}'.")
            print("Run src/build_supervised_windows.py first.")
            return 1

    X      = np.load(X_PATH)        # (N, 48, 5)
    y_load = np.load(Y_LOAD_PATH)   # (N, 24)
    y_wind = np.load(Y_WIND_PATH)   # (N, 24)
    timestamps = pd.read_csv(TS_PATH, parse_dates=["forecast_start"])

    # Last observed value in the look-back window: X[:, -1, col]
    last_load = X[:, -1, LOAD_IDX]  # (N,)
    last_wind = X[:, -1, WIND_IDX]  # (N,)

    # Repeat across all horizon steps → (N, 24)
    yhat_load = np.tile(last_load[:, None], (1, HORIZON))
    yhat_wind = np.tile(last_wind[:, None], (1, HORIZON))

    load_mae  = mae(y_load,  yhat_load)
    load_rmse = rmse(y_load, yhat_load)
    wind_mae  = mae(y_wind,  yhat_wind)
    wind_rmse = rmse(y_wind, yhat_wind)

    first_ts = timestamps["forecast_start"].iloc[0]
    last_ts  = timestamps["forecast_start"].iloc[-1]

    print(f"yhat_load shape       : {yhat_load.shape}")
    print(f"yhat_wind shape       : {yhat_wind.shape}")
    print(f"\nLoad  MAE             : {load_mae:>10,.2f} MW")
    print(f"Load  RMSE            : {load_rmse:>10,.2f} MW")
    print(f"\nWind  MAE             : {wind_mae:>10,.2f} MW")
    print(f"Wind  RMSE            : {wind_rmse:>10,.2f} MW")
    print(f"\nFirst forecast        : {first_ts}")
    print(f"Last  forecast        : {last_ts}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame([{
        "model":                    "persistence",
        "load_mae":                 round(load_mae,  4),
        "load_rmse":                round(load_rmse, 4),
        "wind_mae":                 round(wind_mae,  4),
        "wind_rmse":                round(wind_rmse, 4),
        "n_samples":                len(y_load),
        "horizon":                  HORIZON,
        "first_forecast_timestamp": first_ts,
        "last_forecast_timestamp":  last_ts,
    }])
    metrics.to_csv(OUT_CSV, index=False)
    print(f"\nMetrics saved: {OUT_CSV}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
