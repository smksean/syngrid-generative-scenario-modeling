"""
baseline_linear_regression.py
------------------------------
Step 8 of the Syngrid data pipeline.

Trains two multi-output linear regression models (one for load, one for wind)
on the flattened 48-hour look-back windows and evaluates them on the
validation and test splits.

Input shape per sample : (48, 5)  → flattened to 240 features
Output shape per sample: (24,)    — one value per forecast hour
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROC       = Path("data/processed")
RESULTS    = Path("results")
OUT_CSV    = RESULTS / "baseline_linear_regression_metrics.csv"

HORIZON = 24


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_split(tag: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    X      = np.load(PROC / f"X_{tag}.npy")
    y_load = np.load(PROC / f"y_load_{tag}.npy")
    y_wind = np.load(PROC / f"y_wind_{tag}.npy")
    ts     = pd.read_csv(PROC / f"forecast_timestamps_{tag}.csv",
                         parse_dates=["forecast_start"])
    return X, y_load, y_wind, ts


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def metrics_row(model: str, split: str,
                y_load: np.ndarray, yhat_load: np.ndarray,
                y_wind: np.ndarray, yhat_wind: np.ndarray,
                ts: pd.DataFrame) -> dict:
    return {
        "model":                    model,
        "split":                    split,
        "load_mae":                 round(mae(y_load, yhat_load),  4),
        "load_rmse":                round(rmse(y_load, yhat_load), 4),
        "wind_mae":                 round(mae(y_wind, yhat_wind),  4),
        "wind_rmse":                round(rmse(y_wind, yhat_wind), 4),
        "n_samples":                len(y_load),
        "horizon":                  HORIZON,
        "first_forecast_timestamp": ts["forecast_start"].iloc[0],
        "last_forecast_timestamp":  ts["forecast_start"].iloc[-1],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    for tag in ("train", "val", "test"):
        for stem in (f"X_{tag}", f"y_load_{tag}", f"y_wind_{tag}",
                     f"forecast_timestamps_{tag}"):
            p = PROC / (stem + (".csv" if "timestamp" in stem else ".npy"))
            if not p.exists():
                print(f"ERROR: File not found at '{p}'.")
                print("Run src/split_supervised_windows.py first.")
                return 1

    X_train, y_load_train, y_wind_train, _        = load_split("train")
    X_val,   y_load_val,   y_wind_val,   ts_val   = load_split("val")
    X_test,  y_load_test,  y_wind_test,  ts_test  = load_split("test")

    # Flatten (N, 48, 5) → (N, 240)
    X_train_f = X_train.reshape(len(X_train), -1)
    X_val_f   = X_val.reshape(len(X_val),   -1)
    X_test_f  = X_test.reshape(len(X_test),  -1)

    print(f"X_train flattened : {X_train_f.shape}")
    print(f"X_val   flattened : {X_val_f.shape}")
    print(f"X_test  flattened : {X_test_f.shape}\n")

    # Add a bias column to the flattened feature matrix.
    def add_bias(X: np.ndarray) -> np.ndarray:
        return np.hstack([X, np.ones((len(X), 1), dtype=X.dtype)])

    X_train_b = add_bias(X_train_f)
    X_val_b   = add_bias(X_val_f)
    X_test_b  = add_bias(X_test_f)

    # Ordinary least squares via numpy — equivalent to sklearn LinearRegression.
    print("Fitting load model  (OLS via numpy lstsq)...")
    W_load, _, _, _ = np.linalg.lstsq(X_train_b, y_load_train, rcond=None)

    print("Fitting wind model  (OLS via numpy lstsq)...")
    W_wind, _, _, _ = np.linalg.lstsq(X_train_b, y_wind_train, rcond=None)

    # Predict
    yhat_load_val  = X_val_b  @ W_load
    yhat_wind_val  = X_val_b  @ W_wind
    yhat_load_test = X_test_b @ W_load
    yhat_wind_test = X_test_b @ W_wind

    print(f"\nVal  yhat_load shape : {yhat_load_val.shape}")
    print(f"Val  yhat_wind shape : {yhat_wind_val.shape}")
    print(f"Test yhat_load shape : {yhat_load_test.shape}")
    print(f"Test yhat_wind shape : {yhat_wind_test.shape}")

    # Metrics
    print(f"\n{'='*60}")
    print("Validation metrics:")
    print(f"  Load  MAE  : {mae(y_load_val,  yhat_load_val):>10,.2f} MW")
    print(f"  Load  RMSE : {rmse(y_load_val, yhat_load_val):>10,.2f} MW")
    print(f"  Wind  MAE  : {mae(y_wind_val,  yhat_wind_val):>10,.2f} MW")
    print(f"  Wind  RMSE : {rmse(y_wind_val, yhat_wind_val):>10,.2f} MW")
    print(f"  Range      : {ts_val['forecast_start'].iloc[0]}  →  {ts_val['forecast_start'].iloc[-1]}")

    print(f"\nTest metrics:")
    print(f"  Load  MAE  : {mae(y_load_test,  yhat_load_test):>10,.2f} MW")
    print(f"  Load  RMSE : {rmse(y_load_test, yhat_load_test):>10,.2f} MW")
    print(f"  Wind  MAE  : {mae(y_wind_test,  yhat_wind_test):>10,.2f} MW")
    print(f"  Wind  RMSE : {rmse(y_wind_test, yhat_wind_test):>10,.2f} MW")
    print(f"  Range      : {ts_test['forecast_start'].iloc[0]}  →  {ts_test['forecast_start'].iloc[-1]}")
    print(f"{'='*60}")

    # Save
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows = [
        metrics_row("linear_regression", "val",
                    y_load_val,  yhat_load_val,
                    y_wind_val,  yhat_wind_val,  ts_val),
        metrics_row("linear_regression", "test",
                    y_load_test, yhat_load_test,
                    y_wind_test, yhat_wind_test, ts_test),
    ]
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"\nMetrics saved: {OUT_CSV}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
