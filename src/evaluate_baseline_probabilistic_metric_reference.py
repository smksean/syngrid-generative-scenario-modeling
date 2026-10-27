"""
evaluate_baseline_probabilistic_metric_reference.py
---------------------------------------------------
Step 34 of the Syngrid data pipeline.

Computes deterministic baseline references for the probabilistic metrics used
to evaluate diffusion scenarios.

For a deterministic forecast:
  - CRPS reduces to absolute error, averaged over the 24-hour horizon
  - energy score reduces to Euclidean trajectory error over the 24-hour
    joint wind-load vector
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROC = Path("data/processed")
RESULTS = Path("results")
DEFAULT_OUT_CSV = RESULTS / "baseline_probabilistic_metric_reference_50.csv"

LOAD_IDX = 0
WIND_IDX = 1
HORIZON = 24
DEFAULT_N_SAMPLES = 50

COLS = [
    "model",
    "split",
    "sample_id",
    "horizon",
    "wind_crps_mean",
    "load_crps_mean",
    "joint_energy_score",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute deterministic baseline references for CRPS and energy score."
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=DEFAULT_N_SAMPLES,
        help=f"Number of validation and test samples to evaluate (default: {DEFAULT_N_SAMPLES}).",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=DEFAULT_OUT_CSV,
        help=f"Output CSV path (default: {DEFAULT_OUT_CSV}).",
    )
    return parser.parse_args()


def load_split(tag: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.load(PROC / f"X_{tag}.npy")
    y_load = np.load(PROC / f"y_load_{tag}.npy")
    y_wind = np.load(PROC / f"y_wind_{tag}.npy")
    return X, y_load, y_wind


def add_bias(X: np.ndarray) -> np.ndarray:
    return np.hstack([X, np.ones((len(X), 1), dtype=X.dtype)])


def fit_linear_regression_models(
    X_train: np.ndarray,
    y_load_train: np.ndarray,
    y_wind_train: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    X_train_flat = X_train.reshape(len(X_train), -1)
    X_train_b = add_bias(X_train_flat)

    W_load, _, _, _ = np.linalg.lstsq(X_train_b, y_load_train, rcond=None)
    W_wind, _, _, _ = np.linalg.lstsq(X_train_b, y_wind_train, rcond=None)

    return W_load, W_wind


def predict_linear(
    X: np.ndarray,
    W_load: np.ndarray,
    W_wind: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    X_flat = X.reshape(len(X), -1)
    X_b = add_bias(X_flat)
    yhat_load = X_b @ W_load
    yhat_wind = X_b @ W_wind
    return yhat_load, yhat_wind


def predict_persistence(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    last_load = X[:, -1, LOAD_IDX]
    last_wind = X[:, -1, WIND_IDX]
    yhat_load = np.tile(last_load[:, None], (1, HORIZON))
    yhat_wind = np.tile(last_wind[:, None], (1, HORIZON))
    return yhat_load, yhat_wind


def metrics_row(
    model: str,
    split: str,
    sample_id: int,
    y_wind: np.ndarray,
    y_load: np.ndarray,
    yhat_wind: np.ndarray,
    yhat_load: np.ndarray,
) -> dict:
    wind_crps = float(np.mean(np.abs(yhat_wind - y_wind)))
    load_crps = float(np.mean(np.abs(yhat_load - y_load)))

    true_joint = np.stack([y_wind, y_load], axis=-1)
    pred_joint = np.stack([yhat_wind, yhat_load], axis=-1)
    joint_es = float(np.linalg.norm(pred_joint.reshape(-1) - true_joint.reshape(-1)))

    return {
        "model": model,
        "split": split,
        "sample_id": sample_id,
        "horizon": HORIZON,
        "wind_crps_mean": round(wind_crps, 2),
        "load_crps_mean": round(load_crps, 2),
        "joint_energy_score": round(joint_es, 2),
    }


def rows_for_split(
    model: str,
    split: str,
    y_load: np.ndarray,
    y_wind: np.ndarray,
    yhat_load: np.ndarray,
    yhat_wind: np.ndarray,
    n_samples: int,
) -> list[dict]:
    rows = []
    max_samples = min(n_samples, len(y_load))

    for sample_id in range(max_samples):
        rows.append(
            metrics_row(
                model=model,
                split=split,
                sample_id=sample_id,
                y_wind=y_wind[sample_id],
                y_load=y_load[sample_id],
                yhat_wind=yhat_wind[sample_id],
                yhat_load=yhat_load[sample_id],
            )
        )

    return rows


def print_summary(df: pd.DataFrame) -> None:
    summary = (
        df.groupby(["model", "split"])[
            ["wind_crps_mean", "load_crps_mean", "joint_energy_score"]
        ]
        .agg(["mean", "std"])
        .round(2)
    )

    print(f"\n{'=' * 60}")
    print("Baseline probabilistic metric reference:")
    print(summary.to_string())


def main() -> int:
    args = parse_args()

    if args.n_samples <= 0:
        print("ERROR: --n-samples must be positive.")
        return 1

    required = [
        PROC / "X_train.npy",
        PROC / "y_load_train.npy",
        PROC / "y_wind_train.npy",
        PROC / "X_val.npy",
        PROC / "y_load_val.npy",
        PROC / "y_wind_val.npy",
        PROC / "X_test.npy",
        PROC / "y_load_test.npy",
        PROC / "y_wind_test.npy",
    ]

    for path in required:
        if not path.exists():
            print(f"ERROR: File not found at '{path}'.")
            return 1

    print(f"N_SAMPLES : {args.n_samples}")
    print(f"OUT_CSV   : {args.out_csv}")

    X_train, y_load_train, y_wind_train = load_split("train")
    X_val, y_load_val, y_wind_val = load_split("val")
    X_test, y_load_test, y_wind_test = load_split("test")

    print("Fitting linear regression baselines...")
    W_load, W_wind = fit_linear_regression_models(
        X_train=X_train,
        y_load_train=y_load_train,
        y_wind_train=y_wind_train,
    )

    yhat_load_val_lr, yhat_wind_val_lr = predict_linear(X_val, W_load, W_wind)
    yhat_load_test_lr, yhat_wind_test_lr = predict_linear(X_test, W_load, W_wind)

    yhat_load_val_p, yhat_wind_val_p = predict_persistence(X_val)
    yhat_load_test_p, yhat_wind_test_p = predict_persistence(X_test)

    rows = []
    rows.extend(
        rows_for_split(
            model="persistence",
            split="val",
            y_load=y_load_val,
            y_wind=y_wind_val,
            yhat_load=yhat_load_val_p,
            yhat_wind=yhat_wind_val_p,
            n_samples=args.n_samples,
        )
    )
    rows.extend(
        rows_for_split(
            model="persistence",
            split="test",
            y_load=y_load_test,
            y_wind=y_wind_test,
            yhat_load=yhat_load_test_p,
            yhat_wind=yhat_wind_test_p,
            n_samples=args.n_samples,
        )
    )
    rows.extend(
        rows_for_split(
            model="linear_regression",
            split="val",
            y_load=y_load_val,
            y_wind=y_wind_val,
            yhat_load=yhat_load_val_lr,
            yhat_wind=yhat_wind_val_lr,
            n_samples=args.n_samples,
        )
    )
    rows.extend(
        rows_for_split(
            model="linear_regression",
            split="test",
            y_load=y_load_test,
            y_wind=y_wind_test,
            yhat_load=yhat_load_test_lr,
            yhat_wind=yhat_wind_test_lr,
            n_samples=args.n_samples,
        )
    )

    df = pd.DataFrame(rows, columns=COLS)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)

    print_summary(df)
    print(f"\nSaved: {args.out_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
