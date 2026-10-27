"""
evaluate_historical_stochastic_baseline_50.py
---------------------------------------------
Builds a historical stochastic baseline for the 50-sample probabilistic
comparison.

For each validation/test instance, this baseline samples K historical 24-hour
wind/load windows from Y_train and treats them as scenarios. It then computes
the same empirical CRPS and energy score used for the diffusion scenarios.

Channel order: [..., 0] = wind   [..., 1] = load
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROC = Path("data/processed")
RESULTS = Path("results")
DEFAULT_OUT_CSV = RESULTS / "historical_stochastic_baseline_comparison_50.csv"

DEFAULT_N_SAMPLES = 50
DEFAULT_N_SCENARIOS = 20
DEFAULT_SEED = 2026

METRIC_COLS = [
    "wind_crps_mean",
    "load_crps_mean",
    "joint_energy_score",
]

OUT_COLS = [
    "model",
    "split",
    "wind_crps_mean_mean",
    "wind_crps_mean_std",
    "load_crps_mean_mean",
    "load_crps_mean_std",
    "joint_energy_score_mean",
    "joint_energy_score_std",
]


def empirical_crps_1d(samples: np.ndarray, truth: np.ndarray) -> float:
    """Empirical CRPS for scalar ensemble forecasts over a 24-hour horizon."""
    abs_error = np.abs(samples - truth[None, :]).mean(axis=0)
    pairwise = np.abs(samples[:, None, :] - samples[None, :, :]).mean(axis=(0, 1))
    crps_by_hour = abs_error - 0.5 * pairwise
    return float(crps_by_hour.mean())


def energy_score(samples: np.ndarray, truth: np.ndarray) -> float:
    """Empirical energy score for 24-hour joint wind-load trajectories."""
    samples_flat = samples.reshape(samples.shape[0], -1)
    truth_flat = truth.reshape(-1)

    dist_to_truth = np.linalg.norm(samples_flat - truth_flat[None, :], axis=1).mean()
    pairwise_diff = samples_flat[:, None, :] - samples_flat[None, :, :]
    pairwise_dist = np.linalg.norm(pairwise_diff, axis=2).mean()

    return float(dist_to_truth - 0.5 * pairwise_dist)


def metrics_row(
    split: str,
    sample_id: int,
    scenarios: np.ndarray,
    truth: np.ndarray,
) -> dict:
    wind_crps = empirical_crps_1d(scenarios[:, :, 0], truth[:, 0])
    load_crps = empirical_crps_1d(scenarios[:, :, 1], truth[:, 1])
    joint_es = energy_score(scenarios, truth)

    return {
        "model": "historical_stochastic",
        "split": split,
        "sample_id": sample_id,
        "wind_crps_mean": round(wind_crps, 2),
        "load_crps_mean": round(load_crps, 2),
        "joint_energy_score": round(joint_es, 2),
    }


def evaluate_split(
    split: str,
    y_train: np.ndarray,
    y_eval: np.ndarray,
    rng: np.random.Generator,
    n_samples: int,
    n_scenarios: int,
) -> list[dict]:
    rows = []
    max_samples = min(n_samples, len(y_eval))

    for sample_id in range(max_samples):
        train_ids = rng.integers(0, len(y_train), size=n_scenarios)
        scenarios = y_train[train_ids]
        truth = y_eval[sample_id]
        rows.append(metrics_row(split, sample_id, scenarios, truth))

    return rows


def aggregate_rows(rows: list[dict]) -> pd.DataFrame:
    per_sample = pd.DataFrame(rows)
    summary = (
        per_sample.groupby(["model", "split"])[METRIC_COLS]
        .agg(["mean", "std"])
        .round(2)
    )
    summary.columns = [
        f"{metric}_{stat}" for metric, stat in summary.columns.to_flat_index()
    ]
    summary = summary.reset_index()
    return summary[OUT_COLS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate historical stochastic baseline on the 50-sample subset."
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=DEFAULT_N_SAMPLES,
        help=f"Number of validation and test samples to evaluate (default: {DEFAULT_N_SAMPLES}).",
    )
    parser.add_argument(
        "--n-scenarios",
        type=int,
        default=DEFAULT_N_SCENARIOS,
        help=f"Historical windows sampled per instance (default: {DEFAULT_N_SCENARIOS}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for historical sampling (default: {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=DEFAULT_OUT_CSV,
        help=f"Output CSV path (default: {DEFAULT_OUT_CSV}).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.n_samples <= 0:
        print("ERROR: --n-samples must be positive.")
        return 1
    if args.n_scenarios <= 0:
        print("ERROR: --n-scenarios must be positive.")
        return 1

    required = [
        PROC / "Y_train.npy",
        PROC / "Y_val.npy",
        PROC / "Y_test.npy",
    ]
    for path in required:
        if not path.exists():
            print(f"ERROR: File not found at '{path}'.")
            return 1

    print(f"N_SAMPLES   : {args.n_samples}")
    print(f"N_SCENARIOS : {args.n_scenarios}")
    print(f"Seed        : {args.seed}")
    print(f"OUT_CSV     : {args.out_csv}")

    y_train = np.load(PROC / "Y_train.npy")
    y_val = np.load(PROC / "Y_val.npy")
    y_test = np.load(PROC / "Y_test.npy")

    rng = np.random.default_rng(args.seed)

    rows = []
    rows.extend(
        evaluate_split(
            split="val",
            y_train=y_train,
            y_eval=y_val,
            rng=rng,
            n_samples=args.n_samples,
            n_scenarios=args.n_scenarios,
        )
    )
    rows.extend(
        evaluate_split(
            split="test",
            y_train=y_train,
            y_eval=y_test,
            rng=rng,
            n_samples=args.n_samples,
            n_scenarios=args.n_scenarios,
        )
    )

    summary = aggregate_rows(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out_csv, index=False)

    print("\nHistorical stochastic baseline:")
    print(summary.to_string(index=False))
    print(f"\nSaved: {args.out_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
