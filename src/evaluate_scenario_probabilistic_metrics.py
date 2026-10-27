"""
evaluate_scenario_probabilistic_metrics.py
------------------------------------------
Step 31 of the Syngrid data pipeline.

Computes first probabilistic scenario-quality metrics for validation sample 0
and test sample 0 using the best normalized diffusion checkpoint outputs.

Metrics:
  - empirical CRPS for wind and load, averaged over the 24-hour horizon
  - multivariate energy score for the full 24-hour joint wind-load trajectory
"""

from pathlib import Path

import numpy as np
import pandas as pd


PROC = Path("data/processed")
RESULTS = Path("results")
OUT_CSV = RESULTS / "scenario_probabilistic_metrics_sample0.csv"

SAMPLE_ID = 0

COLS = [
    "split",
    "sample_id",
    "n_scenarios",
    "horizon",
    "wind_crps_mean",
    "load_crps_mean",
    "joint_energy_score",
]


def empirical_crps_1d(samples: np.ndarray, truth: np.ndarray) -> float:
    """
    Empirical CRPS for scalar ensemble forecasts.

    Parameters
    ----------
    samples:
        Scenario values with shape (n_scenarios, horizon).
    truth:
        Observed values with shape (horizon,).
    """
    abs_error = np.abs(samples - truth[None, :]).mean(axis=0)
    pairwise = np.abs(samples[:, None, :] - samples[None, :, :]).mean(axis=(0, 1))
    crps_by_hour = abs_error - 0.5 * pairwise
    return float(crps_by_hour.mean())


def energy_score(samples: np.ndarray, truth: np.ndarray) -> float:
    """
    Empirical energy score for vector-valued ensemble forecasts.

    Each scenario is treated as one 24-hour joint wind-load trajectory.
    """
    samples_flat = samples.reshape(samples.shape[0], -1)
    truth_flat = truth.reshape(-1)

    dist_to_truth = np.linalg.norm(samples_flat - truth_flat[None, :], axis=1).mean()

    pairwise_diff = samples_flat[:, None, :] - samples_flat[None, :, :]
    pairwise_dist = np.linalg.norm(pairwise_diff, axis=2).mean()

    return float(dist_to_truth - 0.5 * pairwise_dist)


def denormalize_truth(split: str) -> np.ndarray:
    y_norm = np.load(PROC / f"Y_{split}_norm.npy")
    stats = np.load(PROC / "diffusion_normalization_stats.npz")
    y_mean = stats["y_mean"].astype(np.float32)
    y_std = stats["y_std"].astype(np.float32)
    return y_norm[SAMPLE_ID] * y_std + y_mean


def compute_row(split: str, scenario_path: Path) -> dict:
    scenarios = np.load(scenario_path)  # (n_scenarios, 24, 2), denormalised
    truth = denormalize_truth(split)    # (24, 2), denormalised

    wind_crps = empirical_crps_1d(scenarios[:, :, 0], truth[:, 0])
    load_crps = empirical_crps_1d(scenarios[:, :, 1], truth[:, 1])
    joint_es = energy_score(scenarios, truth)

    return {
        "split": split,
        "sample_id": SAMPLE_ID,
        "n_scenarios": int(scenarios.shape[0]),
        "horizon": int(scenarios.shape[1]),
        "wind_crps_mean": round(wind_crps, 2),
        "load_crps_mean": round(load_crps, 2),
        "joint_energy_score": round(joint_es, 2),
    }


def main() -> int:
    required = [
        RESULTS / "Y_val_sample0_scenarios_best_denorm.npy",
        RESULTS / "Y_test_sample0_scenarios_best_denorm.npy",
        PROC / "Y_val_norm.npy",
        PROC / "Y_test_norm.npy",
        PROC / "diffusion_normalization_stats.npz",
    ]

    for path in required:
        if not path.exists():
            print(f"ERROR: File not found at '{path}'.")
            return 1

    rows = [
        compute_row("val", RESULTS / "Y_val_sample0_scenarios_best_denorm.npy"),
        compute_row("test", RESULTS / "Y_test_sample0_scenarios_best_denorm.npy"),
    ]

    df = pd.DataFrame(rows, columns=COLS)

    RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    print(df.T.to_string(header=False))
    print(f"\nSaved: {OUT_CSV}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
