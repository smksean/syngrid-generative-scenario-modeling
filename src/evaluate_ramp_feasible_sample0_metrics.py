"""
evaluate_ramp_feasible_sample0_metrics.py
-----------------------------------------
Step 38 of the Syngrid data pipeline.

Compares probabilistic metrics before and after ramp-rate feasibility
correction for validation/test sample 0.
"""

from pathlib import Path

import numpy as np
import pandas as pd


PROC = Path("data/processed")
RESULTS = Path("results")
OUT_CSV = RESULTS / "ramp_feasible_sample0_probabilistic_metrics.csv"

SAMPLE_ID = 0

SCENARIO_FILES = [
    ("val", "raw", RESULTS / "Y_val_sample0_scenarios_best_denorm.npy"),
    ("val", "ramp_feasible", RESULTS / "Y_val_sample0_scenarios_best_ramp_feasible.npy"),
    ("test", "raw", RESULTS / "Y_test_sample0_scenarios_best_denorm.npy"),
    ("test", "ramp_feasible", RESULTS / "Y_test_sample0_scenarios_best_ramp_feasible.npy"),
]

COLS = [
    "split",
    "scenario_type",
    "sample_id",
    "n_scenarios",
    "horizon",
    "wind_crps_mean",
    "load_crps_mean",
    "joint_energy_score",
]


def empirical_crps_1d(samples: np.ndarray, truth: np.ndarray) -> float:
    abs_error = np.abs(samples - truth[None, :]).mean(axis=0)
    pairwise = np.abs(samples[:, None, :] - samples[None, :, :]).mean(axis=(0, 1))
    return float((abs_error - 0.5 * pairwise).mean())


def energy_score(samples: np.ndarray, truth: np.ndarray) -> float:
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


def compute_row(split: str, scenario_type: str, scenario_path: Path) -> dict:
    scenarios = np.load(scenario_path)
    truth = denormalize_truth(split)

    return {
        "split": split,
        "scenario_type": scenario_type,
        "sample_id": SAMPLE_ID,
        "n_scenarios": int(scenarios.shape[0]),
        "horizon": int(scenarios.shape[1]),
        "wind_crps_mean": round(empirical_crps_1d(scenarios[:, :, 0], truth[:, 0]), 2),
        "load_crps_mean": round(empirical_crps_1d(scenarios[:, :, 1], truth[:, 1]), 2),
        "joint_energy_score": round(energy_score(scenarios, truth), 2),
    }


def main() -> int:
    required = [
        PROC / "Y_val_norm.npy",
        PROC / "Y_test_norm.npy",
        PROC / "diffusion_normalization_stats.npz",
        *[path for _, _, path in SCENARIO_FILES],
    ]

    for path in required:
        if not path.exists():
            print(f"ERROR: File not found at '{path}'.")
            return 1

    rows = [
        compute_row(split, scenario_type, scenario_path)
        for split, scenario_type, scenario_path in SCENARIO_FILES
    ]
    df = pd.DataFrame(rows, columns=COLS)

    RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    print(df.to_string(index=False))
    print(f"\nSaved: {OUT_CSV}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
