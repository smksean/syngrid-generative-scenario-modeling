"""
apply_ramp_rate_feasibility_sample0.py
--------------------------------------
Step 37 of the Syngrid data pipeline.

Applies a simple second-stage feasibility correction to the best sample-0
diffusion scenarios:
  1. clip wind/load to non-negative values
  2. clip hourly ramps using 99th-percentile absolute ramps from Y_train

Channel order: [..., 0] = wind   [..., 1] = load
"""

from pathlib import Path

import numpy as np
import pandas as pd


PROC = Path("data/processed")
RESULTS = Path("results")

Y_TRAIN_PATH = PROC / "Y_train.npy"
OUT_CSV = RESULTS / "scenario_feasibility_ramp_sample0.csv"
RAMP_TOL = 1e-3

INPUTS = {
    "val_sample0": RESULTS / "Y_val_sample0_scenarios_best_denorm.npy",
    "test_sample0": RESULTS / "Y_test_sample0_scenarios_best_denorm.npy",
}

OUTPUTS = {
    "val_sample0": RESULTS / "Y_val_sample0_scenarios_best_ramp_feasible.npy",
    "test_sample0": RESULTS / "Y_test_sample0_scenarios_best_ramp_feasible.npy",
}


def compute_ramp_limits() -> np.ndarray:
    y_train = np.load(Y_TRAIN_PATH)  # (N, 24, 2)
    abs_ramps = np.abs(np.diff(y_train, axis=1))
    return np.percentile(abs_ramps.reshape(-1, 2), 99, axis=0)


def max_abs_ramp(values: np.ndarray) -> np.ndarray:
    return np.abs(np.diff(values, axis=1)).max(axis=(0, 1))


def ramp_violations(values: np.ndarray, ramp_limits: np.ndarray) -> np.ndarray:
    abs_ramps = np.abs(np.diff(values, axis=1))
    return (abs_ramps > ramp_limits.reshape(1, 1, 2) + RAMP_TOL).sum(axis=(0, 1))


def clip_ramps(values: np.ndarray, ramp_limits: np.ndarray) -> np.ndarray:
    fixed = np.clip(values.copy(), 0.0, None)

    for scen_idx in range(fixed.shape[0]):
        for t in range(1, fixed.shape[1]):
            prev = fixed[scen_idx, t - 1]
            lower = prev - ramp_limits
            upper = prev + ramp_limits
            fixed[scen_idx, t] = np.clip(fixed[scen_idx, t], lower, upper)

    return np.clip(fixed, 0.0, None)


def build_row(
    split: str,
    raw: np.ndarray,
    fixed: np.ndarray,
    ramp_limits: np.ndarray,
) -> dict:
    raw_max_ramp = max_abs_ramp(raw)
    fixed_max_ramp = max_abs_ramp(fixed)
    raw_violations = ramp_violations(raw, ramp_limits)
    fixed_violations = ramp_violations(fixed, ramp_limits)

    return {
        "split": split,
        "n_scenarios": raw.shape[0],
        "horizon": raw.shape[1],
        "wind_ramp_limit": round(float(ramp_limits[0]), 2),
        "load_ramp_limit": round(float(ramp_limits[1]), 2),
        "neg_wind_before": int((raw[:, :, 0] < 0).sum()),
        "neg_load_before": int((raw[:, :, 1] < 0).sum()),
        "neg_wind_after": int((fixed[:, :, 0] < 0).sum()),
        "neg_load_after": int((fixed[:, :, 1] < 0).sum()),
        "wind_max_abs_ramp_before": round(float(raw_max_ramp[0]), 2),
        "load_max_abs_ramp_before": round(float(raw_max_ramp[1]), 2),
        "wind_max_abs_ramp_after": round(float(fixed_max_ramp[0]), 2),
        "load_max_abs_ramp_after": round(float(fixed_max_ramp[1]), 2),
        "wind_ramp_violations_before": int(raw_violations[0]),
        "load_ramp_violations_before": int(raw_violations[1]),
        "wind_ramp_violations_after": int(fixed_violations[0]),
        "load_ramp_violations_after": int(fixed_violations[1]),
    }


def main() -> int:
    required = [Y_TRAIN_PATH, *INPUTS.values()]
    for path in required:
        if not path.exists():
            print(f"ERROR: File not found at '{path}'.")
            return 1

    ramp_limits = compute_ramp_limits()

    print("Ramp limits from Y_train 99th-percentile absolute hourly ramps:")
    print(f"  Wind : {ramp_limits[0]:,.2f} MW/hour")
    print(f"  Load : {ramp_limits[1]:,.2f} MW/hour")

    rows = []
    RESULTS.mkdir(parents=True, exist_ok=True)

    for split, in_path in INPUTS.items():
        raw = np.load(in_path)
        fixed = clip_ramps(raw, ramp_limits)
        np.save(OUTPUTS[split], fixed)

        row = build_row(split, raw, fixed, ramp_limits)
        rows.append(row)

        print(f"\n{split}:")
        print(f"  saved: {OUTPUTS[split]}")
        print(
            f"  wind ramp violations: "
            f"{row['wind_ramp_violations_before']} -> {row['wind_ramp_violations_after']}"
        )
        print(
            f"  load ramp violations: "
            f"{row['load_ramp_violations_before']} -> {row['load_ramp_violations_after']}"
        )

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    print(f"\nSaved: {OUT_CSV}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
