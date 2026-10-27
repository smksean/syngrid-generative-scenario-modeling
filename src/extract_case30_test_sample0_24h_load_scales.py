"""
extract_case30_test_sample0_24h_load_scales.py
----------------------------------------------
Step 44 of the Syngrid data pipeline.

Extracts 24-hour load multipliers from ramp-feasible generated scenarios for
test sample 0. These multipliers can be used as MATPOWER load scales for
IEEE 30-bus scenario dispatch across the full forecast horizon.

Channel order: [..., 0] = wind   [..., 1] = load
"""

from pathlib import Path

import numpy as np
import pandas as pd


PROC = Path("data/processed")
RESULTS = Path("results")

SCENARIOS_PATH = RESULTS / "Y_test_sample0_scenarios_best_ramp_feasible.npy"
Y_TRUE_PATH = PROC / "Y_test_norm.npy"
STATS_PATH = PROC / "diffusion_normalization_stats.npz"
OUT_CSV = RESULTS / "case30_test_sample0_24h_load_scales.csv"

SAMPLE_ID = 0


def main() -> int:
    for path in (SCENARIOS_PATH, Y_TRUE_PATH, STATS_PATH):
        if not path.exists():
            print(f"ERROR: File not found at '{path}'.")
            return 1

    scenarios = np.load(SCENARIOS_PATH)  # (20, 24, 2)
    y_test_norm = np.load(Y_TRUE_PATH)
    stats = np.load(STATS_PATH)

    y_mean = stats["y_mean"].astype(np.float32)
    y_std = stats["y_std"].astype(np.float32)

    y_true = y_test_norm[SAMPLE_ID] * y_std + y_mean
    true_load = y_true[:, 1]             # (24,)
    scenario_loads = scenarios[:, :, 1]  # (20, 24)
    load_scales = scenario_loads / true_load[None, :]

    rows = []
    for scenario_id in range(scenarios.shape[0]):
        for hour in range(scenarios.shape[1]):
            rows.append({
                "scenario_id": scenario_id,
                "hour": hour,
                "scenario_load_mw": round(float(scenario_loads[scenario_id, hour]), 2),
                "reference_true_load_mw": round(float(true_load[hour]), 2),
                "load_scale": round(float(load_scales[scenario_id, hour]), 6),
            })

    df = pd.DataFrame(rows)

    RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    print(f"Rows saved : {len(df)}")
    print(f"Scenarios  : {scenarios.shape[0]}")
    print(f"Horizon    : {scenarios.shape[1]}")
    print(f"Scale min  : {df['load_scale'].min():.4f}")
    print(f"Scale mean : {df['load_scale'].mean():.4f}")
    print(f"Scale max  : {df['load_scale'].max():.4f}")
    print(f"\nSaved: {OUT_CSV}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
