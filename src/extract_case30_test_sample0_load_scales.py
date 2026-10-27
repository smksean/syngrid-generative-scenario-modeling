"""
extract_case30_test_sample0_load_scales.py
------------------------------------------
Step 40 of the Syngrid data pipeline.

Extracts first-hour load multipliers from ramp-feasible generated scenarios
for test sample 0. These multipliers can be used as MATPOWER load scales for
single-hour IEEE 30-bus scenario dispatch.

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
OUT_CSV = RESULTS / "case30_test_sample0_load_scales.csv"

SAMPLE_ID = 0
HOUR = 0


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
    true_load_mw = float(y_true[HOUR, 1])

    scenario_loads_mw = scenarios[:, HOUR, 1]
    load_scales = scenario_loads_mw / true_load_mw

    df = pd.DataFrame({
        "scenario_id": np.arange(len(scenario_loads_mw)),
        "hour": HOUR,
        "scenario_load_mw": np.round(scenario_loads_mw, 2),
        "reference_true_load_mw": round(true_load_mw, 2),
        "load_scale": np.round(load_scales, 6),
    })

    RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    print(df.to_string(index=False))
    print(f"\nSaved: {OUT_CSV}")
    print("\nSummary:")
    print(f"  min scale : {load_scales.min():.4f}")
    print(f"  mean scale: {load_scales.mean():.4f}")
    print(f"  max scale : {load_scales.max():.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
