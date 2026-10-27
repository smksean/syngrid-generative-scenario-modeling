"""
compare_val_test_scenario_quality.py
--------------------------------------
Step 29 of the Syngrid data pipeline.

Builds a two-row val / test scenario quality comparison table and applies
a non-negativity feasibility correction to the test scenario array.
"""

from pathlib import Path

import numpy as np
import pandas as pd


RESULTS = Path("results")
OUT_CSV  = RESULTS / "scenario_quality_val_vs_test.csv"
OUT_FIX  = RESULTS / "Y_test_sample0_scenarios_best_feasible.npy"

COLS = [
    "split", "n_scenarios", "horizon",
    "wind_mean_std", "load_mean_std",
    "wind_min", "wind_max",
    "load_min", "load_max",
    "wind_first_hour_bias", "load_first_hour_bias",
    "neg_wind", "neg_load",
]


def main() -> int:
    for p in (
        RESULTS / "scenario_quality_comparison_runs.csv",
        RESULTS / "scenario_quality_test_sample0.csv",
        RESULTS / "Y_test_sample0_scenarios_best_denorm.npy",
    ):
        if not p.exists():
            print(f"ERROR: File not found at '{p}'.")
            return 1

    # Load CSVs
    runs_df = pd.read_csv(RESULTS / "scenario_quality_comparison_runs.csv")
    test_df = pd.read_csv(RESULTS / "scenario_quality_test_sample0.csv")

    # Extract the 20-epoch best validation row
    val_row = runs_df.loc[runs_df["run"] == "20_epoch_best"].iloc[0]
    tst_row = test_df.iloc[0]

    # Apply non-negativity correction to test scenario array
    Y_test = np.load(RESULTS / "Y_test_sample0_scenarios_best_denorm.npy")  # (20, 24, 2)

    wind_raw = Y_test[:, :, 0]
    load_raw = Y_test[:, :, 1]

    neg_wind_before = int((wind_raw < 0).sum())
    neg_load_before = int((load_raw < 0).sum())
    wind_min_before = float(wind_raw.min())
    load_min_before = float(load_raw.min())

    wind_fixed = np.clip(wind_raw, 0.0, None)
    load_fixed = np.clip(load_raw, 0.0, None)

    neg_wind_after = int((wind_fixed < 0).sum())
    neg_load_after = int((load_fixed < 0).sum())
    wind_min_after = float(wind_fixed.min())
    load_min_after = float(load_fixed.min())

    Y_test_feasible = np.stack([wind_fixed, load_fixed], axis=-1)
    RESULTS.mkdir(parents=True, exist_ok=True)
    np.save(OUT_FIX, Y_test_feasible)

    # Build comparison rows
    val_entry = {
        "split":               "val_sample0",
        "n_scenarios":         int(val_row["n_scenarios"]),
        "horizon":             int(val_row["horizon"]),
        "wind_mean_std":       round(float(val_row["wind_mean_std"]),        2),
        "load_mean_std":       round(float(val_row["load_mean_std"]),        2),
        "wind_min":            round(float(val_row["wind_min"]),             2),
        "wind_max":            round(float(val_row["wind_max"]),             2),
        "load_min":            round(float(val_row["load_min"]),             2),
        "load_max":            round(float(val_row["load_max"]),             2),
        "wind_first_hour_bias": round(float(val_row["wind_first_hour_bias"]), 2),
        "load_first_hour_bias": round(float(val_row["load_first_hour_bias"]), 2),
        "neg_wind":            int(val_row["neg_wind"]),
        "neg_load":            int(val_row["neg_load"]),
    }

    tst_entry = {
        "split":               "test_sample0",
        "n_scenarios":         int(tst_row["n_scenarios"]),
        "horizon":             int(tst_row["horizon"]),
        "wind_mean_std":       round(float(tst_row["wind_mean_std"]),        2),
        "load_mean_std":       round(float(tst_row["load_mean_std"]),        2),
        "wind_min":            round(float(tst_row["wind_min"]),             2),
        "wind_max":            round(float(tst_row["wind_max"]),             2),
        "load_min":            round(float(tst_row["load_min"]),             2),
        "load_max":            round(float(tst_row["load_max"]),             2),
        "wind_first_hour_bias": round(float(tst_row["wind_first_hour_bias"]), 2),
        "load_first_hour_bias": round(float(tst_row["load_first_hour_bias"]), 2),
        "neg_wind":            int(tst_row["neg_wind"]),
        "neg_load":            int(tst_row["neg_load"]),
    }

    df = pd.DataFrame([val_entry, tst_entry], columns=COLS)
    df.to_csv(OUT_CSV, index=False)

    print(df.T.to_string(header=False))
    print(f"\nSaved: {OUT_CSV}")

    print(f"\n{'='*60}")
    print("Test scenario non-negativity correction:")
    print(f"  Negative wind entries : {neg_wind_before:>4}  →  {neg_wind_after}")
    print(f"  Negative load entries : {neg_load_before:>4}  →  {neg_load_after}")
    print(f"  Wind min before clip  : {wind_min_before:>10,.2f} MW")
    print(f"  Wind min after  clip  : {wind_min_after:>10,.2f} MW")
    print(f"  Load min before clip  : {load_min_before:>10,.2f} MW")
    print(f"  Load min after  clip  : {load_min_after:>10,.2f} MW")
    print(f"\nSaved: {OUT_FIX}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
