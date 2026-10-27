"""
make_scenario_quality_summary.py
---------------------------------
Step 24 of the Syngrid data pipeline.

Aggregates first-pass scenario-quality diagnostics for validation sample 0
into a single compact one-row CSV summary table.
"""

from pathlib import Path

import numpy as np
import pandas as pd


PROC    = Path("data/processed")
RESULTS = Path("results")
OUT_CSV = RESULTS / "scenario_quality_summary_sample0.csv"

SAMPLE_ID = 0


def main() -> int:
    for p in (
        RESULTS / "Y_val_sample0_scenarios_denorm.npy",
        RESULTS / "Y_val_sample0_scenarios_denorm_feasible.npy",
        PROC    / "Y_val_norm.npy",
        PROC    / "diffusion_normalization_stats.npz",
    ):
        if not p.exists():
            print(f"ERROR: File not found at '{p}'.")
            return 1

    Y_scen_raw  = np.load(RESULTS / "Y_val_sample0_scenarios_denorm.npy")      # (20, 24, 2)
    Y_scen_fix  = np.load(RESULTS / "Y_val_sample0_scenarios_denorm_feasible.npy")  # (20, 24, 2)
    Y_val_norm  = np.load(PROC    / "Y_val_norm.npy")
    stats       = np.load(PROC    / "diffusion_normalization_stats.npz")

    y_mean = stats["y_mean"].astype(np.float32)
    y_std  = stats["y_std"].astype(np.float32)

    # Denormalise true target for sample 0
    y_true = Y_val_norm[SAMPLE_ID] * y_std + y_mean   # (24, 2)

    # Convenience slices
    wind_raw = Y_scen_raw[:, :, 0]   # (20, 24)
    load_raw = Y_scen_raw[:, :, 1]   # (20, 24)
    wind_fix = Y_scen_fix[:, :, 0]
    load_fix = Y_scen_fix[:, :, 1]

    n_scenarios = Y_scen_raw.shape[0]
    horizon     = Y_scen_raw.shape[1]

    # Diversity
    wind_mean_std = float(wind_raw.std(axis=0).mean())
    load_mean_std = float(load_raw.std(axis=0).mean())

    # Range
    wind_min = float(wind_raw.min())
    wind_max = float(wind_raw.max())
    load_min = float(load_raw.min())
    load_max = float(load_raw.max())

    # First-hour ensemble mean
    wind_fh_mean = float(wind_raw[:, 0].mean())
    load_fh_mean = float(load_raw[:, 0].mean())

    # First-hour true values
    wind_fh_true = float(y_true[0, 0])
    load_fh_true = float(y_true[0, 1])

    # First-hour bias
    wind_fh_bias = wind_fh_mean - wind_fh_true
    load_fh_bias = load_fh_mean - load_fh_true

    # Feasibility counts — before / after
    neg_wind_before = int((wind_raw < 0).sum())
    neg_load_before = int((load_raw < 0).sum())
    neg_wind_after  = int((wind_fix < 0).sum())
    neg_load_after  = int((load_fix < 0).sum())

    row = {
        "sample_id":                   SAMPLE_ID,
        "n_scenarios":                 n_scenarios,
        "horizon":                     horizon,
        "wind_mean_std":               round(wind_mean_std, 2),
        "load_mean_std":               round(load_mean_std, 2),
        "wind_min":                    round(wind_min, 2),
        "wind_max":                    round(wind_max, 2),
        "load_min":                    round(load_min, 2),
        "load_max":                    round(load_max, 2),
        "wind_first_hour_ensemble_mean": round(wind_fh_mean, 2),
        "wind_first_hour_true":          round(wind_fh_true, 2),
        "wind_first_hour_bias":          round(wind_fh_bias, 2),
        "load_first_hour_ensemble_mean": round(load_fh_mean, 2),
        "load_first_hour_true":          round(load_fh_true, 2),
        "load_first_hour_bias":          round(load_fh_bias, 2),
        "neg_wind_before":             neg_wind_before,
        "neg_load_before":             neg_load_before,
        "neg_wind_after":              neg_wind_after,
        "neg_load_after":              neg_load_after,
    }

    df = pd.DataFrame([row])

    RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    print(df.T.to_string(header=False))
    print(f"\nSaved: {OUT_CSV}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
