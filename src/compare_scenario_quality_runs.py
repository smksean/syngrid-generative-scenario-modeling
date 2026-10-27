"""
compare_scenario_quality_runs.py
---------------------------------
Step 27 of the Syngrid data pipeline.

Side-by-side scenario quality comparison for validation sample 0:
  row 1 — 5-epoch checkpoint
  row 2 — 20-epoch best checkpoint
"""

from pathlib import Path

import numpy as np
import pandas as pd


PROC    = Path("data/processed")
RESULTS = Path("results")
OUT_CSV = RESULTS / "scenario_quality_comparison_runs.csv"

COLS = [
    "run", "n_scenarios", "horizon",
    "wind_mean_std", "load_mean_std",
    "wind_min", "wind_max",
    "load_min", "load_max",
    "wind_first_hour_ensemble_mean", "wind_first_hour_true", "wind_first_hour_bias",
    "load_first_hour_ensemble_mean", "load_first_hour_true", "load_first_hour_bias",
    "neg_wind", "neg_load",
]


def compute_row(
    run_label:    str,
    Y_scen:       np.ndarray,   # (20, 24, 2)
    y_true_denorm: np.ndarray,  # (24, 2)
) -> dict:
    wind = Y_scen[:, :, 0]   # (20, 24)
    load = Y_scen[:, :, 1]   # (20, 24)

    wind_fh_mean = float(wind[:, 0].mean())
    load_fh_mean = float(load[:, 0].mean())
    wind_fh_true = float(y_true_denorm[0, 0])
    load_fh_true = float(y_true_denorm[0, 1])

    return {
        "run":                          run_label,
        "n_scenarios":                  Y_scen.shape[0],
        "horizon":                      Y_scen.shape[1],
        "wind_mean_std":                round(float(wind.std(axis=0).mean()), 2),
        "load_mean_std":                round(float(load.std(axis=0).mean()), 2),
        "wind_min":                     round(float(wind.min()), 2),
        "wind_max":                     round(float(wind.max()), 2),
        "load_min":                     round(float(load.min()), 2),
        "load_max":                     round(float(load.max()), 2),
        "wind_first_hour_ensemble_mean": round(wind_fh_mean, 2),
        "wind_first_hour_true":          round(wind_fh_true, 2),
        "wind_first_hour_bias":          round(wind_fh_mean - wind_fh_true, 2),
        "load_first_hour_ensemble_mean": round(load_fh_mean, 2),
        "load_first_hour_true":          round(load_fh_true, 2),
        "load_first_hour_bias":          round(load_fh_mean - load_fh_true, 2),
        "neg_wind":                      int((wind < 0).sum()),
        "neg_load":                      int((load < 0).sum()),
    }


def pct_change(old: float, new: float) -> str:
    if old == 0:
        return "N/A"
    return f"{100 * (new - old) / abs(old):+.1f}%"


def main() -> int:
    for p in (
        RESULTS / "Y_val_sample0_scenarios_denorm.npy",
        RESULTS / "Y_val_sample0_scenarios_best_denorm.npy",
        PROC    / "Y_val_norm.npy",
        PROC    / "diffusion_normalization_stats.npz",
    ):
        if not p.exists():
            print(f"ERROR: File not found at '{p}'.")
            return 1

    Y_5ep   = np.load(RESULTS / "Y_val_sample0_scenarios_denorm.npy")       # (20, 24, 2)
    Y_best  = np.load(RESULTS / "Y_val_sample0_scenarios_best_denorm.npy")  # (20, 24, 2)
    Y_val   = np.load(PROC    / "Y_val_norm.npy")
    stats   = np.load(PROC    / "diffusion_normalization_stats.npz")

    y_mean = stats["y_mean"].astype(np.float32)
    y_std  = stats["y_std"].astype(np.float32)

    y_true_denorm = Y_val[0] * y_std + y_mean   # (24, 2)

    row_5ep  = compute_row("5_epoch",        Y_5ep,  y_true_denorm)
    row_best = compute_row("20_epoch_best",  Y_best, y_true_denorm)

    df = pd.DataFrame([row_5ep, row_best], columns=COLS)

    RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    print(df.T.to_string(header=False))
    print(f"\nSaved: {OUT_CSV}")

    # Percentage-change summary
    print(f"\n{'='*60}")
    print("Improvement (5-epoch → 20-epoch best):")
    print(f"  Wind mean std   : "
          f"{row_5ep['wind_mean_std']:>10,.2f}  →  {row_best['wind_mean_std']:>10,.2f} MW  "
          f"({pct_change(row_5ep['wind_mean_std'], row_best['wind_mean_std'])})")
    print(f"  Load mean std   : "
          f"{row_5ep['load_mean_std']:>10,.2f}  →  {row_best['load_mean_std']:>10,.2f} MW  "
          f"({pct_change(row_5ep['load_mean_std'], row_best['load_mean_std'])})")
    print(f"  Wind global max : "
          f"{row_5ep['wind_max']:>10,.2f}  →  {row_best['wind_max']:>10,.2f} MW  "
          f"({pct_change(row_5ep['wind_max'], row_best['wind_max'])})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
