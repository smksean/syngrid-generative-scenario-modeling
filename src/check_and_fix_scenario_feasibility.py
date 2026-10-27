"""
check_and_fix_scenario_feasibility.py
--------------------------------------
Step 23 of the Syngrid data pipeline.

Checks generated scenarios for basic physical feasibility and applies
a minimal first-stage correction: non-negativity clipping for wind and load.

Channel order: [..., 0] = wind   [..., 1] = load
"""

from pathlib import Path

import numpy as np


RESULTS   = Path("results")
IN_PATH   = RESULTS / "Y_val_sample0_scenarios_denorm.npy"
OUT_PATH  = RESULTS / "Y_val_sample0_scenarios_denorm_feasible.npy"


def feasibility_report(tag: str, wind: np.ndarray, load: np.ndarray) -> None:
    n_total = wind.size

    neg_wind = int((wind < 0).sum())
    neg_load = int((load < 0).sum())

    print(f"  [{tag}]")
    print(f"    Negative wind entries : {neg_wind:>6}  "
          f"({100 * neg_wind / n_total:.2f}%)")
    print(f"    Negative load entries : {neg_load:>6}  "
          f"({100 * neg_load / n_total:.2f}%)")
    print(f"    Wind  min : {wind.min():>12,.2f} MW")
    print(f"    Load  min : {load.min():>12,.2f} MW")


def main() -> int:
    if not IN_PATH.exists():
        print(f"ERROR: File not found at '{IN_PATH}'.")
        print("Run src/sample_multiple_diffusion_scenarios.py first.")
        return 1

    scenarios = np.load(IN_PATH)   # (20, 24, 2)

    n_scenarios = scenarios.shape[0]
    horizon     = scenarios.shape[1]

    wind = scenarios[:, :, 0].copy()   # (20, 24)
    load = scenarios[:, :, 1].copy()   # (20, 24)

    print(f"{'='*60}")
    print(f"Scenario array shape : {scenarios.shape}")
    print(f"N scenarios          : {n_scenarios}")
    print(f"Forecast horizon     : {horizon} hours")
    print(f"{'='*60}")

    print("\nFeasibility diagnostics — BEFORE correction:")
    feasibility_report("raw", wind, load)

    # Non-negativity correction
    wind_fixed = np.clip(wind, a_min=0.0, a_max=None)
    load_fixed = np.clip(load, a_min=0.0, a_max=None)

    n_wind_corrected = int((wind < 0).sum())
    n_load_corrected = int((load < 0).sum())

    print(f"\nCorrection applied: clip(x, 0, ∞)")
    print(f"  Wind entries corrected : {n_wind_corrected}")
    print(f"  Load entries corrected : {n_load_corrected}")

    print("\nFeasibility diagnostics — AFTER correction:")
    feasibility_report("clipped", wind_fixed, load_fixed)

    # Re-assemble into (20, 24, 2) and save
    scenarios_feasible = np.stack([wind_fixed, load_fixed], axis=-1)
    RESULTS.mkdir(parents=True, exist_ok=True)
    np.save(OUT_PATH, scenarios_feasible)

    print(f"\nSaved: {OUT_PATH}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
