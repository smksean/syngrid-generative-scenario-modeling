"""
compare_probabilistic_metrics_50.py
-----------------------------------
Step 35 of the Syngrid data pipeline.

Combines diffusion scenario metrics and deterministic baseline metric
references for the same 50-sample validation/test subset.
"""

from pathlib import Path

import pandas as pd


DIFFUSION_CSV = Path("results/scenario_probabilistic_metrics_subset_50.csv")
BASELINE_CSV = Path("results/baseline_probabilistic_metric_reference_50.csv")
OUT_CSV = Path("results/probabilistic_metrics_comparison_50.csv")

METRIC_COLS = [
    "wind_crps_mean",
    "load_crps_mean",
    "joint_energy_score",
]


def main() -> int:
    for path in (DIFFUSION_CSV, BASELINE_CSV):
        if not path.exists():
            print(f"ERROR: File not found at '{path}'.")
            return 1

    diffusion = pd.read_csv(DIFFUSION_CSV)
    diffusion.insert(0, "model", "diffusion")

    baseline = pd.read_csv(BASELINE_CSV)

    combined = pd.concat([diffusion, baseline], ignore_index=True)

    comparison = (
        combined.groupby(["model", "split"])[METRIC_COLS]
        .agg(["mean", "std"])
        .round(2)
    )
    comparison.columns = [
        f"{metric}_{stat}" for metric, stat in comparison.columns.to_flat_index()
    ]
    comparison = comparison.reset_index()

    sort_order = {
        "diffusion": 0,
        "linear_regression": 1,
        "persistence": 2,
    }
    comparison["_sort"] = comparison["model"].map(sort_order).fillna(99)
    comparison = (
        comparison.sort_values(["_sort", "split"])
        .drop(columns="_sort")
        .reset_index(drop=True)
    )

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(OUT_CSV, index=False)

    print(comparison.to_string(index=False))
    print(f"\nSaved: {OUT_CSV}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
