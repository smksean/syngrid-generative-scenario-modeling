"""
make_baseline_results_table.py
------------------------------
Step 9 of the Syngrid data pipeline.

Combines the persistence and linear regression baseline metrics into a
single clean comparison table for use in the paper.
"""

from pathlib import Path

import pandas as pd


PERSIST_CSV = Path("results/baseline_persistence_metrics.csv")
LR_CSV      = Path("results/baseline_linear_regression_metrics.csv")
OUT_CSV     = Path("results/baseline_comparison_table.csv")

METRIC_COLS = ["load_mae", "load_rmse", "wind_mae", "wind_rmse"]
FINAL_COLS  = ["model", "split", "load_mae", "load_rmse",
               "wind_mae", "wind_rmse", "n_samples", "horizon"]

SPLIT_ORDER = [
    ("persistence",        "val"),
    ("linear_regression",  "val"),
    ("persistence",        "test"),
    ("linear_regression",  "test"),
]


def main() -> int:
    for p in (PERSIST_CSV, LR_CSV):
        if not p.exists():
            print(f"ERROR: File not found at '{p}'.")
            print("Run the baseline scripts first.")
            return 1

    persist = pd.read_csv(PERSIST_CSV)
    lr      = pd.read_csv(LR_CSV)

    # Pull the single persistence metric values (same for both splits).
    p_row = persist.iloc[0]

    # Look up per-split sample counts from the LR results.
    n_val  = int(lr.loc[lr["split"] == "val",  "n_samples"].iloc[0])
    n_test = int(lr.loc[lr["split"] == "test", "n_samples"].iloc[0])

    persist_rows = pd.DataFrame([
        {
            "model":     "persistence",
            "split":     "val",
            "load_mae":  p_row["load_mae"],
            "load_rmse": p_row["load_rmse"],
            "wind_mae":  p_row["wind_mae"],
            "wind_rmse": p_row["wind_rmse"],
            "n_samples": n_val,
            "horizon":   int(p_row["horizon"]),
        },
        {
            "model":     "persistence",
            "split":     "test",
            "load_mae":  p_row["load_mae"],
            "load_rmse": p_row["load_rmse"],
            "wind_mae":  p_row["wind_mae"],
            "wind_rmse": p_row["wind_rmse"],
            "n_samples": n_test,
            "horizon":   int(p_row["horizon"]),
        },
    ])

    combined = pd.concat([persist_rows, lr[FINAL_COLS]], ignore_index=True)

    # Sort into the required order.
    sort_key = {pair: i for i, pair in enumerate(SPLIT_ORDER)}
    combined["_order"] = combined.apply(
        lambda r: sort_key.get((r["model"], r["split"]), 99), axis=1
    )
    combined = (
        combined.sort_values("_order")
        .drop(columns="_order")
        .reset_index(drop=True)
    )

    combined[METRIC_COLS] = combined[METRIC_COLS].round(2)
    combined = combined[FINAL_COLS]

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUT_CSV, index=False)

    print(combined.to_string(index=False))
    print(f"\nSaved: {OUT_CSV}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
