"""
summarize_case30_scenario_dispatch.py
-------------------------------------
Step 41 of the Syngrid data pipeline.

Summarizes IEEE 30-bus MATPOWER OPF results for scenario-derived load scales.
"""

from pathlib import Path

import pandas as pd


IN_CSV = Path("results/case30_test_sample0_dispatch_results.csv")
OUT_CSV = Path("results/case30_test_sample0_dispatch_summary.csv")


def main() -> int:
    if not IN_CSV.exists():
        print(f"ERROR: File not found at '{IN_CSV}'.")
        return 1

    df = pd.read_csv(IN_CSV)
    ok = df[df["success"] == 1]
    failed = df[df["success"] == 0]

    row = {
        "n_scenarios": len(df),
        "success_count": int((df["success"] == 1).sum()),
        "failure_count": int((df["success"] == 0).sum()),
        "success_rate": round(float((df["success"] == 1).mean()), 4),
        "load_scale_min": round(float(df["load_scale"].min()), 6),
        "load_scale_mean": round(float(df["load_scale"].mean()), 6),
        "load_scale_max": round(float(df["load_scale"].max()), 6),
        "successful_cost_min": round(float(ok["objective_cost"].min()), 2),
        "successful_cost_mean": round(float(ok["objective_cost"].mean()), 2),
        "successful_cost_max": round(float(ok["objective_cost"].max()), 2),
        "failed_scenario_ids": " ".join(str(x) for x in failed["scenario_id"].tolist()),
    }

    summary = pd.DataFrame([row])

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_CSV, index=False)

    print(summary.T.to_string(header=False))
    print(f"\nSaved: {OUT_CSV}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
