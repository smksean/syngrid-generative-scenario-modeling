"""
summarize_case30_24h_scenario_dispatch.py
-----------------------------------------
Step 45 of the Syngrid data pipeline.

Summarizes 24-hour IEEE 30-bus MATPOWER OPF results for scenario-derived
load scales.
"""

from pathlib import Path

import pandas as pd


IN_CSV = Path("results/case30_test_sample0_24h_dispatch_results.csv")
OUT_SUMMARY_CSV = Path("results/case30_test_sample0_24h_dispatch_summary.csv")
OUT_FAILURES_CSV = Path("results/case30_test_sample0_24h_dispatch_failures_by_scenario.csv")


def main() -> int:
    if not IN_CSV.exists():
        print(f"ERROR: File not found at '{IN_CSV}'.")
        return 1

    df = pd.read_csv(IN_CSV)
    ok = df[df["success"] == 1]
    failed = df[df["success"] == 0]

    failure_counts = (
        failed.groupby("scenario_id")
        .size()
        .rename("failure_count")
        .reset_index()
        .sort_values(["failure_count", "scenario_id"], ascending=[False, True])
    )

    row = {
        "n_dispatch_cases": len(df),
        "n_scenarios": int(df["scenario_id"].nunique()),
        "horizon": int(df["hour"].nunique()),
        "success_count": int((df["success"] == 1).sum()),
        "failure_count": int((df["success"] == 0).sum()),
        "success_rate": round(float((df["success"] == 1).mean()), 4),
        "load_scale_min": round(float(df["load_scale"].min()), 6),
        "load_scale_mean": round(float(df["load_scale"].mean()), 6),
        "load_scale_max": round(float(df["load_scale"].max()), 6),
        "successful_cost_min": round(float(ok["objective_cost"].min()), 2),
        "successful_cost_mean": round(float(ok["objective_cost"].mean()), 2),
        "successful_cost_max": round(float(ok["objective_cost"].max()), 2),
        "n_failed_scenarios": int(failure_counts["scenario_id"].nunique()),
    }

    summary = pd.DataFrame([row])

    OUT_SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_SUMMARY_CSV, index=False)
    failure_counts.to_csv(OUT_FAILURES_CSV, index=False)

    print(summary.T.to_string(header=False))
    print(f"\nSaved: {OUT_SUMMARY_CSV}")
    print(f"Saved: {OUT_FAILURES_CSV}")

    if len(failure_counts) > 0:
        print("\nFailure counts by scenario:")
        print(failure_counts.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
