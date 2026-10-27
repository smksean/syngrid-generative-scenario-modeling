"""
summarize_scenario_probabilistic_metrics_subset.py
--------------------------------------------------
Step 33 of the Syngrid data pipeline.

Summarizes subset probabilistic scenario-quality metrics by split.
"""

from pathlib import Path
import argparse

import pandas as pd


DEFAULT_IN_CSV = Path("results/scenario_probabilistic_metrics_subset.csv")
DEFAULT_OUT_CSV = Path("results/scenario_probabilistic_metrics_subset_summary.csv")

METRIC_COLS = [
    "wind_crps_mean",
    "load_crps_mean",
    "joint_energy_score",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize subset probabilistic scenario-quality metrics by split."
    )
    parser.add_argument(
        "--in-csv",
        type=Path,
        default=DEFAULT_IN_CSV,
        help=f"Input per-sample metrics CSV (default: {DEFAULT_IN_CSV}).",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=DEFAULT_OUT_CSV,
        help=f"Output summary CSV (default: {DEFAULT_OUT_CSV}).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.in_csv.exists():
        print(f"ERROR: File not found at '{args.in_csv}'.")
        print("Run src/evaluate_scenario_probabilistic_metrics_subset.py first.")
        return 1

    df = pd.read_csv(args.in_csv)

    summary = (
        df.groupby("split")[METRIC_COLS]
        .agg(["mean", "std", "min", "max"])
        .round(2)
    )

    summary.columns = [
        f"{metric}_{stat}" for metric, stat in summary.columns.to_flat_index()
    ]
    summary = summary.reset_index()

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out_csv, index=False)

    print(summary.to_string(index=False))
    print(f"\nSaved: {args.out_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
