"""
plot_case30_scenario_dispatch.py
--------------------------------
Step 43 of the Syngrid data pipeline.

Plots IEEE 30-bus OPF objective cost against scenario-derived load scale.
"""

from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/syngrid_mplconfig")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


IN_CSV = Path("results/case30_test_sample0_dispatch_results.csv")
OUT_PNG = Path("figures/case30_scenario_dispatch_cost.png")


def main() -> int:
    if not IN_CSV.exists():
        print(f"ERROR: File not found at '{IN_CSV}'.")
        return 1

    df = pd.read_csv(IN_CSV)
    ok = df[df["success"] == 1]
    failed = df[df["success"] == 0]

    plt.figure(figsize=(7.0, 4.2))

    plt.scatter(
        ok["load_scale"],
        ok["objective_cost"],
        s=46,
        label="Successful OPF",
        color="#1f77b4",
        edgecolor="black",
        linewidth=0.4,
    )

    if len(failed) > 0:
        plt.scatter(
            failed["load_scale"],
            failed["objective_cost"],
            s=70,
            marker="x",
            label="Failed OPF",
            color="#d62728",
            linewidth=1.8,
        )

    plt.xlabel("Scenario load scale")
    plt.ylabel("OPF objective cost")
    plt.title("IEEE 30-bus scenario dispatch cost")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PNG, dpi=200)
    plt.close()

    print(f"Saved: {OUT_PNG}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
