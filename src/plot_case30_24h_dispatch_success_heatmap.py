"""
plot_case30_24h_dispatch_success_heatmap.py
-------------------------------------------
Step 47 of the Syngrid data pipeline.

Plots OPF success/failure over scenario ID and forecast hour for the 24-hour
IEEE 30-bus scenario dispatch run.
"""

from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/syngrid_mplconfig")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


IN_CSV = Path("results/case30_test_sample0_24h_dispatch_results.csv")
OUT_PNG = Path("figures/case30_24h_dispatch_success_heatmap.png")


def main() -> int:
    if not IN_CSV.exists():
        print(f"ERROR: File not found at '{IN_CSV}'.")
        return 1

    df = pd.read_csv(IN_CSV)
    grid = df.pivot(index="scenario_id", columns="hour", values="success")
    grid = grid.sort_index().sort_index(axis=1)

    plt.figure(figsize=(8.0, 4.8))
    im = plt.imshow(grid.values, aspect="auto", interpolation="nearest", cmap="RdYlGn")
    cbar = plt.colorbar(im, ticks=[0, 1])
    cbar.ax.set_yticklabels(["Failed", "Succeeded"])

    plt.xlabel("Forecast hour")
    plt.ylabel("Scenario ID")
    plt.title("IEEE 30-bus OPF success over 24-hour scenario dispatch")
    plt.xticks(range(grid.shape[1]), grid.columns.astype(int), fontsize=7)
    plt.yticks(range(grid.shape[0]), grid.index.astype(int), fontsize=7)
    plt.tight_layout()

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PNG, dpi=200)
    plt.close()

    print(f"Saved: {OUT_PNG}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
