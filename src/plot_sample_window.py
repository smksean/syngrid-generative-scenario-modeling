"""
plot_sample_window.py
---------------------
Step 4 of the Syngrid data pipeline.

Loads the processed hourly dataset and plots a single 72-hour window of
Germany load and wind generation for use as a forecasting-context figure
in the paper.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd


DATA_PATH = Path("data/processed/opsd_de_wind_load_hourly.csv")
OUT_FIG   = Path("figures/sample_72h_window.png")

WINDOW_START = pd.Timestamp("2019-01-15 00:00:00", tz="UTC")
WINDOW_HOURS = 72


def main() -> int:
    if not DATA_PATH.exists():
        print(f"ERROR: File not found at '{DATA_PATH}'.")
        print("Run src/prepare_opsd_subset.py first.")
        return 1

    df = pd.read_csv(DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    window_end = WINDOW_START + pd.Timedelta(hours=WINDOW_HOURS)
    mask = (df["timestamp"] >= WINDOW_START) & (df["timestamp"] < window_end)
    window = df.loc[mask].copy()

    if window.empty:
        print(f"ERROR: No data found in the window {WINDOW_START} → {window_end}.")
        return 1

    print(f"Plotted range : {window['timestamp'].iloc[0]}  →  {window['timestamp'].iloc[-1]}")
    print(f"Load  — min: {window['load'].min():,.1f}  max: {window['load'].max():,.1f}")
    print(f"Wind  — min: {window['wind'].min():,.1f}  max: {window['wind'].max():,.1f}")

    fig, ax = plt.subplots(figsize=(11, 4))

    ax.plot(
        window["timestamp"],
        window["load"],
        label="Load (MW)",
        color="#1f77b4",
        linewidth=1.6,
    )
    ax.plot(
        window["timestamp"],
        window["wind"],
        label="Wind generation (MW)",
        color="#2ca02c",
        linewidth=1.6,
        linestyle="--",
    )

    ax.set_title(
        "Germany: 72-Hour Load and Wind Generation Window (15–17 Jan 2019)",
        fontsize=13,
    )
    ax.set_xlabel("Date / Time (UTC)", fontsize=11)
    ax.set_ylabel("Power (MW)", fontsize=11)
    ax.legend(fontsize=10, framealpha=0.85)

    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 12]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=3))

    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.7)
    ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved: {OUT_FIG}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
