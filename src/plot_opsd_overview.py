"""
plot_opsd_overview.py
---------------------
Step 3 of the Syngrid data pipeline.

Loads the train/val/test splits, concatenates them in chronological order,
and produces a clean two-line figure (load + wind) over the first 14 days
of data for publication / proof-of-concept use.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd


SPLIT_PATHS = [
    Path("data/processed/train.csv"),
    Path("data/processed/val.csv"),
    Path("data/processed/test.csv"),
]
OUT_FIG = Path("figures/opsd_wind_load_overview.png")

WINDOW_DAYS = 14


def load_splits() -> pd.DataFrame:
    frames = []
    for p in SPLIT_PATHS:
        if not p.exists():
            raise FileNotFoundError(
                f"Split file not found: '{p}'.\n"
                "Run src/prepare_opsd_subset.py first."
            )
        frames.append(pd.read_csv(p))
    df = pd.concat(frames, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def main() -> int:
    df = load_splits()

    window_start = df["timestamp"].iloc[0]
    window_end = window_start + pd.Timedelta(days=WINDOW_DAYS)
    mask = (df["timestamp"] >= window_start) & (df["timestamp"] < window_end)
    window = df.loc[mask]

    print(f"Plotted date range : {window['timestamp'].iloc[0]}  →  {window['timestamp'].iloc[-1]}")
    print(f"Load  — min: {window['load'].min():,.1f}  max: {window['load'].max():,.1f}")
    print(f"Wind  — min: {window['wind'].min():,.1f}  max: {window['wind'].max():,.1f}")

    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(
        window["timestamp"],
        window["load"],
        label="Load (MW)",
        color="#1f77b4",
        linewidth=1.4,
    )
    ax.plot(
        window["timestamp"],
        window["wind"],
        label="Wind generation (MW)",
        color="#2ca02c",
        linewidth=1.4,
        linestyle="--",
    )

    ax.set_title("Germany: Electricity Load and Wind Generation (first 14 days)", fontsize=13)
    ax.set_xlabel("Date (UTC)", fontsize=11)
    ax.set_ylabel("Power (MW)", fontsize=11)
    ax.legend(fontsize=10, framealpha=0.85)

    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
    fig.autofmt_xdate(rotation=30, ha="right")

    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{int(x):,}")
    )
    ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved: {OUT_FIG}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
