"""
plot_diffusion_sample_vs_truth.py
----------------------------------
Step 20 of the Syngrid data pipeline.

Produces two simple qualitative comparison figures for sample index 0:
  Figure 1 — generated vs true wind over the 24-hour forecast horizon
  Figure 2 — generated vs true load over the 24-hour forecast horizon
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROC     = Path("data/processed")
RESULTS  = Path("results")
FIGURES  = Path("figures")

SAMPLE_IDX = 0
HOURS      = np.arange(1, 25)   # forecast hours 1 → 24


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def denorm(arr: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Reverse z-score: (N, 24, 2) * (2,) + (2,)"""
    return arr * std + mean


def save_figure(
    hours:      np.ndarray,
    gen_vals:   np.ndarray,
    true_vals:  np.ndarray,
    label:      str,
    title:      str,
    out_path:   Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(hours, true_vals, label=f"True {label}",      linewidth=1.8)
    ax.plot(hours, gen_vals,  label=f"Generated {label}", linewidth=1.8, linestyle="--")

    ax.set_title(title,            fontsize=13)
    ax.set_xlabel("Forecast hour", fontsize=11)
    ax.set_ylabel("Power (MW)",    fontsize=11)
    ax.set_xticks(hours)
    ax.legend(fontsize=10, framealpha=0.85)
    ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    for p in (
        PROC    / "Y_val_norm.npy",
        PROC    / "diffusion_normalization_stats.npz",
        RESULTS / "Y_sampled_val_denorm.npy",
    ):
        if not p.exists():
            print(f"ERROR: File not found at '{p}'.")
            return 1

    Y_val_norm         = np.load(PROC    / "Y_val_norm.npy")
    stats              = np.load(PROC    / "diffusion_normalization_stats.npz")
    Y_sampled_denorm   = np.load(RESULTS / "Y_sampled_val_denorm.npy")

    y_mean = stats["y_mean"].astype(np.float32)   # (2,)
    y_std  = stats["y_std"].astype(np.float32)    # (2,)

    # Denormalise the first 8 validation ground-truth targets
    Y_true_denorm = denorm(Y_val_norm[:8], y_mean, y_std)   # (8, 24, 2)

    # Extract sample 0 trajectories
    gen_wind  = Y_sampled_denorm[SAMPLE_IDX, :, 0]   # (24,)
    gen_load  = Y_sampled_denorm[SAMPLE_IDX, :, 1]   # (24,)
    true_wind = Y_true_denorm   [SAMPLE_IDX, :, 0]   # (24,)
    true_load = Y_true_denorm   [SAMPLE_IDX, :, 1]   # (24,)

    # -----------------------------------------------------------------------
    # Figure 1 — Wind
    # -----------------------------------------------------------------------
    wind_path = FIGURES / "diffusion_sample_vs_truth_wind.png"
    save_figure(
        HOURS, gen_wind, true_wind,
        label    = "wind generation (MW)",
        title    = "Diffusion Sample vs Truth — Wind Generation (24-hour horizon)",
        out_path = wind_path,
    )

    # -----------------------------------------------------------------------
    # Figure 2 — Load
    # -----------------------------------------------------------------------
    load_path = FIGURES / "diffusion_sample_vs_truth_load.png"
    save_figure(
        HOURS, gen_load, true_load,
        label    = "load (MW)",
        title    = "Diffusion Sample vs Truth — Electricity Load (24-hour horizon)",
        out_path = load_path,
    )

    # -----------------------------------------------------------------------
    # Print summary
    # -----------------------------------------------------------------------
    print(f"Saved: {wind_path}")
    print(f"Saved: {load_path}")

    print(f"\nWind  — first hour  |  generated: {gen_wind[0]:>10,.1f} MW   "
          f"true: {true_wind[0]:>10,.1f} MW")
    print(f"Load  — first hour  |  generated: {gen_load[0]:>10,.1f} MW   "
          f"true: {true_load[0]:>10,.1f} MW")

    print(f"\nWind  generated   min: {gen_wind.min():>10,.1f}   max: {gen_wind.max():>10,.1f} MW")
    print(f"Wind  true        min: {true_wind.min():>10,.1f}   max: {true_wind.max():>10,.1f} MW")
    print(f"Load  generated   min: {gen_load.min():>10,.1f}   max: {gen_load.max():>10,.1f} MW")
    print(f"Load  true        min: {true_load.min():>10,.1f}   max: {true_load.max():>10,.1f} MW")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
