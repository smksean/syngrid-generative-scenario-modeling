"""
plot_multiple_diffusion_scenarios.py
-------------------------------------
Step 22 of the Syngrid data pipeline.

Plots 20 independently sampled diffusion scenarios against the true
validation trajectory for sample index 0 — one figure per target
variable (wind / load).
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROC    = Path("data/processed")
RESULTS = Path("results")
FIGURES = Path("figures")

HOURS = np.arange(1, 25)   # forecast hours 1 → 24

SCENARIO_ALPHA  = 0.25   # transparency for individual scenario lines
SCENARIO_LW     = 0.9    # linewidth for scenario lines
TRUE_LW         = 2.4    # linewidth for the true trajectory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def denorm(arr: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return arr * std + mean


def save_scenario_figure(
    hours:        np.ndarray,
    scenarios:    np.ndarray,    # (20, 24) — generated trajectories
    true_vals:    np.ndarray,    # (24,)    — ground truth
    var_label:    str,
    title:        str,
    out_path:     Path,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.5))

    # Draw all generated scenarios first (behind truth)
    for i, s in enumerate(scenarios):
        label = "Generated scenarios" if i == 0 else None
        ax.plot(
            hours, s,
            color="tab:blue",
            linewidth=SCENARIO_LW,
            alpha=SCENARIO_ALPHA,
            label=label,
        )

    # Draw true trajectory on top
    ax.plot(
        hours, true_vals,
        color="tab:red",
        linewidth=TRUE_LW,
        label=f"True {var_label}",
        zorder=5,
    )

    ax.set_title(title,             fontsize=13)
    ax.set_xlabel("Forecast hour",  fontsize=11)
    ax.set_ylabel("Power (MW)",     fontsize=11)
    ax.set_xticks(hours)
    ax.legend(fontsize=10, framealpha=0.85)
    ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))

    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    for p in (
        RESULTS / "Y_val_sample0_scenarios_denorm.npy",
        PROC    / "Y_val_norm.npy",
        PROC    / "diffusion_normalization_stats.npz",
    ):
        if not p.exists():
            print(f"ERROR: File not found at '{p}'.")
            return 1

    Y_scenarios_denorm = np.load(RESULTS / "Y_val_sample0_scenarios_denorm.npy")  # (20, 24, 2)
    Y_val_norm         = np.load(PROC    / "Y_val_norm.npy")
    stats              = np.load(PROC    / "diffusion_normalization_stats.npz")

    y_mean = stats["y_mean"].astype(np.float32)
    y_std  = stats["y_std"].astype(np.float32)

    # Denormalise true target for sample 0
    y_true_denorm = denorm(Y_val_norm[0], y_mean, y_std)   # (24, 2)

    n_scenarios        = Y_scenarios_denorm.shape[0]
    scenarios_wind     = Y_scenarios_denorm[:, :, 0]   # (20, 24)
    scenarios_load     = Y_scenarios_denorm[:, :, 1]   # (20, 24)
    true_wind          = y_true_denorm[:, 0]            # (24,)
    true_load          = y_true_denorm[:, 1]            # (24,)

    # -----------------------------------------------------------------------
    # Figure 1 — Wind
    # -----------------------------------------------------------------------
    wind_path = FIGURES / "diffusion_20_scenarios_wind.png"
    save_scenario_figure(
        HOURS, scenarios_wind, true_wind,
        var_label = "wind generation (MW)",
        title     = f"Diffusion: {n_scenarios} Scenarios vs True — Wind Generation",
        out_path  = wind_path,
    )

    # -----------------------------------------------------------------------
    # Figure 2 — Load
    # -----------------------------------------------------------------------
    load_path = FIGURES / "diffusion_20_scenarios_load.png"
    save_scenario_figure(
        HOURS, scenarios_load, true_load,
        var_label = "electricity load (MW)",
        title     = f"Diffusion: {n_scenarios} Scenarios vs True — Electricity Load",
        out_path  = load_path,
    )

    # -----------------------------------------------------------------------
    # Print summary
    # -----------------------------------------------------------------------
    print(f"Saved: {wind_path}")
    print(f"Saved: {load_path}")
    print(f"\nScenarios plotted : {n_scenarios}")

    print(f"\nTrue wind  min: {true_wind.min():>10,.1f} MW   "
          f"max: {true_wind.max():>10,.1f} MW")
    print(f"True load  min: {true_load.min():>10,.1f} MW   "
          f"max: {true_load.max():>10,.1f} MW")

    print(f"\nFirst forecast hour:")
    print(f"  Wind  — ensemble mean: {scenarios_wind[:, 0].mean():>10,.1f} MW   "
          f"true: {true_wind[0]:>10,.1f} MW")
    print(f"  Load  — ensemble mean: {scenarios_load[:, 0].mean():>10,.1f} MW   "
          f"true: {true_load[0]:>10,.1f} MW")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
