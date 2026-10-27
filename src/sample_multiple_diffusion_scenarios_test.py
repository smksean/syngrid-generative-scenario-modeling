"""
sample_multiple_diffusion_scenarios_test.py
--------------------------------------------
Step 28 of the Syngrid data pipeline.

Generates 20 independent 24-hour scenarios for test sample 0 using the
best 20-epoch checkpoint, then produces the same quality diagnostics
used on validation to check generalisation to the unseen test set.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_diffusion import DiffusionForecastModel
from src.diffusion_utils import make_diffusion_schedule


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
NUM_TIMESTEPS = 1000
N_SCENARIOS   = 20
CONTEXT_IDX   = 0

PROC      = Path("data/processed")
RESULTS   = Path("results")
CKPT_PATH = Path("checkpoints/diffusion_model_normalized_epoch_best.pt")

OUT_NORM    = RESULTS / "Y_test_sample0_scenarios_best_norm.npy"
OUT_DENORM  = RESULTS / "Y_test_sample0_scenarios_best_denorm.npy"
OUT_CSV     = RESULTS / "scenario_quality_test_sample0.csv"


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Single DDPM reverse pass
# ---------------------------------------------------------------------------

@torch.no_grad()
def ddpm_sample_one(
    model:    torch.nn.Module,
    x:        torch.Tensor,     # (1, 48, 5)
    schedule: dict,
    device:   torch.device,
) -> torch.Tensor:              # (1, 24, 2)
    """One full DDPM reverse chain from independent Gaussian noise."""
    betas    = schedule["betas"].to(device)
    alphas   = schedule["alphas"].to(device)
    sqrt_omc = schedule["sqrt_one_minus_alphas_cumprod"].to(device)

    y = torch.randn(1, 24, 2, device=device)

    for t_idx in reversed(range(NUM_TIMESTEPS)):
        t_tensor   = torch.tensor([t_idx], dtype=torch.long, device=device)
        noise_pred = model(x, y, t_tensor)

        beta_t     = betas[t_idx]
        alpha_t    = alphas[t_idx]
        sqrt_omc_t = sqrt_omc[t_idx]

        y = (1.0 / alpha_t.sqrt()) * (
            y - (beta_t / sqrt_omc_t) * noise_pred
        )
        if t_idx > 0:
            y = y + beta_t.sqrt() * torch.randn_like(y)

    return y   # (1, 24, 2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    for p in (
        PROC / "X_test_norm.npy",
        PROC / "Y_test_norm.npy",
        PROC / "diffusion_normalization_stats.npz",
        CKPT_PATH,
    ):
        if not p.exists():
            print(f"ERROR: File not found at '{p}'.")
            return 1

    device = get_device()

    X_test = np.load(PROC / "X_test_norm.npy")
    Y_test = np.load(PROC / "Y_test_norm.npy")
    stats  = np.load(PROC / "diffusion_normalization_stats.npz")
    y_mean = stats["y_mean"].astype(np.float32)
    y_std  = stats["y_std"].astype(np.float32)

    x_context     = torch.from_numpy(
        X_test[CONTEXT_IDX : CONTEXT_IDX + 1]
    ).float().to(device)                               # (1, 48, 5)
    y_true_denorm = Y_test[CONTEXT_IDX] * y_std + y_mean   # (24, 2)

    model = DiffusionForecastModel().to(device)
    ckpt  = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    ckpt_epoch    = ckpt.get("epoch",    "?")
    ckpt_val_loss = ckpt.get("val_loss", float("nan"))
    print(f"Checkpoint epoch   : {ckpt_epoch}")
    print(f"Checkpoint val loss: {ckpt_val_loss:.6f}")
    print(f"Device             : {device}")
    print(f"N_SCENARIOS        : {N_SCENARIOS}")
    print(f"Context            : test sample {CONTEXT_IDX}")
    print(f"\nRunning {N_SCENARIOS} reverse chains ({NUM_TIMESTEPS} steps each)...")

    schedule = make_diffusion_schedule(num_timesteps=NUM_TIMESTEPS)

    scenarios_norm = []
    for _ in range(N_SCENARIOS):
        y_s = ddpm_sample_one(model, x_context, schedule, device)
        scenarios_norm.append(y_s.cpu().numpy()[0])   # (24, 2)

    Y_scenarios_norm   = np.stack(scenarios_norm, axis=0)      # (20, 24, 2)
    Y_scenarios_denorm = Y_scenarios_norm * y_std + y_mean     # (20, 24, 2)

    RESULTS.mkdir(parents=True, exist_ok=True)
    np.save(OUT_NORM,   Y_scenarios_norm)
    np.save(OUT_DENORM, Y_scenarios_denorm)

    wind_d = Y_scenarios_denorm[:, :, 0]   # (20, 24)
    load_d = Y_scenarios_denorm[:, :, 1]   # (20, 24)

    wind_mean_std = float(wind_d.std(axis=0).mean())
    load_mean_std = float(load_d.std(axis=0).mean())
    wind_fh_mean  = float(wind_d[:, 0].mean())
    load_fh_mean  = float(load_d[:, 0].mean())
    wind_fh_true  = float(y_true_denorm[0, 0])
    load_fh_true  = float(y_true_denorm[0, 1])
    neg_wind      = int((wind_d < 0).sum())
    neg_load      = int((load_d < 0).sum())

    print(f"\n{'='*60}")
    print(f"Y_scenarios_norm   shape : {Y_scenarios_norm.shape}")
    print(f"Y_scenarios_denorm shape : {Y_scenarios_denorm.shape}")

    print(f"\nDiversity (std across {N_SCENARIOS} scenarios, denormalised):")
    print(f"  Wind mean std across 24 h : {wind_mean_std:>10,.2f} MW")
    print(f"  Load mean std across 24 h : {load_mean_std:>10,.2f} MW")

    print(f"\nScenario global range (denormalised):")
    print(f"  Wind  min: {wind_d.min():>10,.1f} MW   max: {wind_d.max():>10,.1f} MW")
    print(f"  Load  min: {load_d.min():>10,.1f} MW   max: {load_d.max():>10,.1f} MW")

    print(f"\nFirst forecast hour:")
    print(f"  Wind  ensemble mean: {wind_fh_mean:>10,.1f} MW   "
          f"true: {wind_fh_true:>10,.1f} MW   "
          f"bias: {wind_fh_mean - wind_fh_true:>+10,.1f} MW")
    print(f"  Load  ensemble mean: {load_fh_mean:>10,.1f} MW   "
          f"true: {load_fh_true:>10,.1f} MW   "
          f"bias: {load_fh_mean - load_fh_true:>+10,.1f} MW")

    print(f"\nFeasibility:")
    print(f"  Negative wind entries : {neg_wind}")
    print(f"  Negative load entries : {neg_load}")

    row = {
        "sample_id":                    CONTEXT_IDX,
        "n_scenarios":                  N_SCENARIOS,
        "horizon":                      24,
        "wind_mean_std":                round(wind_mean_std, 2),
        "load_mean_std":                round(load_mean_std, 2),
        "wind_min":                     round(float(wind_d.min()), 2),
        "wind_max":                     round(float(wind_d.max()), 2),
        "load_min":                     round(float(load_d.min()), 2),
        "load_max":                     round(float(load_d.max()), 2),
        "wind_first_hour_ensemble_mean": round(wind_fh_mean, 2),
        "wind_first_hour_true":          round(wind_fh_true, 2),
        "wind_first_hour_bias":          round(wind_fh_mean - wind_fh_true, 2),
        "load_first_hour_ensemble_mean": round(load_fh_mean, 2),
        "load_first_hour_true":          round(load_fh_true, 2),
        "load_first_hour_bias":          round(load_fh_mean - load_fh_true, 2),
        "neg_wind":                      neg_wind,
        "neg_load":                      neg_load,
    }
    pd.DataFrame([row]).to_csv(OUT_CSV, index=False)

    print(f"\nSaved: {OUT_NORM}")
    print(f"Saved: {OUT_DENORM}")
    print(f"Saved: {OUT_CSV}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
