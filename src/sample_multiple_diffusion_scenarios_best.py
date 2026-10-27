"""
sample_multiple_diffusion_scenarios_best.py
--------------------------------------------
Step 26 of the Syngrid data pipeline.

Re-runs the multi-scenario sampler using the best 20-epoch checkpoint
(lowest validation loss) instead of the 5-epoch checkpoint.

Goal: verify that longer training reduced the excessive wind spread seen
in the earlier sampling run.
"""

import sys
from pathlib import Path

import numpy as np
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

OUT_NORM   = RESULTS / "Y_val_sample0_scenarios_best_norm.npy"
OUT_DENORM = RESULTS / "Y_val_sample0_scenarios_best_denorm.npy"


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
    """Run one full DDPM reverse chain from independent Gaussian noise."""
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
        PROC / "X_val_norm.npy",
        PROC / "Y_val_norm.npy",
        PROC / "diffusion_normalization_stats.npz",
        CKPT_PATH,
    ):
        if not p.exists():
            print(f"ERROR: File not found at '{p}'.")
            return 1

    device = get_device()

    X_val  = np.load(PROC / "X_val_norm.npy")
    Y_val  = np.load(PROC / "Y_val_norm.npy")
    stats  = np.load(PROC / "diffusion_normalization_stats.npz")
    y_mean = stats["y_mean"].astype(np.float32)   # (2,)
    y_std  = stats["y_std"].astype(np.float32)    # (2,)

    x_context = torch.from_numpy(
        X_val[CONTEXT_IDX : CONTEXT_IDX + 1]
    ).float().to(device)   # (1, 48, 5)

    y_true_denorm = Y_val[CONTEXT_IDX] * y_std + y_mean   # (24, 2)

    # Load best checkpoint
    model = DiffusionForecastModel().to(device)
    ckpt  = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    ckpt_epoch    = ckpt.get("epoch",     "?")
    ckpt_val_loss = ckpt.get("val_loss",  float("nan"))
    print(f"Checkpoint epoch   : {ckpt_epoch}")
    print(f"Checkpoint val loss: {ckpt_val_loss:.6f}")
    print(f"Device             : {device}")
    print(f"N_SCENARIOS        : {N_SCENARIOS}")
    print(f"Context idx        : {CONTEXT_IDX}")
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

    # Diversity statistics
    wind_d = Y_scenarios_denorm[:, :, 0]   # (20, 24)
    load_d = Y_scenarios_denorm[:, :, 1]   # (20, 24)

    mean_wind_std = float(wind_d.std(axis=0).mean())
    mean_load_std = float(load_d.std(axis=0).mean())

    print(f"\n{'='*60}")
    print(f"Y_scenarios_norm   shape : {Y_scenarios_norm.shape}")
    print(f"Y_scenarios_denorm shape : {Y_scenarios_denorm.shape}")

    print(f"\nDiversity (std across {N_SCENARIOS} scenarios, denormalised):")
    print(f"  Wind mean std across 24 h : {mean_wind_std:>10,.2f} MW")
    print(f"  Load mean std across 24 h : {mean_load_std:>10,.2f} MW")

    print(f"\nScenario global range (denormalised):")
    print(f"  Wind  min: {wind_d.min():>10,.1f} MW   max: {wind_d.max():>10,.1f} MW")
    print(f"  Load  min: {load_d.min():>10,.1f} MW   max: {load_d.max():>10,.1f} MW")

    print(f"\nFirst forecast hour — mean across scenarios:")
    print(f"  Wind  : {wind_d[:, 0].mean():>10,.1f} MW   "
          f"(true: {y_true_denorm[0, 0]:>10,.1f} MW)")
    print(f"  Load  : {load_d[:, 0].mean():>10,.1f} MW   "
          f"(true: {y_true_denorm[0, 1]:>10,.1f} MW)")

    print(f"\nSaved: {OUT_NORM}")
    print(f"Saved: {OUT_DENORM}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
