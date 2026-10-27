"""
sample_diffusion_normalized.py
------------------------------
Step 19 of the Syngrid data pipeline.

Runs the DDPM reverse diffusion loop to generate sampled 24-hour future
trajectories from the trained normalised model.

Reverse update (Ho et al., 2020):
  y_{t-1} = 1/√α_t · (y_t − β_t/√(1−ᾱ_t) · ε_θ(y_t, x, t))
            + σ_t · z,    z ~ N(0,I)  [z = 0 at t = 0]

where σ_t = √β_t  (simplest DDPM variance choice).
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_diffusion import DiffusionForecastModel
from src.diffusion_utils import make_diffusion_schedule


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROC       = Path("data/processed")
RESULTS    = Path("results")
CKPT_PATH  = Path("checkpoints/diffusion_model_normalized_epoch_last.pt")

NUM_TIMESTEPS = 1000
N_SAMPLES     = 8       # number of validation contexts to sample


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
# DDPM reverse sampler
# ---------------------------------------------------------------------------

@torch.no_grad()
def ddpm_sample(
    model:    torch.nn.Module,
    x:        torch.Tensor,         # (B, 48, 5)  conditioning context
    schedule: dict,
    device:   torch.device,
) -> torch.Tensor:                  # (B, 24, 2)  sampled future
    """
    Iterates the DDPM reverse chain from t = T-1 down to t = 0.
    Returns a single sampled trajectory per context window.
    """
    B = x.shape[0]
    betas     = schedule["betas"].to(device)
    alphas    = schedule["alphas"].to(device)
    alphas_cp = schedule["alphas_cumprod"].to(device)
    sqrt_omc  = schedule["sqrt_one_minus_alphas_cumprod"].to(device)

    # Start from pure Gaussian noise
    y = torch.randn(B, 24, 2, device=device)

    for t_idx in reversed(range(NUM_TIMESTEPS)):
        t_tensor = torch.full((B,), t_idx, dtype=torch.long, device=device)

        # Predict noise
        noise_pred = model(x, y, t_tensor)  # (B, 24, 2)

        # Scalar schedule values for this step, shaped (B, 1, 1) for broadcasting
        beta_t    = betas[t_idx]
        alpha_t   = alphas[t_idx]
        sqrt_omc_t = sqrt_omc[t_idx]

        # Reverse update: mean component
        y = (1.0 / alpha_t.sqrt()) * (
            y - (beta_t / sqrt_omc_t) * noise_pred
        )

        # Add noise for all steps except the final one (t > 0)
        if t_idx > 0:
            sigma_t = beta_t.sqrt()
            y = y + sigma_t * torch.randn_like(y)

    return y     # (B, 24, 2) in normalised space


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

    # Load validation data
    X_val = np.load(PROC / "X_val_norm.npy")
    Y_val = np.load(PROC / "Y_val_norm.npy")

    # Load normalisation statistics
    stats  = np.load(PROC / "diffusion_normalization_stats.npz")
    y_mean = stats["y_mean"].astype(np.float32)   # (2,)
    y_std  = stats["y_std"].astype(np.float32)    # (2,)

    # Select first N_SAMPLES validation contexts
    x_context = torch.from_numpy(X_val[:N_SAMPLES]).float().to(device)  # (8, 48, 5)

    # Recreate model and load checkpoint
    model = DiffusionForecastModel().to(device)
    ckpt  = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} "
          f"(val loss {ckpt['val_loss']:.6f})")

    schedule = make_diffusion_schedule(num_timesteps=NUM_TIMESTEPS)

    print(f"Device              : {device}")
    print(f"Running reverse diffusion chain ({NUM_TIMESTEPS} steps)...")

    Y_sampled_norm_t = ddpm_sample(model, x_context, schedule, device)
    Y_sampled_norm   = Y_sampled_norm_t.cpu().numpy()   # (8, 24, 2)

    # Denormalise:  y_raw = y_norm * std + mean
    Y_sampled_denorm = Y_sampled_norm * y_std + y_mean  # broadcast (2,)

    # Save
    RESULTS.mkdir(parents=True, exist_ok=True)
    np.save(RESULTS / "Y_sampled_val_norm.npy",   Y_sampled_norm)
    np.save(RESULTS / "Y_sampled_val_denorm.npy", Y_sampled_denorm)

    # Print summary
    print(f"\nx_context shape              : {tuple(x_context.shape)}")
    print(f"Y_sampled_norm   shape       : {Y_sampled_norm.shape}")
    print(f"Y_sampled_denorm shape       : {Y_sampled_denorm.shape}")
    print(f"Y_val[0] shape (reference)   : {Y_val[0].shape}")

    print(f"\nFirst sample, first forecast hour:")
    print(f"  normalised   [wind, load] : {Y_sampled_norm[0, 0, :]}")
    print(f"  denormalised [wind, load] : {Y_sampled_denorm[0, 0, :]}")
    print(f"  true val     [wind, load] : {Y_val[0, 0, :]}  (normalised reference)")

    print(f"\nSaved: {RESULTS / 'Y_sampled_val_norm.npy'}")
    print(f"Saved: {RESULTS / 'Y_sampled_val_denorm.npy'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
