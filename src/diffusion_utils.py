"""
diffusion_utils.py
------------------
Step 15 of the Syngrid data pipeline.

Core DDPM noise-schedule utilities for the forward diffusion process.

Terminology follows Ho et al. (DDPM, 2020):
  betas                        β_t   — noise variance schedule
  alphas                       α_t = 1 − β_t
  alphas_cumprod               ᾱ_t = ∏_{s=1}^{t} α_s
  sqrt_alphas_cumprod          √ᾱ_t
  sqrt_one_minus_alphas_cumprod √(1 − ᾱ_t)

Forward noising (q_sample):
  y_noisy = √ᾱ_t · y_start + √(1 − ᾱ_t) · ε,   ε ~ N(0, I)
"""

from typing import Optional

import torch


# ---------------------------------------------------------------------------
# Beta schedule
# ---------------------------------------------------------------------------

def make_beta_schedule(
    num_timesteps: int,
    beta_start: float = 1e-4,
    beta_end:   float = 2e-2,
) -> torch.Tensor:
    """
    Linear beta schedule from beta_start to beta_end.

    Returns
    -------
    betas : (num_timesteps,) float32
    """
    return torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Full schedule
# ---------------------------------------------------------------------------

def make_diffusion_schedule(
    num_timesteps: int = 1000,
    beta_start:    float = 1e-4,
    beta_end:      float = 2e-2,
) -> dict[str, torch.Tensor]:
    """
    Compute and return the standard DDPM schedule tensors.

    Returns a dict with keys:
      betas, alphas, alphas_cumprod,
      sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod
    All tensors have shape (num_timesteps,) and dtype float32.
    """
    betas        = make_beta_schedule(num_timesteps, beta_start, beta_end)
    alphas       = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    return {
        "betas":                        betas,
        "alphas":                       alphas,
        "alphas_cumprod":               alphas_cumprod,
        "sqrt_alphas_cumprod":          torch.sqrt(alphas_cumprod),
        "sqrt_one_minus_alphas_cumprod": torch.sqrt(1.0 - alphas_cumprod),
    }


# ---------------------------------------------------------------------------
# Forward noising: q(y_t | y_0)
# ---------------------------------------------------------------------------

def _gather(tensor: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Gather schedule scalars for each batch element and reshape for
    broadcasting over (batch_size, 24, 2).

    Parameters
    ----------
    tensor : (T,)     1-D schedule tensor
    t      : (B,)     integer timestep indices

    Returns
    -------
    out    : (B, 1, 1) so it broadcasts over (B, 24, 2)
    """
    return tensor[t].reshape(-1, 1, 1)


def q_sample(
    y_start:                       torch.Tensor,   # (B, 24, 2)
    t:                             torch.Tensor,   # (B,)
    sqrt_alphas_cumprod:           torch.Tensor,   # (T,)
    sqrt_one_minus_alphas_cumprod: torch.Tensor,   # (T,)
    noise:                         Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample y_t from the forward diffusion kernel q(y_t | y_0).

    y_t = √ᾱ_t · y_0 + √(1 − ᾱ_t) · ε,   ε ~ N(0, I)

    Parameters
    ----------
    y_start                       : (B, 24, 2) clean future targets
    t                             : (B,)       integer timestep indices in [0, T)
    sqrt_alphas_cumprod           : (T,)       from make_diffusion_schedule
    sqrt_one_minus_alphas_cumprod : (T,)       from make_diffusion_schedule
    noise                         : (B, 24, 2) optional; sampled internally if None

    Returns
    -------
    y_noisy : (B, 24, 2)  noisy target at timestep t
    noise   : (B, 24, 2)  the Gaussian noise that was added
    """
    if noise is None:
        noise = torch.randn_like(y_start)

    sqrt_ac     = _gather(sqrt_alphas_cumprod,           t)   # (B, 1, 1)
    sqrt_one_mc = _gather(sqrt_one_minus_alphas_cumprod, t)   # (B, 1, 1)

    y_noisy = sqrt_ac * y_start + sqrt_one_mc * noise

    return y_noisy, noise


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------

def main() -> int:
    schedule = make_diffusion_schedule(num_timesteps=1000)

    betas               = schedule["betas"]
    alphas_cumprod      = schedule["alphas_cumprod"]
    sqrt_ac             = schedule["sqrt_alphas_cumprod"]
    sqrt_one_mc         = schedule["sqrt_one_minus_alphas_cumprod"]

    B = 64
    y_start = torch.randn(B, 24, 2)
    t       = torch.randint(0, 1000, (B,))

    y_noisy, noise = q_sample(y_start, t, sqrt_ac, sqrt_one_mc)

    print(f"betas.shape            : {tuple(betas.shape)}")
    print(f"alphas_cumprod.shape   : {tuple(alphas_cumprod.shape)}")
    print(f"y_start.shape          : {tuple(y_start.shape)}")
    print(f"t.shape                : {tuple(t.shape)}")
    print(f"y_noisy.shape          : {tuple(y_noisy.shape)}")
    print(f"noise.shape            : {tuple(noise.shape)}")
    print(f"y_noisy.dtype          : {y_noisy.dtype}")

    # Sanity: at t=0, y_noisy ≈ y_start (ᾱ_0 ≈ 1, √(1-ᾱ_0) ≈ 0)
    t_zero = torch.zeros(1, dtype=torch.long)
    y_noisy_0, _ = q_sample(y_start[:1], t_zero, sqrt_ac, sqrt_one_mc,
                             noise=torch.zeros_like(y_start[:1]))
    max_diff = (y_noisy_0 - y_start[:1]).abs().max().item()
    print(f"\nSanity (t=0, zero noise) max|y_noisy - y_start| : {max_diff:.6f}")

    # Sanity: at t=T-1, signal should be heavily corrupted (ᾱ_{T-1} ≈ 0)
    t_end = torch.full((1,), 999, dtype=torch.long)
    ac_end = alphas_cumprod[999].item()
    print(f"ᾱ at t=999 (should be close to 0)              : {ac_end:.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
