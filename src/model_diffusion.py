"""
model_diffusion.py
------------------
Step 14 of the Syngrid data pipeline.

Minimal diffusion-model skeleton for multivariate time-series forecasting.
At each denoising step the model is given:
  x        : (B, 48, 5)  — conditioning look-back history
  y_noisy  : (B, 24, 2)  — noisy future target at diffusion step t
  t        : (B,)        — integer diffusion timestep

It predicts the noise added to the future target:
  output   : (B, 24, 2)

Architecture
------------
  1. Flatten x         → (B, 240)
  2. Flatten y_noisy   → (B, 48)
  3. Embed t           → (B, T_EMB)  via sinusoidal + linear projection
  4. Concatenate       → (B, 240 + 48 + T_EMB)
  5. MLP [Linear → SiLU] × 2 + final Linear
  6. Reshape output    → (B, 24, 2)
"""

import sys
import math
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Dimensions
# ---------------------------------------------------------------------------
HISTORY_LEN   = 48
N_FEATURES    = 5
HORIZON       = 24
N_TARGETS     = 2
T_EMB_DIM     = 32
HIDDEN_1      = 256
HIDDEN_2      = 128

HISTORY_FLAT  = HISTORY_LEN * N_FEATURES   # 240
FUTURE_FLAT   = HORIZON * N_TARGETS        # 48
INPUT_DIM     = HISTORY_FLAT + FUTURE_FLAT + T_EMB_DIM   # 320
OUTPUT_FLAT   = HORIZON * N_TARGETS        # 48


# ---------------------------------------------------------------------------
# Sinusoidal timestep embedding
# ---------------------------------------------------------------------------

class SinusoidalTimestepEmbedding(nn.Module):
    """
    Maps integer timestep t → (B, T_EMB_DIM) via sinusoidal encoding
    followed by a learned linear projection.

    Sinusoidal frequencies cover a range suitable for T ≈ 1000 steps.
    """

    def __init__(self, emb_dim: int) -> None:
        super().__init__()
        assert emb_dim % 2 == 0, "emb_dim must be even for sinusoidal encoding."
        self.emb_dim = emb_dim
        self.proj    = nn.Linear(emb_dim, emb_dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t : (B,)  integer or float
        half = self.emb_dim // 2
        freqs = torch.exp(
            -math.log(10_000) * torch.arange(half, device=t.device) / (half - 1)
        )                                          # (half,)
        args  = t.float().unsqueeze(1) * freqs.unsqueeze(0)   # (B, half)
        emb   = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, emb_dim)
        return self.proj(emb)                      # (B, emb_dim)


# ---------------------------------------------------------------------------
# Diffusion forecast model
# ---------------------------------------------------------------------------

class DiffusionForecastModel(nn.Module):
    """
    MLP-based noise predictor for the DDPM denoising objective.

    Parameters
    ----------
    t_emb_dim : int   Timestep embedding dimension (default 32).
    hidden_1  : int   First hidden layer width (default 256).
    hidden_2  : int   Second hidden layer width (default 128).
    """

    def __init__(
        self,
        t_emb_dim: int = T_EMB_DIM,
        hidden_1:  int = HIDDEN_1,
        hidden_2:  int = HIDDEN_2,
    ) -> None:
        super().__init__()

        self.t_emb = SinusoidalTimestepEmbedding(t_emb_dim)

        in_dim = HISTORY_FLAT + FUTURE_FLAT + t_emb_dim

        self.mlp = nn.Sequential(
            nn.Linear(in_dim,  hidden_1), nn.SiLU(),
            nn.Linear(hidden_1, hidden_2), nn.SiLU(),
            nn.Linear(hidden_2, OUTPUT_FLAT),
        )

    def forward(
        self,
        x:       torch.Tensor,   # (B, 48, 5)
        y_noisy: torch.Tensor,   # (B, 24, 2)
        t:       torch.Tensor,   # (B,)
    ) -> torch.Tensor:           # (B, 24, 2)

        B = x.shape[0]

        x_flat = x.reshape(B, -1)           # (B, 240)
        y_flat = y_noisy.reshape(B, -1)     # (B, 48)
        t_emb  = self.t_emb(t)             # (B, T_EMB_DIM)

        h = torch.cat([x_flat, y_flat, t_emb], dim=-1)  # (B, 320)
        out = self.mlp(h)                               # (B, 48)
        return out.reshape(B, HORIZON, N_TARGETS)       # (B, 24, 2)


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------

def main() -> int:
    from src.dataloader_diffusion import get_train_loader

    loader = get_train_loader(batch_size=64)
    x_batch, y_batch = next(iter(loader))

    T_MAX = 1000
    t = torch.randint(0, T_MAX, (x_batch.shape[0],))
    y_noisy = torch.randn_like(y_batch)

    model = DiffusionForecastModel()
    model.eval()

    with torch.no_grad():
        output = model(x_batch, y_noisy, t)

    print(f"x_batch.shape  : {tuple(x_batch.shape)}")
    print(f"y_noisy.shape  : {tuple(y_noisy.shape)}")
    print(f"t.shape        : {tuple(t.shape)}")
    print(f"output.shape   : {tuple(output.shape)}")
    print(f"output.dtype   : {output.dtype}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {total_params:,}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
