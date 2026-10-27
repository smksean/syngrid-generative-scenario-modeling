"""
train_diffusion.py
------------------
Step 16 of the Syngrid data pipeline.

Minimal end-to-end DDPM training loop for multivariate 24-hour forecasting.

Objective (Ho et al., 2020):
  L = E_{y_0, ε, t} [ || ε − ε_θ(y_t, x, t) ||² ]

where ε ~ N(0, I) is the true noise and ε_θ is the model prediction.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataloader_diffusion import get_train_loader, get_val_loader
from src.model_diffusion       import DiffusionForecastModel
from src.diffusion_utils       import make_diffusion_schedule, q_sample


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
NUM_TIMESTEPS  = 1000
NUM_EPOCHS     = 5
LEARNING_RATE  = 1e-3
BATCH_SIZE     = 64

RESULTS_DIR    = Path("results")
CKPT_DIR       = Path("checkpoints")
HISTORY_CSV    = RESULTS_DIR / "diffusion_training_history.csv"
CKPT_PATH      = CKPT_DIR    / "diffusion_model_epoch_last.pt"


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
# Training loop
# ---------------------------------------------------------------------------
def train_one_epoch(
    model:       nn.Module,
    loader:      torch.utils.data.DataLoader,
    optimizer:   torch.optim.Optimizer,
    schedule:    dict,
    device:      torch.device,
) -> float:
    model.train()
    total_loss = 0.0

    sqrt_ac    = schedule["sqrt_alphas_cumprod"].to(device)
    sqrt_omc   = schedule["sqrt_one_minus_alphas_cumprod"].to(device)

    for x, y in loader:
        x = x.to(device)   # (B, 48, 5)
        y = y.to(device)   # (B, 24, 2)

        B = x.shape[0]
        t = torch.randint(0, NUM_TIMESTEPS, (B,), device=device)

        y_noisy, noise = q_sample(y, t, sqrt_ac, sqrt_omc)

        noise_pred = model(x, y_noisy, t)   # (B, 24, 2)
        loss = torch.mean((noise_pred - noise) ** 2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(
    model:    nn.Module,
    loader:   torch.utils.data.DataLoader,
    schedule: dict,
    device:   torch.device,
) -> float:
    model.eval()
    total_loss = 0.0

    sqrt_ac  = schedule["sqrt_alphas_cumprod"].to(device)
    sqrt_omc = schedule["sqrt_one_minus_alphas_cumprod"].to(device)

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        B = x.shape[0]
        t = torch.randint(0, NUM_TIMESTEPS, (B,), device=device)

        y_noisy, noise = q_sample(y, t, sqrt_ac, sqrt_omc)
        noise_pred     = model(x, y_noisy, t)
        loss           = torch.mean((noise_pred - noise) ** 2)

        total_loss += loss.item()

    return total_loss / len(loader)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    device = get_device()
    print(f"Device           : {device}")

    train_loader = get_train_loader(batch_size=BATCH_SIZE)
    val_loader   = get_val_loader(batch_size=BATCH_SIZE)
    print(f"Train batches    : {len(train_loader):,}")
    print(f"Val   batches    : {len(val_loader):,}")

    model     = DiffusionForecastModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    schedule  = make_diffusion_schedule(num_timesteps=NUM_TIMESTEPS)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters : {total_params:,}")
    print(f"\nStarting training for {NUM_EPOCHS} epoch(s)...\n")

    history = []

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, schedule, device)
        val_loss   = evaluate(model, val_loader, schedule, device)

        print(f"Epoch {epoch:>2}/{NUM_EPOCHS}  |  "
              f"train loss: {train_loss:.6f}  |  "
              f"val loss: {val_loss:.6f}")

        history.append({
            "epoch":      epoch,
            "train_loss": round(train_loss, 8),
            "val_loss":   round(val_loss,   8),
        })

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(history).to_csv(HISTORY_CSV, index=False)
    print(f"\nHistory saved  : {HISTORY_CSV}")

    torch.save({
        "epoch":                epoch,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss":           train_loss,
        "val_loss":             val_loss,
    }, CKPT_PATH)
    print(f"Checkpoint saved: {CKPT_PATH}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
