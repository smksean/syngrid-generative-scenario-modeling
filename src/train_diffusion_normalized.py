"""
train_diffusion_normalized.py
------------------------------
Step 18 of the Syngrid data pipeline.

Same DDPM training loop as train_diffusion.py, but operating on
z-score-normalised arrays so the noise prediction loss is on a
unit-scale target (mean ≈ 0, std ≈ 1) rather than raw MW.

Expected outcome: train/val MSE should fall clearly below 1.0 as the
model learns to predict Gaussian noise on the normalised future targets.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_diffusion  import DiffusionForecastModel
from src.diffusion_utils  import make_diffusion_schedule, q_sample


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
NUM_TIMESTEPS = 1000
NUM_EPOCHS    = 5
LEARNING_RATE = 1e-3
BATCH_SIZE    = 64

PROC       = Path("data/processed")
RESULTS    = Path("results")
CKPT_DIR   = Path("checkpoints")
HISTORY_CSV = RESULTS  / "diffusion_training_history_normalized.csv"
CKPT_PATH   = CKPT_DIR / "diffusion_model_normalized_epoch_last.pt"


# ---------------------------------------------------------------------------
# Minimal local Dataset
# ---------------------------------------------------------------------------

class NormalisedWindowDataset(Dataset):
    """Wraps pre-normalised X (N,48,5) and Y (N,24,2) numpy arrays."""

    def __init__(self, X: np.ndarray, Y: np.ndarray) -> None:
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).float()

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.Y[idx]


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
# Training / evaluation
# ---------------------------------------------------------------------------

def train_one_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    schedule:  dict,
    device:    torch.device,
) -> float:
    model.train()
    total_loss = 0.0

    sqrt_ac  = schedule["sqrt_alphas_cumprod"].to(device)
    sqrt_omc = schedule["sqrt_one_minus_alphas_cumprod"].to(device)

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        t          = torch.randint(0, NUM_TIMESTEPS, (x.shape[0],), device=device)
        y_noisy, noise = q_sample(y, t, sqrt_ac, sqrt_omc)
        noise_pred = model(x, y_noisy, t)

        loss = torch.mean((noise_pred - noise) ** 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(
    model:    nn.Module,
    loader:   DataLoader,
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

        t              = torch.randint(0, NUM_TIMESTEPS, (x.shape[0],), device=device)
        y_noisy, noise = q_sample(y, t, sqrt_ac, sqrt_omc)
        noise_pred     = model(x, y_noisy, t)

        total_loss += torch.mean((noise_pred - noise) ** 2).item()

    return total_loss / len(loader)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    for stem in ("X_train_norm", "Y_train_norm", "X_val_norm", "Y_val_norm"):
        p = PROC / f"{stem}.npy"
        if not p.exists():
            print(f"ERROR: File not found at '{p}'.")
            print("Run src/normalize_diffusion_data.py first.")
            return 1

    X_train = np.load(PROC / "X_train_norm.npy")
    Y_train = np.load(PROC / "Y_train_norm.npy")
    X_val   = np.load(PROC / "X_val_norm.npy")
    Y_val   = np.load(PROC / "Y_val_norm.npy")

    train_ds = NormalisedWindowDataset(X_train, Y_train)
    val_ds   = NormalisedWindowDataset(X_val,   Y_val)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    device   = get_device()
    model    = DiffusionForecastModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    schedule  = make_diffusion_schedule(num_timesteps=NUM_TIMESTEPS)

    print(f"Device             : {device}")
    print(f"Train dataset size : {len(train_ds):,}")
    print(f"Val   dataset size : {len(val_ds):,}")
    print(f"Train batches      : {len(train_loader):,}")
    print(f"Val   batches      : {len(val_loader):,}")
    print(f"Model parameters   : {sum(p.numel() for p in model.parameters()):,}")
    print(f"\nStarting normalised training for {NUM_EPOCHS} epoch(s)...\n")

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

    RESULTS.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(history).to_csv(HISTORY_CSV, index=False)
    print(f"\nHistory saved   : {HISTORY_CSV}")

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
