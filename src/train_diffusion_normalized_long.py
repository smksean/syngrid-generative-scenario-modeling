"""
train_diffusion_normalized_long.py
-----------------------------------
Step 25 of the Syngrid data pipeline.

Extended normalised DDPM training run (20 epochs) with best-checkpoint
tracking. Identical architecture and loss to the 5-epoch run; the only
change is more training time and saving the epoch with the lowest val loss.
"""

import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_diffusion import DiffusionForecastModel
from src.diffusion_utils import make_diffusion_schedule, q_sample


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
NUM_TIMESTEPS = 1000
NUM_EPOCHS    = 20
LEARNING_RATE = 1e-3
BATCH_SIZE    = 64

PROC      = Path("data/processed")
RESULTS   = Path("results")
CKPT_DIR  = Path("checkpoints")

HISTORY_CSV  = RESULTS  / "diffusion_training_history_normalized_long.csv"
CKPT_LAST    = CKPT_DIR / "diffusion_model_normalized_epoch_last_long.pt"
CKPT_BEST    = CKPT_DIR / "diffusion_model_normalized_epoch_best.pt"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class NormalisedWindowDataset(Dataset):
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
# Train / evaluate
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
        x, y = x.to(device), y.to(device)
        t = torch.randint(0, NUM_TIMESTEPS, (x.shape[0],), device=device)
        y_noisy, noise = q_sample(y, t, sqrt_ac, sqrt_omc)
        loss = torch.mean((model(x, y_noisy, t) - noise) ** 2)
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
        x, y = x.to(device), y.to(device)
        t = torch.randint(0, NUM_TIMESTEPS, (x.shape[0],), device=device)
        y_noisy, noise = q_sample(y, t, sqrt_ac, sqrt_omc)
        total_loss += torch.mean((model(x, y_noisy, t) - noise) ** 2).item()

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

    train_loader = DataLoader(
        NormalisedWindowDataset(X_train, Y_train),
        batch_size=BATCH_SIZE, shuffle=True,
    )
    val_loader = DataLoader(
        NormalisedWindowDataset(X_val, Y_val),
        batch_size=BATCH_SIZE, shuffle=False,
    )

    device    = get_device()
    model     = DiffusionForecastModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    schedule  = make_diffusion_schedule(num_timesteps=NUM_TIMESTEPS)

    print(f"Device             : {device}")
    print(f"Train dataset size : {len(train_loader.dataset):,}")
    print(f"Val   dataset size : {len(val_loader.dataset):,}")
    print(f"Train batches      : {len(train_loader):,}")
    print(f"Val   batches      : {len(val_loader):,}")
    print(f"Model parameters   : {sum(p.numel() for p in model.parameters()):,}")
    print(f"\nStarting training for {NUM_EPOCHS} epochs...\n")

    history        = []
    best_val_loss  = float("inf")
    best_epoch     = -1
    best_state     = None

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, schedule, device)
        val_loss   = evaluate(model, val_loader, schedule, device)

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_epoch    = epoch
            best_state    = copy.deepcopy({
                "epoch":                epoch,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss":           train_loss,
                "val_loss":             val_loss,
            })

        marker = "  ← best" if is_best else ""
        print(f"Epoch {epoch:>2}/{NUM_EPOCHS}  |  "
              f"train: {train_loss:.6f}  |  "
              f"val: {val_loss:.6f}  |  "
              f"best val: {best_val_loss:.6f}{marker}")

        history.append({
            "epoch":      epoch,
            "train_loss": round(train_loss, 8),
            "val_loss":   round(val_loss,   8),
            "is_best":    is_best,
        })

    # Save
    RESULTS.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(history).to_csv(HISTORY_CSV, index=False)

    torch.save({
        "epoch":                epoch,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss":           train_loss,
        "val_loss":             val_loss,
    }, CKPT_LAST)

    torch.save(best_state, CKPT_BEST)

    print(f"\n{'='*60}")
    print(f"Best epoch         : {best_epoch}")
    print(f"Best val loss      : {best_val_loss:.6f}")
    print(f"History saved      : {HISTORY_CSV}")
    print(f"Last checkpoint    : {CKPT_LAST}")
    print(f"Best checkpoint    : {CKPT_BEST}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
