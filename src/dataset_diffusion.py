"""
dataset_diffusion.py
--------------------
Step 12 of the Syngrid data pipeline.

Defines a PyTorch Dataset for the diffusion-model pipeline.
Each sample exposes:
  x : (48, 5)  float32 — conditioning look-back window
  y : (24, 2)  float32 — joint future target (wind, load)
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


PROC = Path("data/processed")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DiffusionTimeSeriesDataset(Dataset):
    """
    Wraps pre-built supervised-window arrays as a PyTorch Dataset.

    Parameters
    ----------
    X : np.ndarray  shape (N, 48, 5)
    Y : np.ndarray  shape (N, 24, 2)
    """

    def __init__(self, X: np.ndarray, Y: np.ndarray) -> None:
        assert len(X) == len(Y), "X and Y must have the same number of samples."
        self.X = torch.from_numpy(X).float()   # (N, 48, 5)
        self.Y = torch.from_numpy(Y).float()   # (N, 24, 2)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.Y[idx]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def load_dataset(tag: str) -> "DiffusionTimeSeriesDataset":
    """Load a single split ('train', 'val', or 'test') and return a Dataset."""
    X = np.load(PROC / f"X_{tag}.npy")
    Y = np.load(PROC / f"Y_{tag}.npy")
    return DiffusionTimeSeriesDataset(X, Y)


def load_all_datasets() -> tuple[
    "DiffusionTimeSeriesDataset",
    "DiffusionTimeSeriesDataset",
    "DiffusionTimeSeriesDataset",
]:
    """Convenience function: return (train_ds, val_ds, test_ds)."""
    return load_dataset("train"), load_dataset("val"), load_dataset("test")


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

def main() -> int:
    for tag in ("train", "val", "test"):
        for stem in (f"X_{tag}", f"Y_{tag}"):
            p = PROC / f"{stem}.npy"
            if not p.exists():
                print(f"ERROR: File not found at '{p}'.")
                print("Run src/diffusion_data_prep.py first.")
                return 1

    train_ds, val_ds, test_ds = load_all_datasets()

    print(f"Dataset lengths:")
    print(f"  train : {len(train_ds):,}")
    print(f"  val   : {len(val_ds):,}")
    print(f"  test  : {len(test_ds):,}")

    x0, y0 = train_ds[0]
    print(f"\nFirst training sample:")
    print(f"  x  shape : {tuple(x0.shape)}   dtype : {x0.dtype}")
    print(f"  y  shape : {tuple(y0.shape)}   dtype : {y0.dtype}")
    print(f"\n  x[0]  (first look-back hour)  : {x0[0].tolist()}")
    print(f"  y[0]  (first forecast hour)   : {y0[0].tolist()}   [wind, load]")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
