"""
dataloader_diffusion.py
-----------------------
Step 13 of the Syngrid data pipeline.

Wraps the DiffusionTimeSeriesDataset in PyTorch DataLoaders ready for
diffusion-model training loops.

Batch shapes:
  x : (batch_size, 48, 5)  float32 — conditioning look-back window
  y : (batch_size, 24, 2)  float32 — joint future target (wind, load)
"""

import sys
from pathlib import Path

# Allow direct `python src/dataloader_diffusion.py` execution from project root.
sys.path.insert(0, str(Path(__file__).parent.parent))

from torch.utils.data import DataLoader

from src.dataset_diffusion import load_dataset


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_BATCH_SIZE = 64


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_train_loader(batch_size: int = DEFAULT_BATCH_SIZE) -> DataLoader:
    """DataLoader for the training split (shuffled)."""
    return DataLoader(
        load_dataset("train"),
        batch_size=batch_size,
        shuffle=True,
    )


def get_val_loader(batch_size: int = DEFAULT_BATCH_SIZE) -> DataLoader:
    """DataLoader for the validation split (ordered)."""
    return DataLoader(
        load_dataset("val"),
        batch_size=batch_size,
        shuffle=False,
    )


def get_test_loader(batch_size: int = DEFAULT_BATCH_SIZE) -> DataLoader:
    """DataLoader for the test split (ordered)."""
    return DataLoader(
        load_dataset("test"),
        batch_size=batch_size,
        shuffle=False,
    )


def get_all_loaders(
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Convenience function: return (train_loader, val_loader, test_loader)."""
    return (
        get_train_loader(batch_size),
        get_val_loader(batch_size),
        get_test_loader(batch_size),
    )


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------

def main() -> int:
    train_loader, val_loader, test_loader = get_all_loaders()

    print(f"Batch size        : {DEFAULT_BATCH_SIZE}")
    print(f"Batches in train  : {len(train_loader):,}")
    print(f"Batches in val    : {len(val_loader):,}")
    print(f"Batches in test   : {len(test_loader):,}")

    x_batch, y_batch = next(iter(train_loader))

    print(f"\nFirst training batch:")
    print(f"  x_batch.shape : {tuple(x_batch.shape)}")
    print(f"  y_batch.shape : {tuple(y_batch.shape)}")
    print(f"  x_batch.dtype : {x_batch.dtype}")
    print(f"  y_batch.dtype : {y_batch.dtype}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
