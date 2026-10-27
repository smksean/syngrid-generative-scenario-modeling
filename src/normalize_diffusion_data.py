"""
normalize_diffusion_data.py
---------------------------
Step 17 of the Syngrid data pipeline.

Computes z-score normalisation statistics from the training split and
applies them to train / val / test arrays.

Statistics are computed per feature / target channel across all samples
and all time steps, so the axes averaged over are (samples, time):

  X : (N, 48, 5)  → x_mean / x_std shape (5,)
  Y : (N, 24, 2)  → y_mean / y_std shape (2,)

Standard score:  z = (x − μ) / σ
If σ = 0 for any channel it is replaced with 1.0 to avoid division by zero.
"""

from pathlib import Path

import numpy as np


PROC = Path("data/processed")
STATS_PATH = PROC / "diffusion_normalization_stats.npz"


def load_split(tag: str) -> tuple[np.ndarray, np.ndarray]:
    X = np.load(PROC / f"X_{tag}.npy")
    Y = np.load(PROC / f"Y_{tag}.npy")
    return X, Y


def compute_stats(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-channel mean and std over all samples and time steps.

    arr shape : (N, T, C)
    Returns   : mean (C,), std (C,)
    """
    mean = arr.mean(axis=(0, 1))    # (C,)
    std  = arr.std(axis=(0, 1))     # (C,)
    std  = np.where(std == 0.0, 1.0, std)   # safe division
    return mean, std


def normalize(arr: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Broadcast (C,) stats over (N, T, C) and apply z-score."""
    return (arr - mean) / std


def main() -> int:
    for tag in ("train", "val", "test"):
        for stem in (f"X_{tag}", f"Y_{tag}"):
            p = PROC / f"{stem}.npy"
            if not p.exists():
                print(f"ERROR: File not found at '{p}'.")
                print("Run src/diffusion_data_prep.py first.")
                return 1

    X_train, Y_train = load_split("train")
    X_val,   Y_val   = load_split("val")
    X_test,  Y_test  = load_split("test")

    x_mean, x_std = compute_stats(X_train)
    y_mean, y_std = compute_stats(Y_train)

    print(f"x_mean.shape : {x_mean.shape}   x_std.shape : {x_std.shape}")
    print(f"y_mean.shape : {y_mean.shape}   y_std.shape : {y_std.shape}")
    print(f"\nx_mean : {x_mean}")
    print(f"x_std  : {x_std}")
    print(f"y_mean : {y_mean}")
    print(f"y_std  : {y_std}")

    # Normalise
    X_train_n = normalize(X_train, x_mean, x_std)
    Y_train_n = normalize(Y_train, y_mean, y_std)
    X_val_n   = normalize(X_val,   x_mean, x_std)
    Y_val_n   = normalize(Y_val,   y_mean, y_std)
    X_test_n  = normalize(X_test,  x_mean, x_std)
    Y_test_n  = normalize(Y_test,  y_mean, y_std)

    # Save normalised arrays
    for tag, Xn, Yn in (
        ("train", X_train_n, Y_train_n),
        ("val",   X_val_n,   Y_val_n),
        ("test",  X_test_n,  Y_test_n),
    ):
        np.save(PROC / f"X_{tag}_norm.npy", Xn)
        np.save(PROC / f"Y_{tag}_norm.npy", Yn)

    # Save statistics
    np.savez(STATS_PATH, x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std)

    # Summary
    print(f"\n{'='*60}")
    for tag, X_raw, Y_raw, Xn, Yn in (
        ("train", X_train, Y_train, X_train_n, Y_train_n),
        ("val",   X_val,   Y_val,   X_val_n,   Y_val_n),
        ("test",  X_test,  Y_test,  X_test_n,  Y_test_n),
    ):
        print(f"{tag:<5}  X: {X_raw.shape} → {Xn.shape}   "
              f"Y: {Y_raw.shape} → {Yn.shape}")
    print(f"{'='*60}")

    print(f"\nFirst look-back hour  X_train[0, 0, :]")
    print(f"  raw  : {X_train[0, 0, :]}")
    print(f"  norm : {X_train_n[0, 0, :]}")

    print(f"\nFirst forecast hour   Y_train[0, 0, :]   [wind, load]")
    print(f"  raw  : {Y_train[0, 0, :]}")
    print(f"  norm : {Y_train_n[0, 0, :]}")

    print(f"\nStats saved : {STATS_PATH}")
    print("Normalised arrays saved to data/processed/")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
