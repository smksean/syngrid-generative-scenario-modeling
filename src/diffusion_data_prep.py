"""
diffusion_data_prep.py
----------------------
Step 11 of the Syngrid data pipeline.

Builds the joint future target tensor Y by stacking y_wind and y_load along
the last dimension, producing shape (N, 24, 2) per split.

  Y[:, :, 0] = wind
  Y[:, :, 1] = load

X stays unchanged at (N, 48, 5).
"""

from pathlib import Path

import numpy as np


PROC = Path("data/processed")


def load_split(tag: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X      = np.load(PROC / f"X_{tag}.npy")
    y_load = np.load(PROC / f"y_load_{tag}.npy")
    y_wind = np.load(PROC / f"y_wind_{tag}.npy")
    return X, y_load, y_wind


def build_Y(y_wind: np.ndarray, y_load: np.ndarray) -> np.ndarray:
    """Stack wind and load along a new last axis → (N, 24, 2)."""
    return np.stack([y_wind, y_load], axis=-1)


def main() -> int:
    for tag in ("train", "val", "test"):
        for stem in (f"X_{tag}", f"y_load_{tag}", f"y_wind_{tag}"):
            p = PROC / f"{stem}.npy"
            if not p.exists():
                print(f"ERROR: File not found at '{p}'.")
                print("Run src/split_supervised_windows.py first.")
                return 1

    X_train, y_load_train, y_wind_train = load_split("train")
    X_val,   y_load_val,   y_wind_val   = load_split("val")
    X_test,  y_load_test,  y_wind_test  = load_split("test")

    Y_train = build_Y(y_wind_train, y_load_train)
    Y_val   = build_Y(y_wind_val,   y_load_val)
    Y_test  = build_Y(y_wind_test,  y_load_test)

    np.save(PROC / "Y_train.npy", Y_train)
    np.save(PROC / "Y_val.npy",   Y_val)
    np.save(PROC / "Y_test.npy",  Y_test)

    print(f"Train  —  X: {X_train.shape}   Y: {Y_train.shape}")
    print(f"Val    —  X: {X_val.shape}     Y: {Y_val.shape}")
    print(f"Test   —  X: {X_test.shape}    Y: {Y_test.shape}")

    print(f"\nVerification (first training sample):")
    print(f"  y_wind_train[0] shape : {y_wind_train[0].shape}")
    print(f"  y_load_train[0] shape : {y_load_train[0].shape}")
    print(f"  Y_train[0]      shape : {Y_train[0].shape}")
    print(f"  Y_train[0, :3]        :")
    print(f"    wind | load")
    for t in range(3):
        print(f"    {Y_train[0, t, 0]:>10.1f} | {Y_train[0, t, 1]:>10.1f}")

    print(f"\nSaved Y_train, Y_val, Y_test to {PROC}/")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
