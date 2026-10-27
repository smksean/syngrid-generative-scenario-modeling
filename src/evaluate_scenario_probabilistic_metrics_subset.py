"""
evaluate_scenario_probabilistic_metrics_subset.py
-------------------------------------------------
Step 32 of the Syngrid data pipeline.

Generates diffusion scenarios for a small validation/test subset and computes
probabilistic scenario-quality metrics:
  - empirical CRPS for wind and load, averaged over the 24-hour horizon
  - multivariate energy score for the full 24-hour joint wind-load trajectory

This script intentionally does not save raw scenario arrays. It writes one
metrics row per evaluated sample plus a compact printed split summary.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.diffusion_utils import make_diffusion_schedule
from src.model_diffusion import DiffusionForecastModel


PROC = Path("data/processed")
RESULTS = Path("results")
CKPT_PATH = Path("checkpoints/diffusion_model_normalized_epoch_best.pt")
DEFAULT_OUT_CSV = RESULTS / "scenario_probabilistic_metrics_subset.csv"

NUM_TIMESTEPS = 1000
DEFAULT_N_SAMPLES = 10
DEFAULT_N_SCENARIOS = 20
DEFAULT_SEED = 2026

COLS = [
    "split",
    "sample_id",
    "n_scenarios",
    "horizon",
    "wind_crps_mean",
    "load_crps_mean",
    "joint_energy_score",
]


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def ddpm_sample_batch(
    model: torch.nn.Module,
    x_context: torch.Tensor,  # (1, 48, 5)
    schedule: dict,
    device: torch.device,
    n_scenarios: int,
) -> torch.Tensor:            # (n_scenarios, 24, 2)
    """Run a batched DDPM reverse chain for one conditioning context."""
    betas = schedule["betas"].to(device)
    alphas = schedule["alphas"].to(device)
    sqrt_omc = schedule["sqrt_one_minus_alphas_cumprod"].to(device)

    x = x_context.repeat(n_scenarios, 1, 1)
    y = torch.randn(n_scenarios, 24, 2, device=device)

    for t_idx in reversed(range(NUM_TIMESTEPS)):
        t_tensor = torch.full((n_scenarios,), t_idx, dtype=torch.long, device=device)
        noise_pred = model(x, y, t_tensor)

        beta_t = betas[t_idx]
        alpha_t = alphas[t_idx]
        sqrt_omc_t = sqrt_omc[t_idx]

        y = (1.0 / alpha_t.sqrt()) * (
            y - (beta_t / sqrt_omc_t) * noise_pred
        )

        if t_idx > 0:
            y = y + beta_t.sqrt() * torch.randn_like(y)

    return y


def empirical_crps_1d(samples: np.ndarray, truth: np.ndarray) -> float:
    """
    Empirical CRPS for scalar ensemble forecasts.

    samples has shape (n_scenarios, horizon); truth has shape (horizon,).
    """
    abs_error = np.abs(samples - truth[None, :]).mean(axis=0)
    pairwise = np.abs(samples[:, None, :] - samples[None, :, :]).mean(axis=(0, 1))
    crps_by_hour = abs_error - 0.5 * pairwise
    return float(crps_by_hour.mean())


def energy_score(samples: np.ndarray, truth: np.ndarray) -> float:
    """Empirical energy score for 24-hour joint wind-load trajectories."""
    samples_flat = samples.reshape(samples.shape[0], -1)
    truth_flat = truth.reshape(-1)

    dist_to_truth = np.linalg.norm(samples_flat - truth_flat[None, :], axis=1).mean()
    pairwise_diff = samples_flat[:, None, :] - samples_flat[None, :, :]
    pairwise_dist = np.linalg.norm(pairwise_diff, axis=2).mean()

    return float(dist_to_truth - 0.5 * pairwise_dist)


def compute_metrics_row(
    split: str,
    sample_id: int,
    scenarios: np.ndarray,  # (n_scenarios, 24, 2), denormalised
    truth: np.ndarray,      # (24, 2), denormalised
) -> dict:
    wind_crps = empirical_crps_1d(scenarios[:, :, 0], truth[:, 0])
    load_crps = empirical_crps_1d(scenarios[:, :, 1], truth[:, 1])
    joint_es = energy_score(scenarios, truth)

    return {
        "split": split,
        "sample_id": sample_id,
        "n_scenarios": int(scenarios.shape[0]),
        "horizon": int(scenarios.shape[1]),
        "wind_crps_mean": round(wind_crps, 2),
        "load_crps_mean": round(load_crps, 2),
        "joint_energy_score": round(joint_es, 2),
    }


def evaluate_split(
    split: str,
    model: torch.nn.Module,
    schedule: dict,
    device: torch.device,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    n_samples: int,
    n_scenarios: int,
) -> list[dict]:
    x_path = PROC / f"X_{split}_norm.npy"
    y_path = PROC / f"Y_{split}_norm.npy"

    X = np.load(x_path)
    Y = np.load(y_path)

    rows = []
    max_samples = min(n_samples, X.shape[0])

    print(f"\nEvaluating {split}: {max_samples} samples, {n_scenarios} scenarios each")

    for sample_id in range(max_samples):
        x_context = torch.from_numpy(
            X[sample_id : sample_id + 1]
        ).float().to(device)
        truth = Y[sample_id] * y_std + y_mean

        scenarios_norm = ddpm_sample_batch(
            model=model,
            x_context=x_context,
            schedule=schedule,
            device=device,
            n_scenarios=n_scenarios,
        )
        scenarios_norm_np = scenarios_norm.cpu().numpy()
        scenarios_denorm = scenarios_norm_np * y_std + y_mean

        row = compute_metrics_row(
            split=split,
            sample_id=sample_id,
            scenarios=scenarios_denorm,
            truth=truth,
        )
        rows.append(row)

        print(
            f"  sample {sample_id:02d}: "
            f"wind CRPS={row['wind_crps_mean']:>9,.2f}, "
            f"load CRPS={row['load_crps_mean']:>9,.2f}, "
            f"ES={row['joint_energy_score']:>10,.2f}"
        )

    return rows


def print_summary(df: pd.DataFrame) -> None:
    summary = (
        df.groupby("split")[["wind_crps_mean", "load_crps_mean", "joint_energy_score"]]
        .agg(["mean", "std"])
        .round(2)
    )
    print(f"\n{'=' * 60}")
    print("Subset summary:")
    print(summary.to_string())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate diffusion scenario CRPS and energy score on a subset."
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=DEFAULT_N_SAMPLES,
        help=f"Number of validation and test samples to evaluate (default: {DEFAULT_N_SAMPLES}).",
    )
    parser.add_argument(
        "--n-scenarios",
        type=int,
        default=DEFAULT_N_SCENARIOS,
        help=f"Number of scenarios per sample (default: {DEFAULT_N_SCENARIOS}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for scenario sampling (default: {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=DEFAULT_OUT_CSV,
        help=f"Output CSV path (default: {DEFAULT_OUT_CSV}).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.n_samples <= 0:
        print("ERROR: --n-samples must be positive.")
        return 1
    if args.n_scenarios <= 0:
        print("ERROR: --n-scenarios must be positive.")
        return 1

    required = [
        PROC / "X_val_norm.npy",
        PROC / "Y_val_norm.npy",
        PROC / "X_test_norm.npy",
        PROC / "Y_test_norm.npy",
        PROC / "diffusion_normalization_stats.npz",
        CKPT_PATH,
    ]

    for path in required:
        if not path.exists():
            print(f"ERROR: File not found at '{path}'.")
            return 1

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = get_device()
    print(f"Device      : {device}")
    print(f"Checkpoint  : {CKPT_PATH}")
    print(f"N_SAMPLES   : {args.n_samples}")
    print(f"N_SCENARIOS : {args.n_scenarios}")
    print(f"Timesteps   : {NUM_TIMESTEPS}")
    print(f"Seed        : {args.seed}")

    stats = np.load(PROC / "diffusion_normalization_stats.npz")
    y_mean = stats["y_mean"].astype(np.float32)
    y_std = stats["y_std"].astype(np.float32)

    model = DiffusionForecastModel().to(device)
    ckpt = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    ckpt_epoch = ckpt.get("epoch", "?")
    ckpt_val_loss = ckpt.get("val_loss", float("nan"))
    print(f"Checkpoint epoch   : {ckpt_epoch}")
    print(f"Checkpoint val loss: {ckpt_val_loss:.6f}")

    schedule = make_diffusion_schedule(num_timesteps=NUM_TIMESTEPS)

    rows = []
    rows.extend(
        evaluate_split(
            split="val",
            model=model,
            schedule=schedule,
            device=device,
            y_mean=y_mean,
            y_std=y_std,
            n_samples=args.n_samples,
            n_scenarios=args.n_scenarios,
        )
    )
    rows.extend(
        evaluate_split(
            split="test",
            model=model,
            schedule=schedule,
            device=device,
            y_mean=y_mean,
            y_std=y_std,
            n_samples=args.n_samples,
            n_scenarios=args.n_scenarios,
        )
    )

    df = pd.DataFrame(rows, columns=COLS)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)

    print_summary(df)
    print(f"\nSaved: {args.out_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
