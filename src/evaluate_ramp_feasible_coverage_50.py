"""
evaluate_ramp_feasible_coverage_50.py
-------------------------------------
Computes empirical coverage of the 90% prediction interval for ramp-feasible
diffusion scenarios on the same 50-sample validation/test subset used in the
probabilistic metrics comparison.

For each sample, hour, and channel, the script computes the 5th and 95th
percentiles across K generated scenarios and checks whether the truth lies
inside that interval. Joint coverage requires wind and load to both be inside
their channel intervals at the same forecast hour.

Channel order: [..., 0] = wind   [..., 1] = load
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
DEFAULT_OUT_CSV = RESULTS / "ramp_feasible_coverage_50.csv"

NUM_TIMESTEPS = 1000
DEFAULT_N_SAMPLES = 50
DEFAULT_N_SCENARIOS = 20
DEFAULT_SEED = 2026
TARGET_COVERAGE = 0.90

OUT_COLS = [
    "split",
    "channel",
    "n_samples",
    "n_scenarios",
    "horizon",
    "mean_coverage",
    "std_coverage",
    "target_coverage",
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
    x_context: torch.Tensor,
    schedule: dict,
    device: torch.device,
    n_scenarios: int,
) -> torch.Tensor:
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


def compute_ramp_limits() -> np.ndarray:
    y_train = np.load(PROC / "Y_train.npy")
    abs_ramps = np.abs(np.diff(y_train, axis=1))
    return np.percentile(abs_ramps.reshape(-1, 2), 99, axis=0)


def clip_ramps(values: np.ndarray, ramp_limits: np.ndarray) -> np.ndarray:
    fixed = np.clip(values.astype(np.float64, copy=True), 0.0, None)

    for scen_idx in range(fixed.shape[0]):
        for t in range(1, fixed.shape[1]):
            prev = fixed[scen_idx, t - 1]
            lower = prev - ramp_limits
            upper = prev + ramp_limits
            fixed[scen_idx, t] = np.clip(fixed[scen_idx, t], lower, upper)

    return np.clip(fixed, 0.0, None)


def coverage_rows_for_sample(
    split: str,
    sample_id: int,
    scenarios: np.ndarray,
    truth: np.ndarray,
) -> list[dict]:
    lower = np.percentile(scenarios, 5, axis=0)
    upper = np.percentile(scenarios, 95, axis=0)
    inside = (truth >= lower) & (truth <= upper)

    return [
        {
            "split": split,
            "channel": "wind",
            "sample_id": sample_id,
            "n_scenarios": int(scenarios.shape[0]),
            "horizon": int(scenarios.shape[1]),
            "coverage": float(inside[:, 0].mean()),
        },
        {
            "split": split,
            "channel": "load",
            "sample_id": sample_id,
            "n_scenarios": int(scenarios.shape[0]),
            "horizon": int(scenarios.shape[1]),
            "coverage": float(inside[:, 1].mean()),
        },
        {
            "split": split,
            "channel": "joint",
            "sample_id": sample_id,
            "n_scenarios": int(scenarios.shape[0]),
            "horizon": int(scenarios.shape[1]),
            "coverage": float((inside[:, 0] & inside[:, 1]).mean()),
        },
    ]


def evaluate_split(
    split: str,
    model: torch.nn.Module,
    schedule: dict,
    device: torch.device,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    ramp_limits: np.ndarray,
    n_samples: int,
    n_scenarios: int,
) -> list[dict]:
    X = np.load(PROC / f"X_{split}_norm.npy")
    Y = np.load(PROC / f"Y_{split}_norm.npy")

    rows = []
    max_samples = min(n_samples, X.shape[0])

    print(f"\nEvaluating {split}: {max_samples} samples, {n_scenarios} scenarios each")

    for sample_id in range(max_samples):
        x_context = torch.from_numpy(X[sample_id : sample_id + 1]).float().to(device)
        truth = Y[sample_id] * y_std + y_mean

        scenarios_norm = ddpm_sample_batch(
            model=model,
            x_context=x_context,
            schedule=schedule,
            device=device,
            n_scenarios=n_scenarios,
        )
        scenarios = scenarios_norm.cpu().numpy() * y_std + y_mean
        scenarios = clip_ramps(scenarios, ramp_limits)

        rows.extend(
            coverage_rows_for_sample(
                split=split,
                sample_id=sample_id,
                scenarios=scenarios,
                truth=truth,
            )
        )

        if sample_id == 0 or (sample_id + 1) % 10 == 0 or sample_id + 1 == max_samples:
            print(f"  completed {split} sample {sample_id + 1:02d}/{max_samples:02d}")

    return rows


def aggregate_rows(rows: list[dict]) -> pd.DataFrame:
    per_sample = pd.DataFrame(rows)

    summary = (
        per_sample.groupby(["split", "channel"])
        .agg(
            n_samples=("sample_id", "nunique"),
            n_scenarios=("n_scenarios", "first"),
            horizon=("horizon", "first"),
            mean_coverage=("coverage", "mean"),
            std_coverage=("coverage", "std"),
        )
        .reset_index()
    )

    summary["mean_coverage"] = summary["mean_coverage"].round(4)
    summary["std_coverage"] = summary["std_coverage"].round(4)
    summary["target_coverage"] = TARGET_COVERAGE

    split_order = {"test": 0, "val": 1}
    channel_order = {"wind": 0, "load": 1, "joint": 2}
    summary["_split_sort"] = summary["split"].map(split_order).fillna(99)
    summary["_channel_sort"] = summary["channel"].map(channel_order).fillna(99)
    summary = (
        summary.sort_values(["_split_sort", "_channel_sort"])
        .drop(columns=["_split_sort", "_channel_sort"])
        .reset_index(drop=True)
    )

    return summary[OUT_COLS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate ramp-feasible 90% prediction interval coverage."
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
        help=f"Random seed for diffusion sampling (default: {DEFAULT_SEED}).",
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
        PROC / "Y_train.npy",
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
    print(f"Seed        : {args.seed}")
    print(f"OUT_CSV     : {args.out_csv}")

    stats = np.load(PROC / "diffusion_normalization_stats.npz")
    y_mean = stats["y_mean"].astype(np.float32)
    y_std = stats["y_std"].astype(np.float32)

    ramp_limits = compute_ramp_limits()
    print("Ramp limits from Y_train 99th-percentile absolute hourly ramps:")
    print(f"  Wind : {ramp_limits[0]:,.2f} MW/hour")
    print(f"  Load : {ramp_limits[1]:,.2f} MW/hour")

    model = DiffusionForecastModel().to(device)
    ckpt = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"Checkpoint epoch   : {ckpt.get('epoch', '?')}")
    print(f"Checkpoint val loss: {ckpt.get('val_loss', float('nan')):.6f}")

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
            ramp_limits=ramp_limits,
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
            ramp_limits=ramp_limits,
            n_samples=args.n_samples,
            n_scenarios=args.n_scenarios,
        )
    )

    summary = aggregate_rows(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out_csv, index=False)

    print("\nRamp-feasible 90% prediction interval coverage:")
    print(summary.to_string(index=False))
    print(f"\nSaved: {args.out_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
