"""
export_probabilistic_metrics_comparison_50_latex.py
---------------------------------------------------
Step 36 of the Syngrid data pipeline.

Exports the 50-sample probabilistic metrics comparison as an IEEE-style
LaTeX table.
"""

from pathlib import Path

import pandas as pd


IN_CSV = Path("results/probabilistic_metrics_comparison_50.csv")
OUT_TEX = Path("results/probabilistic_metrics_comparison_50.tex")

MODEL_MAP = {
    "diffusion": "Diffusion",
    "linear_regression": "Linear Regression",
    "persistence": "Persistence",
}

SPLIT_MAP = {
    "val": "Validation",
    "test": "Test",
}


def build_latex(df: pd.DataFrame) -> str:
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"  \centering")
    lines.append(
        r"  \caption{Probabilistic scenario-quality comparison on the 50-sample "
        r"validation and test subsets. Lower values are better.}"
    )
    lines.append(r"  \label{tab:probabilistic_metrics_comparison}")
    lines.append(r"  \begin{tabular}{llrrr}")
    lines.append(r"    \toprule")
    lines.append(
        r"    \textbf{Model} & \textbf{Split} & "
        r"\textbf{Wind CRPS} & \textbf{Load CRPS} & "
        r"\textbf{Energy Score} \\"
    )
    lines.append(r"    \midrule")

    for _, row in df.iterrows():
        model = MODEL_MAP.get(row["model"], row["model"])
        split = SPLIT_MAP.get(row["split"], row["split"])
        lines.append(
            f"    {model} & {split} & "
            f"{row['wind_crps_mean_mean']:.2f} & "
            f"{row['load_crps_mean_mean']:.2f} & "
            f"{row['joint_energy_score_mean']:.2f} \\\\"
        )

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def main() -> int:
    if not IN_CSV.exists():
        print(f"ERROR: File not found at '{IN_CSV}'.")
        print("Run src/compare_probabilistic_metrics_50.py first.")
        return 1

    df = pd.read_csv(IN_CSV)
    latex = build_latex(df)

    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUT_TEX.write_text(latex + "\n", encoding="utf-8")

    print(latex)
    print(f"\nSaved: {OUT_TEX}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
