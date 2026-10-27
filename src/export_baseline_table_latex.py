"""
export_baseline_table_latex.py
------------------------------
Step 10 of the Syngrid data pipeline.

Generates a manuscript-ready IEEE-style LaTeX table from the baseline
comparison results CSV.
"""

from pathlib import Path

import pandas as pd


IN_CSV  = Path("results/baseline_comparison_table.csv")
OUT_TEX = Path("results/baseline_comparison_table.tex")

MODEL_MAP = {
    "persistence":        "Persistence",
    "linear_regression":  "Linear Regression",
}

SPLIT_MAP = {
    "val":  "Validation",
    "test": "Test",
}


def build_latex(df: pd.DataFrame) -> str:
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{Baseline forecasting results for 24-hour ahead load and wind prediction.}")
    lines.append(r"  \label{tab:baseline_results}")
    lines.append(r"  \begin{tabular}{llrrrr}")
    lines.append(r"    \toprule")
    lines.append(
        r"    \textbf{Model} & \textbf{Split} & "
        r"\textbf{Load MAE} & \textbf{Load RMSE} & "
        r"\textbf{Wind MAE} & \textbf{Wind RMSE} \\"
    )
    lines.append(r"    \midrule")

    for _, row in df.iterrows():
        model = MODEL_MAP.get(row["model"], row["model"])
        split = SPLIT_MAP.get(row["split"], row["split"])
        lines.append(
            f"    {model} & {split} & "
            f"{row['load_mae']:.2f} & {row['load_rmse']:.2f} & "
            f"{row['wind_mae']:.2f} & {row['wind_rmse']:.2f} \\\\"
        )

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def main() -> int:
    if not IN_CSV.exists():
        print(f"ERROR: File not found at '{IN_CSV}'.")
        print("Run src/make_baseline_results_table.py first.")
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
