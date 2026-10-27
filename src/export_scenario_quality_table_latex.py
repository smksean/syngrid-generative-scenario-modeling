"""
export_scenario_quality_table_latex.py
---------------------------------------
Step 30 of the Syngrid data pipeline.

Generates a manuscript-ready IEEE-style LaTeX table comparing diffusion
scenario quality across validation and test sample 0.
"""

from pathlib import Path

import pandas as pd


IN_CSV  = Path("results/scenario_quality_val_vs_test.csv")
OUT_TEX = Path("results/scenario_quality_val_vs_test.tex")

SPLIT_MAP = {
    "val_sample0":  "Validation",
    "test_sample0": "Test",
}


def build_latex(df: pd.DataFrame) -> str:
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"  \centering")
    lines.append(
        r"  \caption{Scenario-quality diagnostics for one validation sample and one "
        r"test sample using the best normalized diffusion checkpoint.}"
    )
    lines.append(r"  \label{tab:scenario_quality}")
    lines.append(r"  \begin{tabular}{lrrrrrrrrr}")
    lines.append(r"    \toprule")
    lines.append(
        r"    \textbf{Split} & "
        r"\textbf{Wind Std} & \textbf{Load Std} & "
        r"\textbf{Wind Min} & \textbf{Wind Max} & "
        r"\textbf{Load Min} & \textbf{Load Max} & "
        r"\textbf{Wind Bias} & \textbf{Load Bias} & "
        r"\textbf{Neg.\ Wind} \\"
    )
    lines.append(r"    \midrule")

    for _, row in df.iterrows():
        split = SPLIT_MAP.get(row["split"], row["split"])
        lines.append(
            f"    {split} & "
            f"{row['wind_mean_std']:.2f} & {row['load_mean_std']:.2f} & "
            f"{row['wind_min']:.2f} & {row['wind_max']:.2f} & "
            f"{row['load_min']:.2f} & {row['load_max']:.2f} & "
            f"{row['wind_first_hour_bias']:.2f} & {row['load_first_hour_bias']:.2f} & "
            f"{int(row['neg_wind'])} \\\\"
        )

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def main() -> int:
    if not IN_CSV.exists():
        print(f"ERROR: File not found at '{IN_CSV}'.")
        print("Run src/compare_val_test_scenario_quality.py first.")
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
