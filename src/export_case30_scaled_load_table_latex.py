"""
export_case30_scaled_load_table_latex.py
----------------------------------------
Step 39 of the Syngrid data pipeline.

Exports the IEEE 30-bus MATPOWER load-scale OPF sweep as a LaTeX table.
"""

from pathlib import Path

import pandas as pd


IN_CSV = Path("results/case30_scaled_load_results.csv")
OUT_TEX = Path("results/case30_scaled_load_results.tex")


def build_latex(df: pd.DataFrame) -> str:
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"  \centering")
    lines.append(
        r"  \caption{IEEE 30-bus MATPOWER OPF results under uniform load scaling.}"
    )
    lines.append(r"  \label{tab:case30_load_scaling}")
    lines.append(r"  \begin{tabular}{rrrrr}")
    lines.append(r"    \toprule")
    lines.append(
        r"    \textbf{Load Scale} & \textbf{Success} & "
        r"\textbf{Cost} & \textbf{Load (MW)} & \textbf{Generation (MW)} \\"
    )
    lines.append(r"    \midrule")

    for _, row in df.iterrows():
        lines.append(
            f"    {row['LoadScale']:.2f} & "
            f"{int(row['Success'])} & "
            f"{row['ObjectiveCost']:.2f} & "
            f"{row['ScaledTotalLoadMW']:.2f} & "
            f"{row['TotalGenerationMW']:.2f} \\\\"
        )

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def main() -> int:
    if not IN_CSV.exists():
        print(f"ERROR: File not found at '{IN_CSV}'.")
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
