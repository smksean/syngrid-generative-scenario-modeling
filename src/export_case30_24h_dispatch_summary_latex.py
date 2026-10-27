"""
export_case30_24h_dispatch_summary_latex.py
-------------------------------------------
Step 46 of the Syngrid data pipeline.

Exports the 24-hour IEEE 30-bus scenario-dispatch summary as a LaTeX table.
"""

from pathlib import Path

import pandas as pd


IN_CSV = Path("results/case30_test_sample0_24h_dispatch_summary.csv")
OUT_TEX = Path("results/case30_test_sample0_24h_dispatch_summary.tex")


def build_latex(row: pd.Series) -> str:
    load_range = f"{row['load_scale_min']:.4f}--{row['load_scale_max']:.4f}"

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"  \centering")
    lines.append(
        r"  \caption{Twenty-four-hour IEEE 30-bus OPF results for scenario-derived "
        r"test sample 0 load scales.}"
    )
    lines.append(r"  \label{tab:case30_24h_scenario_dispatch}")
    lines.append(r"  \begin{tabular}{rrrrrrr}")
    lines.append(r"    \toprule")
    lines.append(
        r"    \textbf{Cases} & \textbf{Scen.} & \textbf{Horizon} & "
        r"\textbf{Success Rate} & \textbf{Load Scale Range} & "
        r"\textbf{Mean Cost} & \textbf{Failed Scen.} \\"
    )
    lines.append(r"    \midrule")
    lines.append(
        f"    {int(row['n_dispatch_cases'])} & "
        f"{int(row['n_scenarios'])} & "
        f"{int(row['horizon'])} & "
        f"{100 * row['success_rate']:.2f}\\% & "
        f"{load_range} & "
        f"{row['successful_cost_mean']:.2f} & "
        f"{int(row['n_failed_scenarios'])} \\\\"
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
    latex = build_latex(df.iloc[0])

    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUT_TEX.write_text(latex + "\n", encoding="utf-8")

    print(latex)
    print(f"\nSaved: {OUT_TEX}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
