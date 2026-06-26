#!/usr/bin/env python3
"""Merge the MCS inclusion table (tab:mcs) and the DM win-rate table
(tab:dm_summary) into one side-by-side table (paper/tables/table_mcs_dm.tex).

This parses the two existing locked .tex source files and re-emits the merged
table verbatim: no value is recomputed. Both tables share the same 17 model
rows in the same order and the same three horizons (h=1, 5, 22). The merged
table carries BOTH labels (\\label{tab:mcs} and \\label{tab:dm_summary}, plus
\) so every existing cross-reference still resolves.

Run:  python3 code/gen_mcs_dm.py
"""
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
TABLE_DIR = HERE.parent / "paper" / "tables"

MCS_SRC = TABLE_DIR / "table_equity_mcs.tex"
DM_SRC = TABLE_DIR / "table_dm_summary.tex"
OUT = TABLE_DIR / "table_mcs_dm.tex"


def parse_rows(path):
    """Return an ordered list of (model, [c1, c2, c3]) parsed from the body
    rows (between \\midrule and \\bottomrule) of a 4-column LaTeX table.
    Values are kept verbatim as written in the source."""
    text = path.read_text()
    # body between the first \midrule and \bottomrule
    body = text.split("\\midrule", 1)[1].split("\\bottomrule", 1)[0]
    rows = []
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith("\\") or "&" not in line:
            continue
        # strip trailing \\
        line = line.rstrip()
        if line.endswith("\\\\"):
            line = line[:-2].strip()
        cells = [c.strip() for c in line.split("&")]
        if len(cells) != 4:
            raise ValueError(f"Unexpected row in {path.name}: {line!r}")
        rows.append((cells[0], cells[1:]))
    return rows


def main():
    mcs = parse_rows(MCS_SRC)
    dm = parse_rows(DM_SRC)

    if len(mcs) != len(dm):
        raise ValueError(f"row count mismatch: MCS {len(mcs)} vs DM {len(dm)}")
    for (mm, _), (dmm, _) in zip(mcs, dm):
        if mm != dmm:
            raise ValueError(f"model order mismatch: {mm!r} vs {dmm!r}")

    caption = (
        "Model Confidence Set inclusion rates and Diebold--Mariano pairwise win "
        "rates for 50 assets (40 equities, 5 FX, 5 futures). "
        "MCS inclusion rates on the left, Diebold--Mariano pairwise win rates on "
        "the right. "
        "MCS columns report the percentage of assets for which the model is "
        "included in the MCS at the 10\\% significance level (QLIKE loss, "
        "$T_{\\max}$ statistic, block bootstrap with $B = 10{,}000$). "
        "DM columns report the percentage of pairwise comparisons (across 50 "
        "assets $\\times$ 16 opponents = 800 tests) in which the row model "
        "achieves significantly lower QLIKE at the 5\\% level."
    )

    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\singlespacing",
        f"\\caption{{{caption}}}",
        "\\label{tab:mcs}",
        "\\label{tab:dm_summary}",
        
        "\\small",
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "& \\multicolumn{3}{c}{Model Confidence Set inclusion (\\%)} "
        "& \\multicolumn{3}{c}{Diebold--Mariano win rate (\\%)} \\\\",
        "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
        "Model & $h=1$ & $h=5$ & $h=22$ & $h=1$ & $h=5$ & $h=22$ \\\\",
        "\\midrule",
    ]

    for (model, mcs_vals), (_, dm_vals) in zip(mcs, dm):
        cells = [model] + mcs_vals + dm_vals
        lines.append(" & ".join(cells) + " \\\\")

    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]

    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT} ({len(mcs)} model rows)")


if __name__ == "__main__":
    main()
