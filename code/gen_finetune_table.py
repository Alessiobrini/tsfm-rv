"""table_finetune.tex, from finetune_summary.csv and finetune_by_group.csv of score_finetune.py.

With --head the scores hold the head-only fine-tuning too (score_finetune.py --fth), and the table
has three rows per model: zero-shot, the fine-tuning of all the weights of TTM or of low-rank
adapters on Sundial, and the head-only fine-tuning with the pretrained backbone fixed.

Usage:  python code/gen_finetune_table.py [--metrics DIR] [--out DIR] [--head]
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluation.forecast_sets import METRICS, TABLES  # noqa: E402

ROWS = [("ttm", "TTM, zero-shot"), ("ttm_ft", "TTM, fine-tuned"), ("sundial", "Sundial, zero-shot"),
        ("sundial_ft", "Sundial, fine-tuned")]
ROWS_HEAD = [("ttm", "TTM, zero-shot"), ("ttm_ft", "TTM, all weights"), ("ttm_fth", "TTM, head only"),
             ("sundial", "Sundial, zero-shot"), ("sundial_ft", "Sundial, adapters"),
             ("sundial_fth", "Sundial, head only")]
H = (1, 5, 22)


def one(df, m, h, col, **kw):
    r = df[(df.model == m) & (df.h == h)]
    for k, v in kw.items():
        r = r[r[k] == v]
    if len(r) != 1:
        raise ValueError(f"{m} h={h} {kw}: {len(r)} rows")
    return r[col].iloc[0]


def caption(rows):
    if rows is ROWS:
        return (r"Fine-tuned TTM and Sundial. Mean across assets of the per-asset QLIKE loss ratio to Log-HAR from 3 January 2022 to 30 January 2026, for each model applied zero-shot as in the main comparison and fine-tuned as in Appendix~\ref{app:finetune}. Panel~A: the 50 assets, with the number of assets with a ratio below one ($<1$). Panel~B: the 40 stocks, which are in the fine-tuning corpus, and the 10 FX pairs and futures, which are not. Dates common to Log-HAR and the four models.")
    return (r"Fine-tuned TTM and Sundial. Mean across assets of the per-asset QLIKE loss ratio to Log-HAR from 3 January 2022 to 30 January 2026, for each model applied zero-shot as in the main comparison and fine-tuned in the two ways of Appendix~\ref{app:finetune}: all the weights of TTM or low-rank adapters on Sundial, and the head only, with the pretrained backbone fixed. Panel~A: the 50 assets, with the number of assets with a ratio below one ($<1$). Panel~B: the 40 stocks, which are in the fine-tuning corpus, and the 10 FX pairs and futures, which are not. Dates common to Log-HAR and the six models.")


def table(s, g, rows=ROWS):
    a = [f"{name} & " + " & ".join(f"{one(s, m, h, 'ratio'):.3f} & {int(one(s, m, h, 'beat'))}" for h in H) + r" \\"
         for m, name in rows]
    b = [f"{name} & " + " & ".join(f"{one(g, m, h, 'ratio', group=grp):.3f}" for grp in ("stocks", "fx+futures") for h in H) + r" \\"
         for m, name in rows]
    return "\n".join([
        r"\begin{table}[htbp]",
        r"\centering",
        r"\singlespacing",
        r"\caption{" + caption(rows) + "}",
        r"\label{tab:finetune}",
        r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"\multicolumn{7}{l}{\textit{Panel A: all 50 assets}} \\",
        r"& \multicolumn{2}{c}{$h=1$} & \multicolumn{2}{c}{$h=5$} & \multicolumn{2}{c}{$h=22$} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
        r"Model & Ratio & $<1$ & Ratio & $<1$ & Ratio & $<1$ \\",
        r"\midrule",
        *a,
        r"\midrule",
        r"\multicolumn{7}{l}{\textit{Panel B: by asset class}} \\",
        r"& \multicolumn{3}{c}{40 stocks} & \multicolumn{3}{c}{10 FX pairs and futures} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        r"Model & $h=1$ & $h=5$ & $h=22$ & $h=1$ & $h=5$ & $h=22$ \\",
        r"\midrule",
        *b,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--metrics", type=Path, default=METRICS, help="output folder of score_finetune.py")
    ap.add_argument("--out", type=Path, default=TABLES)
    ap.add_argument("--head", action="store_true", help="include the head-only fine-tuning")
    a = ap.parse_args()
    s = pd.read_csv(a.metrics / "finetune_summary.csv", float_precision="round_trip")
    g = pd.read_csv(a.metrics / "finetune_by_group.csv", float_precision="round_trip")
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "table_finetune.tex").write_text(table(s, g, ROWS_HEAD if a.head else ROWS))
    print("written to", a.out / "table_finetune.tex")


if __name__ == "__main__":
    main()
