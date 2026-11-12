"""Score the fine-tuned TTM and Sundial against their zero-shot versions and Log-HAR, the input of the
fine-tuning table.

The fine-tuned models (finetune/train.py, volatility input, TTM on 512 days and Sundial on 1,000
days) forecast through the paper's runner (run_foundation_volare.py with --checkpoint,
cluster/run_tsfm_finetuned.slurm), so their files have the format, dates and clipping of the
zero-shot files. The fine-tuning corpus is not distributed with the code, since it is built from
licensed CRSP and TAQ data, but the scoring needs only the forecast files. Scoring as in the main
comparison: forecasts clipped to the window bounds, QLIKE on the variance scale, per-asset ratios
to Log-HAR averaged across assets. The window starts on 2022-01-03, after the end of the
fine-tuning data (training to 2020, early stopping on 2021), on dates common to Log-HAR, the two
zero-shot models and the two fine-tuned ones.

With --fth the head-only fine-tuning (finetune/train.py --train-part head) is scored as ttm_fth and
sundial_fth on the same dates. The head-only files carry the file names of the fully fine-tuned
models and must cover the common dates with the same actuals. With --check, the scores of the
other five models must equal an earlier finetune_long.csv, so adding the head-only models changes
no other number.

Output: finetune_long.csv, finetune_summary.csv, finetune_by_group.csv and finetune_wins.csv.

Usage:  python code/score_finetune.py [--forecasts DIR] [--ft DIR] [--fth DIR] [--check CSV] [--out DIR]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluation.forecast_sets import (ALL, FORECASTS, L, METRICS, ROOT, asset_class,  # noqa: E402
                                      load_sigma_all, qlike_series as ql, read)
from forecasting.bounds import clip_to_window  # noqa: E402

START = pd.Timestamp("2022-01-03")
# model label -> (folder key, file stem). The head-only files carry the stem of the fine-tuned model.
FILES = {L: ("main", L), "ttm": ("main", "ttm"), "sundial": ("main", "sundial"),
         "ttm_ft": ("ft", "ttm_ft"), "sundial_ft": ("ft", "sundial_ft")}
HEAD = {"ttm_fth": ("fth", "ttm_ft"), "sundial_fth": ("fth", "sundial_ft")}
PAIRS = [("ttm_ft", "ttm"), ("sundial_ft", "sundial"), ("ttm_ft", L), ("sundial_ft", L)]
HEAD_PAIRS = [("ttm_fth", "ttm"), ("sundial_fth", "sundial"), ("ttm_fth", L), ("sundial_fth", L),
              ("ttm_fth", "ttm_ft"), ("sundial_fth", "sundial_ft")]


def load(t, h, sig, dirs, files):
    out = {}
    for m, (k, stem) in files.items():
        d = read(dirs[k] / f"{stem}_{t}_h{h}.csv")
        d["forecast"] = clip_to_window(d["forecast"], sig)
        out[m] = d
    return out


def run(sigma_all, dirs, horizons=(1, 5, 22), head=False):
    rows = []
    for t in ALL:
        sig = sigma_all[t]
        for h in horizons:
            files = load(t, h, sig, dirs, FILES)
            idx = None
            for d in files.values():
                k = set(d["forecast"].dropna().index)
                idx = k if idx is None else idx & k
            idx = pd.DatetimeIndex(sorted(i for i in idx if i >= START))
            if head:
                extra = load(t, h, sig, dirs, HEAD)
                for m, d in extra.items():
                    missing = idx.difference(d["forecast"].dropna().index)
                    if len(missing):
                        raise ValueError(f"{m} {t} h={h}: {len(missing)} common dates missing")
                files.update(extra)
            y = files[L].loc[idx, "actual"].values
            for m, d in files.items():
                if not np.allclose(d.loc[idx, "actual"].values, y, rtol=1e-10, atol=0):
                    raise ValueError(f"{t} h={h} {m}: actuals differ from Log-HAR")
            rv = y ** 2
            base = ql(rv, files[L].loc[idx, "forecast"].values ** 2).mean()
            for m, d in files.items():
                f = d.loc[idx, "forecast"].values
                e = np.log(y / f)
                q = ql(rv, f ** 2).mean()
                rows.append(dict(ticker=t, cls=asset_class(t), h=h, model=m, n=len(idx), start=idx[0].date(),
                                 qlike=q, ratio=q / base, bias=e.mean(), sd=e.std()))
        print("done", t, flush=True)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--forecasts", type=Path, default=FORECASTS,
                    help="point forecasts of the main comparison (Log-HAR and the zero-shot models)")
    ap.add_argument("--ft", type=Path, default=ROOT / "results/tsfm_ft/forecasts",
                    help="forecasts of the fine-tuned models (ttm_ft and sundial_ft files)")
    ap.add_argument("--fth", type=Path, default=None, help="forecasts of the head-only fine-tuning")
    ap.add_argument("--check", type=Path, default=None,
                    help="earlier finetune_long.csv that the five models must reproduce")
    ap.add_argument("--out", type=Path, default=METRICS)
    a = ap.parse_args()
    dirs = {"main": a.forecasts, "ft": a.ft, "fth": a.fth}
    res = run(load_sigma_all(), dirs, head=a.fth is not None)
    if a.check:
        old = pd.read_csv(a.check, float_precision="round_trip")
        new = res[res.model.isin(FILES)].reset_index(drop=True)
        m = old.merge(new, on=["ticker", "h", "model"], suffixes=("_old", ""))
        if len(m) != len(old) or len(new) != len(old):
            raise AssertionError(f"rows differ: old {len(old)}, new {len(new)}, matched {len(m)}")
        for c in ["n", "qlike", "ratio", "bias", "sd"]:
            if not np.allclose(m[c + "_old"], m[c], rtol=1e-12, atol=0):
                raise AssertionError(f"{c} differs from {a.check}")
        print("the five models reproduce", a.check)
    a.out.mkdir(parents=True, exist_ok=True)
    res.to_csv(a.out / "finetune_long.csv", index=False)
    res["group"] = np.where(res.cls == "stocks", "stocks", "fx+futures")
    s = res.groupby(["h", "model"]).agg(ratio=("ratio", "mean"), beat=("ratio", lambda r: int((r < 1).sum())),
                                        bias=("bias", "mean"), sd=("sd", "mean"), n=("n", "mean")).reset_index()
    g = res.groupby(["group", "h", "model"]).agg(ratio=("ratio", "mean"), n_assets=("ratio", "size")).reset_index()
    w = res.pivot_table(index=["ticker", "h"], columns="model", values="qlike")
    grp = res.drop_duplicates("ticker").set_index("ticker")["group"]
    wins = []
    for a_, b_ in PAIRS + (HEAD_PAIRS if a.fth is not None else []):
        for h in sorted(res.h.unique()):
            x = w.xs(h, level="h")
            better = x[a_] < x[b_]
            wins.append(dict(ft=a_, ref=b_, h=h, all=int(better.sum()),
                             stocks=int(better[grp.loc[x.index] == "stocks"].sum()),
                             fx_futures=int(better[grp.loc[x.index] != "stocks"].sum())))
    s.to_csv(a.out / "finetune_summary.csv", index=False)
    g.to_csv(a.out / "finetune_by_group.csv", index=False)
    pd.DataFrame(wins).to_csv(a.out / "finetune_wins.csv", index=False)
    print("written to", a.out)


if __name__ == "__main__":
    main()
