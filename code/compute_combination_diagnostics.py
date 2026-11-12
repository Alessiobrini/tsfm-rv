"""Diagnostics of the equal-weight combination of TTM and Log-HAR, the inputs of the two combination
diagnostics tables.

The combination averages the two volatility forecasts, c = ((s_T + s_L) / 2)^2. Every forecast is
clipped to the window bounds, and every model is scored on the dates common to the 17 models, with
QLIKE on the variance scale. Per asset and horizon the script computes:

  error moments   on the volatility scale, the ratio k of the larger to the smaller error RMSE, the
                  error correlation rho, both around zero, and the threshold (3 - k^2) / (2k) below
                  which the average beats the better member under squared error.
  combiners       QLIKE ratios of four combinations (harmonic, geometric and arithmetic means of the
                  variance forecasts, and the square of the average volatility forecast), and the
                  diversity term A = mean ln(h_i / h_H), which equals the mean member loss minus the
                  loss of the harmonic combination (Wood et al., 2023, Thm. 11).
  level           both members rescaled in real time (evaluation.forecast_sets.rescale_level), then
                  recombined.
  encompassing    Harvey-Leybourne-Newbold on the volatility scale and the QLIKE first-order
                  condition for a move from one model toward the other, Newey-West t-statistics with
                  max(22, 2h) lags.
  stability       the recursive Bates-Granger weight from the errors observed by each origin, the
                  share of 126-day blocks in which TTM has the lower QLIKE, and the number of changes
                  of the model with the lower QLIKE between consecutive blocks.
  pairs           Log-HAR averaged with every other model, TTM with every econometric model, and HAR
                  with every model.
  terciles        loss ratios by tercile of the volatility on the day before the origin.

Output: combination_diagnostics_long.csv, combination_pairs_long.csv and
combination_terciles_long.csv in the output folder.

Usage:  python code/compute_combination_diagnostics.py [--forecasts DIR] [--out DIR]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluation.combination import bates_granger_weights  # noqa: E402
from evaluation.forecast_sets import (ALL, DIRECT, ECON, FORECASTS, L, METRICS, MODELS, T,  # noqa: E402
                                      common_index, hac_t, load_clipped, load_sigma_all,
                                      qlike_series as ql, rescale_level)
from evaluation.targets import target_dates  # noqa: E402

ECON_ALL = ECON + DIRECT
BLOCK = 126


def combiners(hT, hL):
    sT, sL = np.sqrt(hT), np.sqrt(hL)
    return {"harm": 2.0 / (1.0 / hT + 1.0 / hL), "geo": np.sqrt(hT * hL),
            "vol": ((sT + sL) / 2.0) ** 2, "arith": (hT + hL) / 2.0}


def diversity(hT, hL):
    hH = 2.0 / (1.0 / hT + 1.0 / hL)
    return 0.5 * (np.log(hT / hH) + np.log(hL / hH))


def moments(sig, s1, s2):
    e1, e2 = sig - s1, sig - s2
    m1, m2, m12 = np.mean(e1 ** 2), np.mean(e2 ** 2), np.mean(e1 * e2)
    k = np.sqrt(max(m1, m2) / min(m1, m2))
    return dict(k=k, rho=m12 / np.sqrt(m1 * m2), thr=(3 - k ** 2) / (2 * k),
                rho_c=np.corrcoef(e1, e2)[0, 1],
                rho_log=np.corrcoef(np.log(sig / s1), np.log(sig / s2))[0, 1])


def pair_row(h, t, partner, anchor, s, Q, QL, sig, rv):
    cm = combiners(s[partner] ** 2, s[anchor] ** 2)
    return dict(h=h, ticker=t, partner=partner, anchor=anchor, r_member=Q[partner].mean() / QL,
                r_pair=ql(rv, cm["vol"]).mean() / QL, r_pair_harm=ql(rv, cm["harm"]).mean() / QL,
                **moments(sig, s[partner], s[anchor]))


def run(folder, sigma_all, horizons=(1, 5, 22)):
    main, pairs, terc = [], [], []
    for h in horizons:
        lags = max(22, 2 * h)
        for t in ALL:
            sig_s = sigma_all[t]
            dfs = load_clipped(folder, MODELS, t, h, sig_s, round_trip=False)
            idx = common_index(dfs)
            sig = dfs[L].loc[idx, "actual"].values
            for m in MODELS:
                if not np.allclose(dfs[m].loc[idx, "actual"].values, sig, rtol=1e-10, atol=0):
                    raise ValueError(f"{t} h={h} {m}: actuals differ from Log-HAR")
            tdate = target_dates(sig_s.index, idx, h)
            if not np.allclose(sig_s.loc[tdate].values, sig, rtol=1e-10, atol=0):
                raise ValueError(f"{t} h={h}: the target dates do not match the actuals")
            rv = sig ** 2
            s = {m: dfs[m].loc[idx, "forecast"].values for m in MODELS}
            Q = {m: ql(rv, s[m] ** 2) for m in MODELS}
            QL = Q[L].mean()
            sT, sL = s[T], s[L]
            hT, hL = sT ** 2, sL ** 2

            # error moments and combiners
            rec = dict(h=h, ticker=t, n=len(idx), **moments(sig, sT, sL))
            rec["r_T"] = Q[T].mean() / QL
            comb = combiners(hT, hL)
            qc = {k: ql(rv, v) for k, v in comb.items()}
            for k in comb:
                rec[f"r_{k}"] = qc[k].mean() / QL
            A = diversity(hT, hL)
            gap = 0.5 * (Q[T].mean() + Q[L].mean()) - qc["harm"].mean()
            if abs(gap - A.mean()) >= 1e-9 * max(1.0, abs(gap)):
                raise ValueError(f"{t} h={h}: the diversity identity does not hold")
            rec.update(A=A.mean() / QL, combiner=(qc["harm"].mean() - qc["vol"].mean()) / QL,
                       halfgap=abs(Q[T].mean() - QL) / 2 / QL)
            rec.update(mse_T=np.mean((sig - sT) ** 2), mse_L=np.mean((sig - sL) ** 2),
                       mse_vol=np.mean((sig - (sT + sL) / 2) ** 2))
            rec["bias_T"] = np.mean(np.log(rv / hT))
            rec["bias_L"] = np.mean(np.log(rv / hL))

            # real-time level correction of both members
            FT = rescale_level(dfs[T].dropna(), h).loc[idx].values
            FL = rescale_level(dfs[L].dropna(), h).loc[idx].values
            cr = combiners(FT, FL)
            rec.update(rc_T=ql(rv, FT).mean() / QL, rc_L=ql(rv, FL).mean() / QL,
                       rc_vol=ql(rv, cr["vol"]).mean() / QL, rc_harm=ql(rv, cr["harm"]).mean() / QL)

            # encompassing: t > 0 means the other model adds information
            eT, eL = sig - sT, sig - sL
            rec["hln_L_adds"] = hac_t(eT * (eT - eL), lags)
            rec["hln_T_adds"] = hac_t(eL * (eL - eT), lags)
            rec["foc_L_adds"] = hac_t(-((hT - rv) / hT ** 2) * 2 * sT * (sL - sT), lags)
            rec["foc_T_adds"] = hac_t(-((hL - rv) / hL ** 2) * 2 * sL * (sT - sL), lags)

            # stability: recursive Bates-Granger weight and 126-day blocks
            w_obs = bates_granger_weights(sig, sT, sL, horizon=h)
            w_all = bates_granger_weights(sig, sT, sL, horizon=1)
            for lab, w in [("bg", w_obs), ("bg_all_rows", w_all)]:
                rec[f"r_{lab}"] = ql(rv, (w * sT + (1 - w) * sL) ** 2).mean() / QL
            live = w_obs[100 + h - 1:]
            rec.update(w_mean=live.mean(), w_sd=live.std(), w_mid=np.mean((live >= 0.3) & (live <= 0.7)))
            nb = len(idx) // BLOCK
            bT = Q[T][:nb * BLOCK].reshape(nb, BLOCK).mean(1)
            bL = Q[L][:nb * BLOCK].reshape(nb, BLOCK).mean(1)
            bC = qc["vol"][:nb * BLOCK].reshape(nb, BLOCK).mean(1)
            rec.update(blocks=nb, share_T_leads=np.mean(bT < bL),
                       share_comb_beats_both=np.mean(bC < np.minimum(bT, bL)),
                       switches=int(np.sum(np.diff((bT < bL).astype(int)) != 0)))

            # terciles of the volatility on the day before the origin
            prev = np.log(sig_s.shift(1).loc[idx].values)
            rec["gw_vsT"] = hac_t(qc["vol"] - Q[T], lags, prev)
            rec["gw_vsL"] = hac_t(qc["vol"] - Q[L], lags, prev)
            grp = np.digitize(prev, np.quantile(prev, [1 / 3, 2 / 3]))
            for g in range(3):
                mk = grp == g
                ql_g = Q[L][mk].mean()
                terc.append(dict(h=h, ticker=t, tercile=g + 1, r_T=Q[T][mk].mean() / ql_g,
                                 r_vol=qc["vol"][mk].mean() / ql_g, r_harm=qc["harm"][mk].mean() / ql_g,
                                 bias_T=np.mean(np.log(rv[mk] / hT[mk])),
                                 bias_L=np.mean(np.log(rv[mk] / hL[mk]))))
            main.append(rec)

            # pairs
            pairs += [pair_row(h, t, m, L, s, Q, QL, sig, rv) for m in MODELS if m != L]
            pairs += [pair_row(h, t, m, T, s, Q, QL, sig, rv) for m in ECON_ALL if m != L]
            pairs += [pair_row(h, t, m, "HAR", s, Q, QL, sig, rv) for m in MODELS if m != "HAR"]
        print(f"h={h} done", flush=True)
    return pd.DataFrame(main), pd.DataFrame(pairs), pd.DataFrame(terc)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--forecasts", type=Path, default=FORECASTS, help="point forecasts of the 17 models")
    ap.add_argument("--out", type=Path, default=METRICS)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    M, P, G = run(a.forecasts, load_sigma_all())
    M.to_csv(a.out / "combination_diagnostics_long.csv", index=False)
    P.to_csv(a.out / "combination_pairs_long.csv", index=False)
    G.to_csv(a.out / "combination_terciles_long.csv", index=False)
    print("written to", a.out)


if __name__ == "__main__":
    main()
