"""Supervised machine-learning baselines for realized volatility: XGBoost and feed-forward networks
on HAR-style inputs, and an LSTM on a sequence of past log volatility.

Both learners forecast log volatility, log(sigma), directly at each horizon h, and the forecast is
converted back to volatility as exp(m + v/2), the retransformation of Log-HAR, with v the variance
of the model's errors on the validation part of its training window. A flexible model fits its
training sample closely, so its in-sample residuals understate its forecast errors; the validation
errors do not.

Rolling scheme (``rolling_ml``). At origin p the information set is rows 0 to p-1, as for every
other model. The training window holds the origins q in [p - window + lookback, p - h], so every
input row lies in the window and every target, dated q + h - 1, is observed before p. Every
``retune_every`` origins the learner is tuned on a small grid: each setting is fitted on the
origins whose targets end before the last ``val_size`` origins of the window and scored by QLIKE on
those last origins, and the best setting fixes the number of trees or epochs and the variance v.
Every ``refit_every`` origins the chosen setting is refitted on the whole window, and the forecasts
until the next refit come from that fit.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from config import RANDOM_SEED


# ---------------------------------------------------------------------------- targets and losses
def log_targets(vol: np.ndarray, horizon: int, target_kind: str = "point") -> np.ndarray:
    """Target of each origin q on the log scale: log(sigma_{q+h-1}) for the point target, the log of
    the mean of sigma_q .. sigma_{q+h-1} for the averaged target. NaN where the target lies past the
    end of the series."""
    n = len(vol)
    y = np.full(n, np.nan)
    if target_kind == "point":
        y[: n - horizon + 1] = np.log(vol[horizon - 1:])
    elif target_kind == "avg":
        cs = np.concatenate([[0.0], np.cumsum(vol)])
        q = np.arange(n - horizon + 1)
        y[: n - horizon + 1] = np.log((cs[q + horizon] - cs[q]) / horizon)
    else:
        raise ValueError(f"target_kind must be 'point' or 'avg', got {target_kind!r}")
    return y


def to_vol(m: np.ndarray, v: float) -> np.ndarray:
    """exp(m + v/2), the volatility forecast from a forecast m of log volatility."""
    return np.exp(np.asarray(m, dtype=float) + 0.5 * v)


def qlike_vol(actual_vol: np.ndarray, forecast_vol: np.ndarray) -> float:
    """Mean QLIKE on the variance scale, as in the evaluation."""
    r = actual_vol ** 2 / forecast_vol ** 2
    return float(np.mean(r - np.log(r) - 1.0))


# ---------------------------------------------------------------------------- learners
class XGBHAR:
    """XGBoost on the HAR inputs of log volatility: the log of the mean volatility over the last 1, 5
    and 22 days before the origin, as Log-HAR uses them."""

    lookback = 22
    grid = [dict(max_depth=d, learning_rate=lr, min_child_weight=mcw)
            for d, lr, mcw in itertools.product([2, 3, 4], [0.03, 0.1], [1, 10])]

    def __init__(self, n_jobs: int = 1, max_trees: int = 2000, early_stop: int = 50):
        self.n_jobs, self.max_trees, self.early_stop = n_jobs, max_trees, early_stop

    def features(self, vol: np.ndarray) -> np.ndarray:
        n = len(vol)
        cs = np.concatenate([[0.0], np.cumsum(vol)])
        X = np.full((n, 3), np.nan)
        q = np.arange(self.lookback, n + 1)
        q = q[q < n] if len(q) else q
        for j, k in enumerate((1, 5, 22)):
            X[q, j] = np.log((cs[q] - cs[q - k]) / k)
        return X

    def _model(self, cfg, n_estimators, early_stop=None):
        import xgboost as xgb
        return xgb.XGBRegressor(n_estimators=n_estimators, subsample=0.8, objective="reg:squarederror",
                                tree_method="hist", n_jobs=self.n_jobs, random_state=RANDOM_SEED,
                                early_stopping_rounds=early_stop, **cfg)

    def tune(self, Xtr, ytr, Xva, yva):
        best = None
        for cfg in self.grid:
            m = self._model(cfg, self.max_trees, self.early_stop)
            m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
            pred = m.predict(Xva, iteration_range=(0, m.best_iteration + 1))
            v = float(np.var(yva - pred))
            score = qlike_vol(np.exp(yva), to_vol(pred, v))
            if best is None or score < best["score"]:
                best = dict(cfg=cfg, n=int(m.best_iteration + 1), v=v, score=score)
        return best

    def fit(self, X, y, tuned):
        return self._model(tuned["cfg"], tuned["n"]).fit(X, y, verbose=False)

    def predict(self, model, X):
        return model.predict(X)


class LSTMSeq:
    """A one-layer LSTM on the last ``lookback`` values of log volatility, with a linear output. The
    input is standardized with the mean and standard deviation of the training window. The forecast
    is the average over ``seeds`` networks."""

    grid = [dict(hidden=h, dropout=d) for h, d in itertools.product([16, 32], [0.0, 0.2])]

    def __init__(self, lookback: int = 22, seeds: int = 3, max_epochs: int = 200, patience: int = 20,
                 batch: int = 64, lr: float = 1e-3, threads: int = 1):
        self.lookback, self.seeds, self.max_epochs = lookback, seeds, max_epochs
        self.patience, self.batch, self.lr, self.threads = patience, batch, lr, threads

    def features(self, vol: np.ndarray) -> np.ndarray:
        s = np.log(vol)
        n, L = len(s), self.lookback
        X = np.full((n, L), np.nan)
        if n > L:
            idx = np.arange(L, n)[:, None] + np.arange(-L, 0)[None, :]
            X[L:] = s[idx]
        return X

    def _net(self, cfg):
        import torch
        import torch.nn as nn

        class Net(nn.Module):
            def __init__(self, hidden, dropout):
                super().__init__()
                self.lstm = nn.LSTM(1, hidden, batch_first=True)
                self.drop = nn.Dropout(dropout)
                self.out = nn.Linear(hidden, 1)

            def forward(self, x):
                h, _ = self.lstm(x.unsqueeze(-1))
                return self.out(self.drop(h[:, -1])).squeeze(-1)

        return Net(cfg["hidden"], cfg["dropout"])

    def _train(self, cfg, X, y, seed, epochs, Xva=None, yva=None):
        """Train one network. With a validation set, stop early and return the best epoch."""
        import torch
        torch.set_num_threads(self.threads)
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        mu, sd = float(np.mean(X)), float(np.std(X)) or 1.0
        ymu = float(np.mean(y))
        net = self._net(cfg)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        Xt = torch.tensor((X - mu) / sd, dtype=torch.float32)
        yt = torch.tensor((y - ymu) / sd, dtype=torch.float32)
        if Xva is not None:
            Xv = torch.tensor((Xva - mu) / sd, dtype=torch.float32)
        best, best_ep, best_state, wait = np.inf, 0, None, 0
        for ep in range(1, epochs + 1):
            net.train()
            perm = rng.permutation(len(Xt))
            for i in range(0, len(perm), self.batch):
                b = perm[i:i + self.batch]
                opt.zero_grad()
                loss = torch.mean((net(Xt[b]) - yt[b]) ** 2)
                loss.backward()
                opt.step()
            if Xva is not None:
                net.eval()
                with torch.no_grad():
                    val = float(torch.mean((net(Xv) * sd + ymu - torch.tensor(yva, dtype=torch.float32)) ** 2))
                if val < best - 1e-12:
                    best, best_ep, wait = val, ep, 0
                    best_state = {k: t.clone() for k, t in net.state_dict().items()}
                else:
                    wait += 1
                    if wait >= self.patience:
                        break
        if best_state is not None:
            net.load_state_dict(best_state)
        net.eval()
        return dict(net=net, mu=mu, sd=sd, ymu=ymu, best_epoch=best_ep)

    def _pred_one(self, fitted, X):
        import torch
        with torch.no_grad():
            Xt = torch.tensor((X - fitted["mu"]) / fitted["sd"], dtype=torch.float32)
            return fitted["net"](Xt).numpy() * fitted["sd"] + fitted["ymu"]

    def tune(self, Xtr, ytr, Xva, yva):
        best = None
        for cfg in self.grid:
            fits = [self._train(cfg, Xtr, ytr, RANDOM_SEED + k, self.max_epochs, Xva, yva)
                    for k in range(self.seeds)]
            pred = np.mean([self._pred_one(f, Xva) for f in fits], axis=0)
            v = float(np.var(yva - pred))
            score = qlike_vol(np.exp(yva), to_vol(pred, v))
            if best is None or score < best["score"]:
                epochs = int(round(np.mean([max(f["best_epoch"], 1) for f in fits])))
                best = dict(cfg=cfg, n=epochs, v=v, score=score)
        return best

    def fit(self, X, y, tuned):
        return [self._train(tuned["cfg"], X, y, RANDOM_SEED + k, tuned["n"]) for k in range(self.seeds)]

    def predict(self, model, X):
        return np.mean([self._pred_one(f, X) for f in model], axis=0)


class FFNHAR(XGBHAR):
    """Feed-forward networks on the HAR inputs, after Christensen, Siggaard and Veliyev (2023, Sec. 2
    and Appendix A.4 to A.5): four pyramid architectures with 2; 4, 2; 8, 4, 2; and 16, 8, 4, 2
    neurons, leaky ReLU, Glorot normal initialization, dropout on the hidden layers, Adam with learning
    rate 0.001, at most 500 epochs with early stopping at patience 100 on the validation MSE, and the
    forecast averaged over the 10 best of 100 networks with different seeds, ranked by validation
    MSE. The architecture is tuned by QLIKE on the validation days, as for the other learners. A
    refit trains the 10 chosen seeds on the whole window, each for its own best number of epochs.

    The 100 networks of one architecture are trained together as one batched model (the weights of
    network s are the slice s of each weight tensor), so they share the order of the minibatches and
    differ in their initial weights and dropout masks."""

    grid = [dict(layers=(2,)), dict(layers=(4, 2)), dict(layers=(8, 4, 2)), dict(layers=(16, 8, 4, 2))]

    def __init__(self, n_nets: int = 100, n_best: int = 10, max_epochs: int = 500, patience: int = 100,
                 dropout: float = 0.2, batch: int = 64, lr: float = 1e-3, slope: float = 0.01, threads: int = 1):
        super().__init__()
        self.n_nets, self.n_best, self.max_epochs, self.patience = n_nets, n_best, max_epochs, patience
        self.dropout, self.batch, self.lr, self.slope, self.threads = dropout, batch, lr, slope, threads

    def _init(self, layers, seeds):
        """Weights and biases of one network per seed, stacked along the first axis."""
        import torch
        sizes = (3,) + tuple(layers) + (1,)
        params = []
        for i, o in zip(sizes[:-1], sizes[1:]):
            W = torch.empty(len(seeds), i, o)
            for k, s in enumerate(seeds):
                g = torch.Generator().manual_seed(int(RANDOM_SEED + s))
                W[k] = torch.randn(i, o, generator=g) * np.sqrt(2.0 / (i + o))
            params += [W.requires_grad_(), torch.zeros(len(seeds), 1, o, requires_grad=True)]
        return params

    def _forward(self, params, x, train):
        import torch
        h = x
        n = len(params) // 2
        for j in range(n):
            h = torch.baddbmm(params[2 * j + 1], h, params[2 * j])
            if j < n - 1:
                h = torch.nn.functional.leaky_relu(h, self.slope)
                if train and self.dropout > 0:
                    h = torch.nn.functional.dropout(h, self.dropout, training=True)
        return h.squeeze(-1)

    def _train(self, layers, seeds, X, y, epochs, Xva=None, yva=None):
        """Train one network per seed. With validation data, stop each network at patience and keep
        its best state. Without, keep each network's state after its own number of epochs."""
        import torch
        torch.set_num_threads(self.threads)
        torch.manual_seed(RANDOM_SEED)
        rng = np.random.default_rng(RANDOM_SEED)
        mu, sd = X.mean(axis=0), X.std(axis=0)
        sd = np.where(sd > 0, sd, 1.0)
        ymu, ysd = float(np.mean(y)), float(np.std(y)) or 1.0
        S = len(seeds)
        Xt = torch.tensor((X - mu) / sd, dtype=torch.float32)
        yt = torch.tensor((y - ymu) / ysd, dtype=torch.float32)
        params = self._init(layers, seeds)
        opt = torch.optim.Adam(params, lr=self.lr)
        ep_target = np.asarray(epochs if np.ndim(epochs) else [epochs] * S)
        best = [None] * S
        best_val, best_ep = np.full(S, np.inf), np.zeros(S, dtype=int)
        wait, active = np.zeros(S, dtype=int), np.ones(S, dtype=bool)
        if Xva is not None:
            Xv = torch.tensor((Xva - mu) / sd, dtype=torch.float32).expand(S, -1, -1)
            yv = torch.tensor(yva, dtype=torch.float32)
        for ep in range(1, int(ep_target.max()) + 1):
            perm = rng.permutation(len(Xt))
            for i in range(0, len(perm), self.batch):
                b = perm[i:i + self.batch]
                opt.zero_grad()
                pred = self._forward(params, Xt[b].expand(S, -1, -1), True)
                torch.mean((pred - yt[b]) ** 2, dim=1).sum().backward()
                opt.step()
            with torch.no_grad():
                if Xva is not None:
                    val = torch.mean((self._forward(params, Xv, False) * ysd + ymu - yv) ** 2, dim=1).numpy()
                    for s in np.flatnonzero(active):
                        if val[s] < best_val[s] - 1e-12:
                            best_val[s], best_ep[s], wait[s] = val[s], ep, 0
                            best[s] = [p[s].clone() for p in params]
                        else:
                            wait[s] += 1
                            active[s] = wait[s] < self.patience
                    if not active.any():
                        break
                else:
                    for s in np.flatnonzero(ep_target == ep):
                        best[s] = [p[s].clone() for p in params]
        with torch.no_grad():
            for s in range(S):
                if best[s] is None:                     # never improved: keep the last state
                    best[s] = [p[s].clone() for p in params]
            final = [torch.stack([best[s][j] for s in range(S)]) for j in range(len(params))]
        return dict(params=final, mu=mu, sd=sd, ymu=ymu, ysd=ysd, best_val=best_val,
                    best_epoch=np.maximum(best_ep, 1))

    def _pred(self, fitted, X):
        import torch
        with torch.no_grad():
            S = fitted["params"][0].shape[0]
            Xt = torch.tensor((X - fitted["mu"]) / fitted["sd"], dtype=torch.float32).expand(S, -1, -1)
            out = self._forward(fitted["params"], Xt, False).numpy() * fitted["ysd"] + fitted["ymu"]
        return out.mean(axis=0)

    def tune(self, Xtr, ytr, Xva, yva):
        best = None
        for cfg in self.grid:
            fit = self._train(cfg["layers"], list(range(self.n_nets)), Xtr, ytr, self.max_epochs, Xva, yva)
            keep = np.argsort(fit["best_val"])[: self.n_best]
            sub = dict(fit, params=[p[keep] for p in fit["params"]])
            pred = self._pred(sub, Xva)
            v = float(np.var(yva - pred))
            score = qlike_vol(np.exp(yva), to_vol(pred, v))
            if best is None or score < best["score"]:
                best = dict(cfg=cfg, n=int(np.median(fit["best_epoch"][keep])), v=v, score=score,
                            seeds=[int(s) for s in keep], epochs=[int(e) for e in fit["best_epoch"][keep]])
        return best

    def fit(self, X, y, tuned):
        return self._train(tuned["cfg"]["layers"], tuned["seeds"], X, y, tuned["epochs"])

    def predict(self, model, X):
        return self._pred(model, X)


# ---------------------------------------------------------------------------- rolling engine
def rolling_ml(vol: pd.Series, learner, horizon: int, target_kind: str = "point", window: int = 1000,
               refit_every: int = 22, retune_every: int = 252, val_size: int = 250,
               start: Optional[int] = None) -> pd.DataFrame:
    """Rolling direct forecasts of one learner. Returns, indexed by origin date, the realized target
    on the volatility scale (``actual``), the volatility forecast exp(m + v/2) (``forecast``), the
    forecast of log volatility ``m``, the variance ``v`` and the tuned setting."""
    v_arr = np.asarray(vol, dtype=float)
    n = len(v_arr)
    X = learner.features(v_arr)
    y = log_targets(v_arr, horizon, target_kind)
    L = learner.lookback
    start = window if start is None else start
    if start < window or window <= L + horizon + val_size:
        raise ValueError("the first origin needs a full window with room for validation")
    rows, tuned, model = [], None, None
    for p in range(start, n - horizon + 1):
        k = p - start
        lo, hi = p - window + L, p - horizon            # training origins, inclusive
        if tuned is None or k % retune_every == 0:
            va_lo = p - val_size
            tr = np.arange(lo, va_lo - horizon + 1)     # targets end before the validation origins
            va = np.arange(va_lo, hi + 1)
            tuned = learner.tune(X[tr], y[tr], X[va], y[va])
        if model is None or k % refit_every == 0:
            idx = np.arange(lo, hi + 1)
            model = learner.fit(X[idx], y[idx], tuned)
        m = float(np.asarray(learner.predict(model, X[p:p + 1])).ravel()[0])
        rows.append((vol.index[p], float(np.exp(y[p])), float(to_vol(m, tuned["v"])), m, tuned["v"],
                     str(tuned["cfg"]), tuned["n"]))
    out = pd.DataFrame(rows, columns=["date", "actual", "forecast", "m", "v", "config", "n_fit"])
    return out.set_index("date")


LEARNERS: Dict[str, Callable] = {
    "xgb-har": lambda threads=1: XGBHAR(n_jobs=threads),
    "lstm-22": lambda threads=1: LSTMSeq(lookback=22, threads=threads),
    "lstm-252": lambda threads=1: LSTMSeq(lookback=252, threads=threads),
    # Early stopping as in Christensen, Siggaard and Veliyev (2023, Table 15): patience 100, at most
    # 500 epochs.
    "lstm-22-p100": lambda threads=1: LSTMSeq(lookback=22, max_epochs=500, patience=100, threads=threads),
    "ffn-har": lambda threads=1: FFNHAR(threads=threads),
}
