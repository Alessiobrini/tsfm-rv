"""Training and validation windows for fine-tuning, from a panel of daily realized variance.

The panel holds one series per stock (`permno`, `date`, `rv5`). Each window pairs a context of
`context_length` days with the next `prediction_length` days. A window belongs to the training set
when its last target day is on or before TRAIN_END, and to the validation set when its last target
day falls after TRAIN_END and at least PURGE_DAYS trading days before EMBARGO, the first day of the
evaluation period. No window therefore has a target on or after EMBARGO.

The input is volatility, the square root of realized variance, as in the paper's zero-shot runs.
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

TRAIN_END = pd.Timestamp("2020-12-31")
EMBARGO = pd.Timestamp("2022-01-01")
PURGE_DAYS = 22


def purge_cutoff(dates: np.ndarray, embargo: pd.Timestamp = EMBARGO, purge_days: int = PURGE_DAYS) -> pd.Timestamp:
    """The last allowed target day of a validation window: `purge_days` trading days of the panel's
    calendar before `embargo`."""
    cal = np.unique(np.asarray(dates).astype("datetime64[D]"))
    j = int(np.searchsorted(cal, np.datetime64(embargo.date()))) - purge_days
    if j <= 0:
        return pd.Timestamp(cal[0]) - pd.Timedelta(days=1)
    return pd.Timestamp(cal[min(j - 1, len(cal) - 1)])


def series_values(rv: np.ndarray, value: str) -> np.ndarray:
    if value == "vol":
        return np.sqrt(rv)
    if value == "log_rv":
        return np.log(rv)
    raise ValueError(f"value must be 'vol' or 'log_rv', got {value!r}")


class RVWindowDataset(Dataset):
    def __init__(self, panel: pd.DataFrame, split: str, context_length: int, prediction_length: int,
                 stride: int = 1, value: str = "vol"):
        if split not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")
        df = panel[["permno", "date", "rv5"]].copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df[np.isfinite(df["rv5"]) & (df["rv5"] > 0)]
        cut = purge_cutoff(df["date"].values)
        L, H = context_length, prediction_length
        self.series, self.index = {}, []
        for permno, g in df.sort_values(["permno", "date"]).groupby("permno"):
            d = g["date"].values
            v = series_values(g["rv5"].to_numpy(dtype=np.float64), value).astype(np.float32)
            self.series[permno] = v
            for t in range(L, len(v) - H + 1, stride):
                fe = pd.Timestamp(d[t + H - 1])
                ok = fe <= TRAIN_END if split == "train" else (TRAIN_END < fe <= cut and fe < EMBARGO)
                if ok:
                    self.index.append((permno, t))
        self.L, self.H = L, H

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict:
        permno, t = self.index[i]
        v = self.series[permno]
        return {"past_values": torch.from_numpy(np.ascontiguousarray(v[t - self.L:t])).float().unsqueeze(-1),
                "future_values": torch.from_numpy(np.ascontiguousarray(v[t:t + self.H])).float().unsqueeze(-1)}
