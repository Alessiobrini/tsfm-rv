"""Fine-tune TTM or Sundial on a panel of daily realized variance of US stocks.

Both models receive volatility, the square root of realized variance, with the context of the
paper's zero-shot runs: 512 days for TTM (the r2.1 checkpoint pretrained with daily series) and
1,000 days for Sundial. Windows end on or before 2020-12-31 for training, and validation windows end
in 2021, at least 22 trading days before 2022-01-01 (finetune/windows.py), so no target falls in the
period on which the fine-tuned models are evaluated, from 2022 on. The panel can include the
stocks of the evaluation, whose data up to 2021 then enter training.

  TTM      all weights trained on the mean squared error of the 96-day forecast path, with the daily
           frequency token, as the model is pretrained.
  Sundial  LoRA adapters (rank 16, alpha 32, dropout 0.05) on the query, key, value and output
           projections of the attention layers, trained on the model's flow-matching loss for the
           96 days after the context. The adapters are merged into the saved checkpoint.

Training is step-driven: AdamW with learning rate 1e-4 and a linear warmup over 500 steps, windows
taken every 5 days, a validation loss every 1,000 steps on a fixed subset of the validation windows,
early stopping after 6 validations without improvement or at 40,000 steps, and the weights with the
best validation loss restored before saving. Interrupted runs continue with --resume.

Usage (cluster, GPU):
  python finetune/train.py --model ttm --panel <panel.parquet> --out <checkpoint dir>
  python finetune/train.py --model sundial --panel <panel.parquet> --out <checkpoint dir>
"""
import argparse
import glob
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import MODEL_REVISIONS, RANDOM_SEED  # noqa: E402
from finetune.common import (EarlyStopper, load_resume_state, lr_lambda, pick_device,  # noqa: E402
                             restore_trainable, save_resume_state, trainable_state)
from finetune.windows import RVWindowDataset  # noqa: E402

TTM_ID = "ibm-granite/granite-timeseries-ttm-r2"
SUNDIAL_ID = "thuml/sundial-base-128m"
TTM_DAILY_FREQ_TOKEN = 8
DEFAULTS = {"ttm": dict(context_length=512, batch_size=64, val_max_windows=8192),
            "sundial": dict(context_length=1000, batch_size=16, val_max_windows=4096)}


# ---------------------------------------------------------------------------------------- TTM
def load_ttm(context_length: int, prediction_length: int):
    from tsfm_public import get_model
    return get_model(model_path=TTM_ID, context_length=context_length,
                     prediction_length=prediction_length, freq="D")


def ttm_loss(model, past, future, device):
    tok = torch.full((past.shape[0], 1), TTM_DAILY_FREQ_TOKEN, dtype=torch.long, device=device)
    return model(past_values=past.to(device), future_values=future.to(device), freq_token=tok).loss


# ------------------------------------------------------------------------------------ Sundial
def load_sundial():
    from transformers import AutoModelForCausalLM, DynamicCache
    if not hasattr(DynamicCache, "get_max_length"):
        DynamicCache.get_max_length = (DynamicCache.get_max_cache_shape
                                       if hasattr(DynamicCache, "get_max_cache_shape") else lambda self: None)
    return AutoModelForCausalLM.from_pretrained(SUNDIAL_ID, revision=MODEL_REVISIONS.get(SUNDIAL_ID),
                                                trust_remote_code=True, torch_dtype=torch.float32)


def sundial_loss_inputs(past, future, input_token_len: int, output_token_len: int):
    """Inputs of Sundial's training forward for a context of any length L and H target days.

    Sundial splits its input into patches of `input_token_len` days, padding the context on the
    left to a whole number of patches, as it does at inference. The hidden state of patch i forecasts
    the `output_token_len` days that follow it, read from `labels` at offset i * input_token_len.
    Laying out `labels` as the padded series shifted by one patch makes the window of the last patch
    start on the first day after the context. Only the last patch and its first H days enter the
    loss, which supervises the forecast the model issues at inference.
    """
    B, L, _ = past.shape
    H = future.shape[1]
    it, ot = input_token_len, output_token_len
    if not 1 <= H <= ot:
        raise ValueError(f"prediction length {H} must be between 1 and {ot}")
    pad = (it - L % it) % it
    n_patches = (L + pad) // it
    series = torch.cat([torch.zeros(B, pad), past.squeeze(-1), future.squeeze(-1)], dim=1)
    labels = torch.zeros(B, L + pad - it + ot)
    shifted = series[:, it:]
    labels[:, :shifted.shape[1]] = shifted
    loss_masks = torch.zeros(B, n_patches)
    loss_masks[:, -1] = 1.0
    mask_y = torch.zeros(B, ot)
    mask_y[:, :H] = 1.0
    return past.squeeze(-1), labels, loss_masks, mask_y


def sundial_loss(model, past, future, device):
    cfg = model.config
    ids, labels, lm, my = sundial_loss_inputs(past, future, cfg.input_token_len, cfg.output_token_lens[-1])
    out = model(input_ids=ids.to(device), labels=labels.to(device), loss_masks=lm.to(device),
                mask_y=my.to(device), revin=True)
    return out.loss


def add_lora(model):
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    return get_peft_model(model, cfg)


def copy_remote_code(model, out_dir: str) -> None:
    """Copy Sundial's model code into the checkpoint, so it loads without the hub cache."""
    src = os.path.dirname(sys.modules[type(model).__module__].__file__)
    for py in glob.glob(os.path.join(src, "*.py")):
        dst = os.path.join(out_dir, os.path.basename(py))
        if not os.path.exists(dst):
            shutil.copy2(py, dst)


# --------------------------------------------------------------------------------------- loop
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["ttm", "sundial"])
    ap.add_argument("--panel", required=True, help="parquet with columns permno, date, rv5")
    ap.add_argument("--out", required=True)
    ap.add_argument("--context-length", type=int, default=None)
    ap.add_argument("--prediction-length", type=int, default=96)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--val-max-windows", type=int, default=None)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup-steps", type=int, default=500)
    ap.add_argument("--max-steps", type=int, default=40000)
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--max-names", type=int, default=None, help="smoke tests only")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    for k, v in DEFAULTS[a.model].items():
        if getattr(a, k) is None:
            setattr(a, k, v)

    device = pick_device(a.device)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    panel = pd.read_parquet(a.panel, columns=["permno", "date", "rv5"])
    if a.max_names:
        keep = panel.groupby("permno").size().sort_values(ascending=False).index[: a.max_names]
        panel = panel[panel.permno.isin(keep)]
    train_ds = RVWindowDataset(panel, "train", a.context_length, a.prediction_length, a.stride)
    val_ds = RVWindowDataset(panel, "val", a.context_length, a.prediction_length, a.stride)
    print(f"[{a.model}] device={device} train windows={len(train_ds):,} val windows={len(val_ds):,} "
          f"names={len(train_ds.series):,}", flush=True)

    if a.model == "ttm":
        model, loss_fn = load_ttm(a.context_length, a.prediction_length), ttm_loss
    else:
        model, loss_fn = add_lora(load_sundial()), sundial_loss
    model.to(device).train()
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{a.model}] trainable parameters={n_trainable:,}", flush=True)

    idx = np.random.RandomState(a.seed).permutation(len(val_ds))[: a.val_max_windows]
    val_loader = DataLoader(Subset(val_ds, sorted(int(i) for i in idx)), batch_size=a.batch_size)

    def validate() -> float:
        # Sundial's flow-matching loss draws noise at every forward, so the validation draws are
        # fixed to make the losses comparable across steps; the training draws are left untouched.
        rng = torch.get_rng_state()
        torch.manual_seed(a.seed + 1)
        model.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for b in val_loader:
                tot += loss_fn(model, b["past_values"], b["future_values"], device).item() * len(b["past_values"])
                n += len(b["past_values"])
        model.train()
        torch.set_rng_state(rng)
        return tot / max(n, 1)

    loader = DataLoader(train_ds, batch_size=a.batch_size, shuffle=True, drop_last=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_lambda(s, a.warmup_steps))
    stopper = EarlyStopper(patience=a.patience)
    resume = os.path.join(a.out, "_resume", "state.pt")
    step, best_state = 0, None
    if a.resume and os.path.exists(resume):
        step, best_state = load_resume_state(resume, model, opt, sched, stopper)
        print(f"[{a.model}] resumed at step {step}, best validation loss {stopper.best:.6f}", flush=True)

    t0, done = time.time(), step >= a.max_steps
    while not done:
        for b in loader:
            loss = loss_fn(model, b["past_values"], b["future_values"], device)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            if step == 1 or step % 100 == 0:
                print(f"  step {step} loss {loss.item():.6f} ({(time.time() - t0) / step:.3f} s/step)", flush=True)
            if step % a.val_every == 0:
                v = validate()
                if stopper.update(v):
                    best_state = trainable_state(model)
                save_resume_state(resume, model, opt, sched, stopper, step, best_state)
                print(f"[{a.model}] step {step} validation {v:.6f} best {stopper.best:.6f} "
                      f"bad {stopper.num_bad}/{a.patience}", flush=True)
                if stopper.should_stop:
                    done = True
                    break
            if step >= a.max_steps:
                done = True
                break

    if best_state is not None:
        restore_trainable(model, best_state)
    final = validate()
    print(f"[{a.model}] validation loss of the saved weights {final:.6f}", flush=True)
    os.makedirs(a.out, exist_ok=True)
    save = model.merge_and_unload() if a.model == "sundial" else model
    save.save_pretrained(a.out)
    if a.model == "sundial":
        copy_remote_code(save, a.out)
    meta = dict(vars(a), base=TTM_ID if a.model == "ttm" else SUNDIAL_ID,
                base_revision=None if a.model == "ttm" else MODEL_REVISIONS.get(SUNDIAL_ID),
                input="volatility", trainable_params=n_trainable, train_windows=len(train_ds),
                val_windows=len(val_ds), names=len(train_ds.series), steps=step, best_val=stopper.best,
                final_val=final, early_stopped=stopper.should_stop)
    json.dump(meta, open(os.path.join(a.out, "ft_meta.json"), "w"), indent=2, default=str)
    print(f"[{a.model}] saved {a.out}", flush=True)


if __name__ == "__main__":
    main()
