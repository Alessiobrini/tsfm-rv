"""Training helpers shared by the TTM and Sundial fine-tuning scripts: learning-rate warmup, early
stopping on the validation loss, and resumable checkpoints."""
import os

import numpy as np
import torch


def lr_lambda(step: int, warmup_steps: int) -> float:
    """Multiplier of the learning rate: linear warmup over `warmup_steps`, then 1."""
    if warmup_steps <= 0 or step >= warmup_steps:
        return 1.0
    return (step + 1) / warmup_steps


class EarlyStopper:
    """Stop after `patience` validations without an improvement larger than `min_delta`
    (patience <= 0 never stops). `update` returns True for a new best."""

    def __init__(self, patience: int = 0, min_delta: float = 0.0):
        self.patience, self.min_delta = patience, min_delta
        self.best, self.num_bad = float("inf"), 0

    def update(self, value: float) -> bool:
        improved = value < self.best - self.min_delta
        if improved:
            self.best, self.num_bad = value, 0
        else:
            self.num_bad += 1
        return improved

    @property
    def should_stop(self) -> bool:
        return self.patience > 0 and self.num_bad >= self.patience


def trainable_state(model) -> dict:
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}


def restore_trainable(model, state: dict) -> None:
    own = dict(model.named_parameters())
    for n, t in state.items():
        if n in own:
            own[n].data.copy_(t.to(own[n].device))


def save_resume_state(path, model, optimizer, scheduler, stopper, step, best_state) -> None:
    payload = {"step": step, "trainable": trainable_state(model), "optimizer": optimizer.state_dict(),
               "scheduler": scheduler.state_dict(), "stopper_best": stopper.best,
               "stopper_num_bad": stopper.num_bad, "best_state": best_state,
               "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state()}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(payload, path + ".tmp")
    os.replace(path + ".tmp", path)


def load_resume_state(path, model, optimizer, scheduler, stopper):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    restore_trainable(model, payload["trainable"])
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    stopper.best, stopper.num_bad = payload["stopper_best"], payload["stopper_num_bad"]
    torch.set_rng_state(payload["torch_rng"])
    np.random.set_state(payload["numpy_rng"])
    return payload["step"], payload["best_state"]


def pick_device(want: str) -> str:
    if want == "cuda" and torch.cuda.is_available():
        return "cuda"
    if want == "mps" and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
