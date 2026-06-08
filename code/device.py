"""device.py — local compute-device selection for TSFM inference.

The project moved from the SLURM GPU cluster to local execution on an Apple-Silicon
MacBook. This picks the best available torch device: MPS (Metal) when available,
then CUDA, else CPU. Enable PYTORCH_ENABLE_MPS_FALLBACK=1 (the run scripts set it)
so the few ops without an MPS kernel fall back to CPU instead of erroring.
"""


def get_device() -> str:
    """Return 'mps', 'cuda', or 'cpu' (best available)."""
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"
