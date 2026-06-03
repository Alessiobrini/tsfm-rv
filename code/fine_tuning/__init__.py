"""
fine_tuning/ — Shared utilities for aligned TSFM fine-tuning experiments.

This package intentionally separates the experiment protocol (rolling refits,
window generation, metric selection, output layout) from model-specific
adapter code. The goal is to keep all TSFMs on the same evaluation design
while allowing different implementation details under the hood.
"""

