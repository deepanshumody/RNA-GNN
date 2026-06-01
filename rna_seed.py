"""Deterministic seeding for reproducible RNA-GNN runs.

The original training / evaluation scripts shuffled data and initialised models
without fixing any random seed, so the train/val/test split and the reported
metrics changed from run to run. Call :func:`set_seed` once at the top of a
script to make a run reproducible.

PyTorch is imported lazily so this module stays importable in environments that
only have the scientific-Python stack (e.g. lightweight CI).
"""
import os
import random

import numpy as np


def set_seed(seed=42):
    """Seed Python, NumPy and (if installed) PyTorch RNGs. Returns ``seed``."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed
