"""Tests for the GNN-DTI model: the focal-loss fix and width-agnostic graph
embedding / collation.

Requires torch (+ scipy, pulled in by the GNN-DTI utils). Skipped where absent.
"""
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("scipy")

# The GNN-DTI modules use sibling imports (``from utils import *``), so put their
# directory on the path before importing them.
GNN_DTI_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "GNN_models", "GNN-DTI"
)
sys.path.insert(0, GNN_DTI_DIR)

import torch.nn.functional as F  # noqa: E402

from gnn import FocalLoss, gnn  # noqa: E402
from dataset import collate_fn  # noqa: E402


def test_focal_reduces_to_alpha_bce_when_gamma_zero():
    loss = FocalLoss()
    logits = torch.tensor([2.0, -1.0, 0.5, -3.0])
    targets = torch.tensor([1.0, 0.0, 1.0, 0.0])
    focal = loss(logits, targets, alpha=1.0, gamma=0.0)
    bce = F.binary_cross_entropy(torch.sigmoid(logits), targets, reduction="mean")
    assert torch.allclose(focal, bce, atol=1e-6)


def test_focal_downweights_an_easy_confident_batch():
    loss = FocalLoss()
    logits = torch.tensor([6.0, -6.0])  # confident and correct
    targets = torch.tensor([1.0, 0.0])
    focal = float(loss(logits, targets, alpha=1.0, gamma=2.0))
    bce = float(F.binary_cross_entropy(torch.sigmoid(logits), targets, reduction="mean"))
    assert focal < bce  # the whole point of focal loss


def test_focal_weights_a_hard_example_above_an_easy_one():
    loss = FocalLoss()
    easy = float(loss(torch.tensor([8.0]), torch.tensor([1.0]), gamma=2.0))   # confident-correct
    hard = float(loss(torch.tensor([-8.0]), torch.tensor([1.0]), gamma=2.0))  # confident-wrong
    assert hard > easy


@pytest.mark.parametrize("n_atom_features", [28, 30])
def test_model_and_collate_are_width_agnostic(n_atom_features):
    width = 2 * n_atom_features
    batch = []
    for natom in (5, 7):
        batch.append(
            {
                "H": np.random.rand(natom, width),
                "A1": np.eye(natom),
                "A2": np.random.rand(natom, natom),
                "C": 1.0,
                "V": np.ones(natom),
                "key": "k",
            }
        )
    H, A1, A2, C, V, keys = collate_fn(batch)
    assert H.shape[-1] == width  # width inferred from data, not hard-coded

    args = SimpleNamespace(
        n_graph_layer=2, d_graph_layer=12, n_FC_layer=2, d_FC_layer=8,
        dropout_rate=0.0, initial_mu=4.0, initial_dev=1.0, n_atom_features=n_atom_features,
    )
    model = gnn(args)
    out = model.train_model((H, A1, A2, V))
    assert out.shape[0] == 2  # one score per graph
