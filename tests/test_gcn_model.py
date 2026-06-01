"""Tests for the shared GCN model.

Requires torch / torch-geometric (skipped where absent). The checkpoint-loading
test is the important one: it guarantees the de-duplicated model still loads the
released ``best_model.pth``, so the Streamlit demo keeps working.
"""
import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from torch_geometric.data import Data  # noqa: E402

import gcn_model  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _one_graph(num_features):
    x = torch.randint(0, 2, (6, num_features), dtype=torch.long)
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 0], [1, 2, 3, 4, 5, 5]], dtype=torch.long)
    data = Data(x=x, edge_index=edge_index)
    data.batch = torch.zeros(6, dtype=torch.long)
    return data


def test_default_width_forward():
    model = gcn_model.GCN_Graph(hidden_dim=16, output_dim=1, num_layers=3, dropout=0.0).eval()
    with torch.no_grad():
        out = model(_one_graph(gcn_model.DEFAULT_NUM_FEATURES))
    assert tuple(out.shape) == (1, 1)


def test_custom_width_forward():
    model = gcn_model.GCN_Graph(hidden_dim=16, output_dim=1, num_layers=3, dropout=0.0, num_features=60).eval()
    with torch.no_grad():
        out = model(_one_graph(60))
    assert tuple(out.shape) == (1, 1)


def test_released_checkpoint_still_loads():
    ckpt = os.path.join(ROOT, "best_model.pth")
    if not os.path.exists(ckpt):
        pytest.skip("best_model.pth not present")
    # Same hyperparameters the checkpoint was trained with.
    model = gcn_model.GCN_Graph(hidden_dim=256, output_dim=1, num_layers=5, dropout=0.5)
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state)  # must not raise (demo depends on this)
