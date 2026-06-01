"""Smoke test: the GCN graph classifier runs a forward pass and returns one
score per graph.

Requires torch / torch-geometric / streamlit-molstar (the demo app's deps), so
it is skipped automatically in environments that don't have them installed
(e.g. lightweight CI). It runs locally where the full stack is present.
"""
import pytest


def test_gcn_graph_forward_shape():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    pytest.importorskip("streamlit_molstar")
    from torch_geometric.data import Data
    import moleculestreamlit as app

    model = app.GCN_Graph(hidden_dim=32, output_dim=1, num_layers=3, dropout=0.0)
    model.eval()

    n_nodes = 8
    # AtomEncoder embeds 56 binary feature columns, so node features are 0/1.
    x = torch.randint(0, 2, (n_nodes, 56), dtype=torch.long)
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 0], [1, 2, 3, 4, 5, 6, 7, 7]], dtype=torch.long
    )
    data = Data(x=x, edge_index=edge_index)
    data.batch = torch.zeros(n_nodes, dtype=torch.long)

    with torch.no_grad():
        out = model(data)

    # One graph in, one binding score out.
    assert tuple(out.shape) == (1, 1)
