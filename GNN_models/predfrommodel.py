"""GCN single-structure inference.

Loads ``best_model.pth`` and scores every candidate site of one RNA structure,
writing per-site predictions to a CSV and reporting imbalance-aware metrics
(ROC-AUC, PR-AUC, enrichment) rather than ROC-AUC alone.

NOTE ON 1FUF: the bundled ``1fuf.pkl`` is the demo structure, and 1FUF is part
of the non-redundant training list (``data/nonredundantRNA.txt``). Scoring it
here is therefore an **in-sample** sanity check, not a held-out generalisation
estimate. For an honest number, train with ``train_gnn.py`` (which holds whole
structures out) and read its ``HELD-OUT`` line.
"""
import os
import pickle
import sys

import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.utils.convert import from_networkx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gcn_model import GCN_Graph  # noqa: E402
from rna_metrics import binding_site_metrics, format_metrics  # noqa: E402

DATA_FILE = "./data/RNA-graph-pickles/1fuf.pkl"
CHECKPOINT = "./best_model.pth"
PDB_ID = "1FUF"

device = 'cuda' if torch.cuda.is_available() else 'cpu'

with open(DATA_FILE, 'rb') as f:
    entries = pickle.load(f)

graphs = []
for coord, graph in entries.items():
    G = nx.from_numpy_array(np.array(graph['A1']))
    for node in G.nodes:
        G.nodes[node]['x'] = torch.tensor(graph['H'][node], dtype=torch.int32)
    pyg_graph = from_networkx(G)
    pyg_graph.y = torch.tensor(graph['C'])
    pyg_graph.coords = coord
    pyg_graph.pdb = PDB_ID
    graphs.append(pyg_graph)

num_features = int(graphs[0].x.shape[1])
loader = DataLoader(graphs, batch_size=32, shuffle=False, num_workers=0)

model = GCN_Graph(hidden_dim=256, output_dim=1, num_layers=5, dropout=0.5,
                  num_features=num_features).to(device)
model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
model.eval()

y_true, y_pred, coords = [], [], []
for batch in loader:
    batch = batch.to(device)
    if batch.x.shape[0] == 1:
        continue
    with torch.no_grad():
        pred = model(batch)
    y_true.append(batch.y.view(pred.shape).detach().cpu())
    y_pred.append(pred.detach().cpu())
    coords = coords + list(batch.coords)

y_true = torch.cat(y_true, dim=0).numpy().reshape(-1)
y_pred = torch.cat(y_pred, dim=0).numpy().reshape(-1)

pd.DataFrame({'y_pred': y_pred, 'y_true': y_true, 'coord': np.array(coords).reshape(-1)}) \
    .to_csv('preds_RNA' + PDB_ID + 'f.csv', sep=',', index=False)

metrics = binding_site_metrics(y_true, y_pred)
print(f"{PDB_ID} (in-sample):", format_metrics(metrics))
