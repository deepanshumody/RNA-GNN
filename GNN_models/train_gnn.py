"""Main GCN graph-classification training script.

Builds candidate-site subgraphs from per-PDB ``*_pos.pkl`` / ``*_neg.pkl``
pickle files (one adjacency/feature dict per binding-site candidate), converts
them to PyTorch Geometric graphs, trains a GCN graph classifier, and saves the
best-performing weights to ``best_model.pth``.

Methodology notes (fixed from the original version):
  * **Structure-level split.** Candidate sites are split by *PDB id*
    (``rna_splits.group_train_val_test_split``), never at random, so near-
    identical sites from the same structure can't leak across train/val/test.
    Named structures in ``HELD_OUT_PDBS`` are removed from the corpus entirely
    and scored separately as a clean held-out set.
  * **Imbalance-aware.** ``BCEWithLogitsLoss`` uses a ``pos_weight`` derived from
    the training-split class ratio, and models are reported with PR-AUC and
    enrichment (``rna_metrics``) alongside ROC-AUC, which is optimistic under
    ~0.07% positives.
  * **Reproducible.** A single seed (``rna_seed.set_seed``) fixes the split and
    initialisation.
"""
import copy
import glob
import os
import pickle
import sys

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import RocCurveDisplay
from torch_geometric.loader import DataLoader
from torch_geometric.utils.convert import from_networkx
from tqdm import tqdm

# Make the repo-root shared modules importable when run as
# ``python GNN_models/train_gnn.py`` from the repository root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gcn_model import GCN_Graph  # noqa: E402
from rna_metrics import binding_site_metrics, format_metrics  # noqa: E402
from rna_seed import set_seed  # noqa: E402
from rna_splits import group_train_val_test_split, split_out_groups  # noqa: E402

# Directory containing the per-PDB candidate-site pickle files.
PICKLE_DIR = "./data/RNA-graph-pickles"
# Structures kept completely out of training and scored as a clean held-out set.
HELD_OUT_PDBS = {"4RUM"}
SEED = 42

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device: {}'.format(device))
set_seed(SEED)


def load_pickle_graphs(path, pdb_id):
    """Load one ``*_pos.pkl`` / ``*_neg.pkl`` into a list of PyG graphs."""
    with open(path, 'rb') as f:
        entries = pickle.load(f)
    graphs = []
    for coord, graph in entries.items():
        G = nx.from_numpy_array(np.array(graph['A1']))
        for node in G.nodes:
            G.nodes[node]['x'] = torch.tensor(graph['H'][node], dtype=torch.int32)
        pyg_graph = from_networkx(G)
        pyg_graph.y = torch.tensor(graph['C'])
        pyg_graph.coords = coord
        pyg_graph.pdb = pdb_id
        graphs.append(pyg_graph)
    return graphs


# ---- Load every candidate-site graph, tagged with its source structure. ----
data_list = []
for path in sorted(glob.glob(os.path.join(PICKLE_DIR, "*_pos.pkl"))) + \
        sorted(glob.glob(os.path.join(PICKLE_DIR, "*_neg.pkl"))):
    pdb_id = os.path.basename(path)[:4].upper()
    data_list.extend(load_pickle_graphs(path, pdb_id))

if not data_list:
    raise SystemExit(
        f"No graphs found in {PICKLE_DIR}. Generate them with "
        f"dataset_creation/gnn_rna.py (only the 1FUF demo pickle ships in this repo)."
    )

# ---- Hold named structures out of the corpus entirely (no leakage). ----
groups = [g.pdb for g in data_list]
kept_idx, held_idx = split_out_groups(groups, HELD_OUT_PDBS)
held_out_list = [data_list[i] for i in held_idx]
data_list = [data_list[i] for i in kept_idx]
print(f"{len(data_list)} candidate sites across {len(set(groups)) - len(HELD_OUT_PDBS & set(groups))} "
      f"structures; {len(held_out_list)} held-out sites ({sorted(HELD_OUT_PDBS)}).")

# ---- Structure-level train/val/test split (group = PDB id). ----
kept_groups = [g.pdb for g in data_list]
train_idx, valid_idx, test_idx = group_train_val_test_split(kept_groups, (0.8, 0.1, 0.1), seed=SEED)
train_data = [data_list[i] for i in train_idx]
valid_data = [data_list[i] for i in valid_idx]
test_data = [data_list[i] for i in test_idx]

# Feature width comes from the data, so an extended atom vocabulary just works.
num_features = int(data_list[0].x.shape[1])

train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=0)
valid_loader = DataLoader(valid_data, batch_size=32, shuffle=False, num_workers=0)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=0)
held_out_loader = DataLoader(held_out_list, batch_size=32, shuffle=False, num_workers=0) \
    if held_out_list else None

args = {
    'device': device,
    'num_layers': 5,
    'hidden_dim': 256,
    'dropout': 0.5,
    'lr': 0.001,
    'epochs': 30,
}


def train(model, device, data_loader, optimizer, loss_fn):
    model.train()
    loss = torch.tensor(0.0)
    for batch in tqdm(data_loader, desc="Iteration"):
        batch = batch.to(device)
        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            continue
        is_labeled = batch.y == batch.y
        optimizer.zero_grad()
        out = model(batch)
        loss = loss_fn(out[is_labeled], batch.y[is_labeled].float().unsqueeze(1))
        loss.backward()
        optimizer.step()
    return loss.item()


def eval(model, device, loader, save_file=None):
    """Return imbalance-aware metrics; optionally dump predictions + ROC curve."""
    model.eval()
    y_true, y_pred, pdb_id, coord = [], [], [], []
    for batch in tqdm(loader, desc="Iteration"):
        batch = batch.to(device)
        if batch.x.shape[0] == 1:
            continue
        with torch.no_grad():
            pred = model(batch)
        y_true.append(batch.y.view(pred.shape).detach().cpu())
        y_pred.append(pred.detach().cpu())
        pdb_id.append(batch.pdb)
        coord = coord + list(batch.coords)

    y_true = torch.cat(y_true, dim=0).numpy().reshape(-1)
    y_pred = torch.cat(y_pred, dim=0).numpy().reshape(-1)
    metrics = binding_site_metrics(y_true, y_pred)

    if save_file is not None:
        df = pd.DataFrame({
            'y_pred': y_pred,
            'y_true': y_true,
            'pdb_id': np.array(np.concatenate(pdb_id).flat),
            'coord': np.array(coord).reshape(-1),
        })
        df.to_csv('preds_RNA' + save_file + '.csv', sep=',', index=False)
        if metrics["n_pos"] and metrics["n_pos"] < metrics["n_total"]:
            RocCurveDisplay.from_predictions(y_true, y_pred)
            plt.savefig("roc_curve.png")
            plt.close()
    return metrics


# ---- Imbalance-aware loss: weight positives by the training class ratio. ----
n_pos = sum(int(g.y.item() == 1) for g in train_data)
n_neg = len(train_data) - n_pos
pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float, device=device)
print(f"train positives={n_pos} negatives={n_neg} pos_weight={pos_weight.item():.1f}")

model = GCN_Graph(args['hidden_dim'], 1, args['num_layers'], args['dropout'],
                  num_features=num_features).to(device)
model.reset_parameters()
optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

best_model = None
best_valid_auc = 0
for epoch in range(1, 1 + args["epochs"]):
    loss = train(model, device, train_loader, optimizer, loss_fn)
    valid_metrics = eval(model, device, valid_loader)
    valid_auc = valid_metrics["roc_auc"]
    if valid_auc == valid_auc and valid_auc > best_valid_auc:  # not NaN and improved
        best_valid_auc = valid_auc
        best_model = copy.deepcopy(model)
    print(f"Epoch {epoch:02d} | loss {loss:.4f} | "
          f"valid ROC-AUC {valid_auc:.3f} PR-AUC {valid_metrics['pr_auc']:.3f}")

best_model = best_model if best_model is not None else model
torch.save(best_model.state_dict(), "./best_model.pth")

print("\n=== Best model ===")
print("train   :", format_metrics(eval(best_model, device, train_loader, save_file="train")))
print("valid   :", format_metrics(eval(best_model, device, valid_loader, save_file="valid")))
print("test    :", format_metrics(eval(best_model, device, test_loader, save_file="test")))
if held_out_loader is not None:
    print("HELD-OUT:", format_metrics(eval(best_model, device, held_out_loader, save_file="heldout")))
