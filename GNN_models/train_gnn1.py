"""GCN training variant: precomputed tensors, all negatives.

Like ``train_gnn2.py`` but loads every negative graph (no subsampling) and uses
large batches. The same methodology fixes apply: structure-level (PDB-grouped)
split, a held-out structure kept out of the corpus, imbalance-aware metrics and
loss weighting, and a fixed seed.
"""
import copy
import glob
import os
import sys

import torch
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gcn_model import GCN_Graph  # noqa: E402
from rna_metrics import binding_site_metrics, format_metrics  # noqa: E402
from rna_seed import set_seed  # noqa: E402
from rna_splits import group_train_val_test_split  # noqa: E402

NONREDUNDANT = "nonredundantRNA.txt"
TENSOR_DIR = "./saved_tensors/"
HELD_OUT_PATH = "pd4rum_graphs.pt"
SEED = 42
BATCH = 2048

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device: {}'.format(device))
set_seed(SEED)


def pdb_from_tensor_path(path):
    name = os.path.basename(path)
    return name.replace("pos_graph.pt", "").replace("neg_graph.pt", "").upper()


def tag(graphs, pdb_id):
    for g in graphs:
        g.pdb = pdb_id
    return graphs


with open(NONREDUNDANT) as f:
    lines = [line.strip() for line in f if line.strip()]

data_list = []
for line in lines:
    for path in glob.glob(TENSOR_DIR + line + "neg_graph.pt") + glob.glob(TENSOR_DIR + line + "pos_graph.pt"):
        data_list.extend(tag(torch.load(path), pdb_from_tensor_path(path)))

if not data_list:
    raise SystemExit(f"No tensors found in {TENSOR_DIR}; build them before training.")

groups = [g.pdb for g in data_list]
train_idx, valid_idx, test_idx = group_train_val_test_split(groups, (0.8, 0.1, 0.1), seed=SEED)
train_data = [data_list[i] for i in train_idx]
valid_data = [data_list[i] for i in valid_idx]
test_data = [data_list[i] for i in test_idx]
num_features = int(data_list[0].x.shape[1])

held_out_list = torch.load(HELD_OUT_PATH) if os.path.exists(HELD_OUT_PATH) else []
train_loader = DataLoader(train_data, batch_size=BATCH, shuffle=True, num_workers=0)
valid_loader = DataLoader(valid_data, batch_size=BATCH, shuffle=False, num_workers=0)
test_loader = DataLoader(test_data, batch_size=BATCH, shuffle=False, num_workers=0)
held_out_loader = DataLoader(held_out_list, batch_size=BATCH, shuffle=False, num_workers=0) \
    if held_out_list else None

args = {'num_layers': 5, 'hidden_dim': 256, 'dropout': 0.5, 'lr': 0.001, 'epochs': 30}


def train(model, loader, optimizer, loss_fn):
    model.train()
    loss = torch.tensor(0.0)
    for batch in loader:
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


def eval(model, loader):
    model.eval()
    y_true, y_pred = [], []
    for batch in loader:
        batch = batch.to(device)
        if batch.x.shape[0] == 1:
            continue
        with torch.no_grad():
            pred = model(batch)
        y_true.append(batch.y.view(pred.shape).detach().cpu())
        y_pred.append(pred.detach().cpu())
    y_true = torch.cat(y_true, dim=0).numpy().reshape(-1)
    y_pred = torch.cat(y_pred, dim=0).numpy().reshape(-1)
    return binding_site_metrics(y_true, y_pred)


n_pos = sum(int(g.y.item() == 1) for g in train_data)
pos_weight = torch.tensor([(len(train_data) - n_pos) / max(n_pos, 1)], dtype=torch.float, device=device)
model = GCN_Graph(args['hidden_dim'], 1, args['num_layers'], args['dropout'],
                  num_features=num_features).to(device)
model.reset_parameters()
optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

best_model, best_valid_auc = None, 0
for epoch in range(1, 1 + args["epochs"]):
    loss = train(model, train_loader, optimizer, loss_fn)
    vm = eval(model, valid_loader)
    if vm["roc_auc"] == vm["roc_auc"] and vm["roc_auc"] > best_valid_auc:
        best_valid_auc = vm["roc_auc"]
        best_model = copy.deepcopy(model)
    print(f"Epoch {epoch:02d} | loss {loss:.4f} | valid ROC-AUC {vm['roc_auc']:.3f} PR-AUC {vm['pr_auc']:.3f}")

best_model = best_model if best_model is not None else model
torch.save(best_model.state_dict(), "./best_model1.pth")
print("test    :", format_metrics(eval(best_model, test_loader)))
if held_out_loader is not None:
    print("HELD-OUT:", format_metrics(eval(best_model, held_out_loader)))
