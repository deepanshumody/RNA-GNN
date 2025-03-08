import streamlit as st
import pickle
import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from torch_geometric.utils.convert import from_networkx
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
from tqdm import tqdm
from ogb.graphproppred import Evaluator
import matplotlib.pyplot as plt
from torch_geometric.nn import GCNConv
from sklearn.metrics import roc_curve

# For merging predicted coordinates into a PDB file
from createpredictedionpdb import write_merged_pdb_with_hetatms

# Mol* viewer for Streamlit
from streamlit_molstar import st_molstar

##################################################
#                   Constants
##################################################
pdb_clean_file = "./data/RNA-only-PDB-clean/1fuf_clean.pdb"  # Path to "cleaned" PDB (RNA atoms only)
output_merged_file = "1fuf_merged.pdb"                       # Output path for the merged PDB
checkpoint_path = "./best_model.pth"                         # Model checkpoint
data_file = "./data/RNA-graph-pickles/1fuf.pkl"              # Pickle containing graph data
pdb_file_path = "./data/RNA-only-PDB/1fuf.pdb"               # Raw PDB file (with actual ions)

##################################################
#          High-Level Description Text
##################################################
description_text = """
# RNA–Metal Ion Binding Predictor

This application demonstrates how a **Graph Neural Network (GNN)** can be used to predict
possible metal ion binding sites in RNA structures. 

### How It Works

1. **Graph Construction**  
   - RNA structures from PDB files are read and processed via RDKit.  
   - Each candidate site (potential metal ion coordinate) is converted into a **graph**:
     - **Nodes** represent atoms (with their associated chemical features).
     - **Edges** come from adjacency (bonds) or spatial proximity.
   - This way, local neighborhoods around each candidate coordinate are captured.

2. **Model Architecture (GNN)**  
   - We use a **GCN** (Graph Convolutional Network) that takes node features and adjacency as input.  
   - The GCN learns to output a score indicating whether a coordinate patch is likely to bind a metal ion (e.g., Mg^2+).  
   - During training, known binding sites from crystal structures are labeled as positives, and non-binding sites as negatives.

3. **Inference & Visualization**  
   - After training, the model scores each candidate site in a new RNA structure.  
   - High scores indicate likely metal ion binding.  
   - We **merge** these predictions back into the original PDB as new HETATM records, allowing direct 3D visualization of predicted ions next to the real RNA structure.

### Using This App

- **Load Model & Data**: The script automatically loads a pre-trained GNN model (`best_model.pth`).
- **Inference**: We evaluate the GNN on the 1FUF RNA structure, applying a threshold to decide which coordinates are “positives.”
- **3D Visualization**: We display both the original PDB (with real ions) and a merged PDB (with predicted ions) side by side.

---

Continue below to see everything in action!
"""

##################################################
#              Model Definitions
##################################################
full_atom_feature_dims = [2 for _ in range(56)]

class AtomEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.atom_embedding_list = torch.nn.ModuleList()
        for dim in full_atom_feature_dims:
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x):
        x_embedding = 0
        for i in range(x.shape[1]):
            x_embedding += self.atom_embedding_list[i](x[:, i])
        return x_embedding


class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout, return_embeds=False):
        super().__init__()
        self.convs = torch.nn.ModuleList(
            [GCNConv(in_channels=input_dim, out_channels=hidden_dim)]
            + [GCNConv(in_channels=hidden_dim, out_channels=hidden_dim) for _ in range(num_layers - 2)]
            + [GCNConv(in_channels=hidden_dim, out_channels=output_dim)]
        )
        self.bns = torch.nn.ModuleList(
            [torch.nn.BatchNorm1d(num_features=hidden_dim) for _ in range(num_layers - 1)]
        )
        self.dropout = dropout
        self.return_embeds = return_embeds
        self.softmax = torch.nn.LogSoftmax(dim=-1)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index):
        for conv, bn in zip(self.convs[:-1], self.bns):
            x = bn(conv(x, edge_index))
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        if self.return_embeds:
            return x
        else:
            return self.softmax(x)


class GCN_Graph(torch.nn.Module):
    """
    A graph-level GCN that pools node embeddings into a single graph embedding,
    then classifies whether a coordinate patch is likely to bind a metal ion.
    """
    def __init__(self, hidden_dim, output_dim, num_layers, dropout):
        super().__init__()
        self.node_encoder = AtomEncoder(hidden_dim)
        self.gnn_node = GCN(hidden_dim, hidden_dim, hidden_dim, num_layers, dropout, return_embeds=True)
        self.pool = global_mean_pool  # Summarize node embeddings
        self.linear = torch.nn.Linear(hidden_dim, output_dim)

    def reset_parameters(self):
        self.gnn_node.reset_parameters()
        self.linear.reset_parameters()

    def forward(self, batched_data):
        x, edge_index, batch = batched_data.x, batched_data.edge_index, batched_data.batch
        x = self.node_encoder(x)
        x = self.gnn_node(x, edge_index)
        x = self.pool(x, batch)
        return self.linear(x)

##################################################
#           Utility / Helper Functions
##################################################
def eval_with_outputs(model, device, loader):
    """
    Returns (y_true, y_pred, coords) after running inference.

    - y_true: ground-truth labels (0 or 1)
    - y_pred: model predictions (float scores)
    - coords: coordinate strings for each example
    """
    model.eval()
    y_true_all = []
    y_pred_all = []
    coords_all = []

    for step, batch in enumerate(loader):
        batch = batch.to(device)
        # Skip trivial single-node batch
        if batch.x.shape[0] == 1:
            continue

        with torch.no_grad():
            logits = model(batch)  # shape [batch_size, 1]

        y_true_all.append(batch.y.view(logits.shape).cpu())
        y_pred_all.append(logits.cpu())
        coords_all.extend(batch.coords)

    y_true_all = torch.cat(y_true_all, dim=0).numpy().flatten()
    y_pred_all = torch.cat(y_pred_all, dim=0).numpy().flatten()
    coords_all = np.array(coords_all)

    return y_true_all, y_pred_all, coords_all

##################################################
#                 Main App
##################################################
def main():
    st.set_page_config(page_title="RNA-Metal Ion Binding Predictor", layout="wide")

    # Display the overall description at the top
    st.markdown(description_text, unsafe_allow_html=True)

    # Sidebar
    st.sidebar.title("Configuration")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    st.sidebar.write(f"**Compute device**: `{device}`")

    # Hyperparameters (for reference)
    args = {
        'hidden_dim': 256,
        'num_layers': 5,
        'dropout': 0.5,
        'lr': 0.001,
        'epochs': 30,
    }

    # --- MODEL LOADING ---
    with st.spinner("Loading GNN model checkpoint..."):
        best_model = GCN_Graph(
            hidden_dim=args['hidden_dim'],
            output_dim=1,
            num_layers=args['num_layers'],
            dropout=args['dropout']
        ).to(device)

        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            best_model.load_state_dict(checkpoint)
            st.success(f"Model loaded successfully from `{checkpoint_path}`")
        except Exception as e:
            st.error(f"Error loading checkpoint: {e}")
            st.stop()

    # --- LOAD 1FUF DATA ---
    with st.expander("Load & Inspect Graph Data (1FUF)"):
        st.write(
            "We'll now load pre-processed graph data for the 1FUF RNA structure. "
            "Each candidate coordinate is represented as a sub-graph with node features "
            "and adjacency information."
        )
        try:
            with open(data_file, 'rb') as f:
                loaded_dict = pickle.load(f)

            # Convert dictionary to a list of PyG data objects
            pyg_list = []
            for coord_str, entry in loaded_dict.items():
                A = np.array(entry['A1'])  # adjacency
                G_nx = nx.from_numpy_array(A)

                # Add node features
                for idx in G_nx.nodes:
                    G_nx.nodes[idx]['x'] = torch.tensor(entry['H'][idx], dtype=torch.int32)

                # Convert networkx -> PyG
                pyg_graph = from_networkx(G_nx)
                pyg_graph.y = torch.tensor(entry['C'], dtype=torch.float).unsqueeze(-1)  # shape [1,1]
                pyg_graph.coords = coord_str
                pyg_graph.pdb = '1FUF'
                pyg_list.append(pyg_graph)

            # DataLoader
            pdb_loader = DataLoader(pyg_list, batch_size=32, shuffle=False, num_workers=0)
            st.success(f"Loaded graph data with {len(pyg_list)} items.")
        except Exception as e:
            st.error(f"Error loading data file: {e}")
            st.stop()

    # --- INFERENCE ---
    st.subheader("Run Inference on 1FUF Graph Data")
    st.write(
        "We pass each graph through the GNN to obtain a prediction of whether it represents a true binding site."
    )

    evaluator = Evaluator(name='ogbg-molhiv')
    y_true, y_pred, coords = eval_with_outputs(best_model, device, pdb_loader)

    # Calculate threshold via Youden's J statistic
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    youden_j = tpr - fpr
    best_threshold = thresholds[np.argmax(youden_j)]
    st.write(f"**Best threshold from Youden's J**: `{best_threshold:.3f}`")

    # Binarize predictions at this threshold
    predicted_positive = (y_pred >= best_threshold)
    positive_coords = coords[predicted_positive]
    st.write(f"Number of predicted positive coordinates: `{len(positive_coords)}`")

    # --- MERGE COORDINATES ---
    with st.spinner("Merging predicted coordinates into a new PDB..."):
        def parse_coordinate(cstr):
            # cstr like '[12 24 30]'
            arr = cstr.strip('[]')
            return np.fromstring(arr, sep=' ')

        if len(positive_coords) > 0:
            coords_np = np.array([parse_coordinate(c) for c in positive_coords])
        else:
            coords_np = np.array([])

        if coords_np.shape[0] > 0:
            try:
                write_merged_pdb_with_hetatms(
                    pdb_clean_path=pdb_clean_file,
                    output_pdb_path=output_merged_file,
                    fixed_coords=coords_np
                )
                st.success("Successfully wrote merged PDB with predicted ions.")
            except Exception as e:
                st.error(f"Failed to write merged PDB: {e}")
        else:
            st.info("No positive coordinates to merge.")

    # --- VISUALIZATION ---
    st.subheader("3D Visualization with Mol* Viewer")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Original 1FUF (with actual ions)**")
        try:
            st_molstar(pdb_file_path, key='original_pdb')
        except Exception as e:
            st.warning("Could not display the original PDB.")
            st.info(str(e))
    with col2:
        st.write("**Predicted Ion-Binding Sites**")
        try:
            st_molstar(output_merged_file, key='merged_pdb')
        except Exception as e:
            st.warning("Could not display the merged PDB.")
            st.info(str(e))

    st.markdown("---")
    st.success("Inference complete. Compare the original and predicted ions above! We have predicted the ions(small dots) in the RNA structure. The left image shows the original RNA structure with ions, and the right image shows the predicted ions in the RNA structure.")

if __name__ == "__main__":
    main()
