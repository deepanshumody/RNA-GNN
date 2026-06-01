"""Streamlit demo: RNA–Metal Ion Binding Predictor.

Loads the trained GCN checkpoint, runs inference on the bundled 1FUF RNA
structure, merges the predicted Mg²⁺ positions back into a PDB, and renders the
experimental vs. predicted ions side by side in an interactive Mol* viewer.

The model / inference logic is unchanged from the training pipeline; this file
adds caching (so reruns are instant) and a custom design layer on top.
"""
import streamlit as st
import pickle
import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils.convert import from_networkx
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import GCNConv
from sklearn.metrics import roc_curve, roc_auc_score

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

GITHUB_URL = "https://github.com/deepanshumody/RNA-GNN"
PAPER_URL = "https://pubs.acs.org/doi/10.1021/acs.jcim.9b00387"

##################################################
#                   Design layer
##################################################
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600&family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {
  --bg: #0a0e15;
  --surface: rgba(255,255,255,0.025);
  --surface-2: rgba(255,255,255,0.04);
  --line: rgba(255,255,255,0.09);
  --text: #e6edf3;
  --muted: #8893a5;
  --accent: #34d399;     /* magnesium green (predicted ions) */
  --accent-2: #5eead4;   /* teal */
  --warm: #fbbf24;       /* amber (experimental ions) */
  --font-display: 'Fraunces', Georgia, serif;
  --font-sans: 'IBM Plex Sans', system-ui, sans-serif;
  --font-mono: 'IBM Plex Mono', ui-monospace, monospace;
}

/* ---- canvas ---- */
.stApp {
  background:
    radial-gradient(1100px 620px at 82% -12%, rgba(52,211,153,0.12), transparent 60%),
    radial-gradient(820px 520px at -5% 2%, rgba(94,234,212,0.07), transparent 55%),
    var(--bg);
  color: var(--text);
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"], footer, #MainMenu { visibility: hidden; }
.block-container { max-width: 1180px; padding-top: 2.2rem; padding-bottom: 4rem; }

html, body, .stApp, .stMarkdown, p, li, span, div, label { font-family: var(--font-sans); }
.stApp h1, .stApp h2, .stApp h3 { font-family: var(--font-display); font-weight: 600; letter-spacing: -0.015em; }

/* ---- hero ---- */
.hero { padding: 0.4rem 0 1.6rem; border-bottom: 1px solid var(--line); margin-bottom: 2rem; }
.hero .kicker {
  font-family: var(--font-mono); text-transform: uppercase; letter-spacing: 0.26em;
  font-size: 0.7rem; color: var(--accent); margin-bottom: 1.1rem;
}
.hero h1 {
  font-family: var(--font-display); font-weight: 600;
  font-size: clamp(2.5rem, 6vw, 4.3rem); line-height: 1.0; letter-spacing: -0.03em; margin: 0;
}
.hero h1 .grad {
  background: linear-gradient(100deg, var(--accent), var(--accent-2));
  -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
}
.hero h1 .dot { color: var(--muted); font-weight: 400; }
.hero .sub { color: var(--muted); max-width: 64ch; font-size: 1.06rem; line-height: 1.65; margin: 1.2rem 0 0; }
.badges { display: flex; flex-wrap: wrap; gap: 0.55rem; margin-top: 1.5rem; }
.badge {
  font-family: var(--font-mono); font-size: 0.74rem; color: var(--muted);
  border: 1px solid var(--line); border-radius: 999px; padding: 0.32rem 0.8rem; background: var(--surface);
}
.badge.live { color: var(--accent); border-color: rgba(52,211,153,0.35); }
.badge.live .pulse {
  display: inline-block; width: 7px; height: 7px; border-radius: 50%; background: var(--accent);
  margin-right: 0.45rem; box-shadow: 0 0 0 0 rgba(52,211,153,0.6); animation: pulse 2s infinite;
}
@keyframes pulse {
  0% { box-shadow: 0 0 0 0 rgba(52,211,153,0.55); }
  70% { box-shadow: 0 0 0 7px rgba(52,211,153,0); }
  100% { box-shadow: 0 0 0 0 rgba(52,211,153,0); }
}

/* ---- section headers ---- */
.sec { margin: 2.6rem 0 1.2rem; }
.sec-label {
  font-family: var(--font-mono); text-transform: uppercase; letter-spacing: 0.2em;
  font-size: 0.7rem; color: var(--accent); margin-bottom: 0.5rem;
}
.sec-title { font-family: var(--font-display); font-size: 1.7rem; margin: 0; }
.sec-sub { color: var(--muted); margin: 0.5rem 0 0; max-width: 70ch; line-height: 1.6; }

/* ---- metric "instrument readout" cards ---- */
[data-testid="stMetric"] {
  background: var(--surface); border: 1px solid var(--line); border-radius: 16px;
  padding: 1.05rem 1.2rem; position: relative; overflow: hidden;
}
[data-testid="stMetric"]::before {
  content: ""; position: absolute; left: 0; top: 0; height: 100%; width: 3px;
  background: linear-gradient(180deg, var(--accent), var(--accent-2));
}
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p {
  font-family: var(--font-mono) !important; text-transform: uppercase;
  letter-spacing: 0.12em; font-size: 0.68rem !important; color: var(--muted) !important;
}
[data-testid="stMetricValue"] { font-family: var(--font-mono); font-weight: 600; font-size: 1.85rem; }

/* ---- legend + viewer captions ---- */
.legend { display: flex; flex-wrap: wrap; gap: 1.6rem; margin: 0.4rem 0 1.4rem; }
.legend .item { display: flex; align-items: center; gap: 0.55rem; font-size: 0.9rem; color: var(--muted); }
.dot-c { width: 11px; height: 11px; border-radius: 50%; display: inline-block; }
.dot-c.pred { background: var(--accent); box-shadow: 0 0 11px rgba(52,211,153,0.8); }
.dot-c.real { background: var(--warm); box-shadow: 0 0 11px rgba(251,191,36,0.6); }
.viewer-cap {
  font-family: var(--font-mono); font-size: 0.78rem; letter-spacing: 0.04em; color: var(--text);
  display: flex; align-items: center; gap: 0.55rem; margin-bottom: 0.6rem;
}

/* ---- native widgets ---- */
[data-testid="stSidebar"] { background: #0c111b; border-right: 1px solid var(--line); }
[data-testid="stSidebar"] .sb-title {
  font-family: var(--font-mono); text-transform: uppercase; letter-spacing: 0.18em;
  font-size: 0.72rem; color: var(--accent); margin-bottom: 0.4rem;
}
[data-testid="stExpander"] details {
  background: var(--surface); border: 1px solid var(--line) !important; border-radius: 14px !important;
}
[data-testid="stAlert"] { border-radius: 12px; border: 1px solid var(--line); }

/* ---- prose blocks ---- */
.prose { color: var(--muted); line-height: 1.7; }
.prose strong { color: var(--text); font-weight: 600; }
.prose code, code { font-family: var(--font-mono); background: var(--surface-2); padding: 0.1rem 0.4rem; border-radius: 6px; font-size: 0.85em; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 1rem; margin-top: 0.4rem; }
.card { background: var(--surface); border: 1px solid var(--line); border-radius: 16px; padding: 1.3rem 1.4rem; }
.card .n { font-family: var(--font-mono); color: var(--accent); font-size: 0.8rem; }
.card h4 { font-family: var(--font-display); margin: 0.5rem 0 0.4rem; font-size: 1.1rem; color: var(--text); }
.card p { color: var(--muted); font-size: 0.9rem; line-height: 1.6; margin: 0; }

/* ---- footer ---- */
.footer {
  margin-top: 3.2rem; padding-top: 1.6rem; border-top: 1px solid var(--line);
  display: flex; flex-wrap: wrap; justify-content: space-between; gap: 0.8rem;
  font-size: 0.86rem; color: var(--muted);
}
.footer a { color: var(--accent); text-decoration: none; }
.footer a:hover { text-decoration: underline; }
"""


def inject_css():
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


def render_hero():
    st.markdown(
        """
        <div class="hero">
          <div class="kicker">Graph Neural Networks · Structural Biology</div>
          <h1>RNA <span class="dot">·</span> Metal-Ion <span class="grad">Binding</span></h1>
          <p class="sub">A graph neural network that predicts where <strong>Mg²⁺</strong> ions
          bind in RNA 3D structures. Every candidate position is turned into a local atomic graph,
          the GNN scores it, and the predicted ions are rendered straight back into the structure below.</p>
          <div class="badges">
            <span class="badge live"><span class="pulse"></span>Live inference</span>
            <span class="badge">GCN</span>
            <span class="badge">1FUF ribozyme</span>
            <span class="badge">ROC&nbsp;AUC&nbsp;≈&nbsp;0.95</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section(label, title, subtitle=None):
    sub = f'<p class="sec-sub">{subtitle}</p>' if subtitle else ""
    st.markdown(
        f"""
        <div class="sec">
          <div class="sec-label">{label}</div>
          <h2 class="sec-title">{title}</h2>
          {sub}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer():
    st.markdown(
        f"""
        <div class="footer">
          <div>RNA · Metal-Ion GNN — predicting Mg²⁺ binding sites with graph neural networks.</div>
          <div><a href="{GITHUB_URL}">GitHub</a> &nbsp;·&nbsp;
               <a href="{PAPER_URL}">GNN-DTI (Lim et al., 2019)</a> &nbsp;·&nbsp;
               Built by Deepanshu Mody</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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
@st.cache_resource(show_spinner=False)
def load_model(_checkpoint_path, device, hidden_dim, num_layers, dropout):
    """Load the trained GCN checkpoint once and keep it in memory across reruns."""
    model = GCN_Graph(hidden_dim=hidden_dim, output_dim=1, num_layers=num_layers, dropout=dropout).to(device)
    checkpoint = torch.load(_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    return model


@st.cache_resource(show_spinner=False)
def load_graphs(_data_file):
    """Load and convert the 1FUF candidate-site graphs once (cached across reruns)."""
    with open(_data_file, 'rb') as f:
        loaded_dict = pickle.load(f)

    pyg_list = []
    for coord_str, entry in loaded_dict.items():
        A = np.array(entry['A1'])  # adjacency
        G_nx = nx.from_numpy_array(A)
        for idx in G_nx.nodes:
            G_nx.nodes[idx]['x'] = torch.tensor(entry['H'][idx], dtype=torch.int32)
        pyg_graph = from_networkx(G_nx)
        pyg_graph.y = torch.tensor(entry['C'], dtype=torch.float).unsqueeze(-1)  # shape [1,1]
        pyg_graph.coords = coord_str
        pyg_graph.pdb = '1FUF'
        pyg_list.append(pyg_graph)
    return pyg_list


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
    st.set_page_config(
        page_title="RNA · Metal-Ion Binding Predictor",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()
    render_hero()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Hyperparameters (must match the trained checkpoint)
    args = {
        'hidden_dim': 256,
        'num_layers': 5,
        'dropout': 0.5,
        'lr': 0.001,
        'epochs': 30,
    }

    # ---- Sidebar: project context ----
    with st.sidebar:
        st.markdown('<div class="sb-title">About</div>', unsafe_allow_html=True)
        st.markdown(
            '<p class="prose">Graph neural network for predicting <strong>Mg²⁺</strong> '
            'binding sites in RNA. This demo runs the trained <strong>GCN</strong> on the '
            '1FUF ribozyme and visualizes predicted vs. experimental ions.</p>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="sb-title">Model</div>', unsafe_allow_html=True)
        st.markdown(
            f'<p class="prose">Architecture: <code>GCN</code><br>'
            f'Hidden dim: <code>{args["hidden_dim"]}</code> · Layers: <code>{args["num_layers"]}</code><br>'
            f'Pooling: <code>global mean</code><br>'
            f'Compute: <code>{device}</code></p>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="sb-title">Links</div>', unsafe_allow_html=True)
        st.markdown(
            f'<p class="prose"><a href="{GITHUB_URL}" target="_blank">GitHub repository</a><br>'
            f'<a href="{PAPER_URL}" target="_blank">GNN-DTI paper</a></p>',
            unsafe_allow_html=True,
        )

    # ---- Load model + data (cached: instant on rerun) ----
    try:
        with st.spinner("Loading model & 1FUF graph data…"):
            best_model = load_model(checkpoint_path, device, args['hidden_dim'], args['num_layers'], args['dropout'])
            pyg_list = load_graphs(data_file)
            pdb_loader = DataLoader(pyg_list, batch_size=32, shuffle=False, num_workers=0)
    except Exception as e:
        st.error(f"Failed to load model or data: {e}")
        st.stop()

    # ---- Inference ----
    y_true, y_pred, coords = eval_with_outputs(best_model, device, pdb_loader)

    # Threshold via Youden's J statistic (favors recall under heavy imbalance)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    youden_j = tpr - fpr
    best_threshold = thresholds[np.argmax(youden_j)]
    predicted_positive = (y_pred >= best_threshold)
    positive_coords = coords[predicted_positive]

    try:
        auc = roc_auc_score(y_true, y_pred)
        auc_str = f"{auc:.3f}"
    except Exception:
        auc_str = "—"

    section(
        "Inference · 1FUF ribozyme",
        "Scoring every candidate site",
        "Each of the candidate grid positions around the RNA is scored by the GCN. "
        "Predictions are binarized at the threshold that maximizes Youden's J (TPR − FPR).",
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Candidate sites", f"{len(y_true):,}")
    m2.metric("Predicted ion sites", f"{len(positive_coords):,}")
    m3.metric("ROC AUC", auc_str)
    m4.metric("Youden's J threshold", f"{best_threshold:.2f}")

    with st.expander("Inspect the loaded graph data"):
        st.markdown(
            '<p class="prose">Each candidate coordinate is a sub-graph: <strong>nodes</strong> are atoms '
            '(with chemical features), <strong>edges</strong> encode covalent bonds and spatial proximity. '
            f'Loaded <code>{len(pyg_list)}</code> candidate graphs for 1FUF.</p>',
            unsafe_allow_html=True,
        )

    # ---- Merge predicted coordinates into a new PDB ----
    with st.spinner("Merging predicted coordinates into a PDB…"):
        def parse_coordinate(cstr):
            # cstr like '[12 24 30]'
            arr = cstr.strip('[]')
            return np.fromstring(arr, sep=' ')

        if len(positive_coords) > 0:
            coords_np = np.array([parse_coordinate(c) for c in positive_coords])
        else:
            coords_np = np.array([])

        merge_ok = False
        if coords_np.shape[0] > 0:
            try:
                write_merged_pdb_with_hetatms(
                    pdb_clean_path=pdb_clean_file,
                    output_pdb_path=output_merged_file,
                    fixed_coords=coords_np,
                )
                merge_ok = True
            except Exception as e:
                st.error(f"Failed to write merged PDB: {e}")
        else:
            st.info("No positive coordinates to merge.")

    # ---- 3D visualization ----
    section(
        "3D structure · Mol* viewer",
        "Predicted vs. experimental ions",
        "Left: the crystal structure with its experimentally observed ions. "
        "Right: the same RNA with the GNN's predicted Mg²⁺ sites merged in. Rotate and zoom either viewer.",
    )

    st.markdown(
        '<div class="legend">'
        '<div class="item"><span class="dot-c real"></span>Experimental ions (crystal structure)</div>'
        '<div class="item"><span class="dot-c pred"></span>GNN-predicted Mg²⁺ sites</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="viewer-cap"><span class="dot-c real"></span>Original 1FUF — experimental ions</div>', unsafe_allow_html=True)
        try:
            st_molstar(pdb_file_path, key='original_pdb')
        except Exception as e:
            st.warning("Could not display the original PDB.")
            st.info(str(e))
    with col2:
        st.markdown('<div class="viewer-cap"><span class="dot-c pred"></span>Predicted ion-binding sites</div>', unsafe_allow_html=True)
        try:
            st_molstar(output_merged_file, key='merged_pdb')
        except Exception as e:
            st.warning("Could not display the merged PDB.")
            st.info(str(e))

    # ---- How it works ----
    section("Method", "How it works")
    st.markdown(
        """
        <div class="cards">
          <div class="card">
            <div class="n">01</div>
            <h4>Graph construction</h4>
            <p>RNA structures are read with RDKit. A 3D grid of candidate ion positions is laid over the
            molecule, and each candidate becomes a local atomic graph — nodes are atoms with chemical
            features, edges encode bonds and spatial proximity.</p>
          </div>
          <div class="card">
            <div class="n">02</div>
            <h4>GNN scoring</h4>
            <p>A Graph Convolutional Network embeds each sub-graph, mean-pools to a single vector, and
            outputs a binding score. Known crystallographic ion sites are positives; everything else is
            negative. A GNN-DTI variant was also explored.</p>
          </div>
          <div class="card">
            <div class="n">03</div>
            <h4>Inference &amp; visualization</h4>
            <p>The model scores every candidate site in a new structure; high-scoring positions are merged
            back into the PDB as <code>HETATM</code> records so predicted and experimental ions can be
            compared directly in 3D.</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_footer()


if __name__ == "__main__":
    main()
