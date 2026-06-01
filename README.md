# RNA–Metal Ion GNN

**Predicting Mg²⁺ binding sites in RNA 3D structures with graph neural networks.**

[![CI](https://github.com/deepanshumody/RNA-GNN/actions/workflows/ci.yml/badge.svg)](https://github.com/deepanshumody/RNA-GNN/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyTorch_Geometric-3C2179)](https://pytorch-geometric.readthedocs.io/)
[![Streamlit Demo](https://img.shields.io/badge/Live_Demo-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://gnnrna.streamlit.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Metal ions such as Mg²⁺ are essential cofactors that stabilize RNA tertiary structure and enable catalysis, yet locating their binding sites from structure alone is hard. This project frames the problem as **graph classification**: tile the space around an RNA molecule with candidate ion positions, build a local atomic graph around each candidate, and train a GNN to score how likely that position is to bind a metal ion.

On the bundled 1FUF structure the Graph Convolutional Network achieves **ROC&nbsp;AUC ≈ 0.95**, ranking the single true Mg²⁺ site within the **top ~5%** of ~1,500 candidate positions — a useful candidate-site *filter* under extreme class imbalance. An [interactive demo](https://gnnrna.streamlit.app/) renders its predictions in 3D alongside the experimentally observed ions.

> **▶ Try it live:** **https://gnnrna.streamlit.app/**

<p align="center">
  <img src="assets/demo.png" alt="Mol* 3D viewer: experimentally observed ions (left) vs. GNN-predicted Mg²⁺ binding sites in green (right) on the 1FUF ribozyme" width="100%">
  <br>
  <em>The deployed app on the 1FUF ribozyme — experimentally observed ions (left) vs. the GNN's predicted Mg²⁺ binding sites (right, green dots).</em>
</p>

---

## Highlights

- **Two GNN architectures** for the same task — a graph-convolutional baseline (GCN) and a gated graph-attention model adapted from the GNN-DTI drug–target framework ([Lim et al., 2019](https://pubs.acs.org/doi/10.1021/acs.jcim.9b00387)).
- **End-to-end pipeline** from raw PDB structures → candidate-site graphs → trained model → predicted ions merged back into a `.pdb` for visualization.
- **Imbalance-aware evaluation:** with positives at ~0.07% of candidate sites, results are framed as a ranking/filtering task — ROC AUC ≈ 0.95 reported alongside PR-AUC and precision/recall, not a single cherry-picked number.
- **Deployed demo** — a Streamlit + Mol\* app that runs inference on the 1FUF ribozyme and shows predicted vs. real ions side by side.
- **Reproducible dataset construction**, including the scripts used to derive the non-redundant set of RNA structures.

---

## Results

Locating ion sites is an **extreme class-imbalance** problem: on the bundled 1FUF structure only **1 of 1,484** candidate positions is a true Mg²⁺ site (≈ 0.07% positive). The GCN is therefore best read as a **ranker / filter** rather than a hard classifier, and is evaluated with imbalance-aware metrics rather than ROC AUC alone.

<p align="center">
  <img src="assets/roc_curve.png" alt="ROC curve for the GCN ion-binding classifier on 1FUF (AUC ≈ 0.95)" width="460">
</p>

| Metric — GCN on the 1FUF demo structure | Value | How to read it |
| --- | --- | --- |
| ROC AUC | **0.95** | ranks the true site near the top of all candidates |
| PR-AUC (average precision) | 0.01 | low by construction when positives are ~0.07% |
| Recall @ Youden's-J threshold | 100% (1 / 1) | the true site is recovered |
| Precision @ same threshold | 1.4% (1 / 73) | catching it costs ~72 false positives |
| Enrichment | true site in **top ~5%** | narrows ~1,500 candidates → ~70 for follow-up |

**Why report both?** ROC AUC is optimistic under heavy imbalance, so it's paired with PR-AUC and precision/recall for an honest picture. The practical value is as a first-pass filter that shrinks the candidate search space for downstream analysis. Predictions are binarized at the threshold that maximizes Youden's J (TPR − FPR), which favors recall.

> These figures are for the single bundled 1FUF structure (the demo). The GNN-DTI variant logs ROC-AUC / PR per epoch during training and is included as an exploratory alternative.

---

## How it works

```mermaid
flowchart LR
    A["RNA PDB<br/>structure"] --> B["Strip HETATM<br/>(clean RNA only)"]
    B --> C["Tile 3D grid of<br/>candidate ion sites"]
    C --> D["Per candidate:<br/>build local atom graph<br/>(nodes = atoms, edges =<br/>bonds + spatial proximity)"]
    D --> E{"GNN<br/>classifier"}
    E -->|GCN| F["binding score"]
    E -->|GNN-DTI| F
    F --> G["Threshold<br/>(Youden's J)"]
    G --> H["Merge predicted ions<br/>as HETATM → PDB"]
    H --> I["3D visualization<br/>(Mol* demo)"]
```

1. **Graph construction.** Each RNA structure is read with RDKit. A 3D grid of candidate ion positions is laid over the molecule, and for every candidate a subgraph is extracted from the atoms within range — nodes carry chemical atom features, edges encode covalent bonds and spatial proximity. The grid point nearest a real crystallographic ion is labeled positive.
2. **Models.**
   - **GCN** — stacked `GCNConv` layers with batch norm and dropout, mean-pooled to a graph embedding and passed through a linear head (binary output). Implemented in `GNN_models/train_gnn.py`.
   - **GNN-DTI** — a gated graph-attention network with a learned distance-based adjacency (Gaussian over interatomic distances), adapted from the drug–target interaction model of Lim et al. (2019). Implemented in `GNN_models/GNN-DTI/`.
3. **Inference & visualization.** The trained model scores every candidate site in a new structure; high-scoring positions are merged back into the cleaned PDB as new `HETATM` records so predicted and experimental ions can be compared directly in 3D.

---

## Repository structure

```
RNA-GNN/
├── moleculestreamlit.py          # Streamlit + Mol* demo app (inference + 3D viewer)
├── createpredictedionpdb.py      # Merge predicted coordinates into a PDB as HETATM records
├── best_model.pth                # Trained GCN checkpoint (used by the demo)
│
├── GNN_models/
│   ├── train_gnn.py              # GCN training — builds graphs from candidate-site pickles
│   ├── train_gnn1.py             #   variant: loads precomputed graph tensors
│   ├── train_gnn2.py             #   variant: precomputed tensors + negative subsampling
│   ├── predfrommodel.py          # GCN inference on a single RNA structure
│   └── GNN-DTI/                  # Gated graph-attention model (Lim et al. 2019)
│       ├── gnn.py                #   model + FocalLoss
│       ├── layers.py             #   GAT_gate graph-attention layer
│       ├── dataset.py            #   collate_fn + weighted sampler
│       ├── train.py / test.py    #   training / evaluation
│       └── utils.py
│
├── dataset_creation/             # PDB → candidate-site graph pickles (4 labeling strategies)
│   ├── gnn_rna.py                #   3Å grid, nearest point to ion = positive
│   ├── gnn_rna_0A.py             #   grid stops at the molecular surface
│   ├── gnn_rna_autodock.py       #   AutoDock Vina–placed candidate points
│   └── gnn_rna_morepos.py        #   8 grid points around each ion = positive
│
├── preprocessing/                # Build the non-redundant RNA list (exploratory/reference)
│   ├── clustering1.py … clustering3.py
│   └── bestresolutionfromcluster.py
│
├── data/                         # Sample data for the demo (1FUF) + curated lists
│   ├── RNA-only-PDB/ , RNA-only-PDB-clean/ , RNA-graph-pickles/
│   ├── Mg_ions.sdf , nonredundantRNA.txt , OnlyRNAlist.txt
│   └── preds_RNA1FUFf.csv
│
└── assets/roc_curve.png          # ROC curve on the 1FUF demo (AUC ≈ 0.95)
```

---

## Installation

```bash
git clone https://github.com/deepanshumody/RNA-GNN.git
cd RNA-GNN

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

> **Note:** install [PyTorch](https://pytorch.org/) and [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) with the build that matches your CUDA / CPU setup before (or alongside) the other requirements.
>
> For a known-good, fully pinned set of versions (Python 3.12), use [`requirements-lock.txt`](requirements-lock.txt).

---

## Quickstart — run the demo locally

The demo loads the released checkpoint (`best_model.pth`), runs inference on the bundled 1FUF structure, and shows predicted vs. real ions in an interactive 3D viewer:

```bash
streamlit run moleculestreamlit.py
```

(Or skip setup entirely and open the hosted version: **https://gnnrna.streamlit.app/**.)

---

## Reproducing the pipeline

### 1. Create the dataset

Place RNA-only `.pdb` files in `RNA-only-PDB/`, then run **one** of the dataset-creation scripts depending on the labeling strategy you want:

```bash
python dataset_creation/gnn_rna.py            # 3Å grid (default)
# or gnn_rna_0A.py / gnn_rna_autodock.py / gnn_rna_morepos.py
```

Each writes per-structure graph pickles (`<pdb>_pos.pkl`, `<pdb>_neg.pkl`). Dataset variants:

| Variant            | Candidate placement                          | ~Positives | ~Negatives |
| ------------------ | -------------------------------------------- | ---------- | ---------- |
| `gnn_rna`          | 3Å grid; nearest point to ion = positive     | ~3,000     | ~1,000,000 |
| `gnn_rna_0A`       | grid stops at the molecular surface          | ~2,500     | ~500,000   |
| `gnn_rna_autodock` | AutoDock Vina–placed candidate points        | —          | —          |
| `gnn_rna_morepos`  | 8 grid points around each ion = positive     | ~12,000    | ~1,000,000 |

> A Mg²⁺ ideal-geometry SDF (e.g. `data/Mg_ions.sdf`) is used as the ligand template when building graphs.

### 2. Train

```bash
# GCN
python GNN_models/train_gnn.py

# GNN-DTI (set the GPU index)
CUDA_VISIBLE_DEVICES=0 python GNN_models/GNN-DTI/train.py
```

The best GCN checkpoint is saved to `best_model.pth`. Update the data-directory constants near the top of each script to point at your generated pickles.

### 3. Predict / evaluate

```bash
# GCN — single structure
python GNN_models/predfrommodel.py

# GNN-DTI
CUDA_VISIBLE_DEVICES=0 python GNN_models/GNN-DTI/test.py
```

### (Optional) Rebuild the non-redundant RNA list

The `preprocessing/` scripts (`clustering1.py` → `clustering3.py` → `bestresolutionfromcluster.py`) reproduce `nonredundantRNA.txt` — RNAs deduplicated by sequence similarity, keeping the best resolution (< 6 Å) per cluster — starting from `data/OnlyRNAlist.txt`. These are exploratory/reference scripts and may need light adaptation to your environment; the resulting list is already provided.

---

## Notes & limitations

- Reported metrics are for the single bundled **1FUF** structure; aggregate metrics across the full training corpus are not included in this repository. Given the extreme class imbalance (positives ≈ 0.07% of candidate sites), the model is intended as a candidate-site **ranker / filter**, not a precise point locator.
- Results are reported for the GCN; the GNN-DTI variant is an exploratory alternative and its metrics are logged per epoch during training rather than as a single headline number.
- The demo ships with one structure (1FUF) for a fast, self-contained showcase; the full training corpus is built from the non-redundant RNA list.
- Several `dataset_creation/` and `preprocessing/` scripts began life as research notebooks; they are included for transparency and reproducibility of the data pipeline.

---

## References

- Lim, J. et al. *Predicting Drug–Target Interaction Using a Novel Graph Neural Network with 3D Structure-Embedded Graph Representation.* **J. Chem. Inf. Model.** 2019. [DOI: 10.1021/acs.jcim.9b00387](https://pubs.acs.org/doi/10.1021/acs.jcim.9b00387) — basis for the GNN-DTI architecture.

### Slides

- [Slide Deck 1](https://purdue0-my.sharepoint.com/:p:/g/personal/modyd_purdue_edu/EXJN6pxMfNZBnivvTjUdbCABFum4tNid0VJ6X5CW7WLyXA?e=6eIJPs)
- [Slide Deck 2](https://purdue0-my.sharepoint.com/:p:/g/personal/modyd_purdue_edu/EdZh7vnDzwZClZ6i372E_DUB3SrtWZm17wQpZd03VlAa8w?e=kDzZlA)

---

## Development

```bash
pip install -r requirements-dev.txt
ruff check .      # lint (critical-error rules)
pytest -q         # tests
```

[GitHub Actions CI](.github/workflows/ci.yml) runs linting, byte-compilation, and the test suite on every push and pull request. Tests cover the PDB-merge formatting and a GCN forward-pass smoke test (the latter skips automatically where the deep-learning stack isn't installed).

---

## License

Released under the [MIT License](LICENSE).

## Contact

**Deepanshu Mody** · [GitHub](https://github.com/deepanshumody)
Questions and contributions welcome — open an issue or reach out directly.
