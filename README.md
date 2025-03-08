# RNA-Metal-Ion-GNN

This repository contains two primary GNN-based architectures for predicting metal ion binding sites in RNA:

1. **GCN (Graph Convolutional Network)**
2. **GNN-DTI** (based on [Wang et al., 2019](https://pubs.acs.org/doi/10.1021/acs.jcim.9b00387))

The project expands on the approach of encoding RNA structures as graphs and applying neural networks for site prediction.
The [Demo](https://gnnrna.streamlit.app/) can be viewed at the link.

---

## Table of Contents
1. [Introduction](#introduction)
2. [Key Features](#key-features)
3. [Project Overview](#project-overview)
    - [GCN](#gcn)
    - [GNN-DTI](#gnn-dti)
4. [Installation](#installation)
5. [Generating Non-Redundant RNA List](#generating-non-redundant-rna-list)
6. [Creating the Dataset](#creating-the-dataset)
    - [Dataset Variants](#dataset-variants)
7. [Usage](#usage)
8. [Training](#training)
    - [3.a) Training GCN](#3a-training-gcn)
    - [3.b) Training GNN-DTI](#3b-training-gnn-dti)
9. [Testing](#testing)
    - [4.a) Testing GCN](#4a-testing-gcn)
    - [4.b) Testing GNN-DTI](#4b-testing-gnn-dti)
10. [Slides](#slides)
11. [License](#license)
12. [Contact](#contact)

---

## Introduction

Metal ions are crucial cofactors in many RNA-related biochemical processes. Accurately predicting the location of metal ion binding sites in RNA can aid in understanding RNA structure and function, accelerating both academic research and potential therapeutic applications.

This repository uses graph neural networks to model potential metal ion binding sites:
- **GCN** uses traditional graph convolutional layers.
- **GNN-DTI** follows drug-target interaction paradigms to capture more complex relationship patterns.

---

## Key Features

1. **Graph Construction**: Scripts to convert RNA structures into graph data (nodes represent nucleotides/atoms, edges represent connectivity or distance thresholds).
2. **Multiple Dataset Variants**: Provides different ways of placing metal ions and negative points, allowing for flexible experimentation.
3. **Scalable Training**: Scripts for training both GCN and GNN-DTI frameworks.
4. **Customizable**: Easy to tweak hyperparameters, adjacency strategies, or input data.
5. **Reproducibility**: Steps to generate the exact same list of PDB IDs (nonredundantRNA.txt) included.

---

## Project Overview

### GCN
The **GCN** model is implemented primarily in `train_gnn.py`. It treats potential metal ion binding sites as nodes on a 3D grid over the RNA structure. Each site is classified in a binary manner: binding vs. nonbinding.

### GNN-DTI
The **GNN-DTI** model is adapted from the drug-target interaction approach ([Wang et al., 2019](https://pubs.acs.org/doi/10.1021/acs.jcim.9b00387)). It is located in the `model_GNN_DTI` folder with a primary training script `train.py`. This architecture may capture more nuanced features and interactions.

---

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repo_url>
   cd RNA-GNN
   ```
2. **Set up a Python environment (recommended)**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # on Linux/Mac
   # or
   venv\Scripts\activate   # on Windows
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
---

## Generating Non-Redundant RNA List

To reproduce the exact list of PDBs used for training:
1. **Input**: `OnlyRNAlist.txt` containing comma-separated RNA-only PDB IDs from the RCSB PDB.
2. **Scripts**: `clustering1.py`, `clustering2.py`, `clustering3.py`, `bestresolutionfromcluster.py` (run in this order).
3. **Output**: `nonredundantRNA.txt` containing a list of non-redundant RNAs based on sequence with resolution better than 6Å.

This step is optional as the `nonredundantRNA.txt` file is already provided.

---

## Creating the Dataset

To create your dataset, you must:
1. Download the PDB files of your RNA structures and store them in `RNA-only-PDB` (within the working directory).
2. Run **one** of the following scripts to generate data pickles:
   - `gnn_rna.py`
   - `gnn_rna_0A.py`
   - `gnn_rna_autodock.py`
   - `gnn_rna_morepos.py`

Each script differs in how it places potential ion locations and labels positives. For instance, `gnn_rna_morepos.py` considers 8 cubic grid points as positive.

### Dataset Variants

1. **RNA-graph-pickles**:
   - Points placed on a 3Å grid over the entire molecule.
   - Nearest grid point to the actual ion location is considered positive.
   - ~3000 positive points, ~1,000,000 negative points.
2. **RNA-graph-pickles0A**:
   - Similar to above but stops at 0Å from the molecule's outer edge.
   - ~2500 positive points, ~500,000 negative points.
3. **RNA-graph-pickles-autodock**:
   - Uses Autodock Vina to place points over the molecule.
4. **RNA-graph-picklesmorepos**:
   - 8 grid points around the actual ion location are labeled positive.
   - ~12,000 positive points, ~1,000,000 negative points.

**MG_ideal.sdf** provides the structure of the binding ion for reference.

---

## Usage

1. **Set environment**:
   - Make sure your dataset pickles are created and stored in the correct folders.
2. **Adjust paths**:
   - Update input folder paths within the training and testing scripts (`train_gnn.py`, `train.py`, etc.) based on which dataset you want to use.
3. **Run training**:
   - GCN: `python train_gnn.py`
   - GNN-DTI: `CUDA_VISIBLE_DEVICES="x" python train.py` (inside `model_GNN_DTI` folder)
4. **Run testing**:
   - GCN: `python predfrommodel.py`
   - GNN-DTI: `CUDA_VISIBLE_DEVICES="x" python test.py`

---

## Training

### 3.a) Training GCN

- **Script**: `train_gnn.py`
- **Usage**:
  ```bash
  python train_gnn.py
  ```
- The best model is saved automatically. To evaluate on individual PDB files, use `predfrommodel.py`.

### 3.b) Training GNN-DTI

- **Folder**: `model_GNN_DTI`
- **Script**: `train.py`
- **Usage**:
  ```bash
  CUDA_VISIBLE_DEVICES="x" python train.py   # x is the GPU index
  ```
- All models are saved; use `test.py` for individual structure tests.

---

## Testing

### 4.a) Testing Individual Structures (GCN)

1. Update any path variables in `predfrommodel.py`.
2. Run:
   ```bash
   python predfrommodel.py
   ```

### 4.b) Testing Individual Structures (GNN-DTI)

1. Update the path to your dataset.
2. Run:
   ```bash
   CUDA_VISIBLE_DEVICES="x" python test.py
   ```

---

## Slides

Additional background and results can be found in the following slides:

- [Slide Deck 1](https://purdue0-my.sharepoint.com/:p:/g/personal/modyd_purdue_edu/EXJN6pxMfNZBnivvTjUdbCABFum4tNid0VJ6X5CW7WLyXA?e=6eIJPs)
- [Slide Deck 2](https://purdue0-my.sharepoint.com/:p:/g/personal/modyd_purdue_edu/EdZh7vnDzwZClZ6i372E_DUB3SrtWZm17wQpZd03VlAa8w?e=kDzZlA)

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Contact

- **Author**: [Deepanshu Mody](https://github.com/deepanshumody)
- For questions, feel free to open an issue or reach out directly.

---

