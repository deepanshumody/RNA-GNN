"""Dataset utilities for the GNN-DTI pipeline.

Provides a weighted sampler (``DTISampler``) for class-balanced batching and a
``collate_fn`` that pads variable-size molecular graphs in a batch to a common
maximum atom count before stacking them into tensors.
"""
from torch.utils.data.sampler import Sampler
import numpy as np
import torch
import random
class DTISampler(Sampler):
    """Weighted random sampler that draws indices proportional to ``weights``."""

    def __init__(self, weights, num_samples, replacement=True):
        """Normalize ``weights`` to a probability distribution and store sampling settings."""
        weights = np.array(weights)/np.sum(weights)
        self.weights = weights
        self.num_samples = num_samples
        self.replacement = replacement

    def __iter__(self):
        """Yield ``num_samples`` indices drawn according to the weight distribution."""
        retval = np.random.choice(len(self.weights), self.num_samples, replace=self.replacement, p=self.weights)
        return iter(retval.tolist())

    def __len__(self):
        """Return the number of samples drawn per iteration."""
        return self.num_samples

def collate_fn(batch):
    """Pad each graph in ``batch`` to the max atom count and stack into batched tensors.

    Returns a tuple ``(H, A1, A2, C, V, keys)`` of padded float tensors plus the
    list of sample keys.
    """
    items = [item for item in batch if item is not None]
    max_natoms = max([len(item['H']) for item in items])
    # Infer the feature width from the data instead of hard-coding 56, so an
    # extended atom vocabulary (wider H) collates without code changes.
    feat_dim = int(np.asarray(items[0]['H']).shape[1])

    H = np.zeros((len(batch), max_natoms, feat_dim))
    A1 = np.zeros((len(batch), max_natoms, max_natoms))
    A2 = np.zeros((len(batch), max_natoms, max_natoms))
    C = np.zeros((len(batch),))
    V = np.zeros((len(batch), max_natoms))
    keys = []
    
    for i in range(len(batch)):
        natom = len(batch[i]['H'])
        
        H[i,:natom] = batch[i]['H']
        A1[i,:natom,:natom] = batch[i]['A1']
        A2[i,:natom,:natom] = batch[i]['A2']
        C[i] = batch[i]['C']
        V[i,:natom] = batch[i]['V']
        keys.append(batch[i]['key'])

    H = torch.from_numpy(H).float()
    A1 = torch.from_numpy(A1).float()
    A2 = torch.from_numpy(A2).float()
    C = torch.from_numpy(C).float()
    V = torch.from_numpy(V).float()
    
    return H, A1, A2, C, V, keys
