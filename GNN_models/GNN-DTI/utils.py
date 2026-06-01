"""Helper utilities for the GNN-DTI model.

This module provides small support functions used throughout the GNN-DTI
pipeline: GPU device selection, model parameter initialization, one-hot
feature encoders, and per-atom feature extraction.
"""

import numpy as np
import torch
from scipy import sparse
import os.path
import time
import torch.nn as nn

N_atom_features = 28


def set_cuda_visible_device(ngpus):
    """Return a comma-separated string of GPU ids that have no running processes.

    Scans up to 8 GPUs via nvidia-smi and selects the first ``ngpus`` idle ones.
    Exits the program if fewer idle GPUs are available than requested.
    """
    import subprocess
    import os
    empty = []
    for i in range(8):
        command = 'nvidia-smi -i '+str(i)+' | grep "No running" | wc -l'
        output = subprocess.check_output(command, shell=True).decode("utf-8")
        if int(output)==1:
            empty.append(i)
    if len(empty)<ngpus:
        print ('avaliable gpus are less than required')
        exit(-1)
    cmd = ''
    for i in range(ngpus):        
        cmd+=str(empty[i])+','
    return cmd

def initialize_model(model, device, load_save_file=False):
    """Initialize ``model`` weights and move it onto ``device``.

    When ``load_save_file`` is provided, the model's state dict is loaded from
    that path; otherwise multi-dimensional parameters are initialized with
    Xavier normal. Wraps the model in ``DataParallel`` when multiple GPUs exist.
    """
    if load_save_file:
        model.load_state_dict(torch.load(load_save_file))
    else:
        for param in model.parameters():
            if param.dim() == 1:
                continue
            else:
                nn.init.xavier_normal_(param)

    if torch.cuda.device_count() > 1:
      print("Let's use", torch.cuda.device_count(), "GPUs!")
      # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
      model = nn.DataParallel(model)
    model.to(device)
    return model

def one_of_k_encoding(x, allowable_set):
    """One-hot encode ``x`` against ``allowable_set``.

    Raises an exception if ``x`` is not a member of the allowable set.
    """
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def atom_feature(m, atom_i, i_donor, i_acceptor):
    """Build the 28-dim feature vector for atom ``atom_i`` of molecule ``m``.

    Concatenates one-hot encodings of the atom symbol, degree, total hydrogens,
    and implicit valence with an aromaticity flag.
    """
    atom = m.GetAtomWithIdx(atom_i)
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                      ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'B', 'H']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5]) +
                    [atom.GetIsAromatic()])    # (10, 6, 5, 6, 1) --> total 28
