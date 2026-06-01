"""GNN-DTI model definitions.

Provides a binary FocalLoss and the ``gnn`` drug-target interaction model,
which embeds molecular graphs through stacked GAT_gate layers and predicts an
interaction score via a fully connected head.
"""
import torch
import torch.nn.functional as F
import torch.nn as nn
from utils import *
import time
from multiprocessing import Pool
from layers import GAT_gate

N_atom_features = 28

class FocalLoss(nn.Module):
    """Binary focal loss that down-weights easy examples during training.

    The focal modulation ``(1 - p_t)**gamma`` must be applied *per example*
    before reducing, otherwise hard and easy samples are weighted identically
    and the loss collapses to a scaled BCE. The previous implementation reduced
    BCE to a scalar first and then applied a single modulation, which defeated
    the purpose; this computes BCE with ``reduction='none'`` and modulates each
    element, matching Lin et al. (2017).
    """
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.size_average = size_average

    def forward(self, inputs, targets, alpha=0.8, gamma=2, smooth=1):
        # Comment out the sigmoid if your model already outputs probabilities.
        inputs = torch.sigmoid(inputs)

        # Flatten label and prediction tensors.
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # Per-example BCE, then the per-example focal modulation.
        bce = F.binary_cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-bce)  # probability assigned to the true class
        focal_loss = alpha * (1 - p_t) ** gamma * bce

        return focal_loss.mean() if self.size_average else focal_loss.sum()

class gnn(torch.nn.Module):
    """Graph neural network for drug-target interaction prediction.

    Embeds an input molecular graph with stacked GAT_gate layers and maps the
    pooled graph representation to an interaction score through a fully
    connected head.
    """
    def __init__(self, args):
        super(gnn, self).__init__()
        n_graph_layer = args.n_graph_layer
        d_graph_layer = args.d_graph_layer
        n_FC_layer = args.n_FC_layer
        d_FC_layer = args.d_FC_layer
        self.dropout_rate = args.dropout_rate 


        self.layers1 = [d_graph_layer for i in range(n_graph_layer+1)]
        self.gconv1 = nn.ModuleList([GAT_gate(self.layers1[i], self.layers1[i+1]) for i in range(len(self.layers1)-1)]) 
        
        self.FC = nn.ModuleList([nn.Linear(self.layers1[-1], d_FC_layer) if i==0 else
                                 nn.Linear(d_FC_layer, 1) if i==n_FC_layer-1 else
                                 nn.Linear(d_FC_layer, d_FC_layer) for i in range(n_FC_layer)])
        
        self.mu = nn.Parameter(torch.Tensor([args.initial_mu]).float())
        self.dev = nn.Parameter(torch.Tensor([args.initial_dev]).float())
        # Feature width is a property of the dataset: receptor + ligand one-hot
        # blocks, each ``n_atom_features`` wide. Read it from args so an extended
        # atom vocabulary (e.g. adding 'Mg') trains without code changes; falls
        # back to the released width when not provided.
        n_atom_features = getattr(args, "n_atom_features", N_atom_features)
        self.embede = nn.Linear(2 * n_atom_features, d_graph_layer, bias=False)
        

    def embede_graph(self, data):
        c_hs, c_adjs1, c_adjs2, c_valid = data
        c_hs = self.embede(c_hs)
        hs_size = c_hs.size()
        c_adjs2 = torch.exp(-torch.pow(c_adjs2-self.mu.expand_as(c_adjs2), 2)/self.dev) + c_adjs1
        regularization = torch.empty(len(self.gconv1), device=c_hs.device)

        for k in range(len(self.gconv1)):
            c_hs1 = self.gconv1[k](c_hs, c_adjs1)
            c_hs2 = self.gconv1[k](c_hs, c_adjs2)
            c_hs = c_hs2-c_hs1
            c_hs = F.dropout(c_hs, p=self.dropout_rate, training=self.training)
        c_hs = c_hs*c_valid.unsqueeze(-1).repeat(1, 1, c_hs.size(-1))
        c_hs = c_hs.sum(1)
        return c_hs

    def fully_connected(self, c_hs):
        regularization = torch.empty(len(self.FC)*1-1, device=c_hs.device)

        for k in range(len(self.FC)):
            if k<len(self.FC)-1:
                c_hs = self.FC[k](c_hs)
                c_hs = F.dropout(c_hs, p=self.dropout_rate, training=self.training)
                c_hs = F.relu(c_hs)
            else:
                c_hs = self.FC[k](c_hs)

        c_hs = torch.sigmoid(c_hs)

        return c_hs

    def train_model(self, data):
        #embede a graph to a vector
        c_hs = self.embede_graph(data)

        #fully connected NN
        c_hs = self.fully_connected(c_hs)
        c_hs = c_hs.view(-1) 

        #note that if you don't use concrete dropout, regularization 1-2 is zero
        return c_hs
    
    def test_model(self,data1 ):
        c_hs = self.embede_graph(data1)
        c_hs = self.fully_connected(c_hs)
        c_hs = c_hs.view(-1)
        return c_hs
