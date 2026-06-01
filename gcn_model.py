"""Shared GCN graph-classification model for the RNA ion-binding task.

Single source of truth for the GCN that was previously copy-pasted across the
training scripts, the inference script and the Streamlit demo. The architecture
is unchanged, so the released ``best_model.pth`` still loads as-is; the only
addition is a ``num_features`` argument so the model adapts to the width of
whatever feature set the data was built with (56 for the released data; wider
if the atom vocabulary is extended when regenerating the dataset).

Each input feature column is a 0/1 indicator, so every column is embedded with a
small ``Embedding(2, hidden)`` table and the per-column embeddings are summed
into a node embedding (this matches the released checkpoint's parameter layout).
"""
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

# Width of the released candidate-site feature vectors (receptor + ligand
# one-hot blocks). Override via ``num_features`` for regenerated datasets.
DEFAULT_NUM_FEATURES = 56


class AtomEncoder(torch.nn.Module):
    """Embed each binary atom-feature column and sum into a node embedding."""

    def __init__(self, emb_dim, num_features=DEFAULT_NUM_FEATURES):
        super().__init__()
        self.atom_embedding_list = torch.nn.ModuleList()
        for _ in range(num_features):
            emb = torch.nn.Embedding(2, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x):
        x_embedding = 0
        for i in range(x.shape[1]):
            x_embedding = x_embedding + self.atom_embedding_list[i](x[:, i])
        return x_embedding


class GCN(torch.nn.Module):
    """Stacked GCNConv layers with batch-norm, ReLU and dropout."""

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
        return self.softmax(x)


class GCN_Graph(torch.nn.Module):
    """Graph-level GCN: pool node embeddings, then score a candidate patch.

    Outputs a raw logit per graph (use with ``BCEWithLogitsLoss``).
    """

    def __init__(self, hidden_dim, output_dim, num_layers, dropout, num_features=DEFAULT_NUM_FEATURES):
        super().__init__()
        self.node_encoder = AtomEncoder(hidden_dim, num_features)
        self.gnn_node = GCN(hidden_dim, hidden_dim, hidden_dim, num_layers, dropout, return_embeds=True)
        self.pool = global_mean_pool
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
