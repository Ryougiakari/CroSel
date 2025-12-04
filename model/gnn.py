import torch
from torch import nn
from torch import Tensor
from typing import Tuple


def normalize_adj(edge_index: Tensor, num_nodes: int) -> Tensor:
    row, col = edge_index
    deg = torch.bincount(row, minlength=num_nodes).float()
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    values = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    return torch.sparse_coo_tensor(edge_index, values, (num_nodes, num_nodes))


class GraphConv(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x: Tensor, edge_index: Tensor, norm_adj: Tensor) -> Tensor:
        x = self.linear(x)
        return torch.sparse.mm(norm_adj, x)


class SimpleGCN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, emb_dim: int, num_clusters: int):
        super().__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, emb_dim)
        self.activation = nn.PReLU()
        self.head = nn.Linear(emb_dim, num_clusters)

    def forward(self, x: Tensor, edge_index: Tensor, norm_adj: Tensor) -> Tuple[Tensor, Tensor]:
        h = self.conv1(x, edge_index, norm_adj)
        h = self.activation(h)
        z = self.conv2(h, edge_index, norm_adj)
        z = self.activation(z)
        logits = self.head(z)
        return z, logits
