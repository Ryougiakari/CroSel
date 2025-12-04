import torch
from torch import nn
from torch import Tensor
from typing import Tuple


def drop_edges(edge_index: Tensor, drop_prob: float) -> Tensor:
    if drop_prob <= 0 or edge_index.size(1) == 0:
        return edge_index
    device = edge_index.device
    mask = torch.rand(edge_index.size(1), device=device) > drop_prob
    return edge_index[:, mask]


def mask_features(x: Tensor, mask_prob: float) -> Tensor:
    if mask_prob <= 0:
        return x
    mask = torch.rand_like(x) > mask_prob
    return x * mask


def graph_augmentations(x: Tensor, edge_index: Tensor, edge_drop: float, feature_mask: float) -> Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]:
    weak_x, weak_edge = x, edge_index
    strong_edge = drop_edges(edge_index, edge_drop)
    strong_x = mask_features(x, feature_mask)
    return (weak_x, weak_edge), (strong_x, strong_edge)


class Mixup(nn.Module):
    def __init__(self, alpha: float = 0.75):
        super().__init__()
        self.alpha = alpha

    def forward(self, z: Tensor, targets: Tensor) -> Tuple[Tensor, Tensor]:
        if self.alpha <= 0:
            return z, targets
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample().to(z.device)
        lam = max(lam, 1 - lam)
        index = torch.randperm(z.size(0), device=z.device)
        z_mix = lam * z + (1 - lam) * z[index]
        t_mix = lam * targets + (1 - lam) * targets[index]
        return z_mix, t_mix
