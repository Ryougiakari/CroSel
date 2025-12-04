import logging
from collections import deque
from typing import Deque, Tuple

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F

from utils.graph_augment import graph_augmentations, Mixup
from model.gnn import normalize_adj


logger = logging.getLogger(__name__)


def compute_hard_labels(logits: Tensor) -> Tensor:
    return logits.argmax(dim=1)


def average_max_probs(pred_history: Deque[Tensor]) -> Tensor:
    stacked = torch.stack(pred_history, dim=0)
    return stacked.softmax(dim=-1).max(dim=-1).values.mean(dim=0)


def mode_labels(history: Deque[Tensor]) -> Tensor:
    hard = torch.stack([h.argmax(dim=1) for h in history], dim=0)
    modes = []
    for node_col in hard.t():
        counts = torch.bincount(node_col)
        modes.append(counts.argmax())
    return torch.stack(modes)


def structural_consistency(hard_labels: Tensor, edge_index: Tensor, num_nodes: int, threshold: float) -> Tensor:
    row, col = edge_index
    agree = (hard_labels[row] == hard_labels[col]).float()
    degree = torch.bincount(row, minlength=num_nodes).float().clamp(min=1)
    scores = torch.zeros(num_nodes, device=hard_labels.device)
    scores.index_add_(0, row, agree)
    scores = scores / degree
    return (scores >= threshold).float()


class GraphCroSelTrainer:
    def __init__(self, model1: nn.Module, model2: nn.Module, optimizer1: torch.optim.Optimizer,
                 optimizer2: torch.optim.Optimizer, num_nodes: int, num_clusters: int, edge_index: Tensor,
                 device: torch.device, args):
        self.model1 = model1
        self.model2 = model2
        self.optimizer1 = optimizer1
        self.optimizer2 = optimizer2
        self.edge_index = edge_index
        self.device = device
        self.num_nodes = num_nodes
        self.num_clusters = num_clusters
        self.args = args
        self.mixup = Mixup(alpha=args.alpha)
        self.memory1: Deque[Tensor] = deque(maxlen=args.memory_length)
        self.memory2: Deque[Tensor] = deque(maxlen=args.memory_length)

    def _update_memory(self, bank: Deque[Tensor], logits: Tensor):
        bank.append(logits.detach())

    def _select_reliable(self, history: Deque[Tensor], hard_current: Tensor) -> Tensor:
        if len(history) == 0:
            return torch.zeros_like(hard_current, dtype=torch.bool)
        temporal = (hard_current == mode_labels(history)).float()
        confidence = (average_max_probs(history) >= self.args.confidence_threshold).float()
        structural = structural_consistency(hard_current, self.edge_index, self.num_nodes, self.args.struct_threshold)
        return (temporal * confidence * structural).bool()

    def _graphmix_consistency(self, model: nn.Module, other_logits: Tensor, x_s: Tensor, edge_s: Tensor) -> Tensor:
        pseudo = other_logits.softmax(dim=-1)
        sharp = pseudo / pseudo.sum(dim=1, keepdim=True)
        strong_norm = normalize_adj(edge_s, self.num_nodes).to(self.device)
        z_s, logits_s = model(x_s, edge_s, strong_norm)
        z_mix, t_mix = self.mixup(z_s, sharp)
        logits_mix = model.head(z_mix) if hasattr(model, 'head') else logits_s
        return F.kl_div(F.log_softmax(logits_mix, dim=-1), t_mix, reduction='batchmean')

    def train_epoch(self, x: Tensor, norm_adj: Tensor, epoch: int, warmup: bool) -> Tuple[Tensor, Tensor]:
        self.model1.train()
        self.model2.train()
        self.optimizer1.zero_grad()
        self.optimizer2.zero_grad()

        (weak_x, weak_edge), (strong_x, strong_edge) = graph_augmentations(
            x, self.edge_index, self.args.edge_drop, self.args.feature_mask)

        _, logits1 = self.model1(weak_x, weak_edge, norm_adj)
        _, logits2 = self.model2(weak_x, weak_edge, norm_adj)

        self._update_memory(self.memory1, logits1)
        self._update_memory(self.memory2, logits2)

        if warmup or len(self.memory1) < self.args.memory_length or len(self.memory2) < self.args.memory_length:
            loss1 = F.cross_entropy(logits1, logits1.softmax(dim=-1).detach())
            loss2 = F.cross_entropy(logits2, logits2.softmax(dim=-1).detach())
            loss = loss1 + loss2
        else:
            reliable1 = self._select_reliable(self.memory1, logits1.argmax(dim=1))
            reliable2 = self._select_reliable(self.memory2, logits2.argmax(dim=1))

            pseudo1 = logits1.argmax(dim=1).detach()
            pseudo2 = logits2.argmax(dim=1).detach()

            sup_loss1 = F.cross_entropy(logits1[reliable2], pseudo2[reliable2]) if reliable2.any() else torch.tensor(0.0, device=self.device)
            sup_loss2 = F.cross_entropy(logits2[reliable1], pseudo1[reliable1]) if reliable1.any() else torch.tensor(0.0, device=self.device)

            cons1 = self._graphmix_consistency(self.model1, logits2.detach(), strong_x, strong_edge)
            cons2 = self._graphmix_consistency(self.model2, logits1.detach(), strong_x, strong_edge)

            rs = (reliable1.float().mean() + reliable2.float().mean()) / 2
            lambda_cons = (1 - rs) * self.args.base_consistency_weight

            loss = sup_loss1 + sup_loss2 + lambda_cons * (cons1 + cons2)

        loss.backward()
        self.optimizer1.step()
        self.optimizer2.step()
        return logits1.detach(), logits2.detach()

    def inference(self, x: Tensor, edge_index: Tensor, norm_adj: Tensor) -> Tensor:
        self.model1.eval()
        self.model2.eval()
        with torch.no_grad():
            _, logits1 = self.model1(x, edge_index, norm_adj)
            _, logits2 = self.model2(x, edge_index, norm_adj)
            prob = 0.5 * (logits1.softmax(dim=-1) + logits2.softmax(dim=-1))
        return prob.argmax(dim=1)
