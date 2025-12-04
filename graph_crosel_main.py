import logging
import os
import numpy as np
import torch
from torch import optim

from model.gnn import SimpleGCN, normalize_adj
from utils.graph_parser import set_graph_parser
from utils.graph_trainer import GraphCroSelTrainer


def load_graph_data(path):
    data = np.load(path)
    x = torch.from_numpy(data['x']).float()
    edge_index = torch.from_numpy(data['edge_index']).long()
    return x, edge_index


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = set_graph_parser()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    x, edge_index = load_graph_data(args.data_path)
    num_nodes, in_dim = x.shape
    x = x.to(device)
    edge_index = edge_index.to(device)
    norm_adj = normalize_adj(edge_index, num_nodes).to(device)

    model1 = SimpleGCN(in_dim, args.hidden_dim, args.embedding_dim, args.num_clusters).to(device)
    model2 = SimpleGCN(in_dim, args.hidden_dim, args.embedding_dim, args.num_clusters).to(device)

    optimizer1 = optim.Adam(model1.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer2 = optim.Adam(model2.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    trainer = GraphCroSelTrainer(model1, model2, optimizer1, optimizer2, num_nodes, args.num_clusters, edge_index, device, args)

    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
    logger = logging.getLogger(__name__)

    for epoch in range(args.epochs):
        warmup = epoch < args.warmup_epochs
        trainer.train_epoch(x, norm_adj, epoch, warmup)
        if epoch % args.log_interval == 0:
            logger.info(
                f"Epoch {epoch:03d} | warmup={warmup} | memory1={len(trainer.memory1)} | memory2={len(trainer.memory2)}")

    assignments = trainer.inference(x, edge_index, norm_adj)
    save_path = os.path.splitext(args.data_path)[0] + '_clusters.npy'
    np.save(save_path, assignments.cpu().numpy())
    logger.info('Saved cluster assignments to %s', save_path)


if __name__ == '__main__':
    main()
