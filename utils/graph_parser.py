import argparse


def set_graph_parser():
    parser = argparse.ArgumentParser(description="GraphCroSel for deep graph clustering")
    parser.add_argument('--gpu-id', default=0, type=int, help='CUDA device id to use')
    parser.add_argument('--data-path', required=True, help='Path to a .npz file containing x and edge_index')
    parser.add_argument('--num-clusters', type=int, required=True, help='Number of target clusters')
    parser.add_argument('--hidden-dim', type=int, default=128, help='Hidden dimension for the GNN encoder')
    parser.add_argument('--embedding-dim', type=int, default=64, help='Output embedding dimension before the cluster head')
    parser.add_argument('--epochs', type=int, default=200, help='Maximum training epochs')
    parser.add_argument('--warmup-epochs', type=int, default=10, help='Warmup epochs using only clustering loss')
    parser.add_argument('--memory-length', type=int, default=3, help='Number of historical predictions to keep')
    parser.add_argument('--struct-threshold', type=float, default=0.6, help='Structural consistency threshold (tau_s)')
    parser.add_argument('--confidence-threshold', type=float, default=0.6, help='Average confidence threshold (gamma)')
    parser.add_argument('--base-consistency-weight', type=float, default=1.0, help='Lambda_0 for consistency term')
    parser.add_argument('--edge-drop', type=float, default=0.2, help='Drop rate for strong graph augmentation')
    parser.add_argument('--feature-mask', type=float, default=0.1, help='Feature masking rate for strong augmentation')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--alpha', type=float, default=0.75, help='Beta distribution parameter for GraphMix')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--log-interval', type=int, default=20, help='Iterations between log prints')
    parser.add_argument('--device', default='cuda', help='Device identifier (e.g., "cuda" or "cpu")')

    return parser.parse_args()
