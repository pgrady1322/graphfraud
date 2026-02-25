# ═══════════════════════════════════════════════════════════════════════
# GraphFraud — GCN Baseline
# ═══════════════════════════════════════════════════════════════════════
"""
Graph Convolutional Network baseline for fraud classification.

Simplest GNN baseline — no attention, no sampling.
Reference: Kipf & Welling "Semi-Supervised Classification with GCNs" (ICLR 2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GCNConv
except ImportError:
    raise ImportError("PyTorch Geometric required. Install with: pip install -e '.[gnn]'")


class GCNFraudClassifier(nn.Module):
    """Simple GCN node classifier."""

    def __init__(
        self,
        in_channels: int = 166,
        hidden_channels: int = 128,
        out_channels: int = 2,
        num_layers: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, hidden_channels))

        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        return self.mlp(x)
