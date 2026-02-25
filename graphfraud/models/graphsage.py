"""
GraphFraud v0.1.0

graphsage.py — GraphSAGE inductive fraud classifier.

Inductive model that samples and aggregates neighbor features.
Good baseline — faster than GATv2 but doesn't learn attention weights.

Reference:
    Hamilton, Ying, Leskovec. "Inductive Representation Learning on
    Large Graphs" (NeurIPS 2017)

Author: Patrick Grady
Anthropic Claude Opus 4.6 used for code formatting and cleanup assistance.
License: MIT License - See LICENSE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import SAGEConv
except ImportError:
    raise ImportError("PyTorch Geometric required. Install with: pip install -e '.[gnn]'")


class GraphSAGEFraudClassifier(nn.Module):
    """GraphSAGE node classifier for fraud detection."""

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
        self.norms = nn.ModuleList()

        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.norms.append(nn.BatchNorm1d(hidden_channels))

        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.norms.append(nn.BatchNorm1d(hidden_channels))

        self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.norms.append(nn.BatchNorm1d(hidden_channels))

        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, x, edge_index):
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        return self.mlp(x)

# GraphFraud v0.1.0
# Any usage is subject to this software's license.
