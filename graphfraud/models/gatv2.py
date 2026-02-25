# ═══════════════════════════════════════════════════════════════════════
# GraphFraud — GATv2Conv Fraud Classifier
# ═══════════════════════════════════════════════════════════════════════
"""
Graph Attention Network v2 (GATv2Conv) for node-level fraud classification.

Architecture mirrors the GATv2 pattern used in strandweaver's PathGNN
for assembly graph path resolution, adapted for transaction graphs.

Reference:
    Brody, Alon, Yahav. "How Attentive are Graph Attention Networks?" (ICLR 2022)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GATv2Conv, global_mean_pool
except ImportError:
    raise ImportError("PyTorch Geometric required. Install with: pip install -e '.[gnn]'")


class GATv2FraudClassifier(nn.Module):
    """
    Multi-layer GATv2Conv classifier for transaction-level fraud detection.

    Architecture:
        Input (166 features)
          → GATv2Conv (multi-head) + ELU + Dropout
          → GATv2Conv (multi-head) + ELU + Dropout
          → GATv2Conv (single-head)  + ELU
          → MLP Head → 2 classes

    The attention mechanism learns *which* neighboring transactions
    are most informative for classifying a given node.
    """

    def __init__(
        self,
        in_channels: int = 166,
        hidden_channels: int = 32,
        out_channels: int = 2,
        heads: int = 8,
        num_layers: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.dropout = dropout
        self.num_layers = num_layers

        # GATv2 convolution layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # First layer: in_channels → hidden_channels * heads
        self.convs.append(
            GATv2Conv(in_channels, hidden_channels, heads=heads, dropout=dropout, concat=True)
        )
        self.norms.append(nn.BatchNorm1d(hidden_channels * heads))

        # Middle layers: hidden_channels * heads → hidden_channels * heads
        for _ in range(num_layers - 2):
            self.convs.append(
                GATv2Conv(
                    hidden_channels * heads, hidden_channels, heads=heads,
                    dropout=dropout, concat=True,
                )
            )
            self.norms.append(nn.BatchNorm1d(hidden_channels * heads))

        # Final GATv2 layer: → hidden_channels (single head, no concat)
        self.convs.append(
            GATv2Conv(
                hidden_channels * heads, hidden_channels, heads=1,
                dropout=dropout, concat=False,
            )
        )
        self.norms.append(nn.BatchNorm1d(hidden_channels))

        # MLP classification head
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, x, edge_index, return_attention=False):
        """
        Forward pass.

        Args:
            x: Node features (N, in_channels)
            edge_index: Edge connectivity (2, E)
            return_attention: If True, return attention weights from last layer

        Returns:
            logits: (N, out_channels)
            attention_weights: (optional) edge attention from the final layer
        """
        attention_weights = None

        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            if i < self.num_layers - 1:
                x = conv(x, edge_index)
                x = norm(x)
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            else:
                # Last layer — optionally return attention
                if return_attention:
                    x, (edge_idx, attn) = conv(x, edge_index, return_attention_weights=True)
                    attention_weights = attn
                else:
                    x = conv(x, edge_index)
                x = norm(x)
                x = F.elu(x)

        logits = self.mlp(x)

        if return_attention:
            return logits, attention_weights
        return logits

    def predict_proba(self, x, edge_index):
        """Return softmax probabilities."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x, edge_index)
            return F.softmax(logits, dim=-1)
