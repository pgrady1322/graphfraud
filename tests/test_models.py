"""
GraphFraud v0.1.0

test_models.py — Tests for GNN and baseline model architectures.

Author: Patrick Grady
Anthropic Claude Opus 4.6 used for code formatting and cleanup assistance.
License: MIT License - See LICENSE
"""

import pytest

# GNN tests require PyTorch Geometric
torch = pytest.importorskip("torch")
pyg = pytest.importorskip("torch_geometric")


class TestGATv2:
    @pytest.fixture
    def sample_graph(self):
        """Small random graph for testing."""
        n_nodes, n_features = 50, 166
        n_edges = 100
        x = torch.randn(n_nodes, n_features)
        edge_index = torch.randint(0, n_nodes, (2, n_edges))
        return x, edge_index

    def test_forward_shape(self, sample_graph):
        from graphfraud.models.gatv2 import GATv2FraudClassifier

        model = GATv2FraudClassifier(in_channels=166, hidden_channels=16, heads=4, num_layers=2)
        x, edge_index = sample_graph
        out = model(x, edge_index)
        assert out.shape == (50, 2)

    def test_attention_weights(self, sample_graph):
        from graphfraud.models.gatv2 import GATv2FraudClassifier

        model = GATv2FraudClassifier(in_channels=166, hidden_channels=16, heads=4, num_layers=2)
        x, edge_index = sample_graph
        out, attn = model(x, edge_index, return_attention=True)
        assert out.shape == (50, 2)
        assert attn is not None

    def test_predict_proba(self, sample_graph):
        from graphfraud.models.gatv2 import GATv2FraudClassifier

        model = GATv2FraudClassifier(in_channels=166, hidden_channels=16, heads=4, num_layers=2)
        x, edge_index = sample_graph
        probs = model.predict_proba(x, edge_index)
        assert probs.shape == (50, 2)
        # Probabilities should sum to 1
        sums = probs.sum(dim=-1)
        torch.testing.assert_close(sums, torch.ones(50), atol=1e-5, rtol=1e-5)


class TestGraphSAGE:
    def test_forward_shape(self):
        from graphfraud.models.graphsage import GraphSAGEFraudClassifier

        model = GraphSAGEFraudClassifier(in_channels=166, hidden_channels=64, num_layers=2)
        x = torch.randn(50, 166)
        edge_index = torch.randint(0, 50, (2, 100))
        out = model(x, edge_index)
        assert out.shape == (50, 2)


class TestGCN:
    def test_forward_shape(self):
        from graphfraud.models.gcn import GCNFraudClassifier

        model = GCNFraudClassifier(in_channels=166, hidden_channels=64, num_layers=2)
        x = torch.randn(50, 166)
        edge_index = torch.randint(0, 50, (2, 100))
        out = model(x, edge_index)
        assert out.shape == (50, 2)


# GraphFraud v0.1.0
# Any usage is subject to this software's license.
