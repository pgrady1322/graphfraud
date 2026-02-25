# ═══════════════════════════════════════════════════════════════════════
# GraphFraud — Dataset Tests
# ═══════════════════════════════════════════════════════════════════════
"""Tests for Elliptic dataset loading and preprocessing."""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from graphfraud.data.dataset import EllipticDataset, LABEL_MAP


class TestEllipticDataset:
    """Tests for the EllipticDataset dataclass."""

    @pytest.fixture
    def sample_dataset(self):
        """Create a small synthetic dataset for testing."""
        n_nodes = 100
        n_edges = 200
        n_features = 166

        np.random.seed(42)

        labels = np.full(n_nodes, -1, dtype=np.int64)
        labels[:40] = 0  # licit
        labels[40:46] = 1  # illicit (6 nodes — simulates imbalance)

        return EllipticDataset(
            node_features=np.random.randn(n_nodes, n_features).astype(np.float32),
            edge_index=np.random.randint(0, n_nodes, size=(2, n_edges)).astype(np.int64),
            labels=labels,
            timesteps=np.repeat(np.arange(1, 11), 10).astype(np.int64),
            node_ids=np.arange(n_nodes),
            train_mask=np.arange(n_nodes) < 60,
            val_mask=(np.arange(n_nodes) >= 60) & (np.arange(n_nodes) < 80),
            test_mask=np.arange(n_nodes) >= 80,
        )

    def test_properties(self, sample_dataset):
        assert sample_dataset.num_nodes == 100
        assert sample_dataset.num_edges == 200
        assert sample_dataset.num_features == 166

    def test_class_counts(self, sample_dataset):
        counts = sample_dataset.class_counts
        assert counts[0] == 40
        assert counts[1] == 6

    def test_imbalance_ratio(self, sample_dataset):
        ratio = sample_dataset.imbalance_ratio
        assert ratio == pytest.approx(40 / 6, rel=0.01)

    def test_num_labeled(self, sample_dataset):
        assert sample_dataset.num_labeled == 46

    def test_summary(self, sample_dataset):
        summary = sample_dataset.summary()
        assert "100" in summary
        assert "200" in summary
        assert "166" in summary

    def test_label_map(self):
        assert LABEL_MAP["1"] == 0  # licit
        assert LABEL_MAP["2"] == 1  # illicit
