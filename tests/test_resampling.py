"""
GraphFraud v0.1.0

test_resampling.py — Tests for hybrid resampling and class weight computation.

Author: Patrick Grady
Anthropic Claude Opus 4.6 used for code formatting and cleanup assistance.
License: MIT License - See LICENSE
"""

import numpy as np
import pytest

from graphfraud.data.resampling import (
    compute_class_weights,
    focal_loss_weights,
    hybrid_resample,
)


class TestHybridResample:
    """Tests for hybrid_resample."""

    @pytest.fixture
    def imbalanced_data(self):
        np.random.seed(42)
        n_majority = 10_000
        n_minority = 500
        X = np.random.randn(n_majority + n_minority, 10).astype(np.float32)
        y = np.concatenate([np.zeros(n_majority), np.ones(n_minority)]).astype(np.int64)
        return X, y

    def test_resampling_balances(self, imbalanced_data):
        X, y = imbalanced_data
        X_rs, y_rs = hybrid_resample(X, y, max_majority=5000)

        unique, counts = np.unique(y_rs, return_counts=True)
        # After resampling, classes should be more balanced
        ratio = max(counts) / min(counts)
        assert ratio < 3.0, f"Ratio still too high after resampling: {ratio}"

    def test_caps_majority(self, imbalanced_data):
        X, y = imbalanced_data
        X_rs, y_rs = hybrid_resample(X, y, max_majority=3000)

        majority_count = (y_rs == 0).sum()
        assert majority_count <= 3000

    def test_deterministic(self, imbalanced_data):
        X, y = imbalanced_data
        X1, y1 = hybrid_resample(X, y, random_state=42)
        X2, y2 = hybrid_resample(X, y, random_state=42)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_preserves_features(self, imbalanced_data):
        X, y = imbalanced_data
        X_rs, y_rs = hybrid_resample(X, y)
        assert X_rs.shape[1] == X.shape[1]


class TestClassWeights:
    def test_inverse_frequency(self):
        y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])  # 9:1 imbalance
        weights = compute_class_weights(y)
        assert weights[1] > weights[0], "Minority class should get higher weight"
        assert len(weights) == 2

    def test_ignores_unlabeled(self):
        y = np.array([0, 0, 1, -1, -1, -1])
        weights = compute_class_weights(y)
        assert -1 not in weights
        assert len(weights) == 2


class TestFocalLossWeights:
    def test_minority_gets_higher_weight(self):
        y = np.array([0] * 90 + [1] * 10)
        weights = focal_loss_weights(y, gamma=2.0)
        avg_majority = weights[y == 0].mean()
        avg_minority = weights[y == 1].mean()
        assert avg_minority > avg_majority

    def test_gamma_zero_uniform(self):
        y = np.array([0] * 50 + [1] * 50)
        weights = focal_loss_weights(y, gamma=0.0)
        # gamma=0 → all weights = (1-0.5)^0 = 1.0
        np.testing.assert_array_almost_equal(weights, 1.0)


# GraphFraud v0.1.0
# Any usage is subject to this software's license.
