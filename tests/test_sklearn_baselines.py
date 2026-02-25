"""
GraphFraud v0.1.0

test_sklearn_baselines.py — Tests for Logistic Regression and Random Forest baselines.

Author: Patrick Grady
Anthropic Claude Opus 4.6 used for code formatting and cleanup assistance.
License: MIT License - See LICENSE
"""

import numpy as np
import pytest


@pytest.fixture
def synthetic_data():
    """Small synthetic binary classification dataset for testing."""
    rng = np.random.RandomState(42)
    n_train, n_val, n_features = 200, 50, 166

    X_train = rng.randn(n_train, n_features)
    # Imbalanced labels: ~85% class 0, ~15% class 1 (mimic Elliptic)
    y_train = np.zeros(n_train, dtype=int)
    y_train[rng.choice(n_train, size=30, replace=False)] = 1

    X_val = rng.randn(n_val, n_features)
    y_val = np.zeros(n_val, dtype=int)
    y_val[rng.choice(n_val, size=8, replace=False)] = 1

    return X_train, y_train, X_val, y_val


class TestLogisticRegression:
    def test_train_returns_model_and_results(self, synthetic_data):
        from graphfraud.models.sklearn_baselines import train_logistic_regression

        X_train, y_train, X_val, y_val = synthetic_data
        model, results = train_logistic_regression(
            X_train, y_train, X_val, y_val, resample=False
        )
        assert hasattr(model, "predict")
        assert hasattr(model, "predict_proba")
        assert "f1_illicit" in results
        assert "f1_macro" in results
        assert results["model"] == "logistic_regression"

    def test_predict_shape(self, synthetic_data):
        from graphfraud.models.sklearn_baselines import train_logistic_regression

        X_train, y_train, X_val, y_val = synthetic_data
        model, _ = train_logistic_regression(
            X_train, y_train, X_val, y_val, resample=False
        )
        preds = model.predict(X_val)
        probs = model.predict_proba(X_val)
        assert preds.shape == (50,)
        assert probs.shape == (50, 2)

    def test_with_resampling(self, synthetic_data):
        from graphfraud.models.sklearn_baselines import train_logistic_regression

        X_train, y_train, X_val, y_val = synthetic_data
        model, results = train_logistic_regression(
            X_train, y_train, X_val, y_val, resample=True, max_majority=100
        )
        assert results["f1_illicit"] >= 0.0

    def test_save_model(self, synthetic_data, tmp_path):
        from graphfraud.models.sklearn_baselines import train_logistic_regression

        save_path = tmp_path / "lr_model.pkl"
        model, _ = train_logistic_regression(
            *synthetic_data, resample=False, save_path=save_path
        )
        assert save_path.exists()


class TestRandomForest:
    def test_train_returns_model_and_results(self, synthetic_data):
        from graphfraud.models.sklearn_baselines import train_random_forest

        X_train, y_train, X_val, y_val = synthetic_data
        model, results = train_random_forest(
            X_train, y_train, X_val, y_val, resample=False, n_estimators=10
        )
        assert hasattr(model, "predict")
        assert hasattr(model, "predict_proba")
        assert "f1_illicit" in results
        assert results["model"] == "random_forest"

    def test_predict_shape(self, synthetic_data):
        from graphfraud.models.sklearn_baselines import train_random_forest

        X_train, y_train, X_val, y_val = synthetic_data
        model, _ = train_random_forest(
            X_train, y_train, X_val, y_val, resample=False, n_estimators=10
        )
        preds = model.predict(X_val)
        probs = model.predict_proba(X_val)
        assert preds.shape == (50,)
        assert probs.shape == (50, 2)

    def test_with_resampling(self, synthetic_data):
        from graphfraud.models.sklearn_baselines import train_random_forest

        X_train, y_train, X_val, y_val = synthetic_data
        model, results = train_random_forest(
            X_train, y_train, X_val, y_val, resample=True,
            max_majority=100, n_estimators=10
        )
        assert results["f1_illicit"] >= 0.0

    def test_feature_importances(self, synthetic_data):
        from graphfraud.models.sklearn_baselines import train_random_forest

        X_train, y_train, X_val, y_val = synthetic_data
        model, _ = train_random_forest(
            X_train, y_train, X_val, y_val, resample=False, n_estimators=10
        )
        importances = model.feature_importances_
        assert importances.shape == (166,)
        assert np.all(importances >= 0)

    def test_save_model(self, synthetic_data, tmp_path):
        from graphfraud.models.sklearn_baselines import train_random_forest

        save_path = tmp_path / "rf_model.pkl"
        model, _ = train_random_forest(
            *synthetic_data, resample=False, n_estimators=10, save_path=save_path
        )
        assert save_path.exists()

# GraphFraud v0.1.0
# Any usage is subject to this software's license.
