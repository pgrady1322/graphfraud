"""
GraphFraud v0.1.0

sklearn_baselines.py — Logistic Regression and Random Forest baselines.

Non-graph sklearn baselines for fraud detection benchmarking.
If the GNN can't beat these, graph topology isn't adding signal.

Author: Patrick Grady
Anthropic Claude Opus 4.6 used for code formatting and cleanup assistance.
License: MIT License - See LICENSE
"""

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import StandardScaler

from graphfraud.data.resampling import hybrid_resample

logger = logging.getLogger("graphfraud")


def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    resample: bool = True,
    max_majority: int = 30_000,
    C: float = 1.0,
    max_iter: int = 1000,
    random_state: int = 42,
    save_path: Optional[Path] = None,
) -> tuple[LogisticRegression, dict]:
    """
    Train L2-regularized Logistic Regression baseline.

    Features are standardized before fitting (required for L2 regularization
    to penalize coefficients on the same scale).

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        resample: Apply hybrid resampling
        max_majority: Cap for majority class
        C: Inverse regularization strength
        max_iter: Maximum solver iterations
        random_state: Random seed
        save_path: Optional path to save model

    Returns:
        (model, results_dict)
    """
    if resample:
        X_train, y_train = hybrid_resample(X_train, y_train, max_majority=max_majority)

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    logger.info(f"Training Logistic Regression: {X_train_scaled.shape[0]:,} samples, C={C}")

    model = LogisticRegression(
        C=C,
        l1_ratio=0,  # equivalent to penalty="l2" (avoids deprecation warning)
        solver="lbfgs",
        max_iter=max_iter,
        random_state=random_state,
        class_weight="balanced",
    )
    model.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = model.predict(X_val_scaled)
    y_prob = model.predict_proba(X_val_scaled)[:, 1]
    f1 = f1_score(y_val, y_pred, average="binary", pos_label=1)
    report = classification_report(y_val, y_pred, target_names=["licit", "illicit"], digits=3)

    logger.info(f"Logistic Regression — F1 (illicit): {f1:.4f}")
    logger.info(f"\n{report}")

    results = {
        "model": "logistic_regression",
        "f1_illicit": float(f1),
        "f1_macro": float(f1_score(y_val, y_pred, average="macro")),
        "classification_report": report,
        "params": {"C": C, "max_iter": max_iter, "penalty": "l2"},
    }

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump({"model": model, "scaler": scaler}, f)
        logger.info(f"✓ Saved: {save_path}")

    return model, results


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    resample: bool = True,
    max_majority: int = 30_000,
    n_estimators: int = 300,
    max_depth: int = 12,
    min_samples_leaf: int = 5,
    random_state: int = 42,
    n_jobs: int = -1,
    save_path: Optional[Path] = None,
) -> tuple[RandomForestClassifier, dict]:
    """
    Train Random Forest baseline.

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        resample: Apply hybrid resampling
        max_majority: Cap for majority class
        n_estimators: Number of trees
        max_depth: Maximum tree depth
        min_samples_leaf: Minimum samples per leaf
        random_state: Random seed
        n_jobs: Parallel jobs (-1 = all cores)
        save_path: Optional path to save model

    Returns:
        (model, results_dict)
    """
    if resample:
        X_train, y_train = hybrid_resample(X_train, y_train, max_majority=max_majority)

    logger.info(
        f"Training Random Forest: {X_train.shape[0]:,} samples, "
        f"{n_estimators} trees, max_depth={max_depth}"
    )

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=n_jobs,
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]
    f1 = f1_score(y_val, y_pred, average="binary", pos_label=1)
    report = classification_report(y_val, y_pred, target_names=["licit", "illicit"], digits=3)

    logger.info(f"Random Forest — F1 (illicit): {f1:.4f}")
    logger.info(f"\n{report}")

    results = {
        "model": "random_forest",
        "f1_illicit": float(f1),
        "f1_macro": float(f1_score(y_val, y_pred, average="macro")),
        "classification_report": report,
        "params": {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
        },
    }

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(model, f)
        logger.info(f"✓ Saved: {save_path}")

    return model, results

# GraphFraud v0.1.0
# Any usage is subject to this software's license.
