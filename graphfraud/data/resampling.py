"""
GraphFraud v0.1.0

resampling.py — Hybrid resampling for extreme class imbalance.

The Elliptic dataset has ~9:1 licit:illicit ratio among labeled nodes.
These methods balance training data without discarding too much majority class.

Author: Patrick Grady
Anthropic Claude Opus 4.6 used for code formatting and cleanup assistance.
License: MIT License - See LICENSE
"""

import logging
from typing import Optional

import numpy as np
from sklearn.utils import resample

logger = logging.getLogger("graphfraud")


def hybrid_resample(
    X: np.ndarray,
    y: np.ndarray,
    max_majority: int = 30_000,
    min_minority: Optional[int] = None,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Hybrid resampling: undersample majority, oversample minority.

    Strategy:
        1. Cap majority class at max_majority (undersample without replacement)
        2. Oversample minority class(es) to median of capped sizes
        3. If min_minority specified, ensure at least that many minority samples

    This is adapted from strandweaver's ErrorSmith training pipeline,
    where 5-class imbalance (>100:1 correct:error) required aggressive rebalancing.

    Args:
        X: Feature matrix (N, D)
        y: Labels (N,)
        max_majority: Cap for the largest class
        min_minority: Minimum samples for minority classes (optional)
        random_state: Random seed for reproducibility

    Returns:
        X_resampled, y_resampled
    """
    classes, counts = np.unique(y, return_counts=True)
    class_counts = dict(zip(classes, counts))

    logger.info(f"Original distribution: {class_counts}")

    # Cap majority
    capped = {c: min(n, max_majority) for c, n in class_counts.items()}
    median_size = int(np.median(list(capped.values())))

    if min_minority is not None:
        median_size = max(median_size, min_minority)

    X_parts, y_parts = [], []
    for cls in classes:
        mask = y == cls
        X_cls, y_cls = X[mask], y[mask]
        target = max(median_size, min(len(X_cls), max_majority))

        if len(X_cls) > target:
            X_rs, y_rs = resample(
                X_cls, y_cls,
                n_samples=target,
                random_state=random_state,
                replace=False,
            )
        elif len(X_cls) < target:
            X_rs, y_rs = resample(
                X_cls, y_cls,
                n_samples=target,
                random_state=random_state,
                replace=True,
            )
        else:
            X_rs, y_rs = X_cls, y_cls

        X_parts.append(X_rs)
        y_parts.append(y_rs)

    X_out = np.vstack(X_parts)
    y_out = np.concatenate(y_parts)

    new_counts = dict(zip(*np.unique(y_out, return_counts=True)))
    logger.info(f"Resampled distribution: {new_counts} (total: {len(y_out):,})")

    return X_out, y_out


def compute_class_weights(y: np.ndarray) -> dict:
    """
    Compute inverse-frequency class weights for loss weighting.

    Args:
        y: Labels array (may contain -1 for unlabeled)

    Returns:
        Dict mapping class → weight (float)
    """
    labeled = y[y >= 0]
    classes, counts = np.unique(labeled, return_counts=True)
    total = len(labeled)
    weights = {int(c): total / (len(classes) * n) for c, n in zip(classes, counts)}
    logger.info(f"Class weights: {weights}")
    return weights


def focal_loss_weights(y: np.ndarray, gamma: float = 2.0) -> np.ndarray:
    """
    Compute per-sample focal loss weights.

    Focal loss down-weights well-classified examples and focuses on
    hard-to-classify ones. Useful for extreme imbalance.

    w_i = (1 - p_i)^gamma where p_i is the class prior probability.

    Args:
        y: Labels (N,)
        gamma: Focusing parameter (higher = more focus on minority)

    Returns:
        Per-sample weights (N,)
    """
    classes, counts = np.unique(y, return_counts=True)
    priors = dict(zip(classes, counts / len(y)))
    weights = np.array([(1 - priors.get(label, 0.5)) ** gamma for label in y])
    return weights

# GraphFraud v0.1.0
# Any usage is subject to this software's license.
