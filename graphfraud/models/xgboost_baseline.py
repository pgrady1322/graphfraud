# ═══════════════════════════════════════════════════════════════════════
# GraphFraud — XGBoost Baseline (No Graph Structure)
# ═══════════════════════════════════════════════════════════════════════
"""
XGBoost baseline classifier that uses only node features (ignores graph topology).

This establishes a strong non-graph baseline. If the GNN can't beat this,
the graph structure isn't providing useful signal — an important sanity check.

This mirrors the XGBoost training pattern from strandweaver's ErrorSmith
and immunoclassifier pipelines.
"""

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report, f1_score

from graphfraud.data.resampling import hybrid_resample

logger = logging.getLogger("graphfraud")


def train_xgboost_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    resample: bool = True,
    max_majority: int = 30_000,
    use_gpu: bool = False,
    n_estimators: int = 500,
    max_depth: int = 8,
    learning_rate: float = 0.05,
    random_state: int = 42,
    save_path: Optional[Path] = None,
) -> tuple[xgb.XGBClassifier, dict]:
    """
    Train an XGBoost baseline (no graph structure).

    Args:
        X_train, y_train: Training features and labels
        X_val, y_val: Validation features and labels
        resample: Whether to apply hybrid resampling
        max_majority: Cap for majority class in resampling
        use_gpu: Use GPU acceleration
        n_estimators: Number of boosting rounds
        max_depth: Maximum tree depth
        learning_rate: Learning rate
        random_state: Random seed
        save_path: Optional path to save trained model

    Returns:
        (model, results_dict)
    """
    if resample:
        X_train, y_train = hybrid_resample(X_train, y_train, max_majority=max_majority)

    params = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "tree_method": "hist",
        "device": "cuda" if use_gpu else "cpu",
        "eval_metric": "logloss",
        "random_state": random_state,
    }

    logger.info(f"Training XGBoost baseline: {X_train.shape[0]:,} samples, {X_train.shape[1]} features")

    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # Evaluate
    y_pred = model.predict(X_val)
    f1 = f1_score(y_val, y_pred, average="binary", pos_label=1)
    report = classification_report(y_val, y_pred, target_names=["licit", "illicit"], digits=3)

    logger.info(f"XGBoost Baseline — F1 (illicit): {f1:.4f}")
    logger.info(f"\n{report}")

    results = {
        "model": "xgboost_baseline",
        "f1_illicit": float(f1),
        "f1_macro": float(f1_score(y_val, y_pred, average="macro")),
        "classification_report": report,
        "params": params,
    }

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(model, f)
        logger.info(f"✓ Saved: {save_path}")

    return model, results
