# ═══════════════════════════════════════════════════════════════════════
# GraphFraud — Model Evaluation
# ═══════════════════════════════════════════════════════════════════════
"""
Evaluation metrics and reporting for fraud detection models.

Emphasizes precision-recall tradeoffs and F1 on the minority (illicit) class,
since accuracy is misleading with ~9:1 class imbalance.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
    average_precision_score,
)

logger = logging.getLogger("graphfraud")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None) -> dict:
    """
    Compute comprehensive evaluation metrics.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities for positive class (optional)

    Returns:
        Dict of metrics
    """
    results = {
        "f1_illicit": float(f1_score(y_true, y_pred, average="binary", pos_label=1)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, target_names=["licit", "illicit"], digits=4
        ),
    }

    if y_prob is not None:
        results["auc_roc"] = float(roc_auc_score(y_true, y_prob))
        results["avg_precision"] = float(average_precision_score(y_true, y_prob))

    return results


def plot_evaluation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    save_dir: Optional[Path] = None,
    model_name: str = "model",
):
    """
    Generate evaluation plots: confusion matrix, ROC, PR curve.

    Args:
        y_true: Ground truth
        y_pred: Predictions
        y_prob: Positive class probabilities
        save_dir: Directory to save plots
        model_name: Name for plot titles and filenames
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Evaluation — {model_name}", fontsize=14, fontweight="bold")

    # ── Confusion Matrix ────────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred)
    im = axes[0].imshow(cm, interpolation="nearest", cmap="Blues")
    axes[0].set_title("Confusion Matrix")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    axes[0].set_xticks([0, 1])
    axes[0].set_yticks([0, 1])
    axes[0].set_xticklabels(["Licit", "Illicit"])
    axes[0].set_yticklabels(["Licit", "Illicit"])

    # Annotate cells
    for i in range(2):
        for j in range(2):
            axes[0].text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                        fontsize=14, color="white" if cm[i, j] > cm.max() / 2 else "black")

    if y_prob is not None:
        # ── ROC Curve ───────────────────────────────────────────────
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        axes[1].plot(fpr, tpr, color="coral", lw=2, label=f"AUC = {auc:.3f}")
        axes[1].plot([0, 1], [0, 1], "k--", lw=1)
        axes[1].set_xlabel("False Positive Rate")
        axes[1].set_ylabel("True Positive Rate")
        axes[1].set_title("ROC Curve")
        axes[1].legend()

        # ── Precision-Recall Curve ──────────────────────────────────
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        axes[2].plot(recall, precision, color="steelblue", lw=2, label=f"AP = {ap:.3f}")
        axes[2].set_xlabel("Recall")
        axes[2].set_ylabel("Precision")
        axes[2].set_title("Precision-Recall Curve")
        axes[2].legend()
    else:
        axes[1].text(0.5, 0.5, "No probabilities\nprovided", ha="center", va="center")
        axes[2].text(0.5, 0.5, "No probabilities\nprovided", ha="center", va="center")

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_dir / f"{model_name}_evaluation.png", dpi=150, bbox_inches="tight")
        logger.info(f"✓ Saved plot: {save_dir / f'{model_name}_evaluation.png'}")

    plt.show()


def evaluate_model(model_path: str, data_dir: str, output_dir: str = "results/") -> dict:
    """
    Load a saved model and evaluate on test data.

    Args:
        model_path: Path to saved model (.pt or .pkl)
        data_dir: Path to Elliptic dataset directory
        output_dir: Directory for evaluation outputs

    Returns:
        Results dict
    """
    import torch

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(data_dir)

    dataset = load_elliptic(data_dir, temporal_split=True)

    model_path = Path(model_path)

    if model_path.suffix == ".pt":
        # GNN model
        pyg_data = to_pyg(dataset)
        checkpoint = torch.load(model_path, weights_only=False)
        cfg = checkpoint["config"]

        from graphfraud.training.trainer import build_model
        model = build_model(cfg, num_features=dataset.num_features)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        with torch.no_grad():
            out = model(pyg_data.x, pyg_data.edge_index)
            probs = torch.softmax(out, dim=-1)

            test_pred = out[pyg_data.test_mask].argmax(dim=-1).numpy()
            test_true = pyg_data.y[pyg_data.test_mask].numpy()
            test_prob = probs[pyg_data.test_mask][:, 1].numpy()

        model_name = cfg["model"]["type"]

    elif model_path.suffix == ".pkl":
        # XGBoost model
        import pickle
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        X_test = dataset.node_features[dataset.test_mask]
        test_true = dataset.labels[dataset.test_mask]
        test_pred = model.predict(X_test)
        test_prob = model.predict_proba(X_test)[:, 1]
        model_name = "xgboost"

    else:
        raise ValueError(f"Unknown model format: {model_path.suffix}")

    # Compute metrics
    results = compute_metrics(test_true, test_pred, test_prob)
    results["model"] = model_name

    # Plot
    plot_evaluation(test_true, test_pred, test_prob, save_dir=output_dir, model_name=model_name)

    # Save results
    results_path = output_dir / f"{model_name}_test_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"✓ Results saved: {results_path}")

    return results


# Import here to avoid circular
from graphfraud.data.dataset import load_elliptic, to_pyg
