"""
GraphFraud v0.1.0

trainer.py — GNN and baseline model training loop.

Training loop for GNN fraud classifiers with:
- Focal loss for class imbalance
- Early stopping on validation F1
- Temporal train/val/test split (no data leakage)
- Semi-supervised: unlabeled nodes participate in message passing

Author: Patrick Grady
Anthropic Claude Opus 4.6 used for code formatting and cleanup assistance.
License: MIT License - See LICENSE
"""

import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import StandardScaler

from graphfraud.data.dataset import load_elliptic, to_pyg
from graphfraud.data.resampling import compute_class_weights

logger = logging.getLogger("graphfraud")


class FocalLoss(nn.Module):
    """
    Focal loss for class imbalance.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Reduces loss contribution from well-classified examples,
    focuses learning on hard-to-classify minority cases.
    """

    def __init__(self, alpha: torch.Tensor | None = None, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        return focal.mean()


def get_device():
    """Detect best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(cfg: dict, num_features: int = 166):
    """Instantiate a GNN model from config."""
    model_type = cfg["model"]["type"]
    params = cfg["model"].get("params", {})

    if model_type == "gatv2":
        from graphfraud.models.gatv2 import GATv2FraudClassifier

        return GATv2FraudClassifier(in_channels=num_features, **params)
    elif model_type == "graphsage":
        from graphfraud.models.graphsage import GraphSAGEFraudClassifier

        return GraphSAGEFraudClassifier(in_channels=num_features, **params)
    elif model_type == "gcn":
        from graphfraud.models.gcn import GCNFraudClassifier

        return GCNFraudClassifier(in_channels=num_features, **params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train_gnn(cfg: dict):
    """
    Full GNN training pipeline.

    Args:
        cfg: YAML config dict with keys: model, training, data, output
    """
    device = get_device()
    logger.info(f"Device: {device}")

    # ── Load data ───────────────────────────────────────────────────
    data_cfg = cfg.get("data", {})
    data_dir = Path(data_cfg.get("data_dir", "data/"))

    dataset = load_elliptic(
        data_dir,
        temporal_split=True,
        train_timesteps=tuple(data_cfg.get("train_timesteps", [1, 34])),
        val_timesteps=tuple(data_cfg.get("val_timesteps", [35, 42])),
        test_timesteps=tuple(data_cfg.get("test_timesteps", [43, 49])),
    )

    # ── Feature standardization ─────────────────────────────────────
    # Fit scaler on training nodes only, transform all nodes.
    # 166 features span different scales; standardization improves GNN convergence.
    scaler = StandardScaler()
    train_features = dataset.node_features[dataset.train_mask]
    scaler.fit(train_features)
    dataset.node_features = scaler.transform(dataset.node_features).astype(np.float32)
    logger.info("Feature standardization applied (fit on train, transform all)")

    pyg_data = to_pyg(dataset).to(device)
    logger.info(f"PyG Data: {pyg_data}")

    # ── Build model ─────────────────────────────────────────────────
    model = build_model(cfg, num_features=dataset.num_features).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {cfg['model']['type']} ({total_params:,} parameters)")

    # ── Training config ─────────────────────────────────────────────
    train_cfg = cfg.get("training", {})
    epochs = train_cfg.get("epochs", 200)
    lr = train_cfg.get("learning_rate", 0.001)
    weight_decay = train_cfg.get("weight_decay", 5e-4)
    patience = train_cfg.get("patience", 20)
    focal_gamma = train_cfg.get("focal_gamma", 2.0)
    use_focal = train_cfg.get("use_focal_loss", True)

    # ── Class weights for loss ──────────────────────────────────────
    class_weights = compute_class_weights(dataset.labels)
    weight_tensor = torch.tensor(
        [class_weights.get(i, 1.0) for i in range(2)], dtype=torch.float32
    ).to(device)

    if use_focal:
        criterion = FocalLoss(alpha=weight_tensor, gamma=focal_gamma)
        logger.info(f"Loss: FocalLoss (gamma={focal_gamma}, weights={class_weights})")
    else:
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        logger.info(f"Loss: CrossEntropy (weights={class_weights})")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=patience // 3
    )

    # ── Training loop ───────────────────────────────────────────────
    best_val_f1 = 0.0
    best_epoch = 0
    no_improve = 0

    output_dir = Path(cfg.get("output", {}).get("dir", "trained_models/"))
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\nTraining for {epochs} epochs (patience={patience})...")
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        # ── Train ───────────────────────────────────────────────────
        model.train()
        optimizer.zero_grad()

        out = model(pyg_data.x, pyg_data.edge_index)

        # Loss only on labeled training nodes
        train_out = out[pyg_data.train_mask]
        train_y = pyg_data.y[pyg_data.train_mask]
        loss = criterion(train_out, train_y)

        loss.backward()
        optimizer.step()

        # ── Validate ────────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            out = model(pyg_data.x, pyg_data.edge_index)
            val_out = out[pyg_data.val_mask]
            val_y = pyg_data.y[pyg_data.val_mask]

            val_pred = val_out.argmax(dim=-1).cpu().numpy()
            val_true = val_y.cpu().numpy()
            val_f1 = f1_score(val_true, val_pred, average="binary", pos_label=1)

        scheduler.step(val_f1)

        if epoch % 10 == 0 or epoch == 1:
            logger.info(
                f"  Epoch {epoch:3d}/{epochs} — loss={loss.item():.4f}, "
                f"val_F1={val_f1:.4f}, lr={optimizer.param_groups[0]['lr']:.2e}"
            )

        # ── Early stopping ──────────────────────────────────────────
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            no_improve = 0

            # Save best model
            model_path = output_dir / f"{cfg['model']['type']}_best.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": cfg,
                    "epoch": epoch,
                    "val_f1": val_f1,
                },
                model_path,
            )
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info(f"  Early stopping at epoch {epoch} (patience={patience})")
                break

    elapsed = time.time() - t0
    logger.info(f"\nTraining complete in {elapsed:.1f}s")
    logger.info(f"Best val F1: {best_val_f1:.4f} (epoch {best_epoch})")

    # ── Test evaluation ─────────────────────────────────────────────
    checkpoint = torch.load(model_path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        out = model(pyg_data.x, pyg_data.edge_index)
        test_out = out[pyg_data.test_mask]
        test_y = pyg_data.y[pyg_data.test_mask]

        test_pred = test_out.argmax(dim=-1).cpu().numpy()
        test_true = test_y.cpu().numpy()
        test_f1 = f1_score(test_true, test_pred, average="binary", pos_label=1)

    report = classification_report(
        test_true, test_pred, target_names=["licit", "illicit"], digits=3
    )
    logger.info(f"\nTest F1 (illicit): {test_f1:.4f}")
    logger.info(f"\n{report}")

    # Save results
    results = {
        "model": cfg["model"]["type"],
        "best_epoch": best_epoch,
        "val_f1": float(best_val_f1),
        "test_f1": float(test_f1),
        "classification_report": report,
        "elapsed_seconds": elapsed,
        "config": cfg,
    }
    results_path = output_dir / f"{cfg['model']['type']}_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"✓ Model saved: {model_path}")
    logger.info(f"✓ Results saved: {results_path}")

    return model, results


def train_xgboost(cfg: dict):
    """Train XGBoost baseline from config."""
    from graphfraud.models.xgboost_baseline import train_xgboost_baseline

    data_dir = Path(cfg.get("data", {}).get("data_dir", "data/"))
    dataset = load_elliptic(data_dir, temporal_split=True)

    X_train = dataset.node_features[dataset.train_mask]
    y_train = dataset.labels[dataset.train_mask]
    X_val = dataset.node_features[dataset.val_mask]
    y_val = dataset.labels[dataset.val_mask]

    output_dir = Path(cfg.get("output", {}).get("dir", "trained_models/"))
    save_path = output_dir / "xgboost_baseline.pkl"

    xgb_cfg = cfg.get("model", {}).get("params", {})

    model, results = train_xgboost_baseline(
        X_train,
        y_train,
        X_val,
        y_val,
        save_path=save_path,
        **xgb_cfg,
    )

    return model, results


def train_sklearn_baseline(cfg: dict):
    """Train Logistic Regression or Random Forest baseline from config."""
    from graphfraud.models.sklearn_baselines import (
        train_logistic_regression,
        train_random_forest,
    )

    model_type = cfg["model"]["type"]
    data_dir = Path(cfg.get("data", {}).get("data_dir", "data/"))
    dataset = load_elliptic(data_dir, temporal_split=True)

    X_train = dataset.node_features[dataset.train_mask]
    y_train = dataset.labels[dataset.train_mask]
    X_val = dataset.node_features[dataset.val_mask]
    y_val = dataset.labels[dataset.val_mask]

    output_dir = Path(cfg.get("output", {}).get("dir", "trained_models/"))
    params = cfg.get("model", {}).get("params", {})

    if model_type == "logistic_regression":
        save_path = output_dir / "logistic_regression.pkl"
        model, results = train_logistic_regression(
            X_train,
            y_train,
            X_val,
            y_val,
            save_path=save_path,
            **params,
        )
    elif model_type == "random_forest":
        save_path = output_dir / "random_forest.pkl"
        model, results = train_random_forest(
            X_train,
            y_train,
            X_val,
            y_val,
            save_path=save_path,
            **params,
        )
    else:
        raise ValueError(f"Unknown sklearn baseline: {model_type}")

    return model, results


# GraphFraud v0.1.0
# Any usage is subject to this software's license.
