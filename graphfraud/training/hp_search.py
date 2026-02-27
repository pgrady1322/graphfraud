"""
GraphFraud v0.1.0

hp_search.py — Optuna hyperparameter search for GNN and XGBoost models.

Mirrors the Optuna pattern from strandweaver's ErrorSmith training
and immunoclassifier's benchmark pipeline.

Author: Patrick Grady
Anthropic Claude Opus 4.6 used for code formatting and cleanup assistance.
License: MIT License - See LICENSE
"""

import logging

import torch
from sklearn.metrics import f1_score

logger = logging.getLogger("graphfraud")

try:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False


def gnn_objective(trial, pyg_data, model_type: str = "gatv2", device=None):
    """Optuna objective for GNN HP search."""
    if device is None:
        device = torch.device("cpu")

    # Suggest hyperparameters
    hidden_channels = trial.suggest_categorical("hidden_channels", [16, 32, 64, 128])
    heads = trial.suggest_categorical("heads", [4, 8]) if model_type == "gatv2" else 1
    num_layers = trial.suggest_int("num_layers", 2, 4)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    focal_gamma = trial.suggest_float("focal_gamma", 0.5, 5.0)

    cfg = {
        "model": {
            "type": model_type,
            "params": {
                "hidden_channels": hidden_channels,
                "heads": heads,
                "num_layers": num_layers,
                "dropout": dropout,
            },
        },
    }

    from graphfraud.data.resampling import compute_class_weights
    from graphfraud.training.trainer import FocalLoss, build_model

    model = build_model(cfg, num_features=pyg_data.x.shape[1]).to(device)

    # Compute class weights
    train_labels = pyg_data.y[pyg_data.train_mask].cpu().numpy()
    weights = compute_class_weights(train_labels)
    weight_tensor = torch.tensor([weights.get(i, 1.0) for i in range(2)], dtype=torch.float32).to(
        device
    )

    criterion = FocalLoss(alpha=weight_tensor, gamma=focal_gamma)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Quick training (fewer epochs for HP search)
    model.train()
    for epoch in range(50):
        optimizer.zero_grad()
        out = model(pyg_data.x, pyg_data.edge_index)
        loss = criterion(out[pyg_data.train_mask], pyg_data.y[pyg_data.train_mask])
        loss.backward()
        optimizer.step()

        # Pruning
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                out = model(pyg_data.x, pyg_data.edge_index)
                val_pred = out[pyg_data.val_mask].argmax(dim=-1).cpu().numpy()
                val_true = pyg_data.y[pyg_data.val_mask].cpu().numpy()
                val_f1 = f1_score(val_true, val_pred, average="binary", pos_label=1)

            trial.report(val_f1, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
            model.train()

    # Final validation
    model.eval()
    with torch.no_grad():
        out = model(pyg_data.x, pyg_data.edge_index)
        val_pred = out[pyg_data.val_mask].argmax(dim=-1).cpu().numpy()
        val_true = pyg_data.y[pyg_data.val_mask].cpu().numpy()
        val_f1 = f1_score(val_true, val_pred, average="binary", pos_label=1)

    return val_f1


def xgboost_objective(trial, X_train, y_train, X_val, y_val):
    """Optuna objective for XGBoost HP search."""
    import xgboost as xgb

    params = {
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 800),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 15),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "gamma": trial.suggest_float("gamma", 0, 5.0),
        "tree_method": "hist",
        "eval_metric": "logloss",
        "random_state": 42,
    }

    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    y_pred = model.predict(X_val)
    return f1_score(y_val, y_pred, average="binary", pos_label=1)


def run_hp_search(
    model_type: str,
    n_trials: int = 50,
    pyg_data=None,
    X_train=None,
    y_train=None,
    X_val=None,
    y_val=None,
    device=None,
) -> dict:
    """
    Run Optuna HP search.

    Args:
        model_type: 'gatv2', 'graphsage', 'gcn', or 'xgboost'
        n_trials: Number of Optuna trials
        pyg_data: PyG Data object (for GNN models)
        X_train, y_train, X_val, y_val: numpy arrays (for XGBoost)
        device: torch device

    Returns:
        Best parameters dict
    """
    if not HAS_OPTUNA:
        raise ImportError("Optuna required. Install with: pip install -e '.[optuna]'")

    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
    )

    if model_type in ("gatv2", "graphsage", "gcn"):
        assert pyg_data is not None, "pyg_data required for GNN HP search"
        study.optimize(
            lambda trial: gnn_objective(trial, pyg_data, model_type, device),
            n_trials=n_trials,
            show_progress_bar=True,
        )
    elif model_type == "xgboost":
        assert X_train is not None, "X_train/y_train required for XGBoost HP search"
        study.optimize(
            lambda trial: xgboost_objective(trial, X_train, y_train, X_val, y_val),
            n_trials=n_trials,
            show_progress_bar=True,
        )

    logger.info(f"\nBest trial: F1={study.best_value:.4f}")
    logger.info(f"Best params: {study.best_params}")

    return study.best_params


# GraphFraud v0.1.0
# Any usage is subject to this software's license.
