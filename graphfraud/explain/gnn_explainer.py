"""
GraphFraud v0.1.0

gnn_explainer.py — GNNExplainer for transaction-level attribution.

Answers: "Why did the model flag this transaction as illicit?"
- Which input features were most important?
- Which neighboring transactions influenced the decision?
- What subgraph structure triggered the classification?

Author: Patrick Grady
Anthropic Claude Opus 4.6 used for code formatting and cleanup assistance.
License: MIT License - See LICENSE
"""

import logging
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger("graphfraud")


def explain_node(
    model_path: str,
    node_id: int,
    data_dir: str = "data/",
    top_k_features: int = 10,
    top_k_neighbors: int = 5,
) -> dict:
    """
    Explain a single node prediction using GNNExplainer.

    Args:
        model_path: Path to saved GNN model (.pt)
        node_id: Node index to explain
        data_dir: Path to Elliptic dataset
        top_k_features: Number of top features to return
        top_k_neighbors: Number of top neighbor nodes to return

    Returns:
        Dict with explanation details
    """
    try:
        from torch_geometric.explain import Explainer, GNNExplainer
    except ImportError:
        raise ImportError(
            "PyTorch Geometric required for explanations. Install with: pip install -e '.[gnn]'"
        ) from None

    from graphfraud.data.dataset import load_elliptic, to_pyg
    from graphfraud.training.trainer import build_model

    # Load model
    checkpoint = torch.load(model_path, weights_only=False)
    cfg = checkpoint["config"]

    dataset = load_elliptic(Path(data_dir), temporal_split=True)
    pyg_data = to_pyg(dataset)

    model = build_model(cfg, num_features=dataset.num_features)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Get prediction for this node
    with torch.no_grad():
        out = model(pyg_data.x, pyg_data.edge_index)
        probs = torch.softmax(out, dim=-1)
        pred_class = out[node_id].argmax().item()
        pred_prob = probs[node_id].numpy()

    # Run GNNExplainer
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=200),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=dict(
            mode="multiclass_classification",
            task_level="node",
            return_type="log_probs",
        ),
    )

    explanation = explainer(pyg_data.x, pyg_data.edge_index, index=node_id)

    # Extract feature importances
    node_mask = explanation.node_mask
    if node_mask is not None:
        feature_importance = node_mask[node_id].numpy()
        top_feat_idx = np.argsort(np.abs(feature_importance))[-top_k_features:][::-1]
        top_features = [
            {"feature_idx": int(idx), "importance": float(feature_importance[idx])}
            for idx in top_feat_idx
        ]
    else:
        top_features = []

    # Extract important edges/neighbors
    edge_mask = explanation.edge_mask
    if edge_mask is not None:
        edge_importance = edge_mask.numpy()
        # Find edges connected to this node
        edge_src, edge_dst = pyg_data.edge_index.numpy()
        node_edges = np.where((edge_src == node_id) | (edge_dst == node_id))[0]

        if len(node_edges) > 0:
            important_edges = node_edges[
                np.argsort(edge_importance[node_edges])[-top_k_neighbors:][::-1]
            ]
            key_neighbors = []
            for eidx in important_edges:
                neighbor = int(edge_dst[eidx]) if edge_src[eidx] == node_id else int(edge_src[eidx])
                key_neighbors.append(
                    {
                        "neighbor_id": neighbor,
                        "edge_importance": float(edge_importance[eidx]),
                        "neighbor_label": int(pyg_data.y[neighbor].item())
                        if pyg_data.y[neighbor] >= 0
                        else "unknown",
                    }
                )
        else:
            key_neighbors = []
    else:
        key_neighbors = []

    result = {
        "node_id": node_id,
        "predicted_class": "illicit" if pred_class == 1 else "licit",
        "prediction_probabilities": {"licit": float(pred_prob[0]), "illicit": float(pred_prob[1])},
        "true_label": int(pyg_data.y[node_id].item()) if pyg_data.y[node_id] >= 0 else "unknown",
        "top_features": top_features,
        "key_neighbors": key_neighbors,
    }

    logger.info(f"Explanation for node {node_id}:")
    logger.info(f"  Predicted: {result['predicted_class']} (p={pred_prob[pred_class]:.3f})")
    logger.info(f"  True: {result['true_label']}")
    logger.info(f"  Top features: {[f['feature_idx'] for f in top_features]}")
    logger.info(f"  Key neighbors: {[n['neighbor_id'] for n in key_neighbors]}")

    return result


# GraphFraud v0.1.0
# Any usage is subject to this software's license.
