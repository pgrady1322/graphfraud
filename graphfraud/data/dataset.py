"""
GraphFraud v0.1.0

dataset.py — Elliptic Bitcoin dataset loader and preprocessing.

The Elliptic dataset contains:
- 203,769 Bitcoin transactions (nodes)
- 234,355 directed payment flows (edges)
- 166 features per node (94 local + 72 aggregated neighbor features)
- 49 timesteps
- Labels: 1 (licit), 2 (illicit), unknown

References:
    Weber et al. "Anti-Money Laundering in Bitcoin: Experimenting with
    Graph Convolutional Networks for Financial Forensics" (KDD 2019)

Author: Patrick Grady
Anthropic Claude Opus 4.6 used for code formatting and cleanup assistance.
License: MIT License - See LICENSE
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("graphfraud")

# ── Constants ───────────────────────────────────────────────────────
ELLIPTIC_FILES = {
    "features": "elliptic_txs_features.csv",
    "edges": "elliptic_txs_edgelist.csv",
    "classes": "elliptic_txs_classes.csv",
}

# Label mapping: original → internal
# Elliptic: "1" = licit, "2" = illicit, "unknown" = unlabeled
LABEL_MAP = {"1": 0, "2": 1}  # 0 = licit, 1 = illicit


@dataclass
class EllipticDataset:
    """Parsed Elliptic Bitcoin dataset."""

    node_features: np.ndarray  # (N, 166) float32
    edge_index: np.ndarray  # (2, E) int64 — source/target pairs
    labels: np.ndarray  # (N,) int64 — 0=licit, 1=illicit, -1=unknown
    timesteps: np.ndarray  # (N,) int64 — timestep 1-49
    node_ids: np.ndarray  # (N,) original transaction IDs
    train_mask: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))
    val_mask: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))
    test_mask: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))

    @property
    def num_nodes(self) -> int:
        return self.node_features.shape[0]

    @property
    def num_edges(self) -> int:
        return self.edge_index.shape[1]

    @property
    def num_features(self) -> int:
        return self.node_features.shape[1]

    @property
    def num_labeled(self) -> int:
        return int((self.labels >= 0).sum())

    @property
    def class_counts(self) -> dict:
        labeled = self.labels[self.labels >= 0]
        unique, counts = np.unique(labeled, return_counts=True)
        return {int(u): int(c) for u, c in zip(unique, counts)}

    @property
    def imbalance_ratio(self) -> float:
        counts = self.class_counts
        if len(counts) < 2:
            return 0.0
        return max(counts.values()) / max(min(counts.values()), 1)

    def summary(self) -> str:
        lines = [
            f"Elliptic Bitcoin Dataset",
            f"  Nodes:      {self.num_nodes:,}",
            f"  Edges:      {self.num_edges:,}",
            f"  Features:   {self.num_features}",
            f"  Labeled:    {self.num_labeled:,} / {self.num_nodes:,}",
            f"  Classes:    {self.class_counts}",
            f"  Imbalance:  {self.imbalance_ratio:.1f}:1",
            f"  Timesteps:  {int(self.timesteps.min())}-{int(self.timesteps.max())}",
        ]
        if self.train_mask.sum() > 0:
            lines.append(f"  Train/Val/Test: {self.train_mask.sum()}/{self.val_mask.sum()}/{self.test_mask.sum()}")
        return "\n".join(lines)


def download_elliptic(output_dir: Path) -> Path:
    """
    Download the Elliptic dataset from Kaggle.

    Requires: `kaggle` CLI configured with API credentials.
    Alternative: download manually from https://www.kaggle.com/datasets/ellipticco/elliptic-data-set

    Args:
        output_dir: Directory to save the dataset files.

    Returns:
        Path to the dataset directory.
    """
    import subprocess

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    expected = output_dir / ELLIPTIC_FILES["features"]
    if expected.exists():
        logger.info(f"Dataset already exists at {output_dir}")
        return output_dir

    logger.info("Downloading Elliptic Bitcoin dataset from Kaggle...")
    try:
        subprocess.run(
            [
                "kaggle", "datasets", "download",
                "-d", "ellipticco/elliptic-data-set",
                "-p", str(output_dir),
                "--unzip",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        logger.info(f"✓ Downloaded to {output_dir}")
    except FileNotFoundError:
        logger.error(
            "Kaggle CLI not found. Install with: pip install kaggle\n"
            "Then configure: https://www.kaggle.com/docs/api\n"
            "Or download manually from: https://www.kaggle.com/datasets/ellipticco/elliptic-data-set"
        )
        raise
    except subprocess.CalledProcessError as e:
        logger.error(f"Kaggle download failed: {e.stderr}")
        raise

    return output_dir


def load_elliptic(
    data_dir: Path,
    temporal_split: bool = True,
    train_timesteps: tuple = (1, 34),
    val_timesteps: tuple = (35, 42),
    test_timesteps: tuple = (43, 49),
) -> EllipticDataset:
    """
    Load the Elliptic Bitcoin dataset from CSV files.

    Args:
        data_dir: Directory containing the 3 Elliptic CSV files.
        temporal_split: If True, split by timestep (no data leakage).
        train_timesteps: (start, end) timesteps for training.
        val_timesteps: (start, end) timesteps for validation.
        test_timesteps: (start, end) timesteps for testing.

    Returns:
        EllipticDataset with features, edges, labels, and masks.
    """
    data_dir = Path(data_dir)

    # ── Load features ───────────────────────────────────────────────
    logger.info("Loading node features...")
    feat_path = data_dir / ELLIPTIC_FILES["features"]
    if not feat_path.exists():
        # Try nested directory (Kaggle sometimes nests)
        candidates = list(data_dir.rglob(ELLIPTIC_FILES["features"]))
        if candidates:
            feat_path = candidates[0]
            data_dir = feat_path.parent
        else:
            raise FileNotFoundError(f"Features file not found: {feat_path}")

    df_feat = pd.read_csv(feat_path, header=None)
    # Column 0 = transaction ID, Column 1 = timestep, Columns 2-167 = features
    node_ids = df_feat.iloc[:, 0].values
    timesteps = df_feat.iloc[:, 1].values.astype(np.int64)
    features = df_feat.iloc[:, 2:].values.astype(np.float32)

    logger.info(f"  {len(node_ids):,} nodes, {features.shape[1]} features")

    # ── Build node ID → index mapping ───────────────────────────────
    id_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}

    # ── Load edges ──────────────────────────────────────────────────
    logger.info("Loading edges...")
    edge_path = data_dir / ELLIPTIC_FILES["edges"]
    df_edges = pd.read_csv(edge_path)

    # Map edge endpoints to node indices
    src_col, dst_col = df_edges.columns[0], df_edges.columns[1]
    valid_mask = df_edges[src_col].isin(id_to_idx) & df_edges[dst_col].isin(id_to_idx)
    df_edges = df_edges[valid_mask]

    src_indices = df_edges[src_col].map(id_to_idx).values.astype(np.int64)
    dst_indices = df_edges[dst_col].map(id_to_idx).values.astype(np.int64)
    edge_index = np.stack([src_indices, dst_indices], axis=0)

    logger.info(f"  {edge_index.shape[1]:,} edges")

    # ── Load labels ─────────────────────────────────────────────────
    logger.info("Loading labels...")
    class_path = data_dir / ELLIPTIC_FILES["classes"]
    df_classes = pd.read_csv(class_path)

    id_col, label_col = df_classes.columns[0], df_classes.columns[1]
    labels = np.full(len(node_ids), -1, dtype=np.int64)  # -1 = unknown

    for _, row in df_classes.iterrows():
        nid = row[id_col]
        label_str = str(row[label_col]).strip()
        if nid in id_to_idx and label_str in LABEL_MAP:
            labels[id_to_idx[nid]] = LABEL_MAP[label_str]

    n_licit = (labels == 0).sum()
    n_illicit = (labels == 1).sum()
    n_unknown = (labels == -1).sum()
    logger.info(f"  Licit: {n_licit:,}, Illicit: {n_illicit:,}, Unknown: {n_unknown:,}")

    # ── Temporal split ──────────────────────────────────────────────
    labeled_mask = labels >= 0

    if temporal_split:
        train_mask = labeled_mask & (timesteps >= train_timesteps[0]) & (timesteps <= train_timesteps[1])
        val_mask = labeled_mask & (timesteps >= val_timesteps[0]) & (timesteps <= val_timesteps[1])
        test_mask = labeled_mask & (timesteps >= test_timesteps[0]) & (timesteps <= test_timesteps[1])

        logger.info(
            f"  Temporal split: train={train_mask.sum():,} "
            f"(t{train_timesteps[0]}-{train_timesteps[1]}), "
            f"val={val_mask.sum():,} (t{val_timesteps[0]}-{val_timesteps[1]}), "
            f"test={test_mask.sum():,} (t{test_timesteps[0]}-{test_timesteps[1]})"
        )
    else:
        train_mask = np.zeros(len(node_ids), dtype=bool)
        val_mask = np.zeros(len(node_ids), dtype=bool)
        test_mask = np.zeros(len(node_ids), dtype=bool)

    dataset = EllipticDataset(
        node_features=features,
        edge_index=edge_index,
        labels=labels,
        timesteps=timesteps,
        node_ids=node_ids,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )

    logger.info(f"\n{dataset.summary()}")
    return dataset


def to_pyg(dataset: EllipticDataset):
    """
    Convert EllipticDataset to a PyTorch Geometric Data object.

    Requires: torch, torch_geometric

    Returns:
        torch_geometric.data.Data
    """
    try:
        import torch
        from torch_geometric.data import Data
    except ImportError:
        raise ImportError(
            "PyTorch Geometric required. Install with: pip install -e '.[gnn]'"
        )

    data = Data(
        x=torch.from_numpy(dataset.node_features),
        edge_index=torch.from_numpy(dataset.edge_index),
        y=torch.from_numpy(dataset.labels),
        train_mask=torch.from_numpy(dataset.train_mask),
        val_mask=torch.from_numpy(dataset.val_mask),
        test_mask=torch.from_numpy(dataset.test_mask),
    )

    # Store metadata
    data.timesteps = torch.from_numpy(dataset.timesteps)
    data.node_ids = dataset.node_ids

    return data

# GraphFraud v0.1.0
# Any usage is subject to this software's license.
