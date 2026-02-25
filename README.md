# GraphFraud

**Graph Neural Network Fraud Detection on Financial Transaction Networks**

Detects illicit transactions in Bitcoin/financial networks using GATv2Conv graph attention networks with class-imbalance-aware training. Inspired by graph attention networks build for genomics.

---

## Overview

| Component | Description |
|-----------|-------------|
| **Graph Construction** | Build transaction graphs from tabular edge/node data (NetworkX → PyG) |
| **GNN Models** | GATv2Conv, GraphSAGE, GCN — configurable via YAML |
| **Baselines** | XGBoost, Logistic Regression, Random Forest (no graph structure) |
| **Resampling** | Hybrid under/oversampling for extreme class imbalance (~2% illicit) |
| **HP Search** | Optuna with pruning, StratifiedKFold CV |
| **Explainability** | GNNExplainer for transaction-level attribution |
| **CLI** | Click-based pipeline: `graphfraud train`, `graphfraud evaluate`, `graphfraud explain` |

## Dataset

**Elliptic Bitcoin Dataset** — 203k transactions, 234k edges, 49 timesteps.

| Class | Count | % |
|-------|-------|---|
| Licit | 42,019 | 20.7% |
| Illicit | 4,545 | 2.2% |
| Unknown | 157,205 | 77.1% |

> 166 node features (94 local + 72 aggregated neighbor features). Temporal graph with 49 timesteps.

Download: [Kaggle](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)

## Quick Start

```bash
# Install (CPU-only baselines)
pip install -e .

# Install with GNN support
pip install -e ".[gnn]"

# Install everything
pip install -e ".[all]"

# Download data
graphfraud download --output data/

# Train GATv2 model
graphfraud train --config configs/gatv2.yaml

# Train XGBoost baseline
graphfraud train --config configs/xgboost_baseline.yaml

# Evaluate
graphfraud evaluate --model trained_models/gatv2_best.pt --data data/

# Explain predictions
graphfraud explain --model trained_models/gatv2_best.pt --node-id 12345
```

## Project Structure

```
graphfraud/
├── pyproject.toml
├── README.md
├── configs/
│   ├── gatv2.yaml              # GATv2Conv config
│   ├── graphsage.yaml          # GraphSAGE config
│   └── xgboost_baseline.yaml   # XGBoost (no graph) baseline
├── graphfraud/
│   ├── __init__.py
│   ├── cli.py                  # Click CLI
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py          # Elliptic dataset loader + PyG conversion
│   │   └── resampling.py       # Hybrid resampling for class imbalance
│   ├── models/
│   │   ├── __init__.py
│   │   ├── gatv2.py            # GATv2Conv classifier
│   │   ├── graphsage.py        # GraphSAGE classifier
│   │   ├── gcn.py              # GCN baseline
│   │   └── xgboost_baseline.py # Non-graph XGBoost baseline
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py          # Training loop (GNN)
│   │   ├── hp_search.py        # Optuna HP optimization
│   │   └── evaluation.py       # Metrics, confusion matrix, reports
│   └── explain/
│       ├── __init__.py
│       └── gnn_explainer.py    # GNNExplainer wrapper
├── notebooks/
│   └── 01_eda_and_baselines.ipynb
├── tests/
│   ├── test_dataset.py
│   ├── test_models.py
│   └── test_resampling.py
├── trained_models/             # .pt / .pkl artifacts (gitignored)
├── data/                       # Raw data (gitignored)
└── results/                    # Evaluation outputs (gitignored)
```

## Methods

### Graph Neural Network Architecture

```
Input (166 features/node)
    │
    ├── GATv2Conv (8 heads, 32 dim) + ELU + Dropout
    ├── GATv2Conv (8 heads, 32 dim) + ELU + Dropout
    ├── GATv2Conv (1 head, 64 dim) + ELU
    │
    ├── Global pooling (optional for graph-level tasks)
    │
    └── MLP Head → 2 classes (licit / illicit)
```

### Class Imbalance Strategy
- **Hybrid resampling:** Undersample licit to cap, oversample illicit to median
- **Focal loss:** Down-weight easy examples, focus on hard-to-classify transactions
- **Temporal train/val/test split:** Timesteps 1-34 train, 35-42 val, 43-49 test (no data leakage)

### Baselines
| Model | Features | Graph? |
|-------|----------|--------|
| Logistic Regression | 166 node features | No |
| XGBoost | 166 node features | No |
| GCN | 166 + graph topology | Yes |
| GraphSAGE | 166 + sampled neighbors | Yes |
| **GATv2Conv** | 166 + attention-weighted neighbors | Yes |

## Key Design Decisions

1. **Temporal split** (not random) — prevents future data leaking into training
2. **Semi-supervised** — use the 157k unlabeled nodes during message passing but only compute loss on labeled nodes
3. **Inductive evaluation** — model must generalize to unseen timesteps
4. **Attention interpretability** — GATv2 attention weights show *which* neighbor transactions influenced the classification

## License

MIT
