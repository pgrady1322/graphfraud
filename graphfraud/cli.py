"""
GraphFraud v0.1.0

cli.py — Click-based command-line interface for GraphFraud.

Author: Patrick Grady
Anthropic Claude Opus 4.6 used for code formatting and cleanup assistance.
License: MIT License - See LICENSE
"""

import click
import yaml
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("graphfraud")


@click.group()
@click.version_option(version="0.1.0")
def main():
    """GraphFraud — Graph Neural Network Fraud Detection."""
    pass


@main.command()
@click.option("--output", "-o", type=click.Path(), default="data/", help="Output directory")
def download(output):
    """Download the Elliptic Bitcoin dataset."""
    from graphfraud.data.dataset import download_elliptic

    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)
    download_elliptic(out_dir)
    click.echo(f"✓ Dataset downloaded to {out_dir}")


@main.command()
@click.option("--config", "-c", type=click.Path(exists=True), required=True, help="YAML config")
def train(config):
    """Train a fraud detection model."""
    with open(config) as f:
        cfg = yaml.safe_load(f)

    model_type = cfg.get("model", {}).get("type", "gatv2")
    click.echo(f"Training {model_type} model...")

    if model_type in ("gatv2", "graphsage", "gcn"):
        from graphfraud.training.trainer import train_gnn

        train_gnn(cfg)
    elif model_type == "xgboost":
        from graphfraud.training.trainer import train_xgboost

        train_xgboost(cfg)
    elif model_type in ("logistic_regression", "random_forest"):
        from graphfraud.training.trainer import train_sklearn_baseline

        train_sklearn_baseline(cfg)
    else:
        raise click.BadParameter(f"Unknown model type: {model_type}")


@main.command()
@click.option("--model", "-m", type=click.Path(exists=True), required=True, help="Model path")
@click.option("--data", "-d", type=click.Path(exists=True), default="data/", help="Data directory")
@click.option("--output", "-o", type=click.Path(), default="results/", help="Output directory")
def evaluate(model, data, output):
    """Evaluate a trained model on test data."""
    from graphfraud.training.evaluation import evaluate_model

    results = evaluate_model(model_path=model, data_dir=data, output_dir=output)
    f1 = results.get('f1_illicit', 0.0)
    auc = results.get('auc_roc', 0.0)
    click.echo(f"✓ Evaluation complete — F1={f1:.4f}, AUC-ROC={auc:.4f}")


@main.command()
@click.option("--model", "-m", type=click.Path(exists=True), required=True, help="Model path")
@click.option("--node-id", "-n", type=int, required=True, help="Node ID to explain")
@click.option("--data", "-d", type=click.Path(exists=True), default="data/", help="Data directory")
def explain(model, node_id, data):
    """Explain a prediction using GNNExplainer."""
    from graphfraud.explain.gnn_explainer import explain_node

    explanation = explain_node(model_path=model, node_id=node_id, data_dir=data)
    click.echo(f"✓ Explanation generated for node {node_id}")
    click.echo(f"  Top features: {explanation['top_features']}")
    click.echo(f"  Key neighbors: {explanation['key_neighbors']}")


if __name__ == "__main__":
    main()

# GraphFraud v0.1.0
# Any usage is subject to this software's license.
