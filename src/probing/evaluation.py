"""Probe evaluation helpers."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from .metrics import (
    LABELS,
    baseline_predictions,
    classification_metrics,
    label_counts,
    labels_to_indices,
)
from .split import DomainArtifact


@dataclass
class TrainedLinearProbe:
    """A trained probe paired with the standardization statistics it expects."""

    model: nn.Module
    mean: torch.Tensor
    std: torch.Tensor


def compute_standardization_stats(
    train_x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute train-set statistics used to standardize embeddings."""
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
    return mean, std


def apply_standardization(
    x: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """Apply a precomputed z-score normalization transform."""
    return (x - mean) / std


def standardize_embeddings(
    train_x: torch.Tensor, test_x: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Z-score normalize embeddings using training set statistics.

    Normalization prevents larger embedding dimensions from dominating
    the linear probe's weights and ensures consistency across different
    models. We use train statistics to avoid data leakage.

    Args:
        train_x: Training embedding tensor (count, dim).
        test_x: Test embedding tensor (count, dim).

    Returns:
        A tuple of (standardized_train_x, standardized_test_x).
    """
    mean, std = compute_standardization_stats(train_x)
    return apply_standardization(train_x, mean, std), apply_standardization(
        test_x, mean, std
    )


def build_linear_probe(input_dim: int) -> nn.Linear:
    """Construct an untrained linear classifier for probe persistence/loading."""
    return nn.Linear(input_dim, len(LABELS))


def train_linear_probe(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    *,
    seed: int,
    learning_rate: float,
    num_steps: int,
    weight_decay: float,
) -> nn.Module:
    """Fit a single linear layer to predict labels from frozen embeddings.

    Uses AdamW optimizer and CrossEntropy loss. This is a standard linear
    probing setup used to measure how much label-relevant information is
    linearly accessible in the encoder's representations.

    Args:
        train_x: Training embeddings.
        train_y: Training labels (integer indices).
        seed: Random seed for model initialization.
        learning_rate: Optimizer learning rate.
        num_steps: Number of training steps (full batch).
        weight_decay: Weight decay (L2 regularization) coefficient.

    Returns:
        The trained and evaluated PyTorch model.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = build_linear_probe(train_x.shape[1])
    loss_fn: nn.Module = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    train_x = train_x.float()
    train_y = train_y.long()

    model.train()
    for _ in range(num_steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model(train_x)
        loss = loss_fn(logits, train_y)
        loss.backward()
        optimizer.step()
    return model.eval()


def predict_labels(model: nn.Module, x: torch.Tensor) -> np.ndarray:
    """Generate argmax predictions for a given embedding tensor.

    Args:
        model: The trained linear probe.
        x: Input embeddings.

    Returns:
        A NumPy array of predicted class indices.
    """
    with torch.inference_mode():
        logits = model(x.float())
        return logits.argmax(dim=1).cpu().numpy()


def _prepare_split_inputs(
    train_artifact: DomainArtifact,
    test_artifact: DomainArtifact,
) -> tuple[np.ndarray, np.ndarray, torch.Tensor, torch.Tensor]:
    """Prepare labels and standardized embeddings for one evaluation split."""
    train_labels = labels_to_indices(record.label for record in train_artifact.records)
    test_labels = labels_to_indices(record.label for record in test_artifact.records)
    train_x, test_x = standardize_embeddings(
        train_artifact.embeddings, test_artifact.embeddings
    )
    return train_labels, test_labels, train_x, test_x


def fit_linear_probe(
    train_artifact: DomainArtifact,
    *,
    seed: int,
    learning_rate: float,
    num_steps: int,
    weight_decay: float,
) -> TrainedLinearProbe:
    """Fit a probe and capture the normalization statistics needed at inference."""
    train_labels = labels_to_indices(record.label for record in train_artifact.records)
    mean, std = compute_standardization_stats(train_artifact.embeddings)
    standardized_train_x = apply_standardization(train_artifact.embeddings, mean, std)
    model = train_linear_probe(
        standardized_train_x,
        torch.from_numpy(train_labels),
        seed=seed,
        learning_rate=learning_rate,
        num_steps=num_steps,
        weight_decay=weight_decay,
    )
    return TrainedLinearProbe(model=model, mean=mean, std=std)


def _summarize_split(
    *,
    train_artifact: DomainArtifact,
    test_artifact: DomainArtifact,
    train_labels: np.ndarray,
    test_labels: np.ndarray,
    predictions: np.ndarray,
    baseline: np.ndarray,
) -> dict[str, Any]:
    """Assemble the reporting payload for one evaluated split."""
    return {
        "probe": classification_metrics(test_labels, predictions),
        "baseline": classification_metrics(test_labels, baseline),
        "train_size": len(train_labels),
        "test_size": len(test_labels),
        "train_papers": len({record.paper_id for record in train_artifact.records}),
        "test_papers": len({record.paper_id for record in test_artifact.records}),
        "train_label_counts": label_counts(
            record.label for record in train_artifact.records
        ),
        "test_label_counts": label_counts(
            record.label for record in test_artifact.records
        ),
    }


def evaluate_split(
    train_artifact: DomainArtifact,
    test_artifact: DomainArtifact,
    *,
    seed: int,
    learning_rate: float,
    num_steps: int,
    weight_decay: float,
) -> dict[str, Any]:
    """Train a probe on the training artifact and evaluate it on the test set.

    Also computes a majority-class baseline for comparison.

    Args:
        train_artifact: Training records and embeddings.
        test_artifact: Evaluation records and embeddings.
        seed: Random seed for the probe.
        learning_rate: LR for AdamW.
        num_steps: Training steps.
        weight_decay: L2 regularization strength.

    Returns:
        A dictionary containing probe metrics, baseline metrics, and
        metadata about the split (sizes, label counts, paper counts).
    """
    trained_probe = fit_linear_probe(
        train_artifact,
        seed=seed,
        learning_rate=learning_rate,
        num_steps=num_steps,
        weight_decay=weight_decay,
    )
    return evaluate_trained_probe(
        train_artifact,
        test_artifact,
        trained_probe=trained_probe,
    )


def evaluate_trained_probe(
    train_artifact: DomainArtifact,
    test_artifact: DomainArtifact,
    *,
    trained_probe: TrainedLinearProbe,
) -> dict[str, Any]:
    """Evaluate a previously trained probe on a split."""
    train_labels = labels_to_indices(record.label for record in train_artifact.records)
    test_labels = labels_to_indices(record.label for record in test_artifact.records)
    test_x = apply_standardization(
        test_artifact.embeddings,
        trained_probe.mean,
        trained_probe.std,
    )
    predictions = predict_labels(trained_probe.model, test_x)
    baseline = baseline_predictions(train_labels, len(test_labels))
    return _summarize_split(
        train_artifact=train_artifact,
        test_artifact=test_artifact,
        train_labels=train_labels,
        test_labels=test_labels,
        predictions=predictions,
        baseline=baseline,
    )
