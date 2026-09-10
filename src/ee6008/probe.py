"""Bounded pooled-feature linear probe for smoke and pilot validation."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ee6008.metrics import classification_metrics, require_label_safe_split


@dataclass
class ProbeResult:
    metrics: dict[str, float]
    train_loss: float
    epochs: int


def train_linear_probe(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    validation_features: torch.Tensor,
    validation_labels: torch.Tensor,
    *,
    num_classes: int,
    seed: int = 20260825,
    epochs: int = 80,
    learning_rate: float = 0.05,
    weight_decay: float = 1e-4,
) -> ProbeResult:
    if train_features.ndim != 2 or validation_features.ndim != 2:
        raise ValueError("features must be two-dimensional")
    if train_features.shape[1] != validation_features.shape[1]:
        raise ValueError("train and validation feature dimensions differ")
    torch.manual_seed(seed)
    model = torch.nn.Linear(train_features.shape[1], num_classes)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    loss_fn = torch.nn.CrossEntropyLoss()
    last_loss = 0.0
    for _ in range(epochs):
        model.train()
        logits = model(train_features)
        loss = loss_fn(logits, train_labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.item())
    model.eval()
    with torch.inference_mode():
        validation_logits = model(validation_features)
    metrics = classification_metrics(validation_logits, validation_labels, num_classes)
    return ProbeResult(metrics=metrics, train_loss=last_loss, epochs=epochs)


def assert_no_final_test_label_metrics(split: str) -> None:
    require_label_safe_split(split, allow_final_test_label_metrics=False)
