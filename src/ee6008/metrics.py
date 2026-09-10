"""Small metric implementations used by validation-only probes."""

from __future__ import annotations

import torch


def classification_metrics(
    logits: torch.Tensor, labels: torch.Tensor, num_classes: int
) -> dict[str, float]:
    if logits.ndim != 2 or labels.ndim != 1 or logits.shape[0] != labels.shape[0]:
        raise ValueError("logits and labels have incompatible shapes")
    predictions = logits.argmax(dim=1)
    top1 = float((predictions == labels).float().mean().item())
    k = min(5, logits.shape[1])
    top5_predictions = logits.topk(k, dim=1).indices
    top5 = float((top5_predictions == labels[:, None]).any(dim=1).float().mean().item())
    recalls = []
    recalls_at_5 = []
    f1s = []
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)
    for truth, prediction in zip(labels.tolist(), predictions.tolist()):
        confusion[truth, prediction] += 1
    for class_index in range(num_classes):
        support = confusion[class_index].sum().item()
        if support == 0:
            continue
        class_mask = labels == class_index
        tp = confusion[class_index, class_index].item()
        fp = confusion[:, class_index].sum().item() - tp
        fn = support - tp
        recalls.append(tp / support)
        recalls_at_5.append(
            float(
                (top5_predictions[class_mask] == class_index)
                .any(dim=1)
                .float()
                .mean()
                .item()
            )
        )
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
    return {
        "top1": top1,
        "top5": top5,
        "mean_class_recall_at_1": float(sum(recalls) / len(recalls))
        if recalls
        else 0.0,
        "mean_class_recall_at_5": float(sum(recalls_at_5) / len(recalls_at_5))
        if recalls_at_5
        else 0.0,
        "macro_f1": float(sum(f1s) / len(f1s)) if f1s else 0.0,
    }


def require_label_safe_split(
    split: str, *, allow_final_test_label_metrics: bool = False
) -> None:
    if split == "final_test" and not allow_final_test_label_metrics:
        raise PermissionError(
            "final_test label metrics are disabled by the Core-A contract"
        )
