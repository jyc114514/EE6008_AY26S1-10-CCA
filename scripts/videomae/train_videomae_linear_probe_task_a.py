from __future__ import annotations

import argparse
import csv
import json
import platform
import random
import time
from pathlib import Path

import torch


DEFAULT_FEATURE_DIR = (
    "/usr1/home/s126mdg33_02/EE6008 AY26S1-10-CCA Project/tmp/"
    "s126mdg33_02/videomae_baseline_20260907/outputs/features"
)
DEFAULT_OUTPUT_DIR = (
    "/usr1/home/s126mdg33_02/EE6008 AY26S1-10-CCA Project/tmp/"
    "s126mdg33_02/videomae_baseline_20260907/outputs/linear_probe"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a linear probe on frozen VideoMAE pooled features."
    )
    parser.add_argument("--feature-dir", default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--variant", choices=["dynamic", "static"], default="dynamic")
    parser.add_argument("--seed", type=int, default=6008)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--cpu", action="store_true", help="Force CPU training.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_feature_file(path: Path) -> dict:
    data = torch.load(path, map_location="cpu", weights_only=False)
    required = {"features", "labels", "episode_indices", "metadata"}
    missing = sorted(required - set(data))
    if missing:
        raise ValueError(f"{path} is missing keys: {missing}")
    return data


def validate_static_pair(feature_dir: Path, split: str, static_data: dict) -> None:
    dynamic_data = load_feature_file(feature_dir / f"videomae_{split}_pooled.pt")
    for key in ("episode_indices", "labels"):
        if not torch.equal(static_data[key], dynamic_data[key]):
            raise ValueError(f"{split}: static and dynamic {key} differ")
    for key in (
        "model_name", "manifest", "g1_root", "input_frames",
        "sample_fps", "window_seconds", "encoder_frozen",
    ):
        if static_data["metadata"].get(key) != dynamic_data["metadata"].get(key):
            raise ValueError(f"{split}: static and dynamic {key} differ")


def compact_labels(
    train_labels: torch.Tensor,
    val_labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[int, int]]:
    raw_labels = sorted(set(train_labels.tolist()) | set(val_labels.tolist()))
    mapping = {int(raw_label): idx for idx, raw_label in enumerate(raw_labels)}
    train_compact = torch.tensor(
        [mapping[int(label)] for label in train_labels.tolist()],
        dtype=torch.long,
    )
    val_compact = torch.tensor(
        [mapping[int(label)] for label in val_labels.tolist()],
        dtype=torch.long,
    )
    return train_compact, val_compact, mapping


def macro_f1(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> float:
    scores = []
    for cls in range(num_classes):
        pred_pos = pred == cls
        true_pos = target == cls
        tp = torch.logical_and(pred_pos, true_pos).sum().item()
        fp = torch.logical_and(pred_pos, ~true_pos).sum().item()
        fn = torch.logical_and(~pred_pos, true_pos).sum().item()
        if tp == 0 and fp == 0 and fn == 0:
            continue
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        score = (
            2.0 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        scores.append(score)
    return float(sum(scores) / len(scores)) if scores else 0.0


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
    num_classes: int,
) -> tuple[dict[str, float], torch.Tensor, torch.Tensor]:
    model.eval()
    logits_chunks = []
    for start in range(0, features.shape[0], batch_size):
        end = min(start + batch_size, features.shape[0])
        logits_chunks.append(model(features[start:end].to(device)).cpu())
    logits = torch.cat(logits_chunks, dim=0)
    predictions = logits.argmax(dim=1)
    accuracy = (predictions == labels).float().mean().item()
    metrics = {
        "top1": float(accuracy),
        "micro_f1": float(accuracy),
        "macro_f1": macro_f1(predictions, labels, num_classes),
        "nll": float(torch.nn.functional.cross_entropy(logits, labels).item()),
    }
    return metrics, logits, predictions


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )

    feature_dir = Path(args.feature_dir)
    output_dir = Path(
        args.output_dir
        or (DEFAULT_OUTPUT_DIR + "_static" if args.variant == "static" else DEFAULT_OUTPUT_DIR)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = "_static" if args.variant == "static" else ""
    train_data = load_feature_file(feature_dir / f"videomae_train{suffix}_pooled.pt")
    val_data = load_feature_file(feature_dir / f"videomae_validation{suffix}_pooled.pt")
    for split, data in (("train", train_data), ("validation", val_data)):
        if data["metadata"].get("input_variant", "dynamic") != args.variant:
            raise ValueError(f"{split}: feature variant does not match --variant")
        if data["metadata"].get("final_test_used") is not False:
            raise ValueError(f"{split}: final-test flag is not false")
        if args.variant == "static":
            validate_static_pair(feature_dir, split, data)
    x_train = train_data["features"].float()
    y_train_raw = train_data["labels"].long()
    x_val = val_data["features"].float()
    y_val_raw = val_data["labels"].long()
    y_train, y_val, label_mapping = compact_labels(y_train_raw, y_val_raw)

    feature_dim = int(x_train.shape[1])
    num_classes = len(label_mapping)
    model = torch.nn.Linear(feature_dim, num_classes).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )

    best_state = None
    best_epoch = -1
    best_macro = -1.0
    stale_epochs = 0
    history = []
    start_time = time.perf_counter()

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        permutation = torch.randperm(x_train.shape[0])
        total_loss = 0.0
        total_seen = 0
        for start in range(0, x_train.shape[0], int(args.batch_size)):
            idx = permutation[start : start + int(args.batch_size)]
            batch_x = x_train[idx].to(device)
            batch_y = y_train[idx].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = torch.nn.functional.cross_entropy(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * int(batch_x.shape[0])
            total_seen += int(batch_x.shape[0])

        val_metrics, _, _ = evaluate(
            model,
            x_val,
            y_val,
            batch_size=int(args.batch_size),
            device=device,
            num_classes=num_classes,
        )
        train_loss = total_loss / max(1, total_seen)
        record = {"epoch": epoch, "train_loss": train_loss, **val_metrics}
        history.append(record)
        print(json.dumps(record), flush=True)

        if val_metrics["macro_f1"] > best_macro:
            best_macro = val_metrics["macro_f1"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= int(args.patience):
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    val_metrics, val_logits, val_predictions = evaluate(
        model,
        x_val,
        y_val,
        batch_size=int(args.batch_size),
        device=device,
        num_classes=num_classes,
    )

    metadata = {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "device": str(device),
        "seed": int(args.seed),
        "input_variant": args.variant,
        "epochs_requested": int(args.epochs),
        "best_epoch": int(best_epoch),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "patience": int(args.patience),
        "feature_dim": feature_dim,
        "num_classes": num_classes,
        "raw_label_count": len(label_mapping),
        "raw_label_min": int(min(label_mapping)),
        "raw_label_max": int(max(label_mapping)),
        "raw_to_compact_label": {str(k): v for k, v in label_mapping.items()},
        "train_examples": int(x_train.shape[0]),
        "validation_examples": int(x_val.shape[0]),
        "train_feature_metadata": train_data["metadata"],
        "validation_feature_metadata": val_data["metadata"],
        "final_test_used": False,
        "elapsed_seconds": time.perf_counter() - start_time,
    }
    result = {"metrics": val_metrics, "metadata": metadata, "history": history}

    (output_dir / f"metrics_seed{args.seed}.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    torch.save(model.state_dict(), output_dir / f"linear_probe_seed{args.seed}.pt")
    torch.save(val_logits, output_dir / f"validation_logits_seed{args.seed}.pt")

    with (output_dir / f"validation_predictions_seed{args.seed}.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        inverse_label_mapping = {v: k for k, v in label_mapping.items()}
        writer.writerow(
            [
                "episode_index",
                "raw_label",
                "compact_label",
                "compact_prediction",
                "raw_prediction",
            ]
        )
        for episode_index, raw_label, compact_label, prediction in zip(
            val_data["episode_indices"].tolist(),
            y_val_raw.tolist(),
            y_val.tolist(),
            val_predictions.tolist(),
        ):
            writer.writerow(
                [
                    episode_index,
                    raw_label,
                    compact_label,
                    prediction,
                    inverse_label_mapping[int(prediction)],
                ]
            )

    print("=== final validation metrics ===")
    print(json.dumps(result["metrics"], indent=2))
    print(f"Saved outputs under: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
