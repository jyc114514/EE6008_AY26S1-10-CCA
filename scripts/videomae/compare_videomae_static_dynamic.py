from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import torch

from audit_videomae_task_a import calculate_metrics, require



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only matched VideoMAE static/dynamic comparison."
    )
    parser.add_argument("--run-dir", type=Path, required=True, help="Root directory containing the VideoMAE run outputs.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[6008, 7, 42])
    return parser.parse_args()


def load_feature(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=True)


def load_seed(result_dir: Path, seed: int, labels: torch.Tensor) -> tuple[dict, torch.Tensor]:
    record = json.loads(
        (result_dir / f"metrics_seed{seed}.json").read_text(encoding="utf-8")
    )
    logits = torch.load(
        result_dir / f"validation_logits_seed{seed}.pt",
        map_location="cpu",
        weights_only=True,
    )
    require(logits.shape == (len(labels), 120), f"Seed {seed}: logits shape differs")
    require(torch.isfinite(logits).all().item(), f"Seed {seed}: non-finite logits")
    metrics = calculate_metrics(logits, labels, torch)
    for name in ("top1", "macro_f1", "nll"):
        require(
            math.isclose(metrics[name], record["metrics"][name], abs_tol=1e-6),
            f"Seed {seed}: saved {name} differs from logits",
        )
    require(record["metadata"].get("final_test_used") is False, "Final-test flag differs")
    return {"metrics": metrics, "metadata": record["metadata"]}, logits


def main() -> int:
    args = parse_args()
    feature_dir = args.run_dir / "outputs/features"
    dynamic = load_feature(feature_dir / "videomae_validation_pooled.pt")
    static = load_feature(feature_dir / "videomae_validation_static_pooled.pt")
    require(
        torch.equal(dynamic["episode_indices"], static["episode_indices"]),
        "Static and dynamic episode order differs",
    )
    require(torch.equal(dynamic["labels"], static["labels"]), "Labels differ")
    require(len(dynamic["labels"]) == 617, "Expected 617 validation episodes")
    require(static["metadata"].get("input_variant") == "static", "Static metadata differs")

    comparisons = []
    metric_names = ("top1", "top5", "macro_f1", "nll", "brier", "ece_15")
    for seed in args.seeds:
        # The label mapping is stored with each trained head; compare it before scoring.
        dynamic_record = json.loads(
            (args.run_dir / "outputs/linear_probe" / f"metrics_seed{seed}.json").read_text(
                encoding="utf-8"
            )
        )
        static_record = json.loads(
            (args.run_dir / "outputs/linear_probe_static" / f"metrics_seed{seed}.json").read_text(
                encoding="utf-8"
            )
        )
        dynamic_mapping = dynamic_record["metadata"]["raw_to_compact_label"]
        static_mapping = static_record["metadata"]["raw_to_compact_label"]
        require(dynamic_mapping == static_mapping, f"Seed {seed}: class mapping differs")
        require(static_record["metadata"].get("input_variant") == "static", "Head variant differs")
        labels = torch.tensor(
            [dynamic_mapping[str(int(raw))] for raw in dynamic["labels"].tolist()],
            dtype=torch.long,
        )
        dynamic_result, dynamic_logits = load_seed(
            args.run_dir / "outputs/linear_probe", seed, labels
        )
        static_result, static_logits = load_seed(
            args.run_dir / "outputs/linear_probe_static", seed, labels
        )
        dynamic_correct = dynamic_logits.argmax(dim=1) == labels
        static_correct = static_logits.argmax(dim=1) == labels
        comparisons.append({
            "seed": seed,
            "dynamic": dynamic_result["metrics"],
            "static": static_result["metrics"],
            "static_minus_dynamic": {
                name: static_result["metrics"][name] - dynamic_result["metrics"][name]
                for name in metric_names
            },
            "dynamic_only_correct": int((dynamic_correct & ~static_correct).sum().item()),
            "static_only_correct": int((static_correct & ~dynamic_correct).sum().item()),
        })

    aggregate = {}
    for name in metric_names:
        values = [item["static_minus_dynamic"][name] for item in comparisons]
        aggregate[name] = {
            "mean": statistics.mean(values),
            "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
        }
    print(json.dumps({
        "status": "pass",
        "comparison": "static_minus_dynamic",
        "episodes": 617,
        "same_episode_order_and_raw_labels": True,
        "final_test_access": 0,
        "per_seed": comparisons,
        "aggregate_delta": aggregate,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
