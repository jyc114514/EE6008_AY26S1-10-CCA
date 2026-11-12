"""Audit strict Core A features and train three isolated CPU linear probes."""

import argparse
import json
from pathlib import Path
import random
import statistics

from evaluate_videomae_core_b import digest, require


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    project, features, output = [p.resolve() for p in (args.project_root, args.feature_root, args.output_root)]
    scratch = (project / "tmp/s126mdg33_02").resolve()
    require(features.is_relative_to(scratch) and output.is_relative_to(scratch) and output != scratch
            and not output.exists(), "Fresh personal output required")
    import numpy as np
    import torch
    import pyarrow.parquet as pq
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    manifest_path = features / "FEATURE_MANIFEST.json"
    report = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(report["status"] == "PASS_CORE_A_STRICT_FULL_FEATURES" and report["strict_load"]
            and report["final_test_access"] == 0, "Strict feature gate failed")
    manifest = Path(report["manifest"])
    require(digest(manifest) == report["manifest_sha256"], "Classification manifest changed")
    rows = pq.read_table(manifest, filters=[("split", "in", ["train", "validation"])]).to_pylist()
    rows = [r for r in rows if all(key not in r or r[key] == "primary" for key in
                                  ("classification_inclusion", "v3_benchmark_inclusion"))
            and ("eligible_4s" not in r or r["eligible_4s"] is True)]
    raw_labels = sorted({int(r["contiguous_label_index"]) for r in rows if r["split"] == "train"})
    mapping = {label: index for index, label in enumerate(raw_labels)}
    require(len(mapping) == 120, "Expected 120 train-supported classes")
    data = {}
    for split, expected_count in (("train", 2757), ("validation", 617)):
        path = features / f"videomae_{split}_pooled.pt"
        require(digest(path) == report["feature_files"][split]["sha256"], "Feature file hash mismatch")
        bundle = torch.load(path, map_location="cpu", weights_only=True)
        require(bundle["metadata"]["strict_load"] and bundle["metadata"]["final_test_used"] is False,
                "Invalid feature provenance")
        selected = sorted([r for r in rows if r["split"] == split], key=lambda r: int(r["episode_index"]))
        require(len(selected) == expected_count, "Classification population mismatch")
        ids = torch.tensor([int(r["episode_index"]) for r in selected], dtype=torch.long)
        labels = torch.tensor([mapping[int(r["contiguous_label_index"])] for r in selected], dtype=torch.long)
        require(torch.equal(bundle["episode_indices"], ids) and torch.equal(bundle["labels"], labels), "Feature ID/label join mismatch")
        x = bundle["features"]
        require(x.dtype == torch.float32 and tuple(x.shape) == (expected_count, 384)
                and bool(torch.isfinite(x).all()), "Invalid feature array")
        data[split] = (x, labels, ids)
    require(not set(data["train"][2].tolist()) & set(data["validation"][2].tolist()), "Episode leakage")
    x_train, y_train, _ = data["train"]
    x_val, y_val, _ = data["validation"]

    @torch.inference_mode()
    def score(model, x, labels):
        model.eval()
        logits = torch.cat([model(x[start:start + 256]) for start in range(0, len(x), 256)])
        pred = logits.argmax(dim=1)
        confusion = torch.bincount(labels * 120 + pred, minlength=120 * 120).reshape(120, 120).double()
        denominator = confusion.sum(0) + confusion.sum(1)
        supported = denominator > 0
        f1 = torch.where(supported, 2 * confusion.diag() / denominator.clamp_min(1), 0)
        probs = logits.softmax(dim=1)
        confidence = probs.max(dim=1).values
        correct = pred == labels
        ece = 0.0
        for bin_index in range(15):
            mask = (confidence > bin_index / 15) & (confidence <= (bin_index + 1) / 15)
            if bool(mask.any()):
                ece += float(mask.float().mean() * (correct[mask].float().mean() - confidence[mask].mean()).abs())
        metrics = {"top1": float(correct.float().mean()),
                   "top5": float((logits.topk(5, dim=1).indices == labels[:, None]).any(dim=1).float().mean()),
                   "macro_f1": float(f1[supported].mean()),
                   "nll": float(torch.nn.functional.cross_entropy(logits, labels)),
                   "brier": float((probs - torch.nn.functional.one_hot(labels, 120)).square().sum(dim=1).mean()),
                   "ece_15": ece}
        return metrics, logits, pred

    output.mkdir(parents=True, exist_ok=False)
    config = {"seeds": [6008, 7, 42], "epochs_max": 100, "batch_size": 256, "optimizer": "AdamW",
              "lr": 1e-3, "weight_decay": 1e-4, "betas": [0.9, 0.999], "eps": 1e-8,
              "patience": 15, "selection": "strictly improving validation Macro-F1; earliest maximum",
              "features": "raw frozen token-mean features; no fitted feature normalization",
              "feature_manifest_sha256": digest(manifest_path), "classification_manifest_sha256": digest(manifest),
              "feature_sha256": {s: report["feature_files"][s]["sha256"] for s in data},
              "script_sha256": digest(Path(__file__)), "label_mapping_train_only": mapping,
              "device": "cpu", "torch": torch.__version__, "numpy": np.__version__,
              "final_test_access": 0, "comparison_scope": "Frozen-backbone linear probe; not matched LoRA fine-tuning"}
    write(output / "CONFIG.json", config)
    results = []
    for seed in config["seeds"]:
        root = output / f"seed{seed}"
        root.mkdir()
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        model = torch.nn.Linear(384, 120)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        best_score, best_epoch, stale, best_state = -1.0, -1, 0, None
        history = []
        for epoch in range(1, 101):
            model.train()
            order = torch.randperm(len(x_train))
            loss_sum = 0.0
            for start in range(0, len(order), 256):
                index = order[start:start + 256]
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.cross_entropy(model(x_train[index]), y_train[index])
                require(bool(torch.isfinite(loss)), "Nonfinite loss")
                loss.backward()
                optimizer.step()
                loss_sum += loss.item() * len(index)
            metrics, _, _ = score(model, x_val, y_val)
            history.append({"epoch": epoch, "train_loss": loss_sum / len(x_train), **metrics})
            if metrics["macro_f1"] > best_score:
                best_score, best_epoch, stale = metrics["macro_f1"], epoch, 0
                best_state = {k: value.detach().clone() for k, value in model.state_dict().items()}
            else:
                stale += 1
                if stale >= 15:
                    break
        require(best_state is not None, "No selected checkpoint")
        model.load_state_dict(best_state, strict=True)
        checkpoint = root / "checkpoint_best.pt"
        torch.save({"model": best_state, "seed": seed, "epoch": best_epoch,
                    "config_sha256": digest(output / "CONFIG.json")}, checkpoint)
        check = torch.nn.Linear(384, 120)
        check.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["model"], strict=True)
        check.eval()
        record = {"seed": seed, "best_epoch": best_epoch, "epochs_run": len(history), "metrics": {},
                  "checkpoint_sha256": digest(checkpoint), "prediction_hashes": {}, "final_test_access": 0}
        for split, (x, labels, ids) in data.items():
            metrics, logits, pred = score(model, x, labels)
            with torch.inference_mode():
                require(torch.equal(check(x[:64]), model(x[:64])), "Checkpoint reload mismatch")
            path = root / f"predictions_{split}.npz"
            np.savez_compressed(path, episode_indices=ids.numpy(), labels=labels.numpy(),
                                logits=logits.numpy(), predictions=pred.numpy())
            record["metrics"][split] = metrics
            record["prediction_hashes"][split] = digest(path)
        write(root / "HISTORY.json", history)
        write(root / "METRICS.json", record)
        results.append(record)
        write(output / "RUN_STATUS.json", {"status": "running", "completed_seeds": len(results), "total_seeds": 3})
        print(json.dumps({"completed_seeds": len(results), "seed": seed, "best_epoch": best_epoch,
                          "validation": record["metrics"]["validation"]}), flush=True)
    summary = {metric: {"mean": statistics.mean(r["metrics"]["validation"][metric] for r in results),
                        "sample_sd": statistics.stdev(r["metrics"]["validation"][metric] for r in results)}
               for metric in results[0]["metrics"]["validation"]}
    final = {"status": "PASS_CORE_A_STRICT_THREE_SEED_PROBES", "validation_summary": summary,
             "results": results, "gpu_used": False, "final_test_access": 0,
             "selection_uses_validation": True, "final_test_evaluated": False}
    write(output / "SUMMARY.json", final)
    write(output / "RUN_STATUS.json", {"status": final["status"], "completed_seeds": 3, "total_seeds": 3,
                                      "summary": str(output / "SUMMARY.json"), "final_test_access": 0})
    print(json.dumps({"status": final["status"], "validation_summary": summary,
                      "summary": str(output / "SUMMARY.json"), "gpu_used": False, "final_test_access": 0}, indent=2))


if __name__ == "__main__":
    main()
