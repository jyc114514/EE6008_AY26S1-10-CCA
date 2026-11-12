"""Offline Core A population audit and four-video strict-load CPU smoke."""

import argparse
import json
from pathlib import Path
import os
import time

from evaluate_videomae_core_b import digest, require


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--phase", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--preflight-report", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    project, output = args.project_root.resolve(), args.output_root.resolve()
    scratch = (project / "tmp/s126mdg33_02").resolve()
    require(output.is_relative_to(scratch) and output != scratch and not output.exists(), "Fresh personal output required")
    import cv2
    import numpy as np
    import pyarrow.parquet as pq
    import torch
    from safetensors.torch import load_file
    from transformers import VideoMAEConfig, VideoMAEForVideoClassification, VideoMAEImageProcessor
    torch.set_num_threads(2)
    device = args.device
    if device == "cuda":
        physical = os.environ.get("GPU_WATCHER_PHYSICAL_GPU")
        require(physical in {"0", "1", "2", "3"} and os.environ.get("CUDA_VISIBLE_DEVICES") == physical,
                "Shared GPU watcher assignment required")
        require(torch.cuda.is_available(), "CUDA unavailable")
        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats()
    require(args.phase != "full" or (device == "cuda" and args.preflight_report is not None), "Full extraction requires GPU lease and preflight")
    split_root = project / "repo/ee6008-humanoid-vjepa/manifests/g1_6b6b599def75e3558b2736062e79bdbeec6cd224/splits/core_a_protocol_v3_duplicate_safe"
    manifest = split_root / "train_validation_label_manifest.parquet"
    schema = pq.read_schema(manifest)
    required = {"episode_index", "video_path", "split", "contiguous_label_index"}
    require(required <= set(schema.names), "Classification manifest schema mismatch")
    table = pq.read_table(manifest, filters=[("split", "in", ["train", "validation"])])
    selected = []
    for row in table.to_pylist():
        if any(key in row and row[key] != "primary" for key in ("v3_benchmark_inclusion", "classification_inclusion")):
            continue
        if "eligible_4s" in row and row["eligible_4s"] is not True:
            continue
        selected.append(row)
    panels = {split: sorted([r for r in selected if r["split"] == split], key=lambda r: int(r["episode_index"]))
              for split in ("train", "validation")}
    train_ids, val_ids = [{int(r["episode_index"]) for r in panels[split]} for split in panels]
    require(len(train_ids) == len(panels["train"]) and len(val_ids) == len(panels["validation"]), "Duplicate episodes")
    require(not train_ids & val_ids, "Episode overlap")
    train_labels = sorted({int(r["contiguous_label_index"]) for r in panels["train"]})
    val_labels = {int(r["contiguous_label_index"]) for r in panels["validation"]}
    require(len(train_labels) == 120 and val_labels <= set(train_labels), "120 train-supported classes required")
    require((len(train_ids), len(val_ids)) == (2757, 617), "Population differs from previously audited Core A; stop to review protocol")
    mapping = {label: i for i, label in enumerate(train_labels)}
    snapshots = scratch / "hf_cache/hub/models--MCG-NJU--videomae-small-finetuned-kinetics/snapshots"
    config_dir = snapshots / "1bf2b82c809609f5df1ea7648aee577e2e486b63"
    weights_path = snapshots / "240e9734611173accbbf74cbdf4b641e4c431264/model.safetensors"
    require(digest(config_dir / "config.json") == "20b05a5026313946ec026dc0c7be4f1f641c47ab66ec7125d406403ce251165c", "Config changed")
    require(digest(weights_path) == "6af658f96396ed5f1207244a5a409ca4f6686f05f63b1be6b3609deee195eca7", "Weights changed")
    processor_hash = digest(config_dir / "preprocessor_config.json")
    require(processor_hash == "c3aa722f22a7ff0d234d407862025fe47672f74f46f92c91f757f7a7354107c2", "Processor changed")
    if args.phase == "full":
        accepted = json.loads(args.preflight_report.read_text(encoding="utf-8"))
        require(accepted["status"] == "PASS_CORE_A_STRICT_CPU_PREFLIGHT" and accepted["strict_load"]
                and accepted["final_test_access"] == 0, "Preflight gate failed")
        require(accepted["manifest_sha256"] == digest(manifest) and accepted["weights_sha256"] == digest(weights_path)
                and accepted["processor_sha256"] == processor_hash, "Preflight identity mismatch")
    config = VideoMAEConfig.from_pretrained(config_dir, local_files_only=True)
    require(config.num_frames == 16 and config.hidden_size == 384, "Unexpected native architecture")
    config._attn_implementation = "eager"
    model = VideoMAEForVideoClassification(config).cpu().eval()
    weights = load_file(str(weights_path), device="cpu")
    expected = model.state_dict()
    converted = []
    for layer in range(config.num_hidden_layers):
        prefix = f"videomae.encoder.layer.{layer}.attention.attention."
        if prefix + "q_bias" not in expected:
            weights[prefix + "query.bias"] = weights.pop(prefix + "q_bias")
            weights[prefix + "value.bias"] = weights.pop(prefix + "v_bias")
            weights[prefix + "key.bias"] = torch.zeros_like(weights[prefix + "query.bias"])
            converted.append(layer)
    model.load_state_dict(weights, strict=True)
    model.requires_grad_(False)
    model.to(device)
    processor = VideoMAEImageProcessor.from_pretrained(config_dir, local_files_only=True)
    raw = (project / "data_raw/HumanoidEveryday/G1/USC-PSI-Lab-Humanoid-Everyday-G1__6b6b599d").resolve()
    records, features = [], []
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    total = 3374 if args.phase == "full" else 4
    for split in panels:
        for row in (panels[split] if args.phase == "full" else panels[split][:2]):
            path = (raw / str(row["video_path"])).resolve()
            require(path.is_relative_to(raw) and path.is_file(), "Video outside raw root or missing")
            capture = cv2.VideoCapture(str(path))
            try:
                require(capture.isOpened(), "Video open failure")
                fps = float(capture.get(cv2.CAP_PROP_FPS))
                count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                require(np.isfinite(fps) and fps > 0 and count / fps >= 4.0, "Insufficient four-second video")
                indices = np.rint(np.arange(16, dtype=np.float64) / 4.0 * fps).astype(np.int64)
                require(int(indices[-1]) < count, "Sampling past video end")
                wanted, frames = set(indices.tolist()), {}
                for index in range(int(indices[-1]) + 1):
                    ok, bgr = capture.read()
                    require(ok, f"Video decode failure at {index}")
                    if index in wanted:
                        frames[index] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            finally:
                capture.release()
            pixels = processor([[frames[int(i)] for i in indices]], return_tensors="pt")
            require(tuple(pixels["pixel_values"].shape) == (1, 16, 3, 224, 224), "Processor shape mismatch")
            with torch.inference_mode():
                tokens = model.videomae(**{key: value.to(device) for key, value in pixels.items()}).last_hidden_state
                pooled = tokens.mean(dim=1)
            require(tuple(tokens.shape) == (1, 1568, 384) and torch.isfinite(pooled).all(), "Encoder output mismatch")
            features.append(pooled[0].cpu().numpy())
            records.append({"split": split, "episode_index": int(row["episode_index"]),
                            "label": mapping[int(row["contiguous_label_index"])], "fps": fps,
                            "sampled_frame_indices": indices.tolist()})
            progress = {"status": "running", "processed": len(records), "total": total,
                        "split": split, "elapsed_seconds": time.monotonic() - started}
            temporary = output / "EXTRACTION_STATUS.json.part"
            temporary.write_text(json.dumps(progress) + "\n", encoding="utf-8")
            temporary.replace(output / "EXTRACTION_STATUS.json")
            if args.phase == "smoke" or len(records) % 16 == 0 or len(records) == total:
                print(json.dumps(progress), flush=True)
    require(len(records) == total, "Extraction population incomplete")
    np.savez_compressed(output / "SMOKE_FEATURES.npz", features=np.stack(features),
                        episode_index=np.asarray([r["episode_index"] for r in records]),
                        labels=np.asarray([r["label"] for r in records]), split=np.asarray([r["split"] for r in records]))
    if args.phase == "full":
        for split in panels:
            positions = [i for i, r in enumerate(records) if r["split"] == split]
            path = output / f"videomae_{split}_pooled.pt"
            metadata = {"model_name": "MCG-NJU/videomae-small-finetuned-kinetics", "split": split,
                        "input_variant": "dynamic", "manifest": str(manifest), "manifest_sha256": digest(manifest),
                        "g1_root": str(raw), "input_frames": 16, "sample_fps": 4.0, "window_seconds": 4.0,
                        "encoder_frozen": True, "strict_load": True, "weights_sha256": digest(weights_path),
                        "final_test_used": False, "num_examples": len(positions), "feature_dim": 384}
            temporary = path.with_suffix(".pt.part")
            torch.save({"features": torch.from_numpy(np.stack([features[i] for i in positions])),
                        "labels": torch.tensor([records[i]["label"] for i in positions], dtype=torch.long),
                        "episode_indices": torch.tensor([records[i]["episode_index"] for i in positions], dtype=torch.long),
                        "metadata": metadata}, temporary)
            temporary.replace(path)
    report = {"status": "PASS_CORE_A_STRICT_FULL_FEATURES" if args.phase == "full" else "PASS_CORE_A_STRICT_CPU_PREFLIGHT", "train_examples": len(train_ids),
              "validation_examples": len(val_ids), "classes": len(mapping), "label_mapping_train_only": mapping,
              "manifest": str(manifest), "manifest_sha256": digest(manifest), "schema_columns": schema.names,
              "config_sha256": digest(config_dir / "config.json"), "weights_sha256": digest(weights_path),
              "processor_sha256": processor_hash, "strict_load": True, "converted_layers": converted,
              "records": records, "feature_shape": [total, 384], "script_sha256": digest(Path(__file__)),
              "sampling": "16 frames at t=0,0.25,...,3.75s; nearest frame; no decode fallback",
              "pooling": "encoder token mean, frozen backbone; Kinetics classifier not used",
              "protocol_scope": "Reuses previously audited Core A classification population and prior baseline sampling; current team comparison settings still require alignment.",
              "formal_training_started": False, "gpu_used": device == "cuda", "final_test_access": 0,
              "elapsed_seconds": time.monotonic() - started}
    if args.phase == "full":
        report["feature_files"] = {split: {"path": str(output / f"videomae_{split}_pooled.pt"),
                                           "sha256": digest(output / f"videomae_{split}_pooled.pt")} for split in panels}
        report["preflight_report_sha256"] = digest(args.preflight_report)
        report["peak_reserved_mib"] = (torch.cuda.max_memory_reserved() + 1024**2 - 1) // 1024**2
    report_path = output / ("FEATURE_MANIFEST.json" if args.phase == "full" else "CORE_A_PREFLIGHT.json")
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "EXTRACTION_STATUS.json").write_text(json.dumps({"status": "complete", "processed": total, "total": total}) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("status", "train_examples", "validation_examples", "classes", "strict_load", "feature_shape", "gpu_used", "final_test_access")}, indent=2))
    print("report:", report_path)


if __name__ == "__main__":
    main()
