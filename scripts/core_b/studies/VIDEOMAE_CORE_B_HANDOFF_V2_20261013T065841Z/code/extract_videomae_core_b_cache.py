"""Restartable target-blind VideoMAE Core B cache extraction via GPU watcher."""

from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time


CONFIG_SHA = "20b05a5026313946ec026dc0c7be4f1f641c47ab66ec7125d406403ce251165c"
WEIGHT_SHA = "6af658f96396ed5f1207244a5a409ca4f6686f05f63b1be6b3609deee195eca7"
INPUT_SHA = "61e8c9bbac6bb83b7d323d6ac07d0edaf5a745caf4c58407de97fdedc7d9ce78"
PROCESSOR_SHA = "c3aa722f22a7ff0d234d407862025fe47672f74f46f92c91f757f7a7354107c2"
SHARD_ROWS = 64


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.part")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--stage0", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    project, output = args.project_root.resolve(), args.output_root.resolve()
    scratch = (project / 'tmp/project-team').resolve()
    if output == scratch or not output.is_relative_to(scratch):
        raise ValueError("Cache output must be an isolated personal scratch child")
    physical = os.environ.get("GPU_WATCHER_PHYSICAL_GPU")
    if physical not in {"0", "1", "2", "3"} or os.environ.get("CUDA_VISIBLE_DEVICES") != physical:
        raise RuntimeError("Extraction must run through the shared watcher with an authorized GPU")
    if output.exists() and not args.resume:
        raise FileExistsError("Existing output requires explicit --resume")
    if args.resume and not output.is_dir():
        raise FileNotFoundError("Resume requires an existing cache")

    import cv2
    import numpy as np
    import torch
    from safetensors.torch import load_file
    from transformers import VideoMAEConfig, VideoMAEForVideoClassification, VideoMAEImageProcessor

    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError("Assigned CUDA device unavailable")
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats()
    snapshots = project / 'tmp/project-team/hf_cache/hub/models--MCG-NJU--videomae-small-finetuned-kinetics/snapshots'
    config_dir = snapshots / "1bf2b82c809609f5df1ea7648aee577e2e486b63"
    weights_file = snapshots / "240e9734611173accbbf74cbdf4b641e4c431264/model.safetensors"
    inputs_file = project / "repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z/data/V3_MAIN_INPUTS.npz"
    for path, expected in ((config_dir / "config.json", CONFIG_SHA), (weights_file, WEIGHT_SHA),
                           (inputs_file, INPUT_SHA), (config_dir / "preprocessor_config.json", PROCESSOR_SHA)):
        if digest(path) != expected:
            raise AssertionError(f"Source identity mismatch: {path}")
    with np.load(inputs_file, allow_pickle=False) as archive:
        keys = ("anchor_id", "episode_index", "anchor_frame", "video_path", "split_code")
        data = {key: archive[key].copy() for key in keys}
    codes = data["split_code"]
    if len(codes) != 20079 or int((codes == 0).sum()) != 17757 or int((codes == 1).sum()) != 2322:
        raise AssertionError("Primary train/dev row counts mismatch")
    for split, expected in ((0, 2712), (1, 603)):
        if len(np.unique(data["episode_index"][codes == split])) != expected:
            raise AssertionError("Primary episode counts mismatch")
    if len(set(data["anchor_id"].tolist())) != len(codes):
        raise AssertionError("Anchor IDs must be unique")
    selected = np.concatenate([np.flatnonzero(codes == s)[:2] for s in (0, 1)]) if args.stage0 else np.arange(len(codes))
    identity = {
        "protocol_id": "VIDEOMAE_CORE_B_32FRAME_CACHE_V1", "stage0": args.stage0,
        "script_sha256": digest(Path(__file__)), "config_sha256": CONFIG_SHA,
        "weights_sha256": WEIGHT_SHA, "input_sha256": INPUT_SHA,
        "processor_sha256": PROCESSOR_SHA, "rows": len(selected),
        "frames": 32, "dtype": "float32", "batch_size": 1,
        "sampling": "floor(linspace(anchor-119,anchor,32))",
        "static_rule": "repeat_anchor_frame", "feature_shape": [2, 16, 4, 384],
        "quadrant_order": ["top_left", "top_right", "bottom_left", "bottom_right"],
        "position_encoding": "fixed sinusoidal table regenerated for 32 frames",
        "targets_loaded": False, "final_test_access": 0,
    }
    if args.resume:
        if json.loads((output / "CACHE_CONFIG.json").read_text()) != identity:
            raise AssertionError("Resume cache configuration/source identity changed")
    else:
        required = len(selected) * 2 * 16 * 4 * 384 * 4 + 512 * 1024**2
        if shutil.disk_usage(scratch).free < required:
            raise RuntimeError("Insufficient disk space for cache and reserve")
        output.mkdir(parents=True, exist_ok=False)
        (output / "shards").mkdir()
        (output / "progress").mkdir()
        atomic_json(output / "CACHE_CONFIG.json", identity)
        np.savez_compressed(output / "ANCHOR_INDEX.npz", cache_index=selected,
                            **{key: value[selected] for key, value in data.items()})

    def shard_path(row):
        start = row // SHARD_ROWS * SHARD_ROWS
        end = min(start + SHARD_ROWS, len(selected))
        return output / "shards" / f"grid_{start:06d}_{end:06d}.npy", start, end

    completed = set()
    for journal_path in sorted((output / "progress").glob("*.json")):
        journal = json.loads(journal_path.read_text())
        for entry in journal["rows"]:
            row = int(entry["row"])
            if row in completed or not 0 <= row < len(selected):
                raise AssertionError("Invalid or duplicate committed cache row")
            path, start, end = shard_path(row)
            values = np.load(path, mmap_mode="r", allow_pickle=False)
            if values.shape != (end - start, 2, 16, 4, 384) or values.dtype != np.float32:
                raise AssertionError("Committed shard schema mismatch")
            if hashlib.sha256(values[row - start].tobytes()).hexdigest() != entry["sha256"]:
                raise AssertionError("Committed row checksum mismatch")
            del values
            completed.add(row)

    config = VideoMAEConfig.from_pretrained(config_dir, local_files_only=True)
    config.num_frames = 32
    config._attn_implementation = "eager"
    model = VideoMAEForVideoClassification(config).cpu().eval()
    weights = load_file(str(weights_file), device="cpu")
    expected = model.state_dict()
    for layer in range(config.num_hidden_layers):
        prefix = f"videomae.encoder.layer.{layer}.attention.attention."
        if prefix + "q_bias" not in expected:
            weights[prefix + "query.bias"] = weights.pop(prefix + "q_bias")
            weights[prefix + "value.bias"] = weights.pop(prefix + "v_bias")
            weights[prefix + "key.bias"] = torch.zeros_like(weights[prefix + "query.bias"])
    model.load_state_dict(weights, strict=True)
    del weights
    model.requires_grad_(False)
    model.to("cuda:0")
    processor = VideoMAEImageProcessor.from_pretrained(config_dir, local_files_only=True)

    def encode(frames):
        pixels = processor([frames], return_tensors="pt")
        if tuple(pixels["pixel_values"].shape) != (1, 32, 3, 224, 224):
            raise AssertionError("Processor input mismatch")
        with torch.inference_mode():
            tokens = model.videomae(**{k: v.to("cuda:0") for k, v in pixels.items()}).last_hidden_state
        if tuple(tokens.shape) != (1, 3136, 384) or not torch.isfinite(tokens).all():
            raise AssertionError("Encoder shape/finite mismatch")
        grid = tokens.reshape(1, 16, 14, 14, 384)
        return torch.stack([grid[:, :, r:r + 7, c:c + 7].mean(dim=(2, 3))
                            for r in (0, 7) for c in (0, 7)], dim=2)[0].cpu().numpy()

    groups = {}
    for row, source in enumerate(selected):
        if row not in completed:
            groups.setdefault(str(data["video_path"][source]), []).append((int(data["anchor_frame"][source]), row))
    raw_root = (project / "data_raw/HumanoidEveryday/G1/USC-PSI-Lab-Humanoid-Everyday-G1__6b6b599d").resolve()
    started = time.monotonic()
    for video, anchors in groups.items():
        anchors.sort()
        if anchors[0][0] < 119:
            raise AssertionError("Anchor lacks four-second support")
        video_path = (raw_root / video).resolve()
        if not video_path.is_relative_to(raw_root) or not video_path.is_file():
            raise FileNotFoundError(video_path)
        capture = cv2.VideoCapture(str(video_path))
        buffer = deque(maxlen=120)
        frame_index, commits = -1, []
        try:
            if not capture.isOpened():
                raise RuntimeError(f"Cannot decode {video_path}")
            for anchor, row in anchors:
                while frame_index < anchor:
                    ok, frame = capture.read()
                    if not ok:
                        raise RuntimeError(f"Video decode failure at {frame_index + 1}")
                    frame_index += 1
                    buffer.append((frame_index, frame))
                frame_map = dict(buffer)
                wanted = np.floor(np.linspace(anchor - 119, anchor, 32)).astype(np.int64)
                rgb = [cv2.cvtColor(frame_map[int(i)], cv2.COLOR_BGR2RGB) for i in wanted]
                pair = np.stack((encode(rgb), encode([rgb[-1]] * 32)))
                if not np.isfinite(pair).all():
                    raise AssertionError("Nonfinite cache row")
                path, start, end = shard_path(row)
                if path.exists():
                    mm = np.lib.format.open_memmap(path, mode="r+")
                    if mm.shape != (end - start, 2, 16, 4, 384) or mm.dtype != np.float32:
                        raise AssertionError("Existing shard schema mismatch")
                else:
                    mm = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32,
                                                  shape=(end - start, 2, 16, 4, 384))
                mm[row - start] = pair
                mm.flush()
                del mm
                commits.append({"row": row, "sha256": hashlib.sha256(pair.tobytes()).hexdigest()})
        finally:
            capture.release()
        journal_name = hashlib.sha256(video.encode()).hexdigest() + ".json"
        # Reuse the original journal on resume only if this video was not committed.
        journal_path = output / "progress" / journal_name
        if journal_path.exists():
            raise AssertionError("Unexpected existing video commit")
        atomic_json(journal_path, {"rows": commits})
        completed.update(item["row"] for item in commits)
        atomic_json(output / "EXTRACTION_STATUS.json", {"status": "running", "completed_rows": len(completed),
                    "total_rows": len(selected), "physical_gpu": physical, "elapsed_seconds": time.monotonic() - started})
        print(json.dumps({"completed_rows": len(completed), "total_rows": len(selected)}), flush=True)
    if completed != set(range(len(selected))):
        raise AssertionError("Incomplete cache row coverage")
    shards = []
    for start in range(0, len(selected), SHARD_ROWS):
        path, _, end = shard_path(start)
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        if not np.isfinite(values).all():
            raise AssertionError("Final shard finite check failed")
        del values
        shards.append({"array": str(path), "start_row": start, "end_row": end,
                       "sha256": digest(path), "bytes": path.stat().st_size})
    report = {**identity, "status": "PASS_STAGE0" if args.stage0 else "PASS",
              "covered_rows": len(completed), "logical_shape": [len(selected), 2, 16, 4, 384],
              "shards": shards, "anchor_index_sha256": digest(output / "ANCHOR_INDEX.npz"),
              "peak_allocated_mib": math.ceil(torch.cuda.max_memory_allocated() / 1024**2),
              "peak_reserved_mib": math.ceil(torch.cuda.max_memory_reserved() / 1024**2),
              "physical_gpu": physical, "watcher_job_id": os.environ.get("GPU_WATCHER_JOB_ID"),
              "elapsed_seconds": time.monotonic() - started}
    atomic_json(output / "CACHE_MANIFEST.json", report)
    atomic_json(output / "EXTRACTION_STATUS.json", {"status": "complete", "covered_rows": len(completed)})
    print(json.dumps({key: value for key, value in report.items() if key != "shards"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
