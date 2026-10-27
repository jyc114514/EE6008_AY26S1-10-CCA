"""Archive trusted artifacts, export tensor-only weights, and replay cached-feature inference."""

import argparse
import importlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

from evaluate_videomae_core_b import digest, require

V3 = "repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z/data"
CODE = "reports/benchmarks/D3_EQ_CORRECTED_SCHEDULER_V1_20261006T073406Z/code"
CACHE = 'tmp/project-team/videomae_cache_full_20261008T074027Z'
TARGET = "reports/benchmarks/CORE_B_D3_EPISODE_EQUAL_SENSITIVITY_V1_20260924T095800Z/d3/train_targets_trainonly.npz"
SPLIT = "repo/ee6008-humanoid-vjepa/manifests/g1_6b6b599def75e3558b2736062e79bdbeec6cd224/splits/core_a_protocol_v3_duplicate_safe/split_manifest.parquet"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def relative(path, root):
    path = Path(path).resolve()
    require(path.is_relative_to(root), f"Dependency outside original project: {path}")
    return path.relative_to(root).as_posix()


def load_model(root, cell, device):
    import torch
    from safetensors.torch import load_file

    config = read(root / cell["config"])
    provenance = read(root / cell["manifest"])
    require(digest(root / cell["weights"]) == provenance["weights_sha256"], "Safe weights changed")
    require(digest(root / cell["config"]) == provenance["config_sha256"], "Config changed")
    for name, sha in config["source_hashes"].items():
        require(digest(root / CODE / name) == sha, f"Model code changed: {name}")
    sys.path.insert(0, str(root / CODE))
    models = importlib.import_module("d3_model")
    models.VIDEO_DIM = 768
    with torch.random.fork_rng(devices=[]):
        model = models.D3TimeSpacePredictor(horizon=config["horizon"], seed=config["seed"], video_mode=config["mode"])
        if config["mode"] == "TS11-D":
            model.video_projection = torch.nn.Linear(384, 128)
            models.VIDEO_DIM = 384
    model.load_state_dict(load_file(str(root / cell["weights"]), device="cpu"), strict=True)
    return model.to(device).eval(), config


def prepare(args):
    import torch
    from safetensors.torch import save_file, load_file

    project, output = args.project_root.resolve(), args.archive_root.resolve()
    require(not output.exists() and output.is_relative_to(project / "reports/benchmarks"), "Fresh benchmark archive required")
    index = read(args.handoff_index)
    require(index["status"] == "PASS_VIDEOMAE_HANDOFF_ARTIFACT_AUDIT", "Handoff audit required")
    cells = index["core_b"]["cells"]
    expected = {(m, h, s) for m in ("TS11-D", "STATE-CTRL") for h in (30, 60) for s in (20261001, 20261002, 20261003)}
    require({(c["mode"], c["horizon"], c["seed"]) for c in cells} == expected and len(cells) == 12, "Incomplete matrix")
    output.mkdir(parents=True)
    files = {}

    def copy(path, sha=None):
        path = Path(path).resolve()
        rel = relative(path, project)
        actual = digest(path)
        require(sha is None or sha == actual, f"Source hash mismatch: {path}")
        dest = output / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            shutil.copy2(path, dest)
        require(digest(dest) == actual, "Copy mismatch")
        files[rel] = {"sha256": actual, "bytes": dest.stat().st_size}
        return rel

    # Retain all previously audited evidence, including original checkpoints.
    for path, record in index["artifacts"].items():
        copy(path, record["sha256"])
    for rel in (V3 + "/V3_MAIN_INPUTS.npz", V3 + "/V3_NORMALIZERS.json", TARGET, SPLIT):
        copy(project / rel)
    for path in (project / CODE).glob("*.py"):
        copy(path)
    manifest = read(project / CACHE / "cache/CACHE_MANIFEST.json")
    for shard in manifest["shards"]:
        copy(shard["array"], shard["sha256"])
    copy(project / CACHE / "cache/ANCHOR_INDEX.npz", manifest["anchor_index_sha256"])
    snapshots = project / 'tmp/project-team/hf_cache/hub/models--MCG-NJU--videomae-small-finetuned-kinetics/snapshots'
    for rev, name in (("1bf2b82c809609f5df1ea7648aee577e2e486b63", "config.json"),
                      ("1bf2b82c809609f5df1ea7648aee577e2e486b63", "preprocessor_config.json"),
                      ("240e9734611173accbbf74cbdf4b641e4c431264", "model.safetensors")):
        copy(snapshots / rev / name)
    exported = []
    for c in cells:
        run = Path(c["run_root"])
        config, status = read(run / "CONFIG.json"), read(run / "RUN_STATUS.json")
        require(all(config[k] == c[k] for k in ("mode", "horizon", "seed"))
                and config["final_test_access"] == 0 and not config["resume_check"], "Run scope mismatch")
        checkpoint = run / "checkpoint_last.pt"
        require(digest(checkpoint) == status["checkpoint_sha256"], "Checkpoint binding mismatch")
        require(status["status"] == "PASS_TRAIN_AND_EXPORT" and status["epochs"] == 40, "Incomplete run")
        # Full historical checkpoints include NumPy RNG pickle objects. Unsafe
        # deserialization is never attempted without explicit owner authorization.
        payload = torch.load(checkpoint, map_location="cpu", weights_only=not args.owner_authorized_pickle)
        require(payload["identity"] == config and payload["epoch"] == 40, "Checkpoint identity mismatch")
        name = f'{c["mode"]}_H{c["horizon"]}_S{c["seed"]}'
        dest = output / "models" / name
        dest.mkdir(parents=True)
        tensors = payload["model"]
        require(all(isinstance(k, str) and torch.is_tensor(v) for k, v in tensors.items()), "Tensor-only state required")
        tensors = {k: v.detach().cpu().contiguous().clone() for k, v in tensors.items()}
        weights = dest / "model.safetensors"
        save_file(tensors, str(weights), metadata={"original_checkpoint_sha256": digest(checkpoint),
                  "config_sha256": digest(run / "CONFIG.json"), "epoch": str(payload["epoch"]),
                  "seed": str(config["seed"]), "horizon": str(config["horizon"]), "mode": config["mode"]})
        restored = load_file(str(weights))
        require(tensors.keys() == restored.keys() and all(torch.equal(v, restored[k]) for k, v in tensors.items()), "Export tensor mismatch")
        config_rel = relative(run / "CONFIG.json", project)
        record = {"original_checkpoint": relative(checkpoint, project), "original_checkpoint_sha256": digest(checkpoint),
                  "config_sha256": digest(run / "CONFIG.json"), "weights_sha256": digest(weights),
                  "seed": config["seed"], "mode": config["mode"], "horizon": config["horizon"], "epoch": payload["epoch"],
                  "tensor_count": len(tensors), "final_test_access": 0}
        write(dest / "MODEL_MANIFEST.json", record)
        exported.append({"name": name, "run": relative(run, project), "config": config_rel,
                         "weights": weights.relative_to(output).as_posix(), "manifest": (dest / "MODEL_MANIFEST.json").relative_to(output).as_posix(),
                         "metrics": relative(c["metrics"], project)})
        print(json.dumps({"exported": name}), flush=True)
    (output / "code").mkdir()
    for name in ("close_videomae_core_b_handoff.py", "evaluate_videomae_core_b.py", "extract_videomae_core_b_cache.py"):
        shutil.copy2(Path(__file__).parent / name, output / "code" / name)
    write(output / "ENVIRONMENT_EXPORT.json", {"python": sys.version, "torch": torch.__version__})
    for path in output.rglob("*"):
        if path.is_file():
            files[path.relative_to(output).as_posix()] = {"sha256": digest(path), "bytes": path.stat().st_size}
    write(output / "ARCHIVE.json", {"status": "PREPARED_NOT_REPLAYED", "original_project": str(project), "cells": exported,
          "files": files, "inference_scope": "cached-feature predictor replay; raw-video encoding not replayed", "final_test_access": 0})
    print(json.dumps({"status": "PREPARED_NOT_REPLAYED", "archive": str(output), "cells": len(exported)}))


def max_metric_difference(a, b):
    if isinstance(a, dict):
        require(a.keys() == b.keys(), "Metric keys differ")
        return max((max_metric_difference(a[k], b[k]) for k in a), default=0.0)
    if isinstance(a, list):
        require(len(a) == len(b), "Metric lengths differ")
        return max((max_metric_difference(x, y) for x, y in zip(a, b)), default=0.0)
    if isinstance(a, (int, float)) and not isinstance(a, bool):
        require(isinstance(b, (int, float)), "Metric type mismatch")
        require(math.isfinite(a) and math.isfinite(b), "Nonfinite metric")
        return abs(a - b)
    require(a == b, "Metric value mismatch")
    return 0.0


def replay(args):
    import numpy as np
    import torch

    require(math.isfinite(args.prediction_atol) and math.isfinite(args.metric_atol)
            and args.prediction_atol >= 0 and args.metric_atol >= 0, "Finite nonnegative tolerances required")
    torch.set_num_threads(2)
    root = args.archive_root.resolve()
    archive = read(root / "ARCHIVE.json")
    expected = {f"{m}_H{h}_S{s}" for m in ("TS11-D", "STATE-CTRL") for h in (30, 60) for s in (20261001, 20261002, 20261003)}
    require(len(archive["cells"]) == 12 and {c["name"] for c in archive["cells"]} == expected, "Complete replay matrix required")
    for rel, record in archive["files"].items():
        require((root / rel).resolve().is_relative_to(root), "Unsafe archive path")
        require(digest(root / rel) == record["sha256"], f"Archive changed: {rel}")
    for cell in archive["cells"]:
        provenance = read(root / cell["manifest"])
        require(digest(root / cell["weights"]) == provenance["weights_sha256"], "Safe weights changed")
    dest = root / "replay" / args.replay_name
    require(not dest.exists(), "Fresh replay name required; preserve failed attempts")
    dest.mkdir(parents=True)
    with np.load(root / V3 / "V3_MAIN_INPUTS.npz", allow_pickle=False) as data:
        ids, episodes, codes = [data[k].copy() for k in ("anchor_id", "episode_index", "split_code")]
        x = [data[k].astype(np.float32) for k in ("history_standardized", "history_mask", "history_relative_delta_standardized", "anchor_command_native")]
    require(set(codes.tolist()) == {0, 1}, "Train/dev only required")
    manifest = read(root / CACHE / "cache/CACHE_MANIFEST.json")
    normalizer = read(root / CACHE / "normalization/VIDEOMAE_VIDEO_NORMALIZER.json")
    sigma = torch.tensor(read(root / V3 / "V3_NORMALIZERS.json")["target_sigma"], dtype=torch.float32, device=args.device)
    original_root = Path(archive["original_project"])
    shards = [(s, np.load(root / relative(s["array"], original_root), mmap_mode="r", allow_pickle=False)) for s in manifest["shards"]]
    reports = []
    for cell in archive["cells"]:
        model, config = load_model(root, cell, args.device)
        require(digest(root / CACHE / "cache/CACHE_MANIFEST.json") == config["cache_manifest_sha256"]
                and digest(root / CACHE / "normalization/VIDEOMAE_VIDEO_NORMALIZER.json") == config["normalizer_sha256"], "Feature binding mismatch")
        require(digest(root / V3 / "V3_MAIN_INPUTS.npz") == manifest["input_sha256"]
                and digest(root / V3 / "V3_NORMALIZERS.json") == config["target_normalizer_sha256"], "Input/target scale binding mismatch")
        run = dest / cell["name"]
        run.mkdir()
        source = root / cell["run"]
        shutil.copy2(source / "CONFIG.json", run / "CONFIG.json")
        shutil.copy2(source / "checkpoint_last.pt", run / "checkpoint_last.pt")
        status = read(source / "RUN_STATUS.json")
        errors = {}
        for split, code in (("train", 0), ("dev", 1)):
            rows = np.flatnonzero(codes == code)
            pred = np.empty((len(rows), config["horizon"], 28), dtype=np.float32)
            with torch.inference_mode():
                for start in range(0, len(rows), 64):
                    selected = rows[start:start + 64]
                    state, mask, rel, anchor = [torch.tensor(v[selected], device=args.device) for v in x]
                    if config["mode"] == "STATE-CTRL":
                        dynamic = static = torch.zeros(len(selected), 16, 4, 768, device=args.device)
                    else:
                        grids = np.stack([next(a[int(row) - s["start_row"]].copy() for s, a in shards if s["start_row"] <= row < s["end_row"]) for row in selected])
                        grids = (grids - np.asarray(normalizer["mean"], dtype=np.float32)) / np.asarray(normalizer["safe_std"], dtype=np.float32)
                        dynamic, static = [torch.tensor(grids[:, i], device=args.device) for i in (0, 1)]
                    pred[start:start + len(selected)] = (anchor[:, None] + sigma[None, None] * model(state, mask, rel, dynamic, static, anchor)).cpu().numpy()
            require(np.isfinite(pred).all(), "Nonfinite replay")
            with np.load(source / f"predictions_{split}.npz", allow_pickle=False) as reference:
                require(np.array_equal(reference["anchor_id"], ids[rows]) and np.array_equal(reference["episode_index"], episodes[rows]), "ID mismatch")
                require(reference["predictions_native"].shape == pred.shape, "Shape mismatch")
                errors[split] = float(np.max(np.abs(pred - reference["predictions_native"])))
            np.savez_compressed(run / f"predictions_{split}.npz", anchor_id=ids[rows], episode_index=episodes[rows], predictions_native=pred)
            status[f"{split}_prediction_sha256"] = digest(run / f"predictions_{split}.npz")
        write(run / "RUN_STATUS.json", status)
        subprocess.run([sys.executable, str(root / "code/evaluate_videomae_core_b.py"), "--archive", "--project-root", str(root), "--run-root", str(run), "--output-root", str(run / "evaluation")], check=True)
        metric_error = max_metric_difference(read(run / "evaluation/METRICS.json")["panels"], read(root / cell["metrics"])["panels"])
        passed = max(errors.values()) <= args.prediction_atol and metric_error <= args.metric_atol
        report = {"cell": cell["name"], "prediction_max_abs": errors, "metric_max_abs": metric_error, "passed": passed}
        write(run / "REPLAY.json", report)
        reports.append(report)
        print(json.dumps(report), flush=True)
    passed = all(r["passed"] for r in reports)
    write(dest / "REPLAY_SUMMARY.json", {"status": "PASS_12_SAFE_REPLAYS" if passed else "FAIL_REPLAY_TOLERANCE", "results": reports,
          "prediction_atol": args.prediction_atol, "metric_atol": args.metric_atol, "device": args.device,
          "torch": torch.__version__, "numpy": np.__version__, "python": sys.version, "final_test_access": 0,
          "scope": "cached-feature inference then scoring; no retraining or raw-video replay"})
    require(passed, "Replay exceeds agreed tolerance; see REPLAY_SUMMARY.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--project-root", type=Path, required=True)
    p.add_argument("--handoff-index", type=Path, required=True)
    p.add_argument("--archive-root", type=Path, required=True)
    p.add_argument("--owner-authorized-pickle", action="store_true", help="Explicitly authorize deserializing your trusted, hash-verified original checkpoints; never use for untrusted files")
    p = commands.add_parser("replay")
    p.add_argument("--archive-root", type=Path, required=True)
    p.add_argument("--replay-name", required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--prediction-atol", type=float, required=True)
    p.add_argument("--metric-atol", type=float, required=True)
    args = parser.parse_args()
    if args.command == "replay":
        require(Path(args.replay_name).name == args.replay_name and args.replay_name not in (".", ".."), "Replay name must be a single directory name")
    (prepare if args.command == "prepare" else replay)(args)


if __name__ == "__main__":
    main()
