"""Read small benchmark metadata without loading arrays, weights or CUDA."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path


def inspect_json(path: Path) -> dict:
    if not path.is_file():
        return {"path": str(path), "status": "missing"}
    if path.stat().st_size > 2 * 1024 * 1024:
        return {"path": str(path), "status": "too_large"}
    raw = path.read_bytes()
    content = json.loads(raw)
    # Avoid printing hundreds of shards, normalizer channels and label names.
    if isinstance(content, dict):
        content = {key: value for key, value in content.items()
                   if key not in {"shards", "mean", "safe_std", "id2label", "label2id"}}
    return {
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "content": content,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    bench = root / "reports/benchmarks/CORE_B_BENCHMARK_LAB_V1_20260920T065538Z"
    eq = root / "reports/benchmarks/CORE_B_D3_EPISODE_EQUAL_SENSITIVITY_V1_20260924T095800Z"
    cs = root / "reports/benchmarks/D3_EQ_CORRECTED_SCHEDULER_V1_20261006T073406Z"
    metadata = [
        bench / "d3/cache/D3_CACHE_MANIFEST.json",
        bench / "d3/cache/D3_VIDEO_NORMALIZER.json",
        eq / "d3/TRAIN_TARGET_CACHE_MANIFEST.json",
        eq / "protocol/D3_EQ_PROTOCOL_LOCKED.json",
        cs / "protocol/COMPARISON_FAMILIES.json",
    ]
    versions = {}
    for name in ("torch", "transformers", "huggingface-hub", "numpy", "opencv-python"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not_installed"

    # Only inspect the exact model's small snapshot configuration files.
    cache_roots = {
        root / "tmp/s126mdg33_02/hf_cache/hub",
        Path.home() / ".cache/huggingface/hub",
    }
    if os.environ.get("HF_HOME"):
        cache_roots.add(Path(os.environ["HF_HOME"]) / "hub")
    if os.environ.get("HF_HUB_CACHE"):
        cache_roots.add(Path(os.environ["HF_HUB_CACHE"]))
    model_configs = []
    model_dir = "models--MCG-NJU--videomae-small-finetuned-kinetics"
    for cache_root in sorted(cache_roots):
        snapshots = cache_root / model_dir / "snapshots"
        if snapshots.is_dir():
            for snapshot in sorted(snapshots.iterdir()):
                if snapshot.is_dir():
                    for name in ("config.json", "preprocessor_config.json"):
                        model_configs.append(inspect_json(snapshot / name))
    print(json.dumps({
        "status": "metadata_only",
        "versions": versions,
        "benchmark_metadata": [inspect_json(path) for path in metadata],
        "model_configs": model_configs,
        "arrays_loaded": False,
        "weights_loaded": False,
        "gpu_used": False,
        "final_test_access": 0,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
