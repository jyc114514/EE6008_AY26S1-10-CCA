"""Audit cached safetensors keys and installed VideoMAE bias implementation."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    args = parser.parse_args()
    snapshot = args.snapshot.resolve()
    config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    from safetensors import safe_open
    from transformers.models.videomae import modeling_videomae

    headers = []
    inventory = []
    snapshots = [snapshot]
    if snapshot.parent.name == "snapshots":
        snapshots = sorted(path for path in snapshot.parent.iterdir() if path.is_dir())
    weight_paths = []
    for revision in snapshots:
        for path in sorted(revision.iterdir()):
            if path.suffix not in {".safetensors", ".bin", ".json"}:
                continue
            inventory.append({
                "path": str(path), "exists": path.is_file(),
                "bytes": path.stat().st_size if path.is_file() else None,
            })
            if path.suffix == ".safetensors" and path.is_file():
                weight_paths.append(path)
    for path in weight_paths:
        with safe_open(path, framework="pt", device="cpu") as weights:
            keys = list(weights.keys())
            bias_keys = [key for key in keys if "layer.0." in key and (
                "q_bias" in key or "v_bias" in key or
                "attention.attention" in key and key.endswith(".bias")
            )]
            headers.append({
                "file": str(path), "bytes": path.stat().st_size,
                "key_count": len(keys),
                "first_layer_attention_biases": {
                    key: weights.get_slice(key).get_shape() for key in bias_keys
                },
            })
    classes = {}
    for name in ("VideoMAESelfAttention", "VideoMAEEmbeddings", "VideoMAEPatchEmbeddings"):
        cls = getattr(modeling_videomae, name, None)
        if cls is not None:
            classes[name] = inspect.getsource(cls)
    print(json.dumps({
        "status": "structure_audit_only",
        "config": {key: config.get(key) for key in (
            "hidden_size", "num_frames", "tubelet_size", "patch_size",
            "image_size", "qkv_bias", "num_attention_heads",
        )},
        "weight_headers": headers,
        "snapshot_file_inventory": inventory,
        "installed_implementation": classes,
        "gpu_used": False,
        "model_instantiated": False,
        "weight_tensors_loaded": False,
        "final_test_access": 0,
    }, indent=2))
    return 0 if headers else 1


if __name__ == "__main__":
    raise SystemExit(main())
