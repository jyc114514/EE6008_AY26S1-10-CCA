"""Offline CPU verification of legacy VideoMAE attention-bias conversion."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--forward-smoke", action="store_true")
    parser.add_argument("--input-frames", type=int, choices=(16, 32), default=16,
                        help="32 rebuilds fixed sinusoidal positions; checkpoint tensors are unchanged")
    args = parser.parse_args()
    import torch
    from safetensors.torch import load_file
    from transformers import VideoMAEConfig, VideoMAEForVideoClassification

    torch.set_num_threads(2)
    config = VideoMAEConfig.from_pretrained(args.config_dir, local_files_only=True)
    pretrained_frames = config.num_frames
    config.num_frames = args.input_frames
    config._attn_implementation = "eager"
    original = load_file(str(args.weights), device="cpu")
    bin_path = args.config_dir / "pytorch_model.bin"
    if not bin_path.is_file():
        raise FileNotFoundError("Original revision .bin is required to verify provenance")
    bin_weights = torch.load(bin_path, map_location="cpu", weights_only=True)
    if set(bin_weights) != set(original):
        raise AssertionError("Original .bin and safetensors have different parameter keys")
    different = [name for name in original if not torch.equal(original[name], bin_weights[name])]
    if different:
        raise AssertionError(f"Original .bin and safetensors differ: {different}")
    del bin_weights

    model = VideoMAEForVideoClassification(config).cpu().eval()
    expected = model.state_dict()
    converted = dict(original)
    changes = []
    for layer in range(config.num_hidden_layers):
        prefix = f"videomae.encoder.layer.{layer}.attention.attention."
        old_q, old_v = prefix + "q_bias", prefix + "v_bias"
        if old_q not in converted or old_v not in converted:
            raise AssertionError(f"Legacy q/v biases missing in layer {layer}")
        if old_q in expected and old_v in expected:
            continue
        query, key, value = (prefix + part + ".bias" for part in ("query", "key", "value"))
        if any(name not in expected for name in (query, key, value)):
            raise AssertionError("Unsupported installed attention bias layout")
        converted[query] = converted.pop(old_q)
        converted[value] = converted.pop(old_v)
        converted[key] = torch.zeros_like(converted[query])
        changes.append({"layer": layer, "query_from": old_q, "value_from": old_v,
                        "key_bias": "explicit_zero"})
    model.load_state_dict(converted, strict=True)
    for name, tensor in model.state_dict().items():
        if not torch.equal(tensor, converted[name]):
            raise AssertionError(f"Loaded parameter mismatch: {name}")

    generator = torch.Generator().manual_seed(6008)
    fixture = torch.randn(1, 5, config.hidden_size, generator=generator)
    max_error = 0.0
    with torch.inference_mode():
        for layer in range(config.num_hidden_layers):
            prefix = f"videomae.encoder.layer.{layer}.attention.attention."
            for part, old_bias in (("query", "q_bias"), ("key", None), ("value", "v_bias")):
                bias = original[prefix + old_bias] if old_bias else None
                reference = torch.nn.functional.linear(fixture, original[prefix + part + ".weight"], bias)
                actual = torch.nn.functional.linear(fixture, converted[prefix + part + ".weight"],
                                                    converted.get(prefix + part + ".bias", bias))
                error = float((reference - actual).abs().max())
                max_error = max(max_error, error)
    if max_error > 1e-6:
        raise AssertionError(f"Bias conversion changes projection outputs: {max_error}")
    forward_report = None
    if args.forward_smoke:
        from transformers import VideoMAEImageProcessor

        model.requires_grad_(False)
        processor = VideoMAEImageProcessor.from_pretrained(args.config_dir, local_files_only=True)
        pixels = torch.randint(0, 256, (config.num_frames, 3, 224, 224),
                               dtype=torch.uint8, generator=generator)
        inputs = processor([list(pixels)], return_tensors="pt")
        with torch.inference_mode():
            tokens = model.videomae(**inputs).last_hidden_state
            repeated = model.videomae(**inputs).last_hidden_state
        time_slots = config.num_frames // config.tubelet_size
        side = config.image_size // config.patch_size
        expected_shape = (1, time_slots * side * side, config.hidden_size)
        if tuple(tokens.shape) != expected_shape or not torch.isfinite(tokens).all():
            raise AssertionError(f"Encoder output shape/finite failure: {tokens.shape}")
        repeat_error = float((tokens - repeated).abs().max())
        if repeat_error > 1e-6:
            raise AssertionError(f"Encoder repeatability failure: {repeat_error}")
        if side % 2:
            raise AssertionError("2x2 spatial pooling requires an even patch grid")
        grid = tokens.reshape(1, time_slots, side, side, config.hidden_size)
        half = side // 2
        quadrants = torch.stack([
            grid[:, :, row:row + half, col:col + half].mean(dim=(2, 3))
            for row in (0, half) for col in (0, half)
        ], dim=2)
        pool_error = float((quadrants.mean(dim=(1, 2)) - tokens.mean(dim=1)).abs().max())
        if pool_error > 1e-5:
            raise AssertionError(f"Spatial pooling order failure: {pool_error}")
        forward_report = {
            "fixture": "synthetic_uint8_RGB; official local image processor",
            "input_shape": list(inputs["pixel_values"].shape),
            "token_shape": list(tokens.shape),
            "native_time_space_grid_shape": list(quadrants.shape),
            "pooled_shape": list(tokens.mean(dim=1).shape),
            "repeat_max_abs_error": repeat_error,
            "quadrant_vs_global_mean_max_abs_error": pool_error,
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        }
    print(json.dumps({
        "status": "PASS_STRICT_CPU_FORWARD" if args.forward_smoke else "PASS_STRICT_CPU_LOAD",
        "config_sha256": digest(args.config_dir / "config.json"),
        "original_bin_sha256": digest(bin_path),
        "safetensors_sha256": digest(args.weights),
        "original_and_safetensors_tensors_identical": True,
        "converted_layers": changes,
        "strict_load": True,
        "projection_max_abs_error": max_error,
        "full_encoder_forward_tested": args.forward_smoke,
        "pretrained_input_frames": pretrained_frames,
        "effective_input_frames": args.input_frames,
        "position_encoding_policy": "rebuild fixed sinusoidal positions from effective frame count",
        "forward_report": forward_report,
        "gpu_used": False,
        "final_test_access": 0,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
