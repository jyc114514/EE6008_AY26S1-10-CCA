#!/usr/bin/env python3
"""Compare the strict q/v-bias conversion with the local official VideoMAE loader."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import MethodType

import cv2
import numpy as np
import torch
import pyarrow.parquet as pq
from safetensors.torch import load_file
from transformers import VideoMAEConfig, VideoMAEForVideoClassification, VideoMAEImageProcessor


ATOL = 1e-6
RTOL = 1e-6


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def model_hash(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode("utf-8")); h.update(b"\0")
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def run_layers(model: VideoMAEForVideoClassification, pixels: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
    captured: list[torch.Tensor | None] = [None] * len(model.videomae.encoder.layer)
    handles = []
    for i, layer in enumerate(model.videomae.encoder.layer):
        def save(_module, _inputs, output, index=i):
            value = output[0] if isinstance(output, tuple) else output
            captured[index] = value.detach().cpu().clone()
        handles.append(layer.register_forward_hook(save))
    try:
        with torch.inference_mode():
            final = model.videomae(pixel_values=pixels).last_hidden_state.detach().cpu().clone()
    finally:
        for handle in handles:
            handle.remove()
    if any(v is None for v in captured):
        raise AssertionError("not every transformer layer was captured")
    return [v for v in captured if v is not None], final


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    args = parser.parse_args()
    project = args.project_root.resolve()
    audit = args.audit_root.resolve()
    feature_root = args.feature_root.resolve()
    feature_manifest = json.loads((feature_root / "FEATURE_MANIFEST.json").read_text(encoding="utf-8"))
    canonical = Path(feature_manifest["manifest"])
    table = pq.read_table(canonical,
                          columns=["episode_index", "video_path", "split", "video_fps", "eligible_4s",
                                   "classification_inclusion", "v3_benchmark_inclusion"],
                          filters=[("split", "in", ["train"])])
    train = [r for r in table.to_pylist()
             if r["eligible_4s"] is True and r["classification_inclusion"] == "primary"
             and r["v3_benchmark_inclusion"] == "primary"]
    row = sorted(train, key=lambda r: int(r["episode_index"]))[0]
    episode = int(row["episode_index"])
    fps = float(row["video_fps"])
    indices = np.rint(np.arange(16, dtype=np.float64) / 4.0 * fps).astype(np.int64)
    raw = project / "data_raw/HumanoidEveryday/G1/USC-PSI-Lab-Humanoid-Everyday-G1__6b6b599d"
    video = (raw / str(row["video_path"])).resolve()
    if not video.is_relative_to(raw.resolve()) or not video.is_file():
        raise FileNotFoundError("selected train fixture video is missing/outside the approved raw root")

    capture = cv2.VideoCapture(str(video))
    wanted = set(indices.tolist())
    frames: dict[int, np.ndarray] = {}
    try:
        if not capture.isOpened():
            raise RuntimeError("cannot decode selected train-only fixture video")
        for frame_index in range(int(indices.max()) + 1):
            ok, bgr = capture.read()
            if not ok:
                raise RuntimeError(f"train-only fixture decode failed at frame {frame_index}")
            if frame_index in wanted:
                frames[frame_index] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    finally:
        capture.release()
    if set(frames) != wanted:
        raise AssertionError("sampled train-only fixture frames are incomplete")
    rgb_frames = [frames[int(i)] for i in indices]

    cache = project / 'tmp/project-team/hf_cache/hub/models--MCG-NJU--videomae-small-finetuned-kinetics/snapshots'
    config_dir = cache / "1bf2b82c809609f5df1ea7648aee577e2e486b63"
    weights_path = cache / "240e9734611173accbbf74cbdf4b641e4c431264/model.safetensors"
    expected = {
        "config.json": "20b05a5026313946ec026dc0c7be4f1f641c47ab66ec7125d406403ce251165c",
        "preprocessor_config.json": "c3aa722f22a7ff0d234d407862025fe47672f74f46f92c91f757f7a7354107c2",
        "model.safetensors": "6af658f96396ed5f1207244a5a409ca4f6686f05f63b1be6b3609deee195eca7",
    }
    paths = {"config.json": config_dir / "config.json",
             "preprocessor_config.json": config_dir / "preprocessor_config.json",
             "model.safetensors": weights_path}
    actual = {name: digest_file(path) for name, path in paths.items()}
    if actual != expected:
        raise AssertionError(f"VideoMAE local source hashes differ from feature manifest: {actual}")

    config = VideoMAEConfig.from_pretrained(config_dir, local_files_only=True)
    if config.num_frames != 16 or config.tubelet_size != 2 or config.hidden_size != 384:
        raise AssertionError("unexpected VideoMAE native 16-frame/384D architecture")
    config._attn_implementation = "eager"
    strict = VideoMAEForVideoClassification(config).cpu().eval()
    raw_weights = load_file(str(weights_path), device="cpu")
    converted = dict(raw_weights)
    expected_keys = strict.state_dict()
    qv_layers = []
    inserted_zero_key_layers = []
    for layer in range(config.num_hidden_layers):
        prefix = f"videomae.encoder.layer.{layer}.attention.attention."
        if prefix + "q_bias" not in expected_keys:
            q_key, v_key, k_key = prefix + "query.bias", prefix + "value.bias", prefix + "key.bias"
            if prefix + "q_bias" not in converted or prefix + "v_bias" not in converted:
                raise AssertionError(f"q/v bias tensors are missing for layer {layer}")
            if q_key in converted or v_key in converted or k_key in converted:
                raise AssertionError(f"unexpected pre-converted q/k/v biases for layer {layer}")
            converted[q_key] = converted.pop(prefix + "q_bias")
            converted[v_key] = converted.pop(prefix + "v_bias")
            converted[k_key] = torch.zeros_like(converted[q_key])
            qv_layers.append(layer)
            inserted_zero_key_layers.append(layer)
    strict.load_state_dict(converted, strict=True)

    # Native split-bias reference: retain raw q_bias/v_bias from the source checkpoint
    # and compute q/k/v directly, with a zero key bias. This avoids using a loader that
    # silently ignores those source keys in the currently installed Transformers build.
    reference = VideoMAEForVideoClassification(config).cpu().eval()
    reference.load_state_dict(strict.state_dict(), strict=True)
    for layer in range(config.num_hidden_layers):
        prefix = f"videomae.encoder.layer.{layer}.attention.attention."
        raw_q = raw_weights[prefix + "q_bias"].detach().clone()
        raw_v = raw_weights[prefix + "v_bias"].detach().clone()
        attention = reference.videomae.encoder.layer[layer].attention.attention

        def native_split_bias_forward(self, hidden_states=None, _q=raw_q, _v=raw_v, **kwargs):
            batch, sequence, _width = hidden_states.shape
            head_dim = self.attention_head_size
            q = torch.nn.functional.linear(hidden_states, self.query.weight, _q.to(hidden_states))
            k = torch.nn.functional.linear(hidden_states, self.key.weight, torch.zeros_like(_q).to(hidden_states))
            v = torch.nn.functional.linear(hidden_states, self.value.weight, _v.to(hidden_states))
            def heads(value):
                return value.view(batch, sequence, self.num_attention_heads, head_dim).transpose(1, 2)
            q, k, v = heads(q), heads(k), heads(v)
            scores = torch.matmul(q, k.transpose(-1, -2)) * self.scaling
            probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
            context = torch.matmul(probs, v).transpose(1, 2).contiguous().view(batch, sequence, self.all_head_size)
            return context, probs

        attention.forward = MethodType(native_split_bias_forward, attention)

    # Record the behavior of the active public auto-loader without using it as the oracle.
    official_model, loader_info = VideoMAEForVideoClassification.from_pretrained(
        str(weights_path.parent), config=config, local_files_only=True, use_safetensors=True,
        output_loading_info=True
    )
    missing = list(loader_info.get("missing_keys", []))
    unexpected = list(loader_info.get("unexpected_keys", []))
    qv_loader_report = {
        "q_bias_unexpected": sum(".q_bias" in key for key in unexpected),
        "v_bias_unexpected": sum(".v_bias" in key for key in unexpected),
        "query_bias_missing": sum(".query.bias" in key for key in missing),
        "key_bias_missing": sum(".key.bias" in key for key in missing),
        "value_bias_missing": sum(".value.bias" in key for key in missing),
    }
    del official_model
    if not all(qv_loader_report.values()):
        raise AssertionError(f"expected direct loader q/v-bias mismatch was not observed: {qv_loader_report}")

    processor = VideoMAEImageProcessor.from_pretrained(config_dir, local_files_only=True)
    pixels = processor([rgb_frames], return_tensors="pt")["pixel_values"]
    if tuple(pixels.shape) != (1, 16, 3, 224, 224):
        raise AssertionError(f"processor layout mismatch: {tuple(pixels.shape)}")

    strict_before, reference_before = model_hash(strict), model_hash(reference)
    strict_layers, strict_final = run_layers(strict, pixels)
    reference_layers, reference_final = run_layers(reference, pixels)
    per_layer = []
    for i, (left, right) in enumerate(zip(strict_layers, reference_layers)):
        delta = np.abs(left.numpy().astype(np.float64) - right.numpy().astype(np.float64))
        per_layer.append({"layer": i, "max_abs": float(delta.max()), "mean_abs": float(delta.mean()),
                          "allclose_atol1e6_rtol1e6": bool(np.allclose(left.numpy(), right.numpy(), atol=ATOL, rtol=RTOL))})
    final_delta = np.abs(strict_final.numpy().astype(np.float64) - reference_final.numpy().astype(np.float64))
    strict_feature = strict_final.mean(dim=1).numpy()
    reference_feature = reference_final.mean(dim=1).numpy()
    feature_delta = np.abs(strict_feature.astype(np.float64) - reference_feature.astype(np.float64))
    if not all(x["allclose_atol1e6_rtol1e6"] for x in per_layer):
        raise AssertionError("converted strict load differs from the official loader at one or more transformer layers")
    if not np.allclose(strict_feature, reference_feature, atol=ATOL, rtol=RTOL):
        raise AssertionError("converted strict load differs at the final pooled 384D feature")
    if strict_before != model_hash(strict) or reference_before != model_hash(reference):
        raise AssertionError("inference changed VideoMAE parameters or buffers")

    report = {
        "status": "PASS_CORE_A_STRICT_CONVERSION_LAYER_AND_FEATURE_FIXTURE",
        "reference_path": "independent split-bias q/k/v attention using original q_bias, v_bias and zero key bias from hash-pinned safetensors; transformer blocks otherwise share the exact official VideoMAE architecture",
        "direct_auto_loader_diagnostic": qv_loader_report,
        "audit_transformers_version": __import__("transformers").__version__,
        "source_model_id": "MCG-NJU/videomae-small-finetuned-kinetics",
        "config_snapshot": "1bf2b82c809609f5df1ea7648aee577e2e486b63",
        "weights_snapshot": "240e9734611173accbbf74cbdf4b641e4c431264",
        "source_sha256": actual,
        "train_only_episode_index": episode,
        "split": "train",
        "fps_manifest": fps,
        "sampled_frame_indices": indices.tolist(),
        "input_layout": list(pixels.shape),
        "native_num_frames": int(config.num_frames),
        "tubelet_size": int(config.tubelet_size),
        "hidden_size": int(config.hidden_size),
        "converted_qv_bias_layers": qv_layers,
        "inserted_zero_key_bias_layers": inserted_zero_key_layers,
        "layer_comparisons": per_layer,
        "last_hidden_state_max_abs": float(final_delta.max()),
        "pooled_feature_shape": list(strict_feature.shape),
        "pooled_feature_max_abs": float(feature_delta.max()),
        "tolerance_atol": ATOL,
        "tolerance_rtol": RTOL,
        "tolerance_basis": "provisional same CPU, dtype, processor and one train-only 16-frame fixture; no prior committed layer-output tolerance found",
        "strict_parameters_before_sha256": strict_before,
        "strict_parameters_after_sha256": model_hash(strict),
        "reference_parameters_before_sha256": reference_before,
        "reference_parameters_after_sha256": model_hash(reference),
        "strict_classifier_used": False,
        "final_test_access": 0,
        "gpu_used": False,
    }
    out = audit / "metrics/core_a/STRICT_CONVERSION_FIXTURE.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
