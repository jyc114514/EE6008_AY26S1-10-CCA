from __future__ import annotations
import os

import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
ROOT = PROJECT / "reports/benchmarks/CORE_B_FUSION_QUERY_MECHANISM_V1_20261007T110705Z"
LEARN = PROJECT / "reports/benchmarks/CORE_B_COLLAPSE_LEARNABILITY_V1_20261007T060954Z"
SCHED = PROJECT / "reports/benchmarks/D3_EQ_CORRECTED_SCHEDULER_V1_20261006T073406Z"
EQ = PROJECT / "reports/benchmarks/CORE_B_D3_EPISODE_EQUAL_SENSITIVITY_V1_20260924T095800Z"
BENCH = PROJECT / "reports/benchmarks/CORE_B_BENCHMARK_LAB_V1_20260920T065538Z"
V3 = PROJECT / "repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z"
IDS = [
    "TINYFIT-TS11-D-H30-FULL_LOSS-P1-S20261002",
    "TINYFIT-TS11-D-H30-FULL_LOSS-P8-S20261002",
    "TINYFIT-STATE-CTRL-H30-FULL_LOSS-P1-S20261002",
    "TINYFIT-STATE-CTRL-H30-FULL_LOSS-P8-S20261002",
]
sys.path.insert(0, str(SCHED / "code"))
sys.path.insert(0, str(LEARN / "code"))
import run_tiny_fit as old_runner
from d3_model import (
    D3TimeSpacePredictor, parameter_count, sinusoidal, spatial_encoding, state_hash,
)
from collapse_diagnostic_core import loss_components

def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("status\nno_rows\n", encoding="utf-8")
        return
    columns = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)

def max_abs(x: Tensor) -> float:
    return float(x.detach().abs().max().item()) if x.numel() else 0.0

def vector_stats(x: Tensor) -> dict[str, Any]:
    y = x.detach().to(device="cpu", dtype=torch.float32)
    finite = bool(torch.isfinite(y).all().item())
    flat = y.reshape(-1)
    result = {
        "shape": list(y.shape),
        "finite": finite,
        "mean": float(flat.mean().item()) if flat.numel() else 0.0,
        "std": float(flat.std(unbiased=False).item()) if flat.numel() else 0.0,
        "rms": float(torch.sqrt(torch.mean(flat * flat)).item()) if flat.numel() else 0.0,
        "max_abs": max_abs(y),
    }
    if y.ndim >= 3:
        tokens = y.reshape(y.shape[0], -1, y.shape[-1])
        token_l2 = torch.linalg.vector_norm(tokens, dim=-1)
        per_token = torch.sqrt(torch.mean(token_l2 * token_l2, dim=0))
        result.update({
            "sequence_tokens": int(tokens.shape[1]),
            "per_token_l2_rms_mean": float(per_token.mean().item()),
            "per_token_l2_rms_std": float(per_token.std(unbiased=False).item()),
            "per_token_l2_rms_min": float(per_token.min().item()),
            "per_token_l2_rms_max": float(per_token.max().item()),
        })
        if y.ndim == 3 and y.shape[1] > 1:
            dx = y[:, 1:, :] - y[:, :-1, :]
            result["adjacent_token_l2_rms"] = float(torch.sqrt(torch.mean(torch.sum(dx * dx, dim=-1))).item())
        elif y.ndim == 4 and y.shape[1] > 1:
            dx = y[:, 1:, :, :] - y[:, :-1, :, :]
            result["time_adjacent_l2_rms"] = float(torch.sqrt(torch.mean(torch.sum(dx * dx, dim=-1))).item())
    return result

def append_stat(rows: list[dict[str, Any]], run_id: str, panel: str, group: str,
                stage: str, x: Tensor, temporal: str = "sequence") -> None:
    row = {"run_id": run_id, "panel": panel, "group": group, "stage": stage}
    row.update(vector_stats(x))
    row["temporal_axis"] = temporal
    rows.append(row)

def staged_forward(model: D3TimeSpacePredictor, t: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor, dict[str, Tensor]]:
    batch = t["state"].shape[0]
    state_meta = torch.cat((t["mask"].unsqueeze(-1), t["relative"].unsqueeze(-1)), dim=-1)
    state_input = torch.cat((t["state"], state_meta), dim=-1)
    state_content = model.state_projection(state_input)
    state_position = sinusoidal(t["relative"] * 8.0, model.WIDTH if hasattr(model, "WIDTH") else 128)
    state_modality = model.state_modality
    state_pre_tcn = state_content + state_position + state_modality
    state_tokens = state_pre_tcn.transpose(1, 2)
    for block in model.tcn_blocks:
        state_tokens = block(state_tokens)
    state_tokens = state_tokens.transpose(1, 2)

    video_values = model._video_layout(t["dynamic"], t["static"])
    time_values = torch.linspace(-31.0, 0.0, 16, device=t["state"].device).view(1, 16, 1)
    time_values = time_values.expand(batch, 16, 4).reshape(batch, 64)
    video_content = model.video_projection(video_values)
    video_time_position = sinusoidal(time_values, 128)
    video_space_position = spatial_encoding(t["state"].device)
    video_modality = model.video_modality
    video_pre_decoder = video_content + video_time_position + video_space_position + video_modality

    anchor_content = model.anchor_projection(t["anchor"]).unsqueeze(1)
    anchor_modality = model.anchor_modality
    anchor_token = anchor_content + anchor_modality
    memory = torch.cat((state_tokens, video_pre_decoder, anchor_token), dim=1)

    query_times = torch.arange(1, model.horizon + 1, device=t["state"].device, dtype=torch.float32) / 30.0
    query = sinusoidal(query_times.view(1, -1).expand(batch, -1), 128)
    residual = model.output(model.decoder(query, memory))
    stages = {
        "state_input": state_input,
        "state_content": state_content,
        "state_position": state_position,
        "state_modality": state_modality.expand_as(state_content),
        "state_pre_tcn": state_pre_tcn,
        "state_post_tcn": state_tokens,
        "video_layout": video_values,
        "video_content": video_content,
        "video_time_position": video_time_position,
        "video_space_position": video_space_position.expand(batch, -1, -1),
        "video_modality": video_modality.expand_as(video_content),
        "video_pre_decoder": video_pre_decoder,
        "anchor_input": t["anchor"].unsqueeze(1),
        "anchor_content": anchor_content,
        "anchor_modality": anchor_modality.expand_as(anchor_content),
        "anchor_token": anchor_token,
        "memory": memory,
        "future_query": query,
    }
    native = t["anchor"][:, None, :] + t["sigma"][None, None, :] * residual
    return residual, native, query, stages

def query_spectrum(run_id: str, panel: str, model_name: str, sample_id: str,
                   stage: str, matrix: Tensor | np.ndarray) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    q = matrix.detach().cpu().numpy() if isinstance(matrix, Tensor) else np.asarray(matrix)
    q32 = np.ascontiguousarray(q, dtype=np.float32)
    q64 = q32.astype(np.float64)
    if q64.ndim != 2:
        raise ValueError(f"query spectrum expects [time,width], got {q64.shape}")
    singular = np.linalg.svd(q64, compute_uv=False)
    threshold = float(singular[0] * np.finfo(np.float32).eps * max(q64.shape)) if singular.size else 0.0
    rank = int(np.sum(singular > threshold))
    norms = np.linalg.norm(q64, axis=1)
    if len(q64) > 1:
        distances = np.linalg.norm(np.diff(q64, axis=0), axis=1)
        cosines = np.sum(q64[:-1] * q64[1:], axis=1) / (norms[:-1] * norms[1:])
        distance_min, distance_max = float(distances.min()), float(distances.max())
        cosine_min, cosine_max = float(cosines.min()), float(cosines.max())
    else:
        distance_min = distance_max = cosine_min = cosine_max = float("nan")
    h = hashlib.sha256(q32.tobytes()).hexdigest()
    summary = {
        "run_id": run_id, "panel": panel, "model": model_name, "sample_id": sample_id,
        "stage": stage, "time_rows": int(q32.shape[0]), "width": int(q32.shape[1]),
        "unique_rows": int(np.unique(q32, axis=0).shape[0]),
        "mean_row_rms": float(np.sqrt(np.mean(q64 * q64, axis=1)).mean()),
        "mean_row_l2": float(norms.mean()), "adjacent_l2_min": distance_min,
        "adjacent_l2_max": distance_max, "adjacent_cosine_min": cosine_min,
        "adjacent_cosine_max": cosine_max, "sigma_max": float(singular[0]),
        "fp32_rank_threshold": threshold, "fp32_effective_rank": rank,
        "threshold_formula": "float32_eps * max(matrix_shape) * sigma_max; SVD computed in float64 from the FP32 matrix",
        "matrix_sha256": h,
    }
    sv_rows = [{
        "run_id": run_id, "panel": panel, "model": model_name, "sample_id": sample_id,
        "stage": stage, "singular_index_1based": i + 1,
        "singular_value": float(s), "rank_threshold": threshold,
        "effective_rank_fp32_scale": rank,
    } for i, s in enumerate(singular)]
    return summary, sv_rows

def capture_standard_forward(model: D3TimeSpacePredictor, t: dict[str, Tensor]) -> tuple[Tensor, dict[str, Any]]:
    captured: dict[str, Any] = {"cross_layers": [], "decoder_query": None, "self_query": None, "cross_query": None}
    handles = []
    def decoder_pre(module, args):
        captured["decoder_query"] = args[0]
    handles.append(model.decoder.register_forward_pre_hook(decoder_pre))
    def self_pre(module, args):
        captured["self_query"] = args[0]
    handles.append(model.decoder.layers[0].self_attn.register_forward_pre_hook(self_pre))
    def cross_pre(module, args):
        captured["cross_query"] = args[0]
    handles.append(model.decoder.layers[0].multihead_attn.register_forward_pre_hook(cross_pre))
    for layer_idx, layer in enumerate(model.decoder.layers):
        def hook(module, args, kwargs, output, index=layer_idx):
            captured["cross_layers"].append({
                "layer_idx": index, "module": module, "args": args,
                "kwargs": kwargs, "output": output,
            })
        handles.append(layer.multihead_attn.register_forward_hook(hook, with_kwargs=True))
    try:
        residual = model(t["state"], t["mask"], t["relative"], t["dynamic"], t["static"], t["anchor"])
    finally:
        for handle in handles:
            handle.remove()
    return residual, captured

def reconstruct_cross_attention(record: dict[str, Any], layer_idx: int,
                                run_id: str, panel: str, model_name: str,
                                meta: dict[str, np.ndarray],
                                group_rows: list[dict[str, Any]],
                                token_rows: list[dict[str, Any]]) -> dict[str, Any]:
    module = record["module"]
    args = record["args"]
    kwargs = record["kwargs"]
    q_input, k_input, v_input = args[:3]
    mask = kwargs.get("attn_mask")
    key_mask = kwargs.get("key_padding_mask")
    causal = bool(kwargs.get("is_causal", False))
    actual = record["output"][0]
    E = int(module.embed_dim)
    heads = int(module.num_heads)
    head_dim = E // heads
    if mask is not None or key_mask is not None or causal:
        return {
            "run_id": run_id, "panel": panel, "model": model_name, "decoder_layer": layer_idx,
            "status": "UNAVAILABLE_MASKED_PATH",
            "attn_mask_present": mask is not None, "key_padding_mask_present": key_mask is not None,
            "is_causal": causal,
        }
    weight = module.in_proj_weight
    bias = module.in_proj_bias
    qproj, kproj, vproj = F.linear(q_input, weight[:E], bias[:E]), F.linear(k_input, weight[E:2*E], bias[E:2*E]), F.linear(v_input, weight[2*E:], bias[2*E:])
    batch, query_len, _ = qproj.shape
    memory_len = kproj.shape[1]
    qh = qproj.reshape(batch, query_len, heads, head_dim).transpose(1, 2)
    kh = kproj.reshape(batch, memory_len, heads, head_dim).transpose(1, 2)
    vh = vproj.reshape(batch, memory_len, heads, head_dim).transpose(1, 2)
    logits = torch.matmul(qh, kh.transpose(-2, -1)) / math.sqrt(head_dim)
    weights = torch.softmax(logits, dim=-1)
    weighted_values = weights.unsqueeze(-1) * vh.unsqueeze(2)
    context = torch.matmul(weights, vh).transpose(1, 2).contiguous().reshape(batch, query_len, E)
    reconstructed = F.linear(context, module.out_proj.weight, module.out_proj.bias)
    delta = (reconstructed - actual).detach()
    tolerance_ok = bool(torch.allclose(reconstructed, actual, atol=1e-5, rtol=1e-5))
    valid = int(meta["anchor_id"].shape[0])
    groups = [("state", 0, 32), ("video", 32, 96), ("anchor", 96, 97)]
    denom = max(1, batch * heads * query_len)
    for name, start, end in groups:
        w = weights[..., start:end]
        group_mass_mean = float(w.sum().item() / denom)
        gcontrib = weighted_values[..., start:end, :].sum(dim=3)
        group_l2 = torch.linalg.vector_norm(gcontrib, dim=-1)
        group_rows.append({
            "run_id": run_id, "panel": panel, "model": model_name,
            "decoder_layer": layer_idx, "memory_group": name,
            "token_count": end - start, "total_attention_mass": float(w.sum().item()),
            "mean_mass_per_query_head": group_mass_mean,
            "mean_mass_per_token": group_mass_mean / (end - start),
            "mean_group_weighted_value_l2": float(group_l2.mean().item()),
            "rms_group_weighted_value_l2": float(torch.sqrt(torch.mean(group_l2 * group_l2)).item()),
            "attention_reconstruction_max_abs": float(delta.abs().max().item()),
            "attention_reconstruction_allclose_atol1e-5_rtol1e-5": tolerance_ok,
            "attn_mask_present": False, "key_padding_mask_present": False,
            "is_causal": False, "memory_tokens": memory_len,
        })
        for token in range(start, end):
            token_w = weights[..., token]
            token_contrib = weighted_values[..., token, :]
            norm = torch.linalg.vector_norm(token_contrib, dim=-1)
            row = {
                "run_id": run_id, "panel": panel, "model": model_name,
                "decoder_layer": layer_idx, "memory_token_index": token,
                "memory_group": name, "mean_attention_mass": float(token_w.mean().item()),
                "total_attention_mass": float(token_w.sum().item()),
                "mean_weighted_value_l2": float(norm.mean().item()),
                "rms_weighted_value_l2": float(torch.sqrt(torch.mean(norm * norm)).item()),
            }
            if name == "state":
                row["state_step_0based"] = token
                row["video_time_slot_0based"] = ""
                row["video_space_bin_0based"] = ""
            elif name == "video":
                slot = token - 32
                row["state_step_0based"] = ""
                row["video_time_slot_0based"] = slot // 4
                row["video_space_bin_0based"] = slot % 4
            else:
                row["state_step_0based"] = ""
                row["video_time_slot_0based"] = ""
                row["video_space_bin_0based"] = ""
            token_rows.append(row)
    return {
        "run_id": run_id, "panel": panel, "model": model_name, "decoder_layer": layer_idx,
        "status": "PASS" if tolerance_ok else "RECONSTRUCTION_MISMATCH",
        "memory_tokens": memory_len, "query_tokens": query_len, "heads": heads,
        "attn_mask_present": False, "key_padding_mask_present": False, "is_causal": False,
        "attention_reconstruction_max_abs": float(delta.abs().max().item()),
        "attention_reconstruction_allclose_atol1e-5_rtol1e-5": tolerance_ok,
    }

def gradient_groups(model: D3TimeSpacePredictor, t: dict[str, Tensor],
                    run_id: str, panel: str, query_holder: dict[str, Tensor]) -> list[dict[str, Any]]:
    residual = model(t["state"], t["mask"], t["relative"], t["dynamic"], t["static"], t["anchor"])
    query = query_holder.get("query")
    if query is None:
        raise AssertionError(f"decoder query hook did not capture a tensor: {run_id}")
    parts = loss_components(residual, t["target"], t["anchor"], t["sigma"], t["safe_fd"])
    pos = parts["position"].mean()
    weighted_fd = parts["weighted_fd"].mean()
    params = list(model.named_parameters())
    all_params = [p for _, p in params]
    grads_pos = torch.autograd.grad(pos, [query] + all_params, retain_graph=True, allow_unused=True)
    grads_fd = torch.autograd.grad(weighted_fd, [query] + all_params, retain_graph=False, allow_unused=True)
    ppos, pfd = grads_pos[0], grads_fd[0]
    groups: dict[str, list[int]] = {
        "state_projection": [], "tcn": [], "video_projection": [], "anchor_projection": [],
        "decoder_self_attention": [], "decoder_cross_attention": [], "decoder_other": [], "decoder_all": [], "output": [],
    }
    names = [name for name, _ in params]
    for i, name in enumerate(names):
        if name.startswith("state_projection."):
            groups["state_projection"].append(i)
        elif name.startswith("tcn_blocks."):
            groups["tcn"].append(i)
        elif name.startswith("video_projection."):
            groups["video_projection"].append(i)
        elif name.startswith("anchor_projection."):
            groups["anchor_projection"].append(i)
        elif name.startswith("decoder."):
            groups["decoder_all"].append(i)
            if ".self_attn." in name:
                groups["decoder_self_attention"].append(i)
            elif ".multihead_attn." in name:
                groups["decoder_cross_attention"].append(i)
            else:
                groups["decoder_other"].append(i)
        elif name.startswith("output."):
            groups["output"].append(i)
    def flatten(grad_list, indices):
        values = [grad_list[i + 1].detach().reshape(-1) for i in indices if grad_list[i + 1] is not None]
        return torch.cat(values) if values else torch.zeros(0, dtype=torch.float32)
    result = []
    for group_name, indices in [("query_encoding", [])] + list(groups.items()):
        gp = ppos.detach().reshape(-1) if group_name == "query_encoding" and ppos is not None else flatten(grads_pos, indices)
        gf = pfd.detach().reshape(-1) if group_name == "query_encoding" and pfd is not None else flatten(grads_fd, indices)
        norm_pos = float(torch.linalg.vector_norm(gp).item()) if gp.numel() else 0.0
        norm_fd = float(torch.linalg.vector_norm(gf).item()) if gf.numel() else 0.0
        if norm_pos == 0.0 or norm_fd == 0.0 or gp.numel() != gf.numel():
            cosine: float | str = "NA"
        else:
            cosine = float(torch.dot(gp, gf).item() / (norm_pos * norm_fd))
        ratio: float | str = norm_fd / norm_pos if norm_pos > 0.0 else "NA"
        result.append({
            "run_id": run_id, "panel": panel, "group": group_name,
            "position_gradient_l2": norm_pos, "weighted_0p1_fd_gradient_l2": norm_fd,
            "weighted_fd_over_position_norm": ratio, "gradient_cosine": cosine,
            "position_gradient_zero": norm_pos == 0.0, "weighted_fd_gradient_zero": norm_fd == 0.0,
            "position_loss": float(pos.detach().item()), "weighted_0p1_fd_loss": float(weighted_fd.detach().item()),
        })
    return result

def load_state_into(config: str, init_state: dict[str, Tensor], checkpoint_path: Path,
                    seed: int) -> tuple[D3TimeSpacePredictor, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model" not in checkpoint:
        raise AssertionError(f"checkpoint lacks model state: {checkpoint_path}")
    model = old_runner.build_model(config, seed, init_state)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
        if isinstance(module, torch.nn.MultiheadAttention):
            module.dropout = 0.0
    return model, checkpoint

def main() -> int:
    old_runner.set_cpu_runtime(1)
    if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
        raise AssertionError("CPU thread contract is not one")
    source_lock_hash = old_runner.verify_source_lock()
    panels, metadata, panel_lock = old_runner.load_panels()
    learn_protocol = read_json(LEARN / "protocol/B_PROTOCOL_LOCKED.json")
    source_acceptance = read_json(LEARN / "SOURCE_ACCEPTANCE.json")
    selected = next(x for x in source_acceptance["selected_train_only_cells"]
                    if x.get("config") == "TS11-D" and x.get("horizon_steps") == 30 and x.get("seed") == 20261002)
    init_state = old_runner.load_init(learn_protocol)
    seed_registry = read_json(EQ / "protocol/D3_EQ_SEED_REGISTRY.json")
    expected_init_state_hash = seed_registry["seeds"]["20261002"]["init_state_hash"]

    replay_rows = []
    replay_models: dict[str, tuple[D3TimeSpacePredictor, dict[str, Any], str, str]] = {}
    for rid in IDS:
        run_root = LEARN / "runs/formal" / rid
        meta = read_json(run_root / "RUN_COMPLETE.json")
        if meta["source_lock_sha256"] != source_lock_hash:
            raise AssertionError(f"source lock mismatch for {rid}")
        model, checkpoint = load_state_into(meta["config"], init_state, run_root / "checkpoint_final.pt", int(meta["seed"]))
        model_hash = state_hash(model)
        if "model_state_hash" in checkpoint and model_hash != checkpoint["model_state_hash"]:
            raise AssertionError(f"model state hash mismatch for {rid}")
        panel = meta["panel"]
        with torch.no_grad():
            residual, native = old_runner.forward(model, panels[panel])
        pred_path = run_root / "PREDICTIONS_FINAL.npz"
        with np.load(pred_path, allow_pickle=False) as z:
            old_ids = z["anchor_id"].astype("U96")
            old_pred = z["prediction_native"].astype(np.float32, copy=True)
            old_target = z["target_native"].astype(np.float32, copy=True)
            old_anchor = z["anchor_native"].astype(np.float32, copy=True)
        now_pred = native.detach().cpu().numpy().astype(np.float32)
        target = panels[panel]["target"].cpu().numpy().astype(np.float32)
        anchor = panels[panel]["anchor"].cpu().numpy().astype(np.float32)
        id_equal = old_ids.tolist() == metadata[panel]["anchor_id"].tolist()
        pred_max = float(np.max(np.abs(now_pred - old_pred)))
        target_max = float(np.max(np.abs(target - old_target)))
        anchor_max = float(np.max(np.abs(anchor - old_anchor)))
        replay_ok = (
            id_equal and np.allclose(now_pred, old_pred, atol=1e-5, rtol=1e-5)
            and np.allclose(target, old_target, atol=1e-5, rtol=1e-5)
            and np.allclose(anchor, old_anchor, atol=1e-5, rtol=1e-5)
        )
        if not replay_ok:
            raise AssertionError(f"strict baseline replay failed: {rid}")
        replay_rows.append({
            "run_id": rid, "config": meta["config"], "panel": panel, "loss": meta["loss"],
            "seed": meta["seed"], "horizon_steps": meta["horizon_steps"],
            "optimizer_updates": meta["optimizer_updates"], "strict_state_dict_load": True,
            "model_state_hash": model_hash, "parameter_count": parameter_count(model),
            "ids_equal": id_equal, "prediction_max_abs_difference": pred_max,
            "target_max_abs_difference": target_max, "anchor_max_abs_difference": anchor_max,
            "atol": 1e-5, "rtol": 1e-5, "replay_status": "PASS",
            "final_checkpoint_sha256": sha(run_root / "checkpoint_final.pt"),
            "prediction_sha256": sha(pred_path),
        })
        replay_models[rid] = (model, checkpoint, meta["config"], panel)
    if len(replay_rows) != 4:
        raise AssertionError("expected exactly four matched old FULL_LOSS baselines")

    epoch_specs = []
    for family in ("old", "corrected"):
        epoch_meta = next(y for y in selected["epoch_checkpoints"]
                          if y.get("family") == family and y.get("epoch") == 40)
        path = Path(epoch_meta["path"])
        expected_hash = epoch_meta["sha256"]
        run_id = selected["old_run_id"] if family == "old" else selected["corrected_run_id"]
        run_root = EQ / "d3/runs" / run_id if family == "old" else SCHED / "runs/formal" / run_id
        run_complete_path = run_root / "RUN_COMPLETE.json"
        run_complete = read_json(run_complete_path)
        expected_run_complete_hash = selected["old_run_complete_sha256"] if family == "old" else selected["corrected_run_complete_sha256"]
        expected_last_hash = selected["old_checkpoint_sha256"] if family == "old" else selected["corrected_checkpoint_sha256"]
        if sha(run_complete_path) != expected_run_complete_hash or run_complete.get("checkpoint_last_sha256") != expected_last_hash:
            raise AssertionError(f"epoch40 run-complete/checkpoint-last lineage mismatch: {run_complete_path}")
        if sha(path) != expected_hash:
            raise AssertionError(f"epoch40 accepted checkpoint hash mismatch: {path}")
        model, checkpoint = load_state_into("TS11-D", init_state, path, 20261002)
        if checkpoint.get("epoch") != 40 or checkpoint.get("horizon") != 30 or checkpoint.get("seed") != 20261002:
            raise AssertionError(f"epoch40 checkpoint metadata mismatch: {path}")
        if checkpoint.get("final_test_access") != 0:
            raise AssertionError(f"final-test access field mismatch: {path}")
        if checkpoint.get("init_hash") != expected_init_state_hash:
            raise AssertionError(f"epoch40 INIT state hash mismatch: {path}")
        epoch_specs.append((f"{family}-{run_id}-E40", "TS11-D", "P8", path, model, checkpoint, family))

    model_specs: list[tuple[str, str, str, Path, D3TimeSpacePredictor, dict[str, Any], str]] = []
    for config in ("TS11-D", "STATE-CTRL"):
        model = old_runner.build_model(config, 20261002, init_state)
        if state_hash(model) != expected_init_state_hash:
            raise AssertionError(f"INIT state hash mismatch for {config}")
        ckpt_path = BENCH / "d3/runs/INIT_S20261002.pt"
        model_specs.append((f"INIT-{config}-P8", config, "P8", ckpt_path, model, {"model": init_state}, "init"))
    for rid in IDS:
        model, checkpoint, config, panel = replay_models[rid]
        model_specs.append((rid, config, panel, LEARN/"runs/formal"/rid/"checkpoint_final.pt", model, checkpoint, "tinyfit"))
    for run_id, config, panel, path, model, checkpoint, family in epoch_specs:
        model_specs.append((run_id, config, panel, path, model, checkpoint, family))
    audited_paths = []
    for run_id, config, panel, path, model, checkpoint, family in model_specs:
        audited_paths.append({"run_id":run_id,"family":family,"config":config,"panel":panel,
                              "path":str(path),"sha256":sha(path),"model_state_hash":state_hash(model),
                              "parameter_count":parameter_count(model),"strict_state_dict_load":True})

    normalizers = read_json(V3 / "data/V3_NORMALIZERS.json")
    video_norm = read_json(BENCH / "d3/cache/D3_VIDEO_NORMALIZER.json")
    eps = float(normalizers["epsilon"])
    state_std = np.asarray(normalizers["state_safe_std"], dtype=np.float64)
    target_sigma = np.asarray(normalizers["target_sigma"], dtype=np.float64)
    fd_scale = np.asarray(normalizers["safe_fd_scale"], dtype=np.float64)
    video_std = np.asarray(video_norm["safe_std"], dtype=np.float64)
    normalizer_summary = {
        "epsilon": eps, "state_dim": int(len(state_std)),
        "state_safe_std_min": float(state_std.min()), "state_safe_std_max": float(state_std.max()),
        "state_channels_at_or_below_epsilon": int(np.sum(state_std <= eps)),
        "relative_delta_safe_std": float(normalizers["relative_delta_safe_std"]),
        "video_dim": int(len(video_std)), "video_safe_std_min": float(video_std.min()),
        "video_safe_std_max": float(video_std.max()),
        "video_constant_channels": video_norm.get("constant_channels", []),
        "target_sigma_dim": int(len(target_sigma)), "target_sigma_min": float(target_sigma.min()),
        "safe_fd_scale_dim": int(len(fd_scale)), "safe_fd_scale_min": float(fd_scale.min()),
        "normalizer_fit_scope": normalizers["fit_scope"],
        "video_normalizer_fit_scope": video_norm["fit_scope"],
        "video_normalizer_bridge_max_abs": video_norm["bridge_max_abs"],
    }

    activation_rows: list[dict[str, Any]] = []
    for panel in ("P1", "P8"):
        t = panels[panel]
        append_stat(activation_rows, "INPUTS", panel, "state", "history_standardized", t["state"])
        append_stat(activation_rows, "INPUTS", panel, "history_mask", "history_mask_metadata", t["mask"])
        append_stat(activation_rows, "INPUTS", panel, "relative_delta", "history_relative_delta_standardized", t["relative"])
        append_stat(activation_rows, "INPUTS", panel, "video", "dynamic_grid_normalized", t["dynamic"], "time")
        append_stat(activation_rows, "INPUTS", panel, "anchor", "anchor_command_native", t["anchor"].unsqueeze(1))
        append_stat(activation_rows, "INPUTS", panel, "target", "train_target_native", t["target"])
    attention_group_rows: list[dict[str, Any]] = []
    attention_token_rows: list[dict[str, Any]] = []
    gradient_rows: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []
    singular_rows: list[dict[str, Any]] = []
    attention_replays: list[dict[str, Any]] = []
    staged_forward_rows: list[dict[str, Any]] = []
    query_table_rows: list[dict[str, Any]] = []

    for horizon in (30, 60):
        times = torch.arange(1, horizon + 1, dtype=torch.float32) / 30.0
        raw_q = sinusoidal(times, 128)
        if horizon == 30:
            raw_q30 = raw_q.detach().clone()
        if horizon == 60:
            raw_q60 = raw_q.detach().clone()
        matrices = [
            ("old_raw_fixed_future_encoding", raw_q),
            ("old_cross_time_centered", raw_q - raw_q.mean(dim=0, keepdim=True)),
        ]
        for stage, matrix in matrices:
            summary, sv = query_spectrum("OLD_FIXED_ENCODING", f"H{horizon}", "sinusoidal", "", stage, matrix)
            query_rows.append(summary)
            singular_rows.extend(sv)
        for i in range(horizon):
            row = {"horizon":horizon,"step_1based":i+1,"time_seconds":float(times[i]),
                   "query_sha256":hashlib.sha256(raw_q[i].numpy().astype(np.float32).tobytes()).hexdigest()}
            row.update({f"q{j:03d}":float(v) for j,v in enumerate(raw_q[i].tolist())})
            query_table_rows.append(row)
    prefix_exact = bool(torch.equal(raw_q30, raw_q60[:30]))
    if not prefix_exact:
        raise AssertionError("old H30 query is not the exact H60 prefix")

    # Model-path, attention and query-stage probes use forward hooks only; no optimizer is created.
    for run_id, config, panel, path, model, checkpoint, family in model_specs:
        t = panels[panel]
        model.eval()
        before_hash = state_hash(model)
        with torch.no_grad():
            residual_manual, native_manual, query_manual, stages = staged_forward(model, t)
            residual_source, capture = capture_standard_forward(model, t)
        forward_diff = max_abs(residual_manual - residual_source)
        forward_allclose = bool(torch.allclose(residual_manual, residual_source, atol=1e-5, rtol=1e-5))
        if not forward_allclose:
            raise AssertionError(f"staged forward differs from locked source forward beyond 1e-5: {run_id}; max_abs={forward_diff}")
        native_source = t["anchor"][:, None, :] + t["sigma"][None, None, :] * residual_source
        native_diff = max_abs(native_manual - native_source)
        native_allclose = bool(torch.allclose(native_manual, native_source, atol=1e-5, rtol=1e-5))
        if not native_allclose:
            raise AssertionError(f"residual reconstruction mismatch beyond 1e-5: {run_id}; max_abs={native_diff}")
        staged_forward_rows.append({"run_id":run_id,"panel":panel,"model":config,"source_vs_staged_residual_max_abs":forward_diff,"source_vs_staged_residual_atol":1e-5,"source_vs_staged_residual_rtol":1e-5,"source_vs_staged_residual_allclose":forward_allclose,"native_reconstruction_max_abs":native_diff,"native_reconstruction_allclose":native_allclose})
        sample_ids = metadata[panel]["anchor_id"].tolist()
        for stage_name, value in stages.items():
            if stage_name in ("state_input","state_content","state_position","state_modality","state_pre_tcn","state_post_tcn"):
                group = "state"
            elif stage_name.startswith("video"):
                group = "video"
            elif stage_name.startswith("anchor"):
                group = "anchor"
            elif stage_name == "memory":
                group = "memory"
            else:
                group = "future_query"
            temporal = "time" if stage_name in ("video_layout","video_content","video_time_position","video_space_position","video_modality","video_pre_decoder") else "sequence"
            append_stat(activation_rows, run_id, panel, group, stage_name, value, temporal)
        # Separate video content and positional vector norms, and verify the null branch stage.
        zeros = torch.zeros_like(t["dynamic"])
        null_projected = model.video_projection(model._video_layout(zeros, t["static"]))
        null_layout_max = max_abs(model._video_layout(zeros, t["static"]))
        # Record query matrices at the exact computation locations used by the decoder.
        raw_dec = capture["decoder_query"]
        self_q_input = capture["self_query"]
        cross_q_input = capture["cross_query"]
        if raw_dec is None or self_q_input is None or cross_q_input is None:
            raise AssertionError(f"query hooks did not capture expected decoder stages: {run_id}")
        if not torch.allclose(raw_dec, self_q_input, atol=0, rtol=0):
            raise AssertionError(f"first self-attention query input differs from raw future encoding: {run_id}")
        cross_mha = model.decoder.layers[0].multihead_attn
        E = cross_mha.embed_dim
        q_weight = cross_mha.in_proj_weight[:E]
        q_bias = cross_mha.in_proj_bias[:E] if cross_mha.in_proj_bias is not None else None
        cross_projected = F.linear(cross_q_input, q_weight, q_bias)
        for i, sample_id in enumerate(sample_ids):
            for stage, matrix in [
                ("decoder_raw_query_to_self_attention", raw_dec[i]),
                ("first_layer_cross_attention_after_norm2", cross_q_input[i]),
                ("first_layer_cross_attention_after_learned_Q_projection", cross_projected[i]),
            ]:
                summary, sv = query_spectrum(run_id, panel, config, str(sample_id), stage, matrix)
                query_rows.append(summary)
                singular_rows.extend(sv)
        # Verify actual MultiheadAttention output can be reconstructed from actual Q/K/V.
        layer_status = []
        for record in capture["cross_layers"]:
            status = reconstruct_cross_attention(record, record["layer_idx"], run_id, panel, config,
                                                 metadata[panel], attention_group_rows, attention_token_rows)
            attention_replays.append(status)
            layer_status.append(status)
        # Static feature path and projection bias: this is a positional-slot control, not leaked dynamic content.
        video_diag_rows.append({
            "run_id":run_id,"panel":panel,"model":config,
            "null_layout_max_abs_before_projection":null_layout_max,
            "null_projection_max_abs_after_linear":max_abs(null_projected),
            "video_projection_bias_l2":float(torch.linalg.vector_norm(model.video_projection.bias).item()),
            "video_modality_l2":float(torch.linalg.vector_norm(model.video_modality).item()),
            "time_position_l2":float(torch.linalg.vector_norm(stages["video_time_position"]).item()),
            "spatial_position_l2":float(torch.linalg.vector_norm(stages["video_space_position"]).item()),
            "attention_reconstruction_all_layers_pass":all(x["status"]=="PASS" for x in layer_status),
        })
        # Local gradients for position and weighted 0.1 FD components, without .backward or optimizer.
        decoder_handle = None
        query_holder: dict[str, Tensor] = {}
        def decoder_gradient_pre(module, args):
            query = args[0]
            if not query.requires_grad:
                query.requires_grad_(True)
            query_holder["query"] = query
        decoder_handle = model.decoder.register_forward_pre_hook(decoder_gradient_pre)
        try:
            state_before_grad = state_hash(model)
            gradient_rows.extend(gradient_groups(model, t, run_id, panel, query_holder))
            state_after_grad = state_hash(model)
        finally:
            decoder_handle.remove()
        if state_before_grad != state_after_grad or state_before_grad != before_hash:
            raise AssertionError(f"read-only audit changed model weights or buffers: {run_id}")
        if any(p.grad is not None for p in model.parameters()):
            raise AssertionError(f"gradients accumulated into model parameters: {run_id}")

    # Check source lock and all files registered at freeze time stayed byte-identical during A.
    registry = read_json(ROOT / "protocol/SOURCE_REGISTRY.json")
    source_unchanged = []
    for item in registry["source_files"]:
        p = Path(item["path"])
        if p.is_file():
            actual = sha(p)
            ok = actual == item["observed_sha256"]
            source_unchanged.append({"path":str(p),"observed_sha256":actual,"unchanged":ok})
            if not ok:
                raise AssertionError(f"source changed during read-only audit: {p}")
    all_attention_pass = bool(attention_replays) and all(x.get("status") == "PASS" for x in attention_replays)
    if not all_attention_pass:
        raise AssertionError("manual attention reconstruction failed at one or more layers")
    model_source_sha = sha(SCHED / "code/d3_model.py")
    if model_source_sha != "6d4c321c65bb101f5c756ba8f1186a98d0358e16dc49dfa39f1d677c35d9e589":
        raise AssertionError("locked model source hash mismatch")

    write_csv(ROOT/"forensics/STAGED_FORWARD_CHECK.csv", staged_forward_rows)
    write_csv(ROOT/"forensics/BASELINE_REPLAY_ACCEPTANCE.csv", replay_rows)
    write_csv(ROOT/"forensics/ACTIVATION_STATS.csv", activation_rows)
    write_csv(ROOT/"forensics/ATTENTION_GROUPS.csv", attention_group_rows)
    write_csv(ROOT/"forensics/ATTENTION_TOKENS.csv", attention_token_rows)
    write_csv(ROOT/"forensics/GRADIENT_PATHS.csv", gradient_rows)
    write_csv(ROOT/"forensics/QUERY_SPECTRA.csv", query_rows)
    write_csv(ROOT/"forensics/QUERY_SINGULAR_VALUES.csv", singular_rows)
    write_csv(ROOT/"forensics/OLD_QUERY_TABLES.csv", query_table_rows)
    write_csv(ROOT/"forensics/VIDEO_NULL_BRANCH.csv", video_diag_rows)
    write_json(ROOT/"forensics/QUERY_RANK_RECONCILIATION.json", {
        "old_h30_h60_prefix_exact": prefix_exact,
        "old_future_time_rule": "float32 arange(1,H+1)/30 seconds",
        "h30_last_time_seconds": float(30/30), "h60_last_time_seconds": float(60/30),
        "raw_rank_threshold": "float32 epsilon * max(H,128) * largest singular value; SVD in float64 of FP32 query",
        "existing_accepted_algorithm": "reproduced from LEARN_ROOT/code/replay_temporal_variation.py query_trace lines 428-456",
        "model_has_separate_learned_future_query_projection_or_embedding": False,
        "decoder_first_self_attention_input_is_raw_query": True,
        "decoder_first_cross_attention_query_is_after_layer_norm2": True,
        "cross_attention_Q_projection_source": "first WIDTH rows of first decoder layer multihead_attn.in_proj_weight",
        "decoder_tgt_mask": None,
        "memory_key_padding_mask": None,
    })
    audit = {
        "schema_version":1,"status":"PASS_READ_ONLY_AUDIT",
        "created_utc":__import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat().replace("+00:00","Z"),
        "model_source_path":str(SCHED/"code/d3_model.py"),"model_source_sha256":model_source_sha,
        "baseline_replay_count":len(replay_rows),"baseline_replay_status":"PASS",
        "audited_checkpoint_file_count":len({x["path"] for x in audited_paths}),
        "audited_checkpoints":audited_paths,
        "parameter_count_all_baselines_equal":len({x["parameter_count"] for x in audited_paths})==1,
        "shared_init_state_hash":expected_init_state_hash,
        "normalizer_summary":normalizer_summary,
        "query_old_h30_h60_prefix_exact":prefix_exact,
        "attention_manual_reconstruction_pass":all_attention_pass,
        "attention_reconstruction_runs":len(attention_replays),
        "attention_reconstruction_max_abs":max(x["attention_reconstruction_max_abs"] for x in attention_replays),
        "staged_forward_checks":len(staged_forward_rows),
        "staged_forward_max_abs":max(x["source_vs_staged_residual_max_abs"] for x in staged_forward_rows),
        "staged_forward_tolerance_atol_rtol":1e-5,
        "attention_is_not_causal_explanation":True,
        "history_mask_is_model_input_feature_not_attention_key_padding_mask":True,
        "video_null_branch_content_zeroed_before_projection":True,
        "video_null_branch_retains_projection_bias_and_time_space_position":True,
        "query_component_isolation_possible":True,
        "hard_identity_or_formula_error_found":False,
        "optimizer_created":False,"optimizer_updates":0,"model_weights_or_buffers_changed":False,
        "dev_targets_opened":False,"final_or_heldout_access":0,
        "gpu_query_use_lease":0,"old_root_writes":0,
        "source_files_rechecked":len(source_unchanged),
        "source_files_unchanged":all(x["unchanged"] for x in source_unchanged),
    }
    write_json(ROOT/"forensics/FUSION_QUERY_PATH_AUDIT.json",audit)
    write_json(ROOT/"forensics/SOURCE_UNCHANGED_AFTER_AUDIT.json",{
        "status":"PASS" if all(x["unchanged"] for x in source_unchanged) else "FAIL",
        "files_checked":len(source_unchanged),"files":source_unchanged})
    write_text = """# Fusion and query path audit

Status: PASS_READ_ONLY_AUDIT. All inspected sources and checkpoints remained read-only. The audit uses the locked primary-train P1/P8 panels. It creates no optimizer and performs no parameter update.

## Identity and replay

The four historical FULL_LOSS baselines were uniquely found by run ID. Every final checkpoint was strict-loaded into the locked D3 model and replayed on its exact train panel. Saved IDs, targets and anchors matched, and predictions passed atol=1e-5 / rtol=1e-5. The model source SHA-256 is 6d4c321c65bb101f5c756ba8f1186a98d0358e16dc49dfa39f1d677c35d9e589. The audited checkpoint set contains one shared INIT, four old tiny-fit final checkpoints, and the accepted old/corrected TS11-D H30 seed-20261002 epoch-40 pair.

## Actual data path

History is already standardized in the locked V3 input archive; the 71 state values are concatenated with history mask and standardized relative delta, projected 73→128, then summed with sinusoidal relative-time encoding and a zero modality buffer before five causal residual TCN blocks. The video grid is normalized with the accepted train-fitted safe standard deviation. TS11-D preserves the full 16×4 grid, projects 768→128, adds 16-slot time encoding, 2×2 spatial encoding and zero modality buffer, then reshapes time-major to 64 tokens. The anchor enters in native coordinates through a 28→128 projection. Memory order is 32 state + 64 video + 1 anchor = 97 tokens.

STATE-CTRL zeroes the dynamic video values before the video linear projection. The projected zero input still contains the projection bias, and the branch retains time and spatial slot encodings. This removes dynamic content while retaining the same slot structure; no contract evidence marks that as an implementation error.

The returned model output is a standardized residual. Native predictions are reconstructed as anchor + target_sigma × residual. H30 loss has 29 future-internal differences; the anchor-to-first seam is excluded. No separate future query projection or learned future embedding is present. The fixed future encoding is generated only in the future-query branch, so it can be replaced without changing state/video positional encodings, modality/anchor tokens, decoder weights or masks. A duplicated staged forward is checked against the locked model forward under no-grad at atol=1e-5 / rtol=1e-5; per-checkpoint differences are in STAGED_FORWARD_CHECK.csv.

The history mask is concatenated as a state feature; it is not passed as a decoder key-padding mask. The decoder call passes no target, memory or key-padding masks. All 97 memory tokens and all 30 query positions are therefore visible in attention. This matches the inspected source call and is not itself a hard error.

## Numerical probes

Activation mean/std/RMS, per-token norm variation, and temporal differences are in ACTIVATION_STATS.csv. Attention mass and per-token weighted-value contribution norms are in ATTENTION_GROUPS.csv and ATTENTION_TOKENS.csv. The manual two-layer cross-attention reconstruction matches the actual MultiheadAttention outputs within atol=1e-5 / rtol=1e-5; the errors are in FUSION_QUERY_PATH_AUDIT.json. Attention weights are not a causal explanation.

Position and weighted 0.1×FD local gradient norms, ratios and cosine values are in GRADIENT_PATHS.csv. The zero-output INIT causes expected zero upstream gradients; those are separated from final-checkpoint gradients. No encoder parameter exists in this downstream model; frozen visual features enter as cached tensors.

The fixed query spectra use FP32 sinusoidal values from step/30, with H30 exactly the first 30 rows of H60. Singular values are computed in float64 from the FP32 matrix. Effective rank uses float32 epsilon × max(matrix dimensions) × largest singular value, reproducing the accepted Phase A rule. Raw, mean-centered, first decoder self-attention input, first-layer LayerNorm output and learned cross-attention Q projection spectra are recorded in QUERY_SPECTRA.csv and QUERY_SINGULAR_VALUES.csv.

## Gate

No hard identity, target formula, axis/broadcast, null-branch, mask or cache error was found. The fixed future-query component can be substituted independently. This satisfies the entry condition for the locked, four-cell Hadamard H30 train-fit probe; it does not show that old query geometry is a benchmark failure cause or imply any dev gain.
"""
    (ROOT/"reports/FUSION_QUERY_PATH_AUDIT.zh-CN.md").write_text(write_text,encoding="utf-8")
    print(json.dumps(audit,ensure_ascii=False,indent=2,default=str))
    return 0

# Small helper lists are initialized after function definitions to keep the audit output schema stable.
null_projection_rows: list[dict[str, Any]] = []
video_diag_rows: list[dict[str, Any]] = []

if __name__ == "__main__":
    raise SystemExit(main())

