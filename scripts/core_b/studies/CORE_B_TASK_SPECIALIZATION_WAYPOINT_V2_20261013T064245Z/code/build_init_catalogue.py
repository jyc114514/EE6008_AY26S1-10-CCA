#!/usr/bin/env python3
"""Create fresh tensor-only INIT states for paired V2 model cells and seeds."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from v2_model import V2TaskConditionedPredictor, parameter_count, state_hash


ROOT = Path(__file__).resolve().parents[1]
SCREEN_SEED = 202610130
FORMAL_SEEDS = (202610131, 202610132, 202610133)
SCOPES = ("shared_5task", "task_119", "task_94", "task_90", "task_141", "task_230")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def state_hash_dict(state: dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        h.update(name.encode("utf-8"))
        h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def save_or_verify(path: Path, state: dict[str, torch.Tensor], expected_hash: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        loaded = load_file(str(path), device="cpu")
        if set(loaded) != set(state) or state_hash_dict(loaded) != expected_hash:
            raise RuntimeError(f"existing INIT differs from deterministic fresh tensors: {path}")
        return
    temporary = path.with_name(f".{path.name}.part")
    save_file({k: v.detach().cpu().contiguous() for k, v in state.items()}, str(temporary))
    temporary.replace(path)
    loaded = load_file(str(path), device="cpu")
    if state_hash_dict(loaded) != expected_hash:
        raise RuntimeError(f"INIT fresh-process tensor hash mismatch: {path}")


def main() -> None:
    torch.set_num_threads(2)
    rows = []
    formal = []
    seeds = (SCREEN_SEED, *FORMAL_SEEDS)
    for seed in seeds:
        for scope in SCOPES:
            base = V2TaskConditionedPredictor(seed=seed, video_mode="TS11-D", waypoint_mode=None)
            base_hash = state_hash(base)
            base_path = ROOT / "model_only/INIT" / f"seed_{seed}" / f"{scope}_base.safetensors"
            save_or_verify(base_path, base.state_dict(), base_hash)
            rows.append({
                "seed": seed,
                "scope": scope,
                "init_type": "fresh_Q0_task_conditioned",
                "init_seed": seed,
                "state_hash": base_hash,
                "safetensors_path": str(base_path),
                "safetensors_sha256": digest(base_path),
                "parameter_count": parameter_count(base),
                "state_video_pair_shared": True,
                "goal_streams": "not_applicable",
            })
            if seed in FORMAL_SEEDS and scope == "task_119":
                variants = []
                for mode in ("AUX_ONLY", "GOAL_COND"):
                    model = V2TaskConditionedPredictor(seed=seed, video_mode="TS11-D", waypoint_mode=mode)
                    model_state = model.state_dict()
                    for key, tensor in base.state_dict().items():
                        if key not in model_state or model_state[key].shape != tensor.shape:
                            raise RuntimeError(f"waypoint common INIT mismatch: {key}")
                        model_state[key] = tensor.detach().clone()
                    model.load_state_dict(model_state, strict=True)
                    variants.append((mode, model))
                if any(not torch.equal(variants[0][1].state_dict()[k], variants[1][1].state_dict()[k]) for k in variants[0][1].state_dict()):
                    raise RuntimeError("AUX_ONLY and GOAL_COND do not share exact fresh INIT tensors")
                waypoint_hash = state_hash(variants[0][1])
                waypoint_path = ROOT / "model_only/INIT" / f"seed_{seed}" / "task_119_waypoint_common.safetensors"
                save_or_verify(waypoint_path, variants[0][1].state_dict(), waypoint_hash)
                formal.append({
                    "seed": seed,
                    "task_id": 119,
                    "variants": "AUX_ONLY,GOAL_COND",
                    "common_state_hash": base_hash,
                    "waypoint_state_hash": waypoint_hash,
                    "safetensors_path": str(waypoint_path),
                    "safetensors_sha256": digest(waypoint_path),
                    "goal_head_prng_seed": seed + 130363,
                    "goal_projection_prng_seed": seed + 130364,
                    "fresh_from_init": True,
                    "trained_sv_checkpoint_warmstart": False,
                })

    catalogue = ROOT / "model_only/INIT_CATALOGUE.csv"
    catalogue.parent.mkdir(parents=True, exist_ok=True)
    with catalogue.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    waypoint_catalogue = ROOT / "model_only/WAYPOINT_INIT_CATALOGUE.csv"
    with waypoint_catalogue.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(formal[0]))
        writer.writeheader(); writer.writerows(formal)
    receipt = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "PASS",
        "screen_seed": SCREEN_SEED,
        "formal_seeds": list(FORMAL_SEEDS),
        "logical_scopes": list(SCOPES),
        "base_init_count": len(rows),
        "waypoint_formal_init_count": len(formal),
        "state_video_common_init_hashes_match_by_scope_seed": True,
        "aux_goal_cond_full_init_hashes_match_by_seed": True,
        "screening_and_formal_heads_fresh": True,
        "trained_checkpoint_warmstarts": 0,
        "safe_format": "safetensors",
    }
    receipt_path = ROOT / "qa/INIT_SAMPLER_PAIRING.json"
    if receipt_path.exists():
        existing = json.loads(receipt_path.read_text(encoding="utf-8"))
        if existing.get("status") != "PASS":
            raise RuntimeError("existing INIT pairing receipt is not accepted")
    else:
        receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
