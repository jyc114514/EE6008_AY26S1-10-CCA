from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from v7_common import NORMALIZER, ROOT, V2, atomic_json, load_model_api, sha_file

PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
PYTHON_BIN = PROJECT / "envs/jyc/bin/python"
SEED = 202610131


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def main() -> int:
    torch.set_num_threads(1)
    model_type, _, state_hash = load_model_api()
    init_path = V2 / "model_only/INIT/seed_202610131/shared_5task_base.safetensors"
    model = model_type(seed=SEED, video_mode="TS11-D", waypoint_mode=None)
    init_tensors = load_file(str(init_path), device="cpu")
    model.load_state_dict(init_tensors, strict=True)
    state_digest = state_hash(model)
    model.eval()

    prefix = {
        "state": torch.zeros(2, 32, 71),
        "history_mask": torch.ones(2, 32),
        "relative_delta": torch.zeros(2, 32),
        "anchor_command": torch.linspace(-0.5, 0.5, 56).reshape(2, 28),
        "dynamic": torch.zeros(2, 16, 4, 768),
        "static": torch.zeros(2, 16, 4, 768),
    }
    task_rank = torch.tensor([0, 1], dtype=torch.long)
    anchor = prefix["anchor_command"]
    norm = json.loads(NORMALIZER.read_text(encoding="utf-8"))
    sigma = torch.as_tensor(norm["target_sigma"], dtype=torch.float32)
    with torch.inference_mode():
        output = model(prefix, task_rank)
        expected_native = anchor[:, None, :] + sigma[None, None, :] * output.action_residual.float()

    fixture_path = ROOT / "tmp/TARGET_BLIND_PREFIX_FIXTURE.safetensors"
    expected_path = ROOT / "tmp/TARGET_BLIND_EXPECTED.safetensors"
    model_path = ROOT / "tmp/TARGET_BLIND_MODEL.safetensors"
    save_file({
        **{name: tensor.contiguous() for name, tensor in prefix.items()},
        "task_rank": task_rank.contiguous(),
        "anchor_native": anchor.clone().contiguous(),
        "target_sigma": sigma.contiguous(),
    }, str(fixture_path))
    save_file({"prediction_native": expected_native.contiguous()}, str(expected_path))
    save_file({name: tensor.contiguous() for name, tensor in init_tensors.items()}, str(model_path))

    receipt_path = ROOT / "qa/TARGET_BLIND_REPLAY_QA.json"
    job_path = ROOT / "tmp/TARGET_BLIND_REPLAY_JOB.json"
    atomic_json(job_path, {
        "model_path": str(model_path),
        "seed": SEED,
        "video_mode": "TS11-D",
        "expected_state_hash": state_digest,
        "prefix_fixture_path": str(fixture_path),
        "expected_prediction_path": str(expected_path),
        "replay_atol": 1e-7,
        "replay_rtol": 1e-6,
        "receipt_path": str(receipt_path),
    })
    env = dict(os.environ)
    env.update({"CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1"})
    result = subprocess.run(
        [str(PYTHON_BIN), str(ROOT / "code/v7_reload_check.py"), "--job-json", str(job_path)],
        cwd=str(PROJECT), env=env, check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    if result.returncode != 0 or not receipt_path.is_file():
        raise RuntimeError(f"CPU fresh-process replay check failed: {result.stdout[-1200:]}")
    replay = json.loads(receipt_path.read_text(encoding="utf-8"))
    if replay.get("status") != "PASS" or replay.get("target_blind_eval_replay") != "PASS":
        raise RuntimeError("fresh-process target-blind replay receipt is not PASS")
    final = {
        "schema_version": 1,
        "status": "PASS",
        "created_utc": utc_now(),
        "device": "CPU",
        "cuda_device_queries": 0,
        "model_state_hash": state_digest,
        "accepted_init_sha256": sha_file(init_path),
        "prefix_fixture_path": str(fixture_path),
        "prefix_fixture_sha256": sha_file(fixture_path),
        "expected_native_prediction_sha256": sha_file(expected_path),
        "fresh_process_replay_receipt_path": str(receipt_path),
        "fresh_process_replay_receipt_sha256": sha_file(receipt_path),
        "replay_max_abs_difference": replay["max_abs_difference"],
        "prefix_keys": sorted(prefix),
        "teacher_or_outcome_inputs": False,
        "final_heldout_payload_access": 0,
    }
    atomic_json(ROOT / "qa/TARGET_BLIND_REPLAY_QA.json", final)
    print(json.dumps({"status": "PASS", "max_abs_difference": final["replay_max_abs_difference"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
