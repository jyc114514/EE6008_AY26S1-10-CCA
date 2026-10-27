from __future__ import annotations

import inspect
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from v7_common import (
    NORMALIZER,
    ROOT,
    TASKS,
    V2,
    atomic_json,
    load_model_api,
    load_train_y_after_teacher_seal,
    position_fd_seam_metrics,
    sha_file,
    validate_prefix_keys,
)
from v7_build_targets import explicit_join


PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
PYTHON_BIN = PROJECT / "envs/jyc/bin/python"
SEED = 202610131
ATOL = 1e-7


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def expect_raises(fn, label: str) -> None:
    try:
        fn()
    except Exception:
        return
    raise AssertionError(f"expected rejection did not occur: {label}")


def main() -> int:
    torch.set_num_threads(1)
    report: dict[str, object] = {
        "schema_version": 1,
        "status": "RUNNING",
        "created_utc": utc_now(),
        "scope": "CPU synthetic QA; no project target values are loaded",
        "final_heldout_payload_access": 0,
        "dev_target_values_read": 0,
    }

    # Native target composition, zero soft-target ratio, and linear FD identity.
    rng = np.random.default_rng(20261015)
    y = rng.normal(size=(3, 30, 28)).astype(np.float64)
    s = rng.normal(size=(3, 30, 28)).astype(np.float64)
    v = rng.normal(size=(3, 30, 28)).astype(np.float64)
    t = 0.5 * s + 0.5 * v
    z_self = 0.5 * y + 0.5 * v
    z_mix = 0.5 * y + 0.5 * t
    beta_zero = (1.0 - 0.0) * y + 0.0 * t
    assert np.array_equal(beta_zero, y)
    assert np.allclose(np.diff(z_mix, axis=1), 0.5 * np.diff(y, axis=1) + 0.25 * np.diff(s, axis=1) + 0.25 * np.diff(v, axis=1), rtol=0.0, atol=1e-12)
    s_equal_v = np.array(v, copy=True)
    t_equal = 0.5 * s_equal_v + 0.5 * v
    assert np.array_equal(0.5 * y + 0.5 * t_equal, 0.5 * y + 0.5 * v)
    report["native_target_formula_and_fd_linearity"] = "PASS"
    report["beta_zero_returns_Y"] = "PASS"
    report["S_equals_V_self_mix_targets_identical"] = "PASS"

    # Wrong axis, wrong receiver identity and object payloads are refused.
    metric_sigma = np.ones(28, dtype=np.float64)
    metric_fd = np.ones(28, dtype=np.float64)
    anchor = rng.normal(size=(3, 28))
    expect_raises(
        lambda: position_fd_seam_metrics(y.transpose(0, 2, 1), y, anchor, metric_sigma, metric_fd),
        "wrong horizon/axis",
    )
    canonical = [
        {"row": 0, "anchor_id": "a0", "episode_id": 10, "task_id": 119, "anchor_frame": 119, "source_cluster": "g0"},
        {"row": 1, "anchor_id": "a1", "episode_id": 11, "task_id": 94, "anchor_frame": 119, "source_cluster": "g1"},
    ]
    wrong = [
        {"row": 0, "anchor_id": "a0", "episode_id": 10, "task_id": 119, "anchor_frame": 119, "source_cluster": "g0"},
        {"row": 1, "anchor_id": "wrong", "episode_id": 11, "task_id": 94, "anchor_frame": 119, "source_cluster": "g1"},
    ]
    expect_raises(lambda: explicit_join(canonical, wrong, np.zeros((2, 30, 28)), "synthetic"), "wrong receiver ID")
    object_path = ROOT / "tmp/qa_object_payload.npz"
    object_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(object_path, payload=np.asarray([object()], dtype=object))
    def load_object_payload() -> None:
        with np.load(object_path, allow_pickle=False) as archive:
            _ = archive["payload"]
    expect_raises(load_object_payload, "object payload with allow_pickle=False")
    report["wrong_axis_rejected"] = True
    report["wrong_receiver_id_rejected"] = True
    report["object_payload_rejected"] = True

    # The prefix interface rejects labels/teacher sentinels before model forward.
    fake_prefix = {
        "state": torch.zeros(1, 32, 71),
        "history_mask": torch.ones(1, 32),
        "relative_delta": torch.zeros(1, 32),
        "anchor_command": torch.zeros(1, 28),
        "dynamic": torch.zeros(1, 16, 4, 768),
        "static": torch.zeros(1, 16, 4, 768),
    }
    validate_prefix_keys(fake_prefix, "TS11-D")
    sentinel = dict(fake_prefix)
    sentinel["future_target_sentinel"] = torch.ones(1, 30, 28)
    expect_raises(lambda: validate_prefix_keys(sentinel, "TS11-D"), "future-target input sentinel")
    report["future_input_sentinel_rejected"] = True

    # An empty teacher seal blocks the only train-Y loader before it can open Y.
    teacher_seal = ROOT / "protocol/TEACHER_BANK_SEAL.json"
    if teacher_seal.exists():
        raise RuntimeError("CPU preflight must run before teacher predictions are sealed")
    expect_raises(load_train_y_after_teacher_seal, "empty or absent teacher seal")
    report["empty_teacher_seal_blocks_train_Y_read"] = True

    # Target creation is structurally independent of all development labels.
    builder_source = (ROOT / "code/v7_build_targets.py").read_text(encoding="utf-8")
    if "DEV_TARGET_NPZ" in builder_source or "DEV_NPZ" in builder_source:
        raise RuntimeError("training target builder references development target data")
    report["target_builder_has_no_dev_target_dependency"] = True

    # Load the accepted fresh INIT on CPU. Its zero output layer implies initial persistence.
    model_type, _, state_hash = load_model_api()
    init_path = V2 / "model_only/INIT/seed_202610131/shared_5task_base.safetensors"
    init_manifest = json.loads((V4_COMPAT := ROOT.parent.parent.parent / "reports/benchmarks/CORE_B_V4_RGB_ALIGNMENT_REASSIGNMENT_V1_20261014T083805Z/qa/COMMON_INIT_AND_RECIPE.json").read_text(encoding="utf-8"))
    init_entry = next(row for row in init_manifest["fit_receipts"] if row["seed"] == SEED and row["program_id"] == "A_STATE_COMMON")
    if sha_file(init_path) != init_entry["init_sha256"]:
        raise RuntimeError("accepted fresh INIT hash differs from the V4 common-init receipt")
    model = model_type(seed=SEED, video_mode="TS11-D", waypoint_mode=None)
    model.load_state_dict(load_file(str(init_path), device="cpu"), strict=True)
    if not torch.count_nonzero(model.output.weight).item() == 0 or not torch.count_nonzero(model.output.bias).item() == 0:
        raise RuntimeError("accepted INIT output layer is not the locked zero/persistence initialization")
    if tuple(inspect.signature(type(model).forward).parameters) != ("self", "prefix_inputs", "task_id"):
        raise RuntimeError("student model forward signature unexpectedly accepts outcome/teacher inputs")
    report["accepted_INIT_zero_output_layer"] = True
    report["initial_output_is_persistence"] = True
    report["student_forward_signature_prefix_and_task_only"] = True

    # Synthetic CPU forward, safe reload, and paired loss/gradient identity for S=V.
    prefix = {
        "state": torch.zeros(2, 32, 71),
        "history_mask": torch.ones(2, 32),
        "relative_delta": torch.zeros(2, 32),
        "anchor_command": torch.as_tensor(anchor[:2], dtype=torch.float32),
        "dynamic": torch.zeros(2, 16, 4, 768),
        "static": torch.zeros(2, 16, 4, 768),
    }
    tasks = torch.tensor([0, 1], dtype=torch.long)
    model.eval()
    with torch.inference_mode():
        output = model(prefix, tasks)
        expected_persistence = prefix["anchor_command"][:, None, :].expand(2, 30, 28)
        native = prefix["anchor_command"][:, None, :] + torch.as_tensor(
            json.loads(NORMALIZER.read_text(encoding="utf-8"))["target_sigma"], dtype=torch.float32
        )[None, None, :] * output.action_residual
    if not torch.allclose(native, expected_persistence, rtol=0.0, atol=0.0):
        raise RuntimeError("zero-initialized action output did not produce persistence")
    report["synthetic_prefix_forward_finite"] = bool(torch.isfinite(native).all())
    report["zero_init_prediction_equals_persistence"] = True

    # Save only prefix tensors/task/anchor/scale and expected native prediction.
    # A separate CPU process below replays the student API without GT or teachers.
    from safetensors.torch import save_file
    fixture_path = ROOT / "tmp/CPU_PREFIX_FIXTURE.safetensors"
    expected_prediction_path = ROOT / "tmp/CPU_PREFIX_EXPECTED.safetensors"
    fixture_tensors = {key: value.detach().cpu().contiguous() for key, value in prefix.items()}
    fixture_tensors["task_rank"] = tasks.detach().cpu().contiguous()
    fixture_tensors["anchor_native"] = prefix["anchor_command"].detach().cpu().contiguous()
    fixture_tensors["target_sigma"] = torch.as_tensor(
        json.loads(NORMALIZER.read_text(encoding="utf-8"))["target_sigma"], dtype=torch.float32
    ).contiguous()
    save_file(fixture_tensors, str(fixture_path))
    save_file({"prediction_native": native.detach().cpu().contiguous()}, str(expected_prediction_path))
    report["target_blind_eval_fixture_contains_no_outcomes_or_teachers"] = True

    # Fixed INIT, batch and random stream must produce matching loss and gradients
    # when the SELF and MIX target tensors coincide (S=V).
    from d3_eq_contract import per_sample_loss
    target = torch.as_tensor((0.5 * y[:2] + 0.5 * v[:2]), dtype=torch.float32)
    init_state = load_file(str(init_path), device="cpu")
    gradients: list[dict[str, torch.Tensor]] = []
    losses: list[float] = []
    for _ in range(2):
        candidate = model_type(seed=SEED, video_mode="TS11-D", waypoint_mode=None)
        candidate.load_state_dict(init_state, strict=True)
        candidate.train()
        torch.manual_seed(9917)
        out = candidate(prefix, tasks)
        sigma = torch.as_tensor(json.loads(NORMALIZER.read_text(encoding="utf-8"))["target_sigma"], dtype=torch.float32)
        fd = torch.as_tensor(json.loads(NORMALIZER.read_text(encoding="utf-8"))["safe_fd_scale"], dtype=torch.float32)
        loss = per_sample_loss(out.action_residual, target, prefix["anchor_command"], sigma, fd).mean()
        loss.backward()
        losses.append(float(loss.detach()))
        gradients.append({name: parameter.grad.detach().clone() for name, parameter in candidate.named_parameters() if parameter.grad is not None})
    if not np.allclose(losses[0], losses[1], rtol=0.0, atol=ATOL):
        raise RuntimeError("S=V synthetic paired loss equality failed")
    if gradients[0].keys() != gradients[1].keys() or any(
        not torch.allclose(gradients[0][name], gradients[1][name], rtol=0.0, atol=ATOL)
        for name in gradients[0]
    ):
        raise RuntimeError("S=V synthetic paired gradient equality failed")
    report["S_equals_V_same_INIT_random_stream_loss_grad_equal"] = True
    report["synthetic_loss"] = losses[0]

    # Fresh-process safe reload QA is performed without CUDA visibility.
    weights_path = ROOT / "tmp/CPU_SYNTHETIC_INIT.safetensors"
    save_file({name: tensor.contiguous() for name, tensor in init_state.items()}, str(weights_path))
    reload_receipt_path = ROOT / "tmp/CPU_SYNTHETIC_INIT_RELOAD.json"
    reload_job_path = ROOT / "tmp/CPU_SYNTHETIC_INIT_RELOAD_JOB.json"
    expected_hash = state_hash(model)
    atomic_json(reload_job_path, {
        "model_path": str(weights_path),
        "seed": SEED,
        "video_mode": "TS11-D",
        "expected_state_hash": expected_hash,
        "prefix_fixture_path": str(fixture_path),
        "expected_prediction_path": str(expected_prediction_path),
        "replay_atol": 1e-7,
        "replay_rtol": 1e-6,
        "receipt_path": str(reload_receipt_path),
    })
    env = dict(os.environ)
    env.update({"CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1"})
    subprocess.run(
        [str(PYTHON_BIN), str(ROOT / "code/v7_reload_check.py"), "--job-json", str(reload_job_path)],
        check=True, env=env, cwd=str(PROJECT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    reload_receipt = json.loads(reload_receipt_path.read_text(encoding="utf-8"))
    if reload_receipt.get("status") != "PASS" or reload_receipt.get("fresh_process") is not True:
        raise RuntimeError("synthetic fresh-process safe reload failed")
    report["fresh_process_safe_reload"] = True
    if reload_receipt.get("target_blind_eval_replay") != "PASS":
        raise RuntimeError("fresh-process target-blind evaluation replay did not pass")
    report["target_blind_eval_replay_fresh_process"] = "PASS"

    report.update({
        "status": "PASS",
        "accepted_task_ids": list(TASKS),
        "normalizer_sha256": sha_file(NORMALIZER),
        "init_sha256": sha_file(init_path),
        "target_builder_sha256": sha_file(ROOT / "code/v7_build_targets.py"),
        "final_heldout_payload_access": 0,
        "dev_target_values_read": 0,
        "gpu_queries": 0,
    })
    atomic_json(ROOT / "qa/CPU_PREFLIGHT.json", report)
    atomic_json(ROOT / "qa/SCHEMA_PREFLIGHT.json", {
        "schema_version": 1, "status": "PASS", "created_utc": utc_now(),
        "cpu_preflight_path": str(ROOT / "qa/CPU_PREFLIGHT.json"),
        "cpu_preflight_sha256": sha_file(ROOT / "qa/CPU_PREFLIGHT.json"),
        "prefix_keys_strict": True, "future_target_input_rejected": True,
        "train_target_shape": [1280, 30, 28], "student_prediction_shape": [185, 30, 28],
        "student_output_shape": ["B", 30, 28], "axes": 28, "horizon": 30,
        "allowed_task_ids": list(TASKS), "object_payloads_allowed": False,
        "dev_target_values_read": 0, "final_heldout_payload_access": 0,
    })
    print(json.dumps({"status": "PASS", "qa_path": str(ROOT / "qa/CPU_PREFLIGHT.json")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
