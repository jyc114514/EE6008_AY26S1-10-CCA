"""Synthetic CPU contracts for the public B1 dependency closure."""

from __future__ import annotations

import ast
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

B1_DIR = Path(__file__).resolve().parents[2]
SUPPORT_ROOT = B1_DIR / "public_support"
ANALYSIS_ROOT = SUPPORT_ROOT / "analysis_scripts"
VJEPA_ROOT = SUPPORT_ROOT / "vjepa2"
REPO_ROOT = B1_DIR.parents[4]

for _path in (ANALYSIS_ROOT, B1_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import accept_core_b_b1 as accept_b1  # noqa: E402
import run_core_b_baseline as run_b1  # noqa: E402


def _inside(path: str | Path, root: Path) -> bool:
    return Path(path).resolve().is_relative_to(root.resolve())


def _synthetic_rng_state() -> dict[str, object]:
    return {
        "python": (1, (2, 3), None),
        "numpy": ("synthetic", 1, 2, 3, 4.0),
        "torch_cpu": torch.tensor([1, 2], dtype=torch.uint8),
        "torch_cuda": [],
    }


def _synthetic_checkpoint(checkpoint_utils) -> dict[str, object]:
    return checkpoint_utils.build_checkpoint_payload(
        protocol_id="synthetic-b1-contract",
        run_identity={
            "cell": "B-D-A-F",
            "input": "synthetic",
            "head": "attentive",
            "adaptation": "frozen",
            "seed": 20260903,
        },
        epoch=1,
        global_optimizer_step=0,
        accumulation_state={
            "pending_microbatches": 0,
            "at_cycle_boundary": True,
            "optimizer_step": 0,
        },
        model_state_dict={"weight": torch.tensor([1.0])},
        head_state_dict={},
        lora_adapter_state_dict={},
        optimizer_state_dict={},
        scheduler_state_dict={},
        scaler_state_dict=None,
        early_stopping_state={"best_epoch": 1},
        data_provenance={"stage": "synthetic", "final_test_count": 0},
        rng_state=_synthetic_rng_state(),
    )


def test_support_modules_resolve_inside_public_tree() -> None:
    fixed = run_b1.load_support(SUPPORT_ROOT)
    checkpoint_utils = importlib.import_module("checkpoint_utils")
    accept_b1._ensure_public_checkpoint_support()

    assert _inside(run_b1.__file__, REPO_ROOT)
    assert _inside(accept_b1.__file__, REPO_ROOT)
    assert _inside(fixed.__file__, SUPPORT_ROOT)
    assert _inside(checkpoint_utils.__file__, SUPPORT_ROOT)


def test_non_public_dependency_roots_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="public B1 support package"):
        run_b1.load_support(tmp_path)
    with pytest.raises(ValueError, match="public B1 V-JEPA2 subset"):
        run_b1._load_attentive_pooler(tmp_path)


def test_generic_src_collision_is_rejected_before_import(monkeypatch, tmp_path: Path) -> None:
    import types

    unrelated_src = types.ModuleType("src")
    unrelated_src.__file__ = str(tmp_path / "src" / "__init__.py")
    monkeypatch.setitem(sys.modules, "src", unrelated_src)
    with pytest.raises(ImportError, match="outside the public V-JEPA2 source"):
        run_b1._load_attentive_pooler(VJEPA_ROOT)


def test_cpu_autocast_adapter_uses_null_context() -> None:
    fixed = run_b1.load_support(SUPPORT_ROOT)
    context = fixed.autocast_context(torch.device("cpu"))
    with context:
        value = torch.tensor([2.0]) * 3
    assert value.tolist() == [6.0]


def test_b1_checkpoint_wrapper_builds_strict_synthetic_payload(monkeypatch) -> None:
    fixed = run_b1.load_support(SUPPORT_ROOT)
    checkpoint_utils = importlib.import_module("checkpoint_utils")
    monkeypatch.setattr(
        checkpoint_utils, "capture_rng_state", _synthetic_rng_state
    )

    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
    payload = run_b1.checkpoint_payload(
        fixed=fixed,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        data_manifest={"stage": "synthetic"},
        identity={
            "cell": "B-D-A-F",
            "input": "synthetic",
            "head": "attentive",
            "adaptation": "frozen",
            "seed": 20260903,
        },
        early={"best_epoch": 1},
        epoch=1,
        global_step=0,
    )

    checkpoint_utils.validate_checkpoint(payload)
    assert payload["final_test_count"] == 0
    assert payload["physical_batch"] == 8
    assert payload["gradient_accumulation"] == 4
    assert payload["nominal_effective_batch"] == 32


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("final_test_count", 1, "final-test access"),
        ("schema_version", 999, "unsupported checkpoint schema"),
    ],
    ids=["reject-final-test-access", "reject-unknown-schema"],
)
def test_checkpoint_contract_rejects_invalid_synthetic_payload(
    field: str, value: object, message: str
) -> None:
    run_b1.load_support(SUPPORT_ROOT)
    checkpoint_utils = importlib.import_module("checkpoint_utils")
    payload = _synthetic_checkpoint(checkpoint_utils)
    payload[field] = value

    with pytest.raises(checkpoint_utils.CheckpointContractError, match=message):
        checkpoint_utils.validate_checkpoint(payload)


def test_safe_load_contract_uses_weights_only_and_cpu_map_location(
    tmp_path: Path, monkeypatch
) -> None:
    run_b1.load_support(SUPPORT_ROOT)
    checkpoint_utils = importlib.import_module("checkpoint_utils")
    payload = _synthetic_checkpoint(checkpoint_utils)
    sentinel = tmp_path / "synthetic-sentinel.bin"
    sentinel.write_bytes(b"synthetic-only")
    calls: list[tuple[object, object]] = []

    def fake_torch_load(path, *, map_location, weights_only):
        calls.append((map_location, weights_only))
        return payload

    monkeypatch.setattr(torch, "load", fake_torch_load)
    loaded = checkpoint_utils.load_checkpoint(sentinel, map_location="cpu")

    assert loaded["protocol_id"] == "synthetic-b1-contract"
    assert calls == [("cpu", True)]


def test_official_attentive_pooler_runs_on_small_synthetic_cpu_tensor() -> None:
    pooler_type = run_b1._load_attentive_pooler(VJEPA_ROOT)
    pooler = pooler_type(
        num_queries=1,
        embed_dim=32,
        num_heads=4,
        depth=2,
        use_activation_checkpointing=False,
    ).eval()
    tokens = torch.randn(2, 8, 32)

    with torch.no_grad():
        output = pooler(tokens)

    assert tuple(output.shape) == (2, 1, 32)
    assert torch.isfinite(output).all()


def test_b1_keeps_historical_attentive_architecture_configuration() -> None:
    runner_path = B1_DIR / "run_core_b_baseline.py"
    tree = ast.parse(runner_path.read_text(encoding="utf-8"))
    attentive_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "AttentivePooler"
    ]
    assert len(attentive_calls) == 1
    kwargs = {item.arg: ast.literal_eval(item.value) for item in attentive_calls[0].keywords}
    assert kwargs == {
        "num_queries": 1,
        "embed_dim": 768,
        "num_heads": 16,
        "depth": 4,
        "use_activation_checkpointing": False,
    }


def test_entrypoint_module_scope_has_no_payload_or_workload_calls() -> None:
    forbidden_names = {"open", "load", "run", "preflight", "Popen", "system"}
    for path in (
        B1_DIR / "run_core_b_baseline.py",
        B1_DIR / "accept_core_b_b1.py",
    ):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        top_level_calls = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            top_level_calls.extend(
                child
                for child in ast.walk(node)
                if isinstance(child, ast.Call)
            )
        assert not [
            node
            for node in top_level_calls
            if isinstance(node.func, ast.Name) and node.func.id in forbidden_names
        ]


def test_fresh_process_import_resolves_public_paths_without_payload_or_gpu_calls() -> None:
    child = r'''import importlib, json, os, pathlib, subprocess, sys
import numpy as np
import torch
def forbidden(*args, **kwargs):
    raise AssertionError("import attempted a blocked payload, job, or CUDA operation")
for name in ("load",):
    setattr(torch, name, forbidden)
np.load = forbidden
for name in ("is_available", "manual_seed_all", "synchronize", "mem_get_info"):
    setattr(torch.cuda, name, forbidden)
subprocess.run = forbidden
subprocess.Popen = forbidden
b1 = pathlib.Path(sys.argv[1]).resolve()
support = pathlib.Path(sys.argv[2]).resolve()
analysis = support / "analysis_scripts"
sys.path[:0] = [str(analysis), str(b1)]
runner = importlib.import_module("run_core_b_baseline")
accept = importlib.import_module("accept_core_b_b1")
fixed = runner.load_support(support)
accept._ensure_public_checkpoint_support()
pooler = runner._load_attentive_pooler(support / "vjepa2")
checkpoint = importlib.import_module("checkpoint_utils")
paths = {
  "runner": runner.__file__, "accept": accept.__file__,
  "adapter": fixed.__file__, "checkpoint": checkpoint.__file__,
  "pooler": importlib.import_module("src.models.attentive_pooler").__file__,
  "modules": importlib.import_module("src.models.utils.modules").__file__,
  "tensors": importlib.import_module("src.utils.tensors").__file__,
  "pooler_name": pooler.__name__,
}
print("B1_IMPORT_PATHS=" + json.dumps(paths, sort_keys=True))
'''
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONPATH"] = os.pathsep.join((str(B1_DIR), str(ANALYSIS_ROOT)))
    completed = subprocess.run(
        [sys.executable, "-c", child, str(B1_DIR), str(SUPPORT_ROOT)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    line = next(
        item for item in completed.stdout.splitlines() if item.startswith("B1_IMPORT_PATHS=")
    )
    resolved = json.loads(line.partition("=")[2])
    for key in ("runner", "accept"):
        assert _inside(resolved[key], REPO_ROOT)
    for key in ("adapter", "checkpoint"):
        assert _inside(resolved[key], SUPPORT_ROOT)
    for key in ("pooler", "modules", "tensors"):
        assert _inside(resolved[key], VJEPA_ROOT)
