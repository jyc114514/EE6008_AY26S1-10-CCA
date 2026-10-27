"""Guarded tmux entrypoint using CPU-verified Stage0 checkpoints before formal."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import run_d3_eq_queue as queue


EQ_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EQ_ROOT.parents[2]
PYTHON = PROJECT_ROOT / "envs/jyc/bin/python"
HOST_GATE = EQ_ROOT / "qa/D3_EQ_HOST_APPROVAL.json"
FORMAL_GATE = EQ_ROOT / "qa/D3_EQ_RESOURCE_GATE_PRE_FORMAL.json"
REMAINING_GATE = EQ_ROOT / "qa/D3_EQ_RESOURCE_GATE_REMAINING24.json"


def verified_stage0_acceptance(items):
    verifier = EQ_ROOT / "code/verify_d3_eq_stage0_entry.py"
    result = subprocess.run([str(PYTHON), str(verifier)], cwd=PROJECT_ROOT, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Stage0 CPU acceptance failed: {result.stdout}{result.stderr}")
    return json.loads((EQ_ROOT / "qa/D3_EQ_STAGE0_ACCEPTANCE.json").read_text(encoding="utf-8"))


def install_stage0_hook() -> None:
    original = queue.run_phase

    def run_phase(items, *, max_wait_seconds: int) -> None:
        original(items, max_wait_seconds=max_wait_seconds)
        if items and all(item.get("phase") == "STAGE0" for item in items):
            if len(queue.complete_items(items)) == 12 and not (EQ_ROOT / "qa/D3_EQ_STAGE0_ACCEPTANCE.json").exists():
                verified_stage0_acceptance(items)

    queue.run_phase = run_phase
    queue.build_stage0_acceptance = verified_stage0_acceptance


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--phase", required=True, choices=("stage0", "seed1_formal", "remaining_formal"))
    args, _ = parser.parse_known_args()
    if not HOST_GATE.is_file():
        raise FileNotFoundError("D3-EQ host approval and node-role gate is missing")
    host = json.loads(HOST_GATE.read_text(encoding="utf-8"))
    if host.get("status") != "PASS" or host.get("explicit_user_approval") is not True or host.get("node_role") not in {"interactive_gpu", "approved_scheduler_gpu"}:
        raise PermissionError("explicit host approval and a non-login GPU node classification are required")
    if args.phase == "seed1_formal":
        if not FORMAL_GATE.is_file():
            raise FileNotFoundError("fresh post-Stage0 disk/GPU-hour gate is missing")
        gate = json.loads(FORMAL_GATE.read_text(encoding="utf-8"))
        if gate.get("status") != "PASS" or float(gate.get("projected_gpu_hours", 1e9)) > 48.0 or float(gate.get("filesystem_free_fraction", 0.0)) < 0.10:
            raise PermissionError("formal resource gate is not PASS")
    if args.phase == "remaining_formal":
        if not REMAINING_GATE.is_file():
            raise FileNotFoundError("fresh first-seed-complete resource gate is missing")
        gate = json.loads(REMAINING_GATE.read_text(encoding="utf-8"))
        if gate.get("status") != "PASS" or float(gate.get("projected_gpu_hours", 1e9)) > 48.0 or float(gate.get("filesystem_free_fraction", 0.0)) < 0.10:
            raise PermissionError("remaining-24 resource gate is not PASS")
    install_stage0_hook()
    return queue.main()


if __name__ == "__main__":
    raise SystemExit(main())
