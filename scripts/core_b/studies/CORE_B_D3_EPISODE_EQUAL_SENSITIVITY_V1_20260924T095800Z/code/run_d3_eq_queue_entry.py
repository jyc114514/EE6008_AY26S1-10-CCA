"""Guarded tmux entrypoint for the D3-EQ watcher queue."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import run_d3_eq_queue as queue


EQ_ROOT = Path(__file__).resolve().parents[1]
HOST_GATE = EQ_ROOT / "qa/D3_EQ_HOST_APPROVAL.json"
FORMAL_GATE = EQ_ROOT / "qa/D3_EQ_RESOURCE_GATE_PRE_FORMAL.json"


def install_stage0_acceptance_hook() -> None:
    original = queue.run_phase

    def guarded_run_phase(items, *, max_wait_seconds: int) -> None:
        original(items, max_wait_seconds=max_wait_seconds)
        if items and all(item.get("phase") == "STAGE0" for item in items):
            if len(queue.complete_items(items)) == 12 and not (EQ_ROOT / "qa/D3_EQ_STAGE0_ACCEPTANCE.json").exists():
                queue.build_stage0_acceptance(items)

    queue.run_phase = guarded_run_phase


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--phase", required=True, choices=("stage0", "seed1_formal", "remaining_formal"))
    args, _ = parser.parse_known_args()
    if not HOST_GATE.is_file():
        raise FileNotFoundError("host approval/node/GPU preflight gate is missing")
    host = json.loads(HOST_GATE.read_text(encoding="utf-8"))
    if host.get("status") != "PASS" or host.get("explicit_user_approval") is not True:
        raise PermissionError("explicit host approval is required before any GPU queue phase")
    if args.phase != "stage0":
        if not FORMAL_GATE.is_file():
            raise FileNotFoundError("post-Stage0 disk/GPU-hour resource gate is missing")
        gate = json.loads(FORMAL_GATE.read_text(encoding="utf-8"))
        if gate.get("status") != "PASS" or gate.get("projected_gpu_hours") > 48.0 or gate.get("filesystem_free_fraction") < 0.10:
            raise PermissionError("formal D3-EQ resource gate is not PASS")
    install_stage0_acceptance_hook()
    return queue.main()


if __name__ == "__main__":
    raise SystemExit(main())
