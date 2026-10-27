"""Fail-closed physical GPU identity guard run inside a shared-watcher child."""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

TASK_ROOT = Path(__file__).resolve().parents[1]
ALLOWED = frozenset({2, 3})


def normalize_gpu_uuid(value: object) -> str:
    if isinstance(value, bytes):
        try:
            text = value.decode("ascii").strip()
        except UnicodeDecodeError as exc:
            raise RuntimeError("logical CUDA UUID is not valid ASCII") from exc
    else:
        text = str(value).strip()
    if text.lower().startswith("gpu-"):
        text = text[4:]
    return text.lower()


def verify_assigned_device(nvidia_smi: str = "nvidia-smi") -> tuple[int, str, str]:
    try:
        candidates = frozenset(int(token) for token in os.environ.get("D3_TASK_GPU_ALLOWLIST", "").split(",") if token)
    except ValueError as exc:
        raise RuntimeError("task physical GPU candidate list is malformed") from exc
    if candidates != frozenset({2}):
        raise RuntimeError("this resumed run requires exactly physical GPU2 as its active candidate")
    physical = int(os.environ.get("GPU_WATCHER_PHYSICAL_GPU", "-1"))
    if physical not in candidates:
        raise RuntimeError(f"watcher assigned unauthorized physical GPU {physical}")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical):
        raise RuntimeError("CUDA_VISIBLE_DEVICES does not match watcher physical GPU assignment")
    if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise RuntimeError("CUDA_DEVICE_ORDER must be PCI_BUS_ID before logical mapping")
    try:
        uuid_map = {int(k): str(v) for k, v in json.loads(os.environ["D3_GPU_UUID_MAP"]).items()}
    except Exception as exc:
        raise RuntimeError("missing or invalid task GPU UUID map") from exc
    if set(uuid_map) != set(candidates):
        raise RuntimeError("task UUID map must match the locked active physical GPU candidates")
    expected_uuid = uuid_map[physical]
    if not expected_uuid.startswith("GPU-"):
        raise RuntimeError("expected UUID is malformed")

    result = subprocess.run(
        [nvidia_smi, "-i", str(physical), "--query-gpu=uuid,name",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    )
    rows = list(csv.reader(result.stdout.splitlines()))
    if len(rows) != 1 or len(rows[0]) != 2:
        raise RuntimeError("selected physical GPU identity query returned an unexpected schema")
    actual_uuid, actual_name = (field.strip() for field in rows[0])
    if actual_uuid != expected_uuid:
        raise RuntimeError(f"selected physical GPU UUID changed: expected {expected_uuid}, got {actual_uuid}")

    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("watcher child must see exactly one approved CUDA device")
    torch.cuda.set_device(0)
    properties = torch.cuda.get_device_properties(0)
    expected_name_map = json.loads(os.environ.get("D3_GPU_NAME_MAP", "{}"))
    expected_name = expected_name_map.get(str(physical))
    if expected_name and actual_name != expected_name:
        raise RuntimeError(f"physical GPU model changed: expected {expected_name}, got {actual_name}")
    if str(properties.name) != actual_name:
        raise RuntimeError("logical cuda:0 model differs from the selected physical GPU")
    logical_uuid = getattr(properties, "uuid", None)
    if logical_uuid is None:
        raise RuntimeError("logical cuda:0 UUID is unavailable; physical identity cannot be verified")
    if normalize_gpu_uuid(logical_uuid) != normalize_gpu_uuid(expected_uuid):
        raise RuntimeError(
            f"logical cuda:0 UUID differs from the selected physical GPU UUID: "
            f"expected {expected_uuid}, got {logical_uuid!r}"
        )
    os.environ["D3_EXPECTED_GPU_UUID"] = expected_uuid
    return physical, expected_uuid, actual_name


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments:
        raise ValueError("usage: physical_gpu_guard.py TRAINER.py [trainer arguments...]")
    trainer = Path(arguments[0]).resolve()
    if not trainer.is_relative_to((TASK_ROOT / "code").resolve()):
        raise ValueError("guard may exec only a trainer inside this task's code/")
    physical, uuid, name = verify_assigned_device()
    print(json.dumps({
        "status": "GPU_IDENTITY_PASS",
        "physical_gpu_index": physical,
        "physical_gpu_uuid": uuid,
        "physical_gpu_name": name,
        "logical_device": "cuda:0",
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "watcher_job_id": os.environ.get("GPU_WATCHER_JOB_ID"),
    }, sort_keys=True), flush=True)
    os.execv(sys.executable, [sys.executable, str(trainer), *arguments[1:]])
    raise AssertionError("os.execv unexpectedly returned")


if __name__ == "__main__":
    raise SystemExit(main())
