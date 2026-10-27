#!/usr/bin/env python3
"""Minimal host-context CUDA preflight for the task-local watcher child.

This performs no model loading and allocates only a 4 MiB CUDA tensor.  It
checks the physical UUID reported by nvidia-smi against the watcher binding
and the UUID exposed by the project PyTorch environment.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import torch


def namespace(path: str) -> str:
    try:
        return os.readlink(path)
    except OSError as exc:
        return f"unavailable:{exc}"


def normalized_uuid(value: str) -> str:
    return value.strip().removeprefix("GPU-")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    physical_gpu = int(os.environ["GPU_WATCHER_PHYSICAL_GPU"])
    watcher_uuid = os.environ["GPU_WATCHER_GPU_UUID"]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    query = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(physical_gpu),
            "--query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    fields = [item.strip() for item in query.stdout.strip().splitlines()[0].split(",")]
    smi_uuid = fields[1]

    assert visible == str(physical_gpu), (visible, physical_gpu)
    assert smi_uuid == watcher_uuid, (smi_uuid, watcher_uuid)
    assert torch.cuda.is_available()
    assert torch.cuda.device_count() == 1
    props = torch.cuda.get_device_properties(0)
    torch_uuid = str(getattr(props, "uuid", ""))
    assert normalized_uuid(torch_uuid) == normalized_uuid(watcher_uuid), (torch_uuid, watcher_uuid)

    smoke_bytes = 4 * 1024 * 1024
    tensor = torch.empty(smoke_bytes, dtype=torch.uint8, device="cuda:0")
    tensor.zero_()
    torch.cuda.synchronize()
    del tensor
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    result = {
        "status": "pass",
        "physical_gpu": physical_gpu,
        "cuda_visible_devices": visible,
        "watcher_gpu_uuid": watcher_uuid,
        "nvidia_smi_gpu_uuid": smi_uuid,
        "torch_device": "cuda:0",
        "torch_device_count": torch.cuda.device_count(),
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda_version": str(torch.version.cuda),
        "torch_name": props.name,
        "torch_uuid": torch_uuid,
        "smoke_bytes": smoke_bytes,
        "smoke_synchronized": True,
        "smoke_released": True,
        "mount_namespace": namespace("/proc/self/ns/mnt"),
        "pid_namespace": namespace("/proc/self/ns/pid"),
        "user_namespace": namespace("/proc/self/ns/user"),
        "device_nodes_visible": all(Path(item).exists() for item in ("/dev/nvidiactl", f"/dev/nvidia{physical_gpu}")),
        "non_child_process_signaled": False,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
