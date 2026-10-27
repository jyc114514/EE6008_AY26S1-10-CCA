#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys

REAL_NVIDIA_SMI = "/usr/bin/nvidia-smi"
ALLOWED_ID_ARGUMENT = "--id=2,3"
EXPECTED_QUERY = "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu"
EXPECTED_FORMAT = "--format=csv,noheader,nounits"


def build_command(argv: list[str]) -> list[str]:
    if argv != [EXPECTED_QUERY, EXPECTED_FORMAT]:
        raise ValueError("P1/P2 watcher adapter permits aggregate GPU2/GPU3 inventory fields only")
    return [REAL_NVIDIA_SMI, ALLOWED_ID_ARGUMENT, *argv]


if __name__ == "__main__":
    try:
        command = build_command(sys.argv[1:])
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(subprocess.run(command, check=False).returncode)
