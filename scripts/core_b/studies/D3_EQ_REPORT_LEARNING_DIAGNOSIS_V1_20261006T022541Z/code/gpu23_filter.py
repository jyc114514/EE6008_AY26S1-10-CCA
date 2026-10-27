#!/usr/bin/env python3
"""Aggregate-only physical GPU2/3 filter used by the shared watcher."""
from __future__ import annotations

import csv
import io
import os
import subprocess
import sys

ALLOWED_PHYSICAL_GPUS = frozenset({2, 3})
EXPECTED_QUERY = "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu"
EXPECTED_FORMAT = "--format=csv,noheader,nounits"
REAL_NVIDIA_SMI = "/usr/bin/nvidia-smi"


def filter_csv(text: str) -> str:
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    for fields in csv.reader(io.StringIO(text)):
        if not fields:
            continue
        fields = [field.strip() for field in fields]
        if len(fields) != 6:
            raise ValueError("unexpected aggregate nvidia-smi row schema")
        try:
            index = int(fields[0])
        except ValueError as exc:
            raise ValueError("invalid physical GPU index") from exc
        if index in ALLOWED_PHYSICAL_GPUS:
            writer.writerow(fields)
    return output.getvalue()


def main() -> int:
    args = sys.argv[1:]
    if os.environ.get("D3EQ_TASK_FILTER_TEST") == "1":
        sys.stdout.write(filter_csv(sys.stdin.read()))
        return 0
    if args != [EXPECTED_QUERY, EXPECTED_FORMAT]:
        sys.stderr.write("task filter accepts only the shared watcher's aggregate GPU query\n")
        return 2
    result = subprocess.run([REAL_NVIDIA_SMI, *args], capture_output=True, text=True, check=False)
    if result.returncode:
        sys.stderr.write(result.stderr)
        return result.returncode
    try:
        sys.stdout.write(filter_csv(result.stdout))
    except ValueError as exc:
        sys.stderr.write(f"task GPU2/3 filter: {exc}\n")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
