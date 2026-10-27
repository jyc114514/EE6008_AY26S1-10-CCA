#!${EE6008_PROJECT_ROOT}/envs/jyc/bin/python
"""Watcher-only aggregate inventory filter for the user-authorized GPU2/3 target set."""

from __future__ import annotations

import csv
import io
import os
import subprocess
import sys

ALLOWED_PHYSICAL_GPUS = {2, 3}
EXPECTED_QUERY = "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu"
EXPECTED_FORMAT = "--format=csv,noheader,nounits"
REAL_NVIDIA_SMI = "/usr/bin/nvidia-smi"


def filter_aggregate_csv(text: str) -> str:
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    for fields in csv.reader(io.StringIO(text)):
        if not fields:
            continue
        values = [field.strip() for field in fields]
        if len(values) != 6:
            raise ValueError("unexpected aggregate nvidia-smi CSV schema")
        try:
            physical_gpu = int(values[0])
        except ValueError as exc:
            raise ValueError("invalid physical GPU index in aggregate row") from exc
        if physical_gpu in ALLOWED_PHYSICAL_GPUS:
            writer.writerow(values)
    return output.getvalue()


def main() -> int:
    args = sys.argv[1:]
    if os.environ.get("D3EQ_FILTER_TEST") == "1":
        sys.stdout.write(filter_aggregate_csv(sys.stdin.read()))
        return 0
    if args != [EXPECTED_QUERY, EXPECTED_FORMAT]:
        sys.stderr.write("D3-EQ GPU2/3 adapter accepts only the shared watcher's aggregate GPU query\n")
        return 2
    real = os.environ.get("D3EQ_REAL_NVIDIA_SMI", REAL_NVIDIA_SMI)
    result = subprocess.run([real, *args], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        sys.stderr.write(result.stderr)
        return result.returncode
    try:
        filtered = filter_aggregate_csv(result.stdout)
    except ValueError as exc:
        sys.stderr.write(f"D3-EQ GPU2/3 aggregate filter error: {exc}\n")
        return 2
    sys.stdout.write(filtered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
