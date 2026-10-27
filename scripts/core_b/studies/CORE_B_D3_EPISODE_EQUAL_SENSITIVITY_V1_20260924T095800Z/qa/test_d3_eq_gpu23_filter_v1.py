"""CPU-only test for the watcher inventory's explicit physical GPU2/3 filter."""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys

EQ_ROOT = pathlib.Path(__file__).resolve().parents[1]
FILTER = EQ_ROOT / "code/d3eq_gpu23_bin/gpu23_filter.py"


def main() -> int:
    fixture = (
        "0, GPU-0, NVIDIA GeForce RTX 3090, 100, 24576, 100\n"
        "1, GPU-1, NVIDIA GeForce RTX 3090, 200, 24576, 75\n"
        "2, GPU-2, NVIDIA GeForce RTX 3090, 300, 24576, 10\n"
        "3, GPU-3, NVIDIA GeForce RTX 3090, 400, 24576, 5\n"
    )
    environment = os.environ.copy()
    environment["D3EQ_FILTER_TEST"] = "1"
    result = subprocess.run([sys.executable, str(FILTER), "fixture-mode"], input=fixture, capture_output=True, text=True, env=environment, check=True)
    indices = [int(line.split(",", 1)[0]) for line in result.stdout.splitlines()]
    if indices != [2, 3]:
        raise AssertionError(f"physical GPU allowlist mismatch: {indices}")
    print('{"status":"PASS","fixture_rows_in":4,"rows_visible_to_shared_watcher":[2,3],"gpu_query_performed":false}')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
