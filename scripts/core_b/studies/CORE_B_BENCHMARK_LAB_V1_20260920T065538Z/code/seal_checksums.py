from __future__ import annotations

import hashlib
from pathlib import Path

from common import BENCH_ROOT


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    output = BENCH_ROOT / "CHECKSUMS.sha256"
    files = sorted(path for path in BENCH_ROOT.rglob("*") if path.is_file() and path != output and ".part" not in path.name)
    with output.open("w", encoding="utf-8") as handle:
        for path in files:
            handle.write(f"{sha256(path)}  {path.relative_to(BENCH_ROOT)}\n")
    print({"status": "PASS", "files": len(files), "checksum_path": str(output)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
