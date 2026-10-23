#!/usr/bin/env python3
"""Create a SHA256 manifest for every regular task output except itself."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

OUT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'reports/benchmarks/D3_EQ_REPORT_LEARNING_DIAGNOSIS_V1_20261006T022541Z'))
MANIFEST = OUT / "CHECKSUMS.sha256"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    files = []
    for path in OUT.rglob("*"):
        if path == MANIFEST or not path.is_file():
            continue
        if path.is_symlink():
            raise RuntimeError(f"symlink output is not allowed in checksum manifest: {path}")
        if not path.resolve().is_relative_to(OUT.resolve()):
            raise RuntimeError(f"output escapes OUT_ROOT: {path}")
        files.append(path)
    files.sort(key=lambda p: p.relative_to(OUT).as_posix())
    if not files:
        raise RuntimeError("no task outputs found")
    tmp = MANIFEST.with_name(f".{MANIFEST.name}.{os.getpid()}.part")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        for path in files:
            f.write(f"{digest(path)}  {path.relative_to(OUT).as_posix()}\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, MANIFEST)
    print(f"CHECKSUM_MANIFEST_WRITTEN files={len(files)} manifest={MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
