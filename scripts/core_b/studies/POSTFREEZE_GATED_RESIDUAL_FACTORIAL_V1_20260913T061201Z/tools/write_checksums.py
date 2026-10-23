from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    output = root / "qa/CHECKSUMS.sha256"
    files = sorted(path for path in root.rglob("*") if path.is_file() and path != output and "/__pycache__/" not in str(path) and "/.pytest_cache/" not in str(path))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(f"{digest(path)}  {path.relative_to(root)}" for path in files) + "\n", encoding="utf-8")
    print(f"wrote {len(files)} checksums to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
