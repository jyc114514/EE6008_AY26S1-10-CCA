#!/usr/bin/env python3
"""Run deterministic safety and structure checks on the public Core A clone."""

from __future__ import annotations

import ast
import csv
import json
import re
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "docs/core_a/publication"
FORBIDDEN_SUFFIXES = {
    ".pt", ".pth", ".ckpt", ".bin", ".safetensors", ".npz", ".npy",
    ".parquet", ".mp4", ".avi", ".mov", ".mkv", ".zip", ".7z", ".tar",
}
FORBIDDEN_TEXT = re.compile(
    r"/" + "usr1" + r"/|" + "s126" + "mdg33|" + "gpu" + "33|"
    r"BEGIN [A-Z ]*PRIVATE KEY|password\s*=|api[_-]?key\s*=",
    re.IGNORECASE,
)
REQUIRED = (
    ROOT / "docs/core_a/CURRENT_CONCLUSIONS.md",
    ROOT / "experiments/core_a/EXPERIMENT_CATALOG.csv",
    ROOT / "experiments/core_a/RUN_LEVEL_RESULTS.csv",
    ROOT / "results/core_a/figure_manifest.json",
)


def json_values(value: Any, key: str = "") -> list[tuple[str, Any]]:
    values: list[tuple[str, Any]] = []
    if isinstance(value, dict):
        for name, child in value.items():
            values.extend(json_values(child, name))
    elif isinstance(value, list):
        for child in value:
            values.extend(json_values(child, key))
    else:
        values.append((key, value))
    return values


def main() -> int:
    failures: list[str] = []
    warnings: list[str] = []
    files = [p for p in ROOT.rglob("*") if p.is_file() and ".git" not in p.parts]
    for required in REQUIRED:
        if not required.is_file():
            failures.append(f"missing required file: {required.relative_to(ROOT)}")
    for path in files:
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            failures.append(f"prohibited artifact extension: {path.relative_to(ROOT)}")
        if path.stat().st_size > 10 * 1024 * 1024:
            failures.append(f"file exceeds 10 MiB: {path.relative_to(ROOT)}")
        if path.name in {".env", ".env.local", "credentials.json"} or path.suffix.lower() in {".pem", ".key"}:
            failures.append(f"credential-like file: {path.relative_to(ROOT)}")
        if path.suffix.lower() in {".py", ".sh", ".yaml", ".yml", ".json", ".csv", ".md", ".txt", ".toml"}:
            text = path.read_text(encoding="utf-8", errors="replace")
            if path.name != "validate_public_release.py" and FORBIDDEN_TEXT.search(text):
                failures.append(f"server/credential pattern: {path.relative_to(ROOT)}")
            if path.suffix.lower() == ".py":
                try:
                    ast.parse(text, filename=str(path))
                except SyntaxError as exc:
                    failures.append(f"Python parse failure {path.relative_to(ROOT)}: {exc}")
            if path.suffix.lower() == ".json":
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError as exc:
                    failures.append(f"JSON parse failure {path.relative_to(ROOT)}: {exc}")
                else:
                    for key, value in json_values(payload):
                        if key in {"final_test_access", "final_test_accessed"} and str(value).lower() not in {"0", "false", "none"}:
                            failures.append(f"nonzero final-test access in {path.relative_to(ROOT)}")
            if path.suffix.lower() in {".yaml", ".yml"}:
                try:
                    yaml.safe_load(text)
                except yaml.YAMLError as exc:
                    failures.append(f"YAML parse failure {path.relative_to(ROOT)}: {exc}")
            if path.suffix.lower() == ".csv":
                try:
                    with path.open(encoding="utf-8", newline="") as handle:
                        rows = list(csv.reader(handle))
                    if rows:
                        width = len(rows[0])
                        if width == 0 or any(len(row) != width for row in rows):
                            failures.append(f"CSV column mismatch: {path.relative_to(ROOT)}")
                        header = rows[0]
                        key_fields: tuple[str, ...] | None = None
                        custom_keys: list[tuple[str, ...]] | None = None
                        if path.name == "COMPARISON_EFFECTS.csv" and {"comparison_scope", "comparison_id", "head"}.issubset(header):
                            scope_index = header.index("comparison_scope")
                            comparison_index = header.index("comparison_id")
                            head_index = header.index("head")
                            custom_keys = [
                                (row[scope_index], row[comparison_index] or row[head_index])
                                for row in rows[1:]
                            ]
                        elif "experiment_id" in header:
                            key_fields = ("experiment_id",)
                        elif "comparison_id" in header:
                            key_fields = ("comparison_id",)
                        elif "figure_id" in header:
                            key_fields = ("figure_id",)
                        elif "run_id" in header:
                            key_fields = ("run_id",)
                        elif {"cell", "metric"}.issubset(header):
                            key_fields = ("cell", "metric")
                        elif {"cell", "bin"}.issubset(header):
                            key_fields = ("cell", "bin")
                        elif {"cell", "support_bucket"}.issubset(header):
                            key_fields = ("cell", "support_bucket")
                        if custom_keys is not None:
                            if any(not all(value for value in item) for item in custom_keys):
                                failures.append(f"empty primary key in {path.relative_to(ROOT)}")
                            if len(custom_keys) != len(set(custom_keys)):
                                failures.append(f"duplicate primary key in {path.relative_to(ROOT)}")
                        elif key_fields:
                            indices = [header.index(key) for key in key_fields]
                            values = [tuple(row[index] for index in indices) for row in rows[1:]]
                            if any(not all(value for value in item) for item in values):
                                failures.append(f"empty primary key in {path.relative_to(ROOT)}")
                            if len(values) != len(set(values)):
                                failures.append(f"duplicate primary key in {path.relative_to(ROOT)}")
                except (OSError, UnicodeError) as exc:
                    failures.append(f"CSV read failure {path.relative_to(ROOT)}: {exc}")

    manifest_path = ROOT / "results/core_a/figure_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for entry in manifest:
            for key in ("png", "svg", "pdf", "data_csv"):
                target = ROOT / "results/core_a" / entry[key]
                if not target.is_file() or target.stat().st_size == 0:
                    failures.append(f"figure manifest target missing: {entry.get('figure_id')}:{key}")
            script = ROOT / entry["plot_script"]
            if not script.is_file():
                failures.append(f"figure plot script missing: {entry.get('figure_id')}")

    result_files = [p for p in (ROOT / "results/core_a").rglob("*") if p.is_file()]
    if any("core_b" in p.as_posix().lower() for p in result_files):
        failures.append("Core B result path found under results/core_a")
    if any(p.name == ".git" and p != ROOT / ".git" for p in ROOT.rglob("*")):
        failures.append("nested Git directory found")
    if not failures:
        warnings.append("End-to-end model replay is not part of this QA; external inputs remain excluded.")

    report = {
        "status": "passed" if not failures else "failed",
        "files_checked": len(files),
        "python_files_parsed": sum(p.suffix.lower() == ".py" for p in files),
        "json_files_parsed": sum(p.suffix.lower() == ".json" for p in files),
        "yaml_files_parsed": sum(p.suffix.lower() in {".yaml", ".yml"} for p in files),
        "csv_files_checked": sum(p.suffix.lower() == ".csv" for p in files),
        "figures_checked": len(json.loads(manifest_path.read_text(encoding="utf-8"))) if manifest_path.is_file() else 0,
        "final_test_access": 0,
        "new_training_started": False,
        "failures": failures,
        "warnings": warnings,
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "QA_REPORT.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# Public Core A release QA", "", f"Status: **{report['status']}**.", "", f"- Files checked: `{len(files)}`", f"- Python AST parses: `{report['python_files_parsed']}`", f"- JSON parses: `{report['json_files_parsed']}`", f"- YAML parses: `{report['yaml_files_parsed']}`", f"- CSV checks: `{report['csv_files_checked']}`", f"- Figures checked: `{report['figures_checked']}`", "- Final-test access: `0`", "- New training started by this QA: `false`", ""]
    if failures:
        lines.extend(["## Failures", "", *[f"- {item}" for item in failures], ""])
    if warnings:
        lines.extend(["## Notes", "", *[f"- {item}" for item in warnings], ""])
    (REPORT_DIR / "QA_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
