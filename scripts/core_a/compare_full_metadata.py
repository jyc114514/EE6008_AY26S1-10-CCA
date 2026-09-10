"""Validate the metadata-only full-repository snapshot and compare it to G1."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g1-root", type=Path, required=True)
    parser.add_argument("--full-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--audit-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--g1-revision")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    expected = {item["path"]: int(item["size"]) for item in plan["files"]}
    all_actual_paths = [
        p.relative_to(args.full_root).as_posix()
        for p in args.full_root.rglob("*")
        if p.is_file()
    ]
    partial_paths = sorted(
        path for path in all_actual_paths if path.endswith(".part")
    )
    actual_paths = [
        path for path in all_actual_paths if not path.endswith(".part")
    ]
    forbidden = sorted(
        path for path in actual_paths if path.startswith(("data/", "videos/"))
    )
    size_mismatches = []
    missing = []
    for rel, size in expected.items():
        path = args.full_root / rel
        if not path.exists():
            missing.append(rel)
        elif path.stat().st_size != size:
            size_mismatches.append(
                {"path": rel, "expected": size, "actual": path.stat().st_size}
            )
    invalid_partials = []
    for partial in partial_paths:
        target = partial[: -len(".part")]
        target_path = args.full_root / target
        if target not in expected:
            invalid_partials.append(
                {"path": partial, "reason": "no_expected_destination"}
            )
        elif not target_path.exists() or target_path.stat().st_size != expected[target]:
            invalid_partials.append(
                {"path": partial, "reason": "destination_missing_or_size_mismatch"}
            )
    unexpected = sorted(set(actual_paths) - set(expected))
    if forbidden or missing or size_mismatches or invalid_partials or unexpected:
        raise SystemExit(
            json.dumps(
                {
                    "forbidden": forbidden[:10],
                    "missing_count": len(missing),
                    "size_mismatch_count": len(size_mismatches),
                    "partial_count": len(partial_paths),
                    "invalid_partial_count": len(invalid_partials),
                    "unexpected_count": len(unexpected),
                },
                indent=2,
            )
        )

    full_info = json.loads(
        (args.full_root / "meta/info.json").read_text(encoding="utf-8")
    )
    full_tasks = load_jsonl(args.full_root / "meta/tasks.jsonl")
    full_episodes = load_jsonl(args.full_root / "meta/episodes.jsonl")
    full_episode_stats = load_jsonl(args.full_root / "meta/episodes_stats.jsonl")
    full_errors = load_jsonl(args.full_root / "meta/errors.jsonl")
    g1_info = json.loads((args.g1_root / "meta/info.json").read_text(encoding="utf-8"))
    audit = json.loads(args.audit_json.read_text(encoding="utf-8"))
    resolved_revision = plan.get("resolved_revision") or plan.get("revision")
    if not resolved_revision:
        raise SystemExit("metadata plan has no resolved_revision or revision")
    g1_revision = args.g1_revision or audit.get("dataset_revision") or audit.get(
        "revision"
    )
    if not g1_revision:
        raise SystemExit("G1 revision was not supplied or recorded in the audit")

    def feature_names(info: dict[str, Any]) -> list[str]:
        return sorted((info.get("features") or {}).keys())

    full_features = feature_names(full_info)
    g1_features = feature_names(g1_info)
    full_modality_tokens = sorted(
        {
            token
            for name in full_features
            for token in (name.split(".")[1] if "." in name else name,)
        }
    )
    g1_modality_tokens = sorted(
        {
            token
            for name in g1_features
            for token in (name.split(".")[1] if "." in name else name,)
        }
    )
    lock = {
        "status": "pass",
        "repo_id": plan["repo_id"],
        "resolved_revision": resolved_revision,
        "g1_revision": g1_revision,
        "source_revision_requested": plan.get("source_revision"),
        "local_root": str(args.full_root),
        "allowed_prefixes": plan["allowed_prefixes"],
        "forbidden_prefixes": ["data/", "videos/"],
        "file_count": len(expected),
        "total_bytes": sum(expected.values()),
        "actual_file_count": len(actual_paths),
        "retained_partial_file_count": len(partial_paths),
        "retained_partial_files": partial_paths,
        "downloaded_data_or_videos": False,
        "metadata_counts": {
            "root_readme": int((args.full_root / "README.md").exists()),
            "meta": sum(path.startswith("meta/") for path in actual_paths),
            "resume_meta": sum(
                path.startswith("resume_meta/") for path in actual_paths
            ),
        },
        "parsed": {
            "info": True,
            "tasks": len(full_tasks),
            "episodes": len(full_episodes),
            "episodes_stats": len(full_episode_stats),
            "errors": len(full_errors),
        },
        "full_info_summary": {
            key: full_info.get(key)
            for key in (
                "robot_type",
                "total_tasks",
                "total_episodes",
                "total_frames",
                "fps",
                "total_videos",
            )
        },
        "sha256_manifest": None,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    sha_path = args.output_dir / "FULL_METADATA_SHA256SUMS_20260825.txt"
    with sha_path.open("w", encoding="utf-8") as handle:
        for rel in sorted(expected):
            handle.write(f"{sha256_file(args.full_root / rel)}  {rel}\n")
    lock["sha256_manifest"] = str(sha_path)
    (args.output_dir / "FULL_DATASET_METADATA_LOCK_20260825.json").write_text(
        json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    comparison_lines = [
        "# G1 vs full-repository metadata comparison (2026-08-25)",
        "",
        "## Scope and revisions",
        "",
        f"- G1 local snapshot: `{g1_info.get('robot_type')}`, immutable revision `{g1_revision}`.",
            f"- Full metadata-only snapshot: `{plan['repo_id']}` at immutable revision `{resolved_revision}`.",
        "- Full snapshot contains only README/meta/resume_meta; `data/` and `videos/` were explicitly excluded.",
        "",
        "## Audited summary",
        "",
        "| field | G1 | full repository |",
        "|---|---:|---:|",
    ]
    for key in (
        "robot_type",
        "total_tasks",
        "total_episodes",
        "total_frames",
        "fps",
        "total_videos",
    ):
        comparison_lines.append(
            f"| `{key}` | `{g1_info.get(key)}` | `{full_info.get(key)}` |"
        )
    comparison_lines.extend(
        [
            "",
            f"- G1 feature count: {len(g1_features)}; full feature count: {len(full_features)}.",
            f"- G1 modality tokens observed: `{', '.join(g1_modality_tokens)}`.",
            f"- Full modality tokens observed: `{', '.join(full_modality_tokens)}`.",
            f"- Full task rows parsed: {len(full_tasks)}; episode rows parsed: {len(full_episodes)}; episode-stat rows parsed: {len(full_episode_stats)}.",
            f"- Full metadata error rows parsed: {len(full_errors)}.",
            f"- Retained `.part` files validated and ignored: {len(partial_paths)}; they were not deleted.",
            "",
            "## Interpretation",
            "",
            "The full repository is a mixed-robot, richer-modality corpus. It is not a larger V-JEPA feature cache. The local G1 snapshot already provides the RGB video, episode/task metadata, joint state, and action fields needed by the current Core A RGB pipeline. H1, depth, LiDAR, IMU, odometry, tactile, and the full data/video trees remain outside the current single-robot scope.",
            "",
            "Task labels and aggregate counts are treated as metadata claims until reconciled with episode references; no full-repository statistical test split is inferred from these summaries.",
        ]
    )
    (args.output_dir / "G1_VS_FULL_METADATA_COMPARISON_20260825.md").write_text(
        "\n".join(comparison_lines) + "\n", encoding="utf-8"
    )

    assets = [
        (
            "G1 RGB egocentric H.264 videos",
            "present",
            "present",
            "yes",
            "use current G1 snapshot",
        ),
        (
            "G1 parquet state/action/timestamp",
            "present",
            "present",
            "yes",
            "use current G1 snapshot",
        ),
        (
            "G1 episode/task metadata",
            "present",
            "present",
            "yes",
            "use current G1 snapshot",
        ),
        (
            "H1 episodes and videos",
            "full metadata only",
            "not present",
            "no",
            "do not download for current G1 charter",
        ),
        (
            "depth",
            "full metadata only",
            "not present",
            "no",
            "not required by Core A RGB",
        ),
        (
            "LiDAR",
            "full metadata only",
            "not present",
            "no",
            "not required by Core A RGB",
        ),
        (
            "IMU/odometry/tactile",
            "full metadata only",
            "not present",
            "no",
            "not required by Core A RGB",
        ),
        (
            "full data/ tree",
            "not downloaded by policy",
            "not present",
            "no",
            "935 GB scope excluded",
        ),
        (
            "full videos/ tree",
            "not downloaded by policy",
            "not present",
            "no",
            "scope excluded; local G1 RGB is sufficient",
        ),
        (
            "V-JEPA 2-AC weights",
            "not requested",
            "not present",
            "no",
            "explicitly deferred",
        ),
    ]
    with (args.output_dir / "MISSING_G1_ASSETS_MANIFEST_20260825.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "asset",
                "full_repository_status",
                "local_g1_status",
                "needed_for_current_core_a",
                "decision",
            ]
        )
        writer.writerows(assets)
    print(json.dumps(lock, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
