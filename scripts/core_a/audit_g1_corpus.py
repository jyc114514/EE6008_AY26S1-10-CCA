#!/usr/bin/env python3
"""Build the required read-only G1 corpus audit artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ee6008.data.inventory import (
    audit_g1,
    sha256_file,
    write_csv,
    write_inventory_parquet,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--g1-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--canonical-dir", type=Path, required=True)
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.dry_run:
        print(
            json.dumps(
                {
                    "g1_root": str(args.g1_root),
                    "output_dir": str(args.output_dir),
                    "canonical_dir": str(args.canonical_dir),
                },
                indent=2,
            )
        )
        return 0
    summary = audit_g1(
        root=args.g1_root,
        ffprobe=args.ffprobe,
        manifest_path=args.manifest,
        limit=args.limit,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.canonical_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "G1_CORPUS_AUDIT_20260825.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    episodes = summary["episodes"]
    tasks = summary["tasks"]
    invalid = summary["invalid_episodes"]
    write_inventory_parquet(
        args.output_dir / "G1_EPISODE_INVENTORY_20260825.parquet", episodes
    )
    task_fields = list(tasks[0]) if tasks else ["original_task_index"]
    invalid_fields = (
        list(invalid[0])
        if invalid
        else list(episodes[0])
        if episodes
        else ["episode_index"]
    )
    write_csv(args.output_dir / "G1_TASK_INVENTORY_20260825.csv", tasks, task_fields)
    write_csv(
        args.output_dir / "G1_INVALID_EPISODES_20260825.csv", invalid, invalid_fields
    )
    write_inventory_parquet(args.canonical_dir / "episodes.parquet", episodes)
    write_csv(args.canonical_dir / "tasks.csv", tasks, task_fields)
    labels = {
        str(row["original_task_index"]): {
            "contiguous_label_index": index,
            "task_name_raw": row["task_name_raw"],
            "task_name_normalized": row["task_name_normalized"],
            "category_raw": row["category_raw"],
            "category_normalized": row["category_normalized"],
        }
        for index, row in enumerate(
            sorted(tasks, key=lambda item: item["original_task_index"])
        )
    }
    (args.canonical_dir / "label_map.json").write_text(
        json.dumps(labels, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    hashes = []
    for path in sorted(args.output_dir.glob("G1_*")) + sorted(
        args.canonical_dir.glob("*")
    ):
        if path.is_file() and path.name != "SHA256SUMS":
            hashes.append(f"{sha256_file(path)}  {path.name}")
    (args.canonical_dir / "SHA256SUMS").write_text(
        "\n".join(hashes) + "\n", encoding="utf-8"
    )
    report = [
        "# G1 corpus audit",
        "",
        f"- G1 root: `{args.g1_root}`",
        f"- Generated UTC: `{summary['generated_utc']}`",
        f"- Info total_tasks: `{summary['info'].get('total_tasks')}`",
        f"- tasks.jsonl rows: `{summary['counts']['tasks_jsonl_rows']}`",
        f"- episodes.jsonl rows: `{summary['counts']['episodes_jsonl_rows']}`",
        f"- Parquet files: `{summary['counts']['parquet_file_count']}`",
        f"- MP4 files: `{summary['counts']['video_file_count']}`",
        f"- Invalid episodes: `{summary['counts']['invalid_episode_count']}`",
        f"- Full-corpus status: `{summary['full_corpus_status']}`",
        "",
        "## Task-index reconciliation",
        "",
        "```json",
        json.dumps(summary["task_index_audit"], indent=2, sort_keys=True),
        "```",
        "",
        "## Core A/Core B fields",
        "",
        "The audit reads the original G1 Parquet fields and does not infer neural-network features from the `features` key in `info.json`.",
        "",
        f"- Core A RGB path: `{summary['info'].get('features', {}).get('observation.images.egocentric')}`",
        f"- Core A task field: `task_index`; unique episode task indices: `{summary['task_index_audit']['episode_unique_count']}`",
        f"- Core B action/arm/hand fields present: `{all(name in summary['core_a_fields_present'] for name in ['action', 'observation.arm_joints', 'observation.hand_joints'])}`",
        f"- Vector nonfinite counts: `{summary['vector_nonfinite_counts']}`",
        "",
        "## Output artifacts",
        "",
        "- `G1_CORPUS_AUDIT_20260825.json`",
        "- `G1_EPISODE_INVENTORY_20260825.parquet`",
        "- `G1_TASK_INVENTORY_20260825.csv`",
        "- `G1_INVALID_EPISODES_20260825.csv`",
        "- canonical manifest directory with `episodes.parquet`, `tasks.csv`, `label_map.json`, and `SHA256SUMS`",
    ]
    (args.output_dir / "G1_CORPUS_AUDIT_20260825.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "summary": str(summary_path),
                "episode_inventory": str(
                    args.output_dir / "G1_EPISODE_INVENTORY_20260825.parquet"
                ),
                "task_inventory": str(
                    args.output_dir / "G1_TASK_INVENTORY_20260825.csv"
                ),
                "invalid_episodes": str(
                    args.output_dir / "G1_INVALID_EPISODES_20260825.csv"
                ),
                "invalid_count": summary["counts"]["invalid_episode_count"],
                "full_corpus_status": summary["full_corpus_status"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if summary["full_corpus_status"] in {"pass", "review"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
