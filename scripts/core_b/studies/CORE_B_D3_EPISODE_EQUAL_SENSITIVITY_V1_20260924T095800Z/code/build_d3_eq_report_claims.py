#!/usr/bin/env python3
"""Rebuild the D3-EQ report claim registry from inline evidence markers."""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
from pathlib import Path

CLAIM_RE = re.compile(r"\[claim:(C[0-9]{3,8})\]")
EVIDENCE_RE = re.compile(r"\[evidence:((?:E[0-9]{3,8})(?:\s*,\s*E[0-9]{3,8})*)\]")
MARKER_RE = re.compile(r"\s*\[claim:C[0-9]{3,8}\]\s*|\s*\[evidence:(?:E[0-9]{3,8})(?:\s*,\s*E[0-9]{3,8})*\]\s*")

SECTION_META = {
    "执行结论": ("interpretive", "moderate", "exploratory"),
    "实验与评估合同": ("method", "not_applicable", "not_applicable"),
    "主要开发面板": ("result", "moderate", "exploratory"),
    "因子效应": ("result", "moderate", "exploratory"),
    "训练目标敏感性": ("result", "moderate", "descriptive"),
    "密集窗口结果": ("result", "moderate", "descriptive"),
    "D4 滚动更新": ("result", "moderate", "descriptive"),
    "D2-V3 固定阈值事件次级评估": ("result", "moderate", "descriptive"),
    "资源与失败记录": ("factual", "not_estimated", "not_applicable"),
    "决策和解释边界": ("interpretive", "moderate", "descriptive"),
    "图表与交付审查": ("factual", "not_estimated", "not_applicable"),
}


def build(report: Path, output: Path) -> int:
    text = report.read_text(encoding="utf-8")
    section = "执行结论"
    seen: set[str] = set()
    rows: list[dict[str, str]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if line.startswith("## "):
            section = line[3:].strip()
        claim_ids = CLAIM_RE.findall(line)
        if not claim_ids:
            continue
        if len(claim_ids) != 1 or section not in SECTION_META:
            raise ValueError(f"invalid claim marker or unmapped section at line {line_number}")
        claim_id = claim_ids[0]
        if claim_id in seen:
            raise ValueError(f"duplicate claim id {claim_id}")
        evidence_groups = EVIDENCE_RE.findall(line)
        if len(evidence_groups) != 1:
            raise ValueError(f"expected one evidence marker for {claim_id}")
        evidence_ids = [item.strip() for item in evidence_groups[0].split(",")]
        if not evidence_ids:
            raise ValueError(f"empty evidence set for {claim_id}")
        claim_text = " ".join(MARKER_RE.sub(" ", line).split())
        digest = hashlib.sha256(claim_text.encode("utf-8")).hexdigest()
        claim_kind, uncertainty, analysis_intent = SECTION_META[section]
        seen.add(claim_id)
        rows.append(
            {
                "claim_id": claim_id,
                "section": section,
                "claim_kind": claim_kind,
                "claim_text_sha256": digest,
                "evidence_ids": ";".join(evidence_ids),
                "verification_status": "unverified",
                "uncertainty": uncertainty,
                "analysis_intent": analysis_intent,
            }
        )
    if not rows:
        raise ValueError("no claim markers found")
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "claim_id",
                "section",
                "claim_kind",
                "claim_text_sha256",
                "evidence_ids",
                "verification_status",
                "uncertainty",
                "analysis_intent",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(f"claims_written={build(args.report, args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
