from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import torch  # noqa: E402
import run_orchestrator  # noqa: E402
import train_full_query  # noqa: E402


def test_training_query_loader_uses_canonical_archive_sha_and_h30_hash() -> None:
    table, table_sha = train_full_query.load_candidate_query()
    metadata = json.loads((ROOT / "protocol/QUERY_TABLES_AND_HASHES.json").read_text())
    archive = ROOT / "protocol/HADAMARD_QUERY_TABLES.npz"
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == metadata["archive_sha256"]
    assert table.shape == (30, 128)
    assert table_sha == metadata["candidate_table_sha256_h30"]
    assert torch.cuda.is_initialized() is False


def test_retry_counter_counts_one_watcher_start_per_attempt(tmp_path: Path) -> None:
    ledger = tmp_path / "JOB_LEDGER.jsonl"
    rows = [
        {"event": "watcher_started", "job_id": "cell-a", "attempt": 1},
        {"event": "watcher_finished", "job_id": "cell-a", "attempt": 1},
    ]
    ledger.write_text("\n".join(json.dumps(x) for x in rows) + "\n")
    assert run_orchestrator.attempt_counts_from_ledger(ledger) == {"cell-a": 1}
