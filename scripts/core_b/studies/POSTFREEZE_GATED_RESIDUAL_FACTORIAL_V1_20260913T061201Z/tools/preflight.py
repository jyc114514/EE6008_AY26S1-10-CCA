from __future__ import annotations
import os

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
REPO = PROJECT / "repo/ee6008-humanoid-vjepa"
JY02 = REPO / "reports/core_b/JY02_RESIDUAL_TEMPORAL_DECODER_V1_20260912T095449Z"
STAGE2 = JY02 / "runs/stage2_authorized_20260912T151855Z"
FREEZE = REPO / "reports/core_b/JY02_CANDIDATE_FREEZE_AND_HELDOUT_PROTOCOL_V1_20260913T053157Z"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checksum(path: Path, cwd: Path, expected: int) -> dict:
    r = subprocess.run(["sha256sum", "-c", str(path.relative_to(cwd))], cwd=cwd, text=True, capture_output=True, check=False)
    lines = (r.stdout + r.stderr).splitlines()
    ok = sum(line.endswith(": OK") for line in lines)
    failed = sum(line.endswith(": FAILED") for line in lines)
    if r.returncode != 0 or ok != expected or failed:
        raise RuntimeError(f"checksum failed: {path} exit={r.returncode} ok={ok} failed={failed}")
    return {"path": str(path), "sha256": digest(path), "exit_code": r.returncode, "ok": ok, "failed": failed, "status": "PASS"}


def main() -> None:
    stage2_check = checksum(STAGE2 / "qa/CHECKSUMS.sha256", STAGE2, 284)
    freeze_check = checksum(FREEZE / "qa/CHECKSUMS.sha256", FREEZE, 85)
    stage0 = json.loads((ROOT / "STAGE0_RESULTS.json").read_text())
    protocol = json.loads((ROOT / "EXPLORATION_PROTOCOL_LOCKED.json").read_text())
    schedule = json.loads((ROOT / "RUN_SCHEDULE.json").read_text())
    if stage0.get("status") != "PASS" or stage0.get("accepted") != 8:
        raise RuntimeError("Stage0 is not 8/8 PASS")
    if protocol.get("status") != "LOCKED_BEFORE_TRAINING" or protocol.get("exploratory_seed") != 20260924 or protocol.get("final_test_explicitly_authorized") is not False:
        raise RuntimeError("exploration protocol is not locked fail-closed")
    if len(schedule["rows"]) != 8 or any(row["physical_gpu"] not in [2, 3] for row in schedule["rows"]):
        raise RuntimeError("run schedule violates watcher allowlist")
    for row in schedule["rows"]:
        job = json.loads(Path(row["job_path"]).read_text())
        if job["resource_estimate"]["final_test_access"] != 0 or job["physical_gpu"] == 0:
            raise RuntimeError("job violates final/GPU0 boundary")
    result = {
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "stage2_checksum": stage2_check,
        "freeze_checksum": freeze_check,
        "stage0": {"status": stage0["status"], "accepted": stage0["accepted"], "expected": stage0["expected"]},
        "exploratory_seed": 20260924,
        "schedule_rows": 8,
        "schedule_gpus": [2, 3],
        "max_concurrent_gpu_children": 2,
        "gpu0_used": False,
        "final_test_access": 0,
        "new_lora_or_encoder_training": False,
        "freeze_hashes_before_training": {name: digest(FREEZE / name) for name in ["qa/CHECKSUMS.sha256", "qa/FINAL_QA.json", "FINAL_MODEL_REGISTRY.json", "FINAL_EVALUATION_EXECUTION_PROMPT_PENDING_APPROVAL.md"]},
    }
    (ROOT / "logs/PRELAUNCH_AUDIT.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (ROOT / "logs/PRELAUNCH_AUDIT.md").write_text("# Prelaunch audit\n\nStage2 `284/284` and freeze `85/85` checksums passed; Stage0 is `8/8`; the eight jobs are locked to GPU2/3 with GPU0 excluded and final-test access zero. Freeze hashes are recorded before full training.\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "stage2": 284, "freeze": 85, "stage0": "8/8", "jobs": 8}, sort_keys=True))


if __name__ == "__main__":
    main()
