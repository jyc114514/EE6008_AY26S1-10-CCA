from __future__ import annotations
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
MANIFEST=ROOT/"CHECKSUMS.sha256"
RECEIPT=ROOT/"qa/FINAL_CHECKSUM_VERIFICATION.json"
V7=ROOT.parent/"CORE_B_V7_SINGLE_HEAD_SOFT_TARGET_V1_20261015T112823Z"


def now(): return datetime.now(timezone.utc).isoformat().replace("+00:00","Z")
def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding="utf-8"))


def preseal_gate():
    if MANIFEST.exists() or RECEIPT.exists():
        raise FileExistsError("refusing to overwrite a V8 checksum manifest or read-back receipt")
    required=[
      "inventory/SOURCE_MAP.json","inventory/EXISTING_DIAGNOSTIC_COVERAGE.csv",
      "protocol/ANALYSIS_LOCK.json","protocol/PREDICTION_REUSE_AND_REPLAY.json",
      "results/V7_CONTEXT_AND_TIMELINE_ADDENDUM.md","results/TRAIN_TARGET_AND_FIT.csv",
      "results/TRAIN_CURVE_SUMMARY.csv","results/TRAIN_DEV_IMITATION.csv",
      "results/CORRECTION_GEOMETRY.csv","results/HORIZON_AXIS_ERRORS.csv",
      "results/TEACHER_HELP_HARM.csv","results/SEED_BLOCK_LOO.csv","results/TASK_LOO.csv",
      "results/DECISION_TABLE.csv","results/V7_RESOURCE_SUPPLEMENT.csv",
      "results/V8_RESOURCE_AND_EXPOSURE.csv","RETURN_TO_CHATGPT.zh-CN.md",
      "RETURN_TO_CHATGPT.json","qa/SYNTHETIC_QA.json","qa/SOURCE_AND_ID_QA.json",
      "qa/REAL_DATA_QA.json","qa/ALIAS_QA.json","qa/FINAL_ACCEPTANCE.json",
      "qa/FIGURE_REVIEW.json","coordination/EXTRACT_COMPLETE.json",
      "coordination/FROZEN_FORWARD_EXECUTION.json","coordination/EXTRACTION_ATTEMPTS.json",
      "figure_data/FIGURE_EXPORT_METADATA.json"]
    missing=[rel for rel in required if not (ROOT/rel).is_file()]
    if missing: raise FileNotFoundError(f"required V8 artifacts missing: {missing}")
    final=read(ROOT/"qa/FINAL_ACCEPTANCE.json")
    real=read(ROOT/"qa/REAL_DATA_QA.json")
    replay=read(ROOT/"protocol/PREDICTION_REUSE_AND_REPLAY.json")
    qa=read(ROOT/"qa/SYNTHETIC_QA.json")
    real_qa=read(ROOT/"qa/EXTRACT_SYNTHETIC_QA.json")
    execution=read(ROOT/"coordination/FROZEN_FORWARD_EXECUTION.json")
    if final.get("status")!="complete_with_limits" or real.get("status")!="complete_with_limits":
        raise RuntimeError("final acceptance or real data QA is not complete_with_limits")
    if qa.get("status")!="PASS" or real_qa.get("status")!="PASS":
        raise RuntimeError("synthetic QA is not PASS")
    if execution.get("status")!="PASS" or execution.get("jobs_accepted")!=6:
        raise RuntimeError("six frozen train forwards are not all accepted")
    if replay.get("new_dev_prediction_jobs")!=0 or replay.get("new_teacher_forward_jobs")!=0:
        raise RuntimeError("replay receipt violates zero new dev/teacher prediction boundary")
    if real.get("dev_target_archive_loads_total")!=1 or real.get("final_heldout_payload_access")!=0:
        raise RuntimeError("real data QA target/heldout counts disagree with the lock")
    if not (ROOT/"results/HORIZON_AXIS_ERRORS.csv").is_file():
        raise FileNotFoundError("explicit horizon/axis unavailable table missing")
    horizon=(ROOT/"results/HORIZON_AXIS_ERRORS.csv").read_text(encoding="utf-8")
    if "UNAVAILABLE" not in horizon:
        raise RuntimeError("horizon/axis limitation is not explicit")
    if list(ROOT.rglob("*.npz")) or list(ROOT.rglob("Y_NATIVE.npy")):
        raise RuntimeError("raw archives/target arrays must not be persisted in V8")
    figure_meta=read(ROOT/"figure_data/FIGURE_EXPORT_METADATA.json")
    if len(figure_meta.get("outputs",[]))!=4:
        raise RuntimeError("expected four provisional figure outputs")
    for fig in figure_meta["outputs"]:
        fp=Path(fig["path"])
        if not fp.is_file() or sha(fp)!=fig["sha256"]:
            raise RuntimeError(f"figure metadata/hash mismatch: {fp}")
    with (ROOT/"results/V7_RESOURCE_SUPPLEMENT.csv").open(newline="",encoding="utf-8") as f:
        import csv
        supplement=list(csv.DictReader(f))
    if len(supplement)!=9:
        raise RuntimeError("V7 resource supplement must have nine manifest-bound rows")
    with (ROOT/"results/V8_RESOURCE_AND_EXPOSURE.csv").open(newline="",encoding="utf-8") as f:
        resource=list(csv.DictReader(f))
    def count(metric,phase):
        values=[int(r["count"]) for r in resource if r["metric"]==metric and r["phase"]==phase]
        if len(values)!=1: raise RuntimeError(f"resource row missing/duplicate: {metric}/{phase}")
        return values[0]
    if count("dev_target_archive_loads","V8_total")!=1:
        raise RuntimeError("dev target archive load total differs from actual attempt ledger")
    if count("prediction_native_arrays_loaded","V8_total")!=100:
        raise RuntimeError("prediction tensor load count differs from the attempt ledger")
    if count("new_dev_prediction_jobs","V8_actual")!=0:
        raise RuntimeError("V8 new dev prediction count must be zero")
    v7_receipt=read(V7/"qa/FINAL_CHECKSUM_VERIFICATION.json")
    if v7_receipt.get("status")!="PASS" or sha(V7/"CHECKSUMS.sha256")!=v7_receipt.get("manifest_sha256"):
        raise RuntimeError("V7 sealed manifest changed")
    return {"required_artifacts":len(required),"V7_manifest_sha256":v7_receipt["manifest_sha256"]}


def main():
    gate=preseal_gate()
    files=[]
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file(): continue
        if path.is_symlink(): raise RuntimeError(f"symlink excluded by integrity gate: {path}")
        if path==MANIFEST or path==RECEIPT: continue
        rel=path.relative_to(ROOT).as_posix()
        files.append((sha(path),rel))
    body="".join(f"{digest}  {rel}\n" for digest,rel in files)
    MANIFEST.write_text(body,encoding="utf-8")
    manifest_sha=sha(MANIFEST)
    verified=0;missing=[];mismatches=[]
    for digest,rel in files:
        path=ROOT/rel
        if not path.is_file(): missing.append(rel);continue
        actual=sha(path)
        if actual!=digest:mismatches.append({"path":rel,"expected":digest,"actual":actual})
        else:verified+=1
    receipt={
      "schema_version":1,"status":"PASS" if not missing and not mismatches else "FAIL",
      "created_utc":now(),"manifest_path":"CHECKSUMS.sha256",
      "manifest_entry_count":len(files),"verified_entry_count":verified,
      "missing":missing,"mismatches":mismatches,
      "manifest_sha256":manifest_sha,
      "excluded_from_manifest":[
        {"path":"CHECKSUMS.sha256","reason":"manifest cannot hash itself"},
        {"path":"qa/FINAL_CHECKSUM_VERIFICATION.json","reason":"post-manifest read-back receipt"}],
      "V7_manifest_sha256":gate["V7_manifest_sha256"],
      "V7_old_root_writes":0,"V8_final_heldout_payload_access":0,
      "preseal_gate_artifacts":gate["required_artifacts"]
    }
    RECEIPT.write_text(json.dumps(receipt,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    # Read back the excluded receipt only; no files are written after this point.
    check=read(RECEIPT)
    if check.get("status")!="PASS": raise RuntimeError("V8 manifest read-back failed")
    print(json.dumps({"status":check["status"],"manifest_entries":check["manifest_entry_count"],
                      "verified_entries":check["verified_entry_count"],
                      "manifest_sha256":check["manifest_sha256"],
                      "v7_manifest_sha256":check["V7_manifest_sha256"],
                      "missing":check["missing"],"mismatches":check["mismatches"]},sort_keys=True))


if __name__=="__main__":main()
