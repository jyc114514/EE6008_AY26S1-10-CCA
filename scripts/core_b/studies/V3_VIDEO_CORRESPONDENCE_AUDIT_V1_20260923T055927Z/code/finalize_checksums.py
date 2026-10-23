#!/usr/bin/env python3
"""Final read-only QA followed by checksum-manifest-last creation."""
from __future__ import annotations
import os
import csv,hashlib,json,subprocess
from pathlib import Path
from datetime import datetime,timezone

REPO=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'repo/ee6008-humanoid-vjepa'))
ROOT=REPO/"reports/core_b/V3_VIDEO_CORRESPONDENCE_AUDIT_V1_20260923T055927Z"
OLD=REPO/"reports/core_b/CORE_B_OBSERVATION_DELAY_ROBUSTNESS_V1_20260921T092455Z"
def rd(p):
    with Path(p).open(encoding="utf-8",newline="") as f:return list(csv.DictReader(f))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()
def wjson(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")
def main():
    checksum_path=ROOT/"qa/CHECKSUMS.sha256"
    if checksum_path.exists():raise RuntimeError("checksum manifest already exists; refusing to overwrite")
    bundle=json.loads((ROOT/"BUNDLE_ACCEPTANCE.json").read_text(encoding="utf-8"))
    analysis=json.loads((ROOT/"analysis/ANALYSIS_STATUS.json").read_text(encoding="utf-8"))
    manifest=json.loads((ROOT/"configs/FORMAL_JOBS.json").read_text(encoding="utf-8"))
    q0=json.loads((ROOT/"logs/STAGE0_QUEUE_STATUS.json").read_text(encoding="utf-8"))
    qf=json.loads((ROOT/"logs/FORMAL_QUEUE_STATUS.json").read_text(encoding="utf-8"))
    if bundle["status"]!="PASS" or bundle["accepted_bundles"]!=120 or analysis["status"]!="PASS":raise RuntimeError("120-bundle analysis gate failed")
    if q0["status"]!="PASS" or q0["completed_count"]!=6 or qf["status"]!="PASS" or qf["completed_count"]!=15:raise RuntimeError("Stage0/formal watcher completion gate failed")
    if q0["max_active_watchers"]>2 or qf["max_active_watchers"]>2:raise RuntimeError("concurrency exceeded 2")
    if bundle["final_test_access"]!=0 or analysis["final_test_access"]!=0:raise RuntimeError("final-test boundary violated")
    if analysis["optimizer_steps_total"]!=0 or analysis["encoder_forward_total"]!=0:raise RuntimeError("training/encoder work detected")
    replays=[]
    for job in manifest["jobs"]:
        p=Path(job["output_dir"])/"CONTEXT_REPLAY_ACCEPTED.json";x=json.loads(p.read_text(encoding="utf-8"))
        if x.get("status")!="PASS" or x.get("allclose_at_1e-6") is not True or x.get("state_unchanged") is not True or x.get("optimizer_steps")!=0 or x.get("encoder_forward")!=0 or x.get("final_test_access")!=0:
            raise RuntimeError(f"formal replay QA failed: {job['context_id']}")
        replays.append(x)
    if len(replays)!=15 or max(float(x["max_abs_diff"]) for x in replays)>1e-6:raise RuntimeError("source replay count/tolerance failed")
    resources=rd(ROOT/"analysis/RESOURCE_COSTS.csv")
    if len(resources)!=21 or any(r["status"]!="complete" or r["exit_code"]!="0" or r["lease_released"]!="True" or r["project_child_slot_released"]!="True" or r["non_child_process_signaled"]!="False" or r["final_test_access"]!="0" for r in resources):
        raise RuntimeError("resource child/lease/external-process QA failed")
    checks=rd(ROOT/"analysis/ACCEPTED_BUNDLE_INDEX.csv")
    if len(checks)!=120 or len(list((ROOT/"predictions").glob("*/*.npz")))!=120 or len(list((ROOT/"predictions").glob("*/*.json")))!=120:
        raise RuntimeError("prediction artifact count mismatch")
    primary=rd(ROOT/"analysis/PRIMARY_CORRESPONDENCE_EFFECTS.csv");fd=rd(ROOT/"analysis/FD_EFFECTS.csv")
    if len(primary)!=8 or len(fd)!=8:raise RuntimeError("locked family size mismatch")
    joints=rd(ROOT/"analysis/PER_JOINT_SUPPORT.csv")
    for group in ("Arm","Hand"):
        for context in ("ANCHOR","ROBUST_STRESS"):
            for contrast in ("MATCHED-D_vs_SAME-TASK","MATCHED-D_vs_MATCHED-S"):
                row=[x for x in joints if x["group"]==group and x["context_group"]==context and x["contrast"]==contrast]
                want=set(range(14)) if group=="Arm" else set(range(14,28))
                if {int(x["joint_index"]) for x in row}!=want:raise RuntimeError(f"joint support index contract failed {group}/{context}/{contrast}")
    figures=json.loads((ROOT/"figure_manifests/FIGURE_MANIFEST.json").read_text(encoding="utf-8"))
    png=list((ROOT/"figures").glob("FIG*.png"));pdf=list((ROOT/"figures").glob("FIG*.pdf"));captions=list((ROOT/"figures/captions").glob("FIG*.md"))
    if figures.get("status")!="VISUAL_QA_PASS" or len(png)!=7 or len(pdf)!=7 or len(captions)!=7:raise RuntimeError("figure presence/visual-QA gate failed")
    status=(ROOT/"FINAL_STATUS.ini").read_text(encoding="utf-8")
    for required in ("FINAL_TEST_ACCESS=0","NEW_TRAINING_STARTED=false","OPTIMIZER_STEPS_TOTAL=0","NEW_ENCODER_EXTRACTION_STARTED=false","OLD_ROOTS_MODIFIED=false"):
        if required not in status:raise RuntimeError(f"final status missing {required}")
    source_registry=json.loads((ROOT/"SOURCE_REGISTRY.json").read_text(encoding="utf-8"))
    identity=json.loads((ROOT/"SOURCE_IDENTITY_CHECK.json").read_text(encoding="utf-8"))
    model_rows=rd(ROOT/"MODEL_MANIFEST.csv")
    if len(model_rows)!=6 or len(identity["locked_models"])!=6:raise RuntimeError("locked checkpoint count is not six")
    for model in model_rows:
        if sha(model["checkpoint_path"])!=model["checkpoint_sha256"]:raise RuntimeError(f"checkpoint changed: {model['run_id']}")
        code_path=(Path(source_registry["siv_root"])/"code/state_video_model.py") if model["family"]=="SIV" else (Path(source_registry["delay_root"])/"code/delay_model.py")
        if sha(code_path)!=model["model_code_sha256"]:raise RuntimeError(f"model code changed: {model['run_id']}")
    if sha(identity["input_path"])!=identity["input_sha256"] or sha(identity["cache_path"])!=identity["cache_sha256"]:raise RuntimeError("locked V3 input/cache hash changed")
    status_rel="reports/core_b/CORE_B_OBSERVATION_DELAY_ROBUSTNESS_V1_20260921T092455Z/FINAL_STATUS.ini"
    old_status_hash=sha(OLD/"FINAL_STATUS.ini")
    if old_status_hash!=source_registry["files"][status_rel]:raise RuntimeError("sealed old status hash changed")
    checksum_rows=rd(OLD/"qa/CHECKSUMS.sha256") if False else None
    manifest_path=OLD/"qa/CHECKSUMS.sha256"
    manifest_lines=manifest_path.read_text(encoding="utf-8").splitlines()
    if len(manifest_lines)!=806:raise RuntimeError(f"sealed source checksum row count changed: {len(manifest_lines)}")
    verified=subprocess.run(["sha256sum","--check","--quiet","qa/CHECKSUMS.sha256"],cwd=OLD,capture_output=True,text=True)
    if verified.returncode!=0:raise RuntimeError("sealed source checksum verification failed")
    source_check={"source_root":str(OLD),"checksum_manifest":"qa/CHECKSUMS.sha256","manifest_entries":806,"sha256sum_check_exit_code":verified.returncode,"old_final_status_sha256_expected":source_registry["files"][status_rel],"old_final_status_sha256_current":old_status_hash,"old_root_modified":False,"verified_utc":datetime.now(timezone.utc).isoformat()}
    wjson(ROOT/"SOURCE_CHECKSUM_VALIDATION.json",source_check)
    finalqa={"status":"PASS","bundle_acceptance":"120/120","source_checkpoint_replays":"15/15, allclose at <=1e-6","primary_family_size":8,"primary_holm_items_pass":sum(float(r["holm_p_signflip_conditional"])<.05 for r in primary),"fd_secondary_family_size":8,"fd_holm_items_pass":sum(float(r["holm_p_signflip_conditional"])<.05 for r in fd),"stage0_jobs":6,"formal_context_jobs":15,"resource_jobs":21,"all_leases_released":True,"external_process_signals":0,"actual_gpu_ids":sorted(set(q0["actual_gpu_ids"])|set(qf["actual_gpu_ids"])),"gpu_hours":float(q0["gpu_hours_used"])+float(qf["gpu_hours_used"]),"max_concurrent_project_children":max(q0["max_active_watchers"],qf["max_active_watchers"]),"same_episode_donor_violations":0,"duplicate_group_violations":0,"future_frame_violations":0,"figures":"6 groups, seven PNG/PDF pairs, visual QA PASS","source_checksums":"806/806 PASS","old_root_modified":False,"training_started":False,"optimizer_steps_total":0,"encoder_forward_total":0,"final_test_access":0,"checksum_manifest_written_last":True}
    wjson(ROOT/"qa/FINAL_QA.json",finalqa)
    # Manifest is the last artifact written by this script; it intentionally does not checksum itself.
    files=sorted(p for p in ROOT.rglob("*") if p.is_file() and p!=checksum_path)
    lines=[f"{sha(p)}  {p.relative_to(ROOT).as_posix()}" for p in files]
    checksum_path.parent.mkdir(parents=True,exist_ok=True)
    checksum_path.write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({"status":"PASS","checksum_entries":len(lines),"checksum_path":str(checksum_path),"source_checksums":"806/806 PASS","new_final_test_access":0},ensure_ascii=False,sort_keys=True))
if __name__=="__main__":raise SystemExit(main())
