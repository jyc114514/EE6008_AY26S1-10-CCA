from __future__ import annotations
import os
import csv,hashlib,json,resource,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1]
PROJECT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
V3=PROJECT/"repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z"
EQ=PROJECT/"reports/benchmarks/CORE_B_D3_EPISODE_EQUAL_SENSITIVITY_V1_20260924T095800Z"
INPUTS=V3/"data/V3_MAIN_INPUTS.npz"
CACHE=EQ/"d3/train_targets_trainonly.npz"
LAMBDAS=(0.001,0.01,0.1,1.0)
def now():return datetime.now(timezone.utc).isoformat().replace("+00:00","Z")
def sha_bytes(a):
 return hashlib.sha256(np.ascontiguousarray(a).view(np.uint8)).hexdigest()
def sha_file(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def read_json(p):return json.loads(p.read_text(encoding="utf-8"))
def write_json(p,o):
 if p.exists():raise FileExistsError(p)
 p.write_text(json.dumps(o,indent=2,sort_keys=True)+"\n",encoding="utf-8")
def write_csv(p,fields,rows):
 if p.exists():raise FileExistsError(p)
 with p.open("w",encoding="utf-8",newline="") as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction="raise");w.writeheader();w.writerows(rows)
def main():
 import importlib.util
 spec=importlib.util.spec_from_file_location("ridge_core",ROOT/"code/ridge_core.py")
 core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
 lock=read_json(ROOT/"models/RIDGE_MODEL_LOCK.json")
 protocol=read_json(ROOT/"protocol/PROTOCOL_LOCKED.json")
 if lock["status"]!="PASS_MODEL_AND_PREDICTIONS_LOCKED_BEFORE_DEV_SCORING":raise AssertionError("model lock missing")
 if lock["selected_global_lambda"] not in LAMBDAS:raise AssertionError("selected lambda not from grid")
 if sha_file(INPUTS)!=protocol["input"]["source_sha256"] or sha_file(CACHE)!=protocol["target"]["train_target_sha256"]:
  raise AssertionError("registered train source hash changed")
 with np.load(INPUTS,allow_pickle=False) as z:
  ids=z["anchor_id"].astype("U64",copy=True)
  eps=z["episode_index"].astype(np.int64,copy=True)
  split=z["split_code"].astype(np.int8,copy=True)
  hist=z["history_native"].astype(np.float32,copy=True)
 train_ix=np.flatnonzero(split==0);train_ids=ids[train_ix];train_eps=eps[train_ix]
 x=hist[train_ix].reshape(len(train_ix),-1).astype(np.float64)
 with np.load(CACHE,allow_pickle=False) as z:
  yids=z["anchor_id"].astype("U64",copy=True)
  yeps=z["episode_index"].astype(np.int64,copy=True)
  anchors=z["anchor_command_native"].astype(np.float32,copy=True)
  y30=z["targets_native_60"][:,:30,:].astype(np.float64)
 if not np.array_equal(yids,train_ids) or not np.array_equal(yeps,train_eps):raise AssertionError("train-only target identity mismatch")
 y=(y30-anchors.astype(np.float64)[:,None,:]).reshape(len(yids),-1)
 fold_by_id={}
 group_by_id={}
 with (ROOT/"protocol/FOLD_ASSIGNMENT.csv").open(encoding="utf-8",newline="") as f:
  for r in csv.DictReader(f):
   if r["split"]=="train":
    fold_by_id[r["anchor_id"]]=int(r["fold"])
    group_by_id[r["anchor_id"]]=r["source_group_id"]
 folds=np.asarray([fold_by_id[str(a)] for a in train_ids],dtype=np.int8)
 if len(fold_by_id)!=len(train_ids) or len(np.unique(folds))!=3:raise AssertionError("locked fold coverage mismatch")
 scaler_rows=[];objective_rows=[]
 xmeans=[];xscales=[];constants=[];sigmas=[];fdscales=[];safes=[]
 t0=time.perf_counter();c0=time.process_time()
 with threadpool_limits(limits=8):
  for fold in range(3):
   val=folds==fold;fit=~val;fit_idx=np.flatnonzero(fit)
   w=core.episode_equal_weights(train_eps[fit_idx])
   sigma,fd,safe=core.episode_equal_target_scales(y30[fit_idx],train_eps[fit_idx])
   prep=core.prepare_weighted_gram(x[fit_idx],y[fit_idx],w)
   ycenter=y[fit_idx]-prep["y_mean"]
   c=float(np.sum(w[:,None]*ycenter*ycenter)/w.sum())
   vals=prep["eigenvalues"];proj=prep["projected_rhs"]
   fit_groups=set(group_by_id[str(a)] for a in train_ids[fit_idx])
   held_groups=set(group_by_id[str(a)] for a in train_ids[val])
   if fit_groups & held_groups:raise AssertionError("source group crosses fold")
   xmeans.append(prep["x_mean"]);xscales.append(prep["x_scale"]);constants.append(prep["constant_mask"])
   sigmas.append(sigma);fdscales.append(fd);safes.append(safe)
   scaler_rows.append({"fold":fold,"fit_rows":len(fit_idx),"fit_episodes":len(set(train_eps[fit_idx].tolist())),
     "fit_groups":len(fit_groups),"heldout_rows":int(val.sum()),"heldout_groups":len(held_groups),
     "constant_input_columns":int(prep["constant_mask"].sum()),"input_mean_sha256":sha_bytes(prep["x_mean"]),
     "input_scale_sha256":sha_bytes(prep["x_scale"]),"target_sigma_sha256":sha_bytes(sigma),
     "fd_scale_sha256":sha_bytes(fd),"safe_fd_scale_sha256":sha_bytes(safe),
     "target_scale_fit_scope":"fold-fit H30 targets only"})
   for lam in LAMBDAS:
    den=vals[:,None]+lam
    reduction=float(np.sum((proj*proj)/den))
    penalty=float(lam*np.sum((proj/den)**2))
    total_obj=float(c-reduction)
    mse=float(total_obj-penalty)
    objective_rows.append({"fold":fold,"lambda":lam,"fit_rows":len(fit_idx),"fit_groups":len(fit_groups),
     "heldout_rows":int(val.sum()),"heldout_groups":len(held_groups),
     "weighted_squared_residual_mse":mse,"l2_penalty":penalty,"objective_total":mse+penalty,
     "objective_formula":"c - sum((U.T@R)^2/(eigenvalue+lambda)); intercept unpenalized",
     "objective_reconstructed_analytically":True,"coefficient_matrix_materialized":False,
     "additional_cv_fit_count":0,"additional_full_refit_count":0,"final_test_access":0})
 npz=ROOT/"models/CV_PREPROCESSORS.npz"
 if npz.exists():raise FileExistsError(npz)
 np.savez_compressed(npz,fold_ids=np.arange(3,dtype=np.int8),x_mean=np.asarray(xmeans),
  x_scale=np.asarray(xscales),constant_mask=np.asarray(constants),target_sigma=np.asarray(sigmas),
  fd_scale=np.asarray(fdscales),safe_fd_scale=np.asarray(safes))
 write_csv(ROOT/"results/CV_OBJECTIVE_RECONSTRUCTION.csv",list(objective_rows[0].keys()),objective_rows)
 write_csv(ROOT/"results/CV_SCALER_RECEIPTS.csv",list(scaler_rows[0].keys()),scaler_rows)
 cost={"status":"PASS_ANALYTIC_CV_OBJECTIVE_AND_SCALER_RECONSTRUCTION","created_utc":now(),
  "cpu_seconds":time.process_time()-c0,"wall_seconds":time.perf_counter()-t0,
  "peak_rss_kib":int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
  "additional_model_fits":0,"additional_cv_fits":0,"additional_full_refits":0,
  "prediction_arrays_created":False,"development_targets_read":False,"target_scope":"primary train H30 only",
  "preprocessors_path":str(npz),"preprocessors_sha256":sha_file(npz),
  "objective_rows":len(objective_rows),"scaler_rows":len(scaler_rows),
  "interpretation":"post-fit closed-form scalar reconstruction from fold Gram eigenvalues and projected RHS; no coefficient vector, prediction or model is emitted"}
 write_json(ROOT/"coordination/CV_OBJECTIVE_RECONSTRUCTION_COST.json",cost)
 qa={"status":"PASS_CV_OBJECTIVE_AND_FOLD_SCALERS_RECORDED","cv_model_fits_remain":12,
  "full_train_refits_remain":1,"grid_extended":False,"preprocessors_sha256":sha_file(npz),
  "cv_objective_rows":len(objective_rows),"all_objectives_finite":bool(np.isfinite([float(x["objective_total"]) for x in objective_rows]).all()),
  "final_test_access":0,"heldout_access":0}
 write_json(ROOT/"qa/CV_OBJECTIVE_RECONSTRUCTION.json",qa)
 print(json.dumps({"status":qa["status"],"objective_rows":len(objective_rows),"scaler_rows":len(scaler_rows),
   "additional_fits":0,"cpu_seconds":cost["cpu_seconds"],"wall_seconds":cost["wall_seconds"]},sort_keys=True))
if __name__=="__main__":main()

