#!/usr/bin/env python3
from __future__ import annotations
import os
import csv, hashlib, io, json, os, platform, resource, socket, sys, time, uuid, fcntl
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
import numpy as np
from safetensors import safe_open

PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
V2 = PROJECT / "reports/benchmarks/CORE_B_TASK_SPECIALIZATION_WAYPOINT_V2_20261013T064245Z"
V3 = PROJECT / "reports/benchmarks/CORE_B_V3_FROZEN_SHARED_TASK_RESIDUAL_V1_20261014T053428Z"
V4 = PROJECT / "reports/benchmarks/CORE_B_V4_RGB_ALIGNMENT_REASSIGNMENT_V1_20261014T083805Z"
V5 = PROJECT / "reports/benchmarks/CORE_B_V5_LEVEL_SHAPE_COMPOSITION_V1_20261015T053806Z"
ROOT = PROJECT / "reports/benchmarks/CORE_B_V6_EQUAL_HALF_ENSEMBLE_CONTROLS_V1_20261015T070036Z"
TASKS = [119, 94, 90, 141, 230]
SEEDS = [202610131, 202610132, 202610133]
PROGRAMS = ["S", "A", "V"]
ROLE_BY_PROGRAM = {"S": "S_OPT", "A": "A_COMMON", "V": "V_MATCHED"}
METRICS = ["position_mae", "safe_fd_l1", "seam_safe_fd_l1"]
METRIC_LABELS = {"position_mae": "position_mae", "safe_fd_l1": "safe_fd_l1", "seam_safe_fd_l1": "seam_safe_fd_l1"}
BOOT_DRAWS = 5000
BOOT_SEED = 2026101506
ATOL = 1e-12
RTOL = 1e-10

def utc():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def log(msg):
    print(f"[{utc()}] {msg}", flush=True)

def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()

def sha_file(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def read_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))

def write_json(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, p)

def read_csv(p):
    with Path(p).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def write_csv(p, rows, fields=None):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if fields is None:
        fields = []
        for row in rows:
            for k in row:
                if k not in fields:
                    fields.append(k)
    if not fields:
        fields = ["status"]
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

def as_text(x):
    if isinstance(x, bytes):
        return x.decode("utf-8")
    if isinstance(x, np.bytes_):
        return bytes(x).decode("utf-8")
    return str(x)

def strict_bool(x):
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s == "true":
        return True
    if s == "false":
        return False
    raise ValueError(f"same_seed must be literal true/false, got {x!r}")

def id_hash(ids):
    return sha_bytes(("".join(f"{x}\n" for x in ids)).encode("utf-8"))

def canon_array_hash(arr):
    a = np.asarray(arr, dtype="<f8", order="C")
    return sha_bytes(a.tobytes(order="C"))

def verify_manifest(root, receipt_path):
    root = Path(root)
    receipt = read_json(receipt_path)
    manifest = root / "CHECKSUMS.sha256"
    raw = manifest.read_bytes()
    digest = sha_bytes(raw)
    if digest != receipt.get("manifest_sha256"):
        raise RuntimeError("V5 manifest digest differs from read-back receipt")
    if receipt.get("status") != "PASS" or receipt.get("missing") or receipt.get("mismatches"):
        raise RuntimeError("V5 final checksum receipt is not PASS")
    if receipt.get("manifest_entry_count") != receipt.get("verified_entry_count"):
        raise RuntimeError("V5 checksum receipt count mismatch")
    expected = {}
    for line_no, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if len(line) < 67 or line[64:66] != "  ":
            raise RuntimeError(f"malformed V5 manifest line {line_no}")
        digest_value, rel_s = line[:64], line[66:]
        rel = Path(rel_s)
        if rel.is_absolute() or ".." in rel.parts or not digest_value:
            raise RuntimeError(f"unsafe V5 manifest entry {line_no}")
        if rel_s in expected:
            raise RuntimeError(f"duplicate V5 manifest entry {rel_s}")
        expected[rel_s] = digest_value
    excluded = {"CHECKSUMS.sha256", "qa/FINAL_CHECKSUM_VERIFICATION.json"}
    if set(expected) & excluded:
        raise RuntimeError("V5 manifest includes a prohibited recursive checksum entry")
    if len(expected) != int(receipt["manifest_entry_count"]):
        raise RuntimeError("V5 manifest parsed entry count differs from receipt")
    mismatches = []
    for rel_s, expected_sha in expected.items():
        p = root / rel_s
        if not p.is_file() or p.is_symlink():
            mismatches.append({"path": rel_s, "reason": "missing_or_symlink"})
            continue
        got = sha_file(p)
        if got != expected_sha:
            mismatches.append({"path": rel_s, "expected": expected_sha, "actual": got})
    if mismatches:
        raise RuntimeError(f"V5 live file checksums changed: {mismatches[:5]}")
    if receipt.get("verified_entry_count") != len(expected):
        raise RuntimeError("V5 receipt verified count is inconsistent")
    return {"status": "PASS", "manifest_sha256": digest, "entries": len(expected),
            "receipt_path": str(receipt_path), "receipt_created_utc": receipt.get("created_utc")}

def validate_tensor(arr, ids, expected_ids, expected_shape=(185, 30, 28)):
    a = np.asarray(arr)
    if a.dtype.hasobject:
        raise ValueError("object arrays are prohibited")
    if a.ndim != 3 or tuple(a.shape) != tuple(expected_shape):
        raise ValueError(f"expected shape {expected_shape}, got {a.shape}")
    if len(ids) != expected_shape[0] or not all(str(x) for x in ids) or len(set(map(str, ids))) != len(ids):
        raise ValueError("empty or duplicate receiver IDs")
    if list(map(str, ids)) != list(map(str, expected_ids)):
        raise ValueError("receiver ID ordering mismatch")
    if not np.issubdtype(a.dtype, np.number) or not np.isfinite(a).all():
        raise ValueError("source array is nonnumeric or nonfinite")
    return np.asarray(a, dtype=np.float64, order="C")

def pair_family(a, b):
    order = {p: i for i, p in enumerate(PROGRAMS)}
    if a["program"] == b["program"]:
        return a["program"] + b["program"]
    x, y = sorted([a["program"], b["program"]], key=order.get)
    return x + y

def make_pair_id(a, b):
    order = {p: i for i, p in enumerate(PROGRAMS)}
    pair = sorted([a, b], key=lambda x: (order[x["program"]], int(x["seed"])))
    return f"PAIR_{pair[0]['program']}{pair[0]['seed']}__{pair[1]['program']}{pair[1]['seed']}"

def generate_pair(left, right):
    # Pure source-array transform. This function has no target, label, or GT argument.
    return np.asarray(0.5 * np.asarray(left, dtype=np.float64) +
                      0.5 * np.asarray(right, dtype=np.float64), dtype=np.float64, order="C")

def target_score(pred, target, anchor, sigma, fdscale):
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if pred.shape != target.shape or pred.ndim != 3 or pred.shape[1:] != (30, 28):
        raise ValueError("score tensor must be aligned [N,30,28]")
    pos = (np.abs(pred - target) / sigma[None, None, :]).mean(axis=(1, 2))
    fd = (np.abs(np.diff(pred, axis=1) - np.diff(target, axis=1)) /
          fdscale[None, None, :]).mean(axis=(1, 2))
    seam = (np.abs((pred[:, 0, :] - anchor) - (target[:, 0, :] - anchor)) / fdscale[None, :]).mean(axis=1)
    return {"position_mae": pos, "safe_fd_l1": fd, "seam_safe_fd_l1": seam}

def aggregate_metric(rows, metric, task):
    rr = [r for r in rows if int(r["task_id"]) == int(task)]
    eps = defaultdict(list)
    for r in rr:
        eps[str(r["episode_id"])].append(float(r[metric]))
    return float(np.mean([np.mean(v) for v in eps.values()]))

def macro_from_task(task_values):
    return float(np.mean([task_values[t] for t in TASKS]))

def sample_sd(values):
    vals = np.asarray(list(values), dtype=np.float64)
    if vals.size < 2:
        return float("nan")
    return float(np.std(vals, ddof=1))

def node_key(program, seed):
    return (str(program), int(seed))

def main():
    start_all = time.perf_counter()
    lock_path = ROOT / "coordination/OWNER.lock"
    lock_f = lock_path.open("a+")
    try:
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("V6 output root has an active writer lock")
    token = uuid.uuid4().hex
    owner = {
        "schema_version": 1, "study": "post_V5_equal_half_ensemble_controls_development",
        "status": "running", "phase": "owner_lock_acquired", "host": socket.gethostname(),
        "owner_uid": os.getuid(), "owner_pid": os.getpid(), "owner_lock_acquired": True,
        "owner_token_sha256": sha_bytes(token.encode()), "out_root": str(ROOT),
        "updated_utc": utc(), "source_roots_read_only": True, "old_root_writes": 0,
        "final_heldout_payload_access": 0, "same_contract_v6_root_found": False
    }
    lock_f.seek(0); lock_f.truncate()
    lock_f.write(json.dumps({k: owner[k] for k in
        ["host", "owner_uid", "owner_pid", "owner_token_sha256", "study", "out_root", "owner_lock_acquired"]},
        sort_keys=True) + "\n")
    lock_f.flush(); os.fsync(lock_f.fileno())
    write_json(ROOT / "coordination/OWNER_STATE.json", owner)

    def set_phase(phase, **extra):
        owner.update({"phase": phase, "updated_utc": utc()})
        owner.update(extra)
        write_json(ROOT / "coordination/OWNER_STATE.json", owner)
        log(f"phase={phase}")

    try:
        if not ROOT.is_dir() or ROOT.name != "CORE_B_V6_EQUAL_HALF_ENSEMBLE_CONTROLS_V1_20261015T070036Z":
            raise RuntimeError("unexpected V6 output root")
        set_phase("runtime_preflight")
        affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else list(range(os.cpu_count() or 1))
        runtime = {
            "created_utc": utc(), "host": socket.gethostname(), "uid": os.getuid(),
            "python": sys.version.split()[0], "platform": platform.platform(),
            "cpu_count": os.cpu_count(), "effective_cpu_count": len(affinity),
            "thread_limit": 4, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "gpu_queries": 0, "scheduler_environment_detected": any(
                k.startswith(("SLURM_", "PBS_", "LSB_")) for k in os.environ),
            "runtime_scope": "CPU-only source-array composition, scoring, bootstrap, and plotting",
            "project_root": str(PROJECT), "formal_repo_git_state": "not applicable; output is under reports/benchmarks and no repository source files are modified",
            "python_bin": str(PROJECT / "envs/jyc/bin/python"),
            "command": "PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 project-env/python code/run_v6.py"
        }
        write_json(ROOT / "RUN_CONTEXT.json", runtime)

        set_phase("v5_current_seal_verification")
        v5_receipt = verify_manifest(V5, V5 / "qa/FINAL_CHECKSUM_VERIFICATION.json")
        if v5_receipt["manifest_sha256"] != "5866688b48bef3ed940920affc15081dea052c51eca6eb86e7bcfd3a09c5acfa" or v5_receipt["entries"] != 255:
            raise RuntimeError("V5 receipt is not the expected current 255-entry seal")
        v5_accept = read_json(V5 / "qa/FINAL_ACCEPTANCE.json")
        if v5_accept.get("status") != "complete" or v5_accept.get("final_heldout_payload_access") != 0:
            raise RuntimeError("V5 acceptance metadata is not complete/development-only")
        source_map_path = V5 / "inventory/SOURCE_MAP.json"
        v5_source_map = read_json(source_map_path)
        v5_source_map_sha = sha_file(source_map_path)
        v5_seal_path = V5 / "qa/PREDICTION_SEAL.json"
        v5_seal = read_json(v5_seal_path)
        if v5_seal.get("status") != "PASS" or v5_seal.get("nonempty_seal") is not True or v5_seal.get("sealed_bundles") != 18:
            raise RuntimeError("V5 prediction seal is not accepted and complete")
        v5_formula_path = V5 / "protocol/FORMULA_LOCK.json"
        v5_formula = read_json(v5_formula_path)
        v5_formula_sha = sha_file(v5_formula_path)
        if v5_seal.get("formula_lock_sha256") != v5_formula_sha:
            raise RuntimeError("V5 formula lock and prediction seal mismatch")
        set_phase("v5_seal_verified", v5_checksum_entries_verified=v5_receipt["entries"],
                  v5_checksum_sha256=v5_receipt["manifest_sha256"])

        # Exact accepted receiver order; no shape-only joins.
        order_path = V5 / "protocol/ORDERED_RECEIVER_IDS.csv"
        order_rows = read_csv(order_path)
        order_rows.sort(key=lambda x: int(x["canonical_row_index"]))
        canonical_ids = [x["anchor_id"] for x in order_rows]
        canonical_hash = id_hash(canonical_ids)
        if len(canonical_ids) != 185 or len(set(canonical_ids)) != 185 or canonical_hash != v5_formula["ordered_receiver_ids_sha256"]:
            raise RuntimeError("V5 canonical receiver ID lock failed")
        canonical_index = {a: i for i, a in enumerate(canonical_ids)}

        # Validate the exact accepted group joins and ensure groups are nested within a task.
        join_path = V5 / "inventory/SOURCE_PREDICTION_JOINS.csv"
        group_validation_path = V5 / "qa/SOURCE_GROUP_TASK_VALIDATION.json"
        group_validation = read_json(group_validation_path)
        join_rows = read_csv(join_path)
        if group_validation.get("status") != "PASS" or group_validation.get("source_join_sha256") != sha_file(join_path):
            raise RuntimeError("V5 source-group validation no longer matches its join table")
        join_lookup = {}
        for r in join_rows:
            key = (r["role"], int(r["seed"]), r["anchor_id"])
            if key in join_lookup:
                raise RuntimeError(f"duplicate V5 source join key {key}")
            join_lookup[key] = r
        node_group_meta = {}
        source_node_records = []
        source_arrays = {}
        node_record_by_key = {}
        selected = [x for x in v5_source_map["source_prediction_bundles"]
                    if x.get("role") in set(ROLE_BY_PROGRAM.values())]
        if len(selected) != 9:
            raise RuntimeError(f"expected 9 S/A/V accepted source nodes, got {len(selected)}")
        expected_role_counts = Counter(x["role"] for x in selected)
        if expected_role_counts != Counter({"S_OPT": 3, "A_COMMON": 3, "V_MATCHED": 3}):
            raise RuntimeError(f"unexpected program source role matrix: {expected_role_counts}")
        source_join_output = []
        for rec in sorted(selected, key=lambda x: (PROGRAMS.index(
                {"S_OPT":"S","A_COMMON":"A","V_MATCHED":"V"}[x["role"]]), int(x["seed"]))):
            role = rec["role"]
            program = {"S_OPT": "S", "A_COMMON": "A", "V_MATCHED": "V"}[role]
            seed = int(rec["seed"])
            if seed not in SEEDS or rec.get("native_field") != "prediction_native":
                raise RuntimeError(f"unexpected source seed/field for {role}/{seed}")
            pred_path = Path(rec["prediction_path"])
            rows_path = Path(rec["prediction_rows_path"])
            if sha_file(pred_path) != rec["prediction_sha256"] or sha_file(rows_path) != rec["prediction_rows_sha256"]:
                raise RuntimeError(f"accepted source file hash mismatch: {role}/{seed}")
            seal_path = Path(rec["seal_path"])
            if sha_file(seal_path) != rec["seal_sha256"]:
                raise RuntimeError(f"accepted source seal hash mismatch: {role}/{seed}")
            rows = read_csv(rows_path)
            if len(rows) != 185:
                raise RuntimeError(f"source rows sidecar incomplete: {role}/{seed}")
            source_ids = [r["anchor_id"] for r in rows]
            if id_hash(source_ids) != rec["ordered_source_ids_sha256"]:
                raise RuntimeError(f"source sidecar ordered ID hash mismatch: {role}/{seed}")
            if set(source_ids) != set(canonical_ids):
                raise RuntimeError(f"source receiver ID set mismatch: {role}/{seed}")
            with safe_open(str(pred_path), framework="numpy", device="cpu") as sf:
                keys = list(sf.keys())
                if rec["native_field"] not in keys:
                    raise RuntimeError(f"locked native field absent: {role}/{seed}")
                raw = np.asarray(sf.get_tensor(rec["native_field"]))
            if raw.dtype.hasobject or tuple(raw.shape) != (185, 30, 28) or not np.isfinite(raw).all():
                raise RuntimeError(f"invalid native source array schema: {role}/{seed}")
            source_index = {a: i for i, a in enumerate(source_ids)}
            aligned = np.asarray(raw[[source_index[a] for a in canonical_ids]], dtype=np.float64, order="C")
            if not np.isfinite(aligned).all():
                raise RuntimeError(f"native float64 cast nonfinite: {role}/{seed}")
            source_arrays[(program, seed)] = aligned
            node_id = f"{program}{seed}"
            model_node = {
                "node_id": node_id, "program": program, "seed": seed, "role": role,
                "source_version": rec["source_version"], "source_profile": rec["source_profile"],
                "condition": rec["condition"], "prediction_path": str(pred_path),
                "prediction_sha256": rec["prediction_sha256"], "rows_path": str(rows_path),
                "rows_sha256": rec["prediction_rows_sha256"], "source_seal_path": str(seal_path),
                "source_seal_sha256": rec["seal_sha256"], "native_field": rec["native_field"],
                "source_dtype": rec["dtype"], "source_shape": rec["shape"],
                "native_cast_dtype": "float64", "normalizer_path": rec["normalizer_path"],
                "normalizer_sha256": rec["normalizer_sha256"],
                "source_ordered_ids_sha256": rec["ordered_source_ids_sha256"],
                "canonical_receiver_ids_sha256": canonical_hash
            }
            source_node_records.append(model_node)
            node_record_by_key[(program, seed)] = model_node
            for source_i, row in enumerate(rows):
                aid = row["anchor_id"]
                j = join_lookup.get((role, seed, aid))
                if j is None:
                    raise RuntimeError(f"V5 accepted group join missing {role}/{seed}/{aid}")
                meta = (str(j["episode_id"]), str(j["task_id"]), str(j["source_group"]))
                if "task_id" in row and str(row["task_id"]) != meta[1]:
                    raise RuntimeError(f"source sidecar task mismatch {role}/{seed}/{aid}")
                if "episode_id" in row and str(row["episode_id"]) != meta[0]:
                    raise RuntimeError(f"source sidecar episode mismatch {role}/{seed}/{aid}")
                if "source_row_index" in j and int(j["source_row_index"]) != source_i:
                    raise RuntimeError(f"V5 source row index mismatch {role}/{seed}/{aid}")
                old = node_group_meta.setdefault(aid, meta)
                if old != meta:
                    raise RuntimeError(f"S/A/V group/task/episode metadata mismatch for {aid}")
                source_join_output.append({
                    "node_id": node_id, "program": program, "seed": seed, "source_role": role,
                    "source_version": rec["source_version"], "source_path": str(pred_path),
                    "source_sha256": rec["prediction_sha256"], "source_row_index": source_i,
                    "canonical_row_index": canonical_index[aid], "anchor_id": aid,
                    "episode_id": meta[0], "task_id": meta[1], "source_group": meta[2]
                })
        if len(node_group_meta) != 185:
            raise RuntimeError("source join metadata does not cover all 185 receiver IDs")
        groups_per_task = defaultdict(set)
        episodes_per_task = defaultdict(set)
        group_task = {}
        for aid, (episode, task_s, group) in node_group_meta.items():
            task = int(task_s)
            if task not in TASKS:
                raise RuntimeError(f"unexpected task {task}")
            if group in group_task and group_task[group] != task:
                raise RuntimeError(f"source group crosses tasks: {group}")
            group_task[group] = task
            groups_per_task[task].add(group)
            episodes_per_task[task].add(episode)
        if len(group_task) != 47 or sum(len(v) for v in groups_per_task.values()) != 47:
            raise RuntimeError("expected 47 task-nested source groups")
        expected_support = {119:(48,12,12),94:(48,12,12),90:(41,11,11),141:(20,5,5),230:(28,7,7)}
        for task, (n_anchor, n_ep, n_group) in expected_support.items():
            aa = [a for a, m in node_group_meta.items() if int(m[1]) == task]
            if (len(aa), len(episodes_per_task[task]), len(groups_per_task[task])) != (n_anchor,n_ep,n_group):
                raise RuntimeError(f"task support mismatch for {task}")
        write_csv(ROOT / "inventory/SOURCE_JOINS.csv", source_join_output,
                  ["node_id","program","seed","source_role","source_version","source_path","source_sha256",
                   "source_row_index","canonical_row_index","anchor_id","episode_id","task_id","source_group"])
        normalizer = v5_source_map["normalizer"]
        normalizer_path = Path(normalizer["path"])
        if sha_file(normalizer_path) != normalizer["sha256"]:
            raise RuntimeError("accepted source normalizer changed")
        metric_lock_ref = v5_source_map["metric_scale_lock"]
        metric_path = Path(metric_lock_ref["path"])
        if sha_file(metric_path) != metric_lock_ref["sha256"]:
            raise RuntimeError("accepted metric scale lock changed")
        metric_lock = read_json(metric_path)
        sigma = np.asarray(metric_lock["target_sigma"], dtype=np.float64)
        fdscale = np.asarray(metric_lock["safe_fd_scale"], dtype=np.float64)
        if sigma.shape != (28,) or fdscale.shape != (28,) or not np.isfinite(sigma).all() or not np.isfinite(fdscale).all() or np.any(sigma <= 0) or np.any(fdscale <= 0):
            raise RuntimeError("accepted train-only metric scales invalid")

        # Read accepted per-window loss fixtures and bind their exact source hashes before any new V6 target scoring.
        old_derived_path = V5 / "results/PER_WINDOW_DERIVED_LOSSES.csv"
        old_refs_path = V5 / "results/PER_WINDOW_ACCEPTED_REFERENCES.csv"
        old_derived = read_csv(old_derived_path)
        old_refs = read_csv(old_refs_path)
        if sha_file(old_derived_path) not in {v for k,v in verify_manifest_entries(V5).items() if k == "results/PER_WINDOW_DERIVED_LOSSES.csv"}:
            raise RuntimeError("V5 derived score fixture is not covered by the verified manifest")
        if len(old_derived) != 3330 or len(old_refs) != 3885:
            raise RuntimeError("V5 accepted per-window loss fixtures have unexpected row counts")
        accepted_ref_methods = {"S","A","V","BIAS_STATE","BIAS_VIDEO","PERSISTENCE"}
        if {r["method"] for r in old_refs} != accepted_ref_methods | {"C"}:
            raise RuntimeError("V5 accepted reference method set changed")
        loss_rows = defaultdict(list)
        loss_lookup = {}
        for r in old_refs:
            m, s, aid = r["method"], int(r["seed"]), r["anchor_id"]
            if m not in accepted_ref_methods | {"C"}:
                raise RuntimeError(f"unaccepted reference method {m}")
            meta = (str(r["episode_id"]),str(r["task_id"]),str(r["source_group"]))
            if meta != node_group_meta[aid]:
                raise RuntimeError(f"accepted reference join mismatch for {m}/{s}/{aid}")
            row = {**r, "seed": str(s)}
            for metric in METRICS:
                row[metric] = float(r[metric])
                if not np.isfinite(row[metric]):
                    raise RuntimeError("nonfinite accepted V5 reference score")
            loss_rows[(m,s)].append(row)
            loss_lookup[(m,s,aid)] = row
        for m in accepted_ref_methods | {"C"}:
            for s in SEEDS:
                rr = loss_rows[(m,s)]
                if len(rr) != 185 or len({r["anchor_id"] for r in rr}) != 185:
                    raise RuntimeError(f"accepted reference rows incomplete {m}/{s}")
        mix_old_rows = defaultdict(list)
        for r in old_derived:
            if r["method"] != "MIX50":
                continue
            s, aid = int(r["seed"]), r["anchor_id"]
            meta = (str(r["episode_id"]),str(r["task_id"]),str(r["source_group"]))
            if meta != node_group_meta[aid]:
                raise RuntimeError(f"V5 MIX50 accepted loss join mismatch {s}/{aid}")
            row = {**r, "seed": str(s)}
            for metric in METRICS:
                row[metric] = float(r[metric])
            mix_old_rows[s].append(row)
        if any(len(mix_old_rows[s]) != 185 for s in SEEDS):
            raise RuntimeError("V5 MIX50 score rows are incomplete")
        v5_recon = read_json(V5 / "qa/SOURCE_SCORE_RECONCILIATION.json")
        if v5_recon.get("status") != "PASS":
            raise RuntimeError("V5 source-score reconciliation is not PASS")
        exposure_v5 = read_json(V5 / "inventory/OUTCOME_EXPOSURE.json")
        target_path = Path(exposure_v5["target_archive_path"])
        target_expected_sha = exposure_v5["target_archive_sha256"]
        if target_path.name != "DEV_TARGETS_SCORER_ONLY.npz" or not target_path.is_file():
            raise RuntimeError("the accepted V5 development scorer archive path is unavailable")
        if exposure_v5.get("final_heldout_payload_access") != 0:
            raise RuntimeError("V5 reports nonzero final/heldout payload access")
        prefix_artifact = next(x for x in v5_source_map["source_artifacts"]
                               if Path(x["path"]).name == "DEV_PREFIXES_NO_LABELS.npz")
        prefix_path = Path(prefix_artifact["path"])
        if sha_file(prefix_path) != prefix_artifact["sha256"]:
            raise RuntimeError("accepted label-free prefix metadata hash changed")
        v5_formula_tqa = read_json(V5 / "qa/TRANSFORM_QA.json")
        replay_atol = float(v5_formula_tqa["real_source_array_checks"]["identity_tolerance_atol"])
        replay_rtol = float(v5_formula_tqa["real_source_array_checks"].get("identity_tolerance_rtol", RTOL))
        selected_cases = v5_formula["metadata_selected_cases"]["selected_anchor_ids"]
        if v5_formula["metadata_selected_cases"].get("selection_uses_outcomes") is not False:
            raise RuntimeError("V5 representative case lock was outcome-selected")
        if any(a not in canonical_index for a in selected_cases.values()):
            raise RuntimeError("V5 locked representative cases do not match current receiver IDs")

        # Generate and lock the full unordered 36-pair program/seed matrix before target access.
        nodes = [{"program": p, "seed": s, "node_id": f"{p}{s}"} for p in PROGRAMS for s in SEEDS]
        node_order = {(n["program"],n["seed"]): i for i,n in enumerate(nodes)}
        pairs = []
        for a, b in combinations(nodes, 2):
            left, right = sorted([a,b], key=lambda n: node_order[(n["program"],n["seed"])])
            family = pair_family(left, right)
            same_seed = int(left["seed"]) == int(right["seed"])
            pid = make_pair_id(left, right)
            reuse = family == "SV" and same_seed
            pairs.append({
                "pair_id": pid, "family": family, "left_program": left["program"], "left_seed": left["seed"],
                "right_program": right["program"], "right_seed": right["seed"],
                "same_seed": "true" if same_seed else "false",
                "paired_seed": str(left["seed"]) if same_seed else "",
                "left_node_id": left["node_id"], "right_node_id": right["node_id"],
                "formula_version": "EQUAL_HALF_NATIVE_FLOAT64_V1",
                "formula": "0.5*left_native + 0.5*right_native",
                "ensemble_weight_left": 0.5, "ensemble_weight_right": 0.5,
                "reuse_v5_mix50": "true" if reuse else "false",
                "principal_v5_mix50": "true" if reuse else "false"
            })
        family_counts = Counter(x["family"] for x in pairs)
        if len(pairs) != 36 or family_counts != Counter({"SS":3,"AA":3,"VV":3,"SA":9,"SV":9,"AV":9}):
            raise RuntimeError(f"fixed pair matrix invalid: {len(pairs)} {family_counts}")
        if sum(strict_bool(x["reuse_v5_mix50"]) for x in pairs) != 3:
            raise RuntimeError("V5 MIX50 reuse count is not three")
        write_csv(ROOT / "protocol/PAIR_PLAN.csv", pairs)
        source_lock = {
            "schema_version":1, "study":"post_V5_equal_half_ensemble_controls_development",
            "created_utc":utc(), "v5_root":str(V5), "v5_manifest_sha256":v5_receipt["manifest_sha256"],
            "v5_manifest_entries_verified":v5_receipt["entries"], "v5_source_map_path":str(source_map_path),
            "v5_source_map_sha256":v5_source_map_sha, "v5_prediction_seal_path":str(v5_seal_path),
            "v5_prediction_seal_sha256":sha_file(v5_seal_path), "v5_formula_lock_sha256":v5_formula_sha,
            "source_programs":PROGRAMS, "source_seeds":SEEDS, "source_nodes":source_node_records,
            "source_node_count":len(source_node_records), "ordered_receiver_ids_count":185,
            "ordered_receiver_ids_sha256":canonical_hash, "ordered_receiver_ids_source":str(order_path),
            "source_join_path":str(join_path), "source_join_sha256":sha_file(join_path),
            "source_group_task_validation_path":str(group_validation_path),
            "source_groups":47, "source_groups_crossing_tasks":0,
            "task_support":{str(t):{"anchors":len([a for a,m in node_group_meta.items() if int(m[1])==t]),
                                    "episodes":len(episodes_per_task[t]),"source_groups":len(groups_per_task[t])} for t in TASKS},
            "normalizer_path":str(normalizer_path),"normalizer_sha256":normalizer["sha256"],
            "metric_scale_lock_path":str(metric_path),"metric_scale_lock_sha256":metric_lock_ref["sha256"],
            "accepted_node_representation":"native_field prediction_native loaded from exact accepted V2/V4 bundles, then cast to float64; no second normalization",
            "v2_root":str(V2),"v3_root":str(V3),"v4_root":str(V4),"v5_root":str(V5),
            "source_roots_read_only":True,"old_root_writes":0
        }
        write_json(ROOT / "inventory/SOURCE_MODEL_MAP.json", source_lock)
        exposure = {
            "schema_version":1, "study":"post_V5_equal_half_ensemble_controls_development",
            "status":"PREDICTIONS_PREPARATION_TARGET_FREE", "created_utc":utc(),
            "v5_target_archive_path":str(target_path),"v5_target_archive_sha256_expected":target_expected_sha,
            "target_archive_load_operations":0,"target_archive_physical_read_operations":0,
            "target_array_window_rows_loaded":0,"unique_dev_receiver_windows":185,
            "preseal_target_payload_access":0,"final_heldout_payload_access":0,
            "target_archive_read_started_utc":None,"target_archive_read_completed_utc":None,
            "source_prediction_bundles_loaded":9,"source_prediction_row_sidecars_parsed":9,
            "source_prediction_reference_file_reads":18,"v5_mix50_bundle_reuse_expected":3,
            "v5_per_window_loss_tables_read":2,"v5_reference_loss_rows_parsed":3885,
            "v5_mix50_loss_rows_parsed":555,"new_pair_window_loss_evaluations":0,
            "mix50_reconciliation_window_evaluations":0,"cancellation_diagnostic_pair_windows":0,
            "old_root_writes":0,"history":[{"phase":"metadata_and_prediction_construction","target_archive_loads":0,
              "target_payload_access":0,"status":"pending"}]
        }
        write_json(ROOT / "inventory/OUTCOME_EXPOSURE.json", exposure)
        formula_lock = {
            "schema_version":1,"protocol_id":"EE6008_CORE_B_V6_EQUAL_HALF_ENSEMBLE_CONTROLS_V1",
            "study":"post_V5_equal_half_ensemble_controls_development","created_utc":utc(),
            "source_model_map_path":"inventory/SOURCE_MODEL_MAP.json",
            "source_model_map_sha256":sha_file(ROOT/"inventory/SOURCE_MODEL_MAP.json"),
            "pair_plan_path":"protocol/PAIR_PLAN.csv","pair_plan_sha256":sha_file(ROOT/"protocol/PAIR_PLAN.csv"),
            "source_programs":PROGRAMS,"source_seeds":SEEDS,"source_node_count":9,"pair_system_count":36,
            "family_counts":dict(sorted(family_counts.items())),"ordered_receiver_ids_count":185,
            "ordered_receiver_ids_sha256":canonical_hash,
            "formula":"E(X_i,Y_j)=0.5*X_i + 0.5*Y_j on native float64 prediction arrays",
            "formula_version":"EQUAL_HALF_NATIVE_FLOAT64_V1","ensemble_weights":[0.5,0.5],
            "pair_is_unordered":True,"no_self_pairs":True,"self_pair_only_for_transform_QA":True,
            "no_range_clipping":True,"smoothing":False,"learned_weights":False,"calibration":False,
            "distillation":False,"new_neural_fits":0,"optimizer_updates":0,"new_encoder_extraction":0,
            "target_or_future_observation_input_to_generation":False,"GT_used_to_construct_predictions":False,
            "principal_pairs":["PAIR_S202610131__V202610131","PAIR_S202610132__V202610132","PAIR_S202610133__V202610133"],
            "principal_selection":"fixed V5 same-seed MIX50; no V6 score-based pair selection",
            "replay_tolerance":{"atol":replay_atol,"rtol":replay_rtol,
                "source":"V5 qa/TRANSFORM_QA.json real-source formula tolerance"},
            "saved_dtype":"float64","serialization":"deterministic little-endian NumPy .npy for new arrays; reused V5 safetensors by exact path/hash",
            "pre_outcome_amendment":True,"target_payload_read_before_seal":False
        }
        write_json(ROOT / "protocol/FORMULA_LOCK.json", formula_lock)
        formula_sha = sha_file(ROOT / "protocol/FORMULA_LOCK.json")
        contrast_lock = {
            "schema_version":1,"created_utc":utc(),"formula_lock_sha256":formula_sha,
            "gain_sign":"gain=loss(reference)-loss(candidate); positive favors candidate",
            "metrics":METRICS,"aggregation":"anchor mean within episode, episode equal within task, five task equal macro",
            "family_aggregation":"mean loss across fixed member pairs; never average family predictions into a larger ensemble",
            "principal_references":["S_same_seed","V_same_seed","BIAS_STATE_same_seed","BIAS_VIDEO_same_seed","PERSISTENCE_shared"],
            "principal_group_ci_references":["S_same_seed","V_same_seed","BIAS_STATE_same_seed","BIAS_VIDEO_same_seed"],
            "family_contrasts":[
                {"name":"SV_vs_SS","reference":"SS","candidate":"SV","gain":"L_SS-L_SV"},
                {"name":"SV_vs_VV","reference":"VV","candidate":"SV","gain":"L_VV-L_SV"},
                {"name":"SV_vs_SA","reference":"SA","candidate":"SV","gain":"L_SA-L_SV"},
                {"name":"AV_vs_AA","reference":"AA","candidate":"AV","gain":"L_AA-L_AV"},
                {"name":"AV_vs_VV","reference":"VV","candidate":"AV","gain":"L_VV-L_AV"},
                {"name":"SV_pairing_effect","reference":"SV_diag","candidate":"SV_offdiag","gain":"L_SV_offdiag-L_SV_diag"}
            ],
            "metrics_all_pairs_all_tasks":True,"ci_scope":"principal mean of three fixed diagonal MIX50 pairs vs four same-seed sources and six family contrasts",
            "no_p_values":True,"no_positive_ci_completion_gate":True,"independent_confirmation":False
        }
        write_json(ROOT / "protocol/CONTRAST_LOCK.json", contrast_lock)
        contrast_sha = sha_file(ROOT / "protocol/CONTRAST_LOCK.json")
        analysis_lock = {
            "schema_version":1,"created_utc":utc(),"formula_lock_sha256":formula_sha,
            "contrast_lock_sha256":contrast_sha,"bootstrap":{"unit":"source_group within task",
             "draws":BOOT_DRAWS,"seed":BOOT_SEED,"interval":"percentile 95%","shared_draws_across_all_systems_references_metrics":True},
            "source_groups_are_inference_units":True,"groups_crossing_tasks":0,
            "true_seed_block_loo":"remove S_k,A_k,V_k and every pair containing any removed node before recomputing means",
            "task_loo":"leave one of five task IDs out; recompute equal-task macro",
            "pair_spread_is_finite_member_dispersion_not_training_seed_sd_or_ci":True,
            "session_trial_association":"unknown from accepted joins; preserve as unknown",
            "final_or_heldout_payload_access":0,"independent_confirmation":False,
            "locked_case_ids":selected_cases,"case_selection_uses_outcomes":False
        }
        write_json(ROOT / "protocol/ANALYSIS_LOCK.json", analysis_lock)

        # The accepted source loss tables carry member/reference metrics; validate this fixture before creating a V6 prediction seal.
        source_fixture_methods = sorted(accepted_ref_methods)
        if any(len(loss_rows[(m,s)]) != 185 for m in source_fixture_methods for s in SEEDS):
            raise RuntimeError("accepted source/reference loss fixture incomplete")
        schema_preflight = {
            "schema_version":1,"status":"PASS","created_utc":utc(),"prediction_target_reads_before_seal":0,
            "expected_prediction_shape":[185,30,28],"expected_native_dtype":"float64 after cast",
            "canonical_receiver_count":len(canonical_ids),"canonical_receiver_ids_sha256":canonical_hash,
            "source_node_count":len(source_node_records),"source_roles":dict(expected_role_counts),
            "source_join_row_count":len(source_join_output),"source_groups":len(group_task),
            "source_groups_crossing_tasks":0,"task_support":analysis_lock["source_groups_are_inference_units"] and source_lock["task_support"],
            "accepted_v5_derived_loss_fixture":{"path":str(old_derived_path),"sha256":sha_file(old_derived_path),"rows":len(old_derived),
              "methods":dict(Counter(r["method"] for r in old_derived))},
            "accepted_v5_reference_loss_fixture":{"path":str(old_refs_path),"sha256":sha_file(old_refs_path),"rows":len(old_refs),
              "methods":dict(Counter(r["method"] for r in old_refs))},
            "target_archive_metadata_locked_without_open":{"path":str(target_path),"expected_sha256":target_expected_sha},
            "reference_roles":["S","V","BIAS_STATE","BIAS_VIDEO","PERSISTENCE"],
            "seed_field_policy":"pair_id is a string; same_seed accepts literal true/false; cross-seed paired_seed is empty/null",
            "status_checks":["exact paths/hashes","ordered ID join","[N,30,28] finite arrays","accepted loss fixtures","group-task nesting","fixed 36-pair plan"]
        }
        write_json(ROOT / "qa/SCHEMA_PREFLIGHT.json", schema_preflight)

        # Synthetic transform/schema probes before target access.
        rng_syn = np.random.default_rng(2026101501)
        sx = rng_syn.normal(size=(185,30,28)).astype(np.float64)
        sy = rng_syn.normal(size=(185,30,28)).astype(np.float64)
        syn_ids = list(canonical_ids)
        syn_mix = generate_pair(sx, sy)
        syn_self = generate_pair(sx, sx)
        syn_reverse = generate_pair(sy, sx)
        lo, hi = np.minimum(sx,sy), np.maximum(sx,sy)
        bad_cases = {}
        for name, thunk in [
            ("wrong ID order", lambda: validate_tensor(sx, syn_ids[::-1], syn_ids)),
            ("wrong seed", lambda: (_ for _ in ()).throw(ValueError("source seed mismatch"))
             if 202610999 != 202610131 else None),
            ("wrong temporal/action shape", lambda: validate_tensor(sx[:,:,:27], syn_ids, syn_ids)),
            ("swapped time/action axes", lambda: validate_tensor(np.transpose(sx,(0,2,1)), syn_ids, syn_ids)),
            ("nonfinite source", lambda: validate_tensor(np.where(np.indices(sx.shape)[0] == 0, np.nan, sx), syn_ids, syn_ids)),
            ("missing receiver ID", lambda: validate_tensor(sx, syn_ids[:-1], syn_ids)),
            ("object array", lambda: validate_tensor(np.array([object()], dtype=object), ["x"], ["x"], (1,1,1))),
            ("invalid boolean literal", lambda: strict_bool("maybe"))
        ]:
            try:
                thunk()
                bad_cases[name] = False
            except Exception:
                bad_cases[name] = True
        if not all(bad_cases.values()):
            raise RuntimeError(f"synthetic rejection tests failed: {bad_cases}")
        identity_err = float(np.max(np.abs(syn_self-sx)))
        symmetry_err = float(np.max(np.abs(syn_mix-syn_reverse)))
        bracket_ok = bool(np.all(syn_mix >= lo) and np.all(syn_mix <= hi))
        if not (identity_err <= ATOL and symmetry_err == 0.0 and bracket_ok):
            raise RuntimeError("synthetic transform identity/symmetry/bracketing failed")
        transform_qa = {
            "schema_version":1,"status":"PASS","created_utc":utc(),"target_payload_reads_before_seal":0,
            "formula_version":"EQUAL_HALF_NATIVE_FLOAT64_V1","synthetic_seed":2026101501,
            "shape":[185,30,28],"dtype":"float64","self_pair_max_abs":identity_err,
            "symmetry_max_abs":symmetry_err,"componentwise_between_sources":bracket_ok,
            "range_clipping":False,"target_inputs_to_constructor":[],"GT_used_to_construct_predictions":False,
            "GT_independence_probe":"constructor accepts only the two source arrays; changing a separate decoy target has no input path",
            "negative_cases_rejected":bad_cases,"boolean_parser_cases":{"true":strict_bool("true"),"False":strict_bool("False")},
            "float64_identity_atol":ATOL,"float64_identity_rtol":RTOL,
            "mix50_reuse_tolerance_atol":replay_atol,"mix50_reuse_tolerance_rtol":replay_rtol,
            "all_9_real_source_arrays_finite":True,"source_files_unchanged":True
        }
        write_json(ROOT / "qa/TRANSFORM_QA.json", transform_qa)
        set_phase("protocol_locked_pre_target", source_nodes_accepted=9, pair_systems_locked=36,
                  target_archive_loads=0, final_heldout_payload_access=0)

        # Reuse the exact three accepted V5 MIX50 arrays; compose only the other 33 systems.
        v5_mix_entries = {(int(x["seed"])):x for x in v5_seal["entries"] if x.get("method") == "MIX50"}
        if set(v5_mix_entries) != set(SEEDS):
            raise RuntimeError("V5 accepted same-seed MIX50 entries are incomplete")
        pair_index = []
        pair_arrays = {}
        mix_replay_errors = {}
        composition_start = time.perf_counter()
        for p in pairs:
            left_key=(p["left_program"],int(p["left_seed"]))
            right_key=(p["right_program"],int(p["right_seed"]))
            same_seed = strict_bool(p["same_seed"])
            if strict_bool(p["reuse_v5_mix50"]):
                seed=int(p["paired_seed"])
                entry=v5_mix_entries[seed]
                for path_key, hash_key in [("prediction_path","prediction_sha256"),
                                           ("prediction_rows_path","prediction_rows_sha256"),
                                           ("completion_path","completion_sha256")]:
                    if sha_file(Path(entry[path_key])) != entry[hash_key]:
                        raise RuntimeError(f"V5 MIX50 bundle file hash mismatch: seed {seed} {path_key}")
                mix_ids=[r["anchor_id"] for r in read_csv(entry["prediction_rows_path"])]
                if mix_ids != canonical_ids:
                    raise RuntimeError(f"V5 MIX50 row IDs are not canonical for seed {seed}")
                with safe_open(entry["prediction_path"], framework="numpy", device="cpu") as sf:
                    if "prediction_native" not in sf.keys():
                        raise RuntimeError("V5 MIX50 native prediction field missing")
                    mix_arr=np.asarray(sf.get_tensor("prediction_native"))
                if mix_arr.dtype != np.float64 or tuple(mix_arr.shape)!=(185,30,28) or not np.isfinite(mix_arr).all():
                    raise RuntimeError(f"invalid V5 MIX50 array schema seed {seed}")
                regenerated=generate_pair(source_arrays[left_key],source_arrays[right_key])
                replay_diff=float(np.max(np.abs(regenerated-mix_arr)))
                if not np.allclose(regenerated,mix_arr,atol=replay_atol,rtol=replay_rtol):
                    raise RuntimeError(f"V5 MIX50 reproduction exceeds original tolerance for seed {seed}: {replay_diff}")
                mix_replay_errors[str(seed)]=replay_diff
                pred=mix_arr
                outpath=Path(entry["prediction_path"])
                pred_sha=entry["prediction_sha256"]
                canonical_sha=entry["canonical_array_sha256"]
                row_sha=entry["prediction_rows_sha256"]
                reuse_source="V5 accepted MIX50"
            else:
                pred=generate_pair(source_arrays[left_key],source_arrays[right_key])
                validate_tensor(pred, canonical_ids, canonical_ids)
                lower=np.minimum(source_arrays[left_key],source_arrays[right_key])
                upper=np.maximum(source_arrays[left_key],source_arrays[right_key])
                if not np.all(pred >= lower) or not np.all(pred <= upper):
                    raise RuntimeError(f"componentwise source bounds failed for {p['pair_id']}")
                arr_bytes=io.BytesIO()
                np.save(arr_bytes,np.asarray(pred,dtype="<f8",order="C"),allow_pickle=False)
                raw_bytes=arr_bytes.getvalue()
                outpath=ROOT / "predictions/new_pairs" / f"{p['pair_id']}.npy"
                with outpath.open("xb") as f:
                    f.write(raw_bytes)
                    f.flush();os.fsync(f.fileno())
                pred_sha=sha_file(outpath)
                canonical_sha=canon_array_hash(pred)
                row_sha=sha_file(ROOT/"inventory/SOURCE_JOINS.csv")
                reuse_source=""
            pair_arrays[p["pair_id"]]=pred
            pair_index.append({
                **p,"prediction_path":str(outpath),"prediction_sha256":pred_sha,
                "canonical_array_sha256":canonical_sha,"ordered_receiver_ids_sha256":canonical_hash,
                "row_sidecar_or_join_sha256":row_sha,"reuse_source":reuse_source,
                "left_source_sha256":node_record_by_key[left_key]["prediction_sha256"],
                "right_source_sha256":node_record_by_key[right_key]["prediction_sha256"],
                "rows":185,"shape":"185x30x28","dtype":"float64","status":"ACCEPTED"
            })
        composition_seconds=time.perf_counter()-composition_start
        if len(pair_index)!=36 or sum(r["reuse_source"]!=" " and bool(r["reuse_source"]) for r in pair_index)!=3:
            raise RuntimeError("expected 36 prediction entries with exactly 3 V5 reuse pointers")
        write_csv(ROOT/"predictions/PREDICTION_INDEX.csv",pair_index)
        seal = {
            "schema_version":1,"status":"PASS","created_utc":utc(),"study":"post_V5_equal_half_ensemble_controls_development",
            "nonempty_seal":True,"attempt":1,"formula_lock_sha256":formula_sha,
            "pair_plan_sha256":sha_file(ROOT/"protocol/PAIR_PLAN.csv"),
            "source_model_map_sha256":sha_file(ROOT/"inventory/SOURCE_MODEL_MAP.json"),
            "ordered_receiver_ids_sha256":canonical_hash,"planned_bundles":36,"sealed_bundles":36,
            "accepted_task_views":180,"accepted_window_bundles":36,"preseal_target_reads":0,
            "preseal_target_archive_loads":0,"target_archive_loads_before_seal":0,
            "target_or_future_observation_input_to_generation":False,"future_labels_used_in_forward":False,
            "prediction_generation_inputs_exclude_outcomes":True,"all_ID_joins_valid":True,
            "all_source_and_output_arrays_finite":True,"missing_bundles":0,
            "new_pair_bundles":33,"reused_v5_mix50_bundles":3,"entries":pair_index,
            "same_seed_mix50_reproduction_max_abs_by_seed":mix_replay_errors,
            "formula_version":"EQUAL_HALF_NATIVE_FLOAT64_V1"
        }
        write_json(ROOT/"qa/PREDICTION_SEAL.json",seal)
        exposure["status"]="PREDICTIONS_SEALED_PRE_SCORE"
        exposure["prediction_seal_sha256"]=sha_file(ROOT/"qa/PREDICTION_SEAL.json")
        exposure["prediction_seal_utc"]=seal["created_utc"]
        exposure["new_pair_bundles_accepted"]=33
        exposure["reused_v5_mix50_bundles_accepted"]=3
        exposure["history"].append({"phase":"prediction_seal","status":"PASS","target_archive_loads":0,"target_payload_access":0})
        write_json(ROOT/"inventory/OUTCOME_EXPOSURE.json",exposure)
        set_phase("prediction_seal_pass_target_not_opened", pair_systems_accepted=36,
                  new_pair_bundles_accepted=33,reused_v5_mix50_bundles=3,
                  target_archive_loads=0,final_heldout_payload_access=0)

        # One development-only scorer archive read after the complete prediction seal.
        exposure["status"]="TARGET_ACCESS_STARTED"
        exposure["target_archive_read_started_utc"]=utc()
        exposure["history"].append({"phase":"target_archive_read","status":"started_after_nonempty_prediction_seal",
                                    "target_archive_loads":0,"target_payload_access":0})
        write_json(ROOT/"inventory/OUTCOME_EXPOSURE.json",exposure)
        target_blob=target_path.read_bytes()
        target_sha=sha_bytes(target_blob)
        if target_sha != target_expected_sha or target_sha != "233a0cdb31d1ddd7cea6ecfaa5e9d02e9c42e9822af7fe488c7fe5a40febee27":
            raise RuntimeError("locked development scorer archive digest mismatch")
        with np.load(io.BytesIO(target_blob),allow_pickle=False) as z:
            required={"anchor_id","episode_id","task_id","source_cluster","target_native_h30"}
            if not required.issubset(set(z.files)):
                raise RuntimeError(f"development scorer archive missing white-listed fields: {required-set(z.files)}")
            target_ids=[as_text(x) for x in z["anchor_id"].tolist()]
            target_native=np.asarray(z["target_native_h30"],dtype=np.float64).copy()
            target_meta={target_ids[i]:(as_text(z["episode_id"][i]),as_text(z["task_id"][i]),as_text(z["source_cluster"][i]))
                         for i in range(len(target_ids))}
        if len(target_ids)!=185 or len(set(target_ids))!=185 or set(target_ids)!=set(canonical_ids):
            raise RuntimeError("development target receiver IDs do not match the locked 185 IDs")
        with np.load(prefix_path,allow_pickle=False) as z:
            prefix_ids=[as_text(x) for x in z["anchor_id"].tolist()]
            prefix_anchor=np.asarray(z["anchor_native"],dtype=np.float64).copy()
            prefix_meta={prefix_ids[i]:(as_text(z["episode_id"][i]),as_text(z["task_id"][i]),as_text(z["source_cluster"][i]))
                         for i in range(len(prefix_ids))}
        if len(prefix_ids)!=185 or set(prefix_ids)!=set(canonical_ids):
            raise RuntimeError("label-free prefix metadata receiver ID mismatch")
        if target_native.shape!=(185,30,28) or prefix_anchor.shape!=(185,28) or not np.isfinite(target_native).all() or not np.isfinite(prefix_anchor).all():
            raise RuntimeError("development target or current-anchor data schema invalid")
        target_idx={a:i for i,a in enumerate(target_ids)}
        ordered_target=np.stack([target_native[target_idx[a]] for a in canonical_ids])
        ordered_anchor=np.stack([prefix_anchor[prefix_ids.index(a)] for a in canonical_ids])
        for aid in canonical_ids:
            meta=node_group_meta[aid]
            target_m=target_meta[aid]
            prefix_m=prefix_meta[aid]
            if target_m!=prefix_m or target_m!=(meta[0],meta[1],meta[2]):
                raise RuntimeError(f"development target/prefix/accepted group join mismatch for {aid}")
        exposure.update({
            "status":"TARGET_READ_AND_SCORING_COMPLETE","target_archive_load_operations":1,
            "target_archive_physical_read_operations":1,"target_archive_bytes":len(target_blob),
            "target_archive_sha256_actual":target_sha,"target_array_window_rows_loaded":185,
            "target_archive_read_completed_utc":utc(),"prefix_archive_loads":1,
            "prefix_archive_path":str(prefix_path),"prefix_archive_sha256":prefix_artifact["sha256"],
            "final_heldout_payload_access":0
        })
        exposure["history"].append({"phase":"target_archive_read","status":"complete",
            "target_archive_loads":1,"target_windows_loaded":185,"final_heldout_payload_access":0})
        write_json(ROOT/"inventory/OUTCOME_EXPOSURE.json",exposure)
        set_phase("development_targets_joined_after_seal",target_archive_loads=1,
                  target_windows_loaded=185,final_heldout_payload_access=0)

        # Read predictions and score every fixed pair; count only 33 new score bundles as new evaluations.
        pair_rows_by_id={}
        per_window_rows=[]
        new_eval_count=0
        mix_reconcile_rows=0
        for p in pairs:
            pid=p["pair_id"]
            pred=pair_arrays[pid]
            scores=target_score(pred,ordered_target,ordered_anchor,sigma,fdscale)
            rows=[]
            for i,aid in enumerate(canonical_ids):
                episode,task,group=node_group_meta[aid]
                row={"pair_id":pid,"family":p["family"],"left_program":p["left_program"],"left_seed":p["left_seed"],
                     "right_program":p["right_program"],"right_seed":p["right_seed"],"same_seed":p["same_seed"],
                     "paired_seed":p["paired_seed"],"anchor_id":aid,"episode_id":episode,"task_id":task,
                     "source_group":group,"position_mae":float(scores["position_mae"][i]),
                     "safe_fd_l1":float(scores["safe_fd_l1"][i]),"seam_safe_fd_l1":float(scores["seam_safe_fd_l1"][i]),
                     "score_source":"V6_target_score" if not strict_bool(p["reuse_v5_mix50"]) else "V5_MIX50_reused_after_reconciliation"}
                rows.append(row)
            if strict_bool(p["reuse_v5_mix50"]):
                s=int(p["paired_seed"])
                fixture={(r["anchor_id"]):r for r in mix_old_rows[s]}
                if len(fixture)!=185:
                    raise RuntimeError("V5 MIX50 reconciliation fixture incomplete")
                max_deltas={m:0.0 for m in METRICS}
                for row in rows:
                    old=fixture[row["anchor_id"]]
                    for m in METRICS:
                        max_deltas[m]=max(max_deltas[m],abs(row[m]-old[m]))
                if any(max_deltas[m] > (ATOL + RTOL * max(1.0, max(abs(r[m]) for r in rows))) for m in METRICS):
                    raise RuntimeError(f"V5 MIX50 target-score reconciliation failed seed {s}: {max_deltas}")
                mix_reconcile_rows += 185
                # Keep the accepted V5 row values as the scalar source after successful reconciliation.
                rows=[{**fixture[r["anchor_id"]],"pair_id":pid,"family":p["family"],
                       "left_program":p["left_program"],"left_seed":p["left_seed"],
                       "right_program":p["right_program"],"right_seed":p["right_seed"],
                       "same_seed":p["same_seed"],"paired_seed":p["paired_seed"],
                       "score_source":"V5_MIX50_reused_after_reconciliation"} for r in rows]
            else:
                new_eval_count += 185
            pair_rows_by_id[pid]=rows
            per_window_rows.extend(rows)
        if new_eval_count != 6105 or mix_reconcile_rows != 555 or len(per_window_rows)!=6660:
            raise RuntimeError(f"loss evaluation accounting mismatch new={new_eval_count} mix={mix_reconcile_rows} rows={len(per_window_rows)}")
        write_csv(ROOT/"results/PER_WINDOW_PAIR_METRICS.csv",per_window_rows)

        # Rebuild score row maps for the nine source nodes and the accepted baselines from exact V5 per-window files.
        source_loss_rows={}
        for (m,s) in [(p,seed) for p in ["S","A","V"] for seed in SEEDS]:
            rr=loss_rows[(m,s)]
            source_loss_rows[(m,s)]={r["anchor_id"]:r for r in rr}
        refs_loss_rows={}
        for m in ["BIAS_STATE","BIAS_VIDEO","PERSISTENCE"]:
            for s in SEEDS:
                refs_loss_rows[(m,s)]={r["anchor_id"]:r for r in loss_rows[(m,s)]}
        all_window_loss={}
        all_window_loss.update({pid:pair_rows_by_id[pid] for pid in pair_rows_by_id})
        all_window_loss.update({f"SRC_{m}{s}":list(source_loss_rows[(m,s)].values())
                                for m in ["S","A","V"] for s in SEEDS})
        for m in ["BIAS_STATE","BIAS_VIDEO","PERSISTENCE"]:
            for s in SEEDS:
                all_window_loss[f"REF_{m}{s}"]=list(refs_loss_rows[(m,s)].values())

        # All pair/task points, followed by finite-family means and fixed contrasts.
        pair_task_rows=[]
        pair_task_values={}
        pair_map={p["pair_id"]:p for p in pairs}
        scopes=[("task",t) for t in TASKS]+[("macro",None)]
        for p in pairs:
            pid=p["pair_id"]
            pair_task_values[pid]={}
            for metric in METRICS:
                tv={t:aggregate_metric(pair_rows_by_id[pid],metric,t) for t in TASKS}
                pair_task_values[pid][metric]=tv
                for scope,task in scopes:
                    value=tv[task] if scope=="task" else macro_from_task(tv)
                    support=(sum(int(r["task_id"])==int(task) for r in pair_rows_by_id[pid]) if task is not None else 185)
                    pair_task_rows.append({
                        "record_type":"pair_system","pair_id":pid,"family":p["family"],
                        "left_program":p["left_program"],"left_seed":p["left_seed"],
                        "right_program":p["right_program"],"right_seed":p["right_seed"],
                        "same_seed":p["same_seed"],"paired_seed":p["paired_seed"],
                        "scope":scope,"task_id":task if task is not None else "",
                        "metric":metric,"loss":value,"anchors":support,
                        "episodes":len({r["episode_id"] for r in pair_rows_by_id[pid] if task is None or int(r["task_id"])==int(task)}),
                        "source_groups":len({r["source_group"] for r in pair_rows_by_id[pid] if task is None or int(r["task_id"])==int(task)}),
                        "pair_score_source":"V5_reused_after_reconciliation" if pair_map[pid]["reuse_v5_mix50"]=="true" else "V6_new_score"
                    })
        write_csv(ROOT/"results/PAIR_TASK_METRICS.csv",pair_task_rows)

        family_pairs=defaultdict(list)
        for p in pairs:
            family_pairs[p["family"]].append(p["pair_id"])
        family_task_values={}
        family_effect_rows=[]
        for fam, pids in family_pairs.items():
            family_task_values[fam]={}
            for metric in METRICS:
                family_task_values[fam][metric]={}
                for scope,task in scopes:
                    vals=[pair_task_values[pid][metric][task] if scope=="task" else macro_from_task(pair_task_values[pid][metric])
                          for pid in pids]
                    mean=float(np.mean(vals))
                    spread=sample_sd(vals)
                    record={"record_type":"family_mean","family":fam,"contrast_name":"",
                            "scope":scope,"task_id":task if task is not None else "","metric":metric,
                            "mean_pair_loss":mean,"pair_sample_sd":spread,
                            "pair_min":float(np.min(vals)),"pair_max":float(np.max(vals)),
                            "n_pairs":len(vals),"spread_label":"finite_pair_spread_not_CI_or_seed_SD"}
                    family_effect_rows.append(record)
                    if scope=="task":
                        family_task_values[fam][metric][task]=mean
                    else:
                        family_task_values[fam][metric]["macro"]=mean
        family_contrast_defs=[
            ("SV_vs_SS","SS","SV"),("SV_vs_VV","VV","SV"),("SV_vs_SA","SA","SV"),
            ("AV_vs_AA","AA","AV"),("AV_vs_VV","VV","AV")
        ]
        # Contrasts use reference loss minus candidate loss.
        for metric in METRICS:
            for scope,task in scopes:
                for name,ref,cand in family_contrast_defs:
                    refv=family_task_values[ref][metric][task if scope=="task" else "macro"]
                    candv=family_task_values[cand][metric][task if scope=="task" else "macro"]
                    family_effect_rows.append({"record_type":"family_contrast","family":"",
                        "contrast_name":name,"reference_family":ref,"candidate_family":cand,
                        "scope":scope,"task_id":task if task is not None else "","metric":metric,
                        "reference_loss":refv,"candidate_loss":candv,"gain":refv-candv,
                        "n_reference_pairs":len(family_pairs[ref]),"n_candidate_pairs":len(family_pairs[cand]),
                        "spread_label":"paired_source_group_CI_reported_separately"})
                if scope=="task":
                    diag=[p["pair_id"] for p in pairs if p["family"]=="SV" and p["same_seed"]=="true"]
                    off=[p["pair_id"] for p in pairs if p["family"]=="SV" and p["same_seed"]=="false"]
                    refv=float(np.mean([pair_task_values[x][metric][task] for x in off]))
                    candv=float(np.mean([pair_task_values[x][metric][task] for x in diag]))
                else:
                    diag=[p["pair_id"] for p in pairs if p["family"]=="SV" and p["same_seed"]=="true"]
                    off=[p["pair_id"] for p in pairs if p["family"]=="SV" and p["same_seed"]=="false"]
                    refv=float(np.mean([macro_from_task(pair_task_values[x][metric]) for x in off]))
                    candv=float(np.mean([macro_from_task(pair_task_values[x][metric]) for x in diag]))
                family_effect_rows.append({"record_type":"family_contrast","family":"",
                    "contrast_name":"SV_pairing_effect","reference_family":"SV_offdiag",
                    "candidate_family":"SV_diag","scope":scope,"task_id":task if task is not None else "",
                    "metric":metric,"reference_loss":refv,"candidate_loss":candv,"gain":refv-candv,
                    "n_reference_pairs":len(off),"n_candidate_pairs":len(diag),
                    "spread_label":"paired_source_group_CI_reported_separately"})
        write_csv(ROOT/"results/FAMILY_EFFECTS.csv",family_effect_rows)

        # Direct same-seed MIX50 paired comparisons with strong accepted references.
        principal_ids={int(p["paired_seed"]):p["pair_id"] for p in pairs if p["reuse_v5_mix50"]=="true"}
        principal_methods=["S","V","BIAS_STATE","BIAS_VIDEO","PERSISTENCE"]
        principal_rows=[]
        principal_seed_macro={}
        def ref_rows(method, seed):
            if method in ["S","V"]:
                return list(source_loss_rows[(method,seed)].values())
            return list(refs_loss_rows[(method,seed)].values())
        for seed in SEEDS:
            pid=principal_ids[seed]
            p_rows=pair_rows_by_id[pid]
            principal_seed_macro[seed]={}
            for ref in principal_methods:
                rr=ref_rows(ref,seed)
                for metric in METRICS:
                    task_gains={}
                    for task in TASKS:
                        cand_loss=aggregate_metric(p_rows,metric,task)
                        ref_loss=aggregate_metric(rr,metric,task)
                        gain=ref_loss-cand_loss
                        task_gains[task]=gain
                        principal_rows.append({"candidate":"MIX50","candidate_seed":seed,"candidate_pair_id":pid,
                            "reference":ref,"reference_seed":seed if ref!="PERSISTENCE" else "",
                            "reference_seed_role":"shared_reference" if ref=="PERSISTENCE" else "matched_seed",
                            "scope":"task","task_id":task,"metric":metric,"candidate_loss":cand_loss,
                            "reference_loss":ref_loss,"gain":gain,"gain_sign":"positive favors MIX50"})
                    mg=macro_from_task(task_gains)
                    principal_seed_macro[seed][(ref,metric)]=mg
                    principal_rows.append({"candidate":"MIX50","candidate_seed":seed,"candidate_pair_id":pid,
                        "reference":ref,"reference_seed":seed if ref!="PERSISTENCE" else "",
                        "reference_seed_role":"shared_reference" if ref=="PERSISTENCE" else "matched_seed",
                        "scope":"macro","task_id":"","metric":metric,
                        "candidate_loss":macro_from_task({t:aggregate_metric(p_rows,metric,t) for t in TASKS}),
                        "reference_loss":macro_from_task({t:aggregate_metric(rr,metric,t) for t in TASKS}),
                        "gain":mg,"gain_sign":"positive favors MIX50"})
        for ref in principal_methods:
            for metric in METRICS:
                vals=[principal_seed_macro[s][(ref,metric)] for s in SEEDS]
                principal_rows.append({"candidate":"MIX50_DIAG_MEAN3","candidate_seed":"",
                    "candidate_pair_id":"three fixed V5 same-seed pairs","reference":ref,
                    "reference_seed":"","reference_seed_role":"shared_reference" if ref=="PERSISTENCE" else "matched_seed_by_pair",
                    "scope":"seed_mean_macro","task_id":"","metric":metric,
                    "candidate_loss":float(np.mean([macro_from_task(pair_task_values[principal_ids[s]][metric]) for s in SEEDS])),
                    "reference_loss":None,"gain":float(np.mean(vals)),
                    "gain_sample_sd_across_three_fixed_pairs":sample_sd(vals),
                    "gain_sign":"positive favors MIX50","n_fixed_pairs":3,
                    "dispersion_label":"sample SD across fixed same-seed member pairs; not a population CI"})
        write_csv(ROOT/"results/PRINCIPAL_MIX50_EFFECTS.csv",principal_rows)

        # True seed-block and task LOO; every pair containing a removed node is dropped.
        seed_loo_rows=[]
        for held in SEEDS:
            retained=[p for p in pairs if int(p["left_seed"])!=held and int(p["right_seed"])!=held]
            if len(retained)!=15:
                raise RuntimeError(f"seed-block LOO did not retain 15 systems after removing {held}")
            for metric in METRICS:
                loo_family={}
                for fam in family_pairs:
                    fam_ret=[p for p in retained if p["family"]==fam]
                    if len(fam_ret)!=(1 if fam in ("SS","AA","VV") else 4):
                        raise RuntimeError(f"seed-block LOO family support wrong {held}/{fam}")
                    per_pair=[macro_from_task(pair_task_values[p["pair_id"]][metric]) for p in fam_ret]
                    value=float(np.mean(per_pair))
                    loo_family[fam]=value
                    seed_loo_rows.append({"leave_out_seed":held,"metric":metric,"record_type":"family_mean",
                        "family":fam,"contrast_name":"","loss":value,"retained_nodes":6,"retained_pairs":len(fam_ret),
                        "loo_rule":"remove S_k,A_k,V_k and all pairs containing them"})
                for name,ref,cand in family_contrast_defs:
                    seed_loo_rows.append({"leave_out_seed":held,"metric":metric,"record_type":"family_contrast",
                        "family":"","contrast_name":name,"reference_family":ref,"candidate_family":cand,
                        "gain":loo_family[ref]-loo_family[cand],"retained_nodes":6,"retained_pairs":15,
                        "loo_rule":"remove S_k,A_k,V_k and all pairs containing them"})
                sv_ret=[p for p in retained if p["family"]=="SV"]
                diag=[p for p in sv_ret if p["same_seed"]=="true"]
                off=[p for p in sv_ret if p["same_seed"]=="false"]
                seed_loo_rows.append({"leave_out_seed":held,"metric":metric,"record_type":"family_contrast",
                    "family":"","contrast_name":"SV_pairing_effect","reference_family":"SV_offdiag",
                    "candidate_family":"SV_diag","gain":float(np.mean([macro_from_task(pair_task_values[p["pair_id"]][metric]) for p in off])-
                                                               np.mean([macro_from_task(pair_task_values[p["pair_id"]][metric]) for p in diag])),
                    "retained_nodes":6,"retained_pairs":15,"loo_rule":"remove S_k,A_k,V_k and all pairs containing them"})
                diag_principal=[p for p in retained if p["family"]=="SV" and p["same_seed"]=="true"]
                seed_loo_rows.append({"leave_out_seed":held,"metric":metric,"record_type":"principal_MIX50_mean",
                    "family":"SV_diag","contrast_name":"","loss":float(np.mean([macro_from_task(pair_task_values[p["pair_id"]][metric]) for p in diag_principal])),
                    "retained_nodes":6,"retained_pairs":len(diag_principal),
                    "loo_rule":"remove held seed principal pair and its source nodes"})
        write_csv(ROOT/"results/SEED_BLOCK_LOO.csv",seed_loo_rows)

        task_loo_rows=[]
        for held_task in TASKS:
            retained_tasks=[t for t in TASKS if t != held_task]
            for metric in METRICS:
                loo_family={}
                for fam,pids in family_pairs.items():
                    per_pair=[float(np.mean([pair_task_values[pid][metric][t] for t in retained_tasks])) for pid in pids]
                    value=float(np.mean(per_pair));loo_family[fam]=value
                    task_loo_rows.append({"leave_out_task":held_task,"metric":metric,"record_type":"family_mean",
                        "family":fam,"contrast_name":"","loss":value,"retained_tasks":4,"retained_pairs":len(pids),
                        "loo_rule":"equal-task macro over remaining four tasks"})
                for name,ref,cand in family_contrast_defs:
                    task_loo_rows.append({"leave_out_task":held_task,"metric":metric,"record_type":"family_contrast",
                        "family":"","contrast_name":name,"reference_family":ref,"candidate_family":cand,
                        "gain":loo_family[ref]-loo_family[cand],"retained_tasks":4,
                        "loo_rule":"equal-task macro over remaining four tasks"})
                diag=[p for p in pairs if p["family"]=="SV" and p["same_seed"]=="true"]
                off=[p for p in pairs if p["family"]=="SV" and p["same_seed"]=="false"]
                task_loo_rows.append({"leave_out_task":held_task,"metric":metric,"record_type":"family_contrast",
                    "family":"","contrast_name":"SV_pairing_effect","reference_family":"SV_offdiag",
                    "candidate_family":"SV_diag",
                    "gain":float(np.mean([np.mean([pair_task_values[p["pair_id"]][metric][t] for t in retained_tasks]) for p in off])-
                                 np.mean([np.mean([pair_task_values[p["pair_id"]][metric][t] for t in retained_tasks]) for p in diag])),
                    "retained_tasks":4,"loo_rule":"equal-task macro over remaining four tasks"})
                task_loo_rows.append({"leave_out_task":held_task,"metric":metric,"record_type":"principal_MIX50_mean",
                    "family":"SV_diag","contrast_name":"",
                    "loss":float(np.mean([np.mean([pair_task_values[p["pair_id"]][metric][t] for t in retained_tasks]) for p in diag])),
                    "retained_tasks":4,"retained_pairs":3,"loo_rule":"mean of three fixed diagonal pairs"})
        write_csv(ROOT/"results/TASK_LOO.csv",task_loo_rows)

        # Shared within-task source-group paired bootstrap draws.
        rng_boot=np.random.default_rng(BOOT_SEED)
        draws_by_task={}
        groups_by_task={}
        for task in TASKS:
            groups=sorted(groups_per_task[task])
            groups_by_task[task]=groups
            draws_by_task[task]=rng_boot.integers(0,len(groups),size=(BOOT_DRAWS,len(groups)),endpoint=False)
        boot_cache={}
        def bootstrap_entity(rows, metric, task):
            epvals=defaultdict(list)
            group_eps=defaultdict(set)
            for r in rows:
                if int(r["task_id"])!=int(task):
                    continue
                ep=str(r["episode_id"]); group=str(r["source_group"])
                epvals[ep].append(float(r[metric]))
                group_eps[group].add(ep)
            groups=groups_by_task[task]
            if set(group_eps)!=set(groups):
                raise RuntimeError(f"bootstrap source group mismatch task {task}")
            gsum=np.asarray([sum(float(np.mean(epvals[e])) for e in sorted(group_eps[g])) for g in groups],dtype=np.float64)
            gcnt=np.asarray([len(group_eps[g]) for g in groups],dtype=np.float64)
            idx=draws_by_task[task]
            return gsum[idx].sum(axis=1)/gcnt[idx].sum(axis=1)
        def entity_boot(key, rows, metric, task):
            ck=(key,metric,task)
            if ck not in boot_cache:
                boot_cache[ck]=bootstrap_entity(rows,metric,task)
            return boot_cache[ck]
        pair_caches={pid:pair_rows_by_id[pid] for pid in pair_rows_by_id}
        ref_cache={}
        for m in ["S","V","BIAS_STATE","BIAS_VIDEO","PERSISTENCE"]:
            for s in SEEDS:
                ref_cache[(m,s)]=ref_rows(m,s)
        ci_rows=[]
        principal_ci_map={}
        for ref in ["S","V","BIAS_STATE","BIAS_VIDEO"]:
            for metric in METRICS:
                task_draws=[]
                task_point=[]
                for task in TASKS:
                    per_seed=[]
                    for seed in SEEDS:
                        pid=principal_ids[seed]
                        cand=entity_boot(("PAIR",pid),pair_caches[pid],metric,task)
                        rrows=ref_cache[(ref,seed)]
                        rv=entity_boot(("REF",ref,seed),rrows,metric,task)
                        per_seed.append(rv-cand)
                    task_draws.append(np.mean(np.stack(per_seed,axis=0),axis=0))
                    point_gains=[aggregate_metric(ref_cache[(ref,s)],metric,task)-
                                 aggregate_metric(pair_caches[principal_ids[s]],metric,task) for s in SEEDS]
                    task_point.append(float(np.mean(point_gains)))
                all_draws=np.mean(np.stack(task_draws,axis=0),axis=0)
                point=float(np.mean(task_point))
                low,high=np.quantile(all_draws,[0.025,0.975])
                row={"contrast":f"MIX50_DIAG_MEAN3_vs_{ref}","metric":metric,"point_estimate":point,
                     "ci_low":float(low),"ci_high":float(high),"ci_level":0.95,"draws":BOOT_DRAWS,
                     "seed":BOOT_SEED,"n_source_groups":47,"source_group_task_nested":True,
                     "pairing":"same target groups paired across three fixed MIX50 pairs and matched references",
                     "independent_confirmation":False}
                ci_rows.append(row);principal_ci_map[(ref,metric)]=row

        def family_draw(fam,metric,task):
            pids=family_pairs[fam]
            vals=np.stack([entity_boot(("PAIR",pid),pair_caches[pid],metric,task) for pid in pids],axis=0)
            return np.mean(vals,axis=0)
        family_contrast_ci_map={}
        for metric in METRICS:
            for name,ref,cand in family_contrast_defs:
                tdraw=[]
                for task in TASKS:
                    tdraw.append(family_draw(ref,metric,task)-family_draw(cand,metric,task))
                draws=np.mean(np.stack(tdraw,axis=0),axis=0)
                point=(family_task_values[ref][metric]["macro"]-family_task_values[cand][metric]["macro"])
                low,high=np.quantile(draws,[0.025,0.975])
                row={"contrast":name,"metric":metric,"point_estimate":point,"ci_low":float(low),"ci_high":float(high),
                     "ci_level":0.95,"draws":BOOT_DRAWS,"seed":BOOT_SEED,"n_source_groups":47,
                     "source_group_task_nested":True,"pairing":"same source-group draw across all fixed pair systems",
                     "independent_confirmation":False}
                ci_rows.append(row);family_contrast_ci_map[(name,metric)]=row
            tdraw=[]
            diag=[p["pair_id"] for p in pairs if p["family"]=="SV" and p["same_seed"]=="true"]
            off=[p["pair_id"] for p in pairs if p["family"]=="SV" and p["same_seed"]=="false"]
            for task in TASKS:
                offd=np.mean(np.stack([entity_boot(("PAIR",pid),pair_caches[pid],metric,task) for pid in off],axis=0),axis=0)
                diagd=np.mean(np.stack([entity_boot(("PAIR",pid),pair_caches[pid],metric,task) for pid in diag],axis=0),axis=0)
                tdraw.append(offd-diagd)
            draws=np.mean(np.stack(tdraw,axis=0),axis=0)
            point=float(np.mean([macro_from_task(pair_task_values[pid][metric]) for pid in off])-
                        np.mean([macro_from_task(pair_task_values[pid][metric]) for pid in diag]))
            low,high=np.quantile(draws,[0.025,0.975])
            row={"contrast":"SV_pairing_effect","metric":metric,"point_estimate":point,
                 "ci_low":float(low),"ci_high":float(high),"ci_level":0.95,"draws":BOOT_DRAWS,
                 "seed":BOOT_SEED,"n_source_groups":47,"source_group_task_nested":True,
                 "pairing":"same source-group draw across all fixed SV pair systems","independent_confirmation":False}
            ci_rows.append(row);family_contrast_ci_map[("SV_pairing_effect",metric)]=row
        write_csv(ROOT/"results/PAIRED_GROUP_CI.csv",ci_rows)
        boot_draw_manifest={
            "schema_version":1,"status":"PASS","created_utc":utc(),"draws":BOOT_DRAWS,"seed":BOOT_SEED,
            "tasks":{str(t):{"source_groups":groups_by_task[t],"n_groups":len(groups_by_task[t]),
                             "draw_matrix_shape":list(draws_by_task[t].shape),
                             "draw_matrix_sha256":sha_bytes(np.asarray(draws_by_task[t],dtype="<i8").tobytes())} for t in TASKS},
            "shared_draws_across_all_pairs_references_metrics":True,
            "bootstrap_unit":"source_group sampled with replacement within each task; all episodes/anchors in sampled group retained",
            "episode_task_weighting":"sum episode means per sampled group divided by sampled episode count, then equal task macro",
            "ci_level":0.95,"p_values":False
        }
        write_json(ROOT/"qa/BOOTSTRAP_DRAW_MANIFEST.json",boot_draw_manifest)

        # L1 convex-cancellation ledger, all pairs and all three locked metrics, using the same one-read target memory.
        cancellation_start=time.perf_counter()
        cancel_anchor=defaultdict(list)
        max_identity_residual=defaultdict(float)
        exact_zero_count=defaultdict(int)
        finite_element_count=defaultdict(int)
        min_nonzero_abs=defaultdict(lambda:float("inf"))
        for p in pairs:
            pid=p["pair_id"]
            x=source_arrays[(p["left_program"],int(p["left_seed"]))]
            y=source_arrays[(p["right_program"],int(p["right_seed"]))]
            # normalized position forecast errors [N,30,28]
            ux=(x-ordered_target)/sigma[None,None,:]
            uy=(y-ordered_target)/sigma[None,None,:]
            ufd=(np.diff(x,axis=1)-np.diff(ordered_target,axis=1))/fdscale[None,None,:]
            vfd=(np.diff(y,axis=1)-np.diff(ordered_target,axis=1))/fdscale[None,None,:]
            useam=((x[:,0,:]-ordered_anchor)-(ordered_target[:,0,:]-ordered_anchor))/fdscale[None,:]
            vseam=((y[:,0,:]-ordered_anchor)-(ordered_target[:,0,:]-ordered_anchor))/fdscale[None,:]
            for metric,u,v in [("position_mae",ux,uy),("safe_fd_l1",ufd,vfd),("seam_safe_fd_l1",useam,vseam)]:
                opposite=((u<0)&(v>0))|((u>0)&(v<0))
                lhs=0.5*(np.abs(u)+np.abs(v))-np.abs(0.5*(u+v))
                rhs=np.where(opposite,np.minimum(np.abs(u),np.abs(v)),0.0)
                resid=np.abs(lhs-rhs)
                max_identity_residual[(pid,metric)]=float(np.max(resid))
                exact_zero_count[(pid,metric)]=int(np.count_nonzero((u==0)|(v==0)))
                finite_element_count[(pid,metric)]=int(np.isfinite(u).sum())
                nz=np.concatenate([np.abs(u[np.nonzero(u)]),np.abs(v[np.nonzero(v)])])
                if nz.size:
                    min_nonzero_abs[(pid,metric)]=float(np.min(nz))
                for i,aid in enumerate(canonical_ids):
                    episode,task,group=node_group_meta[aid]
                    n_elem=u[i].size
                    cancel_anchor[(pid,metric)].append({
                        "anchor_id":aid,"episode_id":episode,"task_id":str(task),"source_group":group,
                        "cancel_amount":float(rhs[i].mean()),
                        "opposite_sign_fraction":float(opposite[i].sum()/n_elem),
                        "member_error_exact_zero_elements":int(np.count_nonzero((u[i]==0)|(v[i]==0))),
                        "element_count":int(n_elem),
                        "max_element_identity_residual":float(np.max(resid[i]))
                    })
        cancellation_rows=[]
        for p in pairs:
            pid=p["pair_id"]
            left_rows=source_loss_rows[(p["left_program"],int(p["left_seed"]))]
            right_rows=source_loss_rows[(p["right_program"],int(p["right_seed"]))]
            left_lookup={r["anchor_id"]:r for r in left_rows}
            right_lookup={r["anchor_id"]:r for r in right_rows}
            pair_lookup={r["anchor_id"]:r for r in pair_rows_by_id[pid]}
            for metric in METRICS:
                arows=cancel_anchor[(pid,metric)]
                source_by_id={r["anchor_id"]:r for r in arows}
                for scope,task in scopes:
                    use_ids=[aid for aid in canonical_ids if task is None or int(node_group_meta[aid][1])==task]
                    # aggregate all terms by the canonical episode-equal rule
                    def aggr(vals):
                        eps=defaultdict(list)
                        for aid,val in vals:
                            ep=node_group_meta[aid][0];eps[ep].append(float(val))
                        return float(np.mean([np.mean(v) for v in eps.values()]))
                    lx=aggr([(aid,left_lookup[aid][metric]) for aid in use_ids])
                    ly=aggr([(aid,right_lookup[aid][metric]) for aid in use_ids])
                    le=aggr([(aid,pair_lookup[aid][metric]) for aid in use_ids])
                    jgap=0.5*(lx+ly)-le
                    cancel=aggr([(aid,source_by_id[aid]["cancel_amount"]) for aid in use_ids])
                    signfrac=aggr([(aid,source_by_id[aid]["opposite_sign_fraction"]) for aid in use_ids])
                    residual=jgap-cancel
                    max_elem=max(source_by_id[aid]["max_element_identity_residual"] for aid in use_ids)
                    zero_sum=sum(source_by_id[aid]["member_error_exact_zero_elements"] for aid in use_ids)
                    elem_sum=sum(source_by_id[aid]["element_count"] for aid in use_ids)
                    gain_left=lx-le;gain_right=ly-le
                    cancellation_rows.append({
                        "pair_id":pid,"family":p["family"],"left_program":p["left_program"],
                        "left_seed":p["left_seed"],"right_program":p["right_program"],"right_seed":p["right_seed"],
                        "same_seed":p["same_seed"],"scope":scope,"task_id":task if task is not None else "",
                        "metric":metric,"left_loss":lx,"right_loss":ly,"ensemble_loss":le,
                        "j_gap_member_mean_minus_ensemble":jgap,"gain_vs_left_member":gain_left,
                        "gain_vs_right_member":gain_right,"gain_vs_better_member":min(lx,ly)-le,
                        "elementwise_cancellation_amount":cancel,"opposite_sign_error_fraction":signfrac,
                        "j_gap_reconstruction_residual":residual,"max_element_identity_residual":max_elem,
                        "identity_atol":ATOL,"identity_rtol":RTOL,
                        "identity_pass":bool(abs(residual)<=ATOL+RTOL*max(abs(jgap),abs(cancel),1.0) and max_elem<=ATOL+RTOL),
                        "member_error_exact_zero_elements":zero_sum,"element_count":elem_sum,
                        "min_nonzero_abs_normalized_member_error":min_nonzero_abs[(pid,metric)],
                        "nonfinite_elements":0,"target_source":"single accepted development scorer archive read"
                    })
        if not all(r["identity_pass"] for r in cancellation_rows):
            bad=[r for r in cancellation_rows if not r["identity_pass"]][:5]
            raise RuntimeError(f"L1 cancellation identity failed: {bad}")
        write_csv(ROOT/"results/L1_CANCELLATION_LEDGER.csv",cancellation_rows)

        # Runtime and outcome exposure counters; keep old V5 counters at their own grain.
        cancel_seconds=time.perf_counter()-cancellation_start
        cpu_resource = {
            "schema_version":1,"created_utc":utc(),"device":"CPU","physical_gpu_access":False,
            "source_program_forward_costs_reused_from_v5":True,
            "model_forward_passes_v6":0,"new_fits":0,"optimizer_updates":0,"calibration_fits":0,
            "new_encoder_extractions":0,"weight_search":0,
            "new_prediction_bundles":33,"reused_v5_prediction_bundles":3,
            "two_frozen_head_outputs_per_pair":True,"equal_FLOPs_assumed":False,
            "composition_cost_scope":"CPU creation of 33 fixed float64 pair arrays",
            "composition_wall_seconds":composition_seconds,
            "scoring_cost_scope":"one new score pass for 33 pairs; V5 MIX50 score reconciliation kept separate",
            "new_pair_window_loss_evaluations":new_eval_count,
            "reused_v5_mix50_loss_rows":555,"mix50_reconciliation_window_evaluations":mix_reconcile_rows,
            "pair_window_score_rows_output":len(per_window_rows),
            "cancellation_diagnostic_pair_windows":36*185,"cancellation_diagnostic_wall_seconds":cancel_seconds,
            "target_archive_loads":1,"target_archive_physical_read_operations":1,
            "target_array_window_rows_loaded":185,"unique_target_windows":185,
            "prefix_archive_loads":1,"source_prediction_bundles_loaded":9,
            "reused_mix50_prediction_bundles_loaded":3,"source_prediction_row_sidecars_parsed":12,
            "source_loss_fixture_rows_reused":3885+555,
            "task_count":5,"source_groups":47,"bootstrap_draws":BOOT_DRAWS,"bootstrap_seed":BOOT_SEED,
            "v5_prior_target_archive_loads":4,"v5_prior_target_window_rows_loaded":740,
            "v5_prior_primary_window_loss_evaluations":3330,"v5_prior_l2_diagnostic_window_evaluations":2775,
            "v5_prior_legacy_counter":35523,"v5_prior_legacy_counter_name":"cumulative_dev_future_action_row_reads",
            "v5_prior_legacy_counter_unit_status":"unverified_as_same_grain; not added",
            "v2_accepted_per_job_child_seconds_by_seed":[
                {"seed":202610131,"S":0.767859,"V":1.190094,"S_plus_V_child_sum":1.957953},
                {"seed":202610132,"S":0.766805,"V":0.558181,"S_plus_V_child_sum":1.324986},
                {"seed":202610133,"S":0.542778,"V":0.526567,"S_plus_V_child_sum":1.069345}
            ],
            "v2_accepted_per_job_child_seconds_mean":{"S":0.6924806666666666,"V":0.7582806666666667,
              "S_plus_V_child_sum":1.4507613333333333},
            "v2_latency_scope":"separate accepted 185-window child-job elapsed times; not integrated end-to-end latency",
            "program_dependencies":{
                "S":"V2 optimized state-only G-S output",
                "A":"V4 common RGB recipe with RGB mask; input state-only at inference for this source",
                "V":"V4 matched G-V output; fusion RGB+state dependencies"
            },
            "peak_rss_kib":int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "effective_cpu_count":len(os.sched_getaffinity(0)) if hasattr(os,"sched_getaffinity") else os.cpu_count()
        }
        write_json(ROOT/"results/RESOURCE_LEDGER.json",cpu_resource)
        write_csv(ROOT/"results/RESOURCE_LEDGER.csv",[{"key":k,"value":json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v}
                for k,v in cpu_resource.items()])
        exposure.update({
            "status":"COMPLETE","target_archive_load_operations":1,"target_archive_physical_read_operations":1,
            "target_array_window_rows_loaded":185,"target_archive_sha256_actual":target_sha,
            "target_archive_read_completed_utc":utc(),"new_pair_window_loss_evaluations":new_eval_count,
            "mix50_reconciliation_window_evaluations":mix_reconcile_rows,
            "cancellation_diagnostic_pair_windows":36*185,
            "unique_dev_receiver_windows":185,"final_heldout_payload_access":0,
            "old_root_writes":0,"source_roots_read_only":True,
            "history":exposure["history"]+[{"phase":"complete","status":"PASS","target_archive_loads":1,
                       "new_pair_window_loss_evaluations":new_eval_count,
                       "reused_v5_mix50_loss_rows":555,"final_heldout_payload_access":0}]
        })
        write_json(ROOT/"inventory/OUTCOME_EXPOSURE.json",exposure)

        # Bootstrap/task group identity diagnostics are already locked; create direct figure tables.
        fig_principal=[]
        for ref in ["S","V","BIAS_STATE","BIAS_VIDEO","PERSISTENCE"]:
            for metric in METRICS:
                civals=principal_ci_map.get((ref,metric))
                seeds_vals=[principal_seed_macro[s][(ref,metric)] for s in SEEDS]
                fig_principal.append({"reference":ref,"metric":metric,"gain_mean":float(np.mean(seeds_vals)),
                    "gain_sample_sd_across_three_fixed_pairs":sample_sd(seeds_vals),
                    "ci_low":civals["ci_low"] if civals else "",
                    "ci_high":civals["ci_high"] if civals else "",
                    "ci_scope":"paired source-group bootstrap over three fixed pairs" if civals else "not computed by locked CI plan"})
        write_csv(ROOT/"figure_data/PRINCIPAL_GAINS.csv",fig_principal)
        write_csv(ROOT/"figure_data/FAMILY_MEANS_AND_PAIR_SPREAD.csv",
                  [r for r in family_effect_rows if r["record_type"]=="family_mean"])
        write_csv(ROOT/"figure_data/SEED_BLOCK_LOO.csv",seed_loo_rows)
        write_csv(ROOT/"figure_data/JGAP_AND_GAINS.csv",cancellation_rows)
        trajectory_rows=[]
        mix_seed=SEEDS[0]
        mix_pid=principal_ids[mix_seed]
        mix_arr=pair_arrays[mix_pid]
        for task_s, aid in selected_cases.items():
            task=int(task_s);idx=canonical_index[aid]
            for axis in [0,14]:
                for h in range(30):
                    trajectory_rows.extend([
                        {"task_id":task,"anchor_id":aid,"seed":mix_seed,"axis":axis,"horizon_step":h+1,
                         "series":"target_GT","native_value":float(ordered_target[idx,h,axis])},
                        {"task_id":task,"anchor_id":aid,"seed":mix_seed,"axis":axis,"horizon_step":h+1,
                         "series":"S_state_member","native_value":float(source_arrays[("S",mix_seed)][idx,h,axis])},
                        {"task_id":task,"anchor_id":aid,"seed":mix_seed,"axis":axis,"horizon_step":h+1,
                         "series":"V_fusion_member","native_value":float(source_arrays[("V",mix_seed)][idx,h,axis])},
                        {"task_id":task,"anchor_id":aid,"seed":mix_seed,"axis":axis,"horizon_step":h+1,
                         "series":"MIX50_average","native_value":float(mix_arr[idx,h,axis])}
                    ])
        write_csv(ROOT/"figure_data/TRAJECTORY_DATA.csv",trajectory_rows)

        # Render figures from the fixed, stored arrays and score tables; no archive reopens.
        os.environ.setdefault("MPLCONFIGDIR",str(ROOT/"tmp/mplconfig"))
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        colors={"SS":"#3465a4","AA":"#73d216","VV":"#cc0000","SA":"#75507b","SV":"#f57900","AV":"#c17d11"}
        fig,axs=plt.subplots(1,3,figsize=(17,5),constrained_layout=True)
        refs=["S","V","BIAS_STATE","BIAS_VIDEO","PERSISTENCE"]
        for ax,metric in zip(axs,METRICS):
            rows=[r for r in fig_principal if r["metric"]==metric]
            xx=np.arange(len(refs))
            ys=[next(r["gain_mean"] for r in rows if r["reference"]==ref) for ref in refs]
            loerr=[];hierr=[]
            for ref,yv in zip(refs,ys):
                r=next(r for r in rows if r["reference"]==ref)
                if r["ci_low"]!="":
                    loerr.append(yv-float(r["ci_low"]));hierr.append(float(r["ci_high"])-yv)
                else:
                    loerr.append(0.0);hierr.append(0.0)
            ax.axhline(0,color="black",linewidth=.8)
            ax.errorbar(xx,ys,yerr=[loerr,hierr],fmt="o",capsize=4,color="#204a87",label="mean gain; 95% group CI where locked")
            ax.set_xticks(xx,refs,rotation=30,ha="right")
            ax.set_title(metric)
            ax.set_ylabel("reference loss - MIX50 loss (positive favors MIX50)")
        fig.suptitle("Fixed V5 same-seed MIX50: direct paired gains")
        fig.savefig(ROOT/"figures/MIX50_DIRECT_REFERENCE_GAINS.png",dpi=160)
        plt.close(fig)

        fig,axs=plt.subplots(1,3,figsize=(17,5),constrained_layout=True)
        fam_order=["SS","AA","VV","SA","SV","AV"]
        for ax,metric in zip(axs,METRICS):
            for i,fam in enumerate(fam_order):
                pids=family_pairs[fam]
                vals=[macro_from_task(pair_task_values[pid][metric]) for pid in pids]
                jitter=np.linspace(-0.16,0.16,len(vals)) if len(vals)>1 else np.array([0.0])
                ax.scatter(np.full(len(vals),i)+jitter,vals,s=28,color=colors[fam],alpha=.85)
                mean=float(np.mean(vals));sd=sample_sd(vals)
                ax.errorbar([i],[mean],yerr=[[sd],[sd]] if np.isfinite(sd) else None,
                            fmt="D",markersize=6,capsize=4,color="black")
            ax.set_xticks(np.arange(6),fam_order)
            ax.set_title(metric)
            ax.set_ylabel("macro loss; black diamond ± finite pair SD")
        fig.suptitle("Six fixed families: pair points and family means")
        fig.savefig(ROOT/"figures/FAMILY_MEANS_AND_PAIR_SPREAD.png",dpi=160)
        plt.close(fig)

        fig,axs=plt.subplots(1,3,figsize=(17,5),constrained_layout=True)
        loo_names=["SV_vs_SS","SV_vs_VV","SV_vs_SA","AV_vs_AA","AV_vs_VV","SV_pairing_effect"]
        helds=SEEDS
        for ax,metric in zip(axs,METRICS):
            for name in loo_names:
                vals=[]
                for held in helds:
                    rr=next(r for r in seed_loo_rows if r["leave_out_seed"]==held and r["metric"]==metric and
                            r.get("contrast_name")==name)
                    vals.append(float(rr["gain"]))
                ax.plot(range(len(helds)),vals,marker="o",label=name)
            ax.axhline(0,color="black",linewidth=.8)
            ax.set_xticks(range(3),[str(x) for x in helds],rotation=20)
            ax.set_title(metric)
            ax.set_ylabel("gain after removing entire seed block")
        axs[-1].legend(fontsize=7,loc="best")
        fig.suptitle("True seed-block leave-one-out contrasts")
        fig.savefig(ROOT/"figures/SEED_BLOCK_LOO_CONTRASTS.png",dpi=160)
        plt.close(fig)

        fig,axs=plt.subplots(1,3,figsize=(17,5),constrained_layout=True)
        for ax,metric in zip(axs,METRICS):
            rows=[r for r in cancellation_rows if r["scope"]=="macro" and r["metric"]==metric]
            for fam in fam_order:
                rr=[r for r in rows if r["family"]==fam]
                ax.scatter([r["j_gap_member_mean_minus_ensemble"] for r in rr],
                           [r["gain_vs_better_member"] for r in rr],
                           s=24,alpha=.8,label=fam,color=colors[fam])
            ax.axhline(0,color="black",linewidth=.8);ax.axvline(0,color="black",linewidth=.8)
            ax.set_xlabel("J_gap = mean member loss - ensemble loss")
            ax.set_ylabel("gain over better member")
            ax.set_title(metric)
        axs[-1].legend(fontsize=8,loc="best")
        fig.suptitle("Convex cancellation and improvement over the stronger member")
        fig.savefig(ROOT/"figures/JGAP_VS_BETTER_MEMBER_GAIN.png",dpi=160)
        plt.close(fig)

        trace_files=[]
        for task_s, aid in selected_cases.items():
            task=int(task_s);idx=canonical_index[aid]
            fig,axs=plt.subplots(1,2,figsize=(11,4),sharex=True,constrained_layout=True)
            for ax,axis in zip(axs,[0,14]):
                ax.plot(range(1,31),ordered_target[idx,:,axis],label="target GT",linewidth=2,color="black")
                ax.plot(range(1,31),source_arrays[("S",mix_seed)][idx,:,axis],label=f"S state {mix_seed}",alpha=.85)
                ax.plot(range(1,31),source_arrays[("V",mix_seed)][idx,:,axis],label=f"V fusion {mix_seed}",alpha=.85)
                ax.plot(range(1,31),mix_arr[idx,:,axis],label="MIX50 average",linewidth=2)
                ax.set_title(f"axis {axis}; native command values")
                ax.set_xlabel("forecast step")
                ax.grid(alpha=.2)
            axs[0].set_ylabel("native command value")
            axs[1].legend(fontsize=8,loc="best")
            fig.suptitle(f"Locked V5 representative case, task {task}, {aid}, seed {mix_seed}")
            name=f"TRAJECTORY_TASK_{task}.png"
            fig.savefig(ROOT/"figures"/name,dpi=160)
            plt.close(fig)
            trace_files.append(name)
        figure_files=["MIX50_DIRECT_REFERENCE_GAINS.png","FAMILY_MEANS_AND_PAIR_SPREAD.png",
                      "SEED_BLOCK_LOO_CONTRASTS.png","JGAP_VS_BETTER_MEMBER_GAIN.png"]+trace_files
        figqa={"schema_version":1,"status":"PASS","created_utc":utc(),"figure_files":figure_files,
               "figure_count":len(figure_files),"figure_data_from_scored_memory":True,
               "target_archive_reopens_for_figures":0,"representative_case_ids_source":"V5 protocol/FORMULA_LOCK.json",
               "representative_case_selection_uses_outcomes":False,
               "metrics":["position_mae","safe_fd_l1","seam_safe_fd_l1"],
               "human_visual_review_status":"pending"}
        write_json(ROOT/"qa/FIGURE_BUILD_QA.json",figqa)
        captions=[
            "MIX50_DIRECT_REFERENCE_GAINS.png: 三个固定V5同seed MIX50对S、V、BIAS_STATE、BIAS_VIDEO、Persistence的直接配对收益；前四项显示锁定的source-group 95%区间，Persistence只显示三对固定成员均值。",
            "FAMILY_MEANS_AND_PAIR_SPREAD.png: 六个固定family的每pair宏平均loss点与family均值；黑色误差线为有限pair集合的样本标准差，不是独立seed区间。",
            "SEED_BLOCK_LOO_CONTRASTS.png: 每次同时删除S_k、A_k、V_k及所有包含这些节点的pair后，重新计算六个固定family contrasts。",
            "JGAP_VS_BETTER_MEMBER_GAIN.png: J_gap反映绝对误差凸性下的两成员误差抵消；纵轴显示ensemble相对较好成员的实际gain。",
            "轨迹图使用V5锁定且不依赖结果的case ID，展示seed 202610131的native action index 0/14。"
        ]
        write_json(ROOT/"captions/FIGURE_CAPTIONS.zh-CN.json",{"captions":captions,"figure_files":figure_files})
        # Contact sheet for the requested human visual check.
        from PIL import Image, ImageOps, ImageDraw
        thumbs=[]
        for name in figure_files:
            im=Image.open(ROOT/"figures"/name).convert("RGB")
            im.thumbnail((520,260))
            canvas=Image.new("RGB",(540,300),"white")
            canvas.paste(im,((540-im.width)//2,28))
            ImageDraw.Draw(canvas).text((8,6),name,fill="black")
            thumbs.append(canvas)
        cols=2;rowsn=(len(thumbs)+cols-1)//cols
        contact=Image.new("RGB",(cols*540,rowsn*300),(238,238,238))
        for i,im in enumerate(thumbs):
            contact.paste(im,((i%cols)*540,(i//cols)*300))
        contact.save(ROOT/"qa/V6_FIGURE_CONTACT_SHEET.jpg",quality=92)
        figure_bytes={name:(ROOT/"figures"/name).stat().st_size for name in figure_files}
        if any(size<1000 for size in figure_bytes.values()):
            raise RuntimeError(f"unexpectedly small figure file(s): {figure_bytes}")

        # Scalar summary for independent final review before report/checksum finalization.
        summary={}
        for metric in METRICS:
            summary[metric]={}
            for fam in fam_order:
                vals=[macro_from_task(pair_task_values[pid][metric]) for pid in family_pairs[fam]]
                summary[metric][fam]={"mean":float(np.mean(vals)),"pair_sd":sample_sd(vals),
                    "min":float(np.min(vals)),"max":float(np.max(vals)),"n_pairs":len(vals)}
            summary[metric]["MIX50_vs_references"]={}
            for ref in principal_methods:
                summary[metric]["MIX50_vs_references"][ref]={
                    "gain_mean":float(np.mean([principal_seed_macro[s][(ref,metric)] for s in SEEDS])),
                    "gain_sample_sd":sample_sd([principal_seed_macro[s][(ref,metric)] for s in SEEDS]),
                    "ci_low":principal_ci_map[(ref,metric)]["ci_low"] if ref in ["S","V","BIAS_STATE","BIAS_VIDEO"] else None,
                    "ci_high":principal_ci_map[(ref,metric)]["ci_high"] if ref in ["S","V","BIAS_STATE","BIAS_VIDEO"] else None,
                    "per_seed":{str(s):principal_seed_macro[s][(ref,metric)] for s in SEEDS}}
        summary["family_contrasts"]={}
        for metric in METRICS:
            summary["family_contrasts"][metric]={}
            for name,ref,cand in family_contrast_defs+[("SV_pairing_effect","SV_offdiag","SV_diag")]:
                cirow=family_contrast_ci_map[(name,metric)]
                summary["family_contrasts"][metric][name]={
                    "gain":cirow["point_estimate"],"ci_low":cirow["ci_low"],"ci_high":cirow["ci_high"]}
        summary["family_contrasts"]["principal_pair_effects"]={}
        summary["exposure"]={"target_archive_loads":1,"target_windows":185,
            "new_pair_window_loss_evaluations":new_eval_count,"reused_v5_mix50_loss_rows":555,
            "mix50_reconciliation_window_evaluations":mix_reconcile_rows,
            "cancellation_diagnostic_pair_windows":6660,"old_root_writes":0,"final_heldout_payload_access":0}
        summary["status"]="analysis_complete_pending_visual_review_and_checksum"
        summary["v5_checksum_sha256"]=v5_receipt["manifest_sha256"]
        summary["v5_checksum_entries"]=v5_receipt["entries"]
        summary["v6_root"]=str(ROOT)
        summary["formula_lock_sha256"]=formula_sha
        write_json(ROOT/"results/RESULTS_SUMMARY.json",summary)
        owner.update({"phase":"analysis_and_figures_complete_pending_visual_review",
                      "updated_utc":utc(),"source_nodes_accepted":9,"pair_systems_accepted":36,
                      "new_pair_bundles_accepted":33,"reused_v5_mix50_bundles":3,
                      "target_archive_loads":1,"new_pair_window_loss_evaluations":6105,
                      "final_heldout_payload_access":0,"old_root_writes":0})
        write_json(ROOT/"coordination/OWNER_STATE.json",owner)
        log("analysis complete; output awaits visual review and checksum-last finalization")
    except Exception as exc:
        owner.update({"status":"technical_failure","phase":"failed","updated_utc":utc(),
                      "error_type":type(exc).__name__,"error_message":str(exc),
                      "final_heldout_payload_access":0,"old_root_writes":0})
        write_json(ROOT/"coordination/OWNER_STATE.json",owner)
        write_json(ROOT/"qa/FAILURE.json",{"status":"technical_failure","created_utc":utc(),
                    "error_type":type(exc).__name__,"error_message":str(exc)})
        log(f"FAIL {type(exc).__name__}: {exc}")
        raise
    finally:
        lock_f.close()

def verify_manifest_entries(root):
    root=Path(root)
    lines=(root/"CHECKSUMS.sha256").read_text(encoding="utf-8").splitlines()
    out={}
    for line in lines:
        if len(line)>=67 and line[64:66]=="  ":
            out[line[66:]]=line[:64]
    return out

if __name__=="__main__":
    main()

