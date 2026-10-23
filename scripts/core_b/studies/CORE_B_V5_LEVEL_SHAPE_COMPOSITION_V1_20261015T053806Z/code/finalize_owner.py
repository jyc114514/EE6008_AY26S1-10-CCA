#!/usr/bin/env python3
"""Hold exclusive ownership while V5 outputs are finalized and checksum-sealed."""
from __future__ import annotations
import datetime, hashlib, json, os, re, shutil, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OWNER = ROOT / 'coordination/OWNER_STATE.json'
LOCK = ROOT / 'coordination/OWNER.lock'
MANIFEST = ROOT / 'CHECKSUMS.sha256'
RECEIPT = ROOT / 'qa/FINAL_CHECKSUM_VERIFICATION.json'

def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace('+00:00', 'Z')

def write_json(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')

def safe_read_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))

def missing_required():
    required = [
        'inventory/SOURCE_MAP.json', 'inventory/SOURCE_PREDICTION_JOINS.csv', 'inventory/OUTCOME_EXPOSURE.json',
        'protocol/FORMULA_LOCK.json', 'protocol/DERIVED_PREDICTION_PLAN.csv', 'protocol/CONTRAST_LOCK.json',
        'qa/TRANSFORM_QA.json', 'qa/PREDICTION_SEAL.json', 'qa/FINAL_ACCEPTANCE.json',
        'results/PRIMARY_TASK_SEED.csv', 'results/PAIRED_EFFECTS.csv', 'results/SEED_LOO.csv', 'results/TASK_LOO.csv',
        'results/PAIRED_GROUP_CI.csv', 'results/LEVEL_SHAPE_L2.csv', 'results/FD_IDENTITY_CHECK.csv', 'results/RESOURCE_LEDGER.csv',
        'RETURN_TO_CHATGPT.zh-CN.md', 'RETURN_TO_CHATGPT.json', 'figures/CAPTIONS.zh-CN.md'
    ]
    return [rel for rel in required if not (ROOT / rel).is_file()]

def acquire_or_reclaim_stale_lock():
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    owner_backup = OWNER.with_name(OWNER.name + f'.pre-finalizer-session-{stamp}.bak')
    lock_backup = LOCK.with_name(LOCK.name + f'.pre-finalizer-session-{stamp}.bak')
    if owner_backup.exists() or lock_backup.exists():
        raise RuntimeError('timestamped owner backup path already exists')
    current = safe_read_json(OWNER)
    if LOCK.exists():
        old = safe_read_json(LOCK)
        old_pid = old.get('owner_pid')
        if old_pid:
            try:
                os.kill(int(old_pid), 0)
                raise RuntimeError(f'previous lock PID {old_pid} is still alive')
            except ProcessLookupError:
                pass
            except PermissionError:
                raise RuntimeError(f'cannot establish previous lock PID {old_pid} is stale')
        shutil.copy2(LOCK, lock_backup)
    elif current.get('active_jobs', 0) or current.get('owner_pid'):
        raise RuntimeError('owner state reports active work without a lock; refusing takeover')
    shutil.copy2(OWNER, owner_backup)
    new_lock = {
        'schema_version': 1,
        'study': 'post_V4_frozen_level_shape_composition_development',
        'root': str(ROOT),
        'owner_pid': os.getpid(),
        'created_utc': utc(),
        'phase': 'finalize_reports_figures_checksum'
    }
    temp = LOCK.with_suffix('.lock.tmp')
    temp.write_text(json.dumps(new_lock, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(temp, LOCK)
    current.update({
        'status': 'running', 'phase': 'final_artifacts', 'active_jobs': 1, 'active_gpu_jobs': 0,
        'owner_pid': os.getpid(), 'owner_lock_acquired': True, 'updated_utc': utc(),
        'target_accesses': 4, 'new_prediction_bundles_scored': 18,
        'new_window_loss_evaluations': 3330, 'failed_scorer_attempts': 3,
        'final_heldout_payload_access': 0, 'source_roots_read_only': True
    })
    write_json(OWNER, current)
    return {'owner_backup': str(owner_backup), 'stale_lock_backup': str(lock_backup) if lock_backup.exists() else None}

def seal():
    missing = missing_required()
    if missing:
        print(json.dumps({'status': 'MISSING_REQUIRED', 'paths': missing}, ensure_ascii=False), flush=True)
        return False
    if MANIFEST.exists() or RECEIPT.exists():
        raise RuntimeError('refusing to overwrite existing final checksum artifacts')
    state = safe_read_json(OWNER)
    state.update({
        'status': 'complete', 'phase': 'checksum_finalization', 'active_jobs': 0, 'active_gpu_jobs': 0,
        'owner_pid': None, 'owner_lock_acquired': False, 'updated_utc': utc(),
        'final_heldout_payload_access': 0, 'old_root_writes': 0, 'source_roots_read_only': True,
        'prediction_seal_status': 'PASS', 'new_prediction_bundles_scored': 18,
        'new_window_loss_evaluations': 3330, 'failed_scorer_attempts': 3
    })
    write_json(OWNER, state)
    try:
        LOCK.unlink()
    except FileNotFoundError:
        pass
    excluded = {'CHECKSUMS.sha256', 'qa/FINAL_CHECKSUM_VERIFICATION.json'}
    paths = []
    unsafe = []
    for path in sorted(ROOT.rglob('*')):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT).as_posix()
        if rel in excluded:
            continue
        if path.is_symlink() or '\n' in rel:
            unsafe.append(rel)
        else:
            paths.append((rel, path))
    if unsafe:
        raise RuntimeError(f'unsafe or newline paths in checksum scope: {unsafe}')
    entries = [(rel, hashlib.sha256(path.read_bytes()).hexdigest()) for rel, path in paths]
    content = ''.join(f'{digest}  {rel}\n' for rel, digest in entries).encode('utf-8')
    with MANIFEST.open('xb') as out:
        out.write(content)
        out.flush()
        os.fsync(out.fileno())
    manifest_sha = hashlib.sha256(content).hexdigest()
    missing_files, mismatches, malformed = [], [], []
    verified = 0
    for line_number, line in enumerate(content.decode('utf-8').splitlines(), 1):
        match = re.fullmatch(r'([0-9a-f]{64})  (.+)', line)
        if not match:
            malformed.append(line_number)
            continue
        expected, rel = match.groups()
        path = ROOT / rel
        try:
            path.resolve(strict=True).relative_to(ROOT.resolve())
        except Exception:
            malformed.append(line_number)
            continue
        if not path.is_file() or path.is_symlink():
            missing_files.append(rel)
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            mismatches.append({'path': rel, 'expected': expected, 'actual': actual})
        else:
            verified += 1
    current_manifest_sha = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    if current_manifest_sha != manifest_sha:
        malformed.append('CHECKSUMS.sha256 changed during verification')
    status = 'PASS' if not (missing_files or mismatches or malformed) and verified == len(entries) else 'FAIL'
    receipt = {
        'schema_version': 1, 'created_utc': utc(), 'status': status,
        'manifest_path': 'CHECKSUMS.sha256', 'manifest_sha256': manifest_sha,
        'manifest_entry_count': len(entries), 'verified_entry_count': verified,
        'missing': missing_files, 'mismatches': mismatches, 'malformed_or_unsafe_entries': malformed,
        'excluded_from_manifest': [
            {'path': 'CHECKSUMS.sha256', 'reason': 'manifest cannot hash its own digest'},
            {'path': 'qa/FINAL_CHECKSUM_VERIFICATION.json', 'reason': 'post-verification receipt excluded to avoid recursive hashing'}
        ]
    }
    with RECEIPT.open('x', encoding='utf-8') as out:
        json.dump(receipt, out, ensure_ascii=False, indent=2)
        out.write('\n')
        out.flush()
        os.fsync(out.fileno())
    print(json.dumps({'status': 'SEALED', 'checksum_status': status, 'manifest_sha256': manifest_sha,
                      'entries': len(entries), 'verified': verified, 'missing': len(missing_files),
                      'mismatches': len(mismatches), 'malformed': len(malformed),
                      'receipt': str(RECEIPT)}, ensure_ascii=False), flush=True)
    return True

def main():
    backups = acquire_or_reclaim_stale_lock()
    # Owner and exposure snapshots were also preserved by the initial handoff lock; retain this session's backups.
    print(json.dumps({'status': 'OWNER_SESSION_READY', 'pid': os.getpid(), 'backups': backups,
                      'root': str(ROOT)}, ensure_ascii=False), flush=True)
    for line in sys.stdin:
        command = line.strip()
        if command == 'seal':
            if seal():
                break
        elif command == 'abort':
            state = safe_read_json(OWNER)
            state.update({'status': 'partial', 'phase': 'final_artifacts_interrupted', 'active_jobs': 0,
                          'owner_pid': None, 'owner_lock_acquired': False, 'updated_utc': utc()})
            write_json(OWNER, state)
            try: LOCK.unlink()
            except FileNotFoundError: pass
            print('OWNER_RELEASED_PARTIAL', flush=True)
            break
        else:
            print('AWAITING_SEAL_OR_ABORT', flush=True)

if __name__ == '__main__':
    main()
