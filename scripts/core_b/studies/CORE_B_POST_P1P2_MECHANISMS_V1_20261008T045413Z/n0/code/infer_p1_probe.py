#!/usr/bin/env python3
"""Target-blind N0 replay/ZERO/PERM inference from fixed accepted P1 checkpoints."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
N0_ROOT = HERE.parent
NEXT_ROOT = N0_ROOT.parent
PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
P1_ROOT = PROJECT / 'reports/benchmarks/CORE_B_P1P2_VISUAL_COMPLEMENTARITY_V1_20261007T053432Z'
V3_ROOT = PROJECT / 'repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z'
P1_VENDOR = HERE / 'vendor/p1'
if str(P1_VENDOR) not in sys.path:
    sys.path.insert(0, str(P1_VENDOR))
from model import parameter_count as alt_parameter_count, state_hash as alt_state_hash
from experiment_contracts import make_matched_alt_predictor
from model_v3_reference import V3Full28Predictor, parameter_count as vjepa_parameter_count, state_hash as vjepa_state_hash

MODEL_CHOICES = ('F-POOL-D', 'ALT-POOL-R3D18')
CONDITION_CHOICES = ('REAL', 'ZERO', 'PERM')
N0_SEEDS = (20260950, 20260951, 20260952)
HORIZON = 30
VIDEO_STEPS = 16
ACTION_DIM = 28
INPUT_KEYS = (
    'anchor_id', 'episode_index', 'original_task_index', 'split_code',
    'anchor_frame', 'anchor_timestamp', 'history_standardized', 'history_mask',
    'history_relative_delta_standardized', 'anchor_command_native',
)


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def hash_arrays(items: list[tuple[str, np.ndarray]]) -> str:
    digest = hashlib.sha256()
    for name, value in items:
        arr = np.ascontiguousarray(value)
        digest.update(name.encode('utf-8') + b'\0')
        digest.update(arr.dtype.str.encode('ascii') + b'\0')
        digest.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
        digest.update(arr.tobytes())
    return digest.hexdigest()


def state_hash(model: torch.nn.Module, fn) -> str:
    return fn(model)


def load_target_blind_dev() -> tuple[dict[str, np.ndarray], np.ndarray, dict[int, str]]:
    source = V3_ROOT / 'data/V3_MAIN_INPUTS.npz'
    with np.load(source, allow_pickle=False) as archive:
        present = set(archive.files)
        if not set(INPUT_KEYS).issubset(present):
            raise RuntimeError(f'input schema missing {sorted(set(INPUT_KEYS) - present)}')
        if {'targets_native', 'targets_native_60', 'future_targets'} & set(INPUT_KEYS):
            raise AssertionError('forbidden target key requested by N0 inference')
        data = {key: archive[key].copy() for key in INPUT_KEYS}
    split = np.asarray(data['split_code'])
    if set(np.unique(split).tolist()) != {0, 1}:
        raise RuntimeError('accepted primary input contains a split outside train/dev')
    rows = np.flatnonzero(split == 1).astype(np.int64)
    if len(rows) != 2322 or len(np.unique(data['anchor_id'][rows])) != len(rows):
        raise RuntimeError('P1 primary dev ID population mismatch')
    groups: dict[int, str] = {}
    with (P1_ROOT / 'inventory/SOURCE_GROUP_MAP.csv').open(newline='', encoding='utf-8') as handle:
        for record in csv.DictReader(handle):
            if record['split'] == 'dev':
                groups[int(record['episode_index'])] = record['source_group_id']
    if len(groups) != 603 or any(int(data['episode_index'][row]) not in groups for row in rows):
        raise RuntimeError('source-group map does not cover exact P1 dev episodes')
    return data, rows, groups


def target_normalizer_identity_sha256() -> str:
    """Return the locked V3 target-normalizer hash from this task's accepted identity table."""
    identity = 'V3:data/V3_NORMALIZERS.json'
    with (NEXT_ROOT / 'inventory/INPUT_IDENTITIES.csv').open(newline='', encoding='utf-8') as handle:
        matches = [row for row in csv.DictReader(handle) if row['identity'] == identity]
    if len(matches) != 1:
        raise RuntimeError(f'expected one locked normalizer identity row, found {len(matches)}')
    path = V3_ROOT / 'data/V3_NORMALIZERS.json'
    if Path(matches[0]['path']).resolve() != path.resolve():
        raise RuntimeError('locked target-normalizer path does not match the inference source')
    actual = sha_file(path)
    if actual != matches[0]['sha256']:
        raise RuntimeError('V3 target-normalizer hash differs from the accepted N0 input identity')
    return actual


def load_permutation() -> dict[int, int]:
    path = NEXT_ROOT / 'protocol/PAIR_PERMUTATION.csv'
    mapping: dict[int, int] = {}
    recipients: set[str] = set()
    donors: set[str] = set()
    with path.open(newline='', encoding='utf-8') as handle:
        for record in csv.DictReader(handle):
            row = int(record['recipient_input_row'])
            if row in mapping:
                raise RuntimeError(f'duplicate recipient input row {row}')
            mapping[row] = int(record['donor_input_row'])
            recipients.add(record['recipient_anchor_id'])
            donors.add(record['donor_anchor_id'])
            if record['recipient_source_group_id'] == record['donor_source_group_id']:
                raise RuntimeError(f'within-source-group donor at {record["recipient_anchor_id"]}')
    if len(mapping) != 2322 or len(donors) != 2322 or recipients != donors:
        raise RuntimeError('permutation is not a 2322-row bijection')
    return mapping


def load_model(model_name: str, seed: int, checkpoint_path: Path, device: torch.device):
    if model_name == 'F-POOL-D':
        model = V3Full28Predictor(horizon=HORIZON, seed=seed)
        count = vjepa_parameter_count
        state_hash_fn = vjepa_state_hash
    elif model_name == 'ALT-POOL-R3D18':
        model, _ = make_matched_alt_predictor(HORIZON, seed)
        count = alt_parameter_count
        state_hash_fn = alt_state_hash
    else:
        raise ValueError(model_name)
    # The trusted accepted V3 training checkpoint contains optimizer/RNG state; the original V3 infer_v3.py loads it with weights_only=False. ALT uses a weights-only export. run() SHA-checks the sealed path before this function.
    payload = torch.load(checkpoint_path, map_location='cpu', weights_only=(model_name == 'ALT-POOL-R3D18'))
    state = payload.get('model_state_dict')
    if not isinstance(state, dict):
        raise RuntimeError('fixed P1 checkpoint lacks model_state_dict')
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, count, state_hash_fn


def load_model_inputs(model_name: str, input_rows: np.ndarray, donor_rows: np.ndarray,
                      condition: str, device: torch.device) -> torch.Tensor:
    if condition not in CONDITION_CHOICES:
        raise ValueError(condition)
    if model_name == 'F-POOL-D':
        feature_path = V3_ROOT / 'cache/V3_TEMPORAL_FEATURES.npy'
        feature_cache = np.load(feature_path, mmap_mode='r', allow_pickle=False)
        raw_rows = np.asarray(feature_cache[input_rows if condition != 'PERM' else donor_rows, 0], dtype=np.float32)
        video_norm = json.loads((V3_ROOT / 'data/V3_VIDEO_NORMALIZER.json').read_text(encoding='utf-8'))
        mean = torch.as_tensor(video_norm['mean'], dtype=torch.float32, device=device)
        std = torch.as_tensor(video_norm['safe_std'], dtype=torch.float32, device=device)
        video = (torch.from_numpy(raw_rows).to(device) - mean) / std
        # This is the accepted F-POOL-D operation immediately before video_projection.
        video = video.mean(dim=1, keepdim=True).expand(-1, VIDEO_STEPS, -1)
    else:
        union = np.load(P1_ROOT / 'cache/R3D_FEATURES_UNION.npy', mmap_mode='r', allow_pickle=False)
        primary_index = np.load(P1_ROOT / 'cache/R3D_PRIMARY_INDEX.npy', allow_pickle=False)
        indices = primary_index[input_rows if condition != 'PERM' else donor_rows]
        raw = np.asarray(union[indices], dtype=np.float32)
        normalizer = json.loads((P1_ROOT / 'cache/R3D_NORMALIZER.json').read_text(encoding='utf-8'))
        mean = np.asarray(normalizer['mean'], dtype=np.float32)
        std = np.asarray(normalizer['safe_std'], dtype=np.float32)
        normalized = (raw - mean) / std
        repeated = np.broadcast_to(normalized[:, None, :], (len(input_rows), VIDEO_STEPS, 512)).copy()
        video = torch.from_numpy(repeated).to(device)
        # P1 ALT inference mean-pools the already repeated normalized 512D row.
        video = video.mean(dim=1, keepdim=True).expand(-1, VIDEO_STEPS, -1)
    if condition == 'ZERO':
        # New tensor: never mutate cached/mmap data or a shared batch input in-place.
        video = torch.zeros_like(video)
    if video.shape != (len(input_rows), VIDEO_STEPS, 768 if model_name == 'F-POOL-D' else 512):
        raise RuntimeError(f'video node shape mismatch: {tuple(video.shape)}')
    if not torch.isfinite(video).all():
        raise RuntimeError('nonfinite normalized video node')
    return video


def reference_predictions(path: Path, anchor_ids: np.ndarray, allow_subset: bool = False) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        if 'predictions_native' not in archive.files or 'anchor_id' not in archive.files:
            raise RuntimeError('accepted reference prediction schema missing anchor_id/predictions_native')
        ref_ids = np.asarray(archive['anchor_id']).astype(str)
        ref = np.asarray(archive['predictions_native'], dtype=np.float32)
    lookup = {value: index for index, value in enumerate(ref_ids)}
    requested = anchor_ids.astype(str).tolist()
    if len(lookup) != len(ref_ids) or (not allow_subset and set(lookup) != set(requested)) or (allow_subset and not set(requested).issubset(set(lookup))):
        raise RuntimeError('accepted reference IDs differ from requested P1 primary dev IDs')
    return ref[np.asarray([lookup[value] for value in requested], dtype=np.int64)]


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.seed not in N0_SEEDS:
        raise ValueError(f'unlocked seed {args.seed}')
    if args.mode not in ('full', 'canary'):
        raise ValueError(args.mode)
    target_normalizer_sha256 = target_normalizer_identity_sha256()
    if args.device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable after watcher host launch')
        torch.cuda.set_device(0)
        device = torch.device('cuda:0')
        physical = int(os.environ['GPU_WATCHER_PHYSICAL_GPU'])
        torch.cuda.reset_peak_memory_stats(device)
    else:
        device = torch.device('cpu')
        physical = None
    # Match the accepted inference entrypoints: eval/inference_mode; leave backend defaults unchanged.
    data, dev_rows, source_groups = load_target_blind_dev()
    if args.mode == 'canary':
        dev_rows = dev_rows[:min(args.canary_rows, len(dev_rows))]
    permutation = load_permutation()
    donors = np.asarray([permutation[int(row)] for row in dev_rows], dtype=np.int64)
    recipient_ids = np.asarray(data['anchor_id'][dev_rows]).astype(str)
    donor_ids = np.asarray(data['anchor_id'][donors]).astype(str)
    donor_groups = np.asarray([source_groups[int(data['episode_index'][row])] for row in donors], dtype='U64')
    recipient_groups = np.asarray([source_groups[int(data['episode_index'][row])] for row in dev_rows], dtype='U64')
    if args.condition == 'PERM' and np.any(donor_groups == recipient_groups):
        raise RuntimeError('PERM contains a same-source-group donor')
    non_video_hash = hash_arrays([
        ('history_standardized', data['history_standardized'][dev_rows]),
        ('history_mask', data['history_mask'][dev_rows]),
        ('history_relative_delta_standardized', data['history_relative_delta_standardized'][dev_rows]),
        ('anchor_command_native', data['anchor_command_native'][dev_rows]),
    ])
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file() or sha_file(checkpoint) != args.checkpoint_sha256:
        raise RuntimeError('fixed checkpoint path/hash mismatch')
    model, count, state_hash_fn = load_model(args.model, args.seed, checkpoint, device)
    param_hash_before = state_hash(model, state_hash_fn)
    model_state_dict = model.state_dict()
    buffer_hash_before = hash_arrays([(name, val.detach().cpu().numpy()) for name, val in model.named_buffers()])
    if count(model) not in (1_155_356, 1_122_588):
        raise RuntimeError(f'unexpected P1 predictor parameter count {count(model)}')
    ref = reference_predictions(Path(args.reference), recipient_ids, allow_subset=(args.mode == 'canary')) if args.reference else None
    if ref is None:
        raise RuntimeError('every probe bundle must bind an accepted REAL reference bundle')
    target_norm = json.loads((V3_ROOT / 'data/V3_NORMALIZERS.json').read_text(encoding='utf-8'))
    sigma = torch.as_tensor(target_norm['target_sigma'], dtype=torch.float32, device=device).view(1, 1, ACTION_DIM)
    node_digest = hashlib.sha256()
    predictions = []
    hook_count = 0
    def capture_video_projection_input(_module, inputs):
        nonlocal hook_count
        value = inputs[0].detach().contiguous().cpu().numpy()
        node_digest.update(value.dtype.str.encode('ascii') + b'\0')
        node_digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        node_digest.update(value.tobytes())
        hook_count += 1
    hook = model.video_projection.register_forward_pre_hook(capture_video_projection_input)
    started = time.monotonic()
    try:
        with torch.inference_mode():
            for start in range(0, len(dev_rows), args.batch_size):
                rows = dev_rows[start:start + args.batch_size]
                donor_batch = donors[start:start + args.batch_size]
                state = torch.from_numpy(data['history_standardized'][rows]).to(device)
                mask = torch.from_numpy(data['history_mask'][rows]).to(device)
                relative = torch.from_numpy(data['history_relative_delta_standardized'][rows]).to(device)
                anchor = torch.from_numpy(data['anchor_command_native'][rows]).to(device)
                video = load_model_inputs(args.model, rows, donor_batch, args.condition, device)
                residual = model(state, mask, relative, video, anchor, use_state=True, use_video=True)
                pred = (anchor.unsqueeze(1) + residual * sigma).float().cpu().numpy()
                predictions.append(pred)
    finally:
        hook.remove()
    prediction = np.concatenate(predictions, axis=0).astype(np.float32, copy=False)
    elapsed = time.monotonic() - started
    if prediction.shape != (len(dev_rows), HORIZON, ACTION_DIM) or not np.isfinite(prediction).all():
        raise RuntimeError(f'prediction schema/finite failure {prediction.shape}')
    if any(p.grad is not None for p in model.parameters()):
        raise RuntimeError('inference populated parameter gradients')
    param_hash_after = state_hash(model, state_hash_fn)
    buffer_hash_after = hash_arrays([(name, val.detach().cpu().numpy()) for name, val in model.named_buffers()])
    if param_hash_before != param_hash_after or buffer_hash_before != buffer_hash_after:
        raise RuntimeError('model parameter or persistent buffer changed during N0 inference')
    replay = None
    replay_exact = None
    replay_max_abs = None
    if args.condition == 'REAL':
        replay_exact = bool(np.array_equal(prediction, ref))
        replay_max_abs = float(np.max(np.abs(prediction.astype(np.float64) - ref.astype(np.float64))))
    output_path = Path(args.output).resolve()
    if not output_path.is_relative_to(N0_ROOT.resolve()):
        raise RuntimeError('N0 output must stay under NEXT_ROOT/n0')
    if args.mode == 'full':
        if output_path.exists():
            raise FileExistsError(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'source_id': recipient_ids.astype('U64'), 'anchor_id': recipient_ids.astype('U64'),
            'episode_index': data['episode_index'][dev_rows].astype(np.int64),
            'source_group_id': recipient_groups, 'anchor_frame': data['anchor_frame'][dev_rows].astype(np.int64),
            'anchor_timestamp': data['anchor_timestamp'][dev_rows].astype(np.float32),
            'predictions_native': prediction,
        }
        if args.condition == 'PERM':
            payload['donor_anchor_id'] = donor_ids.astype('U64')
            payload['donor_episode_index'] = data['episode_index'][donors].astype(np.int64)
            payload['donor_source_group_id'] = donor_groups
        np.savez_compressed(output_path, **payload)
    locked_inputs = json.loads((P1_ROOT / 'protocol/LOCKED_INPUT_IDENTITIES.json').read_text(encoding='utf-8'))
    vendor_manifest = json.loads((P1_VENDOR / 'P1_SOURCE_MANIFEST.json').read_text(encoding='utf-8'))
    sidecar = {
        'status': 'PASS_STRICT_REPLAY' if args.condition == 'REAL' and replay_exact else ('FAIL_STRICT_REPLAY' if args.condition == 'REAL' else 'PREDICTIONS_WRITTEN_UNSCORED'),
        'module': 'N0','model':args.model,'seed':args.seed,'condition':args.condition,'mode':args.mode,
        'rows':len(dev_rows),'prediction_shape':list(prediction.shape),'prediction_sha256':hashlib.sha256(np.ascontiguousarray(prediction).tobytes()).hexdigest(),
        'bundle_path':str(output_path) if args.mode=='full' else None,'bundle_sha256':sha_file(output_path) if args.mode=='full' else None,
        'accepted_reference_path':str(Path(args.reference).resolve()) if args.reference else None,
        'accepted_reference_bundle_sha256':sha_file(Path(args.reference)) if args.reference else None,
        'strict_replay_exact':replay_exact,'strict_replay_max_abs':replay_max_abs,
        'checkpoint_path':str(checkpoint),'checkpoint_sha256':sha_file(checkpoint),
        'p1_video_projection_input_sha256':node_digest.hexdigest(),'p1_video_projection_calls':hook_count,
        'non_video_input_sha256':non_video_hash,'parameter_state_sha256_before':param_hash_before,'parameter_state_sha256_after':param_hash_after,
        'buffer_state_sha256_before':buffer_hash_before,'buffer_state_sha256_after':buffer_hash_after,
        'encoder_updated':False,'optimizer_steps':0,'new_encoder_extraction':False,'target_arrays_opened':False,'heldout_access':0,'final_test_access':0,
        'physical_gpu':physical,'logical_device':str(device),'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
        'elapsed_seconds':elapsed,'peak_memory_allocated_mib':int(torch.cuda.max_memory_allocated(device)//(1024*1024)) if device.type=='cuda' else 0,
        'peak_memory_reserved_mib':int(torch.cuda.max_memory_reserved(device)//(1024*1024)) if device.type=='cuda' else 0,
        'permutation_sha256':sha_file(NEXT_ROOT/'protocol/PAIR_PERMUTATION.csv'),
        'canonical_p1_input_sha256':locked_inputs['v3_input_sha256'],
        'v3_feature_cache_sha256':locked_inputs['v3_cache_sha256'],
        'r3d_feature_cache_sha256':sha_file(P1_ROOT/'cache/R3D_FEATURES_UNION.npy'),
        'r3d_primary_index_sha256':sha_file(P1_ROOT/'cache/R3D_PRIMARY_INDEX.npy'),
        'r3d_normalizer_sha256':sha_file(P1_ROOT/'cache/R3D_NORMALIZER.json'),
        'v3_video_normalizer_sha256':sha_file(V3_ROOT/'data/V3_VIDEO_NORMALIZER.json'),
        'v3_target_normalizer_sha256':target_normalizer_sha256,
        'p1_vendor_source_manifest_sha256':sha_file(P1_VENDOR/'P1_SOURCE_MANIFEST.json'),
        'inference_code_sha256':sha_file(Path(__file__)),
        'anchor_ids_sha256':hash_arrays([('anchor_id', data['anchor_id'][dev_rows])]),
        'permission_note':'Only accepted train/dev IDs/features and fixed P1 checkpoints are read; no target/future input keys.'
    }
    sidecar_path = output_path.with_suffix('.json') if args.mode=='full' else output_path
    if sidecar_path.exists():
        raise FileExistsError(sidecar_path)
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps(sidecar, sort_keys=True))
    if args.condition == 'REAL' and replay_exact is not True:
        return 2
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=MODEL_CHOICES, required=True)
    parser.add_argument('--seed', type=int, choices=N0_SEEDS, required=True)
    parser.add_argument('--condition', choices=CONDITION_CHOICES, required=True)
    parser.add_argument('--mode', choices=('full', 'canary'), default='full')
    parser.add_argument('--canary-rows', type=int, default=128)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--checkpoint-sha256', required=True)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--batch-size', type=int, default=64)
    args = parser.parse_args()
    if args.reference is None:
        parser.error('every N0 bundle requires --reference accepted REAL bundle')
    if args.device == 'cuda' and not os.environ.get('GPU_WATCHER_PHYSICAL_GPU'):
        raise RuntimeError('CUDA inference must be launched through the task watcher')
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
