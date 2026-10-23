from __future__ import annotations
import os
import csv
import json
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
NEXT_ROOT = ROOT.parent
PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
P1 = PROJECT / 'reports/benchmarks/CORE_B_P1P2_VISUAL_COMPLEMENTARITY_V1_20261007T053432Z'
VENDOR = ROOT / 'code/vendor/p1'
sys.path.insert(0, str(VENDOR))
sys.path.insert(0, str(ROOT / 'code'))


def test_fixed_permutation_is_cross_group_bijection():
    rows = list(csv.DictReader((NEXT_ROOT / 'protocol/PAIR_PERMUTATION.csv').open()))
    assert len(rows) == 2322
    assert len({r['recipient_anchor_id'] for r in rows}) == 2322
    assert len({r['donor_anchor_id'] for r in rows}) == 2322
    assert {r['recipient_anchor_id'] for r in rows} == {r['donor_anchor_id'] for r in rows}
    assert all(r['recipient_source_group_id'] != r['donor_source_group_id'] for r in rows)
    lock = json.loads((NEXT_ROOT / 'protocol/PERMUTATION_HASH.json').read_text())
    assert lock['status'] == 'PASS_PRELOCKED_BIJECTION'
    assert lock['permutation_seed'] == 20261008
    assert lock['condition_2m_le_N'] is True
    assert lock['no_target_or_score_inputs_used'] is True


def test_video_zero_preserves_projection_bias_and_slot_encodings():
    from model_v3_reference import V3Full28Predictor, sinusoidal
    model = V3Full28Predictor(horizon=30, seed=20260950).eval()
    zero = torch.zeros((2, 16, 768), dtype=torch.float32)
    batch = torch.arange(2)
    slots = torch.linspace(-31.0, 0.0, 16).view(1, -1).expand(2, -1)
    projected = model.video_projection(zero) + sinusoidal(slots, 128) + model.video_modality
    expected = model.video_projection.bias.view(1, 1, -1) + sinusoidal(slots, 128) + model.video_modality
    assert torch.equal(projected, expected)
    assert torch.count_nonzero(projected).item() > 0
    assert model.video_projection.in_features == 768


def test_alt_zero_content_uses_512d_projection():
    from experiment_contracts import make_matched_alt_predictor
    model, _ = make_matched_alt_predictor(30, 20260950)
    zero = torch.zeros((1, 16, 512), dtype=torch.float32)
    assert model.video_projection.in_features == 512
    projected = model.video_projection(zero)
    assert projected.shape == (1, 16, 128)
    assert torch.equal(projected, model.video_projection.bias.view(1, 1, -1).expand_as(projected))


def test_inference_source_never_indexes_target_arrays():
    import ast
    source = (ROOT / 'code/infer_p1_probe.py').read_text()
    tree = ast.parse(source)
    target_subscripts = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
            if node.slice.value in {'targets_native', 'targets_native_60', 'future_targets'}:
                target_subscripts.append(node.slice.value)
    assert target_subscripts == []
    assert "'targets_native'" not in ''.join(
        line for line in source.splitlines() if 'INPUT_KEYS =' in line or line.lstrip().startswith("'")
    )


def test_identity_permutation_reproduces_real_model_output():
    import infer_p1_probe as infer
    data, dev_rows, _groups = infer.load_target_blind_dev()
    rows = dev_rows[:2]
    for model_name in ('F-POOL-D', 'ALT-POOL-R3D18'):
        entry = next(r for r in csv.DictReader((NEXT_ROOT / 'inventory/MODEL_REUSE_REGISTRY.csv').open())
                     if r['module'] == 'N0' and r['model'] == model_name and int(r['seed']) == 20260950)
        model, _count, _state_hash = infer.load_model(model_name, 20260950, Path(entry['checkpoint_path']), torch.device('cpu'))
        video_real = infer.load_model_inputs(model_name, rows, rows, 'REAL', torch.device('cpu'))
        video_identity_perm = infer.load_model_inputs(model_name, rows, rows, 'PERM', torch.device('cpu'))
        assert torch.equal(video_real, video_identity_perm)
        state = torch.from_numpy(data['history_standardized'][rows])
        mask = torch.from_numpy(data['history_mask'][rows])
        rel = torch.from_numpy(data['history_relative_delta_standardized'][rows])
        anchor = torch.from_numpy(data['anchor_command_native'][rows])
        with torch.inference_mode():
            a = model(state, mask, rel, video_real, anchor, use_state=True, use_video=True)
            b = model(state, mask, rel, video_identity_perm, anchor, use_state=True, use_video=True)
        assert torch.equal(a, b)


def test_saved_permutation_matches_reproducible_builder():
    import build_permutation
    meta = build_permutation.validate_existing(NEXT_ROOT / 'protocol/PAIR_PERMUTATION.csv')
    assert meta['status'] == 'PASS_PRELOCKED_BIJECTION'
    assert meta['pair_permutation_sha256'] == json.loads((NEXT_ROOT / 'protocol/PERMUTATION_HASH.json').read_text())['pair_permutation_sha256']


def test_real_p1_bundle_ids_match_canonical_primary_dev_ids():
    with np.load(PROJECT / 'repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z/data/V3_MAIN_INPUTS.npz', allow_pickle=False) as z:
        ids = z['anchor_id'][z['split_code'] == 1].astype(str)
    manifest = list(csv.DictReader((P1 / 'predictions/PRIMARY_MANIFEST.csv').open()))
    for model in ('F-POOL-D', 'ALT-POOL-R3D18'):
        for seed in (20260950, 20260951, 20260952):
            row = next(x for x in manifest if x['condition'] == model and int(x['seed']) == seed)
            path = Path(row['bundle_path'])
            if not path.is_absolute():
                path = P1 / path
            with np.load(path, allow_pickle=False) as bundle:
                got = bundle['anchor_id'].astype(str)
                pred_shape = bundle['predictions_native'].shape
            assert np.array_equal(got, ids)
            assert pred_shape == (2322, 30, 28)


def test_separate_scorer_gates_target_opening_on_prediction_acceptance():
    source = (ROOT / 'code/score_n0.py').read_text()
    acceptance_guard = source.index("if acceptance.get('status')")
    target_index = source.index("source['targets_native']")
    assert acceptance_guard < target_index
    assert "PASS_ALL_6_REAL_REPLAYS_AND_12_INTERVENTIONS" in source
    assert "PARTIAL_N0_ACCEPTANCE" in (ROOT / 'code/accept_n0_predictions.py').read_text()


def test_n0_acceptance_preserves_partial_branches():
    source = (ROOT / 'code/accept_n0_predictions.py').read_text()
    assert 'PARTIAL_N0_ACCEPTANCE' in source
    assert "branch['real_replay']='blocked_or_failed'" in source
    assert 'expected_but_not_accepted' in source


def test_n0_scorer_checks_prediction_acceptance_before_dev_targets():
    source = (ROOT / 'code/score_n0.py').read_text()
    gate = source.index("if acceptance.get('status') not in")
    target = source.index("source['targets_native']")
    assert gate < target


def test_n0_figure_generator_checks_score_gate_before_dev_targets():
    source = (ROOT / 'code/generate_n0_figures.py').read_text()
    acceptance_gate = source.index("if accepted.get('status') not in")
    score_gate = source.index("if score.get('status') not in")
    target = source.index("z['targets_native']")
    assert acceptance_gate < target and score_gate < target


def test_target_normalizer_hash_uses_exact_task_input_identity():
    import infer_p1_probe as infer
    rows = list(csv.DictReader((NEXT_ROOT / 'inventory/INPUT_IDENTITIES.csv').open(newline='', encoding='utf-8')))
    expected = next(r['sha256'] for r in rows if r['identity'] == 'V3:data/V3_NORMALIZERS.json')
    assert infer.target_normalizer_identity_sha256() == expected
    assert "locked_inputs['target_normalizer_sha256']" not in (ROOT / 'code/infer_p1_probe.py').read_text()


def test_n0_prediction_acceptance_records_node_diff_for_each_accepted_probe():
    source = (ROOT / 'code/accept_n0_predictions.py').read_text()
    assert 'rows=[];real_records=[];probe_records=[];branches=[];node_records=[]' in source
    assert 'node_records.append({' in source
    assert "'real_node_sha256':side.get('p1_video_projection_input_sha256')" in source
    assert "'probe_node_sha256':ps.get('p1_video_projection_input_sha256')" in source
    assert 'writer.writerows(node_records)' in source
