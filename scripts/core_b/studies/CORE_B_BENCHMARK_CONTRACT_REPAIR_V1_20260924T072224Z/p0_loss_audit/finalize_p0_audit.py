"""Consolidate P0 source/math tests and retry lineage without rewriting sources."""

from __future__ import annotations
import os

import json
from pathlib import Path


ROOT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'reports/benchmarks/CORE_B_BENCHMARK_CONTRACT_REPAIR_V1_20260924T072224Z'))
P0 = ROOT / "p0_loss_audit"


def main() -> None:
    audit = json.loads((P0 / "P0_LOSS_LINEAGE_AUDIT.json").read_text())
    tests = json.loads((P0 / "P0_CONTRACT_REGRESSION_FINAL.json").read_text())
    lineage = json.loads((P0 / "P0_OLD_D3_EFFECTIVE_LINEAGE.json").read_text())
    result = {
        "status": "P0_AUDIT_COMPLETE_CONTRACT_UNRESOLVED",
        "D3_DEVIATION": "unresolved",
        "deviation_against_task_attachment_target": audit["deviation_against_attachment_target"],
        "prior_lock_episode_equal_weighting": "not specified in MODEL_LOSS_CONTRACT.md or D3_PROTOCOL_LOCKED.json",
        "D3_AFFECTED_OLD_RUNS": "36/36 share the audited trainer path; attachment-target equivalence would affect all 36; historical contract classification remains unresolved",
        "historical_D3_effective_lineage": lineage,
        "real_train_contract_tests": tests,
        "cache_contract": {
            "covered_rows": audit["tests"]["spatial_cache_rows"],
            "expected_rows": audit["tests"]["spatial_cache_expected_rows"],
            "bridge_max_abs": audit["tests"]["spatial_cache_bridge_max_abs"],
        },
        "no_D3R_training": True,
        "no_cuda_started": True,
        "dev_targets_accessed": False,
        "final_test_access": 0,
        "resolution_needed": "Confirm whether the episode-equal per-sample objective in the new task attachment is an authorized amendment to the historical D3 lock, because the named original MODEL_LOSS_CONTRACT only locks Arm/Hand loss components, train-only scales, effective batch and optimizer/epoch settings.",
    }
    (P0 / "P0_LOSS_LINEAGE_AUDIT.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    mixed = tests["real_train_fixed_batches"]["first_mixed_batch64"]
    tail = tests["real_train_fixed_batches"]["last_partial_batch"]
    same = tests["real_train_fixed_batches"]["same_episode_subbatch"]
    toy = tests["toy_two_episode"]
    report = [
        "# P0 — D3 loss contract and lineage audit",
        "",
        "Status: `P0_AUDIT_COMPLETE_CONTRACT_UNRESOLVED`. No D3-R code patch or training was started.",
        "",
        "## Source contract discrepancy",
        "",
        "The existing `MODEL_LOSS_CONTRACT.md` states `L_pos + 0.1*L_delta`, equal Arm/Hand group terms, train-only scales, 40 epochs and effective batch 64. `D3_PROTOCOL_LOCKED.json` repeats the Arm/Hand loss composition but neither file specifies an episode-equal training objective or the `W/(E*n_episode)` window weights. The new task attachment supplies that episode-equal objective and explicitly says to pause if it differs from the original contract. Therefore the objective is confirmed non-equivalent to the attachment target, while whether this constitutes a deviation from the historical lock remains unresolved; no authority was inferred to amend the old protocol.",
        "",
        "## Actual sampler and reduction",
        "",
        f"The old trainer reads only primary-train weights from `split_code==0`: W={audit['train_only_population']['W_train_windows']}, E={audit['train_only_population']['E_train_episodes']}, with 1–8 windows/episode. It shuffles one index permutation per epoch and visits every train window once. Observed epoch-1 batches for all three seeds: every physical-64 batch was mixed-episode (278/278); the 32+32 fallback batches were also mixed (555/555) and leave one 29-sample final accumulation cycle.",
        "",
        "The source computes a scalar `mean_i(l_i)` and multiplies it by `mean_i(w_i)`. The requested target is `sum_i(w_i*l_i)/B_actual`; the difference is the batch covariance between weights and sample losses. Same-episode constant-weight batches are the special case where the two values coincide.",
        "",
        "## Deterministic tests",
        "",
        f"- Toy fixture: unequal episode window counts 1 vs 3; legacy value {toy['toy_legacy_value']:.6f}, episode-equal value {toy['toy_episode_equal_value']:.6f}; scalar gradients have opposite signs ({toy['toy_legacy_gradient']:.6f} vs {toy['toy_episode_equal_gradient']:.6f}).",
        f"- Actual train-only first mixed batch: legacy {mixed['legacy_value']:.6f}, samplewise target {mixed['samplewise_weighted_value']:.6f}; scalar probe gradient delta {mixed['gradient_difference']:.6f}.",
        f"- Actual constructed same-episode batch: value delta {same['value_difference']:.3g}; as expected, constant weights are the equivalent special case.",
        f"- Actual 29-sample final tail: correct denominator is 29; historical divide-by-2 fallback gradient differs from the full tail objective by {tests['partial_cycle']['legacy_divide_by_2_vs_correct_max_abs_parameter_gradient_difference']:.6f}.",
        f"- Deterministic zero-initialized Linear(1,1) output head, dropout disabled: effective-64 and 32+32 per-parameter gradients match to max absolute difference {tests['effective_batch_64_vs_32plus32']['max_abs_parameter_gradient_difference']:.3g}.",
        "- Weight/ID alignment uses train anchor IDs only; future targets do not enter weight construction. The train-only spatial cache covers 20,079/20,079 rows and records a raw-forward bridge max absolute difference of 5.72e-6.",
        "",
        "## Historical 36-cell lineage",
        "",
        f"Effective lineage: {lineage['effective_train_complete']}/36 training completions, {lineage['effective_checkpoints_present']}/36 last checkpoints and {lineage['effective_prediction_pass']}/36 prediction sidecars pass. One base H30/seed20261001 attempt stopped with `KeyboardInterrupt` while opening the cache; the preserved RETRY1 completed 40 epochs/11,120 optimizer steps with the same initialization hash and accepted prediction. No failed attempt was removed. All effective runs use the same trainer path and sealed trainer hash; launch-time trainer bytes were not individually recorded.",
        "",
        "The locked historical loss clause does not resolve the new episode-equal weighting clause. Under the attachment's rule, P0 cannot select `confirmed` or `disproved` relative to the prior protocol; the retraining branch remains paused until the contract is reconciled. This is the reason no GPU/resource admission check or D3-R run was started.",
        "",
        "Final/held-out access: `0`. Old benchmark roots remain read-only.",
    ]
    (P0 / "P0_LOSS_LINEAGE_AUDIT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "D3_DEVIATION": result["D3_DEVIATION"],
        "effective_old_runs": lineage["effective_train_complete"],
        "effective_old_predictions": lineage["effective_prediction_pass"],
        "final_test_access": 0,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
