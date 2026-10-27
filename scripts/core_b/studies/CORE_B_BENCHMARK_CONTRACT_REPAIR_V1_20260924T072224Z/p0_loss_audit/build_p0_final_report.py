"""Build final P0 JSON/Markdown after validating every source input first."""

from __future__ import annotations
import os

import json
from pathlib import Path


ROOT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'reports/benchmarks/CORE_B_BENCHMARK_CONTRACT_REPAIR_V1_20260924T072224Z/p0_loss_audit'))


def main() -> None:
    raw = json.loads((ROOT / "P0_LOSS_LINEAGE_AUDIT.json").read_text())
    tests = json.loads((ROOT / "P0_CONTRACT_REGRESSION_FINAL.json").read_text())
    lineage = json.loads((ROOT / "P0_OLD_D3_EFFECTIVE_LINEAGE.json").read_text())
    if raw.get("status") != "PASS_CPU_AUDIT" or tests.get("status") != "PASS" or lineage.get("status") != "PASS_EFFECTIVE_36_CELL_LINEAGE":
        raise AssertionError("P0 source tests/lineage are not all accepted")

    mixed = tests["real_train_fixed_batches"]["first_mixed_batch64"]
    partial = tests["partial_cycle"]
    same = tests["real_train_fixed_batches"]["same_episode_subbatch"]
    toy = tests["toy_two_episode"]
    output = {
        "status": "P0_AUDIT_COMPLETE_CONTRACT_UNRESOLVED",
        "D3_DEVIATION": "unresolved",
        "deviation_against_task_attachment_target": raw["deviation_against_attachment_target"],
        "historical_lock_episode_equal_weighting": "not specified in MODEL_LOSS_CONTRACT.md or D3_PROTOCOL_LOCKED.json",
        "D3_AFFECTED_OLD_RUNS": "36/36 share the audited trainer path; relative to the attachment's episode-equal target all 36 used the non-equivalent implementation; historical lock classification unresolved",
        "effective_historical_lineage": lineage,
        "real_train_regression_tests": tests,
        "train_only_population": raw["train_only_population"],
        "episode_weight_distribution": raw["weight_distribution"],
        "epoch_sampler": raw["sampler"],
        "spatial_cache_contract": {
            "covered_rows": raw["tests"]["spatial_cache_rows"],
            "expected_rows": raw["tests"]["spatial_cache_expected_rows"],
            "bridge_max_abs": raw["tests"]["spatial_cache_bridge_max_abs"],
        },
        "no_D3R_training": True,
        "no_cuda_started": True,
        "dev_targets_accessed": False,
        "final_test_access": 0,
        "resolution_needed": "Confirm whether the episode-equal per-sample objective from the task attachment amends the historical D3 lock. Until confirmed, the attachment's pause rule blocks D3-R code patch/training.",
    }

    report = [
        "# P0 — D3 loss contract and lineage audit",
        "",
        "Status: `P0_AUDIT_COMPLETE_CONTRACT_UNRESOLVED`. No D3-R code patch or training was started.",
        "",
        "## Historical contract and task-target difference",
        "",
        "The existing `MODEL_LOSS_CONTRACT.md` locks `L_pos + 0.1*L_delta`, equal Arm/Hand terms, train-only scales, 40 epochs and effective batch 64. `D3_PROTOCOL_LOCKED.json` repeats the Arm/Hand loss composition; neither document specifies episode-equal weighting or `w_i=W/(E*n_episode)`. The new task attachment supplies that target and explicitly says to pause when it differs from the original contract. Thus the historical implementation is confirmed non-equivalent to the attachment target, but protocol deviation relative to the old lock is unresolved. I did not assume authority to amend the historical contract.",
        "",
        "## Actual sampler, weights, and reduction",
        "",
        f"Primary train only: W={raw['train_only_population']['W_train_windows']:,} windows, E={raw['train_only_population']['E_train_episodes']:,} eligible episodes, with {raw['train_only_population']['min_windows_per_episode']}–{raw['train_only_population']['max_windows_per_episode']} windows per episode. Weights are computed from train split and episode IDs only; dev targets were not read.",
        "The old source shuffles one permutation per epoch and visits each train window exactly once. For all three seeds, every epoch-1 batch was mixed-episode: physical64 had 278/278 mixed batches; the 32+32 fallback had 555/555 mixed microbatches and ends with one 29-sample partial cycle.",
        "The trainer first reduces sample loss to `mean_i(l_i)` and multiplies by `mean_i(w_i)`. The attachment target is `sum_i(w_i*l_i)/B_actual`; the omitted term is batch covariance between weight and per-sample loss. Same-episode constant-weight batches are the exact special case.",
        "",
        "## CPU-only tests",
        "",
        f"- Unequal toy episodes (1 vs 3 windows): legacy value {toy['legacy_value']:.6f}, episode-equal value {toy['episode_equal_value']:.6f}; gradients have opposite signs ({toy['legacy_gradient']:.6f} vs {toy['episode_equal_gradient']:.6f}).",
        f"- Real train-only first mixed batch: legacy value {mixed['legacy_value']:.6f}, target value {mixed['samplewise_weighted_value']:.6f}; probe gradient difference {mixed['gradient_difference']:.6f}.",
        f"- Constructed same-episode sample: value delta {same['value_difference']:.3g}, consistent with the constant-weight special case.",
        f"- Deterministic zero-initialized 2-parameter output head: effective64 vs 32+32 max per-parameter gradient difference {tests['effective_batch_64_vs_32plus32']['max_abs_parameter_gradient_difference']:.3g}; dropout disabled.",
        f"- Final tail has {partial['actual_samples']} samples: correct denominator is {partial['correct_denominator']}; legacy unconditional divide-by-2 changes parameter gradients by {partial['legacy_mean_product_divide2_vs_correct_gradient_difference']:.6f}.",
        f"- Frozen D3 spatial cache metadata covers {raw['tests']['spatial_cache_rows']:,}/{raw['tests']['spatial_cache_expected_rows']:,} rows; recorded raw-forward bridge max abs is {raw['tests']['spatial_cache_bridge_max_abs']:.3g}.",
        "",
        "## Historical 36-cell run lineage",
        "",
        f"Effective matrix: {lineage['effective_train_complete']}/36 run completions, {lineage['effective_checkpoints_present']}/36 checkpoints and {lineage['effective_prediction_pass']}/36 predictions pass. One base H30/seed20261001 attempt stopped with `KeyboardInterrupt` during a cache shard open; its preserved RETRY1 completed 40 epochs/11,120 optimizer steps, used the same initialization hash, and has an accepted prediction. The failure was retained. All effective cells share the sealed trainer path/hash, but an execution-time code hash was not captured separately per launch.",
        "",
        "Because the old loss lock is silent on episode weighting while the attachment requires the new target to match that lock, P0 remains `unresolved` for the historical protocol. D3-R training is paused here; no GPU admission or resource check was started.",
        "",
        "Final/held-out access: `0`. All historical roots remain read-only.",
    ]

    # Preserve the exact raw machine audit before replacing the P0 summary path.
    (ROOT / "P0_LOSS_LINEAGE_AUDIT_SOURCE.json").write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
    (ROOT / "P0_LOSS_LINEAGE_AUDIT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (ROOT / "P0_LOSS_LINEAGE_AUDIT.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": output["status"],
        "D3_DEVIATION": output["D3_DEVIATION"],
        "effective_old_train": lineage["effective_train_complete"],
        "effective_old_prediction": lineage["effective_prediction_pass"],
        "final_test_access": 0,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
