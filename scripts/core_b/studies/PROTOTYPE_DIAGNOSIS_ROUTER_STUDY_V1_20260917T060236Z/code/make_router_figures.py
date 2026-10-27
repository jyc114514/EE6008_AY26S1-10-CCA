#!/usr/bin/env python3
"""Create truthful CPU-only diagnostic figures for the router study.

The figures are audit outputs, not evidence of a trained router.  All oracle
curves use future targets and are explicitly marked non-deployable.
"""

from __future__ import annotations
import os

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
REPO = PROJECT / "repo/ee6008-humanoid-vjepa"
PROTO = REPO / "reports/core_b/HAND_CALIBRATED_PROTOTYPE_MIXTURE_V1_20260915T125843Z"
INPUT = REPO / "reports/core_b/JY02_RESIDUAL_TEMPORAL_DECODER_V1_20260912T095449Z/inputs/JY02_TARGETS.npz"
NORMALIZER = REPO / "reports/core_b/feasibility_and_smoke_20260901T160933Z/CORE_B_NORMALIZATION_STATS_TRAIN_ONLY.json"
HAND = 14
STEPS = 30
FOLDS = 3
SEED = 20260937

COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "vermilion": "#D55E00",
    "green": "#009E73",
    "purple": "#CC79A7",
    "gray": "#666666",
    "black": "#111111",
    "light_blue": "#56B4E9",
}


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def dct_basis(length: int = STEPS) -> np.ndarray:
    t = np.arange(length, dtype=np.float64)[None, :]
    k = np.arange(length, dtype=np.float64)[:, None]
    basis = np.cos(np.pi / length * (t + 0.5) * k) * np.sqrt(2.0 / length)
    basis[0] *= 1.0 / np.sqrt(2.0)
    return basis


def fold_assignment(ids: np.ndarray) -> np.ndarray:
    ids = np.asarray(ids, dtype=np.uint64)
    return ((ids * np.uint64(2654435761) + np.uint64(1013904223)) % np.uint64(FOLDS)).astype(np.int64)


def normalise_probabilities(probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    row_sum = values.sum(axis=1, keepdims=True)
    values = np.divide(values, row_sum, out=np.full_like(values, 0.5), where=row_sum > 1e-12)
    values = np.clip(values, 1e-12, 1.0)
    return values / values.sum(axis=1, keepdims=True)


def fusion_probe_oof(features: np.ndarray, labels: np.ndarray, ids: np.ndarray) -> np.ndarray:
    folds = fold_assignment(ids)
    probabilities = np.zeros((len(labels), 2), dtype=np.float64)
    for fold in range(FOLDS):
        train = folds != fold
        test = folds == fold
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                ("logistic", LogisticRegression(C=1.0, max_iter=300, solver="lbfgs", random_state=SEED)),
            ]
        )
        model.fit(features[train], labels[train])
        classifier = model.named_steps["logistic"]
        classes = classifier.classes_.astype(np.int64)
        probabilities[np.ix_(test, classes)] = classifier.predict_proba(features[test])
    return normalise_probabilities(probabilities)


def save_figure(fig: plt.Figure, stem: str) -> None:
    figures = ROOT / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures / f"{stem}.png", dpi=300, facecolor="white")
    fig.savefig(figures / f"{stem}.pdf", dpi=300, facecolor="white")
    plt.close(fig)


def caption(stem: str, text: str) -> None:
    path = ROOT / "figures/captions" / f"{stem}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def main() -> int:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    (ROOT / "figures/data").mkdir(parents=True, exist_ok=True)
    data = load_npz(INPUT)
    train_mask = data["split_code"] == 0
    dev_mask = data["split_code"] == 1
    dev_ids = data["episode_index"][dev_mask].astype(np.int64)
    targets_native = data["targets_native"][dev_mask].astype(np.float64)
    persistence_native = data["persistence_native"][dev_mask].astype(np.float64)
    targets_hand = targets_native[:, :, :HAND]
    persistence_hand = persistence_native[:, :, :HAND]
    normalizer = json.loads(NORMALIZER.read_text(encoding="utf-8"))["1.0s"]
    mean = np.asarray(normalizer["mean"], dtype=np.float64)
    scale = np.asarray(normalizer["safe_std"], dtype=np.float64)
    target_std = data["targets_standardized"][dev_mask].astype(np.float64)
    persistence_std = data["persistence_standardized"][dev_mask].astype(np.float64)
    proto = load_npz(PROTO / "predictions/HP-PROTO_PREDICTIONS_DEV.npz")
    single = load_npz(PROTO / "predictions/HP-SINGLE_PREDICTIONS_DEV.npz")
    components_std = proto["components_hand_standardized"].astype(np.float64)
    components_native = components_std * scale[None, None, :, :HAND] + mean[None, None, :, :HAND]
    q_ref = proto["q"].astype(np.float64)[:, :, 0]
    probabilities = normalise_probabilities(proto["mode_probabilities"].astype(np.float64))
    shadow_components_native = persistence_hand[:, None] + q_ref[:, None, :, None] * (components_native - persistence_hand[:, None])
    shadow_components_std = persistence_std[:, None, :, :HAND] + q_ref[:, None, :, None] * (components_std - persistence_std[:, None, :, :HAND])
    chosen = probabilities.argmax(axis=1)
    shadow_map_native = shadow_components_native[np.arange(len(dev_ids)), chosen]
    shadow_mean_native = persistence_hand + q_ref[:, :, None] * (np.sum(probabilities[:, :, None, None] * components_native, axis=1) - persistence_hand)
    shadow_mean_std = persistence_std[:, :, :HAND] + q_ref[:, :, None] * (np.sum(probabilities[:, :, None, None] * components_std, axis=1) - persistence_std[:, :, :HAND])

    centers_payload = load_npz(PROTO / "mode_audit/PROTOTYPE_CENTERS.npz")
    centers_flat = centers_payload["centers_K2"].astype(np.float64)
    coefficient_mean = centers_payload["coefficient_mean"].astype(np.float64)
    coefficient_scale = centers_payload["coefficient_scale"].astype(np.float64)
    centers = centers_flat.reshape(2, 16, HAND)
    train_target_native = data["targets_native"][train_mask, :, :HAND].astype(np.float64)
    train_p_native = data["persistence_native"][train_mask, :, :HAND].astype(np.float64)
    basis = dct_basis()[:16]
    train_coeff = np.einsum("kt,etj->ekj", basis, train_target_native - train_p_native)
    dev_coeff = np.einsum("kt,etj->ekj", basis, targets_hand - persistence_hand)
    train_z = ((train_coeff - coefficient_mean) / coefficient_scale).reshape(len(train_coeff), -1)
    dev_z = ((dev_coeff - coefficient_mean) / coefficient_scale).reshape(len(dev_coeff), -1)
    dev_true_mode = ((dev_z[:, None, :] - centers_flat[None, :, :]) ** 2).sum(axis=2).argmin(axis=1).astype(np.int64)
    with (PROTO / "mode_audit/CLUSTER_ASSIGNMENTS.csv").open(encoding="utf-8", newline="") as handle:
        assignment_rows = list(csv.DictReader(handle))
    train_labels = np.asarray([int(row["K2"]) for row in assignment_rows], dtype=np.int64)
    train_ids = np.asarray([int(row["episode_index"]) for row in assignment_rows], dtype=np.int64)
    state_train = data["state_action_history"][train_mask].astype(np.float64)
    dynamic_train = data["dynamic_pooled"][train_mask].astype(np.float64)
    prefix_train = np.column_stack((state_train.mean(axis=1), state_train.std(axis=1), np.abs(np.diff(state_train, axis=1)).mean(axis=1)))
    fusion_features = np.concatenate((dynamic_train, state_train, prefix_train), axis=1)
    fusion_prob = fusion_probe_oof(fusion_features, train_labels, train_ids)
    task_train = data["original_task_index"][train_mask].astype(np.int64)
    task_count = int(max(task_train.max(), data["original_task_index"][dev_mask].max())) + 1
    task_prob = fusion_probe_oof(np.eye(task_count, dtype=np.float64)[task_train], train_labels, train_ids)
    old_prob = probabilities

    # Figure 1: mode proportions and the old-router confusion matrix.
    pop_names = ["train\ntrue", "dev\ntrue", "old router\ndev pred", "fusion probe\ntrain OOF", "task probe\ntrain OOF"]
    pop_labels = [train_labels, dev_true_mode, chosen, np.argmax(fusion_prob, axis=1), np.argmax(task_prob, axis=1)]
    proportions = np.asarray([[np.mean(labels == cls) for cls in (0, 1)] for labels in pop_labels])
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), layout="constrained")
    x = np.arange(len(pop_names))
    axes[0].bar(x, proportions[:, 0], color=COLORS["blue"], label="mode 0 (minority in train)")
    axes[0].bar(x, proportions[:, 1], bottom=proportions[:, 0], color=COLORS["orange"], label="mode 1")
    axes[0].set_ylim(0, 1.08)
    axes[0].set_ylabel("Proportion")
    axes[0].set_title("Mode proportions by population")
    axes[0].set_xticks(x, pop_names)
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=1, frameon=False)
    confusion = np.zeros((2, 2), dtype=int)
    for truth, prediction in zip(dev_true_mode, chosen):
        confusion[truth, prediction] += 1
    image = axes[1].imshow(confusion, cmap="Blues", vmin=0)
    for row in range(2):
        for col in range(2):
            axes[1].text(col, row, str(confusion[row, col]), ha="center", va="center", color="white" if confusion[row, col] > confusion.max() * 0.55 else COLORS["black"], fontsize=12)
    axes[1].set(xticks=[0, 1], yticks=[0, 1], xlabel="Old router predicted mode", ylabel="Fixed-center dev true mode", title="Old prototype router (dev)")
    fig.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04, label="Episodes")
    save_figure(fig, "MODE_PROPORTIONS_CONFUSION")
    write_csv(ROOT / "figures/data/MODE_PROPORTIONS_CONFUSION.csv", [{"population": name.replace("\n", " "), "mode0_proportion": float(row[0]), "mode1_proportion": float(row[1]), "N": int(len(labels))} for name, row, labels in zip(pop_names, proportions, pop_labels)])
    caption("MODE_PROPORTIONS_CONFUSION", """# Mode proportions and confusion

Development/train descriptive audit. Mode 0 is the minority class in the train clustering. The left panel keeps train true, dev true, old-router dev predictions, and train-fold OOF probes visibly separate because they are different populations. The right panel is the old prototype router against fixed-center dev labels (`N=614`). Counts are episodes; no router was trained in this study. Underlying proportions are in the accompanying CSV.""")

    # Figure 2: minority recall and reliability, with the population caveat visible.
    recall_rows = read_csv(ROOT / "analysis/MODE_IMBALANCE_AUDIT.csv")
    recall_lookup = {row["population"]: float(row["recall"]) for row in recall_rows if row["class"] == "0"}
    recall_order = ["OLD_PROTO_ROUTER_DEV", "LINEAR_PROBE_FUSION_OOF", "LINEAR_PROBE_TASK_ONLY_OOF"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), layout="constrained")
    bars = axes[0].bar([0, 1, 2], [recall_lookup[name] for name in recall_order], color=[COLORS["vermilion"], COLORS["green"], COLORS["purple"]], edgecolor=COLORS["black"])
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Minority-mode recall")
    axes[0].set_title("Recall is descriptive, not cross-population causal")
    axes[0].set_xticks([0, 1, 2], ["old router\ndev", "fusion probe\ntrain OOF", "task probe\ntrain OOF"])
    for bar in bars:
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03, f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    axes[0].axvline(0.5, color=COLORS["gray"], linestyle="--", linewidth=1)
    axes[0].text(0.52, 0.08, "dev vs\ntrain OOF", color=COLORS["gray"], fontsize=8)
    bins = np.linspace(0, 1, 6)
    reliability_rows: list[dict[str, object]] = []
    curve_specs = [("old router dev", dev_true_mode, old_prob[:, 1], COLORS["vermilion"], "o"), ("fusion probe train OOF", train_labels, fusion_prob[:, 1], COLORS["green"], "s"), ("task probe train OOF", train_labels, task_prob[:, 1], COLORS["purple"], "^")]
    axes[1].plot([0, 1], [0, 1], color=COLORS["gray"], linestyle="--", label="ideal")
    for name, labels, probs, color, marker in curve_specs:
        means: list[float] = []
        observed: list[float] = []
        for left, right in zip(bins[:-1], bins[1:]):
            mask = (probs >= left) & ((probs < right) if right < 1 else (probs <= right))
            if not np.any(mask):
                continue
            mean_prob = float(np.mean(probs[mask]))
            observed_rate = float(np.mean(labels[mask] == 1))
            means.append(mean_prob)
            observed.append(observed_rate)
            reliability_rows.append({"population": name, "bin_left": left, "bin_right": right, "N": int(mask.sum()), "mean_predicted_mode1": mean_prob, "observed_mode1_rate": observed_rate})
        axes[1].plot(means, observed, color=color, marker=marker, linewidth=1.5, label=name)
    axes[1].set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted probability of mode 1", ylabel="Observed mode 1 rate", title="Reliability by available population")
    axes[1].legend(frameon=False, loc="upper left")
    save_figure(fig, "MINORITY_RECALL_CALIBRATION")
    write_csv(ROOT / "figures/data/MINORITY_RECALL_CALIBRATION.csv", reliability_rows + [{"population": name, "minority_recall": recall_lookup[name]} for name in recall_order])
    caption("MINORITY_RECALL_CALIBRATION", """# Minority recall and calibration

Mode-0 recall and five-bin reliability curves are shown for the old prototype router on dev and matched linear probes on train-fold OOF predictions. The dashed divider and labels are intentional: dev and train OOF values are not treated as a comparable selector deficit. Reliability uses the probability of mode 1 and retains empty bins as gaps. These are diagnostics only; no router fit was authorized.""")

    # Fixed-q choices and aggregate component metrics.
    component_level_err = np.abs(shadow_components_std - target_std[:, None, :, :HAND]).mean(axis=(2, 3))
    component_fd_err = np.abs(np.diff(shadow_components_native, axis=2) - np.diff(targets_hand[:, None], axis=2)).mean(axis=(2, 3))
    fd_choice = component_fd_err.argmin(axis=1)
    level_choice = component_level_err.argmin(axis=1)
    safe_scale = np.asarray(json.loads((REPO / "reports/core_b/HAND_DCT_DUAL_LOSS_FACTORIAL_V1_20260915T050907Z/TEMPORAL_LOSS_CONTRACT.json").read_text())["safe_scale_j"], dtype=np.float64)
    lambda_fd = float(json.loads((REPO / "reports/core_b/HAND_DCT_DUAL_LOSS_FACTORIAL_V1_20260915T050907Z/TEMPORAL_LOSS_CONTRACT.json").read_text())["lambda_fd"])
    component_fd_scaled = (np.abs(np.diff(shadow_components_native, axis=2) - np.diff(targets_hand[:, None], axis=2)) / safe_scale[None, None, None, :]).mean(axis=(2, 3))
    dual_choice = (component_level_err + lambda_fd * component_fd_scaled).argmin(axis=1)
    oracle_fd_native = shadow_components_native[np.arange(len(dev_ids)), fd_choice]
    oracle_names = ["PERSISTENCE", "SHADOW_MAP", "SHADOW_MEAN", "ORACLE_LEVEL", "ORACLE_FD", "ORACLE_DUAL"]
    metric_rows = {row["model"]: row for row in read_csv(ROOT / "analysis/COMPONENT_ORACLE_RESULTS.csv")}
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), layout="constrained")
    x = np.arange(len(oracle_names))
    axes[0, 0].bar(x, [float(metric_rows[name]["hand_mae_std"]) for name in oracle_names], color=[COLORS["gray"], COLORS["blue"], COLORS["light_blue"], COLORS["orange"], COLORS["vermilion"], COLORS["purple"]])
    axes[0, 0].set(xticks=x, xticklabels=["P", "MAP", "MEAN", "level\noracle", "FD\noracle", "dual\noracle"], ylabel="Hand standardized MAE", title="Level accuracy (oracle rows use future targets)")
    axes[0, 1].bar(x, [float(metric_rows[name]["hand_fd_native"]) for name in oracle_names], color=[COLORS["gray"], COLORS["blue"], COLORS["light_blue"], COLORS["orange"], COLORS["vermilion"], COLORS["purple"]])
    axes[0, 1].set(xticks=x, xticklabels=["P", "MAP", "MEAN", "level\noracle", "FD\noracle", "dual\noracle"], ylabel="Hand native FD error", title="Temporal fidelity")
    rep = 0
    axes[1, 0].plot(targets_hand[rep, :, 0], color=COLORS["black"], linewidth=2, label="target")
    axes[1, 0].plot(persistence_hand[rep, :, 0], color=COLORS["gray"], linestyle="--", label="Persistence")
    axes[1, 0].plot(shadow_mean_native[rep, :, 0], color=COLORS["blue"], label="fixed-q shadow MEAN")
    axes[1, 0].plot(oracle_fd_native[rep, :, 0], color=COLORS["vermilion"], label="FD oracle (non-deployable)")
    axes[1, 0].set(xlabel="Time step", ylabel="Native coordinate (unit unverified)", title=f"One fixed dev episode (ID {int(dev_ids[rep])}), hand joint 0")
    axes[1, 0].legend(frameon=False, loc="best")
    choice_names = ["level oracle", "FD oracle", "dual oracle"]
    choices = [level_choice, fd_choice, dual_choice]
    width = 0.35
    for offset, (name, choice) in enumerate(zip(choice_names, choices)):
        counts = np.bincount(choice, minlength=2) / len(choice)
        axes[1, 1].bar(np.arange(2) + (offset - 1) * width, counts, width=width, label=name)
    axes[1, 1].set(xticks=[-0.35, 0.65], xticklabels=["component 0", "component 1"], ylim=(0, 1), ylabel="Episode fraction", title="Which component wins under each oracle?")
    axes[1, 1].legend(frameon=False, fontsize=7)
    save_figure(fig, "COMPONENT_ORACLE_TRAJECTORIES")
    write_csv(ROOT / "figures/data/COMPONENT_ORACLE_TRAJECTORIES.csv", [{"model": name, **metric_rows[name]} for name in oracle_names] + [{"choice_rule": name, "component0_fraction": float(np.mean(choice == 0)), "component1_fraction": float(np.mean(choice == 1))} for name, choice in zip(choice_names, choices)])
    caption("COMPONENT_ORACLE_TRAJECTORIES", """# Fixed-q component and oracle trajectories

Aggregate Hand level and native 29-step future-to-future FD errors are paired with one fixed development episode and fixed-q component choice fractions. `ORACLE_LEVEL`, `ORACLE_FD`, and `ORACLE_DUAL` select a complete component with knowledge of the future target and are therefore non-deployable headroom references, not candidate models. The representative episode is fixed at the first sorted dev row, not selected by performance.""")

    # Figure 4: fixed-batch loss and gradient decomposition.
    loss_audit = json.loads((ROOT / "analysis/PROTOTYPE_LOSS_AND_CODEBOOK_AUDIT.json").read_text(encoding="utf-8"))
    loss_names = ["position", "FD scaled", "dual composite", "mode CE", "refinement"]
    loss_values = [loss_audit["position_loss_mean"], loss_audit["fd_scaled_loss_mean"], loss_audit["dual_loss_mean"], loss_audit["mode_loss_mean"], loss_audit["refine_loss_mean"]]
    gradient_names = list(loss_audit["gradient_norms"]["dual"].keys())
    gradient = np.asarray([[loss_audit["gradient_norms"][term][name] for name in gradient_names] for term in ("dual", "mode", "refine")], dtype=np.float64)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), layout="constrained")
    axes[0].bar(np.arange(len(loss_names)), loss_values, color=[COLORS["blue"], COLORS["vermilion"], COLORS["purple"], COLORS["green"], COLORS["orange"]])
    axes[0].set(xticks=np.arange(len(loss_names)), xticklabels=loss_names, ylabel="Loss value", title="Fixed train calibration batch (N=32)")
    image = axes[1].imshow(gradient, cmap="viridis", vmin=0)
    for row in range(gradient.shape[0]):
        for col in range(gradient.shape[1]):
            axes[1].text(col, row, f"{gradient[row, col]:.3g}", ha="center", va="center", color="white" if gradient[row, col] > gradient.max() * 0.45 else COLORS["black"], fontsize=7)
    gradient_labels = ["in.bias", "in.weight", "mode.bias", "mode.weight", "refine.bias", "refine.weight"]
    axes[1].set(xticks=np.arange(len(gradient_names)), xticklabels=gradient_labels, yticks=[0, 1, 2], yticklabels=["dual", "mode", "refine"], title="Parameter gradient L2 norms")
    for tick in axes[1].get_xticklabels():
        tick.set_rotation(25)
        tick.set_ha("right")
    fig.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04, label="L2 norm")
    save_figure(fig, "LOSS_GRADIENT_AUDIT")
    write_csv(ROOT / "figures/data/LOSS_GRADIENT_AUDIT.csv", [{"quantity": name, "value": value} for name, value in zip(loss_names, loss_values)] + [{"loss_term": term, "parameter": parameter, "gradient_l2": float(value)} for term, row in zip(("dual", "mode", "refine"), gradient) for parameter, value in zip(gradient_names, row)])
    caption("LOSS_GRADIENT_AUDIT", """# Loss and gradient audit

Values come from one fixed train calibration batch of 32 rows using locked loss scales. The heatmap reports parameter-gradient L2 norms for the three separately differentiated loss terms. This is a contract audit, not a claim about training dominance or a new optimization run.""")

    # Figure 5: level/FD trade-off and the locked one-percent boundary.
    all_rows = {row["model"]: row for row in read_csv(ROOT / "analysis/METRIC_MAIN_RESULTS.csv")}
    all_rows.update(metric_rows)
    trade_names = oracle_names
    p_fd = float(all_rows["PERSISTENCE"]["hand_fd_native"])
    p_mae = float(all_rows["PERSISTENCE"]["hand_mae_std"])
    shadow_gain = p_mae - float(all_rows["SHADOW_MAP"]["hand_mae_std"])
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), layout="constrained")
    styles = {"PERSISTENCE": (COLORS["gray"], "o"), "SHADOW_MAP": (COLORS["blue"], "s"), "SHADOW_MEAN": (COLORS["light_blue"], "D"), "ORACLE_LEVEL": (COLORS["orange"], "^"), "ORACLE_FD": (COLORS["vermilion"], "v"), "ORACLE_DUAL": (COLORS["purple"], "P")}
    annotation_offsets = {"PERSISTENCE": (5, -13), "SHADOW_MAP": (5, 8), "SHADOW_MEAN": (5, -10), "ORACLE_LEVEL": (-42, 10), "ORACLE_FD": (5, 8), "ORACLE_DUAL": (5, -12)}
    for name in trade_names:
        x_value = float(all_rows[name]["hand_mae_std"])
        y_value = float(all_rows[name]["hand_fd_native"])
        color, marker = styles[name]
        axes[0].scatter(x_value, y_value, s=65, color=color, marker=marker, edgecolor=COLORS["black"], linewidth=0.4, label=name.replace("ORACLE_", "oracle ").replace("SHADOW_", "shadow ").title())
        short_name = name.replace("PERSISTENCE", "P").replace("SHADOW_", "S-").replace("ORACLE_", "O-")
        axes[0].annotate(short_name, (x_value, y_value), xytext=annotation_offsets[name], textcoords="offset points", fontsize=7)
    axes[0].axhline(p_fd, color=COLORS["gray"], linestyle="--", linewidth=1, label="Persistence FD")
    axes[0].axhline(1.01 * p_fd, color=COLORS["vermilion"], linestyle=":", linewidth=1.4, label="+1% FD gate")
    axes[0].set(xlabel="Hand standardized MAE (lower is better)", ylabel="Hand native FD error (lower is better)", title="Level–temporal-fidelity trade-off")
    axes[0].legend(frameon=False, fontsize=7, loc="best")
    fd_relative = [(float(all_rows[name]["hand_fd_native"]) / p_fd - 1) * 100 for name in trade_names]
    bars = axes[1].bar(np.arange(len(trade_names)), fd_relative, color=[styles[name][0] for name in trade_names], edgecolor=COLORS["black"])
    axes[1].axhline(0, color=COLORS["gray"], linewidth=1)
    axes[1].axhline(1, color=COLORS["vermilion"], linestyle=":", linewidth=1.4, label="+1% gate")
    axes[1].set(xticks=np.arange(len(trade_names)), xticklabels=["P", "MAP", "MEAN", "level", "FD", "dual"], ylabel="FD change vs Persistence (%)", title="Locked FD acceptance boundary")
    for bar, value in zip(bars, fd_relative):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value + (0.04 if value >= 0 else -0.08), f"{value:.2f}%", ha="center", va="bottom" if value >= 0 else "top", fontsize=7)
    axes[1].legend(frameon=False, loc="upper left")
    save_figure(fig, "LEVEL_FD_TRADEOFF")
    write_csv(ROOT / "figures/data/LEVEL_FD_TRADEOFF.csv", [{"model": name, "hand_mae_std": float(all_rows[name]["hand_mae_std"]), "hand_fd_native": float(all_rows[name]["hand_fd_native"]), "fd_relative_change_vs_persistence_pct": float(fd_relative), "hand_gain_vs_persistence": float(all_rows[name].get("hand_gain_vs_persistence", p_mae - float(all_rows[name]["hand_mae_std"]))), "shadow_map_gain_retention": float((p_mae - float(all_rows[name]["hand_mae_std"])) / shadow_gain) if shadow_gain else np.nan} for name, fd_relative in zip(trade_names, fd_relative)])
    caption("LEVEL_FD_TRADEOFF", """# Level accuracy versus temporal fidelity

Lower is better on both axes. The horizontal dotted boundary is the pre-locked one-percent FD deterioration limit relative to Persistence. Oracle points use future targets and are non-deployable; their presence quantifies headroom and does not authorize a router. No axis transformation or selective exclusion was applied.""")

    # Figure 6: fixed-position representative episodes, not cherry-picked by score.
    representative_positions = [0, len(dev_ids) // 2, len(dev_ids) - 1]
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), layout="constrained", sharey=True)
    rep_rows: list[dict[str, object]] = []
    for axis, position in zip(axes, representative_positions):
        curves = [("target", targets_hand[position, :, 0], COLORS["black"], "-", 2.0), ("Persistence", persistence_hand[position, :, 0], COLORS["gray"], "--", 1.2), ("HP-SINGLE", single["hand_predictions_native"][position, :, 0], COLORS["blue"], "-", 1.2), ("fixed-q MEAN", shadow_mean_native[position, :, 0], COLORS["green"], "-.", 1.2), ("FD oracle*", oracle_fd_native[position, :, 0], COLORS["vermilion"], ":", 1.4)]
        for name, values, color, linestyle, linewidth in curves:
            axis.plot(np.arange(STEPS), values, color=color, linestyle=linestyle, linewidth=linewidth, label=name)
            for time, value in enumerate(values):
                rep_rows.append({"representative_position": position, "episode_index": int(dev_ids[position]), "mode_fixed_center": int(dev_true_mode[position]), "time_step": time, "joint": 0, "series": name, "native_value": float(value)})
        axis.set_title(f"dev position {position}\nID {int(dev_ids[position])}; mode {int(dev_true_mode[position])}")
        axis.set_xlabel("Time step")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Native coordinate (unit unverified)")
    axes[-1].legend(frameon=False, fontsize=7, loc="best")
    fig.suptitle("Fixed representative episodes (hand joint 0)", fontsize=11)
    save_figure(fig, "FIXED_REPRESENTATIVE_EPISODES")
    write_csv(ROOT / "figures/data/FIXED_REPRESENTATIVE_EPISODES.csv", rep_rows)
    caption("FIXED_REPRESENTATIVE_EPISODES", """# Fixed representative episodes

Hand joint 0 trajectories are shown for development positions 0, 307, and 613 in canonical episode order. Positions were fixed before plotting and were not selected by error or mode outcome. `FD oracle*` uses future target information and is non-deployable; native coordinate units remain unverified.""")

    manifest = {
        "status": "PASS_WITH_CAVEAT",
        "figure_count": 6,
        "formats": ["png", "pdf"],
        "audience": "internal development audit; not final evaluation",
        "source_files": [str(INPUT), str(PROTO / "predictions/HP-PROTO_PREDICTIONS_DEV.npz"), str(PROTO / "predictions/HP-SINGLE_PREDICTIONS_DEV.npz"), str(ROOT / "analysis/COMPONENT_ORACLE_RESULTS.csv"), str(ROOT / "analysis/MODE_IMBALANCE_AUDIT.csv"), str(ROOT / "analysis/PROTOTYPE_LOSS_AND_CODEBOOK_AUDIT.json")],
        "transformations": ["development split only", "native and standardized coordinates retained as labelled", "fixed-q replay", "five-bin reliability summaries", "no smoothing or cherry-picked representative positions"],
        "uncertainty": "Aggregate figures show point estimates; episode bootstrap intervals remain in analysis/PAIRED_EFFECTS.csv where applicable.",
        "oracle_warning": "ORACLE_LEVEL, ORACLE_FD, ORACLE_DUAL, and FD oracle trajectories use future targets and are non-deployable.",
        "final_test_access": 0,
        "router_training_started": False,
    }
    (ROOT / "figures/FIGURE_MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "figures": 6, "final_test_access": 0}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
