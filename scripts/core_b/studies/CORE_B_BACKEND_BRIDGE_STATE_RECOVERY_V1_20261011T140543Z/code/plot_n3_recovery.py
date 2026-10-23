#!/usr/bin/env python3
"""Render two provisional, source-backed N3 H30 development figures."""
from __future__ import annotations
import os

import csv
import hashlib
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'reports/benchmarks/CORE_B_BACKEND_BRIDGE_STATE_RECOVERY_V1_20261011T140543Z'))
SEEDS = (20260953, 20260954, 20260955)
STATE_COLOR = {'FULL': '#0072B2', 'OBS': '#A36B00', 'ANCHOR': '#009E73'}
EFFECT_COLOR = {'h_command_at_N': '#0072B2', 'h_observation_at_N': '#009E73', 'J_OBS': '#D55E00', 'J_ANCHOR': '#CC79A7'}
SEED_MARKER = {20260953: 'o', 20260954: 's', 20260955: '^'}


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, data: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for item in data:
        for key in item:
            if key not in fields:
                fields.append(key)
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(data)


def save(fig: plt.Figure, stem: str, title: str) -> list[Path]:
    pdf = ROOT / 'figures' / f'{stem}.pdf'
    png = ROOT / 'figures' / f'{stem}.png'
    if pdf.exists() or png.exists():
        raise FileExistsError(f'figure output already exists: {stem}')
    fig.savefig(pdf, format='pdf', facecolor='white', metadata={'Title': title, 'Subject': 'N3 accepted H30 development re-score'})
    fig.savefig(png, format='png', dpi=300, facecolor='white', metadata={'Title': title, 'Description': 'Fixed-seed N3 H30 development re-score; see caption and data CSV.'})
    plt.close(fig)
    return [pdf, png]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    state_rows = rows(ROOT / 'results/N3_STATE_RESULTS.csv')
    gain_rows = rows(ROOT / 'results/N3_VIDEO_GAINS.csv')
    interaction_rows = rows(ROOT / 'results/N3_INFORMATION_INTERACTIONS.csv')
    main_state = [r for r in state_rows if r['full_mae_std']]
    position_g = {r['estimand']: r for r in gain_rows if r['metric'] == 'full_mae_std'}
    position_i = {r['estimand']: r for r in interaction_rows if r['metric'] == 'full_mae_std'}

    figure_data: list[dict[str, Any]] = []
    for state in ('FULL', 'OBS', 'ANCHOR'):
        for video in ('N', 'D'):
            vals = np.asarray([float(next(r['full_mae_std'] for r in main_state if int(r['seed']) == seed and r['state_level'] == state and r['video_level'] == video)) for seed in SEEDS])
            figure_data.append({'panel': 'A', 'quantity': 'full_mae_std', 'state_level': state, 'video_level': video,
                                'mean_fixed_seeds': float(vals.mean()), 'sample_sd_across_seeds': float(vals.std(ddof=1)),
                                'n_fixed_seeds': 3, 'final_test_access': 0})
        g = position_g[f'g_{state}']
        for seed in SEEDS:
            figure_data.append({'panel': 'B', 'quantity': 'video_gain_N_minus_D', 'state_level': state,
                                'video_level': '', 'seed': seed, 'seed_estimate': float(g[f'seed_{seed}']),
                                'aggregate_estimate': float(g['point_estimate_mean_fixed_seeds']),
                                'ci95_low': float(g['group_bootstrap_ci95_low']),
                                'ci95_high': float(g['group_bootstrap_ci95_high']),
                                'bootstrap_unit': 'source_group', 'n_groups': 603, 'final_test_access': 0})
    for name in ('h_command_at_N', 'h_observation_at_N', 'J_OBS', 'J_ANCHOR'):
        item = position_i[name]
        for seed in SEEDS:
            figure_data.append({'panel': 'C', 'quantity': name, 'state_level': '', 'video_level': '',
                                'seed': seed, 'seed_estimate': float(item[f'seed_{seed}']),
                                'aggregate_estimate': float(item['point_estimate_mean_fixed_seeds']),
                                'ci95_low': float(item['group_bootstrap_ci95_low']),
                                'ci95_high': float(item['group_bootstrap_ci95_high']),
                                'bootstrap_unit': 'source_group', 'n_groups': 603, 'final_test_access': 0})
    write_csv(ROOT / 'figure_data/N3_POSITION_FIGURES.csv', figure_data)

    with mpl.rc_context({'font.family': 'DejaVu Sans', 'font.size': 8.5,
                         'axes.titlesize': 9.5, 'axes.labelsize': 8.5,
                         'xtick.labelsize': 8, 'ytick.labelsize': 8,
                         'legend.fontsize': 7.5, 'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.grid': True, 'grid.color': '#D9DDE2', 'grid.linewidth': .55,
                         'axes.axisbelow': True}):
        # Figure 1: level scores and paired video gain. The nonzero score axis is explicit.
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(190/25.4, 105/25.4), layout='constrained',
                                       gridspec_kw={'width_ratios': [1.08, 1.0]})
        x = np.arange(3); labels = ['FULL', 'OBS', 'ANCHOR']
        for video, color, marker, ls, label in (
            ('N', '#555555', 'o', '--', 'No video (N)'),
            ('D', '#0072B2', 's', '-', 'Dynamic video (D)'),
        ):
            means, sds = [], []
            for state in labels:
                vals = np.asarray([float(next(r['full_mae_std'] for r in main_state if int(r['seed']) == seed and r['state_level'] == state and r['video_level'] == video)) for seed in SEEDS])
                means.append(vals.mean()); sds.append(vals.std(ddof=1))
            ax0.errorbar(x, means, yerr=sds, color=color, linestyle=ls, marker=marker,
                         markersize=5, linewidth=1.4, capsize=3, label=label)
        ax0.set_xticks(x, labels)
        ax0.set_ylabel('Full28 position MAE / train target σ')
        ax0.set_title('A  Absolute error by information level')
        ax0.set_ylim(0.18, 0.245)
        ax0.legend(frameon=False, loc='upper left', ncol=1)

        offsets = {'FULL': -0.07, 'OBS': 0.0, 'ANCHOR': 0.07}
        for i, state in enumerate(labels):
            item = position_g[f'g_{state}']
            y = float(item['point_estimate_mean_fixed_seeds'])
            lo = float(item['group_bootstrap_ci95_low']); hi = float(item['group_bootstrap_ci95_high'])
            ypos = i + offsets[state]
            ax1.errorbar(y, ypos, xerr=[[y-lo], [hi-y]], color=STATE_COLOR[state],
                         marker='D', markersize=5, capsize=3, linewidth=1.5, linestyle='none',
                         label=f'{state}: fixed-seed mean, 95% group CI')
            for seed in SEEDS:
                ax1.scatter(float(item[f'seed_{seed}']), ypos + ({SEEDS[0]:-.16, SEEDS[1]:0, SEEDS[2]:.16}[seed]),
                            marker=SEED_MARKER[seed], color=STATE_COLOR[state], s=24, edgecolor='white', linewidth=.4,
                            zorder=4)
        ax1.axvline(0, color='#333333', linewidth=1.0)
        ax1.set_yticks(np.arange(3), labels)
        ax1.invert_yaxis()
        ax1.set_xlabel('g = L(N) − L(D)')
        ax1.set_title('B  Paired dynamic-video gain')
        ax1.set_xlim(-0.015, 0.011)
        f1_paths = save(fig, 'FIG01_N3_VIDEO_GAIN_BY_STATE', 'N3 H30 state-conditioned video gain')

        # Figure 2: information effects and video-gain interactions, all on one signed scale.
        fig, ax = plt.subplots(1, 1, figsize=(190/25.4, 105/25.4), layout='constrained')
        effect_names = ('h_command_at_N', 'h_observation_at_N', 'J_OBS', 'J_ANCHOR')
        effect_labels = ('h_command (N): OBS − FULL', 'h_observation (N): ANCHOR − OBS',
                         'J_OBS: g_OBS − g_FULL', 'J_ANCHOR: g_ANCHOR − g_FULL')
        ypos = np.arange(len(effect_names))
        for i, (name, label) in enumerate(zip(effect_names, effect_labels)):
            item = position_i[name]
            x0 = float(item['point_estimate_mean_fixed_seeds'])
            lo = float(item['group_bootstrap_ci95_low']); hi = float(item['group_bootstrap_ci95_high'])
            ax.errorbar(x0, ypos[i], xerr=[[x0-lo], [hi-x0]], color=EFFECT_COLOR[name],
                        marker='D', markersize=5, capsize=3, linewidth=1.5, linestyle='none', zorder=3)
            for seed in SEEDS:
                dy = {SEEDS[0]:-.12, SEEDS[1]:0.0, SEEDS[2]:.12}[seed]
                ax.scatter(float(item[f'seed_{seed}']), ypos[i]+dy, marker=SEED_MARKER[seed],
                           color=EFFECT_COLOR[name], s=25, edgecolor='white', linewidth=.4, zorder=4)
        ax.axvline(0, color='#333333', linewidth=1.0)
        ax.set_yticks(ypos, effect_labels)
        ax.invert_yaxis()
        ax.set_xlabel('Difference in Full28 position MAE / train target σ')
        ax.set_title('Information contribution and video-gain interaction')
        ax.set_xlim(-0.008, 0.024)
        f2_paths = save(fig, 'FIG02_N3_INFORMATION_INTERACTIONS', 'N3 H30 information and video-gain interactions')

    # Captions and a small hash manifest; publisher-specific requirements remain pending.
    (ROOT / 'captions/FIG01_N3_VIDEO_GAIN_BY_STATE.md').write_text(
        'Figure 1. N3 H30 primary-dev Full28 standardized position error and paired dynamic-video contrast across FULL, OBS, and ANCHOR. Panel A shows each video condition mean across three fixed seeds with sample SD; lower error is better. Panel B defines g=L(N)−L(D), so positive values favor dynamic video. Diamonds and horizontal whiskers show the mean across the fixed seeds and a 95% percentile CI from 10,000 paired source-group bootstrap draws (603 groups; PRNG 20261011); small symbols show each seed. FULL is near zero with a CI crossing zero, OBS is negative, and ANCHOR is positive. These are fixed-model, previously exposed development estimates, not independent confirmation.\n\nAlt text: Two panels compare H30 position error across FULL, OBS, and ANCHOR. Dynamic video is slightly better in FULL with an interval spanning zero, worse in OBS, and better in ANCHOR. Seed-level points show variation.\n\nPublisher dimensions and submission rules are provisional; no target journal was specified.\n', encoding='utf-8')
    (ROOT / 'captions/FIG02_N3_INFORMATION_INTERACTIONS.md').write_text(
        'Figure 2. Paired N3 H30 information effects and interactions for Full28 standardized position error. h_command=L(OBS,N)−L(FULL,N); h_observation=L(ANCHOR,N)−L(OBS,N). J_OBS=g_OBS−g_FULL and J_ANCHOR=g_ANCHOR−g_FULL, where g=L(N)−L(D). Positive h means the removed history block reduced predictive information under N. Positive J means a more positive video gain under the lower-information condition, but does not by itself establish absolute benefit. Diamonds and horizontal whiskers show fixed-seed means and 95% paired source-group percentile CIs (10,000 draws; 603 groups; PRNG 20261011); small symbols show the three fixed seeds. The dev set had prior exposure; results are exploratory.\n\nAlt text: Command-history removal has a small positive error increase, observation-history removal has a larger increase. J_OBS is negative and J_ANCHOR is positive, indicating the video interaction differs by state level rather than increasing monotonically as history is removed.\n\nPublisher dimensions and submission rules are provisional; no target journal was specified.\n', encoding='utf-8')
    manifest = []
    for path in f1_paths + f2_paths:
        manifest.append({'file': str(path.relative_to(ROOT)), 'sha256': sha(path), 'bytes': path.stat().st_size,
                         'formats': path.suffix[1:], 'source_data': 'figure_data/N3_POSITION_FIGURES.csv',
                         'transform': 'N3 accepted scorer; fixed-seed mean; paired 603 source groups',
                         'final_test_access': 0})
    with (ROOT / 'figures/FIGURE_MANIFEST.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0])); w.writeheader(); w.writerows(manifest)
    print({'status': 'PASS_RENDERED', 'figures': [x['file'] for x in manifest], 'figure_data_rows': len(figure_data)})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
