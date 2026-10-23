#!/usr/bin/env python3
"""Presentation-only revision of FIG05 title precision; reads locked train-only receipts."""
import csv,json,resource,time
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
growth=json.loads((ROOT/'results/TRAIN_TRANSITION_GROWTH.json').read_text())
with (ROOT/'results/TRAIN_TRANSITION_SPECTRUM.csv').open(encoding='utf-8',newline='') as f: eig=list(csv.DictReader(f))
absvals=[float(r['eigen_abs']) for r in eig]
g= growth['growth']; rho=float(growth['spectral_radius']); n_gt=int(growth['eigenvalues_abs_gt_1'])
fig,axs=plt.subplots(1,2,figsize=(10,4.2))
axs[0].hist(absvals,bins=24,color='#4c78a8'); axs[0].axvline(1,color='red',ls='--',label='unit radius'); axs[0].set_xlabel('|eigenvalue|'); axs[0].set_ylabel('count'); axs[0].set_title(f'Train-only 86×86 transition; ρ={rho:.6f}\n{n_gt} eigenvalue(s) above 1'); axs[0].legend(fontsize=8)
axs[1].plot([int(r['horizon']) for r in g],[float(r['spectral_norm_A_power_h']) for r in g],marker='o'); axs[1].set_yscale('log'); axs[1].set_xticks([int(r['horizon']) for r in g]); axs[1].set_xlabel('h'); axs[1].set_ylabel('||A^h||₂ (log scale)'); axs[1].set_title('Homogeneous linear perturbation growth'); axs[1].grid(alpha=.25)
fig.tight_layout(); out=ROOT/'figures/FIG05_TRAIN_TRANSITION_SPECTRUM_AMEND001.png'; fig.savefig(out,dpi=180); plt.close(fig)
print(json.dumps({'status':'PASS_PRESENTATION_ONLY','output':str(out),'spectral_radius_full_precision':rho,'source_growth_sha256':__import__('hashlib').sha256((ROOT/'results/TRAIN_TRANSITION_GROWTH.json').read_bytes()).hexdigest(),'dev_labels_read':False,'model_or_metric_changed':False,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'wall_seconds':time.perf_counter()},sort_keys=True))
