#!/usr/bin/env python3
"""Presentation-only revision using explicit numeric 30 Hz tick labels."""
import json,resource,time
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
fig,ax=plt.subplots(figsize=(11,5.8)); ax.axis('off')
ax.text(.04,.84,'Recorded source fields (different recorded variables)',fontsize=13,weight='bold')
ax.add_patch(plt.Rectangle((.04,.56),.43,.18,facecolor='#e8eef7',edgecolor='#345'))
ax.text(.06,.65,'s(t) = observation.arm[14] + hand[14] + leg[15]\n43 native numeric channels; unit/anatomy unresolved',va='center',fontsize=11)
ax.add_patch(plt.Rectangle((.53,.56),.43,.18,facecolor='#f7eee4',edgecolor='#853'))
ax.text(.55,.65,'u(t) = action[28]\nseparate recorded command field; no 28↔43 pairing',va='center',fontsize=11)
ax.annotate('',xy=(.49,.48),xytext=(.25,.55),arrowprops=dict(arrowstyle='->',lw=2,color='#345'))
ax.annotate('',xy=(.70,.48),xytext=(.74,.55),arrowprops=dict(arrowstyle='->',lw=2,color='#853'))
ax.text(.04,.39,'At t0: prefix inputs s[-1], s[0], u[0] (known-anchor contract assumption)',fontsize=10.5)
ax.plot([.06,.94],[.26,.26],color='black',lw=1.4); ax.scatter([.22,.36,.50,.64,.78,.92],[.26]*6,c='black')
for x,label in zip([.22,.36,.50,.64,.78,.92],['t0−1/30s','t0','t0+1/30s','t0+2/30s','…','t0+1s']): ax.text(x,.18,label,ha='center',fontsize=8.5)
ax.text(.04,.055,'One-step fit: s[0]→s[1] once per train anchor. HOLD uses u[0]; ORACLE uses u[0..29] (privileged); teacher resets to GT state each step (local only).',fontsize=9.3)
fig.tight_layout(); out=ROOT/'figures/FIG01_SOURCE_AND_CLOCK_AMEND001.png'; fig.savefig(out,dpi=180); plt.close(fig)
print(json.dumps({'status':'PASS_PRESENTATION_ONLY','output':str(out),'tick_labels':'explicit 1/30 second nominal grid','dev_labels_read':False,'model_or_metric_changed':False,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},sort_keys=True))
