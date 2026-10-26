"""Regenerate manuscript assets from saved pilot summaries; no optimization runs."""
from pathlib import Path
import hashlib,json,os
os.environ.setdefault("MPLCONFIGDIR","/tmp/wcl-mpl")
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
DATA=ROOT/'experiments/fluid_tx_single_user/outputs/pilot'
s=json.loads((DATA/'summary.json').read_text())
t=json.loads((DATA/'tradeoff_summary.json').read_text())
plt.rcParams.update({'font.family':'serif','font.size':8,'axes.labelsize':8,'legend.fontsize':7,'lines.linewidth':1.1,'pdf.fonttype':42,'ps.fonttype':42})
rows=[]
for case in 'ABC':
 ref=s['geometry_comparisons'][case]['fixed_reference']
 names=[ref,'StaticRandom_budget360','Fluid_budget360']
 vals=[s['cases'][case][n]['shared_union']['objective'] for n in names]
 ci=s['geometry_comparisons'][case]['fluid_vs_fixed']['channel_95pct']
 rows.append(f"{case} & {vals[0]:.4f} & {vals[1]:.4f} & {vals[2]:.4f} & [{ci[0]:.4f}, {ci[1]:.4f}] \\\\")
(HERE/'table_geometry.tex').write_text('\n'.join(rows)+'\n'+r'\bottomrule'+'\n')
rows=[]
for case in 'BC':
 for method,label in [('ULA_budget60','ULA'),('Fluid_budget60','Fluid')]:
  d=s['cases'][case][method];v=d['shared_union'];tm=d['timing']
  rows.append(f"{case}/{label} & {v['comm']:.4f} & {v['sense']:.4f} & {v['objective']:.4f} & {tm['median_ms']:.1f}/{tm['p95_ms']:.1f} \\\\")
(HERE/'table_short.tex').write_text('\n'.join(rows)+'\n'+r'\bottomrule'+'\n')
rows=[]
for case in 'ABC':
 l=s['learning'][case];plain=s['cases'][case]['Fluid_plain_learning_control']['shared_union']['objective']
 rows.append(f"{case} & {plain:.6f} & {l['constant_utility']:.6f} & {l['schedule_utility']:.6f} \\\\")
(HERE/'table_learning.tex').write_text('\n'.join(rows)+'\n'+r'\bottomrule'+'\n')
fig,axes=plt.subplots(1,2,figsize=(3.5,1.9),sharey=False)
for ax,case in zip(axes,'BC'):
 for base,label,marker,color in [('ULA','Fixed ULA','o','#333333'),('StaticRandom','Fixed random','s','#0072B2'),('Fluid','Fluid','^','#D55E00')]:
  values=[]
  for budget in [60,180,360]:
   key=f'{base}_budget{budget}'
   if base=='ULA' and budget!=60:key=f'ULA_retuned_budget{budget}'
   values.append(s['cases'][case][key]['shared_union']['objective'])
  ax.plot([60,180,360],values,marker=marker,markersize=3,label=label,color=color)
 ax.set(title=f'Case {case}',xlabel='Full gradient calls',ylabel='Attacked utility' if case=='B' else '')
 ax.grid(color='#dddddd');ax.set_xticks([60,180,360])
axes[0].legend(loc='lower right',frameon=False)
fig.tight_layout(pad=.4,w_pad=.7)
for ext in ['pdf','png','eps']:fig.savefig(HERE/f'figures/budgets.{ext}',dpi=250)
plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(3.5,1.9))
for ax,case in zip(axes,'BC'):
 for method,marker,color in [('ULA','o','#333333'),('Fluid','^','#D55E00')]:
  vals=[t[f'{case}/{method}/{r}'] for r in [.2,.5,.8]]
  ax.plot([v['comm'] for v in vals],[v['sense'] for v in vals],marker=marker,markersize=3,label=method,color=color)
  for v,r in zip(vals,[.2,.5,.8]):
   ax.annotate(str(r),(v['comm'],v['sense']),xytext=(2,3 if r!=.2 else -9),textcoords='offset points',fontsize=6)
 ax.set(title=f'Case {case}',xlabel='Communication (nats/s/Hz)',ylabel='Sensing metric' if case=='B' else '')
 ax.grid(color='#dddddd')
axes[0].legend(frameon=False,fontsize=6)
fig.tight_layout(pad=.4,w_pad=.6)
for ext in ['pdf','png','eps']:fig.savefig(HERE/f'figures/tradeoff.{ext}',dpi=250)
plt.close(fig)
inputs={str((DATA/f).relative_to(ROOT)):hashlib.sha256((DATA/f).read_bytes()).hexdigest() for f in ['summary.json','tradeoff_summary.json','validation.json']}
(HERE/'asset_provenance.json').write_text(json.dumps({'data_commit':'dd842239b2afbda03d9b3b6d6742c4e07ef5dc81','inputs_sha256':inputs,'note':'Reported utilities use shared-union independent attacks. No new training or test tuning.'},indent=2)+'\n')
