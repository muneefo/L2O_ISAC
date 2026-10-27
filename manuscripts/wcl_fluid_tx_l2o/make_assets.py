"""Letter assets from the newly executed coupled study only."""
from pathlib import Path
import os,json,hashlib
os.environ.setdefault('MPLCONFIGDIR','/tmp/fluid-l2o-letter-mpl')
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
H=Path(__file__).resolve().parent;R=H.parents[1];D=R/'experiments/fluid_tx_l2o/outputs/study';S=json.loads((D/'summary.json').read_text());seeds=[401,402,403]
def model_mean(c,kind,metric='objective'):return float(np.mean([S['cases'][c][f'{kind}_s{seed}']['attacked'][metric] for seed in seeds]))
def rows_file(name,rows):(H/name).write_text('\n'.join(rows)+'\n'+r'\bottomrule'+'\n')
rows=[]
for c in 'ABC':
 z=S['cases'][c]
 vals=[z['ULA_G40']['attacked']['objective'],model_mean(c,'FixedL2O'),z['Fluid_G60']['attacked']['objective'],model_mean(c,'FluidL2O')]
 rows.append(c+' & '+' & '.join(f'{v:.5f}' for v in vals)+r' \\')
rows_file('table_main.tex',rows)
rows=[]
for c in 'ABC':
 d=S['paired_comparisons'][c]['FluidL2O_minus_untrained'];ci=d['paired_channel_95pct'];u=S['cases'][c]['FluidUntrained']['attacked']['objective']
 interval_text=(f"$[{ci[0]*1e5:.2f},{ci[1]*1e5:.2f}]\\times10^{{-5}}$" if c=="A" else f"$[{ci[0]*1e8:.2f},{ci[1]*1e8:.2f}]\\times10^{{-8}}$" if c=="C" else f"[{ci[0]:+.5f}, {ci[1]:+.5f}]")
 rows.append(f"{c} & {u:.5f} & {model_mean(c,'FluidL2O'):.5f} & {interval_text}"+r' \\')
rows_file('table_learning.tex',rows)
rows=[]
for c in 'BC':
 for kind in ['FixedL2O','FluidL2O']:
  vals=[S['cases'][c][f'{kind}_s{seed}'] for seed in seeds]
  ts=[v['timing']['median_ms'] for v in vals];p95=[v['timing']['p95_ms'] for v in vals]
  rows.append(f"{c}/{kind.replace('L2O','')} & {model_mean(c,kind,'comm'):.4f} & {model_mean(c,kind,'sense'):.4f} & {np.mean(ts):.1f}/{np.mean(p95):.1f}"+r' \\')
rows_file('table_components.tex',rows)
plt.rcParams.update({'font.family':'serif','font.size':8,'axes.labelsize':8,'legend.fontsize':6,'pdf.fonttype':42,'ps.fonttype':42})
fig,axes=plt.subplots(2,2,figsize=(6.8,3.6))
for row,c in enumerate('BC'):
 for col,timing in enumerate([False,True]):
  ax=axes[row,col]
  for family,color,marker in [('ULA','#333333','o'),('Fluid','#D55E00','^')]:
   vals=[S['cases'][c][family+f'_G{g}'] for g in [40,60,180,360]]
   ax.plot([v['timing']['median_ms'] if timing else v['counts']['full_analytic_calls'] for v in vals],[v['attacked']['objective'] for v in vals],color=color,marker=marker,markersize=3,label='Classical '+family)
  for kind,color,marker in [('FixedL2O','#0072B2','s'),('FluidL2O','#009E73','D')]:
   vals=[S['cases'][c][f'{kind}_s{seed}'] for seed in seeds]
   xx=np.mean([v['timing']['median_ms'] for v in vals]) if timing else vals[0]['counts']['full_analytic_calls']
   ax.scatter([xx],[model_mean(c,kind)],color=color,marker=marker,s=18,label=kind,zorder=5)
  ax.set(title=f'Case {c}',xlabel='Initialize + design (ms)' if timing else 'Full gradient calls',ylabel='Attacked utility');ax.grid(color='#dddddd')
  if row==0 and col==0:ax.legend(loc='lower right')
fig.tight_layout(pad=.6,h_pad=.7,w_pad=.9)
for ext in ['pdf','png','eps']:fig.savefig(H/f'figures/compute.{ext}',dpi=240)
plt.close(fig)
# Numerical macros are deliberately generated from fresh measurements.
macros={}
for c in 'ABC':
 macros[c+'LearnGain']=S['paired_comparisons'][c]['FluidL2O_minus_untrained']['mean']
 macros[c+'FixedMean']=model_mean(c,'FixedL2O');macros[c+'FluidMean']=model_mean(c,'FluidL2O')
 macros[c+'FluidLong']=S['cases'][c]['Fluid_G360']['attacked']['objective']
 macros[c+'ClassGain']=S['paired_comparisons'][c]['Fluid_classical60_minus_ULA60']['mean']
 for family in ['ULA','Fluid']:
  macros[c+family+'RuntimeGap']=S['paired_comparisons'][c]['FluidL2O_minus_'+family+'_runtime_match']['mean']
macros['ALOS']=S['cases']['A']['AnalyticULA']['attacked']['objective']
(H/'numbers.tex').write_text('\n'.join('\\newcommand{\\'+k+'}{'+f'{v:.5f}'+'}' for k,v in {k:(0. if abs(v)<.000005 else v) for k,v in macros.items()}.items())+'\n')
(H/'asset_provenance.json').write_text(json.dumps({'summary_sha256':hashlib.sha256((D/'summary.json').read_bytes()).hexdigest(),'data_sha256':hashlib.sha256((D/'scenarios.pt').read_bytes()).hexdigest(),'scope':'New coupled study only; no earlier pilot numerical values reused'},indent=2))
print('Generated measured manuscript assets')
