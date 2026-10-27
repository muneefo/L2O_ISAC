"""Generate paired measured tables and scientific figures; never tune test data."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/fluid-l2o-mpl')
from pathlib import Path
import csv,json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .run import Study,ROOT,write
from experiments.fluid_tx_single_user.geometry import steering

def interval(v):
 v=np.asarray(v,dtype=float);g=np.random.default_rng(929100);draws=v[g.integers(0,len(v),(4000,len(v)))].mean(1)
 return {'mean':float(v.mean()),'paired_channel_95pct':np.quantile(draws,[.025,.975]).tolist(),'n_scenarios':len(v)}

def main():
 torch.set_num_threads(1);s=Study();out=s.out;summary={'cases':{},'paired_comparisons':{},'training':{},'scope':'Independent shared-union attack estimates; not certified minima. Seeds are paired training shuffles, not additional independent test channels.'};raw={};rows=[]
 for c in 'ABC':
  raw[c]={};summary['cases'][c]={}
  for path in sorted((out/c).glob('*.pt')):
   r=torch.load(path,weights_only=True);raw[c][path.stem]=r
   z={k:float(v.mean()) for k,v in r['shared_union'].items()};summary['cases'][c][path.stem]={'attacked':z,'terminal':{k:float(v.mean()) for k,v in r['terminal'].items()},'counts':r['counts'],'timing':r.get('timing'),'max_residual':max(r['residuals'].values()),'attack_seconds':r['attack_seconds']}
   for i in range(len(r['positions'])):
    row={'case':c,'method':path.stem,'scenario':i,**{k:float(v[i]) for k,v in r['shared_union'].items()},**{f'terminal_{k}':float(v[i]) for k,v in r['terminal'].items()},**{f'attack_{b}_{k}':float(v[i]) for b,z in r['attack_metrics'].items() for k,v in z.items()},**{f'x{n}':float(v) for n,v in enumerate(r['positions'][i])}}
    rows.append(row)
  learned={kind:np.stack([raw[c][f'{kind}_s{seed}']['shared_union']['objective'].numpy() for seed in s.spec['seeds']]) for kind in ['FixedL2O','FluidL2O']}
  comparisons={'FluidL2O_minus_FixedL2O':learned['FluidL2O'].mean(0)-learned['FixedL2O'].mean(0),'FluidL2O_minus_classical_Fluid60':learned['FluidL2O'].mean(0)-raw[c]['Fluid_G60']['shared_union']['objective'].numpy(),'FluidL2O_minus_untrained':learned['FluidL2O'].mean(0)-raw[c]['FluidUntrained']['shared_union']['objective'].numpy(),'FixedL2O_minus_classical_ULA40':learned['FixedL2O'].mean(0)-raw[c]['ULA_G40']['shared_union']['objective'].numpy(),'Fluid_classical60_minus_ULA60':raw[c]['Fluid_G60']['shared_union']['objective'].numpy()-raw[c]['ULA_G60']['shared_union']['objective'].numpy(),'FluidL2O_minus_no_position_momentum_seed401':learned['FluidL2O'][0]-raw[c]['FluidNoPositionMomentum_s401']['shared_union']['objective'].numpy()}
  for family in ['ULA','Fluid']:
   comparisons['FluidL2O_minus_'+family+'_runtime_match']=learned['FluidL2O'].mean(0)-raw[c][family+'_time_FluidL2O']['shared_union']['objective'].numpy()
  summary['paired_comparisons'][c]={k:interval(v) for k,v in comparisons.items()}
  summary['training'][c]={}
  for f in sorted((out/'training'/c).glob('*/complete.json')):summary['training'][c][f.parent.name]=json.loads(f.read_text())
 with (out/'per_scenario.csv').open('w') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
 write(out/'summary.json',summary)
 plt.rcParams.update({'font.family':'serif','font.size':8,'pdf.fonttype':42,'ps.fonttype':42})
 fig,axes=plt.subplots(1,2,figsize=(6.7,2.5))
 for ax,c in zip(axes,'BC'):
  for family,color,marker in [('ULA','#333333','o'),('Fluid','#D55E00','^')]:
   vals=[raw[c][family+f'_G{g}'] for g in s.spec['classical_budgets']]
   ax.plot([r['counts']['full_analytic_calls'] for r in vals],[r['shared_union']['objective'].mean() for r in vals],marker=marker,color=color,label='Classical '+family)
  for kind,color,marker in [('FixedL2O','#0072B2','s'),('FluidL2O','#009E73','D')]:
   vals=[raw[c][f'{kind}_s{seed}'] for seed in s.spec['seeds']];ax.scatter([vals[0]['counts']['full_analytic_calls']],[np.mean([r['shared_union']['objective'].mean() for r in vals])],marker=marker,color=color,label=kind,zorder=5)
  ax.set(title='Case '+c,xlabel='Full six-block gradient calls',ylabel='Independently attacked utility');ax.grid(color='#dddddd');ax.legend(fontsize=6)
 fig.tight_layout();fig.savefig(out/'utility_vs_gradient_budget.pdf');fig.savefig(out/'utility_vs_gradient_budget.png',dpi=220);plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(6.7,2.5))
 for ax,c in zip(axes,'BC'):
  for family,color,marker in [('ULA','#333333','o'),('Fluid','#D55E00','^')]:
   vals=[raw[c][family+f'_G{g}'] for g in s.spec['classical_budgets']]
   ax.plot([r['timing']['median_ms'] for r in vals],[r['shared_union']['objective'].mean() for r in vals],marker=marker,color=color,label='Classical '+family)
  for kind,color,marker in [('FixedL2O','#0072B2','s'),('FluidL2O','#009E73','D')]:
   vals=[raw[c][f'{kind}_s{seed}'] for seed in s.spec['seeds']];ax.scatter([np.mean([r['timing']['median_ms'] for r in vals])],[np.mean([r['shared_union']['objective'].mean() for r in vals])],marker=marker,color=color,label=kind,zorder=5)
  ax.set(title='Case '+c,xlabel='Online initialize + design (ms)',ylabel='Independently attacked utility');ax.grid(color='#dddddd');ax.legend(fontsize=6)
 fig.tight_layout();fig.savefig(out/'utility_vs_runtime.pdf');fig.savefig(out/'utility_vs_runtime.png',dpi=220);plt.close(fig)
 fig,axes=plt.subplots(1,3,figsize=(9,2.6))
 for ax,c in zip(axes,'ABC'):
  for label in ['ULA_G60','Fluid_G60','FixedL2O_s401','FluidL2O_s401']:
   r=raw[c][label];ax.plot(s.spec['test_attack_budgets'],[float(r['attack_metrics'][b]['objective'].mean()) for b in s.spec['test_attack_budgets']],marker='o',label=label)
  ax.set(xscale='log',title='Case '+c,xlabel='Independent attack steps',ylabel='Lowest utility found');ax.grid(color='#dddddd');ax.legend(fontsize=5)
 fig.tight_layout();fig.savefig(out/'attack_budget_saturation.pdf');fig.savefig(out/'attack_budget_saturation.png',dpi=200);plt.close(fig)
 # Scene0 fixed before inspection; all powers share the Pt reference.
 fig,axes=plt.subplots(2,2,figsize=(6.7,4.3));angle=torch.linspace(-np.pi/2,np.pi/2,361)
 for col,c in enumerate('BC'):
  sc,_=s.data(c,'test')
  for j,(label,color) in enumerate([('ULA_G60','#333333'),('Fluid_G60','#D55E00'),('FluidL2O_s401','#009E73')]):
   r=raw[c][label];x=r['positions'][0];v=r['state']['a'][0]@r['state']['d'][0];at=steering(angle,x);power=(at@v).abs().square().sum(-1)/s.cfg[c].power
   axes[0,col].scatter(x.numpy(),np.full(len(x),j),s=12,label=label,color=color)
   axes[1,col].plot(angle.numpy()*180/np.pi,10*np.log10(power.numpy().clip(1e-8)),label=label,color=color)
  axes[0,col].set(title='Case '+c+', scene0',xlabel='Position / wavelength',yticks=[0,1,2],yticklabels=['ULA','Classical fluid','L2O fluid'],xlim=(-.1,3.6));axes[0,col].grid(color='#dddddd')
  for l,t in enumerate(sc.sensing_angles[0]):axes[1,col].axvline(float(t*180/np.pi),ls='--',color='red' if l==0 else 'purple',lw=.8)
  axes[1,col].set(xlabel='Broadside angle (deg)',ylabel='Transmit power / P (dB)',ylim=(-35,10));axes[1,col].grid(color='#dddddd');axes[1,col].legend(fontsize=5)
 fig.tight_layout();fig.savefig(out/'positions_common_reference_patterns.pdf');fig.savefig(out/'positions_common_reference_patterns.png',dpi=200);plt.close(fig)
 # Measured report: exact table values and contrasts, no assumed successes.
 lines=['# Coupled fluid-transmit L2O: measured study','',summary['scope'],'','## Main comparisons','', '|Case/method|Comm|Sensing|Utility|Full calls|Median ms|','|---|---:|---:|---:|---:|---:|']
 for c in 'ABC':
  for label in ['ULA_G60','Fluid_G60','FixedL2O_s401','FluidL2O_s401','FluidUntrained','FluidNoPositionMomentum_s401','TrainOptimizedFixed','StaticRandom','ULA_G360','Fluid_G360']:
   r=summary['cases'][c][label];v=r['attacked'];lines.append(f"|{c}/{label}|{v['comm']:.6f}|{v['sense']:.6f}|{v['objective']:.6f}|{r['counts']['full_analytic_calls']}|{r['timing']['median_ms']:.2f}|")
 lines+=['','## Paired channel differences','', 'L2O contrasts average the three trained runs before resampling100 paired scenarios. These intervals condition on this training sample and the three fitted models.','']
 for c in 'ABC':
  for name,d in summary['paired_comparisons'][c].items():lines.append(f"- {c}/{name}: {d['mean']:+.6f}, paired95% {d['paired_channel_95pct']}.")
 lines+=['','## Training and interpretation','','Inherited Eq22 loss, full end-to-end derivatives, direct unconstrained coefficients, SGD1e-4, decay.98,10epochs,128 train/32 validation/100 test scenarios per regime;40 optimizer updates per model. Fixed models have240 active scalar parameters, fluid280, no-position-momentum260. Three paired shuffle seeds401/402/403 for the primary fixed/fluid models; the momentum ablation uses401 only. No original checkpoint is transferred.','', 'Classical settings are validation-tuned. Fresh split seeds926101/926102/926103 (+case offsets) differ from the earlier fluid pilot. Epoch0 is eligible under the inherited validation layer loss; an epoch0-selected model cannot support a claim of learning improvement. Negative or unsuccessful outcomes remain in all aggregates.','', 'Matched runtime uses validation-scene timing only to select nearby classical iteration counts; reported test medians/p95/raw samples reveal actual mismatch. Same helper budget is not equal FLOPs. Online times include initialization and full forward; attacks/training separate. Physical movement, CSI acquisition, RF programming and hardware latency are unmeasured.','', 'The aligned LOS analytic reference is geometry invariant. Its exact adverse candidate is included where the single active stream condition holds. This is not a certificate for multipath/clutter. Weighted improvement at rho=.8 is not a Pareto frontier or a minimum QoS guarantee.','', 'See summary.json,per_scenario.csv,training/*/*/complete.json and raw PT outputs for all seeds, selected epochs, costs, parameters, attack budgets and residuals. The classical geometry gains in the earlier pilot are not relabeled as L2O gains here. Wider physical/configuration/initialization studies and competitive cross-literature baselines remain unresolved.']
 for c in 'ABC':
  lines.append('\n### '+c+' selection and seed coverage')
  for name,t in summary['training'][c].items():lines.append(f"- {name}: selected epoch{t['selected_epoch']}; {t['updates']} updates; {t['training_seconds']:.2f}s; mu changes {t['mu_block_changes']}.")
 (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')
 print('Saved report, raw scenario CSV, paired comparisons and figures')
if __name__=='__main__':main()
