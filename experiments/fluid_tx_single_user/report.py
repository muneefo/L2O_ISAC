"""Reports and plots strictly from executed spatial/geometry pilot artifacts."""
from pathlib import Path
import csv,os
import numpy as np
import torch
from l2o_isac.architecture_audit import read,write,means
from .run import Pilot,ROOT
from l2o_isac.solver import SolverState
from .spatial import system
from .geometry import steering


def paired(values,seed=910100):
    v=np.asarray(values,dtype=np.float64);rng=np.random.default_rng(seed)
    boot=v[rng.integers(0,len(v),size=(3000,len(v)))].mean(1)
    return {'mean':float(v.mean()),'channel_95pct':np.quantile(boot,[.025,.975]).tolist(),'std':float(v.std(ddof=1)),'samples':len(v)}


def main():
    torch.set_num_threads(1);pilot=Pilot();rows={};raw={};comparisons={};learning={};timing=read(pilot.out/'timing.json')['cases']
    settings=read(ROOT/'configs/learning.json');seeds=settings['seeds']
    for case in ['A','B','C']:
        rows[case]={};raw[case]={}
        for p in sorted((pilot.out/case).glob('*.pt')):
            r=torch.load(p,weights_only=True)
            if 'shared_union' not in r:continue
            raw[case][p.stem]=r;t=r.get('timing',timing[case].get(p.stem))
            rows[case][p.stem]={'nominal':means(r['nominal']),'terminal':means(r['terminal']),'shared_union':means(r['shared_union']),
              'budgets':{str(b):means(z) for b,z in r['attack_budget_metrics'].items()},'counts':r['counts'],'iterations':r['iterations'],
              'max_residuals':r['max_residuals'],'design_batch_seconds':r['design_batch_seconds'],'attack_seconds':r['attack_seconds'],
              'timing':t,'selected_epoch':r.get('selected_epoch'),'learned_parameter_count':r.get('learned_parameter_count')}
        f=raw[case]['Fluid_budget360']['shared_union']['objective'].numpy()
        u=raw[case]['Analytic_ULA_control' if case=='A' else 'ULA_retuned_budget360']['shared_union']['objective'].numpy()
        random=raw[case]['StaticRandom_budget360']['shared_union']['objective'].numpy()
        comparisons[case]={'fluid_vs_fixed':paired(f-u),'fluid_vs_static_random':paired(f-random),
                           'fixed_reference':'Analytic_ULA_control' if case=='A' else 'ULA_retuned_budget360'}
        base=raw[case]['Plain_tuned_control']['shared_union']['objective'].numpy()
        constant=np.stack([raw[case][f'LearnedConstant_seed{s}']['shared_union']['objective'].numpy() for s in seeds])
        schedule=np.stack([raw[case][f'LearnedSchedule_seed{s}']['shared_union']['objective'].numpy() for s in seeds])
        learning[case]={'constant_utility':float(constant.mean()),'schedule_utility':float(schedule.mean()),
             'schedule_minus_constant':paired((schedule-constant).mean(0)),'schedule_minus_fixed_plain':paired((schedule-base).mean(0)),
             'constant_per_seed':constant.mean(1).tolist(),'schedule_per_seed':schedule.mean(1).tolist(),
             'schedule_minus_constant_per_seed':(schedule-constant).mean(1).tolist(),
             'training_seed_gain_sd':float((schedule-constant).mean(1).std(ddof=1)),
             'selected_epochs':{v:[rows[case][f'{v}_seed{s}']['selected_epoch'] for s in seeds] for v in settings['variants']},
             'scope':'Paired channel interval conditional on trained runs; seed means/SD separate; not300 independent channels'}
    summary={'cases':rows,'geometry_comparisons':comparisons,'learning':learning,'scenarios_per_case':pilot.spec['test_scenarios_per_case'],
             'scope':'Fresh known spatial-field scenarios; independent implementation, no global robustness/publication/hardware-latency claim'}
    write(pilot.out/'summary.json',summary)
    with (pilot.out/'metrics.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['case','method','nominal_utility','comm','sensing_mi_metric','weighted_utility','full_gradient_calls','median_ms'])
        for case,rs in rows.items():
            for name,r in rs.items():w.writerow([case,name,r['nominal']['objective'],r['shared_union']['comm'],r['shared_union']['sense'],r['shared_union']['objective'],r['counts']['full_analytic_calls'],r['timing']['median_ms'] if r['timing'] else 'analytic reference, not timed'])
    lines=['# One-dimensional fluid transmit-array pilot','',
      '**Finding:** geometry optimization improves the independently attacked objective in the tested multipath regimes. Aligned LOS with no clutter has no intrinsic geometry gain: a feasible analytic fixed-ULA solution already reaches the geometry-independent optimum. Learning20 layer-specific position steps does not show a useful advantage over an equally trained single scalar.','',
      '## Isolation and model','',
      'This workflow lives entirely in experiments/fluid_tx_single_user/. The completed reproduction and controlled studies are pinned before this experiment; outputs/baseline_preservation.json hashes all767 pre-existing tracked files. Shared physics, gradients, original PGDA and independent attackers are imported unchanged. No old checkpoint is transferred to the new channel distribution or overwritten.','',
      'M=T1,Nt=Nr8,L4,communication plus dedicated sensing streams, SNR5dB,rho.8,noise1,common normalized uncertainty radii.05. Transmit aperture3.5lambda,min spacing.25lambda; half-wave ULA fits in the same aperture with genuine movement freedom. Coordinates are x=p/lambda; wavelength.01m converts units only. Receive ULA is fixed. All hardware/power/receive-filter constraints and norm-ball radii eps*sqrt(Nt/Nr) are retained. Sensing uses the inherited sum-of-clutter-powers approximation and natural-log MI metric; no coherent-clutter substitution or detection-accuracy claim.','',
      'The communication field is h(p)^H=sum beta_k*a_t(vartheta_k;p)^T, with fixed gains/angles while positions change. Stored h is conjugated consistently with the shared receiver equation. This is a necessary new spatial-channel assumption, **not** the paper’s position-independent CN(0,I) distribution. Fresh fixed-ULA and fluid baselines share identical spatial scenarios; no arbitrary old vector was fitted to a speculative spatial model, redrawn after movement, or normalized separately by position.','',
      'Cases A: unit-magnitude LOS user aligned with the target,J0; B: four fixed multipath gains with dominant user direction near-25deg and target near+25deg,J0; C: multipath,one target and one competing clutter. Clutter count is configurable and tests cover two clutter paths. Each has64 train/32 validation/100 test scenarios with disjoint saved seeds. Positions/weights are optimized per scenario. Random is fixed within a scenario but independently resampled across scenarios; **StaticRandom** is one draw(seed909999) held fixed across all splits/scenarios and is the primary fixed-random comparator. Both share aperture/endpoints/spacing.','',
      '## Classical result at360 full gradient-helper calls','',
      '| Case | Fixed ULA reference | Static random array | Joint fluid | Fluid minus ULA | Paired95% channel interval |','|---|---:|---:|---:|---:|---|']
    for case in ['A','B','C']:
        u=rows[case][comparisons[case]['fixed_reference']]['shared_union']['objective'];r=rows[case]['StaticRandom_budget360']['shared_union']['objective'];f=rows[case]['Fluid_budget360']['shared_union']['objective'];d=comparisons[case]['fluid_vs_fixed']
        lines.append(f"| {case} | {u:.6f} | {r:.6f} | {f:.6f} | {d['mean']:.6f} | [{d['channel_95pct'][0]:.6f},{d['channel_95pct'][1]:.6f}] |")
    lines+=['',
      'For B/C, ULA and StaticRandom use180 iterations and fluid120: all360 calls to the same helper that computes six variable-block gradients. A uses its analytic feasible MRT/MRC reference instead, not an iteratively degraded comparator. The100 paired channels, not a selected geometry, support the B/C observation. Scalar components and all shorter-budget points are in metrics.csv/summary.json.','',
      'Case A is a crucial negative control. Its analytic robust value is independent of positions under these idealized aligned/single-stream assumptions. The inherited finite PGDA updates can move away from that optimal initialization; apparent fluid recovery relative to a degraded finite-iteration ULA is an optimizer effect, **not intrinsic geometry benefit**. All A methods receive an exact feasible error candidate as an added validation control; this does not certify B/C searches.','',
      '## Matched derivative and runtime comparisons','',
      '| Case/method | Communication | Sensing MI metric | Utility | Full helper calls | Warm batch-one median/p95ms |','|---|---:|---:|---:|---:|---|']
    for case in ['A','B','C']:
        for name in ['ULA_budget60','StaticRandom_budget60','Fluid_budget60','ULA_runtime_match']:
            r=rows[case][name];z=r['shared_union'];t=r['timing'];lines.append(f"| {case}/{name} | {z['comm']:.6f} | {z['sense']:.6f} | {z['objective']:.6f} | {r['counts']['full_analytic_calls']} | {t['median_ms']:.2f}/{t['p95_ms']:.2f} |")
    lines+=['',
      'A fixed macro uses2 full analytic calls; a fluid macro uses3, with the third supplying communication/steering partials for the real position chain rule. Budgets60/180/360 mean fixed30/90/180 versus fluid20/60/120. Fluid also pays for field reconstruction, exact PAV projections and line-search metric evaluations. Counts and measured wall time are both reported; equal helper count is not equal FLOPs. A validation-timing-selected ULA budget is included near fluid20 runtime. It is approximate timing matching, not a guaranteed equal deadline.','',
      'BF coefficients are selected on validation with a3x3 grid, separately for ULA/random; fluid reuses ULA BF coefficients and tunes3 position steps. Strong ULA90/180 and StaticRandom budgets are separately retuned on validation. No test performance selects these coefficients. Positions use zero momentum and fixed-error Armijo backtracking; a surrogate improvement is not monotonic improvement after an independently reoptimized adversary. Long runs can lose utility, and those results remain saved.','',
      'Timing uses one CPU thread,complex64/float32,batch1,three warmups and first15 paired test scenarios, with raw times,median/p95. It includes gradients,field reconstruction,projection/backtracking and static learned-parameter access. Dataset construction and initialization are excluded. Independent attacks are timed separately. No antenna movement,RF programming,CSI acquisition,energy,mechanical feasibility or hardware-realistic latency is measured.','',
      '## Learned-position result: small schedules and equal-extra-training control','',
      'There is no compatible pretrained original single-user/spatial checkpoint. New position-only models retain the original physical MinNet/MaxNet structure with frozen validation-tuned **update coefficients**; the beamformer/error iterates still update. We train either one shared step or20 layer-specific positive steps, with position momentum0. No larger NN, EpsNet,RecNet or joint BF fine-tuning is introduced. Models are trained separately per regime; this is not one-model transfer across regimes.','',
      'Both learned variants start from the same validation-tuned no-backtracking constant step and use identical paired batch orders,64 training channels,four Adam epochs,lr.002,batch32,eight updates,three seeds. Each online solve has20 macros/60 full calls and20 position projections. Full derivatives pass through spatial responses,analytic gradient updates and PAV’s exact active-face Jacobian; no straight-through geometry derivative is used. The terminal10-step training attacker is detached, an approximate surrogate rather than an exact envelope/global-minimum derivative. Validation uses independent50-step search with initialization eligible. Actual training histories,optimizer/RNG states,source/data hashes and selected epochs are retained.','',
      '| Case | Fixed tuned plain step | Learned scalar | Learned20-step schedule | Schedule minus scalar | Selected epochs scalar / schedule |','|---|---:|---:|---:|---:|---|']
    for case in ['A','B','C']:
        l=learning[case];lines.append(f"| {case} | {rows[case]['Plain_tuned_control']['shared_union']['objective']:.6f} | {l['constant_utility']:.6f} | {l['schedule_utility']:.6f} | {l['schedule_minus_constant']['mean']:.8f} | {l['selected_epochs']['LearnedConstant']} / {l['selected_epochs']['LearnedSchedule']} |")
    lines+=['',
      '**Decision:** a20-parameter schedule is not justified by this pilot. In B, the small gain over a coarse fixed-step grid is reproduced by the equally trained single scalar. In C, several runs select the untrained initialization, and trained selections can worsen test utility. All seeds, including losses, remain in the averages. Learned20-step designs also do not recover the higher utility of longer classical fluid solves. Faster no-backtracking inference must not be credited to learning when the fixed no-backtracking control has the same overhead. This does not rule out all future learning methods; it does not establish learned acceleration here.','',
      'Channel intervals average paired differences over the three trained runs before resampling100 channel scenarios. Per-seed means/SD are reported separately;300 model/channel pairs are not300 independent test scenarios. Training wall times include epoch validation but exclude initial validation/grid setup; inference comparisons do not hide that offline cost.','',
      '## Validation, plots and limitations','',
      'The tests cover steering recovery,field repeatability,independent NumPy signal equations,transpose/conjugation/physical units,float64 position finite differences,projection against SciPy convex QPs and exhaustive tiny partitions,PAV first/second derivatives away from kinks,full-layer feasibility,disabled-position equivalence,frozen variables,zero-channel handling,short-unroll parameter finite differences and read-only original checkpoint regression. The initial disabled-position mismatch was a coefficient-initialization precision issue; matching the shared PGDA buffer convention fixed it without relaxing the equality test.','',
      'The independent attacker freezes geometry and beamformers, searches only the original separate error sets, includes zero/terminal/five random interior/boundary starts,50 Adam steps and nested50/200/1000 PGD. Best candidates are retained and every method receives a shared feasible candidate pool evaluated at its own frozen positions. Raw budget curves,complex states,positions,per-scenario metrics/residuals and independent NumPy checks are saved. Lowest utility found is an upper bound on the true minimum, not certified robust performance.','',
      'Position and beampattern diagnostics use scenario index0 fixed before inspection; they illustrate rather than prove the average gain. Patterns share the same Pt normalization (power/Pt), not each curve’s peak. Tradeoff results at rho.2,.5,.8 use the same100 spatial scenarios with validation retuning at each rho and matched60 gradient calls. They are observed tradeoffs, not a globally optimized Pareto frontier or measured sensing detection accuracy.','',
      'This is an idealized known-field-response experiment. Coupling,position errors,channel aging,movement cost/calibration overhead and hardware latency are not modeled. The original constant-radius uncertainty assumptions are retained rather than statistically calibrated. The angular/gain distributions and optimizer choices condition the results. More apertures/path laws,initializations,stronger BF optimization and independent larger studies are needed before broad/publishable claims. No novelty is claimed.','',
      '## Reproduce','',
      'Figures: [gradient budgets](outputs/pilot/utility_vs_gradient_budget.png), [runtime](outputs/pilot/utility_vs_runtime.png), [positions/common-reference patterns](outputs/pilot/positions_and_common_reference_patterns.png), [all100 geometries](outputs/pilot/all_scenario_positions.png), [tradeoff](outputs/pilot/communication_sensing_tradeoff.png), [attack saturation](outputs/pilot/attack_budget_saturation.png).', '',
      'Independent evidence replays every saved design using the final source with zero maximum difference and verifies all767 pre-existing tracked files are unchanged.', '',
      'See README.md for exact commands,configs,dependencies and the isolated STATUS.md. Primary raw data are outputs/pilot/{scenarios.pt,data_manifest.json,A/,B/,C/}; learning checkpoints are under outputs/pilot/learning/; tradeoff outputs under outputs/pilot/tradeoff/. Evidence verifies every pinned pre-existing file remains byte-identical.']
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    plot(pilot,raw,rows,learning,comparisons)
    print('Wrote REPORT.md, CSV, summary and measured-data figures')


def plot(pilot,raw,rows,learning,comparisons):
    os.environ.setdefault('MPLCONFIGDIR','/tmp/fluid-pilot-mpl');os.environ.setdefault('XDG_CACHE_HOME','/tmp/fluid-pilot-cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for ax,case in zip(axes,['A','B','C']):
        for method in ['ULA','StaticRandom','Fluid']:
            vals=[]
            for b in [60,180,360]:
                name=(('ULA_retuned' if b>60 else 'ULA') if method=='ULA' else method)+'_budget'+str(b)
                vals.append(rows[case][name]['shared_union']['objective'])
            ax.plot([60,180,360],vals,marker='o',label=method)
        if case=='A':ax.axhline(rows[case]['Analytic_ULA_control']['shared_union']['objective'],linestyle='--',color='black',label='Analytic fixed ULA')
        ax.set_title('Case '+case);ax.set_xlabel('Full six-block gradient calls');ax.set_ylabel('Lowest-found weighted utility');ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(pilot.out/'utility_vs_gradient_budget.png',dpi=180);fig.savefig(pilot.out/'utility_vs_gradient_budget.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for ax,case in zip(axes,['A','B','C']):
        for method in ['ULA','StaticRandom','Fluid']:
            points=[]
            for b in [60,180,360]:
                name=(('ULA_retuned' if b>60 else 'ULA') if method=='ULA' else method)+'_budget'+str(b);r=rows[case][name]
                points.append((r['timing']['median_ms'],r['shared_union']['objective']))
            ax.plot([v[0] for v in points],[v[1] for v in points],marker='o',label=method)
        for variant in ['LearnedConstant','LearnedSchedule']:
            rs=[rows[case][f'{variant}_seed{s}'] for s in [101,202,303]]
            ax.scatter(np.mean([r['timing']['median_ms'] for r in rs]),np.mean([r['shared_union']['objective'] for r in rs]),label=variant)
        ax.set_title(case);ax.set_xlabel('Warm batch-one computation ms');ax.set_ylabel('Lowest-found utility');ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(pilot.out/'utility_vs_runtime.png',dpi=180);fig.savefig(pilot.out/'utility_vs_runtime.pdf');plt.close(fig)
    fig,axes=plt.subplots(3,2,figsize=(12,10))
    for ci,case in enumerate(['A','B','C']):
        sc,_=pilot.data(case,'test');diag=dict(torch.load(pilot.out/case/'diagnostic.pt',weights_only=True)['variants'])
        final=raw[case]['Fluid_budget360'];diag['Joint_fluid_optimized']={'positions':final['positions'][:1],'state':{k:v[:1] for k,v in final['state'].items()}}
        ax=axes[ci,0]
        for name,r in diag.items():ax.plot(np.arange(pilot.spec['nt'])+1,r['positions'][0].numpy(),marker='o',label=name)
        ax.set_ylim(0,pilot.spec['aperture_wavelengths']);ax.set_ylabel(f'{case}: position / wavelength');ax.set_xlabel('Ordered antenna index');ax.legend(fontsize=6)
        ax=axes[ci,1];angle=torch.linspace(-80,80,641);radians=angle*torch.pi/180
        for name,r in diag.items():
            state=SolverState(**r['state']);response=steering(radians[None],r['positions']);power=(response@(state.a@state.d)).abs().square().sum(-1)[0]
            ax.plot(angle.numpy(),10*np.log10(np.maximum(power.numpy()/system(pilot.spec,case).power,1e-12)),label=name)
        ax.axvline(sc.sensing_angles[0,0].item()*180/np.pi,color='r',linestyle='--',linewidth=.7,label='Target')
        ax.axvline(sc.user_angles[0,0].item()*180/np.pi,color='g',linestyle=':',linewidth=.7,label='User dominant path')
        for a in sc.sensing_angles[0,1:]:ax.axvline(a.item()*180/np.pi,color='black',linestyle=':',linewidth=.7)
        ax.set_ylim(-45,12);ax.set_ylabel('Tx illumination/Pt (dB)');ax.set_xlabel('Departure angle (deg)');ax.legend(fontsize=6)
    fig.tight_layout();fig.savefig(pilot.out/'positions_and_common_reference_patterns.png',dpi=180);fig.savefig(pilot.out/'positions_and_common_reference_patterns.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for ax,case in zip(axes,['A','B','C']):
        x=raw[case]['Fluid_budget360']['positions'].numpy();im=ax.imshow(x,aspect='auto',vmin=0,vmax=pilot.spec['aperture_wavelengths']);ax.set_title(case+' all100 optimized geometries');ax.set_xlabel('Antenna index');ax.set_ylabel('Paired scenario')
    fig.colorbar(im,ax=list(axes),label='Position / wavelength');fig.savefig(pilot.out/'all_scenario_positions.png',dpi=180,bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for ax,case in zip(axes,['A','B','C']):
        for name in ['ULA_budget60','Fluid_budget60','ULA_retuned_budget360','Fluid_budget360']:
            ax.plot([50,200,1000],[rows[case][name]['budgets'][str(b)]['objective'] for b in [50,200,1000]],marker='o',label=name)
        ax.set_title(case);ax.set_xlabel('PGD steps/start (+50 Adam)');ax.set_ylabel('Independent lowest-found utility');ax.legend(fontsize=6)
    fig.tight_layout();fig.savefig(pilot.out/'attack_budget_saturation.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    tradeoff={}
    for ax,case in zip(axes,['A','B','C']):
        for method in ['ULA','Fluid']:
            points=[]
            for rho in [.2,.5,.8]:
                r=torch.load(pilot.out/'tradeoff'/f'{case}_rho{rho}.pt',weights_only=True);m=means(r['methods'][method]['shared_union']);points.append(m);tradeoff[f'{case}/{method}/{rho}']=m
            ax.plot([q['comm'] for q in points],[q['sense'] for q in points],marker='o',label=method)
            for rho,q in zip([.2,.5,.8],points):ax.annotate(str(rho),(q['comm'],q['sense']),fontsize=7)
        ax.set_title(case);ax.set_xlabel('Communication sum-rate (nats)');ax.set_ylabel('Sensing MI upper-bound metric (nats)');ax.legend()
    fig.tight_layout();fig.savefig(pilot.out/'communication_sensing_tradeoff.png',dpi=180);fig.savefig(pilot.out/'communication_sensing_tradeoff.pdf');plt.close(fig)
    write(pilot.out/'tradeoff_summary.json',tradeoff)


if __name__=='__main__':main()
