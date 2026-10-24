"""Controlled ablation tables, crossed paired intervals, raw-data plots and manifests."""
import argparse,hashlib,json,os,subprocess
from pathlib import Path
import numpy as np
import torch
from .architecture_audit import read,write,sha,means


def bootstrap_contrast(values,rng):
    # P,F,N: pretrained models / paired fine-tuning runs / shared channel scenarios.
    p,f,n=values.shape;hier=[];conditional=[];pretrained=[]
    for _ in range(3000):
        ps=rng.integers(0,p,p);ns=rng.integers(0,n,n)
        rows=[values[pi,rng.integers(0,f,f)][:,ns].mean() for pi in ps]
        hier.append(float(np.mean(rows)));conditional.append(float(values.mean((0,1))[ns].mean()))
        pretrained.append(float(values.mean((1,2))[ps].mean()))
    return {'mean':float(values.mean()),'per_pretraining_seed_mean':values.mean((1,2)).tolist(),
            'pretraining_mean_sd':float(values.mean((1,2)).std(ddof=1)) if p>1 else None,
            'per_pretraining_finetuning_mean_sd':values.mean(2).std(axis=1,ddof=1).tolist() if f>1 else [None]*p,
            'crossed_hierarchical_95pct':np.quantile(hier,[.025,.975]).tolist(),
            'conditional_channel_95pct':np.quantile(conditional,[.025,.975]).tolist(),
            'pretraining_only_95pct':np.quantile(pretrained,[.025,.975]).tolist(),
            'scope':'Only three pretrained models; exploratory bootstrap. Shared channels resampled jointly across models, fine-tuning runs paired within each contrast.'}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',default='configs/pilot/controlled_ablation.json');parser.add_argument('--output',default='results/controlled_ablation');a=parser.parse_args(argv)
    torch.set_num_threads(1);spec=read(a.config);out=Path(a.output);rows={};arrays={};training={};per_seed={};failures=0;worst_residual=0.;rng=np.random.default_rng(99181)
    pretraining=spec['pretraining_seeds'];fine=spec['paired_finetuning_seeds'];labels={'A':'A','B':'B_balanced','C':'C_balanced','D':'D'}
    for pre in pretraining:
        folder=out/f'pretrain{pre}'/'evaluation';per_seed[str(pre)]={};arrays[pre]={}
        for condition,label in labels.items():
            methods=['A'] if condition=='A' else [f'{label}_s{s}' for s in fine]
            data=[]
            for name in methods:
                p=folder/(name+'.pt');r=torch.load(p,weights_only=True)
                if 'shared_union' not in r:raise RuntimeError('Union missing')
                data.append({k:v.numpy() for k,v in r['shared_union'].items()})
                key=f'{pre}/{name}';failure=int(r['failure_flags'].sum());failures+=failure
                residual=max(v.max().item() for v in r['feasibility_per_scenario'].values());worst_residual=max(worst_residual,residual)
                rows[key]={'independent_200':means(r['budget_metrics'][200]),'independent_1000':means(r['budget_metrics'][1000]),'shared_union':means(r['shared_union']),
                           'nominal':means(r['nominal']),'terminal':means(r['terminal']),'attack_seconds':r['attack_seconds'],
                           'gradient_calls':r['design_full_analytic_calls'],'max_residual':residual,'failures':failure}
                if condition!='A':
                    t=torch.load(out/f'pretrain{pre}'/name/'selected.pt',weights_only=True);last=torch.load(out/f'pretrain{pre}'/name/'latest.pt',weights_only=True)
                    training[key]={'selected_epoch':t['epoch'],'counts':last['counts'],'update_seconds':last['training_update_seconds'],'initial_validation':t['initial_validation'],
                                   'best_validation':t['best_validation'],'cycles_target':t['cycles_per_minibatch']}
            arrays[pre][condition]={k:np.stack([d[k] for d in data]) for k in ['comm','sense','objective']}
            if condition=='A':arrays[pre][condition]={k:np.repeat(v,len(fine),axis=0) for k,v in arrays[pre][condition].items()}
            per_seed[str(pre)][condition]={k:float(v.mean()) for k,v in arrays[pre][condition].items()}
        # Verify paired epoch permutations, identical positive initialization C/D,
        # unchanged momentum mapping and same starting checkpoint hash.
        for seed in fine:
            cb=torch.load(out/f'pretrain{pre}'/f'B_balanced_s{seed}'/'latest.pt',weights_only=True)
            cc=torch.load(out/f'pretrain{pre}'/f'C_balanced_s{seed}'/'latest.pt',weights_only=True)
            dd=torch.load(out/f'pretrain{pre}'/f'D_s{seed}'/'latest.pt',weights_only=True)
            assert cb['reference_sha256']==cc['reference_sha256']==dd['reference_sha256']
            assert [h['batch_order_sha256'] for h in cb['history']]==[h['batch_order_sha256'] for h in cc['history']]==[h['batch_order_sha256'] for h in dd['history']]
            ci=torch.load(out/f'pretrain{pre}'/f'C_balanced_s{seed}'/'initial.pt',weights_only=True)['model']
            di=torch.load(out/f'pretrain{pre}'/f'D_s{seed}'/'initial.pt',weights_only=True)['model']
            for k in ci:torch.testing.assert_close(ci[k],di[k],atol=0,rtol=0)
    contrasts={}
    for name,left,right in [('B_minus_A','B','A'),('C_minus_B','C','B'),('D_minus_C','D','C'),('D_minus_A','D','A')]:
        contrasts[name]={}
        for metric in ['comm','sense','objective']:
            values=np.stack([arrays[p][left][metric]-arrays[p][right][metric] for p in pretraining])
            contrasts[name][metric]=bootstrap_contrast(values,rng)
    mapping={}
    for pre in pretraining:
        r=torch.load(out/f'pretrain{pre}'/'evaluation/Mapped_initial.pt',weights_only=True)
        mapping[str(pre)]={'shared_union':means(r['shared_union']), 'gain_over_A':{k:float(r['shared_union'][k].mean())-per_seed[str(pre)]['A'][k] for k in ['comm','sense','objective']}}
    # Export every supplemental condition and saved original E, without treating E
    # as D or as an independent pretrained model.
    supplemental={}
    for p in sorted((out/f'pretrain{pretraining[0]}'/'evaluation').glob('*.pt')):
        r=torch.load(p,weights_only=True)
        if 'shared_union' in r:supplemental[p.stem]={'shared_union':means(r['shared_union']),'independent_200':means(r['budget_metrics'][200]),'independent_1000':means(r['budget_metrics'][1000])}
    inner={}
    for pre in pretraining:
        for p in (out/f'pretrain{pre}'/'fixed_beam_inner_validation').glob('*.pt'):
            r=torch.load(p,weights_only=True);inner[f'{pre}/{p.stem}']={'numpy_policy':means(r['numpy_policy_metrics']),'independent_refined':means(r['independent_refinement_metrics'])}
    pgda={}
    for p in (out/'pgda').glob('budget*.pt'):
        r=torch.load(p,weights_only=True);pgda[p.stem]={'shared_union':means(r['shared_union']),'independent_200':means(r['budget_metrics'][200]),'independent_1000':means(r['budget_metrics'][1000]),
          'nominal':means(r['nominal']),'full_analytic_calls':r['design_full_analytic_calls'],'attack_seconds':r['attack_seconds'],'max_residual':max(v.max().item() for v in r['feasibility_per_scenario'].values()),'failures':int(r['failure_flags'].sum())}
    report={'per_pretraining_seed':per_seed,'contrasts':contrasts,'mapping':mapping,'primary_rows':rows,'training':training,'supplemental_original_reference':supplemental,
            'inner_frozen_beam_validation':inner,'pgda':pgda,'independent_pretrained_references':len(pretraining),'paired_finetuning_seeds':fine,
            'failure_count':failures,'max_primary_feasibility_residual':worst_residual,'source_pdf_available':False,'scope':'Controlled exploratory study of independent implementation; not author-code verification or certified worst-case performance'}
    write(out/'summary.json',report)
    for condition in ['A','B','C','D']:
        report[condition+'_mean']={m:float(np.stack([arrays[p][condition][m] for p in pretraining]).mean()) for m in ['comm','sense','objective']}
    write(out/'summary.json',report)
    timing=read(out/'timing.json');counts={c:{k:float(np.mean([x['counts'][k] for n,x in training.items() if n.split('/')[1].startswith(labels[c]+'_')])) for k in next(iter(training.values()))['counts']} for c in ['B','C','D']}
    times={c:[x['update_seconds'] for n,x in training.items() if n.split('/')[1].startswith(labels[c]+'_')] for c in ['B','C','D']}
    lines=['# Controlled training and computation audit','',
      'This study continues the completed independent reproduction and adversarial audit; the original checkpoint and prior outputs remain unchanged. It distinguishes extra training, positive inner steps and separated roles. The source PDF is **not present** in the accessible workspace/attachments; `/workspace/reports/paper_extracted.txt` and earlier source records exist. Reattachment is requested. No new paper-equation verification or verified author-code claim is made.','',
      '## Preserved evidence and design','',
      'Pre-study source `73159d4`; immutable full archive `/workspace/evidence_archives/pre_controls_73159d4.zip` is hashed in results/controlled_ablation/preservation.json. Prior saved values were verified: terminal4.66515, independent reference4.03977, earlier separated4.09496, tunedPGDA1804.98123. These concern our implementation. Earlier three fine-tuning seeds share one pretrained reference.','',
      'Now use **three independently pretrained corrected-reference seeds73129,74129,75129**, each with paired fine-tuning seeds101,202,303. New references reuse the unchanged established `l2o_isac.experiment --profile paper` pipeline,5000/1000/1000 channels and10epochs, in fresh directories. Earlier checkpoints were inventoried; remaining ones use legacy EpsNet layouts, different depths, parameterization or array dimensions and are not silently counted as matched corrected references.','',
      'The physical problem and saved audit splits remain fixed: Nt=Nr16,M4,T2,J2,L8,rho.8,SNR5dB,oracle radii.05;256 train/64 validation/128 test channels. Same channel tensors, per-scenario initial states, paired epoch permutations, complex64, three continuation epochs, SGD lr1e-4 with decay.98. Per-case verification asserts identical C/D mapped initial tensors and B/C/D batch-order hashes. No CRB/controller or physical-model change.','',
      'A=frozen reference. B=original joint negative layer-average utility loss. C=same joint loss with direct inner steps constrained to[1e-6,infinity) by post-update projection. D=separate inner/outer objectives with **exactly C’s parameterization**, same LR and unchanged momentum treatment. Mapping flips negative inner steps to their absolute magnitudes, floors exact zeros, and leaves all momentum/outer parameters unchanged. Mapping-only performance is evaluated separately before any weight training. All conditions use the same independent validation-selection protocol, including initialization. This is a controlled continuation under oracle radii and saved initial states, not a claim of replaying the authors’ unavailable training.','',
      '## Main controlled results','',
      '| Condition | Communication | Sensing MI upper-bound metric | Independently attacked shared-union utility |','|---|---:|---:|---:|']
    for c in ['A','B','C','D']:
        v=report[c+'_mean'];lines.append(f"| {c} | {v['comm']:.6f} | {v['sense']:.6f} | {v['objective']:.6f} |")
    lines+=['','| Paired contrast | Weighted gain | Per-pretrained-model mean gains | Crossed hierarchical95% interval |','|---|---:|---|---|']
    for key,z in contrasts.items():
        x=z['objective'];lines.append(f"| {key} | {x['mean']:.6f} | {', '.join(f'{v:.6f}' for v in x['per_pretraining_seed_mean'])} | [{x['crossed_hierarchical_95pct'][0]:.6f}, {x['crossed_hierarchical_95pct'][1]:.6f}] |")
    result=contrasts['D_minus_C']['objective']
    conclusion='Role separation improves the positive-step equal-extra-training control on average in this experiment.' if result['mean']>0 else 'Role separation does not outperform the positive-step equal-extra-training control on average in this experiment.'
    lines+=['',conclusion+' The gain/interval and per-pretraining results above determine its strength; no positive narrative is forced. These are exploratory results with only three pretrained models and128 shared scenarios. Bootstrap reports separate conditional channel, pretraining-only and crossed intervals in summary.json; fine-tuning variability within each pretrained model is also saved. Thousands of channel evaluations are not independent pretrained models.','',
      '| Pretrained seed | A | B | C | D | Mapping-only |','|---|---:|---:|---:|---:|---:|']
    for pre in pretraining:lines.append('| '+str(pre)+' | '+' | '.join(f"{per_seed[str(pre)][c]['objective']:.6f}" for c in ['A','B','C','D'])+f" | {mapping[str(pre)]['shared_union']['objective']:.6f} |")
    lines+=['','## Training computation and role verification','',
      'Short validation profiling found one separated cycle costs about2.42 joint updates. Primary B/C use a deterministic mix of2/3 repeated updates per minibatch to approximate that cost; D makes one inner and one outer update. The earlier2-repeat and1-repeat variants remain supplemental outputs. This matches compute approximately, not optimizer update counts. All batches/permutations/LRs are shared; repeats reuse the same minibatch. Actual measured time and distinct derivative counts are reported rather than called exactly equal.','',
      '| Condition | Mean training update seconds | Full analytic helper calls | Error-only gradient calls | Differentiated layers | Backward passes | Inner/outer block updates |','|---|---:|---:|---:|---:|---:|---|']
    for c in ['B','C','D']:
        x=counts[c];lines.append(f"| {c} | {np.mean(times[c]):.3f} | {x['full_analytic_calls']:.0f} | {x['error_gradient_calls']:.0f} | {x['differentiated_layers']:.0f} | {x['backward_passes']:.0f} | {x['inner_block_updates']:.0f}/{x['outer_block_updates']:.0f} |")
    lines+=['',
      'One full analytic helper call computes all six physical variable gradients; each alternating layer calls it twice. Fifteen online layers cost30 calls for every learned condition. Independent attackers use three-error autograd derivatives, counted separately. Differentiated inner-only layers and alternating layers have different costs; higher derivatives and projections are included in wall time. Raw per-minibatch records, optimizer states, RNG states and source/data hashes are saved. Training-update time excludes common validation/model selection, whose cost is recorded separately. CPU concurrency can affect wall times; derivative counts remain explicit.','',
      'D’s inner training freezes A/deployed beamformers and differentiates its complete15-step search. Outer training freezes attack-policy parameters, retains full derivatives through deployed adversarial iterates, and detaches only a separately computed terminal approximate adversary. This is not exact Danskin/envelope differentiation through a solved minimum. Tests verify intended parameter blocks and independent NumPy whole-search utility, not merely step signs.','',
      'On identical frozen A validation beams, mapped-initial and trained D policy utilities are independently recomputed with NumPy, then refined using independent search. Complete tables are in summary.json. A lowered policy-candidate utility indicates stronger learned search. With beams fixed, the true minimum does not change; independent refinement can therefore return the same value before/after training. Negative steps and uphill stages alone are not sufficient proof of failure.','',
      '## Classical PGDA at several computation budgets','',
      'Each budget is tuned on24 saved validation scenarios using the same3×3 inner/outer grid and30-step independent validation search:63 configurations total. Test results never choose steps. Generation and adversarial-evaluation time are separated.','',
      '| PGDA iterations | Full analytic calls | Comm | Sensing MI upper bound | Lowest-found utility | Batch-one median/p95ms |','|---|---:|---:|---:|---:|---|']
    for budget in spec['pgda_budgets']:
        z=pgda[f'budget{budget}'];t=timing['methods'][f'PGDA{budget}'];v=z['shared_union'];lines.append(f"| {budget} | {2*budget} | {v['comm']:.6f} | {v['sense']:.6f} | {v['objective']:.6f} | {t['median_ms']:.2f}/{t['p95_ms']:.2f} |")
    for name in ['A','B_balanced_s101','C_balanced_s101','D_s101']:
        t=timing['methods'][name];lines.append(f"\n{name}:30 online full calls, median{t['median_ms']:.2f}ms,p95{t['p95_ms']:.2f}ms.")
    lines+=['',
      'These are warm CPU generation measurements: AMD EPYC9V74 virtual CPU, quota two CPUs, one PyTorch thread, complex64,batch1,five warmups,20 repeats. No GPU synchronization is applicable; no GPU was used. Static scalar schedules have no neural-controller overhead. Initialization and EpsNet are timed separately; RF/CSI acquisition is excluded. No acceleration is claimed solely against180 iterations; nearby15/20-step points and the full utility/runtime/gradient curves are shown.','',
      '## Independent evaluation, feasibility and limits','',
      f'Every method uses frozen beamformers, common balls, zero/terminal/five random interior/boundary starts,50 Adam steps and nested200/1,000-step PGD, retaining best candidates and a shared union across all methods/pretrained models. The original independent attacker is preserved. Folding starts into a batch was validated against it (exact component agreement on saved validation cases and complex128 equivalence tests) without changing seeds, domains or schedules. Logical per-start gradient counts and actual batched tensor calls are distinct. There are{failures} primary numerical/feasibility failures, maximum primary residual{worst_residual:.3g} at tolerance1e-5; raw flags/residuals are retained. Failures would remain in accounting rather than be dropped.','',
      'Results are lowest utility found, not certified global worst cases. Attack saturation from200→1,000 is saved and plotted; small changes do not prove global optimality. Nominal/terminal/attacked components, full complex designs/errors, per-scenario results and candidate unions are retained. Sensing improvements refer only to the MI upper-bound metric, not measured detection or estimation accuracy.','',
      'E_previous is the prior unconstrained separated method, with inner lr1e-3 and one pretrained reference. It remains a separate additional condition; its earlier1.37% gain must not be attributed solely to D’s role contrast. All observed mapping, extra-training and sign effects remain explicit. Optional controllers, physical extensions, broad dimension/channel transfer and publication novelty are outside this experiment.','',
      'Executable commands: [CONTROLLED_COMMANDS.md](../docs/CONTROLLED_COMMANDS.md). Protocol: [CONTROLLED_PROTOCOL.md](../docs/CONTROLLED_PROTOCOL.md). Raw outputs and plots: results/controlled_ablation/. Reattach the source PDF for direct source verification; implementation experiments proceeded without assuming access.']
    Path('reports/controlled_ablation.md').write_text('\n'.join(lines)+'\n')
    # Plot only measured artifacts; CSV gives an easily inspectable scalar table.
    import csv
    with (out/'controlled_table.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['pretraining_seed','condition','comm','sensing_mi_upper_bound','weighted_utility'])
        for pre in pretraining:
            for c in ['A','B','C','D']:w.writerow([pre,c,*[per_seed[str(pre)][c][m] for m in ['comm','sense','objective']]])
    os.environ.setdefault('MPLCONFIGDIR','/tmp/isac-controlled-mpl');os.environ.setdefault('XDG_CACHE_HOME','/tmp/isac-controlled-cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,3,figsize=(15,4))
    for i,metric in enumerate(['objective','comm','sense']):
        for pre in pretraining:ax[i].plot(['A','B','C','D'],[per_seed[str(pre)][c][metric] for c in ['A','B','C','D']],marker='o',label=str(pre))
        ax[i].set_ylabel({'objective':'Lowest-found weighted utility','comm':'Communication sum-rate (nats)','sense':'Sensing MI upper-bound metric (nats)'}[metric]);ax[i].legend(title='Pretraining seed')
    fig.tight_layout();fig.savefig(out/'controlled_ablation.png',dpi=180);fig.savefig(out/'controlled_ablation.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    x=[2*b for b in spec['pgda_budgets']];y=[pgda[f'budget{b}']['shared_union']['objective'] for b in spec['pgda_budgets']]
    axes[0].plot(x,y,marker='o',label='Validation-tuned PGDA')
    axes[1].plot([timing['methods'][f'PGDA{b}']['median_ms'] for b in spec['pgda_budgets']],y,marker='o',label='Validation-tuned PGDA')
    for c in ['A','B','C','D']:
        name={'A':'A','B':'B_balanced_s101','C':'C_balanced_s101','D':'D_s101'}[c]
        axes[0].scatter(30,report[c+'_mean']['objective'],label=c)
        axes[1].scatter(timing['methods'][name]['median_ms'],report[c+'_mean']['objective'],label=c)
    axes[0].set_xlabel('Full six-block analytic design calls');axes[1].set_xlabel('Warm generation median ms (batch1)')
    for ax in axes:ax.set_ylabel('Lowest-found weighted utility');ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/'utility_vs_compute.png',dpi=180);fig.savefig(out/'utility_vs_compute.pdf');plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4))
    for c in ['A','B','C','D']:
        vals=[]
        for budget in [200,1000]:
            vals.append(np.mean([row['independent_'+str(budget)]['objective'] for key,row in rows.items() if key.split('/')[1].startswith(labels[c]+('_' if c!='A' else ''))]))
        ax.plot([200,1000],vals,marker='o',label=c)
    ax.set_xlabel('PGD steps per start (+50 Adam)');ax.set_ylabel('Independent lowest-found utility');ax.legend();fig.tight_layout();fig.savefig(out/'attack_saturation.png',dpi=180);plt.close(fig)
    np.savez_compressed(out/'plot_arrays.npz',**{f'{pre}/{c}/{m}':v for pre,cs in arrays.items() for c,ms in cs.items() for m,v in ms.items()})
    print('Wrote controlled report; D-C=',result['mean'],flush=True)


if __name__=='__main__':main()
