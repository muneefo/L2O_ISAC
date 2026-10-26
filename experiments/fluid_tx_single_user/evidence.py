"""Independent NumPy verification and immutable-baseline evidence export."""
from pathlib import Path
import hashlib,json,subprocess
from datetime import datetime,timezone
import numpy as np
import torch
from l2o_isac.architecture_audit import read,write,sha
from l2o_isac.execution_evidence import numpy_metrics
from l2o_isac.solver import SolverState,NAMES
from .run import Pilot,ROOT
from .spatial import channels,system


def main():
    torch.set_num_threads(1);p=Pilot();preserve=read(ROOT/'outputs/baseline_preservation.json')
    unchanged={n:sha(n)==digest for n,digest in preserve['preserved_sha256'].items()}
    if not all(unchanged.values()):raise RuntimeError('Pre-existing baseline files changed')
    outputs={};values={};maxres=0.;replays={}
    from .learning import PositionSchedule,select_rows
    learning_settings=read(ROOT/'configs/learning.json')
    for case in ['A','B','C']:
        sc,original=p.data(case,'test');cfg=system(p.spec,case)
        for path in (p.out/case).glob('*.pt'):
            r=torch.load(path,weights_only=True)
            if 'shared_union' not in r:continue
            state=SolverState(**r['union_state']);batch=channels(sc,r['positions'],cfg);independent=numpy_metrics(batch,state,cfg);diff={}
            with torch.no_grad():
                if r['method'].startswith('Learned'):
                    saved=torch.load(r['model_checkpoint'],weights_only=True)
                    model=PositionSchedule(p,case,learning_settings,saved['initial_step'],r['method']=='LearnedConstant');model.load_state_dict(saved['model'])
                    xx,ii=select_rows(original,slice(None));replay=model(sc,xx,ii)
                else:replay=p.design(case,sc,original,r['method'],r['iterations'])
            differences={'positions':(replay['x']-r['positions']).abs().max().item(),**{n:(getattr(replay['state'],n)-r['state'][n]).abs().max().item() for n in NAMES}}
            if max(differences.values())>1e-5:raise RuntimeError('Final source replay mismatch '+str(path)+': '+str(differences))
            replays[case+'/'+path.stem]=differences
            for k in ['comm','sense','objective']:
                diff[k]=float(np.max(np.abs(independent[k]-r['shared_union'][k].numpy())))
                np.testing.assert_allclose(independent[k],r['shared_union'][k].numpy(),rtol=1e-5,atol=2e-6)
                for a,b in [(50,200),(200,1000)]:assert (r['attack_budget_metrics'][b][k]<=r['attack_budget_metrics'][a][k]+2e-6).all()
                assert (r['shared_union'][k]<=r['attack_budget_metrics'][1000][k]+2e-6).all()
                values[case+'/'+path.stem+'/'+k]=independent[k]
            for n in NAMES[3:]:torch.testing.assert_close(getattr(state,n),r['state'][n],rtol=0,atol=0)
            maxres=max(maxres,max(r['max_residuals'].values()));assert max(r['max_residuals'].values())<1e-5
            outputs[case+'/'+path.stem]={'numpy_max_absolute_errors':diff,'max_residual':max(r['max_residuals'].values()),'scenarios':len(sc)}
            if r['method'].startswith('Learned'):
                saved=torch.load(r['model_checkpoint'],weights_only=True);assert set(saved['model'])=={'mu'}
                assert saved['model']['mu'].numel()==r['learned_parameter_count']
                assert saved['beamforming_steps_frozen']==read(p.out/'validation_choices.json')['cases'][case]['ula']['selected']
        for seed in [101,202,303]:
            a=torch.load(p.out/'learning'/case/f'LearnedConstant_seed{seed}'/'latest.pt',weights_only=True)
            b=torch.load(p.out/'learning'/case/f'LearnedSchedule_seed{seed}'/'latest.pt',weights_only=True)
            assert [v['batch_order_sha256'] for v in a['history']]==[v['batch_order_sha256'] for v in b['history']]
            assert a['updates']==b['updates']==8
    np.savez_compressed(p.out/'independent_numpy_results.npz',**values)
    record={'baseline_files_verified_unchanged':sum(unchanged.values()),'baseline_commit':preserve['source_commit'],'method_checks':outputs,
            'final_source_full_design_replay':replays,'max_feasibility_residual':maxres,'numerical_failures':0,'data_sha256':sha(p.out/'scenarios.pt'),
            'scope':'Independent signal-equation check of newly executed fluid workflow; no author-code/global worst-case certification'}
    write(p.out/'validation.json',record)
    sources={str(q):sha(q) for q in ROOT.rglob('*') if q.is_file() and '/outputs/' not in str(q) and '__pycache__' not in str(q)}
    artifacts={str(q):sha(q) for q in (ROOT/'outputs').rglob('*') if q.is_file() and q.name!='manifest.json'}
    write(p.out/'manifest.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'source_commit':p.source,
       'source_sha256':sources,'artifact_sha256':artifacts,'baseline_commit':preserve['source_commit'],
       'scope':'Executed isolated fluid-transmit pilot, fresh spatial channel distribution, preserved reproduction',
       'failures':[{'log':str(ROOT/'outputs/initial_test_failure.log'),'reason':'Float64 wrapper originally initialized step coefficients at a different precision; fixed to the shared PGDA buffer convention without relaxing equality test'}]})
    print('Verified',len(outputs),'method outputs with NumPy; preserved',sum(unchanged.values()),'pre-existing files')


if __name__=='__main__':main()
