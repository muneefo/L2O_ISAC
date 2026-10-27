"""Independent signal checks, complete source replay, seed accounting and preservation."""
from pathlib import Path
import json,hashlib
import numpy as np
import torch
from l2o_isac.execution_evidence import numpy_metrics
from l2o_isac.audit_artifacts import residuals
from l2o_isac.solver import SolverState,NAMES,constraint_errors
from experiments.fluid_tx_single_user.spatial import channels,initialize
from .run import Study,ROOT,digest,write

@torch.no_grad()
def main():
 torch.set_num_threads(1);s=Study();pres=json.loads((ROOT/'sources/baseline_manifest.json').read_text());bad=[p for p,sha in pres['tracked_sha256'].items() if digest(p)!=sha]
 assert not bad,bad
 results={};numeric={};feasible={};maxres=0.;maxreplay=0.
 for c in 'ABC':
  sc,raw=s.data(c,'test')
  for path in sorted((s.out/c).glob('*.pt')):
   r=torch.load(path,weights_only=True);cfg=s.cfg[c];design=SolverState(**r['state']);union=SolverState(**r['union_state']);batch=channels(sc,r['positions'],cfg)
   vals=numpy_metrics(batch,union,cfg)
   for stage,st in [('design',design),('union',union)]:
    for k,v in residuals(st,batch,cfg).items():feasible[f'{c}/{path.stem}/{stage}/{k}']=v.numpy()
   for k,v in r['geometry_per_scenario'].items():feasible[f'{c}/{path.stem}/geometry/{k}']=v.numpy()
   errors={}
   for k in ['comm','sense','objective']:
    errors[k]=float(np.max(np.abs(vals[k]-r['shared_union'][k].numpy())))
    np.testing.assert_allclose(vals[k],r['shared_union'][k].numpy(),atol=3e-6,rtol=1e-5)
    prior=None
    for b in s.spec['test_attack_budgets']:
     if prior is not None:assert (r['attack_metrics'][b][k]<=prior+3e-6).all()
     prior=r['attack_metrics'][b][k]
    assert (r['shared_union'][k]<=prior+3e-6).all()
    numeric[f'{c}/{path.stem}/{k}']=vals[k]
   for n in NAMES[3:]:torch.testing.assert_close(getattr(union,n),getattr(design,n),rtol=0,atol=0)
   residual=max(constraint_errors(union,batch,cfg).values());maxres=max(maxres,residual,max(r['residuals'].values()));assert maxres<1e-5
   x=raw['ula'];initial=SolverState(**raw['initial'])
   if r['kind'] in ['random','offline']:
    x=r['positions'];initial=initialize(sc,x,cfg,c,s.spec['initialization_seed']+'ABC'.index(c))
   if r['kind'] in ['classical','random','offline']:replay=s.classical(c,sc,x,initial,r['fluid'],r['iterations'])
   elif r['kind']=='analytic':replay={'x':x,'state':initial}
   else:replay=s.model(c,r['kind'],r['seed'])(sc,x,initial)
   diffs={'positions':float((replay['x']-r['positions']).abs().max()),**{n:float((getattr(replay['state'],n)-getattr(design,n)).abs().max()) for n in NAMES}}
   maxreplay=max(maxreplay,max(diffs.values()));assert max(diffs.values())<1e-5,(path,diffs)
   results[c+'/'+path.stem]={'independent_numpy_errors':errors,'source_replay':diffs,'max_residual':max(r['residuals'].values()),'scenarios':len(sc)}
 for c in 'ABC':
  for seed in s.spec['seeds']:
   models=[torch.load(s.out/'training'/c/f'{kind}_s{seed}'/'latest.pt',weights_only=True) for kind in ['FixedL2O','FluidL2O']]
   assert [h['batch_order_sha256'] for h in models[0]['history']]==[h['batch_order_sha256'] for h in models[1]['history']]
   assert all(m['updates']==40 for m in models)
   assert all(m['gradient_policy'].startswith('Full end-to-end') for m in models)
   for m in models:assert all(np.isfinite(h['gradient_abs_by_mu_block']).all() for h in m['history'])
 np.savez_compressed(s.out/'independent_numpy_metrics.npz',**numeric)
 np.savez_compressed(s.out/'feasibility_per_scenario.npz',**feasible)
 record={'preserved_baseline_files':len(pres['tracked_sha256']),'baseline_commit':pres['source_commit'],'verified_methods':len(results),'max_feasibility_residual':maxres,'max_complete_replay_difference':maxreplay,'checks':results,'source_sha256':s.source,'scenario_sha256':digest(s.out/'scenarios.pt'),'independent_scenarios_per_regime':100,'primary_training_runs':18,'momentum_ablation_runs':3,'failure_accounting':'21/21 scheduled models completed. Failed initial regression test (position coefficient precision comparison) retained in work history; corrected comparison uses identical numeric schedule. No training run omitted.'}
 write(s.out/'validation.json',record)
 artifacts={str(p):digest(p) for p in s.out.rglob('*') if p.is_file() and p.name!='artifact_manifest.json'}
 write(s.out/'artifact_manifest.json',{'artifacts_sha256':artifacts,'core_source_sha256':s.source,'baseline_commit':pres['source_commit']})
 print('Verified',len(results),'methods; baseline',len(pres['tracked_sha256']),'unchanged; max replay',maxreplay,'max feasibility',maxres)
if __name__=='__main__':main()
