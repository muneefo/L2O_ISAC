"""Prespecified additional attack-budget check; never used for model selection."""
import argparse
from dataclasses import replace
from pathlib import Path
import time
import numpy as np
import torch
from .architecture_audit import Study,read,write,sha,means,tensor_metrics
from .solver import SolverState,constraint_errors
from .attacks import attack,union_attack


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/pilot/architecture_audit.json');p.add_argument('--output',default='results/architecture_audit')
    p.add_argument('--steps',type=int,default=1000);p.add_argument('--restarts',type=int,default=5);a=p.parse_args(argv)
    torch.set_num_threads(1);study=Study(a.config,a.output);batch,_=study.split('test');out=Path(a.output)/'strong_attack_check';out.mkdir(exist_ok=True)
    chosen=['U15','RU25','PGDA180']+[f'Role_seed{s}' for s in study.spec['training_seeds']]
    states={};candidates=[];raw={};summaries={};choice=read(Path(a.output)/'validation_choices.json')
    for name in chosen:
        design=SolverState(**torch.load(Path(a.output)/'extension_evaluation'/(name+'.pt'),weights_only=True)['design']);states[name]=design
        path=out/(name+'.pt')
        if path.exists():r=torch.load(path,weights_only=True)
        else:
            r=attack(batch,design,study.cfg,budgets=(200,a.steps),restarts=a.restarts,adam_steps=50,step_size=choice['attack_step'],seed=9001)
            r={'state':vars(r['state']),'metrics':r['metrics'],'seconds':r['seconds'],'error_gradient_calls':r['gradient_evaluations'],
               'steps':a.steps,'restarts':a.restarts,'reference_sha256':study.sha,'split_sha256':sha(Path(a.output)/'splits.pt')}
            torch.save(r,path)
        assert r['split_sha256']==sha(Path(a.output)/'splits.pt') and r['steps']==a.steps and r['restarts']==a.restarts
        assert max(constraint_errors(SolverState(**r['state']),batch,study.cfg).values())<1e-5
        raw[name]=r;candidates.append(SolverState(**r['state']))
        # Include all previously found candidates, not only this strategy's winners.
        old=torch.load(Path(a.output)/'extension_evaluation'/(name+'.pt'),weights_only=True)
        candidates.extend([SolverState(**old['attacked_state']),SolverState(**old['terminal_state'])])
        summaries[name]={'200':means(r['metrics'][200]),str(a.steps):means(r['metrics'][a.steps]),'seconds':r['seconds'],'error_gradient_calls':r['error_gradient_calls']}
        study.event('Strong attack '+name+': '+str(summaries[name]))
    union=union_attack(batch,states,candidates,study.cfg);baseline=tensor_metrics(batch,union['U15'],study.cfg)['objective'];rng=np.random.default_rng(9111)
    for name,s in union.items():
        z=tensor_metrics(batch,s,study.cfg);raw[name]['union_metrics']=z;raw[name]['union_state']=vars(s);torch.save(raw[name],out/(name+'.pt'))
        diff=(z['objective']-baseline).numpy();boots=diff[rng.integers(0,len(diff),size=(2000,len(diff)))].mean(1)
        summaries[name]['shared_union']=means(z);summaries[name]['paired_difference_vs_U15']=float(diff.mean());summaries[name]['paired_channel_95pct']=np.quantile(boots,[.025,.975]).tolist()
    write(out/'summary.json',{'methods':summaries,'samples':len(batch),'steps':a.steps,'random_restarts':a.restarts,'scope':'Additional pilot strength check, fixed before execution, not used for model selection; five random restarts, not full-study twenty; no certificate'})


if __name__=='__main__':main()
