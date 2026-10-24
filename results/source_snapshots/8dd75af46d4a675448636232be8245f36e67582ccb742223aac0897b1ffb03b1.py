"""Frozen-weight rho/radius sensitivity; secondary 32-scenario single-seed check."""
import argparse
import copy
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
from .architecture_audit import Study,read,write,state_slice,tensor_metrics,means,sha
from .role_training import RoleSeparatedPGDA
from .solver import NAMES,constraint_errors
from .attacks import attack,union_attack


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/pilot/architecture_audit.json');p.add_argument('--output',default='results/architecture_audit');a=p.parse_args(argv)
    torch.set_num_threads(1);study=Study(a.config,a.output);out=study.output/'secondary_tradeoff';out.mkdir(exist_ok=True)
    batch,init=study.split('test');batch=batch[:32];init=state_slice(init,slice(0,32));rows=[]
    ref=copy.deepcopy(study.reference);role=RoleSeparatedPGDA(study.reference)
    role.load_state_dict(torch.load(study.output/f'role_seed{study.spec["training_seeds"][0]}'/'selected.pt',weights_only=True)['model'])
    for rho in [.2,.5,.8]:
        for eps in [.01,.05]:
            cfg=replace(study.cfg,rho=rho,eps_h=eps,eps_t=eps,eps_r=eps);ref.cfg=cfg;role.cfg=cfg
            b=replace(batch,eps=torch.full_like(batch.eps,eps));s=replace(init,**{n:getattr(init,n)*(eps/.05) for n in NAMES[:3]})
            states={};candidates=[];items={}
            for name,model in [('U15',ref),('Role_seed101',role)]:
                path=out/f'{name}_rho{rho}_eps{eps}.pt'
                if path.exists():entry=torch.load(path,weights_only=True);states[name]=type(s)(**entry['design']);items[name]=entry;candidates.append(type(s)(**entry['attacked_state']));continue
                with torch.no_grad():design=model(b,s)[0]
                result=attack(b,design,cfg,budgets=(50,200),restarts=5,adam_steps=50,step_size=read(study.output/'validation_choices.json')['attack_step'],seed=9001)
                entry={'design':vars(design),'attacked_state':vars(result['state']),'independent':result['metrics'][200],
                       'budget_metrics':result['metrics'],'rho':rho,'epsilon':eps,'samples':len(b),'training_seed':study.spec['training_seeds'][0],
                       'split_sha256':sha(study.output/'splits.pt'),'reference_sha256':study.sha,'source_sha256':study.commit_sources}
                assert max(constraint_errors(result['state'],b,cfg).values())<1e-5
                torch.save(entry,path);states[name]=design;candidates.append(result['state']);items[name]=entry
            union=union_attack(b,states,candidates,cfg)
            for name,state in union.items():
                z=tensor_metrics(b,state,cfg);entry=items[name];entry['shared_union']=z;torch.save(entry,out/f'{name}_rho{rho}_eps{eps}.pt')
                rows.append({'method':name,'rho':rho,'epsilon':eps,**means(z)})
            study.event(f'Secondary rho={rho} epsilon={eps} complete')
    write(out/'summary.json',{'rows':rows,'samples':len(batch),'scope':'Secondary 32 fixed test scenarios; fixed weights trained at rho=.8/epsilon=.05; only extension seed101; no tuning/retraining, no broad generalization claim'})
    import os
    os.environ.setdefault('MPLCONFIGDIR','/tmp/isac-audit-mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for i,key in enumerate(['comm','sense']):
        for name in ['U15','Role_seed101']:
            for eps in [.01,.05]:
                rr=[r for r in rows if r['method']==name and r['epsilon']==eps]
                axes[i].plot([r['rho'] for r in rr],[r[key] for r in rr],marker='o',label=f'{name}, eps={eps}')
        axes[i].set_xlabel('rho');axes[i].set_ylabel('Communication sum-rate (nats)' if key=='comm' else 'Sensing MI upper-bound metric (nats)');axes[i].legend(fontsize=7)
    fig.tight_layout();fig.savefig(out/'components_vs_rho.png',dpi=170);fig.savefig(out/'components_vs_rho.pdf');plt.close(fig)


if __name__=='__main__':main()
