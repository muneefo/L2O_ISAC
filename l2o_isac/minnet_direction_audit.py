"""Measure actual inner-stage directions along deployed trajectories (not attacks)."""
from dataclasses import replace
from pathlib import Path
import math
import torch
from .architecture_audit import Study,write
from .physics import gradients,metrics
from .solver import NAMES,pgda_step,project_ball
from .role_training import RoleSeparatedPGDA


def main():
    torch.set_num_threads(1);study=Study('configs/pilot/architecture_audit.json','results/architecture_audit');batch,initial=study.split('test')
    schedules={'U15':(study.reference.mu,study.reference.eta)}
    for seed in study.spec['training_seeds']:
        m=RoleSeparatedPGDA(study.reference);m.load_state_dict(torch.load(study.output/f'role_seed{seed}'/'selected.pt',weights_only=True)['model'])
        schedules[f'Role_seed{seed}']=(torch.cat((m.inner_mu,m.outer_mu),-1),torch.cat((m.inner_eta,m.outer_eta),-1))
    report={};raw={}
    with torch.no_grad():
        for name,(mus,etas) in schedules.items():
            cur,prev=initial,initial;deltas={k:[] for k in ['comm','sense','objective']}
            for mu,eta in zip(mus,etas):
                before=metrics(batch,cur,study.cfg);g=gradients(batch,cur,study.cfg)
                errs={n:project_ball(getattr(cur,n)-mu[j]*g[n]+eta[j]*(getattr(cur,n)-getattr(prev,n)),
                         batch.eps[:,j,None,None]*math.sqrt(study.cfg.nr if n=='dr' else study.cfg.nt)) for j,n in enumerate(NAMES[:3])}
                after=metrics(batch,replace(cur,**errs),study.cfg)
                for k in deltas:deltas[k].append(after[k]-before[k])
                nxt=pgda_step(batch,cur,prev,study.cfg,mu,eta);prev,cur=cur,nxt
            raw[name]={k:torch.stack(v) for k,v in deltas.items()}
            report[name]={k:{'mean_inner_stage_change':v.mean().item(),'fraction_increasing_above_1e-6':(v>1e-6).float().mean().item(),
                            'per_layer_mean':v.mean(-1).tolist()} for k,v in raw[name].items()}
    torch.save(raw,study.output/'minnet_stage_changes.pt');write(study.output/'minnet_stage_changes.json',{'methods':report,'samples':len(batch),
          'scope':'Changes in deployed inner stage with previous projected momentum state; not an independent attacker or proof of monotonic minimax convergence'})


if __name__=='__main__':main()
