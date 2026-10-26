"""Paired classical communication/sensing tradeoff with validation-only retuning."""
from pathlib import Path
from dataclasses import replace
import itertools
import torch
from l2o_isac.architecture_audit import read,write,tensor_metrics,sha
from l2o_isac.solver import SolverState
from l2o_isac.attacks import union_attack,ERRORS
from .run import Pilot,save
from .spatial import system,channels,single_stream_exact_errors
from .optimizer import solve,all_residuals


def main():
    torch.set_num_threads(1);pilot=Pilot();folder=pilot.out/'tradeoff';folder.mkdir(exist_ok=True)
    for case in ['A','B','C']:
        for rho in [.2,.5,.8]:
            path=folder/f'{case}_rho{rho}.pt'
            if path.exists():continue
            cfg=system(pilot.spec,case,rho);validation,vr=pilot.data(case,'validation');grid=[]
            for mi,mo in itertools.product(pilot.spec['inner_step_grid'],pilot.spec['outer_step_grid']):
                r=solve(validation,vr['ula'],SolverState(**vr['initial']),cfg,pilot.spec,30,mi,mo)
                a=pilot.attack(validation,r['x'],r['state'],cfg,budgets=[50],validation=True)
                grid.append({'inner':mi,'outer':mo,'utility':a['metrics'][50]['objective'].mean().item()})
            bf=max(grid,key=lambda row:row['utility']);position=[]
            for step in pilot.spec['position_step_grid']:
                r=solve(validation,vr['ula'],SolverState(**vr['initial']),cfg,pilot.spec,20,bf['inner'],bf['outer'],step)
                a=pilot.attack(validation,r['x'],r['state'],cfg,budgets=[50],validation=True)
                position.append({'step':step,'utility':a['metrics'][50]['objective'].mean().item()})
            step=max(position,key=lambda row:row['utility'])['step'];scenario,raw=pilot.data(case,'test');entries={};candidates=[]
            for name,iterations,ps in [('ULA',30,0.),('Fluid',20,step)]:
                r=solve(scenario,raw['ula'],SolverState(**raw['initial']),cfg,pilot.spec,iterations,bf['inner'],bf['outer'],ps)
                a=pilot.attack(scenario,r['x'],r['state'],cfg);res=all_residuals(r['x'],r['state'],scenario,cfg,pilot.spec)
                if max(res.values())>1e-5:raise RuntimeError('Tradeoff feasibility failure')
                entries[name]={'positions':r['x'],'state':vars(r['state']),'attacked_state':vars(a['state']),'independent':a['metrics'],
                  'counts':r['counts'],'design_seconds':r['seconds'],'attack_seconds':a['seconds'],'residuals':res}
                candidates.extend([a['state'],r['state']])
                if case=='A':candidates.append(single_stream_exact_errors(channels(scenario,r['x'],cfg),r['state'],cfg))
            for name,r in entries.items():
                state=SolverState(**r['state']);batch=channels(scenario,r['positions'],cfg)
                best=union_attack(batch,{'design':state},candidates,cfg)['design'];r['shared_union']=tensor_metrics(batch,best,cfg);r['union_state']=vars(best)
            save({'case':case,'rho':rho,'methods':entries,'validation_bf_grid':grid,'validation_position_grid':position,'bf_selected':bf,'position_selected':step,
                  'data_sha256':sha(pilot.out/'scenarios.pt'),'scope':'Same test spatial paths, same uncertainty/radiated power; validation-retuned each rho; matched60 full derivative calls'},path)
            pilot.log(f'Classical tradeoff {case} rho={rho} complete')


if __name__=='__main__':main()
