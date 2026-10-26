"""One random nonuniform array held fixed across every channel scenario/split."""
import itertools
import torch
from l2o_isac.architecture_audit import read,write
from l2o_isac.solver import SolverState
from .run import Pilot
from .spatial import random_positions,system,initialize
from .optimizer import solve


def main():
    torch.set_num_threads(1);p=Pilot();path=p.out/'static_random_geometry.json'
    if not path.exists():
        x=random_positions(p.spec,1,909999)[0]
        write(path,{'positions_wavelengths':x.tolist(),'seed':909999,'scope':'One independent draw, fixed for every train/validation/test channel and all three regimes; endpoints0,A'})
    point=torch.tensor(read(path)['positions_wavelengths']);cp=p.out/'static_random_validation.json'
    if not cp.exists():
        choices={}
        for case in ['A','B','C']:
            sc,_=p.data(case,'validation');cfg=system(p.spec,case);x=point.expand(len(sc),-1).clone();initial=initialize(sc,x,cfg,case,p.spec['initialization_seed']+['A','B','C'].index(case));choices[case]={}
            for iterations in [30,90,180]:
                grid=[]
                for mi,mo in itertools.product(p.spec['inner_step_grid'],p.spec['outer_step_grid']):
                    r=solve(sc,x,initial,cfg,p.spec,iterations,mi,mo);a=p.attack(sc,x,r['state'],cfg,budgets=[50],validation=True)
                    grid.append({'inner':mi,'outer':mo,'utility':a['metrics'][50]['objective'].mean().item()})
                choices[case][str(iterations)]={'selected':max(grid,key=lambda v:v['utility']),'grid':grid}
        write(cp,{'cases':choices,'test_used':False,'selection_split':'validation only'})
    for case in ['A','B','C']:
        sc,raw=p.data(case,'test');cfg=system(p.spec,case)
        for iterations in [30,90,180]:
            label='budget'+str(2*iterations)
            if not (p.out/case/('StaticRandom_'+label+'.pt')).exists():p.store_extra(case,'StaticRandom',label,p.design(case,sc,raw,'StaticRandom',iterations),iterations)
        p.shared_pool(case)


if __name__=='__main__':main()
