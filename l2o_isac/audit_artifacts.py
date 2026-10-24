"""Export per-scenario feasibility and verify preserved reference provenance."""
from pathlib import Path
import hashlib
import json
import subprocess
import numpy as np
import torch
from .architecture_audit import Study,write,sha
from .solver import SolverState


def residuals(state,batch,cfg):
    result={'analog_modulus':(state.a.abs()-1).abs().flatten(1).amax(1),
            'transmit_power':((state.a@state.d).abs().square().sum((-2,-1))-cfg.power).abs(),
            'receive_norm':(state.w.norm(dim=-1)-1).abs().amax(1)}
    for i,n in enumerate(['dh','dt','dr']):
        result[n+'_violation']=(getattr(state,n).norm(dim=-1)-batch.eps[:,i,None]*(cfg.nr if n=='dr' else cfg.nt)**.5).clamp_min(0).amax(1)
    return result


def main():
    torch.set_num_threads(1);s=Study('configs/pilot/architecture_audit.json','results/architecture_audit');b,_=s.split('test')
    arrays={};summary={}
    for folder in ['adversarial_audit','extension_evaluation','strong_attack_check','role_ablations']:
        for p in (s.output/folder).glob('*.pt'):
            r=torch.load(p,weights_only=True)
            key='attacked_state' if 'attacked_state' in r else 'state' if 'state' in r else None
            if key is None:continue
            stages={key:r[key]}
            if 'union_state' in r:stages['union_state']=r['union_state']
            for stage,raw in stages.items():
                state=SolverState(**raw);z=residuals(state,b,s.cfg);name=folder+'/'+p.stem+'/'+stage
                finite=torch.stack([torch.isfinite(getattr(state,n)).flatten(1).all(1) for n in ['dh','dt','dr','a','d','w']]).all(0)
                fail=(torch.stack(list(z.values())).amax(0)>1e-5)|~finite
                for k,v in z.items():arrays[name+'/'+k]=v.numpy()
                arrays[name+'/failure']=fail.numpy();summary[name]={'maxima':{k:v.max().item() for k,v in z.items()},'failures':int(fail.sum()),'scenarios':len(b)}
                assert not fail.any()
    np.savez_compressed(s.output/'feasibility_per_scenario.npz',**arrays)
    files=['l2o_isac/physics.py','l2o_isac/solver.py','l2o_isac/data.py','l2o_isac/networks.py','l2o_isac/training.py','l2o_isac/inference.py',
           'reports/corrected_per_channel_seed73129/checkpoint.pt','reports/corrected_per_channel_seed73129/raw_beamformers_and_channels.pt']
    preserved={}
    for name in files:
        pinned=subprocess.check_output(['git','show',s.spec['reference_commit']+':'+name])
        preserved[name]=hashlib.sha256(pinned).hexdigest()==sha(name)
    assert all(preserved.values())
    write(s.output/'artifact_validation.json',{'per_scenario_feasibility':summary,'reference_files_unchanged':preserved,
          'reference_sha256':s.sha,'failures':0,'precision':'complex64; absolute feasibility tolerance 1e-5'})


if __name__=='__main__':main()
