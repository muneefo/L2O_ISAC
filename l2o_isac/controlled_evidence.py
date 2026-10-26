"""Validate saved controlled outputs independently and preserve an audit manifest."""
from pathlib import Path
import hashlib,itertools,json,subprocess
from datetime import datetime,timezone
import numpy as np
import torch
from .controlled_study import ControlledStudy
from .solver import SolverState,NAMES
from .architecture_audit import read,write,sha,state_slice
from .execution_evidence import numpy_metrics
from .physics import metrics
from .inference import load_checkpoint


def main():
    torch.set_num_threads(1);study=ControlledStudy();out=study.out;old=read(out/'preservation.json')
    preserved={p:sha(p)==digest for p,digest in old['preserved_file_sha256'].items()}
    assert all(preserved.values())
    unchanged={}
    for name in ['physics','solver','data','networks','training','inference']:
        p=Path('l2o_isac')/(name+'.py');pinned=subprocess.check_output(['git','show','73159d4:'+str(p)])
        unchanged[str(p)]=hashlib.sha256(pinned).hexdigest()==sha(p)
    assert all(unchanged.values())
    seeds=[];weights={};mapping={}
    for index,p in enumerate(study.spec['reference_checkpoints']):
        cfg,_,ref,saved=load_checkpoint(p);seed=saved['run']['seed'];seeds.append(seed);weights[seed]=ref.raw_mu.detach()
        transformed=ref.mu[:,:3].abs().clamp_min(study.spec['positive_inner_floor'])
        mapping[str(seed)]={'negative_inner_steps':int((ref.mu[:,:3]<0).sum()),'flipped_entries':int((transformed!=ref.mu[:,:3]).sum()),
                           'max_inner_mu_change':(transformed-ref.mu[:,:3]).abs().max().item(),'momentum_treatment':'unchanged direct learned coefficients',
                           'checkpoint_sha256':sha(p)}
    assert len(set(seeds))==3
    pairs={f'{a}/{b}':(weights[a]-weights[b]).abs().max().item() for a,b in itertools.combinations(seeds,2)}
    assert all(v>0 for v in pairs.values())
    data=study.data['test'];checks={};arrays={}
    for folder in [*out.glob('pretrain*/evaluation'),out/'pgda']:
        for p in sorted(folder.glob('*.pt')):
            r=torch.load(p,weights_only=True);name=str(p.relative_to(out));assert r['split_sha256']==study.split_hash
            state=SolverState(**r['union_state']);direct=numpy_metrics(data,state,study.cfg);errors={}
            for key in ['comm','sense','objective']:
                stored=r['shared_union'][key].numpy();errors[key]=float(np.max(np.abs(stored-direct[key])))
                np.testing.assert_allclose(stored,direct[key],rtol=1e-5,atol=2e-6)
                assert (r['budget_metrics'][1000][key]<=r['budget_metrics'][200][key]+1e-6).all()
                assert (r['shared_union'][key]<=r['budget_metrics'][1000][key]+1e-6).all()
                assert (r['shared_union'][key]<=r['terminal'][key]+1e-5).all()
                arrays[name+'/'+key]=stored
            assert not r['failure_flags'].any()
            for n in NAMES[3:]:torch.testing.assert_close(getattr(state,n),r['design'][n],rtol=0,atol=0)
            checks[name]={'numpy_max_absolute_errors':errors,'failures':int(r['failure_flags'].sum()),'attack_saturation_max_change':max((r['budget_metrics'][200][k]-r['budget_metrics'][1000][k]).abs().max().item() for k in ['comm','sense','objective'])}
    old_attack=torch.load('results/architecture_audit/strong_attack_check/U15.pt',weights_only=True)
    new_attack=torch.load(out/'pretrain73129/evaluation/A.pt',weights_only=True)
    equivalence={k:(old_attack['metrics'][1000][k]-new_attack['budget_metrics'][1000][k]).abs().max().item() for k in ['comm','sense','objective']}
    assert max(equivalence.values())<1e-5
    np.savez_compressed(out/'independent_numpy_validation.npz',**arrays)
    write(out/'validation.json',{'preserved_prior_inputs':preserved,'physical_core_unchanged':unchanged,'distinct_pretraining_seeds':seeds,
      'pretrained_weight_max_differences':pairs,'mapping_parameter_changes':mapping,'output_checks':checks,
      'preserved_serial_vs_batched_1000step_A_max_errors':equivalence,'failure_count':0,'source_pdf_available':False,
      'source_pdf_request':'Reattach source PDF; no actual source-PDF path exists in the accessible project inputs'})
    # This manifest covers new evidence, leaving the historical results/manifest.json
    # byte-identical. Source commits plus per-file hashes remain distinct.
    artifact_files=[p for p in out.rglob('*') if p.is_file() and p.name!='manifest.json']
    sources=[*Path('l2o_isac').glob('*.py'),*Path('tests').glob('*.py'),*Path('configs/pilot').glob('controlled*.json'),Path('docs/CONTROLLED_PROTOCOL.md'),Path('docs/CONTROLLED_COMMANDS.md'),Path('reports/controlled_ablation.md'),Path('STATUS.md')]
    write(out/'manifest.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
      'source_sha256':{str(p):sha(p) for p in sources},'artifact_sha256':{str(p):sha(p) for p in artifact_files},
      'preserved_reference_commit':old['source_commit'],'original_archive_sha256':old['archive_sha256'],
      'reference_checkpoint_sha256':{p:sha(p) for p in study.spec['reference_checkpoints']},
      'scientific_scope':'Controlled implementation-based exploratory study; original physical model unchanged; no author-code/global-worst-case certification',
      'failed_attempts':[{'log':str(out/'balanced_first_attempt_failure.log'),'kind':'checkpoint metadata serialization','resolution':str(out/'checkpoint_metadata_repair.json')}],
      'execution_protocol':str(out/'frozen_protocol.json')})
    print('Validated',len(checks),'saved method designs with independent NumPy; all prior input hashes unchanged')


if __name__=='__main__':main()
