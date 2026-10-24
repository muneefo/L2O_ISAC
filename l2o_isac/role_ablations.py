"""Inner-only / outer-only controls, same fixed learning rates and audit splits.

These are extension variants; the original reference is not retrained or changed.
No hyperparameters are selected from the test set.
"""
import argparse
import copy
from dataclasses import replace,asdict
from pathlib import Path
import time
import numpy as np
import torch
from .architecture_audit import Study,read,write,sha,state_slice,tensor_metrics,means
from .role_training import RoleSeparatedPGDA
from .physics import metrics
from .attacks import attack,ERRORS,union_attack
from .solver import SolverState,constraint_errors


def train_control(study,seed,mode,folder):
    folder.mkdir(exist_ok=True);selected=folder/'selected.pt';latest=folder/'latest.pt'
    model=RoleSeparatedPGDA(study.reference);lr=read(study.output/'training_choices.json')['selected_outer_lr'] if mode=='outer' else study.spec['inner_lr']
    opt=torch.optim.SGD(model.inner_parameters() if mode=='inner' else model.outer_parameters(),lr=lr)
    generator=torch.Generator().manual_seed(seed);best=study.validation_score(model);history=[];elapsed=0.;epoch_start=0
    def save(epoch):
        return {'model':copy.deepcopy(model.state_dict()),'optimizer':opt.state_dict(),'torch_rng_state':torch.get_rng_state(),'shuffle_rng_state':generator.get_state(),
                'config':asdict(study.cfg),'spec':study.spec,'mode':mode,'seed':seed,'epoch':epoch,'best_validation':best,'history':history,'seconds':elapsed,
                'source_commit':study.source,'source_sha256':study.commit_sources,'reference_sha256':study.sha,'split_sha256':sha(study.output/'splits.pt')}
    if latest.exists():
        old=torch.load(latest,weights_only=True)
        assert old['reference_sha256']==study.sha and old['spec']==study.spec and old['mode']==mode
        model.load_state_dict(old['model']);opt.load_state_dict(old['optimizer']);generator.set_state(old['shuffle_rng_state']);torch.set_rng_state(old['torch_rng_state'])
        best=old['best_validation'];history=old['history'];elapsed=old['seconds'];epoch_start=old['epoch']
    else:torch.save(save(0),selected)
    train,initial=study.split('train')
    for epoch in range(epoch_start,study.spec['epochs']):
        t=time.perf_counter();losses=[]
        for ids in torch.randperm(len(train),generator=generator).split(study.spec['batch_size']):
            b=train[ids];s=state_slice(initial,ids);model.set_phase(mode);opt.zero_grad()
            if mode=='inner':
                with torch.no_grad():design=model(b,s)[0]
                loss=metrics(b,model.inner_attack(b,design),study.cfg)['objective'].mean()
            else:
                state=model(b,s)[0]
                adverse=attack(b,state,study.cfg,budgets=(study.spec['training_attack_steps'],),restarts=0,adam_steps=0,
                               step_size=read(study.output/'validation_choices.json')['attack_step'])['state']
                loss=-metrics(b,replace(state,**{n:getattr(adverse,n).detach() for n in ERRORS}),study.cfg)['objective'].mean()
            loss.backward();study.check_grads(model);opt.step();losses.append(loss.item())
        score=study.validation_score(model);elapsed+=time.perf_counter()-t
        improved=score>best
        if improved:best=score
        history.append({'epoch':epoch+1,'loss':float(np.mean(losses)),'validation':score,'seconds':time.perf_counter()-t})
        torch.save(save(epoch+1),latest)
        if improved:torch.save(save(epoch+1),selected)
        study.event(f'{mode}-only seed={seed} epoch={epoch+1} validation={score:.6f}')
    return selected


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/pilot/architecture_audit.json');p.add_argument('--output',default='results/architecture_audit');a=p.parse_args(argv)
    torch.set_num_threads(1);study=Study(a.config,a.output)
    if not (study.output/'adversarial_audit/summary.json').exists():raise RuntimeError('Gate D required')
    out=study.output/'role_ablations';out.mkdir(exist_ok=True);batch,initial=study.split('test');states={};raw={};candidates=[];report={}
    for mode in ['inner','outer']:
        for seed in study.spec['training_seeds']:
            name=f'{mode}_only_seed{seed}';model=RoleSeparatedPGDA(study.reference);selected=train_control(study,seed,mode,out/name)
            ckpt=torch.load(selected,weights_only=True);model.load_state_dict(ckpt['model'])
            with torch.no_grad():design=model(batch,initial)[0]
            path=out/(name+'.pt')
            if path.exists():r=torch.load(path,weights_only=True)
            else:
                attack_out=attack(batch,design,study.cfg,budgets=study.spec['attack_budgets'],restarts=study.spec['random_restarts'],adam_steps=study.spec['adam_steps'],
                                  step_size=read(study.output/'validation_choices.json')['attack_step'],seed=9001)
                r={'design':vars(design),'attacked_state':vars(attack_out['state']),'metrics':attack_out['metrics'],
                   'attack_seconds':attack_out['seconds'],'error_gradient_calls':attack_out['gradient_evaluations'],'reference_sha256':study.sha,
                   'selected_epoch':ckpt['epoch'],'seed':seed,'mode':mode,'split_sha256':sha(study.output/'splits.pt')}
                torch.save(r,path)
            states[name]=design;raw[name]=r;candidates.append(SolverState(**r['attacked_state']))
            assert max(constraint_errors(SolverState(**r['attacked_state']),batch,study.cfg).values())<1e-5
    # All controls and primary methods share a common union of candidates.
    for path in (study.output/'extension_evaluation').glob('*.pt'):
        r=torch.load(path,weights_only=True);name=path.stem;states[name]=SolverState(**r['design']);raw[name]=r
        candidates.extend([SolverState(**r['attacked_state']),SolverState(**r['terminal_state'])])
    union=union_attack(batch,states,candidates,study.cfg);base=tensor_metrics(batch,union['U15'],study.cfg)['objective'];rng=np.random.default_rng(9112)
    for name,s in union.items():
        z=tensor_metrics(batch,s,study.cfg);d=(z['objective']-base).numpy();boot=d[rng.integers(0,len(d),size=(2000,len(d)))].mean(1)
        report[name]={'shared_union':means(z),'paired_difference_vs_U15':float(d.mean()),'paired_channel_95pct':np.quantile(boot,[.025,.975]).tolist(),
                      'selected_epoch':raw[name].get('selected_epoch'),'parameters_total':180,'online_full_analytic_calls':30 if 'only_seed' in name or name.startswith('Role') or name.startswith('U15') else None}
        if 'only_seed' in name:
            raw[name]['shared_union']=z;raw[name]['union_state']=vars(s);torch.save(raw[name],out/(name+'.pt'))
    write(out/'summary.json',{'methods':report,'scope':'Three paired fine-tuning shuffle seeds; one shared pretrained reference; selected on validation; channel CIs do not represent training-seed uncertainty'})


if __name__=='__main__':main()
