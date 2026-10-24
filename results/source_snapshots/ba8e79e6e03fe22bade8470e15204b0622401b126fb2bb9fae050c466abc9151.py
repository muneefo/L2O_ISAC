"""Resumable audit/extension pilot reusing the immutable completed checkpoint.

Stages: prepare, tune, audit, train, evaluate, figures, all. No reference training.
All tensors and JSON stay under the chosen new output directory.
"""
import argparse
import copy
from dataclasses import asdict,replace
from datetime import datetime,timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import numpy as np
import torch
from .attacks import attack,union_attack,ERRORS,project_errors
from .config import SystemConfig
from .data import ChannelBatch,generate_channels,initialize_state
from .inference import load_checkpoint
from .physics import gradients,metrics
from .role_training import RoleSeparatedPGDA
from .solver import NAMES,PGDA,SolverState,constraint_errors,project_ball,project_beamformers


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,x):Path(path).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def read(path):return json.loads(Path(path).read_text())
def tensor_metrics(batch,state,cfg):return {k:v.detach().cpu() for k,v in metrics(batch,state,cfg).items()}
def means(z):return {k:float(v.mean()) for k,v in z.items()}
def state_slice(s,ids):return SolverState(**{n:getattr(s,n)[ids] for n in NAMES})


@torch.no_grad()
def classical_strong(batch,initial,cfg,outer_steps,mu_inner,mu_outer,k=4):
    cur=initial
    for _ in range(outer_steps):
        for _ in range(k):
            g=gradients(batch,cur,cfg)
            cur=project_errors(replace(cur,**{n:getattr(cur,n)-mu_inner*g[n] for n in ERRORS}),batch,cfg)
        g=gradients(batch,cur,cfg)
        a,d,w=project_beamformers(*[getattr(cur,n)+mu_outer*g[n] for n in NAMES[3:]],cfg.power)
        cur=replace(cur,a=a,d=d,w=w)
    return cur


class Study:
    def __init__(self,config,output):
        self.spec=read(config);self.output=Path(output);self.output.mkdir(parents=True,exist_ok=True)
        self.config_path=str(config);self.cfg,self.epsnet,self.reference,self.saved=load_checkpoint(self.spec['checkpoint'])
        self.sha=sha(self.spec['checkpoint']);self.source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
        self.commit_sources={str(p):sha(p) for p in sorted(Path('l2o_isac').glob('*.py'))}

    def event(self,msg):
        line=datetime.now(timezone.utc).isoformat()+' '+msg
        print(line,flush=True)
        with (self.output/'events.log').open('a') as f:f.write(line+'\n')

    def prepare(self):
        path=self.output/'splits.pt'
        if path.exists():
            manifest=read(self.output/'split_manifest.json')
            if sha(path)!=manifest['sha256']:raise RuntimeError('Immutable split hash mismatch')
            if manifest['spec']!=self.spec:raise RuntimeError('Existing split uses a different protocol')
            return
        values={};init={}
        for name,size in self.spec['split_sizes'].items():
            batch=generate_channels(self.cfg,size,seed=self.spec['split_seeds'][name])
            values[name]=vars(batch);init[name]=vars(initialize_state(batch,self.cfg,seed=self.spec['initialization_seeds'][name]))
        torch.save({'channels':values,'initial':init},path)
        write(self.output/'split_manifest.json',{'sha256':sha(path),'spec':self.spec,'reference_sha256':self.sha,
              'scope':'Separate new audit splits; original reproduction data are not regenerated','system':asdict(self.cfg)})
        write(self.output/'environment.json',{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,
              'platform':platform.platform(),'cpu':next((x.split(':',1)[1].strip() for x in Path('/proc/cpuinfo').read_text().splitlines() if x.startswith('model name')),'unknown'),
              'cpu_quota':Path('/sys/fs/cgroup/cpu.max').read_text().strip(),'threads':torch.get_num_threads(),
              'cuda_available':torch.cuda.is_available(),'precision':'complex64 experiments; complex128 numerical tests',
              'reference_sha256':self.sha,'source_commit':self.source,'source_sha256':self.commit_sources})
        self.event('Saved disjoint immutable audit splits; no reference retraining')

    def split(self,name):
        raw=torch.load(self.output/'splits.pt',weights_only=True)
        return ChannelBatch(**raw['channels'][name]),SolverState(**raw['initial'][name])

    @torch.no_grad()
    def tune(self):
        path=self.output/'validation_choices.json'
        if path.exists():return
        batch,initial=self.split('validation');batch=batch[:24];initial=state_slice(initial,slice(0,24))
        design=self.reference(batch,initial)[0];records=[]
        for lr in self.spec['attack_step_candidates']:
            out=attack(batch,design,self.cfg,budgets=(self.spec['validation_attack_steps'],),restarts=1,
                       adam_steps=10,step_size=lr,seed=9010)
            records.append({'step':lr,'objective':out['metrics'][self.spec['validation_attack_steps']]['objective'].mean().item()})
        step=min(records,key=lambda x:x['objective'])['step'];families={}
        # Separate tuning of classical families, with all candidates retained.
        for name,outer,k in [('PGDA15',15,1),('PGDA180',180,1),('StrongInnerMatched',6,4),('StrongInner15',15,4)]:
            candidates=[]
            for mi,mo in itertools.product(self.spec['classical_inner_steps'],self.spec['classical_outer_steps']):
                if k==1:s=PGDA(self.cfg,iterations=outer,mu=[mi]*3+[mo]*3)(batch,initial)[0]
                else:s=classical_strong(batch,initial,self.cfg,outer,mi,mo,k)
                out=attack(batch,s,self.cfg,budgets=(20,),restarts=0,adam_steps=0,step_size=step)
                candidates.append({'inner':mi,'outer':mo,'objective':out['metrics'][20]['objective'].mean().item()})
            families[name]={'selected':max(candidates,key=lambda x:x['objective']),'candidates':candidates,
                            'outer_steps':outer,'inner_per_outer':k,'full_analytic_calls':outer*(k+1)}
        write(path,{'attack_step':step,'attack_candidates':records,'families':families,
                    'selection_split':'validation first 24 only','test_used':False,'frozen_utc':datetime.now(timezone.utc).isoformat()})
        self.event('Validation tuning frozen before test; '+json.dumps({k:v['selected'] for k,v in families.items()}))

    @torch.no_grad()
    def methods(self,batch,initial,include_roles=False):
        choices=read(self.output/'validation_choices.json');states={};calls={};seconds={}
        for name,item in choices['families'].items():
            mi,mo=item['selected']['inner'],item['selected']['outer'];started=time.perf_counter()
            if item['inner_per_outer']==1:s=PGDA(self.cfg,iterations=item['outer_steps'],mu=[mi]*3+[mo]*3)(batch,initial)[0]
            else:s=classical_strong(batch,initial,self.cfg,item['outer_steps'],mi,mo,item['inner_per_outer'])
            states[name]=s;calls[name]=item['full_analytic_calls'];seconds[name]=time.perf_counter()-started
        for name,kwargs in [('U15',{}),('RU25',dict(extra_layers=10,beta=self.saved['recurrence']['beta'],phi=self.saved['recurrence']['phi'])),
                            ('U15_EpsNet',{'eps':self.epsnet(batch)})]:
            started=time.perf_counter();states[name]=self.reference(batch,initial,**kwargs)[0]
            calls[name]=50 if name=='RU25' else 30;seconds[name]=time.perf_counter()-started
        if include_roles:
            for seed in self.spec['training_seeds']:
                path=self.output/f'role_seed{seed}'/'selected.pt'
                saved=torch.load(path,weights_only=True);model=RoleSeparatedPGDA(self.reference);model.load_state_dict(saved['model'])
                started=time.perf_counter();name=f'Role_seed{seed}';states[name]=model(batch,initial)[0]
                calls[name]=30;seconds[name]=time.perf_counter()-started
        return states,calls,seconds

    @torch.no_grad()
    def evaluate(self,roles=False):
        folder=self.output/('extension_evaluation' if roles else 'adversarial_audit');folder.mkdir(exist_ok=True)
        batch,initial=self.split('test');designs,calls,seconds=self.methods(batch,initial,roles)
        records={};raw={};candidates=[]
        step=read(self.output/'validation_choices.json')['attack_step']
        for name,design in designs.items():
            target=folder/(name+'.pt')
            cached = target if target.exists() else self.output/'adversarial_audit'/(name+'.pt')
            if cached.exists():
                entry=torch.load(cached,weights_only=True)
                if entry['split_sha256']!=sha(self.output/'splits.pt'):raise RuntimeError('Split mismatch')
                records[name]=entry['summary'];raw[name]=entry;candidates.extend([SolverState(**entry['attacked_state']),SolverState(**entry['terminal_state'])]);continue
            started=time.perf_counter();terminal=project_errors(design,batch,self.cfg)
            out=attack(batch,design,self.cfg,budgets=self.spec['attack_budgets'],restarts=self.spec['random_restarts'],
                       adam_steps=self.spec['adam_steps'],step_size=step,seed=9001)
            term=tensor_metrics(batch,terminal,self.cfg);found=out['metrics'][max(self.spec['attack_budgets'])]
            gap={k:term[k]-found[k] for k in term}
            if min(v.min().item() for v in gap.values()) < -1e-5:raise RuntimeError('Negative attack gap')
            nominal=replace(design,**{n:getattr(design,n)*0 for n in ERRORS})
            residual=constraint_errors(out['state'],batch,self.cfg)
            design_residual=constraint_errors(terminal,batch,self.cfg)
            if max(residual.values())>1e-5 or max(design_residual.values())>1e-5:raise RuntimeError('Feasibility gate failed')
            summary={'nominal':means(tensor_metrics(batch,nominal,self.cfg)),'internal_unprojected':means(tensor_metrics(batch,design,self.cfg)),
                     'terminal_common_set':means(term),'independent':means(found),'gap':means(gap),
                     'gap_quantiles':torch.quantile(gap['objective'],torch.tensor([0.,.25,.5,.75,.95,1.])).tolist(),
                     'budget_means':{str(b):means(z) for b,z in out['metrics'].items()},'constraint_residuals':residual,
                     'design_constraint_residuals':design_residual,'original_error_ball_violation':{k:v for k,v in constraint_errors(design,batch,self.cfg).items() if 'violation' in k},
                     'full_analytic_design_calls':calls[name],'attack_error_gradient_calls':out['gradient_evaluations'],
                     'attack_seconds':out['seconds'],'design_batch_seconds':seconds[name],'failures':0}
            entry={'summary':summary,'design':vars(design),'terminal_state':vars(terminal),'attacked_state':vars(out['state']),
                   'nominal':tensor_metrics(batch,nominal,self.cfg),'terminal':term,'independent':found,'gap':gap,
                   'budget_metrics':out['metrics'],'split_sha256':sha(self.output/'splits.pt'),
                   'reference_sha256':self.sha,'source_sha256':self.commit_sources,'parameters':self.spec}
            torch.save(entry,target);raw[name]=entry;records[name]=summary;candidates.extend([out['state'],terminal])
            self.event(f'{name}: independent {summary["independent"]}, gap={summary["gap"]["objective"]:.5f}, attack_s={time.perf_counter()-started:.1f}')
        union=union_attack(batch,designs,candidates,self.cfg)
        for name,state in union.items():
            raw[name]['union']=tensor_metrics(batch,state,self.cfg);raw[name]['union_state']=vars(state)
            records[name]['union']=means(raw[name]['union']);torch.save(raw[name],folder/(name+'.pt'))
        comparison={}
        baseline=raw['U15']['union']['objective'].numpy();rng=np.random.default_rng(9110)
        for name,entry in raw.items():
            diff=entry['union']['objective'].numpy()-baseline
            boot=diff[rng.integers(0,len(diff),size=(2000,len(diff)))].mean(1)
            comparison[name]={'paired_mean_difference_vs_U15':float(diff.mean()),'channel_bootstrap_95pct':np.quantile(boot,[.025,.975]).tolist(),
                              'scope':'Channel sampling only for fixed checkpoints; not training-seed CI'}
        role_diffs=[comparison[f'Role_seed{s}']['paired_mean_difference_vs_U15'] for s in self.spec['training_seeds']] if roles else []
        report={'methods':records,'paired_comparisons':comparison,'samples':len(batch),'radius_mode':'oracle except separately named EpsNet branch',
                'training_seed_differences':role_diffs,'training_seed_difference_sd':float(np.std(role_diffs,ddof=1)) if roles else None,
                'reference_pretraining_seeds':1,'reference_sha256':self.sha,'claim':'Lowest feasible utility found; no global minimum certification',
                'H1_supported_for_reconstruction':records['U15']['gap']['objective']>1e-3}
        write(folder/'summary.json',report)
        self.event('Gate D passed: frozen designs feasible, nested candidates retained; H1='+str(report['H1_supported_for_reconstruction']))
        return report

    @torch.no_grad()
    def validation_score(self,model):
        b,s=self.split('validation');design=model(b,s)[0]
        result=attack(b,design,self.cfg,budgets=(self.spec['validation_attack_steps'],),restarts=0,adam_steps=0,
                      step_size=read(self.output/'validation_choices.json')['attack_step'])
        return result['metrics'][self.spec['validation_attack_steps']]['objective'].mean().item()

    def train_one(self,seed,lr,folder):
        folder.mkdir(exist_ok=True);latest=folder/'latest.pt';selected=folder/'selected.pt'
        model=RoleSeparatedPGDA(self.reference);inner_opt=torch.optim.SGD(model.inner_parameters(),lr=self.spec['inner_lr'])
        outer_opt=torch.optim.SGD(model.outer_parameters(),lr=lr);generator=torch.Generator().manual_seed(seed)
        train,initial=self.split('train');valid,vinit=self.split('validation');start_epoch=0;history=[];elapsed=0.
        best=self.validation_score(model);initial_score=best
        with torch.no_grad():fixed_design=self.reference(valid,vinit)[0]
        def inner_score():
            with torch.no_grad():return metrics(valid,model.inner_attack(valid,fixed_design),self.cfg)['objective'].mean().item()
        initial_inner=inner_score()
        def checkpoint(epoch):
            return {'model':copy.deepcopy(model.state_dict()),'inner_optimizer':inner_opt.state_dict(),'outer_optimizer':outer_opt.state_dict(),
                    'torch_rng_state':torch.get_rng_state(),'shuffle_rng_state':generator.get_state(),'config':asdict(self.cfg),
                    'spec':self.spec,'outer_lr':lr,'seed':seed,'epoch':epoch,'best_validation':best,'initial_validation':initial_score,
                    'initial_inner_validation':initial_inner,'history':history,'training_seconds':elapsed,
                    'reference_checkpoint_sha256':self.sha,'source_commit':self.source,'source_sha256':self.commit_sources,
                    'split_identifiers':read(self.output/'split_manifest.json'),'outer_adversary_gradient':'terminal stop-gradient; full deployed trajectory gradients'}
        if latest.exists():
            old=torch.load(latest,weights_only=True)
            if old['spec']!=self.spec or old['outer_lr']!=lr or old['reference_checkpoint_sha256']!=self.sha:raise RuntimeError('Resume provenance mismatch')
            model.load_state_dict(old['model']);inner_opt.load_state_dict(old['inner_optimizer']);outer_opt.load_state_dict(old['outer_optimizer'])
            generator.set_state(old['shuffle_rng_state']);torch.set_rng_state(old['torch_rng_state'])
            start_epoch=old['epoch'];history=old['history'];best=old['best_validation'];initial_score=old['initial_validation'];initial_inner=old['initial_inner_validation'];elapsed=old['training_seconds']
        else:torch.save(checkpoint(0),selected)
        for epoch in range(start_epoch,self.spec['epochs']):
            started=time.perf_counter();losses=[]
            for ids in torch.randperm(len(train),generator=generator).split(self.spec['batch_size']):
                batch=train[ids];init=state_slice(initial,ids)
                with torch.no_grad():design=model(batch,init)[0]
                model.set_phase('inner');inner_opt.zero_grad()
                inner_loss=metrics(batch,model.inner_attack(batch,design),self.cfg)['objective'].mean()
                inner_loss.backward();self.check_grads(model);inner_opt.step()
                model.set_phase('outer');outer_opt.zero_grad()
                state=model(batch,init)[0]
                adverse=attack(batch,state,self.cfg,budgets=(self.spec['training_attack_steps'],),restarts=0,adam_steps=0,
                               step_size=read(self.output/'validation_choices.json')['attack_step'])['state']
                surrogate=replace(state,**{n:getattr(adverse,n).detach() for n in ERRORS})
                outer_loss=-metrics(batch,surrogate,self.cfg)['objective'].mean()
                outer_loss.backward();self.check_grads(model);outer_opt.step()
                losses.append([inner_loss.item(),outer_loss.item()])
            value=self.validation_score(model);inner_value=inner_score();elapsed+=time.perf_counter()-started
            history.append({'epoch':epoch+1,'inner_train_mean':float(np.mean(losses,axis=0)[0]),'outer_train_loss':float(np.mean(losses,axis=0)[1]),
                            'independent_validation':value,'fixed_beam_inner_validation':inner_value,'epoch_seconds':time.perf_counter()-started})
            improved=value>best
            if improved:best=value
            torch.save(checkpoint(epoch+1),latest)
            if improved:torch.save(checkpoint(epoch+1),selected)
            write(folder/'training.json',{'history':history,'initial_validation':initial_score,'best_validation':best,
                                         'initial_inner_validation':initial_inner,'final_inner_validation':inner_value,
                                         'selected_epoch':torch.load(selected,weights_only=True)['epoch'],'seconds':elapsed})
            self.event(f'role seed={seed} lr={lr} epoch={epoch+1}: val={value:.6f}, fixed-beam inner={inner_value:.6f}')
        return best

    @staticmethod
    def check_grads(model):
        for p in model.parameters():
            if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all()):raise FloatingPointError('Missing/nonfinite role gradient')

    def train(self):
        if not (self.output/'adversarial_audit'/'summary.json').exists():raise RuntimeError('Gate D must pass before training')
        path=self.output/'training_choices.json'
        if not path.exists():
            scores=[]
            for lr in self.spec['outer_lr_candidates']:
                value=self.train_one(self.spec['training_seeds'][0],lr,self.output/f'tune_outer_{lr}')
                scores.append({'lr':lr,'validation':value})
            chosen=max(scores,key=lambda x:x['validation'])['lr']
            write(path,{'candidates':scores,'selected_outer_lr':chosen,'selection_split':'validation only','test_used':False})
        lr=read(path)['selected_outer_lr']
        for seed in self.spec['training_seeds']:self.train_one(seed,lr,self.output/f'role_seed{seed}')

    @torch.no_grad()
    def timing(self):
        b,s=self.split('test');b=b[:1];s=state_slice(s,slice(0,1));choice=read(self.output/'validation_choices.json');results={}
        functions={'U15':lambda:self.reference(b,s),'RU25':lambda:self.reference(b,s,extra_layers=10,beta=self.saved['recurrence']['beta'],phi=self.saved['recurrence']['phi']),
                   'EpsNet_only':lambda:self.epsnet(b),'initialization_only':lambda:initialize_state(b,self.cfg,seed=843103)}
        for name,item in choice['families'].items():
            mi,mo=item['selected']['inner'],item['selected']['outer']
            if item['inner_per_outer']==1:
                solver=PGDA(self.cfg,iterations=item['outer_steps'],mu=[mi]*3+[mo]*3)
                functions[name]=lambda solver=solver:solver(b,s)
            else:functions[name]=lambda item=item,mi=mi,mo=mo:classical_strong(b,s,self.cfg,item['outer_steps'],mi,mo,item['inner_per_outer'])
        for seed in self.spec['training_seeds']:
            model=RoleSeparatedPGDA(self.reference);model.load_state_dict(torch.load(self.output/f'role_seed{seed}'/'selected.pt',weights_only=True)['model'])
            functions[f'Role_seed{seed}']=lambda model=model:model(b,s)
        for name,fn in functions.items():
            for _ in range(3):fn()
            times=[]
            for _ in range(self.spec['timing_repeats']):
                t=time.perf_counter();fn();times.append((time.perf_counter()-t)*1000)
            results[name]={'median_ms':float(np.median(times)),'p95_ms':float(np.quantile(times,.95)),'samples_ms':times}
        write(self.output/'timing.json',{'batch_size':1,'threads':1,'precision':'complex64','scope':'Warm CPU computation; solver includes projections; initialization and EpsNet separately measured; no RF/CSI/transfer cost','methods':results})

    def figures(self):
        os.environ.setdefault('MPLCONFIGDIR','/tmp/isac-audit-mpl')
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        folder=self.output/'extension_evaluation'
        if not (folder/'summary.json').exists():folder=self.output/'adversarial_audit'
        report=read(folder/'summary.json');names=list(report['methods']);rows=report['methods'];x=np.arange(len(names))
        fig,ax=plt.subplots(2,2,figsize=(14,9))
        for key,label in [('terminal_common_set','Terminal'),('independent','Independent'),('union','Shared union')]:
            ax[0,0].plot(x,[rows[n][key]['objective'] for n in names],marker='o',label=label)
        ax[0,0].set_ylabel('Weighted utility (nats)');ax[0,0].legend();ax[0,0].set_xticks(x,names,rotation=55,ha='right')
        for name in names:
            r=torch.load(folder/(name+'.pt'),weights_only=True)
            ax[0,1].plot(self.spec['attack_budgets'],[r['budget_metrics'][b]['objective'].mean().item() for b in self.spec['attack_budgets']],marker='o',label=name)
            ax[1,0].hist(r['gap']['objective'].numpy(),bins=20,histtype='step',label=name)
        ax[0,1].set_xlabel('PGD steps per start (plus fixed Adam)');ax[0,1].set_ylabel('Lowest-found weighted utility');ax[0,1].legend(fontsize=7)
        ax[1,0].set_xlabel('Terminal minus independent utility');ax[1,0].set_ylabel('Scenario count')
        for name in names:ax[1,1].scatter(rows[name]['full_analytic_design_calls'],rows[name]['union']['objective'],label=name)
        ax[1,1].set_xlabel('Full analytic gradient calls for design');ax[1,1].set_ylabel('Union lowest-found utility');ax[1,1].legend(fontsize=7)
        fig.tight_layout();fig.savefig(self.output/'audit_figures.png',dpi=170);fig.savefig(self.output/'audit_figures.pdf');plt.close(fig)
        if (self.output/'timing.json').exists():
            timing=read(self.output/'timing.json')['methods'];fig,ax=plt.subplots(figsize=(9,5))
            for name in names:
                if name in timing:ax.scatter(timing[name]['median_ms'],rows[name]['union']['objective'],label=name)
            ax.set_xlabel('Measured warm batch-one median ms');ax.set_ylabel('Union lowest-found utility');ax.legend(fontsize=8);fig.tight_layout();fig.savefig(self.output/'utility_vs_latency.png',dpi=170);plt.close(fig)
        self.event('Generated figures from saved measurements')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/pilot/architecture_audit.json');parser.add_argument('--output',default='results/architecture_audit')
    parser.add_argument('--stage',choices=['prepare','tune','audit','train','evaluate','figures','all'],default='all')
    args=parser.parse_args(argv);torch.set_num_threads(1);torch.manual_seed(888)
    study=Study(args.config,args.output)
    try:
        study.prepare()
        if args.stage in ('tune','all'):study.tune()
        if args.stage in ('audit','all'):study.evaluate(False)
        if args.stage in ('train','all'):study.train()
        if args.stage in ('evaluate','all'):study.evaluate(True);study.timing()
        if args.stage in ('figures','all'):study.figures()
    except Exception as exc:
        study.event('FAILED '+type(exc).__name__+': '+str(exc));raise


if __name__=='__main__':main()
