"""Controlled A/B/C/D continuation and matched-compute PGDA study.

Reuses saved audit tensors; additional pretrained seeds use existing experiment.py.
Stages are resumable. No source-paper claims are inferred from implementation.
"""
import argparse,copy,itertools,json,os,platform,subprocess,sys,time
from dataclasses import asdict,replace
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import torch
from .architecture_audit import read,write,sha,state_slice,means,tensor_metrics
from .controlled_training import ControlledSchedule,joint_update,separated_update
from .data import ChannelBatch,initialize_state
from .inference import load_checkpoint
from .physics import metrics
from .solver import SolverState,PGDA,unrolled_loss,NAMES,constraint_errors
from .attacks import attack,union_attack,ERRORS,project_errors
from .batched_attack import batched_attack
from .audit_artifacts import residuals
from .execution_evidence import numpy_metrics

def atomic_save(value,path):
    path=Path(path);temporary=path.with_name(path.name+'.tmp.'+str(os.getpid()))
    torch.save(value,temporary);os.replace(temporary,path)


COUNTERS=['full_analytic_calls','error_gradient_calls','differentiated_layers','differentiated_full_analytic_calls','backward_passes','optimizer_step_calls','inner_block_updates','outer_block_updates']


class ControlledStudy:
    def __init__(self,config='configs/pilot/controlled_ablation.json',output='results/controlled_ablation'):
        self.config_path=config;self.spec=read(config);self.out=Path(output);self.out.mkdir(parents=True,exist_ok=True)
        self.source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
        self.sources={str(p):sha(p) for p in Path('l2o_isac').glob('*.py')}
        self.split_hash=sha(self.spec['split_file']);raw=torch.load(self.spec['split_file'],weights_only=True)
        self.data={n:ChannelBatch(**x) for n,x in raw['channels'].items()};self.initial={n:SolverState(**x) for n,x in raw['initial'].items()}
        self.attack_step=read(self.spec['validation_attack_choices'])['attack_step']
        self.cfg,_,_,_=load_checkpoint(self.spec['reference_checkpoints'][0])

    def log(self,msg):
        line=datetime.now(timezone.utc).isoformat()+' '+msg;print(line,flush=True)
        with (self.out/'events.log').open('a') as f:f.write(line+'\n')

    def prepare(self):
        path=self.out/'frozen_protocol.json'
        payload={'spec':self.spec,'split_sha256':self.split_hash,'attack_step':self.attack_step,'system':asdict(self.cfg),
                 'scope':'Independent reimplementation controls; source PDF currently unavailable'}
        if path.exists():
            if read(path)!=payload:raise RuntimeError('Frozen protocol/splits mismatch; use new output folder')
        else:write(path,payload)

    def reference(self,index):
        p=self.spec['reference_checkpoints'][index];cfg,eps,ref,saved=load_checkpoint(p)
        if asdict(cfg)!=asdict(self.cfg):raise RuntimeError('Pretraining configuration mismatch')
        if ref.layers!=15 or saved['run'].get('eps_layout')!='per-channel':raise RuntimeError('Expected matched corrected reference')
        return ref,p

    @torch.no_grad()
    def val(self,model):
        b,s=self.data['validation'],self.initial['validation'];design=model(b,s)[0]
        out=attack(b,design,self.cfg,budgets=(self.spec['validation_attack_steps'],),restarts=0,adam_steps=0,step_size=self.attack_step)
        return out['metrics'][self.spec['validation_attack_steps']]['objective'].mean().item()

    def profile(self,clean=False):
        path=self.out/('compute_matching_clean.json' if clean else 'compute_matching.json')
        if path.exists():return
        ref,_=self.reference(0);b=self.data['validation'][:32];s=state_slice(self.initial['validation'],slice(0,32));measure={}
        for condition in ['B','C','D']:
            values=[]
            for rep in range(self.spec['profiling_repeats']+1):
                model=ControlledSchedule(ref,condition in ['C','D'],self.spec['positive_inner_floor'])
                opt=torch.optim.SGD(model.parameters(),lr=self.spec['lr']);io=torch.optim.SGD(model.inner_parameters(),lr=self.spec['lr']);oo=torch.optim.SGD(model.outer_parameters(),lr=self.spec['lr'])
                t=time.perf_counter()
                if condition=='D':separated_update(model,b,s,io,oo,self.spec['training_attack_steps'],self.attack_step,outer_loss_mode='joint' if clean else 'terminal')
                else:joint_update(model,b,s,opt)
                if rep:values.append(time.perf_counter()-t)
            measure[condition]={'median_seconds':float(np.median(values)),'samples_seconds':values}
        joint=float(np.mean([measure[n]['median_seconds'] for n in ['B','C']]))
        cycles=min(self.spec['joint_cycle_candidates'],key=lambda n:abs(joint*n-measure['D']['median_seconds']))
        write(path,{'profile_split':'validation first32','methods':measure,'joint_repeats_per_minibatch':cycles,
                    'predicted_joint_to_separate_ratio':joint*cycles/measure['D']['median_seconds'],
                    'scope':'Approximate wall-time matching of training updates; actual full-run costs separately reported; same cycles for B/C'})
        self.log('Frozen training-compute matching: '+str(cycles)+' joint repeats per minibatch')

    def train_case(self,index,condition,seed,equal_updates=False,balanced=False,clean=False):
        ref,refpath=self.reference(index);positive=condition in ['C','D'];model=ControlledSchedule(ref,positive,self.spec['positive_inner_floor'])
        profile_file=self.out/('compute_matching_clean.json' if clean else 'compute_matching.json')
        cycles=1
        if not balanced and not equal_updates:cycles=read(profile_file)['joint_repeats_per_minibatch']
        if balanced:
            prof=read(profile_file)['methods']
            cycles=float(prof['D']['median_seconds']/np.mean([prof[n]['median_seconds'] for n in ['B','C']]))
        if condition=='D':cycles=1
        name=condition+('_clean' if clean else '')+('_balanced' if balanced and not clean else '')+('_updates' if equal_updates else '')+f'_s{seed}';folder=self.out/f'pretrain{self.spec["pretraining_seeds"][index]}'/name;folder.mkdir(parents=True,exist_ok=True)
        optimizers={'joint':torch.optim.SGD(model.parameters(),lr=self.spec['lr']),
                    'inner':torch.optim.SGD(model.inner_parameters(),lr=self.spec['lr']),
                    'outer':torch.optim.SGD(model.outer_parameters(),lr=self.spec['lr'])}
        generator=torch.Generator().manual_seed(seed);history=[];counts={k:0 for k in COUNTERS};elapsed=0.;epoch_start=0
        best=self.val(model);initial_score=best
        latest,selected=folder/'latest.pt',folder/'selected.pt'
        def checkpoint(epoch):
            return {'model':copy.deepcopy(model.state_dict()),'optimizers':{n:o.state_dict() for n,o in optimizers.items()},
                    'torch_rng_state':torch.get_rng_state(),'batch_order_rng_state':generator.get_state(),'condition':condition,'positive_inner':positive,
                    'floor':self.spec['positive_inner_floor'],'seed':seed,'pretraining_seed':self.spec['pretraining_seeds'][index],
                    'reference':refpath,'reference_sha256':sha(refpath),'spec':self.spec,'system':asdict(self.cfg),'split_sha256':self.split_hash,
                    'source_commit':self.source,'source_sha256':self.sources,'epoch':epoch,'history':history,'training_update_seconds':elapsed,
                    'counts':counts,'cycles_per_minibatch':cycles,'initial_validation':initial_score,'best_validation':best,
                    'outer_loss_mode':'joint' if clean or condition!='D' else 'terminal',
                    'gradient_policy':'full deployed differentiation; inner policy frozen in outer phase' if clean else 'joint full differentiation' if condition!='D' else 'full deployed iterates; terminal independent adversary stop-gradient'}
        if latest.exists():
            old=torch.load(latest,weights_only=True)
            if old['reference_sha256']!=sha(refpath) or old['spec']!=self.spec or old['cycles_per_minibatch']!=cycles:raise RuntimeError('Resume provenance mismatch')
            model.load_state_dict(old['model']);generator.set_state(old['batch_order_rng_state']);torch.set_rng_state(old['torch_rng_state'])
            for n,o in optimizers.items():o.load_state_dict(old['optimizers'][n])
            history=old['history'];counts=old['counts'];elapsed=old['training_update_seconds'];epoch_start=old['epoch'];best=old['best_validation'];initial_score=old['initial_validation']
        else:
            atomic_save(checkpoint(0),folder/'initial.pt');atomic_save(checkpoint(0),selected)
        train=self.data['train'];initial=self.initial['train']
        for epoch in range(epoch_start,self.spec['epochs']):
            for opt in optimizers.values():
                for group in opt.param_groups:group['lr']=self.spec['lr']*self.spec['lr_decay']**epoch
            order=torch.randperm(len(train),generator=generator);start=time.perf_counter();batch_records=[]
            for batch_index,ids in enumerate(order.split(self.spec['batch_size'])):
                b,s=train[ids],state_slice(initial,ids)
                step_index=epoch*((len(train)+self.spec['batch_size']-1)//self.spec['batch_size'])+batch_index
                repeats=int(np.floor((step_index+1)*cycles)-np.floor(step_index*cycles)) if balanced else int(cycles)
                for _ in range(repeats):
                    t=time.perf_counter()
                    if condition=='D':r=separated_update(model,b,s,optimizers['inner'],optimizers['outer'],self.spec['training_attack_steps'],self.attack_step,outer_loss_mode='joint' if clean else 'terminal')
                    else:r=joint_update(model,b,s,optimizers['joint'])
                    r['seconds']=time.perf_counter()-t;batch_records.append(r)
                    for k in COUNTERS:counts[k]+=r[k]
            update_time=time.perf_counter()-start;elapsed+=update_time
            t=time.perf_counter();score=self.val(model);validation_seconds=time.perf_counter()-t
            improved=score>best
            if improved:best=score
            history.append({'epoch':epoch+1,'validation':score,'training_update_seconds':update_time,'validation_seconds':validation_seconds,
                            'batch_order_sha256':__import__('hashlib').sha256(order.numpy().tobytes()).hexdigest(),
                            'negative_inner_steps':int((model.inner_mu<0).sum()),'at_positive_floor':int((model.inner_mu<=model.floor).sum()),'updates':batch_records})
            atomic_save(checkpoint(epoch+1),latest)
            if improved:atomic_save(checkpoint(epoch+1),selected)
            write(folder/'summary.json',{'condition':condition,'initial_validation':initial_score,'best_validation':best,'selected_epoch':torch.load(selected,weights_only=True)['epoch'],
                  'history':history,'counts':counts,'training_update_seconds':elapsed,'positive_inner':positive,'cycles':cycles})
            self.log(f'pretrain={self.spec["pretraining_seeds"][index]} {name} epoch={epoch+1} validation={score:.6f}')
        return selected

    def train_index(self,index,pilot=False):
        for condition in ['B','C','D']:
            for seed in self.spec['paired_finetuning_seeds']:self.train_case(index,condition,seed,balanced=index>0 and condition in ['B','C'])
        if index==0:
            for condition in ['B','C']:self.train_case(0,condition,self.spec['paired_finetuning_seeds'][0],True)
        if pilot:write(self.out/'pilot_training_gate.json',{'passed':True,'scope':'Methodology/finite training validation, independent test evaluation follows; no positive-gain gate'})

    def cases(self,index):
        ref,refpath=self.reference(index);cases={'A':ControlledSchedule(ref),'Mapped_initial':ControlledSchedule(ref,True,self.spec['positive_inner_floor'])}
        files={}
        for label in (['B','C','B_balanced','C_balanced','D','B_clean','C_clean','D_clean'] if index==0 else ['B_balanced','C_balanced','D','B_clean','C_clean','D_clean']):
            condition=label[0]
            for seed in self.spec['paired_finetuning_seeds']:
                name=f'{label}_s{seed}';p=self.out/f'pretrain{self.spec["pretraining_seeds"][index]}'/name/'selected.pt'
                if not p.exists():continue
                saved=torch.load(p,weights_only=True);m=ControlledSchedule(ref,condition in ['C','D'],self.spec['positive_inner_floor']);m.load_state_dict(saved['model']);cases[name]=m;files[name]=p
        if index==0:
            for condition in ['B','C']:
                seed=self.spec['paired_finetuning_seeds'][0];name=f'{condition}_updates_s{seed}';p=self.out/f'pretrain{self.spec["pretraining_seeds"][index]}'/name/'selected.pt'
                m=ControlledSchedule(ref,condition=='C',self.spec['positive_inner_floor']);m.load_state_dict(torch.load(p,weights_only=True)['model']);cases[name]=m;files[name]=p
            for seed in self.spec['paired_finetuning_seeds']:
                p=Path(f'results/architecture_audit/role_seed{seed}/selected.pt');m=ControlledSchedule(ref);m.load_state_dict(torch.load(p,weights_only=True)['model']);cases[f'E_previous_s{seed}']=m;files[f'E_previous_s{seed}']=p
        return cases,files

    @torch.no_grad()
    def evaluate_index(self,index,clean_only=False):
        cases,files=self.cases(index)
        if clean_only:cases={n:m for n,m in cases.items() if '_clean_' in n}
        folder=self.out/f'pretrain{self.spec["pretraining_seeds"][index]}'/'evaluation';folder.mkdir(exist_ok=True)
        b,s=self.data['test'],self.initial['test']
        for name,model in cases.items():
            path=folder/(name+'.pt')
            if path.exists():
                old=torch.load(path,weights_only=True)
                if old['split_sha256']!=self.split_hash or old['spec']!=self.spec:raise RuntimeError('Cached evaluation mismatch')
                continue
            start=time.perf_counter();design=model(b,s)[0];generation_seconds=time.perf_counter()-start
            result=batched_attack(b,design,self.cfg,budgets=self.spec['attack_budgets'],restarts=self.spec['attack_restarts'],seed=self.spec['attack_seed'],
                                  step_size=self.attack_step,adam_steps=self.spec['adam_steps'])
            nominal=replace(design,**{n:getattr(design,n)*0 for n in ERRORS});term=project_errors(design,b,self.cfg)
            feas=residuals(result['state'],b,self.cfg);fail=torch.stack(list(feas.values())).amax(0)>1e-5
            if fail.any() or not all(torch.isfinite(v).all() for v in metrics(b,result['state'],self.cfg).values()):raise FloatingPointError('Controlled evaluation feasibility/nonfinite failure')
            payload={'design':vars(design),'attacked_state':vars(result['state']),'budget_metrics':result['metrics'],'nominal':tensor_metrics(b,nominal,self.cfg),
                'terminal':tensor_metrics(b,term,self.cfg),'internal':tensor_metrics(b,design,self.cfg),'feasibility_per_scenario':feas,'failure_flags':fail,
                'generation_batch_seconds':generation_seconds,'attack_seconds':result['seconds'],'tensor_error_gradient_calls':result['tensor_error_gradient_calls'],
                'logical_start_error_gradient_calls':result['logical_start_error_gradient_calls'],'design_full_analytic_calls':2*model.layers,
                'split_sha256':self.split_hash,'reference_sha256':sha(self.spec['reference_checkpoints'][index]),'model_checkpoint':str(files.get(name,'')),
                'spec':self.spec,'source_sha256':self.sources}
            atomic_save(payload,path)
            self.log(f'eval pretrain={self.spec["pretraining_seeds"][index]} {name}: 200={result["metrics"][200]["objective"].mean():.6f},1000={result["metrics"][1000]["objective"].mean():.6f},attack={result["seconds"]:.1f}s')
        self.union_results()

    @torch.no_grad()
    def union_results(self):
        entries={};designs={};candidates=[]
        for p in sorted(self.out.glob('pretrain*/evaluation/*.pt')):
            r=torch.load(p,weights_only=True);key=str(p.relative_to(self.out));entries[key]=(p,r);designs[key]=SolverState(**r['design']);candidates.append(SolverState(**r['attacked_state']))
            candidates.append(project_errors(designs[key],self.data['test'],self.cfg))
        # PGDA candidates join once computed; all methods receive identical pool.
        for p in sorted((self.out/'pgda').glob('budget*.pt')):
            r=torch.load(p,weights_only=True);key=str(p.relative_to(self.out));entries[key]=(p,r);designs[key]=SolverState(**r['design']);candidates.append(SolverState(**r['attacked_state']))
            candidates.append(project_errors(designs[key],self.data['test'],self.cfg))
        states=union_attack(self.data['test'],designs,candidates,self.cfg)
        for key,state in states.items():
            p,r=entries[key];r['shared_union']=tensor_metrics(self.data['test'],state,self.cfg);r['union_state']=vars(state)
            atomic_save(r,p)

    def additional_references(self):
        if not (self.out/'pilot_training_gate.json').exists():raise RuntimeError('Original-reference controls must validate before extra pretraining')
        for index in range(1,len(self.spec['reference_checkpoints'])):
            p=Path(self.spec['reference_checkpoints'][index]);folder=p.parent
            if p.exists():self.reference(index);continue
            if folder.exists():raise RuntimeError('Incomplete additional reference directory exists; preserve it and diagnose before rerunning')
            seed=self.spec['pretraining_seeds'][index]
            command=[sys.executable,'-u','-m','l2o_isac.experiment','--profile','paper','--seed',str(seed),'--output',str(folder)]
            self.log('Explicitly authorized additional independent pretraining: '+' '.join(command))
            start=time.perf_counter()
            with (self.out/f'pretraining_seed{seed}.log').open('w') as log:run=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
            write(self.out/f'pretraining_seed{seed}_execution.json',{'command':command,'returncode':run.returncode,'wall_seconds':time.perf_counter()-start,'source_commit':self.source,'source_sha256':self.sources,
                   'original_checkpoint_preserved':sha(self.spec['reference_checkpoints'][0]),'checkpoint':str(p),'checkpoint_sha256':sha(p) if p.exists() else None})
            if run.returncode:raise RuntimeError('Additional pretraining failed; log retained')
            self.reference(index)

    @torch.no_grad()
    def inner_validation(self,index):
        ref,_=self.reference(index);cases,files=self.cases(index);b=self.data['validation'];s=self.initial['validation'];fixed=ref(b,s)[0]
        folder=self.out/f'pretrain{self.spec["pretraining_seeds"][index]}'/'fixed_beam_inner_validation';folder.mkdir(exist_ok=True)
        for name in ['Mapped_initial']+[f'D_s{x}' for x in self.spec['paired_finetuning_seeds']]+[f'D_clean_s{x}' for x in self.spec['paired_finetuning_seeds'] if f'D_clean_s{x}' in cases]:
            path=folder/(name+'.pt')
            if path.exists():continue
            errors=cases[name].inner_attack(b,fixed);independent=numpy_metrics(b,errors,self.cfg)
            result=batched_attack(b,errors,self.cfg,budgets=(200,),restarts=5,seed=9001,step_size=self.attack_step,adam_steps=50)
            payload={'policy_state':vars(errors),'frozen_beamformers':{n:getattr(fixed,n) for n in NAMES[3:]},
                     'numpy_policy_metrics':{k:torch.tensor(v) for k,v in independent.items()},'independent_refinement_metrics':result['metrics'][200],
                     'refined_state':vars(result['state']),'scope':'Same frozen A validation beams; whole15-step MinNet evaluated by independent NumPy, then refined with independent search'}
            atomic_save(payload,path)
            self.log(f'fixed-beam inner validation pretrain={self.spec["pretraining_seeds"][index]} {name}: candidate={independent["objective"].mean():.6f}, refined={result["metrics"][200]["objective"].mean():.6f}')

    @torch.no_grad()
    def pgda(self):
        folder=self.out/'pgda';folder.mkdir(exist_ok=True);choice_path=folder/'validation_choices.json';b=self.data['validation'][:24];s=state_slice(self.initial['validation'],slice(0,24))
        if not choice_path.exists():
            all_choices={};started=time.perf_counter()
            for budget in self.spec['pgda_budgets']:
                choices=[]
                for mi,mo in itertools.product(self.spec['pgda_inner_grid'],self.spec['pgda_outer_grid']):
                    state=PGDA(self.cfg,iterations=budget,mu=[mi]*3+[mo]*3)(b,s)[0]
                    result=attack(b,state,self.cfg,budgets=(30,),restarts=0,adam_steps=0,step_size=self.attack_step)
                    choices.append({'inner':mi,'outer':mo,'validation_utility':result['metrics'][30]['objective'].mean().item()})
                all_choices[str(budget)]={'selected':max(choices,key=lambda x:x['validation_utility']),'all':choices}
            write(choice_path,{'budgets':all_choices,'selection_scenarios':24,'configurations_searched':len(self.spec['pgda_budgets'])*len(self.spec['pgda_inner_grid'])*len(self.spec['pgda_outer_grid']),
                  'seconds':time.perf_counter()-started,'test_used_for_tuning':False})
        choices=read(choice_path)['budgets'];b=self.data['test'];s=self.initial['test']
        for budget in self.spec['pgda_budgets']:
            p=folder/f'budget{budget}.pt'
            if p.exists():continue
            selected=choices[str(budget)]['selected'];solver=PGDA(self.cfg,iterations=budget,mu=[selected['inner']]*3+[selected['outer']]*3)
            t=time.perf_counter();design=solver(b,s)[0];elapsed=time.perf_counter()-t
            result=batched_attack(b,design,self.cfg,budgets=self.spec['attack_budgets'],restarts=5,adam_steps=50,seed=9001,step_size=self.attack_step)
            feas=residuals(result['state'],b,self.cfg);flags=torch.stack(list(feas.values())).amax(0)>1e-5
            if flags.any():raise FloatingPointError('PGDA feasibility failure')
            nominal=replace(design,**{n:getattr(design,n)*0 for n in ERRORS})
            atomic_save({'design':vars(design),'attacked_state':vars(result['state']),'budget_metrics':result['metrics'],'generation_batch_seconds':elapsed,
                       'attack_seconds':result['seconds'],'design_full_analytic_calls':2*budget,'nominal':tensor_metrics(b,nominal,self.cfg),
                       'terminal':tensor_metrics(b,design,self.cfg),'feasibility_per_scenario':feas,'failure_flags':flags,
                       'spec':self.spec,'split_sha256':self.split_hash,'source_sha256':self.sources},p)
            self.log(f'PGDA budget={budget} calls={2*budget} independent1000={result["metrics"][1000]["objective"].mean():.6f}')
        self.union_results()

    @torch.no_grad()
    def timing(self):
        b=self.data['test'][:1];s=state_slice(self.initial['test'],slice(0,1));functions={};results={}
        cases,_=self.cases(0)
        for name,m in cases.items():functions[name]=lambda m=m:m(b,s)
        choices=read(self.out/'pgda/validation_choices.json')['budgets']
        for budget in self.spec['pgda_budgets']:
            c=choices[str(budget)]['selected'];solver=PGDA(self.cfg,iterations=budget,mu=[c['inner']]*3+[c['outer']]*3)
            functions[f'PGDA{budget}']=lambda solver=solver:solver(b,s)
        _,eps,_,_=load_checkpoint(self.spec['reference_checkpoints'][0]);functions['EpsNet_only']=lambda:eps(b);functions['Initialization_only']=lambda:initialize_state(b,self.cfg,seed=843103)
        for name,fn in functions.items():
            for _ in range(5):fn()
            samples=[]
            for _ in range(self.spec['timing_repeats']):
                t=time.perf_counter();fn();samples.append((time.perf_counter()-t)*1000)
            results[name]={'median_ms':float(np.median(samples)),'p95_ms':float(np.quantile(samples,.95)),'samples_ms':samples}
        write(self.out/'timing.json',{'batch_size':1,'threads':1,'precision':'complex64','warmup':5,'repeats':self.spec['timing_repeats'],
             'hardware':read('results/architecture_audit/environment.json'),'scope':'CPU solver generation only, no GPU, independent attacker excluded; initialization/EpsNet separate','methods':results})


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/pilot/controlled_ablation.json');p.add_argument('--output',default='results/controlled_ablation')
    p.add_argument('--stage',choices=['prepare','profile','pilot','balanced','pretrain','expanded','pgda','timing','condition','clean','all'],default='all');p.add_argument('--condition',choices=['B','C','D']);p.add_argument('--reference-index',type=int,default=0);p.add_argument('--fine-seed',type=int,default=101);a=p.parse_args(argv)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(901)
    study=ControlledStudy(a.config,a.output);study.prepare()
    try:
        if a.stage in ['profile','pilot','all']:study.profile()
        if a.stage in ['pilot','all']:study.train_index(0,True);study.inner_validation(0);study.evaluate_index(0)
        if a.stage in ['balanced','all']:
            for condition in ['B','C']:
                for seed in study.spec['paired_finetuning_seeds']:study.train_case(0,condition,seed,balanced=True)
            study.evaluate_index(0)
        if a.stage=='condition':
            if a.condition is None:p.error('--condition is required for condition stage')
            study.profile();study.train_case(a.reference_index,a.condition,a.fine_seed,balanced=a.condition in ['B','C'])
        if a.stage in ['pretrain','all']:study.additional_references()
        if a.stage in ['expanded','all']:
            for index in range(1,len(study.spec['reference_checkpoints'])):study.train_index(index);study.inner_validation(index);study.evaluate_index(index)
        if a.stage in ['clean','all']:
            study.profile(clean=True)
            for index in range(len(study.spec['reference_checkpoints'])):
                for condition in ['B','C','D']:
                    for seed in study.spec['paired_finetuning_seeds']:study.train_case(index,condition,seed,balanced=condition in ['B','C'],clean=True)
                study.inner_validation(index);study.evaluate_index(index,clean_only=True)
        if a.stage in ['pgda','all']:study.pgda()
        if a.stage in ['timing','all']:study.timing()
    except Exception as exc:
        study.log('FAILED '+type(exc).__name__+': '+str(exc));raise


if __name__=='__main__':main()
