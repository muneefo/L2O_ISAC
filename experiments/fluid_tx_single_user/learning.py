"""Gated position-only L2O; one-scalar equal-training and 20-scalar controls."""
import argparse,copy,hashlib,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from l2o_isac.architecture_audit import read,write,sha,state_slice
from l2o_isac.batched_attack import batched_attack
from l2o_isac.attacks import ERRORS
from l2o_isac.solver import SolverState
from l2o_isac.physics import metrics
from dataclasses import replace
from .run import Pilot,ROOT,save
from .optimizer import solve
from .spatial import channels,system


class PositionSchedule(nn.Module):
    def __init__(self,pilot,case,settings,initial_step,constant=False):
        super().__init__();self.case=case;self.settings=settings;self.spec=pilot.spec;self.cfg=system(pilot.spec,case)
        self.choice=read(pilot.out/'validation_choices.json')['cases'][case]['ula']['selected']
        self.mu=nn.Parameter(torch.full((1 if constant else settings['layers'],),float(initial_step)))
        self.constant=constant
    @property
    def steps(self):return self.mu.expand(self.settings['layers']) if self.constant else self.mu
    def forward(self,scenario,x,initial):
        return solve(scenario,x,initial,self.cfg,self.spec,self.settings['layers'],self.choice['inner'],self.choice['outer'],
                     safeguard=False,position_schedule=self.steps)
    @torch.no_grad()
    def project(self):self.mu.clamp_(*self.settings['step_bounds'])


def select_rows(raw,ids):
    return raw['ula'][ids],SolverState(**{n:v[ids] for n,v in raw['initial'].items()})


@torch.no_grad()
def validation(pilot,model,settings):
    scenario,raw=pilot.data(model.case,'validation');x,initial=select_rows(raw,slice(None))
    r=model(scenario,x,initial);batch=channels(scenario,r['x'],model.cfg)
    attack=batched_attack(batch,r['state'],model.cfg,budgets=(settings['validation_attack_steps'],),restarts=settings['validation_random_restarts'],adam_steps=0,step_size=pilot.spec['attack_step'],seed=908100)
    return attack['metrics'][settings['validation_attack_steps']]['objective'].mean().item()


@torch.no_grad()
def tune_plain(pilot,settings):
    path=pilot.out/'learning_constant_validation.json'
    if path.exists():return read(path)
    rows={}
    for case in ['A','B','C']:
        grid=[]
        for step in settings['constant_step_grid']:
            model=PositionSchedule(pilot,case,settings,step,constant=True)
            grid.append({'step':step,'validation':validation(pilot,model,settings)})
        rows[case]={'selected':max(grid,key=lambda v:v['validation']),'grid':grid}
    result={'cases':rows,'selection_split':'validation only','test_used':False,'settings':settings};write(path,result);return result


def train_one(pilot,case,settings,variant,seed,initial_step):
    folder=pilot.out/'learning'/case/(variant+'_seed'+str(seed));folder.mkdir(parents=True,exist_ok=True)
    model=PositionSchedule(pilot,case,settings,initial_step,variant=='LearnedConstant');optimizer=torch.optim.Adam(model.parameters(),lr=settings['lr'])
    generator=torch.Generator().manual_seed(seed);scenario,raw=pilot.data(case,'train');best=validation(pilot,model,settings);initial_score=best
    history=[];start_epoch=0;elapsed=0.;updates=0
    def checkpoint(epoch):
        return {'model':copy.deepcopy(model.state_dict()),'optimizer':optimizer.state_dict(),'shuffle_rng_state':generator.get_state(),'torch_rng_state':torch.get_rng_state(),
          'case':case,'variant':variant,'seed':seed,'epoch':epoch,'history':history,'settings':settings,'initial_step':initial_step,'initial_validation':initial_score,
          'best_validation':best,'training_seconds':elapsed,'updates':updates,'source_commit':pilot.source,'source_sha256':pilot.source_hashes,
          'data_sha256':sha(pilot.out/'scenarios.pt'),'beamforming_steps_frozen':model.choice,
          'full_analytic_forward_calls':updates*3*settings['layers'],'position_projections':updates*settings['layers'],
          'logical_training_attack_calls':updates*(2+settings['training_random_restarts'])*settings['training_attack_steps'],
          'gradient_policy':'Full differentiation through physical unroll and exact-a.e. PAV Jacobian; independent terminal attack errors detached'}
    if (folder/'latest.pt').exists():
        old=torch.load(folder/'latest.pt',weights_only=True)
        if old['settings']!=settings or old['data_sha256']!=sha(pilot.out/'scenarios.pt'):raise RuntimeError('Learning resume mismatch')
        model.load_state_dict(old['model']);optimizer.load_state_dict(old['optimizer']);generator.set_state(old['shuffle_rng_state']);torch.set_rng_state(old['torch_rng_state'])
        history=old['history'];start_epoch=old['epoch'];best=old['best_validation'];initial_score=old['initial_validation'];elapsed=old['training_seconds'];updates=old['updates']
    else:save(checkpoint(0),folder/'selected.pt')
    for epoch in range(start_epoch,settings['epochs']):
        start=time.perf_counter();order=torch.randperm(len(scenario),generator=generator);losses=[]
        for ids in order.split(settings['batch_size']):
            s=scenario[ids];x,initial=select_rows(raw,ids);optimizer.zero_grad(set_to_none=True);r=model(s,x,initial)
            batch=channels(s,r['x'],model.cfg)
            attack=batched_attack(batch,r['state'],model.cfg,budgets=(settings['training_attack_steps'],),restarts=settings['training_random_restarts'],adam_steps=0,
                                  step_size=pilot.spec['attack_step'],seed=908100)
            adverse=replace(r['state'],**{n:getattr(attack['state'],n).detach() for n in ERRORS})
            loss=-metrics(channels(s,r['x'],model.cfg),adverse,model.cfg)['objective'].mean()
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite learned-position loss')
            loss.backward()
            if model.mu.grad is None or not torch.isfinite(model.mu.grad).all():raise FloatingPointError('Invalid position schedule gradient')
            optimizer.step();model.project();updates+=1;losses.append(loss.item())
        value=validation(pilot,model,settings);elapsed+=time.perf_counter()-start;improved=value>best
        if improved:best=value
        history.append({'epoch':epoch+1,'validation':value,'mean_train_loss':float(np.mean(losses)),'seconds':time.perf_counter()-start,
                        'batch_order_sha256':hashlib.sha256(order.numpy().tobytes()).hexdigest(),'steps':model.mu.detach().tolist()})
        save(checkpoint(epoch+1),folder/'latest.pt')
        if improved:save(checkpoint(epoch+1),folder/'selected.pt')
        write(folder/'history.json',{'initial_validation':initial_score,'best_validation':best,'history':history,'training_seconds':elapsed,'updates':updates})
        pilot.log(f'Position training {case} {variant} seed={seed} epoch={epoch+1}: validation={value:.6f}')
    return folder/'selected.pt'


@torch.no_grad()
def evaluate_one(pilot,case,settings,variant,seed,checkpoint):
    label='seed'+str(seed);path=pilot.out/case/(variant+'_'+label+'.pt')
    if path.exists():return
    saved=torch.load(checkpoint,weights_only=True);model=PositionSchedule(pilot,case,settings,saved['initial_step'],variant=='LearnedConstant');model.load_state_dict(saved['model'])
    s,raw=pilot.data(case,'test');x,initial=select_rows(raw,slice(None));r=model(s,x,initial)
    pilot.store_extra(case,variant,label,r,settings['layers'])
    entry=torch.load(path,weights_only=True);entry['model_checkpoint']=str(checkpoint);entry['selected_epoch']=saved['epoch'];entry['learned_parameter_count']=model.mu.numel()
    entry['initial_step']=saved['initial_step'];entry['learned_steps']=model.mu.detach()
    entry['training_seconds']=read(Path(checkpoint).parent/'history.json')['training_seconds'];save(entry,path)
    single=s[:1];xx,ii=select_rows(raw,slice(0,1))
    for _ in range(pilot.spec['timing_warmup']):model(single,xx,ii)
    times=[]
    for index in range(min(pilot.spec['timing_repeats'],len(s))):
        xx,ii=select_rows(raw,slice(index,index+1));times.append(model(s[index],xx,ii)['seconds']*1000)
    entry['timing']={'median_ms':float(np.median(times)),'p95_ms':float(np.quantile(times,.95)),'samples_ms':times,'scope':'Warm batch-one first15 paired test scenarios, computation only'};save(entry,path)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=ROOT/'configs/pilot.json');p.add_argument('--output',type=Path,default=ROOT/'outputs/pilot')
    p.add_argument('--settings',type=Path,default=ROOT/'configs/learning.json');a=p.parse_args(argv);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    pilot=Pilot(a.config,a.output);settings=read(a.settings)
    gate=read(pilot.out/'classical_gate.json')
    if not gate['passed'] or not gate['learned_position_investigation_justified']:raise RuntimeError('Classical geometry gate must pass before learning')
    choices=tune_plain(pilot,settings)
    for case in ['A','B','C']:
        step=choices['cases'][case]['selected']['step'];s,raw=pilot.data(case,'test')
        baseline=PositionSchedule(pilot,case,settings,step,True)
        if not (pilot.out/case/'Plain_tuned_control.pt').exists():
            with torch.no_grad():r=baseline(s,raw['ula'],SolverState(**raw['initial']))
            pilot.store_extra(case,'Plain_tuned','control',r,settings['layers'])
        for seed in settings['seeds']:
            for variant in settings['variants']:
                path=train_one(pilot,case,settings,variant,seed,step);evaluate_one(pilot,case,settings,variant,seed,path)
        pilot.shared_pool(case)


if __name__=='__main__':main()
