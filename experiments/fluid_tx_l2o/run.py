"""Immutable spatial study, inherited joint training, frozen-design attacks, timing."""
import argparse,copy,hashlib,json,math,os,platform,subprocess,time
from pathlib import Path
from dataclasses import replace
import numpy as np
import torch
from l2o_isac.solver import SolverState,NAMES
from l2o_isac.physics import metrics
from l2o_isac.attacks import ERRORS,keep_best,union_attack
from l2o_isac.batched_attack import batched_attack
from experiments.fluid_tx_single_user.run import Pilot,save
from experiments.fluid_tx_single_user.spatial import system,channels,initialize,random_positions,single_stream_exact_errors,analytic_case_a
from experiments.fluid_tx_single_user.geometry import project_positions,geometry_residuals
from experiments.fluid_tx_single_user.optimizer import solve,all_residuals
from .model import FluidUnroll

ROOT=Path('experiments/fluid_tx_l2o')
def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def detached_result(r):return {'x':r['x'].detach(),'state':r['state'].detach(),'counts':r['counts'],'trace':[ {k:v.detach() for k,v in z.items()} for z in r['trace']]}

class Study:
 def __init__(self,config=ROOT/'configs/study.json',output=ROOT/'outputs/study'):
  self.spec=read(config);self.out=Path(output);self.out.mkdir(parents=True,exist_ok=True);self.pilot=Pilot(config,self.out)
  self.source={str(p):digest(p) for p in [ROOT/'model.py',ROOT/'run.py']};self.cfg={c:system(self.spec,c) for c in 'ABC'}
 def log(self,x):
  print(x,flush=True)
  with (self.out/'events.log').open('a') as f:f.write(x+'\n')
 def data(self,c,split):return self.pilot.data(c,split)
 def prepare(self):
  self.pilot.prepare()
  lock={'spec':self.spec,'source_sha256':self.source,'scenario_sha256':digest(self.out/'scenarios.pt'),'protocol':'Fixed before final test: full inherited layer-averaged loss, direct coefficients, independent frozen-design 50/200/1000 attacks; shared union; paired timing.'}
  if not (self.out/'protocol_lock.json').exists():write(self.out/'protocol_lock.json',lock)
 def attack(self,c,sc,x,st,validation=False):
  return batched_attack(channels(sc,x.detach(),self.cfg[c]),st.detach(),self.cfg[c],budgets=(self.spec['validation_attack_steps'],) if validation else tuple(self.spec['test_attack_budgets']),restarts=self.spec['validation_random_restarts'] if validation else self.spec['attack_random_restarts'],adam_steps=0 if validation else self.spec['attack_adam_steps'],step_size=self.spec['attack_step'],seed=self.spec['attack_seed'])
 @torch.no_grad()
 def tune(self):
  path=self.out/'validation_choices.json'
  if path.exists():return
  choices={}
  for c in 'ABC':
   sc,raw=self.data(c,'validation');cfg=self.cfg[c];x=raw['ula'];st=SolverState(**raw['initial']);fixed={}
   iterations=sorted(set([self.spec['layers']]+[g//2 for g in self.spec['classical_budgets']]))
   for n in iterations:
    grid=[]
    for mi in self.spec['inner_step_grid']:
     for mo in self.spec['outer_step_grid']:
      r=solve(sc,x,st,cfg,self.spec,n,mi,mo);a=self.attack(c,sc,r['x'],r['state'],True)
      grid.append({'inner':mi,'outer':mo,'utility':a['metrics'][self.spec['validation_attack_steps']]['objective'].mean().item()})
    fixed[str(n)]={'selected':max(grid,key=lambda z:z['utility']),'grid':grid}
   bf=fixed[str(self.spec['layers'])]['selected'];fluid={}
   for n in sorted(set([self.spec['layers']]+[g//3 for g in self.spec['classical_budgets']])):
    grid=[]
    for step in self.spec['position_step_grid']:
     r=solve(sc,x,st,cfg,self.spec,n,bf['inner'],bf['outer'],step);a=self.attack(c,sc,r['x'],r['state'],True)
     grid.append({'position':step,'utility':a['metrics'][self.spec['validation_attack_steps']]['objective'].mean().item()})
    fluid[str(n)]={'selected':max(grid,key=lambda z:z['utility']),'grid':grid}
   # Learned fluid starts from a fair no-backtracking, zero-position-momentum grid.
   grid=[]
   for step in self.spec['position_step_grid']:
    m=FluidUnroll(cfg,self.spec,self.spec['layers'],bf['inner'],bf['outer'],step);r=m(sc,x,st);a=self.attack(c,sc,r['x'],r['state'],True)
    grid.append({'position':step,'utility':a['metrics'][self.spec['validation_attack_steps']]['objective'].mean().item()})
   choices[c]={'fixed':fixed,'fluid':fluid,'plain':{'selected':max(grid,key=lambda z:z['utility']),'grid':grid}}
   self.log('Validation tuned '+c+' BF='+str(bf)+' plain='+str(choices[c]['plain']['selected']))
  write(path,{'cases':choices,'selection_split':'validation only','test_used':False,'source_sha256':self.source})
 def model(self,c,variant,seed=None):
  choice=read(self.out/'validation_choices.json')['cases'][c];bf=choice['fixed'][str(self.spec['layers'])]['selected'];step=choice['plain']['selected']['position']
  m=FluidUnroll(self.cfg[c],self.spec,self.spec['layers'],bf['inner'],bf['outer'],step,fluid=variant!='FixedL2O',position_momentum=variant!='FluidNoPositionMomentum')
  if seed is not None:m.load_state_dict(torch.load(self.out/'training'/c/f'{variant}_s{seed}'/'selected.pt',weights_only=True)['model'])
  return m
 @torch.no_grad()
 def validate(self,c,m):
  sc,raw=self.data(c,'validation');r=m(sc,raw['ula'],SolverState(**raw['initial']));loss=m.loss(r).item();a=self.attack(c,sc,r['x'],r['state'],True)
  return {'loss':loss,'independently_attacked':a['metrics'][self.spec['validation_attack_steps']]['objective'].mean().item()}
 def train(self):
  for c in 'ABC':
   for variant in self.spec['primary_variants']+self.spec['ablations_first_seed_only']:
    seeds=self.spec['seeds'] if variant in self.spec['primary_variants'] else self.spec['seeds'][:1]
    for seed in seeds:self.train_one(c,variant,seed)
 def train_one(self,c,variant,seed):
  folder=self.out/'training'/c/f'{variant}_s{seed}';folder.mkdir(parents=True,exist_ok=True)
  if (folder/'complete.json').exists():return
  torch.manual_seed(seed);generator=torch.Generator().manual_seed(seed);m=self.model(c,variant);initial=copy.deepcopy(m.state_dict());opt=torch.optim.SGD(m.parameters(),lr=self.spec['lr']);scheduler=torch.optim.lr_scheduler.ExponentialLR(opt,gamma=self.spec['decay'])
  history=[];v=self.validate(c,m);best=v['loss'];epoch0=0;updates=0;elapsed=0.;sc,raw=self.data(c,'train')
  def checkpoint(epoch):
   return {'model':copy.deepcopy(m.state_dict()),'initial_model':initial,'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),'rng':generator.get_state(),'torch_rng':torch.get_rng_state(),'epoch':epoch,'updates':updates,'best_loss':best,'history':history,'training_seconds':elapsed,'case':c,'variant':variant,'seed':seed,'spec':self.spec,'source_sha256':self.source,'data_sha256':digest(self.out/'scenarios.pt'),'active_parameters':m.active_parameter_count(),'gradient_policy':'Full end-to-end graph through error, beamforming and position iterates; no detached attacks in Eq22 training','transfer':'No original checkpoint transferred; validation-tuned classical scalar initialization'}
  if (folder/'latest.pt').exists():
   old=torch.load(folder/'latest.pt',weights_only=True);assert old['source_sha256']==self.source and old['spec']==self.spec
   m.load_state_dict(old['model']);initial=old['initial_model'];opt.load_state_dict(old['optimizer']);scheduler.load_state_dict(old['scheduler']);generator.set_state(old['rng']);torch.set_rng_state(old['torch_rng']);history=old['history'];epoch0=old['epoch'];updates=old['updates'];elapsed=old['training_seconds'];best=old['best_loss']
  else:save(checkpoint(0),folder/'selected.pt');save(checkpoint(0),folder/'initial.pt')
  for epoch in range(epoch0,self.spec['epochs']):
   start=time.perf_counter();order=torch.randperm(len(sc),generator=generator);losses=[];blocknorm=torch.zeros(7)
   for ids in order.split(self.spec['batch_size']):
    opt.zero_grad(set_to_none=True);r=m(sc[ids],raw['ula'][ids],SolverState(**{n:raw['initial'][n][ids] for n in NAMES}));loss=m.loss(r)
    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss '+str(folder))
    loss.backward()
    for p in m.parameters():
     if p.grad is None or not torch.isfinite(p.grad).all():raise FloatingPointError('Invalid coupled gradient '+str(folder))
    blocknorm[:m.mu.shape[1]]+=m.mu.grad.detach().abs().sum(0)
    opt.step();updates+=1;losses.append(loss.item())
   v=self.validate(c,m);elapsed+=time.perf_counter()-start;improved=v['loss']<best
   if improved:best=v['loss']
   history.append({'epoch':epoch+1,'train_loss':float(np.mean(losses)),**v,'lr':scheduler.get_last_lr()[0],'gradient_abs_by_mu_block':blocknorm.tolist(),'batch_order_sha256':hashlib.sha256(order.numpy().tobytes()).hexdigest()})
   scheduler.step();save(checkpoint(epoch+1),folder/'latest.pt')
   if improved:save(checkpoint(epoch+1),folder/'selected.pt')
   self.log(f'{c} {variant} seed{seed} epoch{epoch+1}: layerloss={v["loss"]:.6f}, independent_val={v["independently_attacked"]:.6f}')
  selected=torch.load(folder/'selected.pt',weights_only=True)
  write(folder/'complete.json',{'selected_epoch':selected['epoch'],'updates':updates,'training_seconds':elapsed,'history':history,'active_parameters':m.active_parameter_count(),'initial_validation':self.validate(c,self.model(c,variant)),'negative_inner_steps':int((m.mu[:,:3]<0).sum()),'mu_block_changes':(selected['model']['mu']-initial['mu']).abs().max(0).values.tolist(),'eta_block_changes':(selected['model']['eta']-initial['eta']).abs().max(0).values.tolist()})
  self.log('Completed '+str(folder))
 def classical(self,c,sc,x,st,fluid,n):
  choices=read(self.out/'validation_choices.json')['cases'][c]
  if fluid:
   bf=choices['fixed'][str(self.spec['layers'])]['selected'];key=min(choices['fluid'],key=lambda k:abs(int(k)-n));ps=choices['fluid'][key]['selected']['position']
  else:
   key=min(choices['fixed'],key=lambda k:abs(int(k)-n));bf=choices['fixed'][key]['selected'];ps=0.
  return solve(sc,x,st,self.cfg[c],self.spec,n,bf['inner'],bf['outer'],ps)
 @torch.no_grad()
 def offline_geometry(self):
  path=self.out/'offline_geometry.pt'
  if path.exists():return
  poses={}
  for c in 'ABC':
   sc,raw=self.data(c,'train');m=self.model(c,'FixedL2O')
   for p in m.parameters():p.requires_grad_(False)
   pose=torch.nn.Parameter(raw['ula'][0].clone());opt=torch.optim.SGD([pose],lr=self.spec['offline_geometry_lr']);hist=[]
   for _ in range(self.spec['offline_geometry_updates']):
    with torch.enable_grad():
     opt.zero_grad();x=pose.expand(len(sc),-1);st=initialize(sc,x,self.cfg[c],c,self.spec['initialization_seed']+'ABC'.index(c));r=m(sc,x,st);loss=m.loss(r);loss.backward();opt.step()
     with torch.no_grad():pose.copy_(project_positions(pose,self.spec['aperture_wavelengths'],self.spec['min_spacing_wavelengths']))
    hist.append(loss.item())
   poses[c]={'positions':pose.detach(),'training_loss':hist,'scope':'One common pose optimized on training Eq22 loss only; fixed at validation/test, no sample-dependent movement'}
  save(poses,path)
 @torch.no_grad()
 def calibration(self):
  path=self.out/'runtime_calibration.json'
  if path.exists():return
  rows={}
  for c in 'ABC':
   sc,raw=self.data(c,'validation');sc=sc[:1];x=raw['ula'][:1];samples={}
   jobs={}
   for fluid in [False,True]:
    for n in self.spec['runtime_candidates']:
     label=('Fluid' if fluid else 'ULA')+str(n)
     jobs[label]=lambda f=fluid,k=n:self.classical(c,sc,x,initialize(sc,x,self.cfg[c],c,self.spec['initialization_seed']+'ABC'.index(c)),f,k)
   for variant in self.spec['primary_variants']:
    m=self.model(c,variant,self.spec['seeds'][0]);jobs[variant]=lambda model=m:model(sc,x,initialize(sc,x,self.cfg[c],c,self.spec['initialization_seed']+'ABC'.index(c)))
   for label,f in jobs.items():
    for _ in range(2):f()
    ts=[]
    for _ in range(5):start=time.perf_counter();f();ts.append((time.perf_counter()-start)*1000)
    samples[label]={'median_ms':float(np.median(ts)),'samples_ms':ts}
   chosen={}
   for variant in self.spec['primary_variants']:
    target=samples[variant]['median_ms'];chosen[variant]={}
    for family in ['ULA','Fluid']:
     name=min([k for k in samples if k.startswith(family) and k[len(family):].isdigit()],key=lambda k:abs(samples[k]['median_ms']-target));chosen[variant][family]=int(name[len(family):])
   rows[c]={'samples':samples,'selected_iterations':chosen}
  write(path,{'cases':rows,'split':'validation scene0 only; median5 after2 warmups; includes initialize and full design call','test_used':False})
 @torch.no_grad()
 def evaluate(self):
  self.offline_geometry();self.calibration()
  for c in 'ABC':
   sc,raw=self.data(c,'test');cfg=self.cfg[c];jobs=[]
   for g in self.spec['classical_budgets']:
    for fluid in [False,True]:jobs.append((('Fluid' if fluid else 'ULA')+f'_G{g}','classical',fluid,g//(3 if fluid else 2),None))
   jobs.extend([('StaticRandom','random',False,self.spec['layers'],None),('TrainOptimizedFixed','offline',False,self.spec['layers'],None),('FluidUntrained','FluidL2O',True,self.spec['layers'],None)])
   for variant in self.spec['primary_variants']+self.spec['ablations_first_seed_only']:
    for seed in self.spec['seeds'] if variant in self.spec['primary_variants'] else self.spec['seeds'][:1]:jobs.append((variant+f'_s{seed}',variant,variant!='FixedL2O',self.spec['layers'],seed))
   cal=read(self.out/'runtime_calibration.json')['cases'][c]['selected_iterations']
   for variant,fs in cal.items():
    for family,n in fs.items():jobs.append((family+'_time_'+variant,'classical',family=='Fluid',n,None))
   if c=='A':jobs.append(('AnalyticULA','analytic',False,0,None))
   for label,kind,fluid,n,seed in jobs:
    path=self.out/c/(label+'.pt');path.parent.mkdir(exist_ok=True)
    if path.exists():continue
    x=raw['ula'];st=SolverState(**raw['initial'])
    if kind=='random':x=random_positions(self.spec,1,929999).expand(len(sc),-1).clone();st=initialize(sc,x,cfg,c,self.spec['initialization_seed']+'ABC'.index(c))
    if kind=='offline':x=torch.load(self.out/'offline_geometry.pt',weights_only=True)[c]['positions'].expand(len(sc),-1).clone();st=initialize(sc,x,cfg,c,self.spec['initialization_seed']+'ABC'.index(c))
    start=time.perf_counter()
    if kind in ['classical','random','offline']:r=self.classical(c,sc,x,st,fluid,n)
    elif kind=='analytic':r={'x':x,'state':st,'counts':{'full_analytic_calls':0},'trace':[metrics(channels(sc,x,cfg),st,cfg)]}
    else:r=self.model(c,kind,seed)(sc,x,st)
    seconds=time.perf_counter()-start;r=detached_result(r);res=all_residuals(r['x'],r['state'],sc,cfg,self.spec)
    if max(res.values())>1e-5:raise FloatingPointError('Infeasible '+label+': '+str(res))
    a=self.attack(c,sc,r['x'],r['state']);batch=channels(sc,r['x'],cfg)
    nominal=replace(r['state'],**{k:getattr(r['state'],k)*0 for k in ERRORS})
    payload={'positions':r['x'],'state':vars(r['state']),'attack_state':vars(a['state']),'attack_metrics':a['metrics'],'attack_seconds':a['seconds'],'attack_counts':{k:v for k,v in a.items() if 'calls' in k},'terminal':metrics(batch,r['state'],cfg),'nominal':metrics(batch,nominal,cfg),'residuals':res,'geometry_per_scenario':geometry_residuals(r['x'],self.spec['aperture_wavelengths'],self.spec['min_spacing_wavelengths']),'counts':r['counts'],'design_batch_seconds':seconds,'method':label,'kind':kind,'fluid':fluid,'iterations':n,'seed':seed,'source_sha256':self.source,'data_sha256':digest(self.out/'scenarios.pt'),'objective_trace':torch.stack([z['objective'] for z in r['trace']])}
    save(payload,path);self.log(f'{c}/{label}: independent={a["metrics"][max(self.spec["test_attack_budgets"])]["objective"].mean().item():.6f}')
   self.union(c)
 @torch.no_grad()
 def union(self,c):
  sc,_=self.data(c,'test');files=list((self.out/c).glob('*.pt'));entries=[torch.load(p,weights_only=True) for p in files];candidates=[SolverState(**r[k]) for r in entries for k in ['state','attack_state']]
  if c=='A':
   for r in entries:
    st=SolverState(**r['state'])
    if torch.count_nonzero((st.a@st.d)[...,1:])==0:candidates.append(single_stream_exact_errors(channels(sc,r['positions'],self.cfg[c]),st,self.cfg[c]))
  for p,r in zip(files,entries):
   batch=channels(sc,r['positions'],self.cfg[c])
   best=union_attack(batch,{'recipient':SolverState(**r['state'])},candidates,self.cfg[c])['recipient']
   r['shared_union']=metrics(batch,best,self.cfg[c]);r['union_state']=vars(best);save(r,p)
 @torch.no_grad()
 def timing(self):
  for c in 'ABC':
   sc,raw=self.data(c,'test')
   for path in (self.out/c).glob('*.pt'):
    r=torch.load(path,weights_only=True)
    if 'timing' in r:continue
    if r['kind']=='analytic':continue
    model=self.model(c,r['kind'],r['seed']) if r['kind'] not in ['classical','random','offline'] else None
    def f(i):
     s=sc[i];x=r['positions'][i:i+1] if r['kind'] in ['random','offline'] else raw['ula'][i:i+1]
     st=initialize(s,x,self.cfg[c],c,self.spec['initialization_seed']+'ABC'.index(c))
     if model is not None:return model(s,x,st)
     return self.classical(c,s,x,st,r['fluid'],r['iterations'])
    for _ in range(self.spec['timing_warmup']):f(0)
    ts=[]
    for i in range(min(len(sc),self.spec['timing_scenarios'])):
     for _ in range(self.spec['timing_repeats']):start=time.perf_counter();f(i);ts.append((time.perf_counter()-start)*1000)
    r['timing']={'samples_ms':ts,'median_ms':float(np.median(ts)),'p95_ms':float(np.quantile(ts,.95)),'scope':'Full initialize+forward including coefficient access, field reconstruction, trace metrics, projections/backtracking; batch1 CPU1 thread; independent attacks separate. No movement, RF or CSI acquisition.'};save(r,path)
    self.log(f'Timed {c}/{r["method"]}: {np.median(ts):.2f} ms')

def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['prepare','tune','train','evaluate','timing','all'],default='all');p.add_argument('--config',type=Path,default=ROOT/'configs/study.json');p.add_argument('--output',type=Path,default=ROOT/'outputs/study');a=p.parse_args();torch.set_num_threads(1);torch.use_deterministic_algorithms(True);study=Study(a.config,a.output)
 for stage in ['prepare','tune','train','evaluate','timing'] if a.stage=='all' else [a.stage]:getattr(study,stage)()
if __name__=='__main__':main()
