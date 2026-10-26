"""Isolated spatial-model/geometry classical pilot. No reproduction training."""
import argparse,copy,itertools,json,os,platform,subprocess,time
from dataclasses import asdict,replace
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from l2o_isac.architecture_audit import read,write,sha,means,state_slice,tensor_metrics
from l2o_isac.batched_attack import batched_attack
from l2o_isac.attacks import union_attack,ERRORS,keep_best
from l2o_isac.solver import SolverState,PGDA,NAMES
from l2o_isac.physics import metrics
from l2o_isac.audit_artifacts import residuals
from .spatial import Scenario,system,make_scenarios,channels,ula_positions,random_positions,initialize,analytic_case_a,single_stream_exact_errors
from .optimizer import solve,all_residuals
from .geometry import geometry_residuals,steering

ROOT=Path('experiments/fluid_tx_single_user')


def save(value,path):
    path=Path(path);tmp=path.with_name(path.name+'.tmp');torch.save(value,tmp);os.replace(tmp,path)


class Pilot:
    def __init__(self,config=ROOT/'configs/pilot.json',output=ROOT/'outputs/pilot'):
        self.spec=read(config);self.out=Path(output);self.out.mkdir(parents=True,exist_ok=True)
        try:self.source=subprocess.check_output(['git','rev-parse','HEAD'],text=True,stderr=subprocess.DEVNULL).strip()
        except subprocess.CalledProcessError:self.source='unversioned export; see saved source hashes'
        self.source_hashes={str(p):sha(p) for p in ROOT.glob('*.py')}
        self._bf_choices=None;self._long_choices=None;self._plain_choices=None

    def log(self,msg):
        line=datetime.now(timezone.utc).isoformat()+' '+msg;print(line,flush=True)
        with (self.out/'events.log').open('a') as f:f.write(line+'\n')

    def prepare(self):
        path=self.out/'scenarios.pt'
        if path.exists():
            m=read(self.out/'data_manifest.json')
            if m['spec']!=self.spec or m['sha256']!=sha(path):raise RuntimeError('Immutable spatial split/config mismatch')
            return
        stored={}
        for ci,case in enumerate(['A','B','C']):
            stored[case]={}
            for split in ['train','validation','test']:
                count=self.spec[split+'_scenarios_per_case'];seed=self.spec['split_seeds'][split]+ci*100
                scenario=make_scenarios(self.spec,case,count,seed);x=ula_positions(self.spec,count)
                xr=random_positions(self.spec,count,seed+5000)
                cfg=system(self.spec,case);seed_init=self.spec['initialization_seed']+ci
                initial=initialize(scenario,x,cfg,case,seed_init);random_initial=initialize(scenario,xr,cfg,case,seed_init)
                stored[case][split]={'scenario':vars(scenario),'ula':x,'random':xr,'initial':vars(initial),'random_initial':vars(random_initial),'seed':seed,'initialization_seed':seed_init}
        save(stored,path);write(self.out/'data_manifest.json',{'sha256':sha(path),'spec':self.spec,'scope':'Fresh field-response scenario splits; not the paper CN channel law','source_commit':self.source,'source_sha256':self.source_hashes})
        import scipy
        write(self.out/'environment.json',{'python':platform.python_version(),'torch':str(torch.__version__),'numpy':np.__version__,'scipy':scipy.__version__,
              'cpu':next(x.split(':',1)[1].strip() for x in Path('/proc/cpuinfo').read_text().splitlines() if x.startswith('model name')),
              'cpu_quota':Path('/sys/fs/cgroup/cpu.max').read_text().strip(),'threads':torch.get_num_threads(),'cuda_available':torch.cuda.is_available(),
              'coordinates':'x=p/lambda in wavelengths; physical wavelength '+str(self.spec['wavelength_m'])+' meters','precision':'complex64/float32 experiments; complex128/float64 checks'})
        self.log('Saved new fixed spatial paths and immutable paired splits')

    def data(self,case,split):
        raw=torch.load(self.out/'scenarios.pt',weights_only=True)[case][split]
        return Scenario(**raw['scenario']),raw

    def attack(self,scenario,x,state,cfg,budgets=None,validation=False):
        b=channels(scenario,x.detach(),cfg)
        return batched_attack(b,state.detach(),cfg,budgets=tuple(budgets or self.spec['test_attack_budgets']),
           restarts=1 if validation else self.spec['attack_random_restarts'],adam_steps=0 if validation else self.spec['attack_adam_steps'],
           step_size=self.spec['attack_step'],seed=908100)

    @torch.no_grad()
    def tune(self):
        path=self.out/'validation_choices.json'
        if path.exists():return
        choices={}
        for case in ['A','B','C']:
            cfg=system(self.spec,case);scenario,raw=self.data(case,'validation');records={}
            for geometry in ['ula','random']:
                x=raw[geometry];initial=SolverState(**raw['initial' if geometry=='ula' else 'random_initial']);grid=[]
                for mi,mo in itertools.product(self.spec['inner_step_grid'],self.spec['outer_step_grid']):
                    r=solve(scenario,x,initial,cfg,self.spec,self.spec['tuning_iterations'],mi,mo)
                    attack=self.attack(scenario,r['x'],r['state'],cfg,budgets=[self.spec['validation_attack_steps']],validation=True)
                    grid.append({'inner':mi,'outer':mo,'utility':attack['metrics'][self.spec['validation_attack_steps']]['objective'].mean().item()})
                records[geometry]={'selected':max(grid,key=lambda q:q['utility']),'grid':grid}
            selected=records['ula']['selected'];position_grid=[]
            for step in self.spec['position_step_grid']:
                r=solve(scenario,raw['ula'],SolverState(**raw['initial']),cfg,self.spec,self.spec['tuning_iterations'],selected['inner'],selected['outer'],step)
                attack=self.attack(scenario,r['x'],r['state'],cfg,budgets=[self.spec['validation_attack_steps']],validation=True)
                position_grid.append({'step':step,'utility':attack['metrics'][self.spec['validation_attack_steps']]['objective'].mean().item()})
            records['position']={'selected':max(position_grid,key=lambda q:q['utility']),'grid':position_grid}
            choices[case]=records;self.log('Validation choices '+case+': '+str({k:v['selected'] for k,v in records.items()}))
        write(path,{'cases':choices,'selection_split':'validation only','test_used':False,'frozen_utc':datetime.now(timezone.utc).isoformat(),'spec':self.spec})

    def design(self,case,scenario,raw,method,iterations,position_schedule=None,rho=None):
        cfg=system(self.spec,case,rho)
        if self._bf_choices is None:self._bf_choices=read(self.out/'validation_choices.json')['cases']
        choice=self._bf_choices[case]
        geometry='random' if method=='Random' else 'ula';x=raw[geometry];initial=SolverState(**raw['random_initial' if geometry=='random' else 'initial'])
        bf=choice[geometry]['selected']
        if method=='StaticRandom':
            static=read(self.out/'static_random_geometry.json')['positions_wavelengths']
            x=torch.tensor(static,dtype=x.dtype).expand(len(scenario),-1).clone()
            initial=SolverState(**raw['static_initial']) if 'static_initial' in raw else initialize(scenario,x,cfg,case,self.spec['initialization_seed']+['A','B','C'].index(case))
            bf=read(self.out/'static_random_validation.json')['cases'][case][str(iterations)]['selected']
        if method=='ULA_retuned':
            if self._long_choices is None:self._long_choices=read(self.out/'long_validation_choices.json')['cases']
            bf=self._long_choices[case][str(iterations)]['selected']
        step=choice['position']['selected']['step'] if method in ['Fluid','Fluid_plain'] else 0.
        if method=='Plain_tuned':
            if self._plain_choices is None:self._plain_choices=read(self.out/'learning_constant_validation.json')['cases']
            step=self._plain_choices[case]['selected']['step']
        return solve(scenario,x,initial,cfg,self.spec,iterations,bf['inner'],bf['outer'],step,safeguard=method not in ['Fluid_plain','Plain_tuned'],position_schedule=position_schedule)

    @torch.no_grad()
    def calibrate_runtime(self):
        path=self.out/'runtime_calibration.json'
        if path.exists():return
        result={}
        for case in ['A','B','C']:
            sc,raw=self.data(case,'validation');sc=sc[:1];raw={k:state_slice(SolverState(**v),slice(0,1)).__dict__ if k in ['initial','random_initial'] else v[:1] if isinstance(v,torch.Tensor) else v for k,v in raw.items()}
            samples={}
            for method,iterations in [('Fluid',20)]+[('ULA',n) for n in [20,30,45,60,90,120,180,240]]:
                for _ in range(2):self.design(case,sc,raw,method,iterations)
                t=[]
                for _ in range(3):
                    r=self.design(case,sc,raw,method,iterations);t.append(r['seconds']*1000)
                samples[method+str(iterations)]={'median_ms':float(np.median(t)),'samples_ms':t,'iterations':iterations}
            target=samples['Fluid20']['median_ms'];selected=min([v for k,v in samples.items() if k.startswith('ULA')],key=lambda v:abs(v['median_ms']-target))
            result[case]={'fluid_iterations':20,'fluid_median_ms':target,'matched_ula_iterations':selected['iterations'],'candidates':samples}
            self.log('Validation runtime match '+case+': Fluid20 vs ULA'+str(selected['iterations']))
        write(path,{'cases':result,'selection_split':'validation scenario0, timing only','test_used':False})

    @torch.no_grad()
    def evaluate(self):
        choices=read(self.out/'validation_choices.json')
        for case in ['A','B','C']:
            cfg=system(self.spec,case);scenario,raw=self.data(case,'test');folder=self.out/case;folder.mkdir(exist_ok=True)
            jobs=[]
            for budget in self.spec['derivative_budgets']:
                jobs.extend([('ULA',budget//2,'budget'+str(budget)),('Random',budget//2,'budget'+str(budget)),('Fluid',budget//3,'budget'+str(budget))])
            matched=read(self.out/'runtime_calibration.json')['cases'][case]['matched_ula_iterations'];jobs.append(('ULA',matched,'runtime_match'))
            # Plain constant-step control for any later learned position schedule.
            jobs.append(('Fluid_plain',self.spec['learning_layers'],'learning_control'))
            for method,iterations,label in jobs:
                path=folder/(method+'_'+label+'.pt')
                if path.exists():continue
                r=self.design(case,scenario,raw,method,iterations);x,state=r['x'],r['state'];res=all_residuals(x,state,scenario,cfg,self.spec)
                if max(res.values())>1e-5:raise FloatingPointError('Feasibility failure: '+str(res))
                attacked=self.attack(scenario,x,state,cfg);batch=channels(scenario,x,cfg);nominal=replace(state,**{n:getattr(state,n)*0 for n in ERRORS})
                payload={'positions':x,'state':vars(state),'attacked_state':vars(attacked['state']),'attack_budget_metrics':attacked['metrics'],
                  'nominal':tensor_metrics(batch,nominal,cfg),'terminal':tensor_metrics(batch,state,cfg),'max_residuals':res,
                  'hardware_per_scenario':residuals(state,batch,cfg),'geometry_per_scenario':geometry_residuals(x,self.spec['aperture_wavelengths'],self.spec['min_spacing_wavelengths']),
                  'counts':r['counts'],'design_batch_seconds':r['seconds'],'attack_seconds':attacked['seconds'],
                  'attack_logical_start_gradient_calls':attacked['logical_start_error_gradient_calls'],'attack_tensor_gradient_calls':attacked['tensor_error_gradient_calls'],
                  'iterations':iterations,'method':method,'label':label,'source_sha256':self.source_hashes,'data_sha256':sha(self.out/'scenarios.pt'),'spec':self.spec,
                  'objective_trace':torch.stack([t['objective'] for t in r['trace']])}
                if case=='A':payload['analytic_robust_optimum']=analytic_case_a(scenario,cfg)
                save(payload,path);self.log(f'{case} {method} {label}: independent={attacked["metrics"][max(self.spec["test_attack_budgets"])]["objective"].mean():.6f}, counts={r["counts"]["full_analytic_calls"]}')
            self.shared_pool(case)

    @torch.no_grad()
    def store_extra(self,case,method,label,result,iterations):
        scenario,_=self.data(case,'test');cfg=system(self.spec,case);x,state=result['x'],result['state'];batch=channels(scenario,x,cfg)
        res=all_residuals(x,state,scenario,cfg,self.spec)
        if max(res.values())>1e-5:raise FloatingPointError('Extra design infeasible')
        attacked=self.attack(scenario,x,state,cfg);nominal=replace(state,**{n:getattr(state,n)*0 for n in ERRORS})
        payload={'positions':x,'state':vars(state),'attacked_state':vars(attacked['state']),'attack_budget_metrics':attacked['metrics'],
          'nominal':tensor_metrics(batch,nominal,cfg),'terminal':tensor_metrics(batch,state,cfg),'max_residuals':res,
          'hardware_per_scenario':residuals(state,batch,cfg),'geometry_per_scenario':geometry_residuals(x,self.spec['aperture_wavelengths'],self.spec['min_spacing_wavelengths']),
          'counts':result['counts'],'design_batch_seconds':result['seconds'],'attack_seconds':attacked['seconds'],
          'attack_logical_start_gradient_calls':attacked['logical_start_error_gradient_calls'],'attack_tensor_gradient_calls':attacked['tensor_error_gradient_calls'],
          'iterations':iterations,'method':method,'label':label,'source_sha256':self.source_hashes,'data_sha256':sha(self.out/'scenarios.pt'),'spec':self.spec,
          'objective_trace':torch.stack([v['objective'] for v in result['trace']])}
        if case=='A':payload['analytic_robust_optimum']=analytic_case_a(scenario,cfg)
        save(payload,self.out/case/(method+'_'+label+'.pt'))
        self.log(f'{case} {method} {label}: independent={attacked["metrics"][max(self.spec["test_attack_budgets"])]["objective"].mean():.6f}')

    @torch.no_grad()
    def support_baselines(self):
        path=self.out/'long_validation_choices.json'
        if not path.exists():
            choices={}
            for case in ['A','B','C']:
                cfg=system(self.spec,case);scenario,raw=self.data(case,'validation');choices[case]={}
                for iterations in [90,180]:
                    grid=[]
                    for mi,mo in itertools.product(self.spec['inner_step_grid'],self.spec['outer_step_grid']):
                        r=solve(scenario,raw['ula'],SolverState(**raw['initial']),cfg,self.spec,iterations,mi,mo)
                        attack=self.attack(scenario,r['x'],r['state'],cfg,budgets=[self.spec['validation_attack_steps']],validation=True)
                        grid.append({'inner':mi,'outer':mo,'utility':attack['metrics'][self.spec['validation_attack_steps']]['objective'].mean().item()})
                    choices[case][str(iterations)]={'selected':max(grid,key=lambda q:q['utility']),'grid':grid}
            write(path,{'cases':choices,'selection_split':'validation only','test_used':False,'purpose':'Stronger per-budget fixed-ULA controls'})
        for case in ['A','B','C']:
            scenario,raw=self.data(case,'test');cfg=system(self.spec,case)
            for iterations in [90,180]:
                label='budget'+str(2*iterations)
                if not (self.out/case/('ULA_retuned_'+label+'.pt')).exists():
                    self.store_extra(case,'ULA_retuned',label,self.design(case,scenario,raw,'ULA_retuned',iterations),iterations)
            if case=='A' and not (self.out/case/'Analytic_ULA_control.pt').exists():
                state=SolverState(**raw['initial']);batch=channels(scenario,raw['ula'],cfg)
                r={'x':raw['ula'],'state':state,'trace':[metrics(batch,state,cfg)],'counts':{'full_analytic_calls':0,'position_projections':0,'position_metric_evaluations':0,'field_reconstructions':1},'seconds':0.}
                self.store_extra(case,'Analytic_ULA','control',r,0)
            self.shared_pool(case)

    @torch.no_grad()
    def shared_pool(self,case):
        cfg=system(self.spec,case);scenario,_=self.data(case,'test');entries={};candidates=[]
        for p in (self.out/case).glob('*.pt'):
            r=torch.load(p,weights_only=True)
            if 'attacked_state' not in r:continue
            entries[p]=(r,SolverState(**r['state']));candidates.extend([SolverState(**r['attacked_state']),SolverState(**r['state'])])
            if case=='A':
                st=SolverState(**r['state']);assert torch.count_nonzero((st.a@st.d)[...,1:])==0
                candidates.append(single_stream_exact_errors(channels(scenario,r['positions'],cfg),st,cfg))
        # Positions differ between methods: reuse uncertainty vectors, but rebuild
        # nominal channels for the recipient's frozen positions before evaluating.
        for p,(r,design) in entries.items():
            batch=channels(scenario,r['positions'],cfg);states=union_attack(batch,{'recipient':design},candidates,cfg)
            r['shared_union']=tensor_metrics(batch,states['recipient'],cfg);r['union_state']=vars(states['recipient']);save(r,p)

    @torch.no_grad()
    def diagnostics(self):
        for case in ['A','B','C']:
            path=self.out/case/'diagnostic.pt'
            if path.exists():continue
            cfg=system(self.spec,case);sc,raw=self.data(case,'test');sc=sc[:1]
            short={**raw,'ula':raw['ula'][:1],'random':raw['random'][:1],
                   'initial':{k:v[:1] for k,v in raw['initial'].items()},'random_initial':{k:v[:1] for k,v in raw['random_initial'].items()}}
            fixed=self.design(case,sc,short,'ULA',90);xrand=short['random'];weights=fixed['state']
            choice=read(self.out/'validation_choices.json')['cases'][case]['ula']['selected']
            retuned=solve(sc,xrand,weights,cfg,self.spec,90,choice['inner'],choice['outer'])
            variants={'ULA_fixed_weights':(fixed['x'],weights),'Changed_same_weights':(xrand,weights),'Changed_retuned_weights':(xrand,retuned['state'])}
            if case=='A':variants['Changed_analytic_MRT']=(xrand,initialize(sc,xrand,cfg,'A',raw['initialization_seed']))
            out={}
            for name,(x,state) in variants.items():
                a=self.attack(sc,x,state,cfg)
                final_metrics=a['metrics'][max(self.spec['test_attack_budgets'])]
                if case=='A':
                    batch=channels(sc,x,cfg);exact=single_stream_exact_errors(batch,state,cfg)
                    final_metrics=tensor_metrics(batch,keep_best(batch,a['state'],exact,cfg),cfg)
                out[name]={'positions':x,'state':vars(state),'nominal':tensor_metrics(channels(sc,x,cfg),replace(state,**{n:getattr(state,n)*0 for n in ERRORS}),cfg),
                           'independent':final_metrics}
            save({'variants':out,'scope':'First test scenario, index0 fixed before inspection; illustration only, not evidence of average gain'},path)

    @torch.no_grad()
    def timing(self):
        result={}
        previous=read(self.out/'timing.json')['cases'] if (self.out/'timing.json').exists() else {}
        for case in ['A','B','C']:
            scenario,original=self.data(case,'test');indices=list(range(min(self.spec['timing_repeats'],len(scenario))));rows=dict(previous.get(case,{}))
            def single(index):
                raw={k:state_slice(SolverState(**v),slice(index,index+1)).__dict__ if k in ['initial','random_initial'] else v[index:index+1] if isinstance(v,torch.Tensor) else v for k,v in original.items()}
                sc=scenario[index]
                if (self.out/'static_random_geometry.json').exists():
                    xx=torch.tensor(read(self.out/'static_random_geometry.json')['positions_wavelengths']).expand(1,-1)
                    raw['static_initial']=vars(initialize(sc,xx,system(self.spec,case),case,self.spec['initialization_seed']+['A','B','C'].index(case)))
                return sc,raw
            first,initial=single(0)
            for p in (self.out/case).glob('*.pt'):
                r=torch.load(p,weights_only=True)
                if p.stem in rows:continue
                if 'iterations' not in r or r.get('method','').startswith('Learned') or r.get('method')=='Analytic_ULA':continue
                for _ in range(self.spec['timing_warmup']):self.design(case,first,initial,r['method'],r['iterations'])
                times=[]
                for index in indices:
                    sc,raw=single(index);started=time.perf_counter();self.design(case,sc,raw,r['method'],r['iterations']);times.append((time.perf_counter()-started)*1000)
                rows[p.stem]={'median_ms':float(np.median(times)),'p95_ms':float(np.quantile(times,.95)),'samples_ms':times,'scenario_indices':indices}
            result[case]=rows;self.log('Completed batch-one timing '+case)
        write(self.out/'timing.json',{'cases':result,'batch_size':1,'precision':'complex64/float32','threads':1,'warmups':self.spec['timing_warmup'],'timed_scenarios_per_method':self.spec['timing_repeats'],
             'scope':'Warm computation on paired first15 test scenarios, includes field reconstruction/PAV/backtracking; independent attacks and initialization excluded; no physical movement/RF/CSI timing'})


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=ROOT/'configs/pilot.json');p.add_argument('--output',type=Path,default=ROOT/'outputs/pilot')
    p.add_argument('--stage',choices=['prepare','tune','evaluate','diagnostics','support','timing','all'],default='all');a=p.parse_args(argv)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);pilot=Pilot(a.config,a.output);pilot.prepare()
    try:
        if a.stage in ['tune','all']:pilot.tune();pilot.calibrate_runtime()
        if a.stage in ['evaluate','all']:pilot.evaluate()
        if a.stage in ['support','all']:pilot.support_baselines()
        if a.stage in ['diagnostics','all']:pilot.diagnostics()
        if a.stage in ['timing','all']:pilot.timing()
    except Exception as exc:pilot.log('FAILED '+type(exc).__name__+': '+str(exc));raise


if __name__=='__main__':main()
