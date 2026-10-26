"""Read-only baseline preservation and classical geometry gate."""
from pathlib import Path
import json
import numpy as np
import torch
from l2o_isac.architecture_audit import read,write,sha
from .run import Pilot,ROOT


def main():
    torch.set_num_threads(1);p=Pilot();preservation=read(ROOT/'outputs/baseline_preservation.json')
    changed=[name for name,digest in preservation['preserved_sha256'].items() if sha(name)!=digest]
    if changed:raise RuntimeError('Pre-existing files changed: '+str(changed))
    report={'baseline_files_unchanged':len(preservation['preserved_sha256']),'cases':{},'scope':'Classical pilot gate; approximate independent search except solvable case-A control'}
    rng=np.random.default_rng(909100)
    for case in ['A','B','C']:
        entries=list((p.out/case).glob('*.pt'));maxres=0.;violations=0
        for path in entries:
            r=torch.load(path,weights_only=True)
            if 'max_residuals' not in r:continue
            maxres=max(maxres,max(r['max_residuals'].values()))
            for key in ['comm','sense','objective']:
                assert torch.isfinite(r['shared_union'][key]).all()
                assert (r['shared_union'][key]<=r['attack_budget_metrics'][1000][key]+2e-6).all()
                assert (r['attack_budget_metrics'][1000][key]<=r['attack_budget_metrics'][200][key]+2e-6).all()
        if maxres>1e-5:raise RuntimeError('Classical feasibility gate failed')
        fixed=torch.load(p.out/case/'ULA_retuned_budget360.pt',weights_only=True);fluid=torch.load(p.out/case/'Fluid_budget360.pt',weights_only=True)
        delta=(fluid['shared_union']['objective']-fixed['shared_union']['objective']).numpy()
        bootstrap=delta[rng.integers(0,len(delta),size=(2000,len(delta)))].mean(1)
        row={'scenarios':len(delta),'mean_fluid_minus_fixed':float(delta.mean()),'paired_channel_95pct':np.quantile(bootstrap,[.025,.975]).tolist(),'max_residual':maxres}
        if case=='A':
            exact=torch.load(p.out/case/'Analytic_ULA_control.pt',weights_only=True)
            expected=exact['analytic_robust_optimum']['objective'];got=exact['shared_union']['objective']
            torch.testing.assert_close(got,expected,rtol=1e-5,atol=2e-6)
            row['analytic_fixed_ula_mean']=got.mean().item();row['fluid_minus_analytic']=float((fluid['shared_union']['objective']-got).mean())
            for path in entries:
                r=torch.load(path,weights_only=True)
                if 'shared_union' in r:assert (r['shared_union']['objective']<=expected+1e-5).all()
        report['cases'][case]=row
    report['geometry_signal_observed']=any(report['cases'][c]['paired_channel_95pct'][0]>0 for c in ['B','C'])
    report['passed']=True;report['learned_position_investigation_justified']=report['geometry_signal_observed']
    write(p.out/'classical_gate.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
