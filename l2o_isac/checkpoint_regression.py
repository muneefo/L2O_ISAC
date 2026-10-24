"""Replay eight existing saved raw scenarios without training or regenerating data."""

def main():
    from pathlib import Path
    from dataclasses import replace
    import json,torch,hashlib
    from l2o_isac.data import ChannelBatch
    from l2o_isac.solver import SolverState,constraint_errors,NAMES
    from l2o_isac.inference import load_checkpoint
    from l2o_isac.physics import metrics
    p=Path('reports/corrected_per_channel_seed73129'); out=Path('results/architecture_audit');torch.set_num_threads(1)
    cfg,epsnet,model,saved=load_checkpoint(p/'checkpoint.pt'); raw=torch.load(p/'raw_beamformers_and_channels.pt',weights_only=True)
    batch=ChannelBatch(**raw['channel']); initial=SolverState(**raw['initial'])
    with torch.no_grad():
     final,trace,_=model(batch,initial,eps=raw['predicted_eps'])
     errors={n:(getattr(final,n)-raw['trained'][n]).abs().max().item() for n in NAMES}
     for n in NAMES: torch.testing.assert_close(getattr(final,n),raw['trained'][n],rtol=1e-5,atol=1e-6)
     objective_error=(torch.stack([x['objective'] for x in trace])-raw['objective_trace']).abs().max().item()
     assert objective_error<1e-5
     result={'checkpoint_sha256':hashlib.sha256((p/'checkpoint.pt').read_bytes()).hexdigest(),'reference_commit':'83ba89148c8d89a7ab60105da0fa5cbfe6105d9e','state_max_absolute_errors':errors,'trace_max_absolute_error':objective_error,'constraint_residuals':constraint_errors(final,batch,cfg,raw['predicted_eps']),'perturbed':{k:v.tolist() for k,v in metrics(batch,final,cfg).items()},'nominal':{k:v.tolist() for k,v in metrics(batch,replace(final,dh=final.dh*0,dt=final.dt*0,dr=final.dr*0),cfg).items()},'initialization_seed':raw['initialization_seed'],'passed':True}
    (out/'checkpoint_regression.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
