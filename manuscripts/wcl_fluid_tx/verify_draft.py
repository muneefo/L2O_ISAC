"""Check draft assets, data linkage, PDF build, and the LOS construction."""
from pathlib import Path
import hashlib,json,re,subprocess
import numpy as np
H=Path(__file__).resolve().parent
R=H.parents[1]
D=R/'experiments/fluid_tx_single_user/outputs/pilot'
s=json.loads((D/'summary.json').read_text())
provenance=json.loads((H/'asset_provenance.json').read_text())
for rel,sha in provenance['inputs_sha256'].items():
 assert hashlib.sha256((R/rel).read_bytes()).hexdigest()==sha,rel
tex=(H/'main.tex').read_text()
abstract=re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}',tex,re.S).group(1)
words=len(abstract.split())
assert 75<=words<=100,words
pdf_pages={}
overfull={}
for stem in ['main','main_single_column','initial_model_user']:
 log=(H/f'{stem}.log').read_text()
 assert 'undefined' not in log.lower(),stem
 overfull[stem]=bool('Overfull' in log)
 if stem!='initial_model_user':assert not overfull[stem],stem
 assert 'Fatal error' not in log,stem
 info=subprocess.check_output(['pdfinfo',str(H/f'{stem}.pdf')],text=True)
 pdf_pages[stem]=int(re.search(r'Pages:\s+(\d+)',info).group(1))
assert pdf_pages['main']<=5
claims={}
for c in 'BC':
 v=s['cases'][c]
 base=v['ULA_retuned_budget360']['shared_union']['objective']
 fluid=v['Fluid_budget360']['shared_union']['objective']
 gain=100*(fluid/base-1)
 assert round(gain,1)=={'B':11.8,'C':10.6}[c]
 comp=s['geometry_comparisons'][c]['fluid_vs_fixed']
 assert abs((fluid-base)-comp['mean'])<1e-6
 claims[c]={'fixed_utility':base,'fluid_utility':fluid,'relative_percent':gain,'paired_interval':comp['channel_95pct']}
 for val in [base,fluid,v['StaticRandom_budget360']['shared_union']['objective']]:
  assert f'{val:.4f}' in (H/'table_geometry.tex').read_text()
 for vname in ['ULA_budget60','Fluid_budget60']:
  z=v[vname]
  for metric in ['comm','sense','objective']:
   assert f"{z['shared_union'][metric]:.4f}" in (H/'table_short.tex').read_text()
for c in 'ABC':
 for metric in ['constant_utility','schedule_utility']:
  assert f"{s['learning'][c][metric]:.6f}" in (H/'table_learning.tex').read_text()
# Independent numerical sanity check for the constructive LOS equality proof.
g=np.random.default_rng(51026);maxerr=0.
for _ in range(30):
 nt,nr=8,8;power=10**.5;theta=g.uniform(-.9,.9);x=np.sort(g.uniform(0,3.5,nt));xr=np.arange(nr)*.5
 at=np.exp(-2j*np.pi*x*np.sin(theta));ar=np.exp(-2j*np.pi*xr*np.sin(theta))
 beta=np.exp(2j*np.pi*g.random());alpha=(g.normal()+1j*g.normal())/np.sqrt(2)
 eh,et,er=g.uniform(0,.5,3);h=(beta*at).conj();vc=np.sqrt(power/nt)*at.conj();w=ar/np.sqrt(nr)
 hc=(1-eh)*h;tc=(1-et)*at;rc=(1-er)*ar
 c=np.log1p(abs(hc.conj()@vc)**2)
 sensing=np.log1p(abs(alpha*(w.conj()@rc)*(tc@vc))**2)
 exactc=np.log1p(power*nt*(1-eh)**2)
 exacts=np.log1p(abs(alpha)**2*power*nt*nr*(1-et)**2*(1-er)**2)
 maxerr=max(maxerr,abs(c-exactc),abs(sensing-exacts))
assert maxerr<1e-12,maxerr
files=['main.tex','initial_model_user.tex','make_assets.py','build.sh','verify_draft.py','table_geometry.tex','table_short.tex','table_learning.tex','figures/budgets.pdf','figures/tradeoff.pdf','vendor/IEEEtran.cls']
report={'abstract_words':words,'pdf_pages':pdf_pages,'latex_unresolved_references':False,'latex_overfull_boxes':overfull,'claims':claims,'los_construction_max_error_30_draws':maxerr,'source_sha256':{f:hashlib.sha256((H/f).read_bytes()).hexdigest() for f in files},'scope':'Draft/data and closed-form-construction checks; no new training or physical experiments; not a full-text novelty audit.'}
(H/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='source_sha256'},indent=2))
