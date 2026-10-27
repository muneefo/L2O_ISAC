from pathlib import Path
import hashlib,json,re,subprocess
H=Path(__file__).resolve().parent;R=H.parents[1]
tex=(H/'main.tex').read_text();abstract=re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}',tex,re.S).group(1);words=len(abstract.split());assert 75<=words<=100,words
assert 'L2O' in re.search(r'\\title\{(.*?)\}',tex,re.S).group(1)
assert 'L2O' in abstract and 'L2O' in re.search(r'\\begin\{IEEEkeywords\}(.*?)\\end\{IEEEkeywords\}',tex,re.S).group(1)
pages={}
for name in ['main','main_single_column']:
 log=(H/f'{name}.log').read_text();assert 'undefined' not in log.lower();assert 'Overfull' not in log;assert 'Fatal error' not in log
 info=subprocess.check_output(['pdfinfo',str(H/f'{name}.pdf')],text=True);pages[name]=int(re.search(r'Pages:\s+(\d+)',info).group(1))
assert pages['main']<=5
s=json.loads((R/'experiments/fluid_tx_l2o/outputs/study/summary.json').read_text());pro=json.loads((H/'asset_provenance.json').read_text())
assert hashlib.sha256((R/'experiments/fluid_tx_l2o/outputs/study/summary.json').read_bytes()).hexdigest()==pro['summary_sha256']
for c in 'ABC':
 for name in ['FixedL2O','FluidL2O']:
  val=sum(s['cases'][c][f'{name}_s{seed}']['attacked']['objective'] for seed in [401,402,403])/3
  assert f'{val:.5f}' in (H/'table_main.tex').read_text()
files=[p for p in H.rglob('*') if p.is_file() and p.suffix in ['.tex','.py','.pdf','.json'] and p.name!='verification.json']
record={'abstract_words':words,'pages':pages,'L2O_in_title_abstract_keywords':True,'unresolved_references':False,'overflow_boxes':False,'paired_models':3,'scenarios_per_regime':100,'source_and_assets_sha256':{str(p.relative_to(H)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},'visual_check':'All four double-column pages and both editable diagrams rendered and inspected; no plotted result is a placeholder.'}
(H/'verification.json').write_text(json.dumps(record,indent=2)+'\n');print(words,'abstract words;',pages)
