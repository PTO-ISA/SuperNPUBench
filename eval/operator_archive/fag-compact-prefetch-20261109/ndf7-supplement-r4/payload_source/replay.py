#!/usr/bin/env python3
"""Replay one frozen zero-input FAG case with paired downloaded Release payloads."""
import argparse,gzip,hashlib,itertools,json,os,re,shutil,subprocess,sys
from pathlib import Path
sys.dont_write_bytecode=True
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__)
 for x in ('operator-root','model-root','outdir'):p.add_argument('--'+x,type=Path,required=True)
 p.add_argument('--preflight',action='store_true',help='verify identities/configuration only; never run simulator');p.add_argument('--case',required=True);p.add_argument('--timeout',type=int,default=3600);a=p.parse_args()
 op,model,out=a.operator_root.resolve(),a.model_root.resolve(),a.outdir.resolve()
 if out.exists() or any(out==x or out.is_relative_to(x) for x in (op,model)):p.error('choose new output directory outside payloads')
 manifest=json.loads((op/'manifest.json').read_text());matches=[c for c in manifest['cases'] if c['id']==a.case]
 if len(matches)!=1:p.error('unknown or duplicate case id')
 case=matches[0]
 env={k:v for k,v in os.environ.items() if k in {'PATH','LANG','LC_ALL','TZ'}};env.update(PYTHONDONTWRITEBYTECODE='1',**case['environment'])
 for root in (op,model):
  expected_files={x['path'] for x in json.loads((root/'files.json').read_text())['files']}|{'files.json'}
  actual_files=set()
  for entry in root.rglob('*'):
   if entry.is_symlink():raise SystemExit('Payload symlink rejected: '+str(entry.relative_to(root)))
   if entry.is_file():actual_files.add(entry.relative_to(root).as_posix())
  if actual_files!=expected_files:raise SystemExit('Payload file set differs: '+str(sorted(actual_files^expected_files)))
  subprocess.run([sys.executable,str(root/'verify.py')],env=env,check=True)
 runtime=model/'bin'/case['runtime_variant']/'gfsim';elf=op/case['elf']
 expected=json.loads((model/'runtime_identity.json').read_text())['variants'][case['runtime_variant']]['sha256']
 if expected!=manifest['runtime_variants'][case['runtime_variant']]['sha256']:raise SystemExit('Model runtime manifest differs from operator binding')
 for name,expected_config in manifest['model_configuration_sha256'].items():
  if sha(model/name)!=expected_config:raise SystemExit('Model configuration differs from operator binding: '+name)
 if sha(runtime)!=expected or sha(elf)!=case['elf_sha256']:raise SystemExit('ELF/runtime identity mismatch')
 if a.preflight:
  print('PREFLIGHT ONLY: identities verified; no simulation or numerical claim');return
 out.mkdir(parents=True);(out/'tmp').mkdir();env['TMPDIR']=str(out/'tmp')
 shutil.copytree(model/'configuration_inputs'/case['config_set'],out/'configs')
 shutil.copy2(model/'case_configs'/(case['config_id']+'.conf'),out/'configs/perf.conf')
 command=[sys.executable,str(op/'tools/run_swimlane.py'),'--gfsim',str(model/'run_model.py'),'--elf',str(elf),'--outdir',str(out/'perf_run'),'--name',a.case,'--gzip-full','--conf','configs/perf.conf','--timeout',str(a.timeout)]
 env['FAG_RUNTIME_VARIANT']=case['runtime_variant']
 result={'command':command,'cwd':str(out),'environment':env,'removed_inherited_environment_names':sorted(set(os.environ)-set(env)),'numeric_status':'not_checked_in_timing_run; zero input; separate functional evidence is scoped baseline equivalence'}
 (out/'invocation.json').write_text(json.dumps(result,indent=2)+'\n');proc=subprocess.run(command,cwd=out,env=env)
 log=out/'perf_run'/(a.case+'_swim.log');matches=re.findall(r'^Total Cycles[.\s]*:\s*(\d+)\s*$',log.read_text() if log.exists() else '',re.M);cycles=int(matches[-1]) if matches else None
 config_error=bool(re.search(r'Error: fail to open file:|(?:failed|cannot|could not|error).*?(?:config|conf)|(?:config|conf).*?(?:not found|failed|error)',log.read_text() if log.exists() else '',re.I))
 expected_overrides=[line.split('#',1)[0].strip() for line in (out/'configs/perf.conf').read_text().splitlines() if '=' in line.split('#',1)[0]]
 observed_log=log.read_text() if log.exists() else ''
 missing_overrides=[line for line in expected_overrides if line.replace(' ','') not in observed_log.replace(' ','')]
 files={}
 for suffix in ('_swim.log','.json.gz','_nocounters.json','_full_identity.json'):
  f=out/'perf_run'/(a.case+suffix)
  if f.is_file():files[f.name]={'sha256':sha(f),'bytes':f.stat().st_size}
 full=out/'perf_run'/(a.case+'.json.gz');identity=out/'perf_run'/(a.case+'_full_identity.json');stream={'sha256':None,'bytes':0}
 if full.is_file() and identity.is_file():
  h=hashlib.sha256();size=0
  with gzip.open(full,'rb') as f:
   for b in iter(lambda:f.read(1048576),b''):h.update(b);size+=len(b)
  stream={'sha256':h.hexdigest(),'bytes':size}
  capture=json.loads(identity.read_text())
  stream['capture_matches']=capture.get('sha256')==stream['sha256'] and capture.get('bytes')==size and size>0 and 'error' not in capture
 historical=case['historical_full_identity'];stream['historical_matches']=all(stream[k]==historical[k] for k in ('sha256','bytes'))
 noc=out/'perf_run'/(a.case+'_nocounters.json')
 semantic={'filtered_full_equal':False,'historical_canonical_equal':False}
 if noc.is_file() and full.is_file():
  sys.path.insert(0,str(op/'tools'));from trace_equivalence import digest
  full_normal=digest(full);filtered=digest(full,filter_counters=True);noc_normal=digest(noc)
  historical_full=case['historical_full_normalized'];historical_noc=case['historical_noc_normalized']
  semantic={'filtered_full_equal':filtered==noc_normal,'full_normalized_historical_equal':all(full_normal[k]==historical_full[k] for k in ('canonical_sha256','events','metadata')),'historical_canonical_equal':all(noc_normal[k]==historical_noc[k] for k in ('canonical_sha256','events','metadata')),'events':noc_normal['events'],'canonical_sha256':noc_normal['canonical_sha256'],'historical_raw_sha256':case['historical_nocounters_sha256'],'actual_raw_sha256':sha(noc),'raw_hash_equal':sha(noc)==case['historical_nocounters_sha256'],'full_normalized':full_normal,'historical_inverse_annotation_events':historical_full['inverse_annotation_events'],'serialization_note':'Historical pure annotations inverted by exact documented rule; all other event fields/order/flows and root metadata canonical compared. Raw hashes remain distinct observations.'}
 trace_match=stream.get('capture_matches',False) and semantic.get('full_normalized_historical_equal',False) and semantic['filtered_full_equal'] and semantic['historical_canonical_equal']
 passed=not config_error and not missing_overrides and proc.returncode==0 and cycles==case['cycles'] and len(files)==4 and trace_match
 result.update(config_error=config_error,missing_overrides=missing_overrides,status='pass' if passed else 'fail',returncode=proc.returncode,total_cycles=cycles,expected_cycles=case['cycles'],files=files,nocounters_semantics=semantic,full_uncompressed=stream,runtime_sha256=sha(runtime),elf_sha256=sha(elf))
 (out/'replay_result.json').write_text(json.dumps(result,indent=2)+'\n')
 if not passed:raise SystemExit('Simulator exit, cycles, capture identity or historical trace differs')
if __name__=='__main__':main()
