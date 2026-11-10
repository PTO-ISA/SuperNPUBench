#!/usr/bin/env python3
"""Replay shipped ELF, from any cwd, with an explicitly supplied hash-pinned model."""
import argparse, hashlib, json, os, re, resource, subprocess, sys
from pathlib import Path
from golden import input_and_golden, compare
root=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('case',choices=['split2','baseline']);p.add_argument('--phase',choices=['performance','gfrun'],required=True)
 p.add_argument('--model-root',type=Path,required=True);p.add_argument('--outdir',type=Path,required=True);p.add_argument('--timeout',type=int,default=180)
 a=p.parse_args();model=a.model_root.resolve();out=a.outdir.resolve();case=root/'cases'/a.case;elf=case/(a.case+'.elf')
 manifest=json.loads((root/'operator_identity.json').read_text());expected=manifest['cases'][a.case]
 assert sha(elf)==expected['elf_sha256'],'Shipped ELF identity mismatch'
 if out.exists():raise SystemExit('Output directory already exists; choose a fresh path')
 exe=model/'bin'/('gfsim' if a.phase=='performance' else 'gfrun')
 expected_runtime=manifest['timing_model_sha256' if a.phase=='performance' else 'gfrun_sha256']
 assert sha(exe)==expected_runtime,'Model hash mismatch'
 generated=input_and_golden()
 for name,blob in zip(['input.bin','golden_payload.bin','golden_scales.bin'],generated):
  assert (root/'inputs'/name).read_bytes()==blob,'Shipped golden/input differs from independent generator'
 out.mkdir(parents=True);env=dict(os.environ);env.pop('MXQ_DUMP_DIR',None)
 result={'case':a.case,'phase':a.phase,'elf_sha256':sha(elf),'runtime_sha256':sha(exe),'numeric_timing_acceptance':False}
 if a.phase=='performance':
  env['MXQUANT_TIMING_MODEL_ROOT']=str(model)
  cmd=[sys.executable,str(root/'run_swimlane.py'),'--gfsim',str(root/'run_model.py'),'--elf',str(elf),'--outdir',str(out),'--name','mxquant','--timeout',str(a.timeout),'--conf',str(root/'configs/rms.conf')]
 else:
  symbols={x.split()[2]:int(x.split()[0],16) for x in (case/'symbols.txt').read_text().splitlines() if len(x.split())==3}
  begin=symbols['input'];end=max(symbols['output']+131072,symbols['scales']+4096)
  cmd=[str(exe),'-s','softcore.multiThreadNum=4','-f',str(elf),'--dump-memory',f'{hex(begin)}:{end-begin}:{out/"memory.bin"}']
 (out/'command.json').write_text(json.dumps(cmd,indent=2)+'\n')
 def limits():
  resource.setrlimit(resource.RLIMIT_CORE,(0,0));resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,3*1024**3));os.nice(15)
 with (out/'replay.log').open('w') as log:
  try:rc=subprocess.run(cmd,cwd=model,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=a.timeout+20,preexec_fn=limits if a.phase=='gfrun' else None).returncode
  except subprocess.TimeoutExpired:rc=124
 result['returncode']=rc
 if a.phase=='performance':
  text=(out/'mxquant_swim.log').read_text() if (out/'mxquant_swim.log').exists() else ''
  matches=re.findall(r'Total Cycles[^\n]*?:\s*(\d+)',text)
  result['total_cycles']=int(matches[-1]) if matches else None
  result['expected_cycles']=expected['expected_cycles']
  result['pass']=rc==0 and result['total_cycles']==result['expected_cycles'] and all((out/name).is_file() and (out/name).stat().st_size for name in ['mxquant.json','mxquant_nocounters.json','mxquant_swim.log'])
  result['scope']='Historical timing replay only; no numeric or TLSU-bound acceptance'
 else:
  dump=(out/'memory.bin').read_bytes() if (out/'memory.bin').exists() else b''
  for name,blob in zip(['input','output','scales'],generated):
   off=symbols[name]-begin;actual=dump[off:off+len(blob)];(out/(name+'.bin')).write_bytes(actual)
   result[name]=compare(actual,blob)
  corrupt=bytearray(generated[1]);corrupt[17]^=1
  result['negative_control_rejected']=not compare(dump[symbols['output']-begin:symbols['output']-begin+len(corrupt)],corrupt)['pass']
  result['pass']=rc==0 and all(result[n]['pass'] for n in ['input','output','scales']) and result['negative_control_rejected']
 (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
 return 0 if result['pass'] else 1
if __name__=='__main__':raise SystemExit(main())
