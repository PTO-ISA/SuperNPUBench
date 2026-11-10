#!/usr/bin/env python3
"""r2 fail-closed functional launcher; historical performance/build are not provided."""
import argparse, hashlib, json, os, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
RUNTIME='9b0cf76eaa3ac62902f9511929a961cb064c8930fa9f9af7f8d967aa57eb1fb9'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('operation',choices=['gfrun','performance','build-recipe'])
 p.add_argument('--operator-root',type=Path,required=True)
 p.add_argument('--model-root',type=Path)
 p.add_argument('--case',choices=['baseline','split2'],default='split2')
 p.add_argument('--outdir',type=Path,required=True)
 a=p.parse_args(); out=a.outdir.resolve()
 if out.exists():p.error('Output must be a fresh directory')
 out.mkdir(parents=True)
 report={'operation':a.operation,'case':a.case,'runtime_executed':False,'status':'rejected','returncode':2}
 def finish(reason,rc=2):
  report['reason']=reason;report['returncode']=rc
  (out/'guard_result.json').write_text(json.dumps(report,indent=2)+'\n');print(reason);return rc
 if a.operation=='performance':return finish('Performance refused: portable historical timing runtime and verified complete frozen configuration closure are not delivered. Historical cycle matching is not numerical acceptance.')
 if a.operation=='build-recipe':return finish('Build refused: only a historical recipe is available; full compiler/linker/API/harness/resource/sysroot closure is not pinned or verified.')
 if a.model_root is None:return finish('Explicit model root is required')
 operator=a.operator_root.resolve(); model=a.model_root.resolve(); runtime=model/'bin/gfrun'
 try:
  if sha(runtime)!=RUNTIME:return finish('Functional runtime identity mismatch')
  pins=json.loads((ROOT/'functional_pins.json').read_text())
  for item in pins['operator_files']:
   f=operator/item['path']
   if not f.is_file() or f.stat().st_size!=item['bytes'] or sha(f)!=item['sha256']:return finish('Operator file missing or identity mismatch: '+item['path'])
 except (OSError,ValueError,KeyError) as e:return finish('Dependency check failed: '+type(e).__name__)
 # Allowlist drops loader, Python, model and user-shell overrides; no inherited configuration.
 env={'PATH':'/usr/bin:/bin','LANG':'C.UTF-8','LC_ALL':'C.UTF-8','PYTHONNOUSERSITE':'1','TMPDIR':str(out)}
 run=out/'run';cmd=[sys.executable,'-B','-E','-s',str(operator/'replay.py'),a.case,'--phase','gfrun','--model-root',str(model),'--outdir',str(run)]
 report.update(runtime_sha256=RUNTIME,operator_pins_sha256=sha(ROOT/'functional_pins.json'),runtime_executed=True,environment_policy='allowlist PATH/LANG/LC_ALL/PYTHONNOUSERSITE/TMPDIR; no inherited loader/model/Python variables')
 with (out/'launcher.log').open('w') as log:
  try:rc=subprocess.run(cmd,cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=220).returncode
  except subprocess.TimeoutExpired:return finish('Functional runner timed out',124)
 try:r=json.loads((run/'result.json').read_text())
 except (OSError,ValueError):return finish('Runner produced no valid result',1)
 report['runner_result']=r
 if rc or not r.get('pass'):return finish('Functional runner failed',rc or 1)
 report['status']='pass';return finish('Fixed-ELF local functional validation only; shared runtime and timing/build acceptance remain unavailable.',0)
if __name__=='__main__':raise SystemExit(main())
