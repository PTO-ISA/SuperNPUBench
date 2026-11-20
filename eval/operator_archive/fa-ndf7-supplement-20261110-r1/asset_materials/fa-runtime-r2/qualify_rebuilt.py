#!/usr/bin/env python3
"""Qualify four FA cases with a rebuilt runtime; archive runtime SHA need not match."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--operator-root',required=True,type=Path)
    p.add_argument('--runtime',required=True,type=Path)
    p.add_argument('--outdir',required=True,type=Path)
    a=p.parse_args();root=Path(__file__).resolve().parent;operator=a.operator_root.resolve();runtime=a.runtime.resolve();out=a.outdir.resolve()
    if out.exists() or not runtime.is_file():p.error('runtime must exist and outdir must be new')
    if any(out.is_relative_to(x) for x in [root,operator]):p.error('output must be outside model and operator payloads')
    expected={'fa-8k-algod-21488-portable-r1','fa-8k-scoped-d2-20519-portable-r1','fa-64k-algod-155269-portable-r1','fa-64k-unscoped-d2-258099-portable-r2'}
    cases=json.loads((operator/'manifest.json').read_text())['cases']
    if len(cases)!=4 or {c.get('id') for c in cases}!=expected:p.error('qualification requires the four fixed archived cases')
    for payload in [root,operator]:
        command=[sys.executable,'-B',str(payload/'verify.py')]
        if payload==root:command.append('--data-only')
        subprocess.run(command,check=True)
    out.mkdir(parents=True)
    rows=[]
    for case in cases:
        run=out/case['id']
        command=[sys.executable,str(operator/'replay.py'),'--operator-root',str(operator),'--model-root',str(root),'--case',case['id'],'--outdir',str(run),'--qualification-mode','--rebuilt-runtime',str(runtime)]
        r=subprocess.run(command,check=False)
        rows.append({'case':case['id'],'returncode':r.returncode,'result':str(run.relative_to(out)/'replay_result.json')})
        if r.returncode:break
    result={'runtime_sha256':hashlib.sha256(runtime.read_bytes()).hexdigest(),'cases':rows,
            'status':'pass' if len(rows)==len(cases) and all(x['returncode']==0 for x in rows) else 'fail',
            'scope':'four measured zero-input timing cases and full traces only; not numerical or all-host equivalence'}
    (out/'qualification.json').write_text(json.dumps(result,indent=2)+'\n')
    if result['status']!='pass':raise SystemExit('FAIL: rebuilt runtime qualification')

if __name__=='__main__':main()
