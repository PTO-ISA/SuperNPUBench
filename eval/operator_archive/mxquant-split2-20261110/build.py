#!/usr/bin/env python3
"""Optional rebuild with explicit historical toolchain and benchmark checkout; ELF replay needs neither."""
import argparse,hashlib,json,os,subprocess
from pathlib import Path
root=Path(__file__).resolve().parent
p=argparse.ArgumentParser(description=__doc__);p.add_argument('case',choices=['baseline','split2']);p.add_argument('--toolchain-root',type=Path,required=True);p.add_argument('--benchmark-root',type=Path,required=True);p.add_argument('--outdir',type=Path,required=True);a=p.parse_args()
case=root/'cases'/a.case;toolchain=a.toolchain_root.resolve();out=a.outdir.resolve()
expected=json.loads((case/'identity.json').read_text())['compiler_sha256']
if hashlib.sha256((toolchain/'bin/clang++').read_bytes()).hexdigest()!=expected:raise SystemExit('Historical compiler hash mismatch')
if out.exists():raise SystemExit('Output exists; choose a fresh directory')
variables={'TOOLCHAIN':toolchain,'BENCHMARK':a.benchmark_root.resolve(),'SOURCE':case/'source','INPUTS':root/'inputs','OUTPUT':out}
def expand(s):
 for key,value in variables.items():s=s.replace('${'+key+'}',str(value))
 if '${' in s:raise ValueError('Unresolved recipe variable')
 return s
commands=[[expand(s) for s in cmd] for cmd in json.loads((case/'build_recipe.json').read_text())['commands']]
out.mkdir(parents=True);env=dict(os.environ,TMPDIR=str(out))
with (out/'build.log').open('w') as log:
 for cmd in commands:subprocess.run(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
elf=out/(a.case+'.elf');sha=hashlib.sha256(elf.read_bytes()).hexdigest();original=hashlib.sha256((case/(a.case+'.elf')).read_bytes()).hexdigest()
with (out/(a.case+'.elf.diss')).open('w') as diss:subprocess.run([str(toolchain/'bin/llvm-objdump'),'-dl',str(elf)],stdout=diss,check=True)
(out/'build_result.json').write_text(json.dumps({'elf_sha256':sha,'historical_elf_sha256':original,'byte_identical':sha==original,'limitation':'Historical compiler/API/benchmark closure required; different source dependencies may change bytes. This script has not been newly compiled during archival preparation.'},indent=2)+'\n')
print(elf,sha)
