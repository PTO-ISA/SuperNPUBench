#!/usr/bin/env python3
"""Build the measured freestanding D/D2 recipe with explicitly pinned tools."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',required=True,help='historical result id without portable suffix')
    p.add_argument('--toolchain-root',required=True,type=Path,help='installation parent containing pinned build/bin tools; headers are shipped here')
    p.add_argument('--outdir',required=True,type=Path)
    a=p.parse_args();t=a.toolchain_root.resolve();out=a.outdir.resolve()
    if out.exists() or out.is_relative_to(ROOT):p.error('choose new output directory outside operator payload')
    subprocess.run([sys.executable, '-B', str(ROOT/'verify.py')], check=True)
    identity=json.loads((ROOT/'dependencies/identity.json').read_text());recipes=json.loads((ROOT/'build_recipes.json').read_text())
    if a.case not in recipes:p.error('unknown case')
    recipe=recipes[a.case]
    for name,key in [('clang++','clang_sha256'),('ld.lld','lld_sha256'),('llvm-objdump','objdump_sha256')]:
        file=t/'build/bin'/name
        if not file.is_file() or sha(file)!=identity['toolchain'][key]:raise SystemExit('FAIL: missing/mismatched tool '+name)
    for base,files in [(ROOT/'dependencies/api',identity['api']['files_sha256']),(ROOT/'dependencies/harness',identity['harness']['files_sha256'])]:
        for rel,digest in files.items():
            file=base/rel
            if not file.is_file() or sha(file)!=digest:raise SystemExit('FAIL: dependency identity '+rel)
    source=ROOT/recipe['source_snapshot'];api=ROOT/'dependencies/api';h=ROOT/'dependencies/harness';sysroot=ROOT/'dependencies/sysroot'
    pins=json.loads((ROOT/'dependencies/sysroot_headers.json').read_text())
    for rel,digest in pins.items():
        file=sysroot/rel
        if not file.is_file() or sha(file)!=digest:raise SystemExit('FAIL: required sysroot header '+rel)
    if not pins:raise SystemExit('FAIL: no validated sysroot header closure')
    out.mkdir(parents=True);(out/'tmp').mkdir();cxx=t/'build/bin/clang++'
    flags=recipe['flags']+['--sysroot='+str(sysroot),'-resource-dir='+str(api/'resource')]
    flags += ['-I'+str(x) for x in [api/'include/tileop-api',source,h,h/'src']]
    flags += ['-ffile-prefix-map='+str(ROOT)+'=FAArchive','-fdebug-prefix-map='+str(ROOT)+'=FAArchive','-ffile-prefix-map='+str(out)+'=build','-fdebug-prefix-map='+str(out)+'=build','-ffile-prefix-map='+str(t)+'=LinxToolchain','-fdebug-prefix-map='+str(t)+'=LinxToolchain']
    elf=out/(a.case+'.elf');commands=[ [str(cxx),'-c']+flags+[str(source/'fa_lowp_algD_driver.cpp'),'-o',str(out/'kernel.o')],
        [str(cxx),'--sysroot='+str(sysroot),'-resource-dir='+str(api/'resource'),'-nostdlib','-fuse-ld='+str(t/'build/bin/ld.lld'),str(h/'_start.s'),str(out/'kernel.o'),'-o',str(elf)] ]
    env={k:v for k,v in os.environ.items() if k in {'PATH','LANG','LC_ALL','TZ'}};env['TMPDIR']=str(out/'tmp')
    for n,cmd in enumerate(commands):
        with (out/('step'+str(n)+'.log')).open('w') as log:subprocess.run(cmd,cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    with (out/(elf.name+'.diss')).open('w') as f:subprocess.run([str(t/'build/bin/llvm-objdump'),'-dl',elf.name],cwd=out,env=env,stdout=f,check=True)
    result={'commands':commands,'cwd':str(out),'elf_sha256':sha(elf),'historical_elf_sha256':recipe['elf_sha256'],'historical_elf_bytes_equal':sha(elf)==recipe['elf_sha256'],'diss_sha256':sha(out/(elf.name+'.diss')),'source_identity':'manifest.json','scope':'build only; qualify with replay.py --qualification-mode --rebuilt-elf'}
    (out/'build_result.json').write_text(json.dumps(result,indent=2)+'\n');(out/'kernel.o').unlink()
    print(json.dumps({k:v for k,v in result.items() if k!='commands'}))

if __name__=='__main__':main()
