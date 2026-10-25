#!/usr/bin/env python3
"""Optional orchestration; all target builds use Makefile/compile.all."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import numpy as np
HERE=Path(__file__).resolve().parent
SRC=HERE/'src'
sys.path.insert(0,str(SRC))
from gen_ctcloss_v2_data import generate_network
from ctcloss_v2_data_compare import compare
ONE=HERE.parents[2]
REPO=ONE.parent.parent
WORKSPACE=REPO.parent

def digest(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--compiler-dir',type=Path,default=Path(os.environ.get('COMPILER_DIR',WORKSPACE/'linx-toolchain-build/output/linx_blockisa_llvm_musl/bin')))
    p.add_argument('--api-include',type=Path,default=Path(os.environ.get('API_INCLUDE',WORKSPACE/'Linx-TileOP-API-b223de6/include')))
    p.add_argument('--model',type=Path,default=Path(os.environ.get('SSM',WORKSPACE/'SuperScalarModel-8e8d8ac7')))
    p.add_argument('--build-only',action='store_true')
    p.add_argument('--functional-only',action='store_true')
    p.add_argument('--out',type=Path)
    a=p.parse_args()
    compiler=a.compiler_dir.resolve(); api=a.api_include.resolve(); model=a.model.resolve()
    out=(a.out or WORKSPACE/'validation_logs'/('ctcloss-normalized-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+str(os.getpid()))).resolve()
    out.mkdir(parents=True,exist_ok=False)
    report=dict(status='RUNNING',steps=[],variants={},output=str(out),rtol=1e-4,atol=2e-5)
    env=dict(os.environ,COMPILER_DIR=str(compiler),API_INCLUDE=str(api),CTC_COMPARE_ROOT=str(out))
    def save(): (out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    def execute(stage,cmd,cwd=HERE,timeout=120,extra_env=None):
        stage_env=env.copy()
        if extra_env: stage_env.update(extra_env)
        item=dict(stage=stage,command=list(map(str,cmd)),cwd=str(cwd))
        report['steps'].append(item); save(); start=time.monotonic()
        print('['+stage+'][running]',flush=True)
        with (out/(stage+'.log')).open('w') as f:
            r=subprocess.run(['timeout','-k','10s',str(timeout)+'s',*item['command']],cwd=cwd,env=stage_env,stdout=f,stderr=subprocess.STDOUT)
        item.update(exit_code=r.returncode,wall_seconds=time.monotonic()-start); save()
        if r.returncode: raise RuntimeError(f'{stage}: exit {r.returncode}; see {out/(stage+".log")}')
    def elf(pe,mode):
        case='ctcloss_v2_single_pe' if pe==1 else 'ctcloss_v2_four_pe'
        return out/'elf'/f'solution_ctcloss_{case}_DTypefloat_T195_N256_D41_U74_PE{pe}_rescheck{mode}.elf'
    try:
        for pe in (1,4):
            work=out/f'pe{pe}'; work.mkdir()
            generate_network(work/'compare',pe_num=pe)
            report['variants'][str(pe)]=dict(gfrun='NOT_RUN',gfsim='NOT_RUN')
        execute('host-build',['g++','-std=c++17','-O2','-Wall','-Wextra','-Werror',
            '-I'+str(ONE/'kernels'),SRC/'validate_ctcloss_v2_inputs.cpp','-o',out/'validate_inputs'])
        for pe in (1,4): execute(f'pe{pe}-input-check',[out/'validate_inputs',out/f'pe{pe}/compare'])
        tools=[compiler/'clang++',compiler/'lld',compiler/'llvm-objdump',model/'bin/gfrun',model/'bin/gfsim']
        report['tools_sha256']={str(t):digest(t) if t.exists() else 'UNAVAILABLE' for t in tools}
        report['repositories']={}
        for name,path in [('benchmark',REPO),('model',model),('tileop',api.parent),
                          ('compiler',WORKSPACE/'linx-toolchain-build/src/llvm-project')]:
            if not path.exists(): continue
            try:
                report['repositories'][name]=dict(
                    sha=subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip(),
                    status=subprocess.check_output(['git','-C',str(path),'status','--short'],text=True),
                    tracked_diff_sha256=hashlib.sha256(subprocess.check_output(['git','-C',str(path),'diff','--binary'])).hexdigest())
            except subprocess.CalledProcessError: pass
        source_paths=[HERE/'Makefile',HERE/'compile.all',HERE/'validate.py',*SRC.glob('*.cpp'),*SRC.glob('*.hpp'),*SRC.glob('*.py'),
                      *(ONE/'kernels/solution/ctcloss').glob('*.hpp')]
        report['source_sha256']={str(t.relative_to(REPO)):digest(t) for t in source_paths}
        args=[f'OBJ_ROOT={out/"objects"}',f'ELF_DIR={out/"elf"}']
        for mode in ('on','off'):
            for pe in (1,4):
                case='ctcloss_v2_single_pe' if pe==1 else 'ctcloss_v2_four_pe'
                plan_cmd=['make','-n','-o','clean',f'TESTCASE={case}',f'PE_NUM={pe}',f'res_check={mode}',
                          f'CHK_DIR={out/f"pe{pe}/compare"}',*args,'diss']
                plan=subprocess.check_output(plan_cmd,cwd=HERE,env=env,text=True)
                (out/f'pe{pe}-{mode}-make-plan.txt').write_text(plan)
                assert not re.search(r'^\s*(?:rm\s|find\s.*-exec)',plan,re.M),'unexpected clean command'
            execute('compile-all-'+mode,['bash',HERE/'compile.all',*args],extra_env={'res_check':mode},timeout=240)
            for pe in (1,4):
                binary=elf(pe,mode)
                dis=Path(str(binary)+'.diss').read_text()
                assert all(op.lower() in dis.lower() for op in ['TEXP','TLOG','TLEA','MGATHER'])
                if pe==4: assert re.search(r'ssrget\s+(?:2050|0x802)\b',dis,re.I)
                symbols=subprocess.check_output([str(compiler/'llvm-objdump'),'-t',str(binary)],text=True)
                assert '__linx_group_worker_start' in symbols,'missing hosted worker entry'
                assert ' main' in symbols and '_start' in symbols
                report['variants'][str(pe)]['res_check_'+mode]=dict(status='BUILD_PASS',elf=str(binary),sha256=digest(binary))
            save()
        if a.build_only:
            report['status']='BUILD_PASS_NOT_RUN'; save()
            print('BUILD_PASS_NOT_RUN: '+str(out)); return
        failed=False
        for pe in (1,4):
            result=report['variants'][str(pe)]
            try:
                execute(f'pe{pe}-gfrun',[model/'bin/gfrun','--pto059-execution-mask-dev','-s',f'softcore.multiThreadNum={pe}','-f',elf(pe,'on')],cwd=model)
                log=(out/f'pe{pe}-gfrun.log').read_text()
                assert 'CTCLOSS_OUTPUT_WRITTEN' in log,'guest did not export outputs'
                assert 'Reach the End of Benchmark! R2 = 0' in log,'guest did not finish successfully'
                result['gfrun']=compare(out/f'pe{pe}/compare')
                if not a.functional_only:
                    extra=[] if pe==1 else ['--conf','fourpe']
                    settings=['core.threadCount=1','core.stdPeCount=1','tlsu.fake_l2_enable=false'] if pe==1 else ['tlsu.fake_l2_enable=false']
                    execute(f'pe{pe}-gfsim',[model/'bin/gfsim','--pto059-execution-mask-dev',*extra,'-s',*settings,'-f',elf(pe,'off')],cwd=model,timeout=900)
                    log=(out/f'pe{pe}-gfsim.log').read_text()
                    assert not re.search(r'LOG_ERROR|ASSERTION FAILED',log)
                    assert re.search(r'invariants:.*stq_conservation=OK.*a3_tile_violation=0\b',log)
                    result['gfsim']=dict(status='EXECUTION_PASS',whole_program_cycles=int(re.search(r'^Total Cycles\.+:\s*(\d+)',log,re.M).group(1)),
                        cycle_scope='includes launch and input I/O; not a kernel-only speedup measurement')
                result['status']='FUNCTIONAL_PASS' if a.functional_only else 'PASS'
            except Exception as e:
                result.update(status='FAIL',error=str(e)); failed=True
            save()
        report['status']='FAIL' if failed else 'FUNCTIONAL_PASS' if a.functional_only else 'PASS'
        save()
        if failed: raise SystemExit(1)
    except Exception as e:
        report.update(status='FAIL',error=str(e)); save(); raise

if __name__=='__main__': main()
