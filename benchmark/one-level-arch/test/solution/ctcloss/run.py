#!/usr/bin/env python3
"""Build and optionally simulate the sole Ctcloss_v2 network specification."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
import numpy as np
from generate import generate_network

HERE=Path(__file__).resolve().parent
ONE=HERE.parents[2]
REPO=ONE.parent.parent
WORKSPACE=REPO.parent

def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()

def repository(path):
    try:
        sha=subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()
        diff=subprocess.check_output(['git','-C',str(path),'diff','--binary'])
        status=subprocess.check_output(['git','-C',str(path),'status','--short'],text=True)
        return dict(path=str(path),sha=sha,tracked_diff_sha256=hashlib.sha256(diff).hexdigest(),status=status)
    except subprocess.CalledProcessError:
        return dict(path=str(path),source_revision='unavailable')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--pe-count',nargs='+',type=int,choices=[1,4],default=[1,4])
    p.add_argument('--build-only',action='store_true',help='Do not claim a simulator PASS')
    p.add_argument('--functional-only',action='store_true')
    p.add_argument('--out',type=Path)
    p.add_argument('--toolchain',type=Path,default=Path(os.environ.get('LINX_RUNTIME_ROOT',WORKSPACE/'linx-toolchain-build/output/linx_blockisa_llvm_musl')))
    p.add_argument('--model',type=Path,default=Path(os.environ.get('SSM',WORKSPACE/'SuperScalarModel-8e8d8ac7')))
    p.add_argument('--tileop-include',type=Path,default=Path(os.environ.get('API_INCLUDE',WORKSPACE/'Linx-TileOP-API-b223de6/include')))
    a=p.parse_args()
    a.toolchain=a.toolchain.resolve(); a.model=a.model.resolve()
    a.tileop_include=a.tileop_include.resolve()
    out=(a.out or WORKSPACE/'validation_logs'/('ctcloss-solution-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+str(os.getpid()))).resolve()
    out.mkdir(parents=True,exist_ok=False)
    data=out/'data'; data.mkdir()
    report=dict(status='RUNNING',output=str(out),steps=[],variants={},
                versions=dict(benchmark=repository(REPO),model=repository(a.model),
                              tileop=repository(a.tileop_include.parent)),
                numpy_version=np.__version__,rtol=1e-4,atol=2e-5)
    compiler_source=WORKSPACE/'linx-toolchain-build/src/llvm-project'
    if compiler_source.exists(): report['versions']['compiler']=repository(compiler_source)
    def save():
        (out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    def execute(stage,cmd,cwd=REPO,timeout=120):
        entry=dict(stage=stage,command=list(map(str,cmd)),cwd=str(cwd))
        report['steps'].append(entry); save(); start=time.monotonic()
        print('['+stage+'][running]',flush=True)
        with (out/(stage+'.log')).open('w') as log:
            r=subprocess.run(['timeout','-k','10s',str(timeout)+'s',*entry['command']],
                             cwd=cwd,stdout=log,stderr=subprocess.STDOUT)
        entry.update(exit_code=r.returncode,wall_seconds=time.monotonic()-start); save()
        if r.returncode: raise RuntimeError(f'{stage}: exit {r.returncode}; see {out/(stage+".log")}')
    try:
        generate_network(data)
        execute('host-build',['g++','-std=c++17','-O2','-Wall','-Wextra','-Werror',
                '-I'+str(ONE/'kernels'),HERE/'validate_inputs.cpp','-o',out/'validate_inputs'])
        execute('host-input-check',[out/'validate_inputs',data])
        clang=a.toolchain/'bin/clang++'; ld=a.toolchain/'bin/lld'
        objdump=a.toolchain/'bin/llvm-objdump'
        tools=[clang,ld,objdump,a.model/'bin/gfrun',a.model/'bin/gfsim']
        report['tools_sha256']={str(t):digest(t) for t in tools}
        sources=[*HERE.glob('*.py'),*HERE.glob('*.cpp'),HERE/'start.s',
                 *(ONE/'kernels/solution/ctcloss').glob('*.hpp')]
        report['sources_sha256']={str(t.relative_to(REPO)):digest(t) for t in sources}
        report['input_sha256']={str(t.relative_to(out)):digest(t) for t in data.glob('*.bin')}
        common=[clang,'--target=linx64v5-unknown-linux-musl','--sysroot='+str(a.toolchain/'sysroot'),
                '-resource-dir='+str(a.toolchain/'lib/clang/15.0.4')]
        flags=['-mlxbc','-fenable-matrix','-O2','-std=c++20','-fno-vectorize',
               '-fno-slp-vectorize','-fno-fast-math','-gline-tables-only','-Werror=unknown-pragmas',
               '-DSUPERNPUBENCH_LINX_TILE_SIZE','-DLINX_ELEMENTWISE_AVAILABLE','-D__linx','-DENABLE_TENSOR_INSTR']
        for flag in ['enable-all-vector-as-tilereg=true','linxv5-enable-HL-Inst-Opt=true',
                     'linxv5-enable-dim-opt=true','linxv5-enable-ldst-bridge=false',
                     'linxv5-enable-continuous-mem-opt=true','linxv5-enable-tile-clock-hand=false',
                     'linxv5-enable-simt-clock-hand=true','enable-misched=false','linxv5-enable-generic-element=true']:
            flags+=['-mllvm','-'+flag]
        flags+=['-I'+str(path) for path in [a.tileop_include,ONE/'kernels',data]]
        execute('data-assembly',common+['-c',data/'data.S','-o',out/'data.o'])
        failed=False
        for pe in dict.fromkeys(a.pe_count):
            work=out/f'pe{pe}'; work.mkdir()
            result=dict(status='BUILDING',gfrun='NOT_RUN',gfsim='NOT_RUN')
            report['variants'][str(pe)]=result
            try:
                variant_flags=flags+[f'-DCTC_PE_COUNT={pe}']
                execute(f'pe{pe}-compile',common+variant_flags+['-c',HERE/'driver.cpp','-o',work/'ctc.o'])
                execute(f'pe{pe}-link',common+['-fuse-ld=lld','-nostartfiles',HERE/'start.s',work/'ctc.o',out/'data.o','-o',work/'ctc.elf'])
                execute(f'pe{pe}-ir',common+variant_flags+['-S','-emit-llvm',HERE/'driver.cpp','-o',work/'ctc.ll'])
                execute(f'pe{pe}-disassembly',[objdump,'-d',work/'ctc.elf'])
                dis=(out/f'pe{pe}-disassembly.log').read_text()
                assert all(op.lower() in dis.lower() for op in ['TEXP','TLOG','TLEA','MGATHER'])
                if pe==4: assert re.search(r'ssrget\s+(?:2050|0x802)\b',dis,re.I)
                assert not re.search(r'call[^\n]*@llvm\.linx\.experimental\.element\.(region|view)',(work/'ctc.ll').read_text())
                result.update(status='BUILD_PASS_NOT_RUN',elf_sha256=digest(work/'ctc.elf'))
                save()
                if a.build_only: continue
                symbols_text=subprocess.check_output([str(objdump),'-t',str(work/'ctc.elf')],text=True)
                (work/'symbols.txt').write_text(symbols_text)
                symbols={}
                for line in symbols_text.splitlines():
                    f=line.split()
                    if f and f[-1] in ['ctc_alpha','ctc_loss','ctc_status']:
                        symbols[f[-1]]=(int(f[0],16),int(f[-2],16))
                assert len(symbols)==3
                low=min(x[0] for x in symbols.values()); high=max(x[0]+x[1] for x in symbols.values())
                def compare(runner):
                    memory=(work/(runner+'.mem')).read_bytes()
                    checked={}
                    for name,golden in [('ctc_alpha','alpha.golden'),('ctc_loss','loss.golden')]:
                        address,size=symbols[name]
                        actual=np.frombuffer(memory[address-low:address-low+size],dtype='<f4')
                        expected=np.fromfile(data/golden,dtype='<f4')
                        assert actual.size==expected.size
                        for guard in [slice(0,4),slice(-4,None)]:
                            assert np.array_equal(actual[guard].view('<u4'),expected[guard].view('<u4')),(name,'guard')
                        close=np.isclose(actual[4:-4],expected[4:-4],rtol=1e-4,atol=2e-5,equal_nan=False)
                        assert close.all(),(runner,name,'mismatch at',int(np.flatnonzero(~close)[0]))
                        finite=np.isfinite(expected)
                        checked[name]=dict(elements=int(actual.size),max_absolute_error=float(np.max(np.abs(actual[finite]-expected[finite]))))
                    address,size=symbols['ctc_status']
                    status=np.frombuffer(memory[address-low:address-low+size],dtype='<u4').reshape(pe,4)
                    expected_status=np.array([[1,i+1,0,256//pe] for i in range(pe)],dtype='<u4')
                    assert np.array_equal(status,expected_status),status
                    checked['per_pe_status']=status.tolist()
                    return checked
                for runner in ['gfrun','gfsim']:
                    if runner=='gfsim' and a.functional_only: continue
                    extra=[] if runner=='gfrun' or pe==1 else ['--conf','fourpe']
                    settings=[f'softcore.multiThreadNum={pe}'] if runner=='gfrun' else (
                        ['core.threadCount=1','core.stdPeCount=1','tlsu.fake_l2_enable=false'] if pe==1 else ['tlsu.fake_l2_enable=false'])
                    execute(f'pe{pe}-{runner}',[a.model/'bin'/runner,'--pto059-execution-mask-dev',*extra,'-s',*settings,
                        '--dump-force','--dump-memory',f'{hex(low)}:{hex(high-low)}:{work/(runner+".mem")}','-f',work/'ctc.elf'],
                        cwd=a.model,timeout=120 if runner=='gfrun' else 900)
                    log=(out/f'pe{pe}-{runner}.log').read_text()
                    if runner=='gfrun': assert 'Reach the End of Benchmark! R2 = 0' in log
                    else:
                        assert re.search(r'invariants:.*stq_conservation=OK.*a3_tile_violation=0\b',log)
                        assert not re.search(r'LOG_ERROR|ASSERTION FAILED',log)
                        result['cycles']=int(re.search(r'^Total Cycles\.+:\s*(\d+)',log,re.M).group(1))
                    result[runner]=compare(runner); save()
                result['status']='FUNCTIONAL_PASS' if a.functional_only else 'PASS'
            except Exception as e:
                result.update(status='FAIL',error=str(e)); failed=True
            finally:
                save()
        report['status']='FAIL' if failed else 'BUILD_PASS_NOT_RUN' if a.build_only else 'FUNCTIONAL_PASS' if a.functional_only else 'PASS'
        save(); print(report['status']+': '+str(out),flush=True)
        if failed: raise SystemExit(1)
    except Exception as e:
        report.update(status='FAIL',error=str(e)); save(); raise

if __name__=='__main__':
    main()
