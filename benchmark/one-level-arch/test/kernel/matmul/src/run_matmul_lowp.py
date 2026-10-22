#!/usr/bin/env python3
"""Build original lowp matmul drivers and compare full gfrun memory independently."""
import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import matmul_lowp_fixture as fixture

SUITE = Path(__file__).resolve().parents[2]/'element_wise/tile_element_suite'
sys.path.insert(0, str(SUITE))
from compare_memory import load_symbols
from check_disassembly import (function_body_exact, matching_function_bodies,
                               normalize_disassembly)
from run_qli_element import OPTIONS, has_success_finisher


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare_output(actual, expected):
    if len(actual) != len(expected) or len(actual) % 4:
        raise ValueError('wrong FP32 result length')
    got, want = [np.frombuffer(raw, dtype='<f4') for raw in (actual, expected)]
    if not np.all(np.isfinite(got)) or not np.array_equal(got, want):
        bad = np.flatnonzero(~np.isfinite(got) | (got != want))
        index = int(bad[0])
        raise ValueError(f'FP32[{index}]: got {got[index]} expected {want[index]}')


def check_matmul_native(text, profile):
    text = normalize_disassembly(text)
    kind, version, m, n, k, tm, tn, tk = profile
    opcode = 'TMATMULMX' if kind == 'HIF4_HIF4' else 'TMATMUL'
    main_body = function_body_exact(text, 'main')
    if kind == 'HIF4_HIF4':
        fragment = 'matmul_hif4x2_mx' + ''.join(
            ('ILi' if index == 0 else 'ELi') + str(value)
            for index, value in enumerate((m,n,k,tm,tn,tk)))
        fragment += 'ELb' + ('1' if version == 'MX_NOGATHER_REUSEA' else '0') + 'E'
    else:
        fragment = 'matmul_mpI'
    matches = matching_function_bodies(text, fragment)
    if matches:
        assert len(matches) == 1, ('ambiguous original specialization', fragment)
        symbol, body = matches[0]
        if kind == 'A16W4':
            shape = 'Li' + 'ELi'.join(map(str, (m,n,k,tm,tn,tk,2))) + 'E'
            assert shape in symbol, ('wrong original dimensions', symbol)
        assert re.search(r'BSTART\.STD\s+DIRECT,[^\n]*<' + re.escape(symbol) + '>', main_body), symbol
    else:
        body = main_body
    assert re.search(r'BSTART(?:\.CUBE\s+|\.)' + opcode + r'(?:\.ACC)?\b', body), (fragment, opcode)


def dump_windows(start, end, directory):
    if end <= start:
        raise ValueError('empty memory observation range')
    limit = 4 * 1024 * 1024
    return [(address, min(limit,end-address), directory/f'memory_{index:03d}.bin')
            for index,address in enumerate(range(start,end,limit))]


def read_dump_windows(windows):
    chunks = []
    expected_address = windows[0][0]
    for address,size,path in windows:
        if address != expected_address:
            raise ValueError('noncontiguous memory observation windows')
        raw = path.read_bytes()
        if len(raw) != size:
            raise ValueError(f'incomplete or oversized memory dump {path}')
        chunks.append(raw)
        expected_address += size
    return b''.join(chunks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('compiler-bin','sysroot','resource-dir','api-include','gfrun','out'):
        parser.add_argument('--'+option, type=Path, required=True)
    parser.add_argument('--case', type=int, choices=range(7), action='append',
                        help='original configuration index; default all seven')
    parser.add_argument('--compile-timeout', type=int, default=600,
                        help='bounded compiler/linker stage timeout in seconds (default 600)')
    args = parser.parse_args()
    if args.compile_timeout <= 0:
        parser.error('--compile-timeout must be positive')
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve().parent
    bench = source.parents[3]
    tools = args.compiler_bin.resolve()
    profiles = fixture.profiles()
    if fixture.active_commands(source.parent/'element_original_profiles.json') != profiles:
        parser.error('active original command inventory changed')
    files = [Path(__file__).resolve(), Path(fixture.__file__), source.parent/'element_original_profiles.json',
             source/'HiF4_HiF4.cpp', source/'A16W4.cpp', bench/'test/common/_start.s',
             SUITE/'compare_memory.py', SUITE/'check_disassembly.py', SUITE/'run_qli_element.py',
             bench/'verification/gfrun_dump.py', bench.parent/'native_bstart.py']
    files += list((bench/'kernels').rglob('*.hpp'))
    files += list(args.api_include.rglob('*.hpp'))
    files += list((args.resource_dir/'include').rglob('*.hpp'))
    files += list((bench/'test/common').rglob('*.h'))
    files += [tools/name for name in ('clang++','llc','ld.lld','llvm-objdump')]
    files += [args.gfrun.resolve()]
    before = {str(path.resolve()): sha(path) for path in files}
    (out/'provenance.json').write_text(json.dumps({'hashes': before, 'spec_pin': fixture.SPEC_PIN,
        'model': 'gfrun', 'pe_count': 1, 'compile_timeout_seconds': args.compile_timeout,
        'gfrun_timeout_seconds': 120}, indent=2)+'\n')
    target = ['--target=linx64v5-unknown-linux-musl','--sysroot='+str(args.sysroot),
              '-resource-dir='+str(args.resource_dir)]
    commands, reports = [], []

    def run(command, log, timeout=None):
        command = list(map(str, command))
        commands.append(command)
        (out/'commands.json').write_text(json.dumps(commands, indent=2)+'\n')
        with log.open('w') as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True,
                           timeout=args.compile_timeout if timeout is None else timeout)

    for index in (range(7) if args.case is None else args.case):
        profile = profiles[index]
        kind, version, m, n, k, tm, tn, tk = profile
        case = out/f'case_{index}'
        case.mkdir()
        fixture.generate(case/'fixture', profile)
        report = {'index': index, 'profile': profile, 'status': 'FAIL'}
        try:
            flags = ['-c','-mlxbc','-fenable-matrix','-O2','-std=c++20',
                '-fno-vectorize','-fno-slp-vectorize','-Werror=unknown-pragmas',
                '-D__linx','-DENABLE_TENSOR_INSTR','-DMATMUL_INPUT_DATA',
                '-DglobM='+str(m),'-DglobN='+str(n),'-DglobK='+str(k),
                '-DtilM='+str(tm),'-DtilN='+str(tn),'-DtilK='+str(tk),'-DBatch=1',
                '-I'+str(args.api_include),'-I'+str(bench/'kernels'),
                '-I'+str(bench/'test/common'),'-I'+str(bench/'test/common/src'),'-I'+str(case/'fixture')]
            if version:
                flags += ['-D'+version]
            for option in OPTIONS:
                flags += ['-mllvm','-'+option]
            cpp = source/('HiF4_HiF4.cpp' if kind == 'HIF4_HIF4' else 'A16W4.cpp')
            run([tools/'clang++',*target,*flags,'-S','-emit-llvm',cpp,'-o',case/'kernel.ll'], case/'frontend.log')
            run([tools/'llc','-mtriple=linx64v5-unknown-linux-musl','-O2','-filetype=obj',
                 *['-'+opt for opt in OPTIONS],case/'kernel.ll','-o',case/'kernel.o'],case/'backend.log')
            run([tools/'clang++',*target,'-nostartfiles','-fuse-ld=lld',
                 bench/'test/common/_start.s',case/'kernel.o','-o',case/'kernel.elf'],case/'link.log')
            run([tools/'llvm-objdump','-d','--disassembler-options=no-tile-macros',case/'kernel.elf'],case/'kernel.diss')
            text = (case/'kernel.diss').read_text()
            check_matmul_native(text, profile)
            run([tools/'llvm-objdump','-t',case/'kernel.elf'],case/'symbols.txt')
            symbols = load_symbols(case/'symbols.txt')
            wants = {p.stem: p.read_bytes() for p in (case/'fixture/golden').glob('*.bin')}
            for name, raw in wants.items():
                assert symbols[name][1] == len(raw), name
            for name in json.loads((case/'fixture/buffers.json').read_text()):
                prev, data, after = [symbols[name+tail] for tail in ('_before','','_after')]
                assert prev[0]+prev[1] == data[0] and data[0]+data[1] == after[0], name
            start = min(symbols[name][0] for name in wants)
            end = max(symbols[name][0]+len(raw) for name, raw in wants.items())
            artifacts = {str(path): sha(path) for path in (case/'fixture').rglob('*') if path.is_file()}
            artifacts[str(case/'kernel.elf')] = sha(case/'kernel.elf')
            (case/'artifacts.json').write_text(json.dumps(artifacts, indent=2)+'\n')
            windows = dump_windows(start,end,case)
            memory_args = []
            for address,size,path in windows:
                memory_args += ['--dump-memory',f'{address:#x}:{size:#x}:{path}']
            run([args.gfrun,'--pto059-execution-mask-dev','-s','softcore.multiThreadNum=1',
                '--dump-force',*memory_args,'-f',case/'kernel.elf'],case/'gfrun.log', timeout=120)
            assert has_success_finisher((case/'gfrun.log').read_text())
            memory = read_dump_windows(windows)
            assert len(memory) == end-start
            for name, raw in wants.items():
                offset = symbols[name][0]-start
                actual = memory[offset:offset+len(raw)]
                if name == 'matmul_dst':
                    count=m*n*4
                    compare_output(actual[:count], raw[:count])
                    assert actual[count:]==raw[count:], 'unchanged output capacity tail'
                else:
                    assert actual == raw, name
            assert all(sha(path) == value for path, value in artifacts.items()), 'artifact mutation'
            report.update(status='PASS', segments=len(wants), bytes=sum(map(len,wants.values())),
                          elf_sha256=sha(case/'kernel.elf'))
        except subprocess.TimeoutExpired as error:
            report.update(status='TIMEOUT', error=str(error))
        except (subprocess.CalledProcessError, AssertionError, ValueError, KeyError) as error:
            report['error'] = str(error)
        reports.append(report)
        (case/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
        (out/'summary.json').write_text(json.dumps({'cases':reports, 'pass':sum(
            r['status']=='PASS' for r in reports), 'total':len(reports), 'original_total':7}, indent=2)+'\n')
        print(f'case {index} {kind} {version}: {report["status"]}', flush=True)
    assert all(sha(path)==value for path,value in before.items()), 'source/tool mutation invalidates run'
    return 0 if all(r['status']=='PASS' for r in reports) else 1


if __name__ == '__main__':
    sys.exit(main())
