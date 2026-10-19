#!/usr/bin/env python3
"""Run original or supplemental QLI drivers with guarded memory observation."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import qli_original_fixture as fixture

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
from native_bstart import normalize_disassembly

sys.path.insert(0, str(fixture.SUITE))
from compare_memory import load_symbols
from check_disassembly import function_body_exact, matching_function_bodies, ir_function_body_exact
from run_qli_element import OPTIONS, has_success_finisher


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reachable_native_functions(text, root='main'):
    text = normalize_disassembly(text)
    pending, reachable = [root], {}
    while pending:
        symbol = pending.pop()
        if symbol in reachable:
            continue
        body = function_body_exact(text, symbol)
        reachable[symbol] = body
        pending.extend(re.findall(r'BSTART\.STD\s+DIRECT,[^\n]*<([^>]+)>', body))
    return reachable


def check_score_native(text, kind):
    fragment = 'qli_ptoI' if kind == 'static' else 'qli_pto_dynamicI'
    reachable = reachable_native_functions(text)
    matches = [(symbol, body) for symbol, body in reachable.items()
               if fragment in symbol]
    if matches:
        assert len(matches) == 1, 'ambiguous original QLI specialization'
        symbol, body = matches[0]
    else:
        body = '\n'.join(reachable.values())
    assert re.search(r'BSTART(?:\.CUBE\s+|\.)TMATMUL\b', body), 'missing QK matmul'
    for operation in ('TCMP', 'TSEL', 'TCOLSUM', 'TMUL', 'TADD'):
        assert re.search(r'BSTART(?:\.TEPL\s+|\.VEC\s+|\.SFU\s+|\.)' +
                         operation + r',\s*FP32\b', body), \
            'missing original FP32 score operation: ' + operation


def check_pe0_domain_native(text):
    """Check every Tile binder reachable from the explicitly PE0-only stage."""
    called = reachable_native_functions(text)
    roots = [symbol for symbol in called if 'run_qli_pe0' in symbol]
    assert len(roots) == 1, 'missing or ambiguous PE0 domain boundary'
    reachable = reachable_native_functions(text, roots[0])
    binders, atomics = 0, 0
    for symbol, body in reachable.items():
        for line in body.splitlines():
            if re.search(r'\bB\.IOT\b', line):
                assert re.search(r'\bmask=1000\b', line), \
                    'Tile binder escaped PE0 domain in ' + symbol + ': ' + line
                binders += 1
        for bundle in re.split(r'(?=\bBSTART\.)', body):
            if re.match(r'BSTART\.TLSU\s+MGATHER\.ADD,', bundle):
                assert re.search(r'\bB\.IOR\b[^\n]*ExecMaskPresent', bundle), \
                    'element atomic lost its execution mask in ' + symbol
                atomics += 1
    assert binders, 'PE0 domain has no native Tile binders'
    assert atomics, 'PE0 domain has no masked element atomic'


def check_topk_native(text, ir, profile):
    kind, sq, skv, topk, _ = profile
    fragment = (f'qli_topk_radixILi{sq}ELi{skv}ELi{topk}E'
                if kind == 'static' else 'qli_topk_radix_dynamic')
    called = reachable_native_functions(text)
    matches = [(symbol, body) for symbol, body in called.items()
               if fragment in symbol]
    if matches:
        assert len(matches) == 1, 'ambiguous original TopK specialization'
        root, _ = matches[0]
        reachable = reachable_native_functions(text, root)
    else:
        reachable = called
    combined = '\n'.join(reachable.values())
    assert 'THISTOGRAM' not in combined, 'retired histogram instruction'
    for engine, operation in (('TLSU', 'MGATHER.ADD'), ('TEPL', 'TROWSUM'),
                              ('TEPL', 'TROWARGMAX'), ('TEPL', 'TCMP'), ('TEPL', 'TSEL')):
        assert re.search(r'BSTART\.' + engine + r'\s+' + re.escape(operation) +
                         r',\s*U32\b', combined), 'missing migrated TopK operation: ' + operation
    helpers = [symbol for symbol in reachable
               if 'PopN' in symbol and re.search(r'\.pto\.local(?:\.pto\.pe\d+\.\d+)?$', symbol)]
    assert helpers, 'no reachable outlined LocalTile selection helper'
    for symbol in helpers:
        body = reachable[symbol]
        first = re.search(r'BSTART\.(?:TLSU|TEPL)\s+(\w+)', body)
        assert first and first[1] == 'TROWARGMAX', 'selection helper reconstructed Tile in GM'
        assert not re.search(r'\b(?:load|store)\s+<\d+ x',
                             ir_function_body_exact(ir, symbol)), 'selection helper Tile payload spill'
        for operation in ('TCMP', 'TSEL'):
            assert re.search(r'BSTART\.TEPL\s+' + operation + r',\s*U32\b', body), \
                'selection clear did not lower to element operations'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('compiler-bin', 'sysroot', 'resource-dir', 'api-include', 'gfrun', 'out'):
        parser.add_argument('--' + option, type=Path, required=True)
    parser.add_argument('--profile-set', choices=('original', 'original-four-pe', 'upstream-four-pe'),
                        default='original')
    parser.add_argument('--case', type=int, choices=range(5), action='append')
    args = parser.parse_args()
    profile_sets = {
        'original': fixture.profiles(),
        'original-four-pe': tuple((*profile[:4], 4) for profile in fixture.profiles()
                                 if profile[0] == 'static'),
        'upstream-four-pe': fixture.supplemental_four_pe_profiles(),
    }
    profiles = profile_sets[args.profile_set]
    if args.case is not None and any(index >= len(profiles) for index in args.case):
        parser.error(f'--case must select one of 0..{len(profiles) - 1} for '
                     f'{args.profile_set}')
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve().parent
    bench = source.parents[3]
    tools = args.compiler_bin.resolve()
    files = [Path(__file__), Path(fixture.__file__), source / 'qli_check_opt.cpp',
             source / 'qli_check_opt_dynamic.cpp', source.parent / 'compile.all',
             bench / 'test/common/_start.s', fixture.SUITE / 'qli_topk_fixture.py',
             fixture.SUITE / 'qli_histogram_fixture.py', fixture.SUITE / 'compare_memory.py',
             fixture.SUITE / 'check_disassembly.py', fixture.SUITE / 'run_qli_element.py',
             bench / 'verification/gfrun_dump.py', bench.parent / 'native_bstart.py']
    files += list((bench / 'kernels/solution/qli').glob('*.hpp'))
    files += list(args.api_include.rglob('*.hpp'))
    files += list((args.resource_dir / 'include').rglob('*.hpp'))
    files += list((bench / 'test/common').rglob('*.h'))
    files += [tools / name for name in ('clang++', 'llc', 'ld.lld', 'llvm-objdump')]
    files.append(args.gfrun.resolve())
    before = {str(path.resolve()): digest(path) for path in files}
    (out / 'provenance.json').write_text(json.dumps(dict(
        hashes=before, model='gfrun', spec_pin=fixture.SPEC_PIN,
        target_triple='linx64v5-unknown-linux-musl', sysroot=str(args.sysroot.resolve()),
        scope='original driver full-memory diagnostics; clean integrated freeze required for promotion',
        compile_timeout_seconds=600, gfrun_timeout_seconds=120,
        profile_set=args.profile_set, profiles=profiles), indent=2) + '\n')
    commands, reports = [], []

    def run(command, log, timeout=600):
        commands.append(list(map(str, command)))
        (out / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
        with log.open('w') as stream:
            subprocess.run(commands[-1], stdout=stream, stderr=subprocess.STDOUT,
                           check=True, timeout=timeout)

    target = ['--target=linx64v5-unknown-linux-musl', '--sysroot=' + str(args.sysroot),
              '-resource-dir=' + str(args.resource_dir)]
    for index in (range(len(profiles)) if args.case is None else args.case):
        profile = profiles[index]
        kind, sq, skv, topk, npe = profile
        case = out / f'case_{index}'
        case.mkdir()
        fixture.generate(case / 'fixture', profile)
        report = dict(index=index, profile=profile, status='FAIL')
        try:
            flags = ['-c', '-mlxbc', '-fenable-matrix', '-O2', '-std=c++20',
                     '-fno-vectorize', '-fno-slp-vectorize', '-Werror=unknown-pragmas',
                     '-D__linx', '-DENABLE_TENSOR_INSTR', '-DQLI_INPUT_DATA',
                     '-DQLI_DTYPE_FP8', '-DBatch=1', '-DTg=64', '-DTm=16', '-DTk=32',
                     '-DTsq=' + str(sq), '-DTskv=' + str(skv), '-DTtopk=' + str(topk),
                     '-DTNPE=' + str(npe), '-I' + str(args.api_include),
                     '-I' + str(bench / 'kernels'), '-I' + str(bench / 'test/common'),
                     '-I' + str(bench / 'test/common/src')]
            for option in OPTIONS:
                flags += ['-mllvm', '-' + option]
            cpp = source / ('qli_check_opt.cpp' if kind == 'static' else 'qli_check_opt_dynamic.cpp')
            run([tools / 'clang++', *target, *flags, '-S', '-emit-llvm', cpp,
                 '-o', case / 'kernel.ll'], case / 'frontend.log')
            run([tools / 'llc', '-mtriple=linx64v5-unknown-linux-musl', '-O2', '-filetype=obj',
                 *['-' + option for option in OPTIONS], case / 'kernel.ll',
                 '-o', case / 'kernel.o'], case / 'backend.log')
            run([tools / 'clang++', *target, '-nostartfiles', '-fuse-ld=lld',
                 bench / 'test/common/_start.s', case / 'fixture/data.s', case / 'kernel.o',
                 '-o', case / 'kernel.elf'], case / 'link.log')
            run([tools / 'llvm-objdump', '-d', '--disassembler-options=no-tile-macros',
                 case / 'kernel.elf'], case / 'kernel.diss')
            check_score_native((case / 'kernel.diss').read_text(), kind)
            check_topk_native((case / 'kernel.diss').read_text(),
                              (case / 'kernel.ll').read_text(), profile)
            if kind == 'static':
                check_pe0_domain_native((case / 'kernel.diss').read_text())
            run([tools / 'llvm-objdump', '-t', case / 'kernel.elf'], case / 'symbols.txt')
            symbols = load_symbols(case / 'symbols.txt')
            wants = {p.name.removesuffix('.golden.bin'): p.read_bytes()
                     for p in (case / 'fixture').glob('*.golden.bin')}
            memory_args = []
            for name, raw in wants.items():
                address, size = symbols[name]
                assert size == len(raw), name
                memory_args += ['--dump-memory', f'{address:#x}:{size:#x}:{case}/{name}.dump.bin']
            for name in json.loads((case / 'fixture/manifest.json').read_text())['buffers']:
                prev, data, after = [symbols[name + suffix] for suffix in ('_before', '', '_after')]
                assert prev[0] + prev[1] == data[0] and data[0] + data[1] == after[0], name
            artifacts = {str(p): digest(p) for p in (case / 'fixture').iterdir() if p.is_file()}
            artifacts[str(case / 'kernel.elf')] = digest(case / 'kernel.elf')
            run([args.gfrun, '--pto059-execution-mask-dev', '-s', f'softcore.multiThreadNum={npe}',
                 '--dump-force', *memory_args, '-f', case / 'kernel.elf'], case / 'gfrun.log', 120)
            assert has_success_finisher((case / 'gfrun.log').read_text())
            for name, expected in wants.items():
                actual = (case / (name + '.dump.bin')).read_bytes()
                if name == 'qli_workspace':
                    fixture.compare_workspace(actual, expected, profile)
                else:
                    assert actual == expected, name
            assert all(digest(path) == value for path, value in artifacts.items()), 'artifact mutation'
            (case / 'artifacts.json').write_text(json.dumps(artifacts, indent=2) + '\n')
            report.update(status='PASS', segments=len(wants), bytes=sum(map(len, wants.values())),
                          elf_sha256=digest(case / 'kernel.elf'))
        except subprocess.TimeoutExpired as error:
            report.update(status='TIMEOUT', error=str(error))
        except (subprocess.CalledProcessError, AssertionError, ValueError, RuntimeError, KeyError) as error:
            report['error'] = str(error)
        reports.append(report)
        (case / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
        (out / 'summary.json').write_text(json.dumps(dict(
            cases=reports, passed=sum(row['status'] == 'PASS' for row in reports),
            total=len(reports), profile_set=args.profile_set, original_total=5,
            supplemental_four_pe_total=len(fixture.supplemental_four_pe_profiles())),
            indent=2) + '\n')
        print(f'{args.profile_set} QLI case {index}: {report["status"]}', flush=True)
    assert all(digest(path) == value for path, value in before.items()), 'source/tool mutation invalidates run'
    return 0 if all(row['status'] == 'PASS' for row in reports) else 1


if __name__ == '__main__':
    sys.exit(main())
