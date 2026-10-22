#!/usr/bin/env python3
"""Compile and compare QLI element diagnostics on gfrun (not whole-app acceptance)."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time

from compare_memory import load_symbols
from check_disassembly import function_body, function_body_exact, ir_function_body_exact
import qli_key_fixture
import qli_histogram_fixture
import qli_topk_fixture
import qli_norm_fixture
import qli_popn_fixture

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "verification"))
from gfrun_dump import has_success_finisher
sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
from native_bstart import normalize_disassembly

OPTIONS = ('enable-all-vector-as-tilereg=true', 'linxv5-enable-HL-Inst-Opt=true',
           'linxv5-enable-dim-opt=true', 'linxv5-enable-ldst-bridge=false',
           'linxv5-enable-continuous-mem-opt=true', 'linxv5-enable-tile-clock-hand=false',
           'linxv5-enable-simt-clock-hand=true', 'enable-misched=false',
           'linxv5-enable-generic-element=true')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_native(text, kind, profiles):
    text = normalize_disassembly(text)
    for capacity, valid in sorted({(p[0], p[1]) for p in profiles}):
        if valid == 0:
            continue
        name = f'{"make_key" if kind == "key" else "histogram"}ILj{capacity}ELj{valid}E'
        body = function_body(text, name)
        assert 'MGATHER, U32' in body, name
        assert 'MSCATTER, U32' in body, name
        assert 0 <= body.rfind('B.ASSEMBLE') < body.index('MSCATTER, U32'), name
        if valid < capacity:
            for block in re.split(r'BSTART\.', body)[1:]:
                if block.startswith('TLSU\tMGATHER, U32'):
                    assert 'ExecMaskPresent' in block, name
        if kind == 'histogram':
            assert 'MGATHER.ADD, U32' in body, name
    assert 'THISTOGRAM' not in text


def check_norm_native(text, profiles, symbol='qli_norm_stage'):
    text = normalize_disassembly(text)
    for capacity, valid in sorted({case[:2] for case in profiles}):
        body = function_body(text, f'{symbol}ILj{capacity}ELj{valid}E')
        assert 'TCMP, U32' in body and 'TSEL, U32' in body
        assert 'B.SUBVIEW' not in body and 'B.ASSEMBLE' not in body
        for block in re.split(r'BSTART\.', body)[1:]:
            operation = re.match(r'TEPL\s+(TCMP|TSEL|TCI|TEXPANDS), U32', block)
            if not operation:
                continue
            dims = {int(index): int(value) for value, index in
                    re.findall(r'(?:C\.B\.DIMI\s+|B\.DIM\s+zero,\s*)(\d+),\s+->lb([012])', block)}
            assert dims.get(0, 1) == valid, (symbol, 'valid columns', dims)
            assert dims.get(1, 1) == 1, (symbol, 'valid rows', dims)
            assert dims.get(2, 1) == capacity, (symbol, 'physical columns', dims)
            if operation[1] == 'TCMP':
                # The pinned generic TMOV descriptor also requires its U8 logical extent
                # to fit the allocation, even though predicate payloads are bit-packed.
                predicate_bytes = max(128, capacity)
                label = f'{predicate_bytes}B' if predicate_bytes < 1024 else f'{predicate_bytes // 1024}KB'
                assert f'<{label}>' in block, (symbol, 'copyable predicate capacity')
            elif operation[1] == 'TCI':
                assert 'B.DATR' not in block, (symbol, 'TCI attributes')
            elif operation[1] == 'TEXPANDS':
                assert not re.search(r'B\.IOR\s+\[\],\s*\[\]', block), (symbol, 'missing scalar operand')

def check_popn_native(disassembly, ir, profiles):
    """Pair each fixture's real call and result with its marked outlined helper."""
    disassembly = normalize_disassembly(disassembly)
    caller_ir = ir_function_body_exact(ir, 'main')
    caller_native = function_body_exact(disassembly, 'main')
    assert not re.search(r'\b(?:load|store)\s+<\d+ x', caller_ir), 'Tile payload GM spill'
    definitions = list(re.finditer(
        r'^define internal inreg <(\d+) x i32> @([^\s(]+)\(.*?#(\d+) \{', ir, re.M))
    helpers = {}
    for definition in definitions:
        carrier, symbol, attributes = definition.groups()
        if 'RadixPopN' not in symbol or not symbol.endswith('.pto.local'):
            continue
        attr = re.search(r'^attributes #' + attributes + r' = \{([^\n]+)\}', ir, re.M)
        assert attr is not None
        marker = re.search(r'"linx-local-tile-abi"="0:(\d+):(\d+)"', attr[1])
        assert marker is not None, symbol
        shape = tuple(map(int, marker.groups()))
        assert int(carrier) == shape[0] and shape not in helpers, symbol
        helper_ir = ir_function_body_exact(ir, symbol)
        assert not re.search(r'\b(?:load|store)\s+<\d+ x', helper_ir), symbol
        body = function_body_exact(disassembly, symbol)
        for operation in ('TROWARGMAX', 'TCMP', 'TSEL'):
            assert re.search(r'BSTART\.TEPL\s+' + operation + r',\s+U32\b', body), (symbol, operation)
        assert 'NORM' in body
        first_tile = re.search(r'BSTART\.(?:TLSU|TEPL)\s+(\w+)', body)
        assert first_tile and first_tile[1] == 'TROWARGMAX', (symbol, 'allocation-free import')
        assert 'B.SUBVIEW' not in body and 'B.ASSEMBLE' not in body
        # No incoming Tile payload is reconstructed through GM.
        assert not re.search(r'BSTART\.TLSU\s+TLOAD\b', body), symbol
        returns = re.split(r'C?\.?BSTART\.STD\s+RET\b', body)
        assert len(returns) == 2, (symbol, 'one native return')
        export = re.split(r'BSTART\.TLSU\s+TMOV,\s+U32\b', returns[0])[-1]
        assert len(re.findall(r'BSTART\.TLSU\s+TMOV,\s+U32\b', returns[0])) >= 1
        assert re.search(r'B\.IOT\s+t#1(?:\.reuse)?,.*last,\s+->t<', export), (symbol, 'return export')
        assert not re.search(r'BSTART\.(?:TLSU|TEPL)', export), (symbol, 'last Tile producer')
        helpers[shape] = symbol
    expected_shapes = {case[:2] for case in profiles}
    assert set(helpers) == expected_shapes, (set(helpers), expected_shapes)
    for capacity, valid, ties, count in profiles:
        stem = f'qli_popn_{capacity}_{valid}_{int(ties)}_{count}'
        stores = [line for line in caller_ir.splitlines()
                  if re.search(r'asm sideeffect "BSTART\.(?:TSTORE\s|TLSU TSTORE,)', line)
                  and '@' + stem + '_remaining,' in line]
        assert len(stores) == 1, stem
        assert '<' + str(capacity) + ' x i32>' in stores[0], stem
        if count == 0:
            assert '@' + stem + '_indices' not in caller_ir, (stem, 'zero-count write')
            continue
        symbol = helpers[(capacity, valid)]
        calls = [line for line in caller_ir.splitlines()
                 if 'call inreg' in line and '@' + symbol + '(' in line and '@' + stem + '_indices,' in line]
        assert len(calls) == 1, stem
        returned = re.match(r'\s*(%[\w.]+) = call inreg <' + str(capacity) + r' x i32>', calls[0])
        assert returned is not None, stem
        assert '<' + str(capacity) + ' x i32> ' + returned[1] + ',' in stores[0], (stem, 'copyback SSA')
        assert re.search(r'i32 noundef signext ' + str(count) + r'\)$', calls[0]), (stem, 'count ABI')
        assert 'i32 noundef signext 4096,' in calls[0], (stem, 'chunk ABI')
    for shape, symbol in helpers.items():
        want = sum(case[:2] == shape and case[3] > 0 for case in profiles)
        transfers = re.findall(r'C?\.?BSTART\.STD\s+DIRECT,[^\n]*<' + re.escape(symbol) + r'>', caller_native)
        assert len(transfers) == want, (symbol, 'native direct calls')
        for prefix in re.split(r'C?\.?BSTART\.STD\s+DIRECT,[^\n]*<' + re.escape(symbol) + r'>', caller_native)[:-1]:
            export = re.split(r'BSTART\.TLSU\s+TMOV,\s+U32\b', prefix)[-1]
            assert re.search(r'B\.IOT\s+t#1(?:\.reuse)?,.*last,\s+->t<', export), (symbol, 'call export')
            assert not re.search(r'BSTART\.(?:TLSU|TEPL)', export), (symbol, 'immediate Tile export')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=('key', 'histogram', 'topk', 'norm', 'popn'), required=True)
    parser.add_argument('--norm-profile', choices=('all', '128', '2048', '128-tail'), default='all')
    for option in ('compiler-bin', 'sysroot', 'resource-dir', 'api-include', 'gfrun', 'out'):
        parser.add_argument('--' + option, type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    suite = Path(__file__).resolve().parent
    bench = suite.parents[3]
    fixture = {'key': qli_key_fixture, 'histogram': qli_histogram_fixture,
               'topk': qli_topk_fixture, 'norm': qli_norm_fixture, 'popn': qli_popn_fixture}[args.kind]
    cases = list(fixture.profiles())
    if args.kind == 'norm':
        if args.norm_profile != 'all':
            wanted = {'128': (128, 128), '2048': (2048, 2048), '128-tail': (128, 32)}[args.norm_profile]
            cases = [case for case in cases if case[:2] == wanted]
        fixture.generate(out / 'fixture', cases)
    else:
        if args.norm_profile != 'all':
            parser.error('--norm-profile requires --kind norm')
        fixture.generate(out / 'fixture')
    tools = args.compiler_bin.resolve()
    binaries = [tools / name for name in ('clang++', 'llc', 'ld.lld', 'llvm-objdump')]
    binaries.append(args.gfrun.resolve())
    before = {str(path): sha(path) for path in binaries}
    inputs = [suite / f'qli_{args.kind}.cpp', Path(fixture.__file__),
              bench / 'kernels/solution/qli/qli_element.hpp', Path(__file__).resolve(),
              suite / 'compare_memory.py', suite / 'check_disassembly.py',
              bench / 'verification/gfrun_dump.py', bench.parent / 'native_bstart.py']
    if args.kind in ('topk', 'popn'):
        inputs += [bench / 'kernels/solution/qli/qli_pto_opt_simple.hpp',
                   Path(qli_histogram_fixture.__file__)]
    before.update({str(path): sha(path) for path in inputs})
    target = ['--target=linx64v5-unknown-linux-musl', '--sysroot=' + str(args.sysroot),
              '-resource-dir=' + str(args.resource_dir)]
    commands = []

    def run(command, output):
        commands.append(list(map(str, command)))
        (out / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
        with output.open('w') as log:
            subprocess.run(commands[-1], stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=120)

    compile_flags = ['-c', '-mlxbc', '-fenable-matrix', '-O2', '-std=c++20',
                     '-fno-vectorize', '-fno-slp-vectorize', '-Werror=unknown-pragmas',
                     '-D__linx', '-DENABLE_TENSOR_INSTR', '-I' + str(args.api_include),
                     '-I' + str(bench / 'kernels'), '-I' + str(bench / 'test/common/src'),
                     '-I' + str(out / 'fixture')]
    for option in OPTIONS:
        compile_flags += ['-mllvm', '-' + option]
    run([tools / 'clang++', *target, *compile_flags, '-S', '-emit-llvm', inputs[0],
         '-o', out / 'kernel.ll'], out / 'frontend.log')
    run([tools / 'llc', '-mtriple=linx64v5-unknown-linux-musl', '-O2', '-filetype=obj',
         *['-' + option for option in OPTIONS], out / 'kernel.ll', '-o', out / 'kernel.o'],
        out / 'backend.log')
    run([tools / 'clang++', *target, '-nostartfiles', '-fuse-ld=lld',
         bench / 'test/common/_start.s', out / 'kernel.o', '-o', out / 'kernel.elf'],
        out / 'link.log')
    run([tools / 'llvm-objdump', '-d', '--disassembler-options=no-tile-macros',
         out / 'kernel.elf'], out / 'kernel.diss')
    disassembly = (out / 'kernel.diss').read_text()
    if args.kind == 'topk':
        for kind in ('key', 'histogram'):
            check_native(disassembly, kind, [(128, 32), (128, 128), (2048, 2048)])
        assert 'TROWSUM' in disassembly and 'TROWARGMAX' in disassembly
    elif args.kind == 'norm':
        check_norm_native(disassembly, cases)
    elif args.kind == 'popn':
        check_popn_native(disassembly, (out / 'kernel.ll').read_text(), cases)
    else:
        check_native(disassembly, args.kind, cases)
    run([tools / 'llvm-objdump', '-t', out / 'kernel.elf'], out / 'symbols.txt')
    symbols = load_symbols(out / 'symbols.txt')
    expected = {p.stem: p.read_bytes() for p in (out / 'fixture/golden').glob('*.bin')}
    for name, raw in expected.items():
        assert symbols[name][1] == len(raw), name
    for stem in json.loads((out / 'fixture/buffers.json').read_text()):
        before_guard, body, after_guard = [symbols[stem + tail]
                                          for tail in ('_before', '', '_after')]
        assert before_guard[0] + before_guard[1] == body[0], stem
        assert body[0] + body[1] == after_guard[0], stem
    start = min(symbols[name][0] for name in expected)
    end = max(symbols[name][0] + len(raw) for name, raw in expected.items())
    started = time.monotonic()
    exit_code, timed_out = None, False
    try:
        run([args.gfrun, '--pto059-execution-mask-dev', '-s', 'softcore.multiThreadNum=1',
             '--dump-force', '--dump-memory', f'{start:#x}:{end-start:#x}:{out}/memory.bin',
             '-f', out / 'kernel.elf'], out / 'gfrun.log')
        exit_code = 0
    except subprocess.TimeoutExpired:
        exit_code, timed_out = 124, True
        raise
    except subprocess.CalledProcessError as error:
        exit_code = error.returncode
        raise
    finally:
        (out / 'gfrun-run.json').write_text(json.dumps({
            'elapsed_seconds': time.monotonic() - started,
            'timeout_seconds': 120, 'timed_out': timed_out,
            'exit_code': exit_code}, indent=2) + '\n')
    assert has_success_finisher((out / 'gfrun.log').read_text())
    memory = (out / 'memory.bin').read_bytes()
    assert len(memory) == end - start
    storage_profiles = (json.loads((out / 'fixture/storage_profiles.json').read_text())
                        if args.kind == 'topk' else {})
    selection_profiles = (json.loads((out / 'fixture/selection_profiles.json').read_text())
                          if args.kind == 'popn' else {})
    selection_results = {stem + suffix for stem in selection_profiles
                         for suffix in ('_indices', '_remaining')}
    for name, raw in expected.items():
        offset = symbols[name][0] - start
        actual = memory[offset:offset + len(raw)]
        if name in selection_results:
            continue
        if name in storage_profiles:
            qli_topk_fixture.compare_storage(actual, raw, *storage_profiles[name])
            continue
        if actual != raw:
            first = next(i for i, (a, b) in enumerate(zip(actual, raw)) if a != b)
            raise AssertionError(f'{name}: first byte mismatch at {first}')
    for stem, profile in selection_profiles.items():
        def observed(suffix):
            name = stem + suffix
            offset, length = symbols[name]
            return memory[offset - start:offset - start + length]
        qli_popn_fixture.compare_selection(observed('_indices'), observed('_remaining'), *profile)
    assert all(sha(Path(path)) == digest for path, digest in before.items()), 'inputs changed during run'
    report = {'kind': args.kind, 'profiles': len(cases), 'segments': len(expected),
              'bytes': sum(map(len, expected.values())), 'pass': True,
              'scope': 'diagnostic only; not frozen whole-application acceptance',
              'norm_profile': args.norm_profile if args.kind == 'norm' else None,
              'comparison': ('unordered TopK indices; exact inputs/scratch/guards'
                             if args.kind == 'topk' else ('unordered unique PopN indices; exact inputs/results/guards'
                                                        if args.kind == 'popn' else 'exact bytes')),
              'input_sha256': before, 'elf_sha256': sha(out / 'kernel.elf')}
    (out / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'{args.kind}: {len(cases)} profiles, {len(expected)} checked segments PASS')


if __name__ == '__main__':
    main()
