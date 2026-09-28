#!/usr/bin/env python3
"""Build network-inspired cases, validate host goldens, and measure four-PE gfsim."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time

import numpy as np

COMMON = Path(__file__).resolve().parent
ARCH = COMMON.parents[2]
REPO = ARCH.parents[1]


def command(argv, cwd, log, timeout=300):
    with log.open('w') as stream:
        stream.write(shlex.join(map(str, argv)) + '\n')
        stream.flush()
        try:
            result = subprocess.run(list(map(str, argv)), cwd=cwd, stdout=stream,
                                    stderr=subprocess.STDOUT, timeout=timeout)
            return result.returncode
        except subprocess.TimeoutExpired:
            stream.write('\nTIMEOUT\n')
            return 124


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def revision(path):
    return subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'],
                                   text=True).strip()


def tree_hash(root):
    if not root.is_dir():
        raise FileNotFoundError(root)
    entries = [f'{p.relative_to(root)} {sha256(p)}\n'
               for p in sorted(root.rglob('*')) if p.is_file()]
    return hashlib.sha256(''.join(entries).encode()).hexdigest()


def fixtures(case, folder):
    """Golden uses NumPy take / explicit scalar-coordinate address maps."""
    dtype = np.dtype({'fp16': '<f2', 'fp32': '<f4'}[case['dtype']])
    size = dtype.itemsize
    tile = case['tile']
    gather = case['operator'] == 'gather_v2'
    prefix = case.get('output_offset_bytes', 128) // size
    defs = dict(DATA_TYPE='__half' if size == 2 else 'float',
                NETWORK_GATHER=int(gather), TILE_ELEMENTS=tile,
                OUTPUT_PREFIX=prefix)
    index = np.zeros(tile, dtype='<u4')
    if gather:
        ishape, oshape, axis = case['input_shape'], case['output_shape'], case['axis']
        assert len(ishape) == len(oshape) and 0 <= axis < len(ishape)
        assert all(a == b for d, (a, b) in enumerate(zip(ishape, oshape)) if d != axis)
        ninput, noutput = math.prod(ishape), math.prod(oshape)
        i = np.arange(oshape[axis], dtype=np.uint64)
        if 'indices' in case:
            indices = np.asarray(case['indices'], dtype='<u4')
        elif case['index_pattern'] == 'hashed':
            # Fixed integer hash, deliberately permits duplicate token IDs.
            x = (i + 0x9e3779b9) & 0xffffffff
            x = ((x ^ (x >> 16)) * 0x85ebca6b) & 0xffffffff
            indices = ((x ^ (x >> 13)) % ishape[axis]).astype('<u4')
        else:
            step = 17
            while math.gcd(step, ishape[axis]) != 1:
                step += 2
            indices = ((i * step + 3) % ishape[axis]).astype('<u4')
        assert len(indices) == oshape[axis] and np.all(indices < ishape[axis])
        index = np.pad(indices, (0, tile))
        defs.update(RANK=len(ishape), AXIS=axis, INPUT_SHAPE=ishape,
                    OUTPUT_SHAPE=oshape, INPUT_ELEMENTS=ninput, OUTPUT_ELEMENTS=noutput)
        input_prefix, input_span, output_span = 0, ninput, noutput
    else:
        shape = case['shape']
        istride, ostride = case['input_stride'], case['output_stride']
        assert len(shape) == len(istride) == len(ostride)
        assert case.get('input_offset_bytes', 128) % size == 0
        assert case.get('output_offset_bytes', 128) % size == 0
        input_prefix = case.get('input_offset_bytes', 128) // size
        input_span = 1 + sum((n - 1) * s for n, s in zip(shape, istride))
        output_span = 1 + sum((n - 1) * s for n, s in zip(shape, ostride))
        noutput = math.prod(shape)
        defs.update(RANK=len(shape), SHAPE=shape, INPUT_STRIDE=istride,
                    OUTPUT_STRIDE=ostride, OUTPUT_ELEMENTS=noutput,
                    INPUT_OFFSET_BYTES=input_prefix * size, OUTPUT_OFFSET_BYTES=prefix * size)
    # Exactly representable FP16/FP32 values; hash breaks row-periodic fixtures.
    pos = np.arange(input_prefix + input_span + tile, dtype=np.uint64)
    x = ((pos ^ (pos >> 11)) * 2654435761) & 0xffffffff
    inp = (((x ^ (x >> 16)) % 2047).astype(np.int32) - 1023).astype(dtype)
    inp *= dtype.type(1 / 32)
    output = np.full(prefix + output_span + tile, -17, dtype=dtype)
    expected = output.copy()
    if gather:
        values = np.take(inp[:ninput].reshape(ishape), indices, axis=axis).ravel()
        expected[prefix:prefix + noutput] = values
    else:
        coord = np.indices(shape, dtype=np.int64).reshape(len(shape), -1)
        src = np.asarray(istride) @ coord
        dst = np.asarray(ostride) @ coord
        assert np.unique(dst).size == noutput, 'Output view must not overlap itself'
        expected[prefix + dst] = inp[input_prefix + src]
    for name, values in [('input', inp), ('index', index), ('output', output), ('golden', expected)]:
        (folder / (name + '.bin')).write_bytes(values.tobytes())
    case.update(payload_bytes=noutput * size, output_storage_bytes=expected.nbytes)
    lines = []
    for name, value in defs.items():
        value = ','.join(map(str, value)) if isinstance(value, list) else str(value)
        lines.append(f'#define {name} {value}\n')
    (folder / 'network_config.hpp').write_text(''.join(lines))
    # ELF-load initialization is outside both simulators' executed instruction stream.
    asm = ['.data\n']
    for name in ('input', 'index', 'output'):
        path = folder / (name + '.bin')
        asm += ['.balign 4096\n', f'.global network_{name}\n',
                f'.type network_{name},@object\n', f'network_{name}:\n',
                f'.incbin {json.dumps(str(path))}\n', f'.size network_{name}, .-network_{name}\n']
    (folder / 'data.s').write_text(''.join(asm))


def build(case, folder, compiler):
    fixtures(case, folder)
    flags = ['-mlxbc', '-fenable-matrix', '-O2', '-std=c++20',
             '-D__linx', '-DENABLE_TENSOR_INSTR']
    for setting in ('enable-all-vector-as-tilereg=true', 'linxv5-enable-HL-Inst-Opt=true',
                    'linxv5-enable-dim-opt=true', 'linxv5-enable-ldst-bridge=false',
                    'linxv5-enable-continuous-mem-opt=true', 'linxv5-enable-tile-clock-hand=false',
                    'linxv5-enable-simt-clock-hand=true', 'enable-misched=false',
                    'linxv5-enable-bfi-opt=false'):
        flags += ['-mllvm', '-' + setting]
    for path in (ARCH / 'include', ARCH / 'test/common', ARCH / 'test/common/src', ARCH / 'kernels', folder):
        flags += ['-I', str(path)]
    cmd = [compiler / 'clang++', *flags, '-c', COMMON / 'network_bench.cpp',
           '-o', folder / 'kernel.o']
    assert command(cmd, folder, folder / 'kernel_compile.log') == 0, 'Kernel compilation failed'
    cmd = [compiler / 'clang++', '-nostartfiles', COMMON / 'start.s',
           folder / 'data.s', folder / 'kernel.o', '-o', folder / 'kernel.elf']
    assert command(cmd, folder, folder / 'kernel_link.log') == 0, 'Kernel link failed'
    cmd = [compiler / 'llvm-objdump', '-d', folder / 'kernel.elf']
    assert command(cmd, folder, folder / 'kernel.diss') == 0


def validate(folder, compiler, model, timeout):
    symbols = subprocess.check_output([str(compiler / 'llvm-nm'), '-S', str(folder / 'kernel.elf')], text=True)
    symbol = next(line.split() for line in symbols.splitlines() if line.endswith(' network_output'))
    address, size = int(symbol[0], 16), int(symbol[1], 16)
    expected = (folder / 'golden.bin').read_bytes()
    assert size == len(expected)
    dump = folder / 'actual.bin'
    dump.unlink(missing_ok=True)
    cmd = [model / 'bin/gfrun', '-f', folder / 'kernel.elf', '-s', 'softcore.multiThreadNum=4',
           '--dump-memory', f'{address:#x}:{size}:{dump}']
    rc = command(cmd, model, folder / 'gfrun.log', timeout)
    log = (folder / 'gfrun.log').read_text(errors='replace')
    exact = dump.is_file() and dump.read_bytes() == expected
    passed = rc == 0 and bool(re.search(r'Reach the End of Benchmark! R2 = 0\s', log)) and exact
    return dict(gfrun_exit=rc, host_golden=exact, functional_pass=passed)


def simulate(folder, model, name, timeout):
    cmd = [model / 'bin/gfsim', '-f', folder / (name + '.elf'), '--conf', 'fourpe',
           '--seed', '1']
    rc = command(cmd, model, folder / (name + '_gfsim.log'), timeout)
    log = (folder / (name + '_gfsim.log')).read_text(errors='replace')
    counts = re.findall(r'^\s*Total Cycles\.+:\s*(\d+)', log, re.M)
    stopped = 'SuperScalar Report Stop' in log
    # Counts from aborted/deadlocked runs are diagnostic, never performance data.
    passed = rc == 0 and len(counts) == 1 and stopped
    result = dict(exit_code=rc, completed=passed, cycles=int(counts[0]) if passed else None)
    if not passed:
        errors = [line for line in log.splitlines()
                  if 'Assertion' in line or 'ASSERTION FAILED' in line or 'TIMEOUT' in line]
        result['error'] = (errors[0][:400] if errors else f'Incomplete run (exit {rc})')
    return result


def bandwidth(row, clock_mhz):
    """Logical useful bytes, never presented as measured HBM/SL2 bus traffic."""
    if 'payload_bytes' not in row:
        return
    payload = row['payload_bytes']
    element_bytes = {'fp16': 2, 'fp32': 4}[row['dtype']]
    is_gather = row['operator'] == 'gather_v2'
    # Each output lane performs an index MGATHER in the current generic kernel.
    # The semantic index vector is much smaller, and can be reused from cache.
    index_vector = 4 * row['output_shape'][row['axis']] if is_gather else 0
    lane_indices = 4 * (payload // element_bytes) if is_gather else 0
    row.update(data_read_bytes=payload, data_write_bytes=payload,
               data_rw_bytes=2 * payload, index_vector_bytes=index_vector,
               index_lane_bytes=lane_indices)
    if not row.get('gfsim_pass'):
        return
    total = row['kernel_sim']['cycles']
    assert total > 0, 'Kernel cycles must be positive'
    us = total / clock_mhz
    row.update(total_cycles=total, estimated_us=us,
               payload_gbps=payload / (us * 1000),
               data_rw_gbps=2 * payload / (us * 1000),
               data_rw_bytes_per_cycle=2 * payload / total,
               semantic_rw_gbps=(2 * payload + index_vector) / (us * 1000),
               indexed_lane_rw_gbps=(2 * payload + lane_indices) / (us * 1000))


def report(rows, out, args):
    for row in rows:
        bandwidth(row, args.clock_mhz)
    (out / 'results.json').write_text(json.dumps(rows, indent=2) + '\n')
    fields = ['operator', 'name', 'dtype', 'payload_bytes', 'functional_pass',
              'gfsim_pass', 'total_cycles', 'estimated_us', 'payload_gbps',
              'data_read_bytes', 'data_write_bytes', 'data_rw_bytes', 'data_rw_gbps',
              'data_rw_bytes_per_cycle',
              'index_vector_bytes', 'index_lane_bytes', 'semantic_rw_gbps', 'indexed_lane_rw_gbps']
    with (out / 'results.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# Network-inspired gather_v2 / view_copy\n\n',
             'Synthetic operator scenarios, not captured end-to-end network traces.\n\n',
             'One modeled core, four PEs; default fourpe configuration, seed 1. '
             'ELF data is preinitialized; no fixture generation, scalar golden, printf, or validation loop is timed. '
             'Total cycles include kernel and entry/exit, without empty-baseline subtraction; this is not an instruction-level ROI.\n\n',
             f'Time conversion assumes **{args.clock_mhz:g} MHz**: total cycles / MHz = us. '
             '**Effective data R+W GB/s = (selected input bytes + output bytes) / time = 2 × payload / time.** '
             'GB uses decimal 1e9 bytes. This is useful logical bandwidth for one modeled core, not physical bus traffic or whole-device bandwidth.\n\n',
             '| Operator / case | dtype | Output KiB | Host golden | Total cycles | Estimated us | Effective data R+W GB/s |\n',
             '|---|---|---:|---|---:|---:|---:|\n']
    for row in rows:
        fmt = lambda key: f'{row[key]:.3f}' if isinstance(row.get(key), float) else str(row.get(key, '-'))
        status = 'PASS' if row.get('functional_pass') else ('FAIL' if 'functional_pass' in row else 'NOT RUN')
        lines.append(f"| {row['operator']}/{row['name']} | {row['dtype']} | {row.get('payload_bytes', 0)/1024:.2f} | "
                     f"{status} | " +
                     ' | '.join(fmt(k) for k in ('total_cycles', 'estimated_us', 'data_rw_gbps')) + ' |\n')
    lines += ['\nCorrectness requires gfrun exit 0, R2=0, and byte-exact comparison of the entire output allocation '
              '(prefix/suffix guards and holes included). gfsim is timing-only; its dump is not used as an accuracy oracle. '
              'Failed functional validation suppresses timing. See per-case logs, ELF/disassembly, inputs, golden and actual dumps.\n',
              '\nIndex/layout sensitivity of gfsim is not calibrated here; these are model cycle measurements, not hardware predictions.\n']
    lines += ['\n## Gather index traffic\n\n',
              'For N output elements, dtype size d, and K entries in the uint32 index vector: '
              '`data_rw = 2*N*d`; `semantic_rw = 2*N*d + 4*K` counts the index vector once; '
              '`indexed_lane_rw = 2*N*d + 4*N` counts the generic kernel\'s index lookup for every output lane. '
              'Neither index formula measures cache misses, line amplification, spills or speculative accesses. '
              'Unselected source rows, storage holes and validation guards are not counted as useful traffic.\n\n',
              '| Gather case | Index vector bytes | Per-lane index bytes | Semantic R+W GB/s | Per-lane indexed R+W GB/s |\n',
              '|---|---:|---:|---:|---:|\n']
    for row in rows:
        if row['operator'] == 'gather_v2':
            fmt = lambda key: f'{row[key]:.3f}' if isinstance(row.get(key), float) else str(row.get(key, '-'))
            lines.append(f"| {row['name']} | " + ' | '.join(fmt(k) for k in
                         ('index_vector_bytes', 'index_lane_bytes', 'semantic_rw_gbps', 'indexed_lane_rw_gbps')) + ' |\n')
    lines += ['\nPhysical HBM/SL2 bandwidth is **not measured** by this report. '
              'The log\'s `Total Served Read/Write Bytes` belong to **CellReg Bank Arbitration**, '
              'so they must not be interpreted as external-memory bytes.\n']
    lines += ['\n## Scenarios\n\n',
              '| Case | Shape / axis | Tile elements | Input stride → output stride (elements) | Input / output offset (bytes) |\n',
              '|---|---|---:|---|---|\n']
    for row in rows:
        shape = lambda dims: '×'.join(map(str, dims))
        if row['operator'] == 'gather_v2':
            geometry = f"{shape(row['input_shape'])} → {shape(row['output_shape'])}; axis {row['axis']}"
            strides, offsets = 'contiguous', '0 / 128'
        else:
            geometry = shape(row['shape'])
            strides = f"{row['input_stride']} → {row['output_stride']}"
            offsets = f"{row.get('input_offset_bytes', 128)} / {row.get('output_offset_bytes', 128)}"
        lines.append(f"| {row['name']} | {geometry} | {row['tile']} | {strides} | {offsets} |\n")
    lines += ['\nManifest `scenario` fields explain '
              'network mapping and `indices`/`index_pattern` define deterministic selections.\n']
    for row in rows:
        if not row.get('gfsim_pass') and not args.build_only:
            reason = row.get('error') or row.get('kernel_sim', {}).get('error', 'Timing not run after validation failure')
            lines.append(f"\n- {row['name']}: {reason}\n")
    provenance_file = out / 'provenance.json'
    if provenance_file.exists():
        provenance = json.loads(provenance_file.read_text())
        lines += ['\n## Versions and reproduction\n\n',
                  f"- Benchmark base: `{provenance['benchmark_revision']}` plus source snapshots/hashes in `provenance.json`.\n",
                  f"- Compiler: `{provenance['compiler_version'].splitlines()[0]}`.\n"]
        for name, commit in provenance.get('toolchain_source_revisions', {}).items():
            lines.append(f'- {name}: `{commit}`.\n')
        if 'model_revision' in provenance:
            lines.append(f"- gfrun/gfsim model: `{provenance['model_revision']}`; binary hashes recorded.\n")
        lines += ['\n```sh\n', shlex.join(['python3', *provenance['command']]), '\n```\n']
    (out / 'REPORT.md').write_text(''.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler-dir', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path)
    parser.add_argument('--operator', choices=['view_copy', 'gather_v2'])
    parser.add_argument('--case', action='append', help='Exact case name; repeatable')
    parser.add_argument('--output', type=Path, default=ARCH / 'output/solution/network_cases')
    parser.add_argument('--build-only', action='store_true')
    parser.add_argument('--timeout', type=int, default=300)
    parser.add_argument('--clock-mhz', type=float, default=1650, help='Explicit conversion assumption, not hardware measurement')
    args = parser.parse_args()
    if not args.build_only and not args.model_dir:
        parser.error('--model-dir is required unless --build-only')
    if args.clock_mhz <= 0 or args.timeout <= 0:
        parser.error('clock and timeout must be positive')
    args.compiler_dir = args.compiler_dir.resolve()
    args.output = args.output.resolve()
    args.model_dir = args.model_dir.resolve() if args.model_dir else None
    args.output.mkdir(parents=True, exist_ok=True)
    cases = []
    for op in ('view_copy', 'gather_v2'):
        if args.operator and op != args.operator:
            continue
        cases += [dict(c, operator=op) for c in json.loads((COMMON.parent / op / 'network_cases.json').read_text())]
    if args.case:
        missing = set(args.case) - {c['name'] for c in cases}
        if missing:
            parser.error(f'Unknown cases: {sorted(missing)}')
        cases = [c for c in cases if c['name'] in args.case]
    provenance = dict(command=sys.argv, benchmark_revision=revision(REPO),
                      compiler_version=subprocess.check_output([str(args.compiler_dir / 'clang'), '--version'], text=True),
                      clock_mhz_assumption=args.clock_mhz)
    provenance['compiler_binaries'] = {name: sha256(args.compiler_dir / name)
                                      for name in ('clang', 'ld.lld', 'llvm-nm', 'llvm-objdump')}
    resource_dir = Path(subprocess.check_output(
        [str(args.compiler_dir / 'clang'), '-print-resource-dir'], text=True).strip())
    provenance['installed_tileop_headers_sha256'] = tree_hash(resource_dir / 'include/tileop-api')
    runtime_dir = args.compiler_dir.parent / 'sysroot/usr/lib'
    provenance['installed_runtime_sha256'] = {name: sha256(runtime_dir / name)
        for name in ('libc.a', 'libc++.a', 'libc++abi.a', 'libunwind.a', 'libjemalloc.a',
                     'crtbeginT.o', 'crtend.o') if (runtime_dir / name).is_file()}
    source_root = args.compiler_dir.parents[2] / 'src'
    provenance['toolchain_source_revisions'] = {name: revision(source_root / name)
        for name in ('llvm-project', 'Linx-TileOP-API', 'musl', 'jemalloc')
        if (source_root / name / '.git').exists()}
    provenance['source_sha256'] = {str(p.relative_to(REPO)): sha256(p) for p in
        [Path(__file__), COMMON / 'network_bench.cpp', COMMON / 'start.s',
         *[COMMON.parent / op / 'network_cases.json' for op in ('view_copy', 'gather_v2')],
         *[ARCH / f'kernels/solution/{op}/{op}.hpp' for op in ('view_copy', 'gather_v2')]]}
    for relative in provenance['source_sha256']:
        snapshot = args.output / 'source' / relative
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / relative, snapshot)
    if args.model_dir:
        provenance.update(model_revision=revision(args.model_dir),
                          model_binaries={name: sha256(args.model_dir / 'bin' / name) for name in ('gfrun', 'gfsim')},
                          model_configs_sha256={p.name: sha256(p) for p in sorted((args.model_dir / 'configs').glob('*.toml'))},
                          fourpe_config=(args.model_dir / 'configs/fourpe.conf').read_text())
    (args.output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    rows = []
    for case in cases:
        folder = args.output / case['operator'] / case['name']
        folder.mkdir(parents=True, exist_ok=True)
        row = dict(case)
        started = time.monotonic()
        try:
            build(row, folder, args.compiler_dir)
            row['elf_sha256'] = sha256(folder / 'kernel.elf')
            if not args.build_only:
                row.update(validate(folder, args.compiler_dir, args.model_dir, args.timeout))
                if row['functional_pass']:
                    row['kernel_sim'] = simulate(folder, args.model_dir, 'kernel', args.timeout)
                    row['gfsim_pass'] = row['kernel_sim']['completed']
                    bandwidth(row, args.clock_mhz)
        except (AssertionError, OSError, ValueError, subprocess.SubprocessError) as exc:
            row['error'] = str(exc)
        row['wall_seconds'] = time.monotonic() - started
        rows.append(row)
        (folder / 'result.json').write_text(json.dumps(row, indent=2) + '\n')
        report(rows, args.output, args)
        print(f"{case['name']}: golden={row.get('functional_pass')} timing={row.get('gfsim_pass')} "
              f"cycles={row.get('total_cycles')} error={row.get('error', '')}", flush=True)
    return 0 if all(('elf_sha256' in r if args.build_only else r.get('functional_pass') and r.get('gfsim_pass')) for r in rows) else 1


if __name__ == '__main__':
    raise SystemExit(main())
