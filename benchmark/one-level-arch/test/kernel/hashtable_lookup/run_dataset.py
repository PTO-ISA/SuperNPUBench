#!/usr/bin/env python3
"""Build separate perf/verify ELFs for the saved 6144-query dataset."""
import argparse
import collections
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
import time

CASE = Path(__file__).resolve().parent
REPO = CASE.parents[4]
COMPILER = REPO.parent / 'linx-toolchain-build/output/linx_blockisa_llvm_musl/bin'
MODEL = REPO.parent / 'SuperScalarModel'


def events(path):
    buf = None
    with path.open() as stream:
        for line in stream:
            if line.startswith('  {'):
                buf = [line]
            elif buf is not None:
                buf.append(line)
                if line.startswith('  }'):
                    yield json.loads(''.join(buf).rstrip().rstrip(','))
                    buf = None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['perf', 'verify'], required=True)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--build-only', action='store_true')
    parser.add_argument('--completion-barrier', action='store_true', help='Experimental all-PE barrier before perf finisher')
    parser.add_argument('--table-guard', type=int, default=48, metavar='BYTES',
                        help='Append BYTES of guard entries mirroring the table head')
    parser.add_argument('--kernel-header', type=Path,
                        help='Experimental kernel header; default is the baseline kernel')
    parser.add_argument('--timeout', type=int, default=600)
    args = parser.parse_args()
    out = (args.output_dir or REPO / 'outputs/hash_dataset').resolve() / args.mode
    out.mkdir(parents=True, exist_ok=True)
    cmd = ['make', '-B', '-C', str(CASE), 'TESTCASE=hashtable_lookup',
           f'hash_mode={args.mode}', 'res_check=off', 'CC_OPT=optimized',
           f'COMPILER_DIR={COMPILER}', f'OBJ_ROOT={out / "build"}']
    if args.completion_barrier:
        cmd.append("hash_perf_barrier=1")
    if args.table_guard:
        cmd.append(f"hash_table_guard={args.table_guard}")
    if args.kernel_header:
        header = args.kernel_header.resolve(strict=True)
        cmd.append(f'HASH_KERNEL_HEADER={header}')
    env = dict(os.environ, COMPILER_DIR=str(COMPILER))
    (out / 'build_command.json').write_text(json.dumps(cmd, indent=2))
    with (out / 'compile.log').open('w') as log:
        subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    elf = out / f'build/kernel/hashtable_lookup/elf/kernel_hashtable_lookup_hashtable_lookup_6144_{args.mode}_PE4.elf'
    with (out / 'kernel.diss').open('w') as f:
        subprocess.run([str(COMPILER / 'llvm-objdump'), '-d', str(elf)], stdout=f, check=True)
    # Record that the performance ELF has no oracle or verification state.
    symbols = subprocess.check_output([str(COMPILER / 'llvm-nm'), str(elf)], text=True)
    (out / 'symbols.txt').write_text(symbols)
    if args.mode == 'perf':
        for symbol in ['golden', 'errors', 'checked', 'sync']:
            if re.search(r'\b' + symbol + r'$', symbols, re.M):
                raise RuntimeError(f'Unexpected verification symbol in perf ELF: {symbol}')
    (out / 'versions.json').write_text(json.dumps({
        'elf': str(elf), 'elf_sha256': hashlib.sha256(elf.read_bytes()).hexdigest(),
        'gfsim_sha256': hashlib.sha256((MODEL / 'bin/gfsim').read_bytes()).hexdigest(),
        'compiler': str(COMPILER),
    }, indent=2))
    if args.build_only:
        print(elf)
        return
    cmd = [str(MODEL / 'bin/gfsim'), '--test-finisher', '1']
    for setting in ['softcore.multiThreadNum=4', 'cube.enable_internal_acc=true',
                    'bctrl.vec_cell_sched_enable=true', 'core.soc_random=true',
                    'core.read_bandwidth_limit=128', 'core.write_bandwidth_limit=128',
                    'core.sharetreg_size_bytes=262144', 'core.sharetreg_ssb_count=64']:
        cmd += ['-s', setting]
    cmd += ['-f', str(elf), '-t', '1']
    trace = out / 'swimlane.json'
    if args.mode == 'perf':
        cmd += ['--swimlane', '1', '--swimfile', str(trace),
                '-s', 'dfx.swimCounterSampleInterval=128']
    (out / 'command.json').write_text(json.dumps({'cwd': str(MODEL), 'argv': cmd}, indent=2))
    print(f'Running {args.mode}: {elf}', flush=True)
    per_pe, total = {}, None
    tail = collections.deque(maxlen=200)
    proc = subprocess.Popen(cmd, cwd=MODEL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    timer = threading.Timer(args.timeout, proc.kill)
    timer.start()
    next_snapshot = time.monotonic() + 10
    try:
        with (out / 'evidence.log').open('w') as log:
            for line in proc.stdout:
                tail.append(line)
                if time.monotonic() >= next_snapshot:
                    (out / 'progress.log').write_text(''.join(tail))
                    next_snapshot = time.monotonic() + 10
                if 'retire entry' in line and '|sw ' in line and 'data=0x10009000' in line:
                    match = re.search(r'-ST(\d+).*?data=(0x[0-9a-f]+)', line)
                    if match:
                        per_pe[match[1]] = match[2]
                        log.write(line)
                        log.flush()
                match = re.search(r'Total Cycles[^\d]*(\d+)', line)
                if match and total is None:
                    total = int(match[1])
                    log.write(line)
                if any(s in line for s in ['ASSERT', 'fatal', 'linx_test_finisher']):
                    log.write(line)
                    log.flush()
            rc = proc.wait()
    finally:
        timer.cancel()
    (out / 'tail.log').write_text(''.join(tail))
    completed = rc == 0 and len(per_pe) == 4 and all(x == '0x5555' for x in per_pe.values())
    result = {'mode': args.mode, 'exit': rc, 'completed': completed,
              'numerical_verified': completed if args.mode == 'verify' else None,
              'per_pe': per_pe, 'total_cycles': total}
    if args.mode == 'perf' and trace.exists():
        loads, stores = [], []
        for e in events(trace):
            if e.get('ph') != 'X':
                continue
            if re.search(r'\bTLOAD\b', e.get('name', '')):
                loads.append(e['ts'])
            if re.search(r'\bTSTORE\b', e.get('name', '')):
                stores.append(e['ts'] + e['dur'])
        result.update(input_loads=len(loads), result_stores=len(stores))
        if loads and stores:
            result.update(first_input_load=min(loads), last_result_store=max(stores),
                          kernel_cycles=max(stores) - min(loads))
        with trace.open('rb') as src, gzip.open(str(trace) + '.gz', 'wb', compresslevel=1) as dst:
            shutil.copyfileobj(src, dst)
        trace.unlink()
        completed = completed and len(loads) == 192 and len(stores) == 192
        result['completed'] = completed
    elif args.mode == 'perf':
        completed = False
        result['completed'] = False
    (out / 'result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    if not completed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
