#!/usr/bin/env python3
"""Build the archived zero-input timing snapshot; optionally produce SwimLane."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--toolchain-bin', required=True, type=Path)
    parser.add_argument('--outdir', required=True, type=Path)
    parser.add_argument('--gfsim', type=Path)
    args = parser.parse_args()
    tools = args.toolchain_bin.resolve()
    out = args.outdir.resolve()
    if out == ROOT or ROOT in out.parents:
        parser.error('choose an output directory outside the shipped archive')
    if out.exists():
        parser.error('output directory already exists; choose a new one')
    identity = json.loads((ROOT / 'identity.json').read_text())
    if sha(tools / 'clang++') != identity['compiler']['sha256']:
        parser.error('compiler hash differs from the frozen historical compiler')
    model = args.gfsim.resolve() if args.gfsim else None
    if model and sha(model) != identity['historical_run']['model_sha256']:
        parser.error('runtime hash differs from the historical timing runtime')
    subprocess.run([sys.executable, str(ROOT / 'verify_sources.py')], check=True)
    out.mkdir(parents=True)
    scratch = out / 'tmp'
    scratch.mkdir()
    env = dict(os.environ, TMPDIR=str(scratch), PYTHONDONTWRITEBYTECODE="1")
    elf = out / 'rmsnorm_single_buffer.elf'
    obj = out / 'kernel.o'
    flags = json.loads((ROOT / 'build_flags.json').read_text())
    flags += [f'-I{ROOT / "source/support"}', f'-I{ROOT / "source"}',
              f'-ffile-prefix-map={ROOT}=.',
              f'-fdebug-prefix-map={ROOT}=.',
              f'-ffile-prefix-map={tools.parent}=toolchain',
              f'-fdebug-prefix-map={tools.parent}=toolchain']
    commands = [
        [str(tools / 'clang++'), *flags, str(ROOT / 'source/driver.cpp'),
         '-o', str(obj)],
        [str(tools / 'clang++'), '-nostartfiles',
         str(ROOT / 'source/support/_start.s'), str(obj), '-o', str(elf)],
        [str(tools / 'llvm-objdump'), '-dl', elf.name],
    ]
    for index, command in enumerate(commands):
        log = out / ('rmsnorm_single_buffer.elf.diss' if index == 2
                     else f'build_{index}.log')
        with log.open('w') as handle:
            subprocess.run(command, cwd=out, env=env, stdout=handle,
                           stderr=subprocess.STDOUT, check=True)
    result = {'commands': commands, 'compiler_sha256': sha(tools / 'clang++'),
              'elf_sha256': sha(elf),
              'diss_sha256': sha(out / 'rmsnorm_single_buffer.elf.diss'),
              'correctness': 'not_checked; zero-input timing build only'}
    (out / 'build_identity.json').write_text(json.dumps(result, indent=2) + '\n')
    obj.unlink()
    if model:
        (out / 'configs').mkdir()
        shutil.copyfile(ROOT / 'configs/r128_w128_shared128.conf',
                        out / 'configs/r128_w128_shared128.conf')
        command = [sys.executable, str(ROOT / 'tools/run_swimlane.py'),
                   '--gfsim', str(model), '--elf', str(elf),
                   '--outdir', str(out / 'perf_run'), '--name', 'rmsnorm',
                   '--conf', 'r128_w128_shared128.conf', '--timeout', '300']
        result.update(run_command=command, model_sha256=sha(model),
                      environment_overrides={'RMS_TAIL_BYPASS': '1'})
        (out / 'build_identity.json').write_text(json.dumps(result, indent=2) + '\n')
        subprocess.run(command, cwd=out, env=dict(env, RMS_TAIL_BYPASS='1'),
                       check=True)
    scratch.rmdir()
    print(f'Built {elf}; no numerical correctness claim')


if __name__ == '__main__':
    main()
