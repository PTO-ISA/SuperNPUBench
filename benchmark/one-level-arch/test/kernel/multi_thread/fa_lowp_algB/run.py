#!/usr/bin/env python3
"""Build and run experimental FA B/C. Requires COMPILER_DIR and MODEL env vars."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--mode', choices=['check', 'perf', 'build'], default='check')
p.add_argument('--algorithm', choices=['B', 'C'], default='C')
p.add_argument('--gm-fused', type=int, choices=[0, 1, 2], default=0)
p.add_argument('--sq', type=int, default=256)
p.add_argument('--skv', type=int, default=256)
p.add_argument('--tk', type=int, default=128)
p.add_argument('--vecq', type=int, choices=[32, 64, 96], default=32)
p.add_argument('--clang-resource', type=Path, help='isolated clang resource directory containing the required TileOP API')
p.add_argument('--run-id', required=True)
a = p.parse_args()
if a.gm_fused == 2 and a.algorithm != 'C':
    p.error('--gm-fused 2 requires --algorithm C')
if a.gm_fused == 2 and a.tk != 128:
    p.error('--gm-fused 2 is currently validated only for --tk 128')
test = Path(__file__).resolve().parent
bm = test.parents[3]
model = Path(os.environ['MODEL']).resolve()
compiler = Path(os.environ['COMPILER_DIR']).resolve()
run = test / 'perf_runs' / a.run_id
run.mkdir(parents=True, exist_ok=False)
check = 'on' if a.mode == 'check' else 'off'
c = int(a.algorithm == 'C')
stem = f'kernel_multi_thread_fa_lowp_algB_C{c}_GM{a.gm_fused}_Sq{a.sq}_Skv{a.skv}_Tk{a.tk}_check{check}'
obj = bm / 'output/kernel/multi_thread/fa_lowp_algB/src/fa_lowp_algB.o'
obj.unlink(missing_ok=True)
cmd = ['make', '-j4', 'all', f'COMPILER_DIR={compiler}', f'Sq={a.sq}', f'Skv={a.skv}', f'Tk={a.tk}', f'ALGO_C={c}', f'GM_FUSED={a.gm_fused}', f'res_check={check}']
if a.clang_resource:
    import shlex
    cmd.append('CFLAGS=-resource-dir ' + shlex.quote(str(a.clang_resource.resolve())))
(run / 'command.txt').write_text(repr(vars(a)) + '\n' + repr(cmd) + '\n')
with (run / 'build.log').open('w') as log:
    subprocess.run(cmd, cwd=test, stdout=log, stderr=subprocess.STDOUT, check=True)
elf = bm / 'output/kernel/multi_thread/fa_lowp_algB/elf' / (stem + '.elf')
import shutil
saved = run / (stem + '.elf')
shutil.copy2(elf, saved)
print('ELF:', saved, flush=True)
if a.mode == 'check':
    data = bm / 'compare' / stem
    verify = [sys.executable, str(test / 'verify_fa_lowp.py'), '--outdir', str(data), '--sq', str(a.sq), '--skv', str(a.skv), '--algorithm', a.algorithm]
    subprocess.run(verify, check=True)
    (data / 'res.bin').unlink(missing_ok=True)
    with (run / 'gfrun.log').open('w') as log:
        subprocess.run([str(model / 'bin/gfrun'), '-s', 'softcore.multiThreadNum=4', '-f', str(saved)], cwd=model, stdout=log, stderr=subprocess.STDOUT, check=True)
    result = subprocess.run(verify + ['--check'], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (run / 'check.log').write_text(result.stdout)
    print(result.stdout)
    shutil.copytree(data, run / "check_data")
    result.check_returncode()
elif a.mode == 'perf':
    # The common helper has no -s passthrough: a wrapper preserves its trace handling.
    import shlex
    wrapper = run / 'gfsim_configured.sh'
    wrapper.write_text('#!/bin/sh\ncd ' + shlex.quote(str(model)) + '\nexec ' + shlex.quote(str(model / 'bin/gfsim')) + f' -s bctrl.vec_cell_sched_enable=false bctrl.vecIssueQDepth={a.vecq} "$@"\n')
    wrapper.chmod(0o755)
    subprocess.run([sys.executable, str(bm / 'test/common/run_swimlane.py'), '--gfsim', str(wrapper), '--elf', str(saved), '--outdir', str(run), '--name', 'algo' + a.algorithm, '--conf', 'fourpe'], cwd=model, check=True)
