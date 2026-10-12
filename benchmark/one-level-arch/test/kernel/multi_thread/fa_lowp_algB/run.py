#!/usr/bin/env python3
"""Build and run experimental FA B/C/D. Requires COMPILER_DIR and MODEL env vars."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--mode', choices=['check', 'perf', 'build'], default='check')
p.add_argument('--algorithm', choices=['B', 'C', 'D'], default='C')
p.add_argument('--d-variant', choices=['base', 'preload', 'dblQ', 'dblQ_preload'], default='base',
               help='Algo D schedule variant')
p.add_argument('--preload-count', type=int, choices=[2, 3, 4, 5], default=2,
               help='continuous K/KScale lookahead distance in KV blocks (2..5)')
p.add_argument('--d-const-kv-address', action='store_true',
               help='load K/KScale/V/VScale from block 0 on every dblQ iteration (timing only)')
p.add_argument('--gm-fused', type=int, choices=[0, 1, 2], default=None, help='C/D require 2; B defaults to 0')
p.add_argument('--sq', type=int, default=256)
p.add_argument('--skv', type=int, default=256)
p.add_argument('--tk', type=int, default=128)
p.add_argument('--vecq', type=int, choices=[32, 64, 96], default=32)
p.add_argument('--l0a-resident', type=int, choices=[0, 1], default=0)
p.add_argument('--sharetreg-bandwidth', type=int, choices=[128, 256], default=256)
p.add_argument('--soc-random', action='store_true')
p.add_argument('--clang-resource', type=Path, help='isolated clang resource directory containing the required TileOP API')
p.add_argument('--run-id', required=True)
p.add_argument('--debug', action='store_true', help='export D functional intermediates (check only)')
p.add_argument('--seed', type=int, default=1, help='functional input seed')
a = p.parse_args()
if a.debug and (a.algorithm != 'D' or a.mode != 'check'):
    p.error('--debug requires --algorithm D --mode check')
if a.d_variant != 'base' and a.algorithm != 'D':
    p.error('--d-variant requires --algorithm D')
if a.d_variant in ('dblQ', 'dblQ_preload') and (a.sq < 256 or a.sq % 256):
    p.error('dblQ requires --sq to be a positive multiple of 256')
if a.d_const_kv_address and (a.algorithm != 'D' or a.d_variant not in ('dblQ', 'dblQ_preload')):
    p.error('--d-const-kv-address requires --algorithm D --d-variant dblQ or dblQ_preload')
if a.d_const_kv_address and a.mode == 'check':
    p.error('--d-const-kv-address is timing-only; use --mode build or perf')
if a.gm_fused is None:
    a.gm_fused = 2 if a.algorithm in ('C', 'D') else 0
if a.algorithm in ('C', 'D') and a.gm_fused != 2:
    p.error('C/D retain only --gm-fused 2')
if a.gm_fused == 2 and a.algorithm not in ('C', 'D'):
    p.error('--gm-fused 2 requires --algorithm C or D')
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
d = int(a.algorithm == 'D')
d_preload = int(a.d_variant in ('preload', 'dblQ_preload'))
d_dblq = int(a.d_variant in ('dblQ', 'dblQ_preload'))
d_preload_n = a.preload_count if d_preload else 0
d_suffix = '_D1' if d else ''
variant_suffix = (('_dblQ' + ('_constKV' if a.d_const_kv_address else '') + '_preload' + str(a.preload_count)) if a.d_variant == 'dblQ_preload'
                  else (f'_preload{a.preload_count}' if d_preload else
                        ('_dblQ' + ('_constKV' if a.d_const_kv_address else '') if d_dblq else '')))
stem = f'kernel_multi_thread_fa_lowp_algB{d_suffix}{variant_suffix}_C{c}_GM{a.gm_fused}_Sq{a.sq}_Skv{a.skv}_Tk{a.tk}_check{check}'
obj = bm / 'output/kernel/multi_thread/fa_lowp_algB/src/fa_lowp_algB.o'
obj.unlink(missing_ok=True)
cmd = ['make', '-j4', 'all', f'COMPILER_DIR={compiler}', f'Sq={a.sq}', f'Skv={a.skv}', f'Tk={a.tk}', f'ALGO_C={c}', f'ALGO_D={d}', f'GM_FUSED={a.gm_fused}', f'D_PRELOAD={d_preload}', f'D_PRELOAD_N={d_preload_n}', f'D_DBLQ={d_dblq}', f'D_CONST_KV_ADDR={int(a.d_const_kv_address)}', f'res_check={check}']
flags = ['-DFA_LOWP_DEBUG'] if a.debug and a.mode == 'check' else []
if a.clang_resource:
    import shlex
    flags.extend(['-resource-dir', shlex.quote(str(a.clang_resource.resolve()))])
if flags:
    cmd.append('CFLAGS=' + ' '.join(flags))
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
    verify = [sys.executable, str(test / 'verify_fa_lowp.py'), '--outdir', str(data), '--sq', str(a.sq), '--skv', str(a.skv), '--tk', str(a.tk), '--algorithm', a.algorithm, '--seed', str(a.seed)]
    subprocess.run(verify, check=True)
    (data / 'res.bin').unlink(missing_ok=True)
    (data / 'debug.bin').unlink(missing_ok=True)
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
    overrides = [
        'bctrl.vec_cell_sched_enable=false', f'bctrl.vecIssueQDepth={a.vecq}',
        'spe.speROBDepth=128',  # fourpe already fixes its ROB partition at 128.
        f'cube.l0aSharetregResident={"true" if a.l0a_resident else "false"}',
        f'core.sharetreg_bandwidth_bytes={a.sharetreg_bandwidth}',
        'cube.enable_internal_acc=true',
    ]
    if a.soc_random:
        overrides.append('core.soc_random=true')
    wrapper.write_text('#!/bin/sh\ncd ' + shlex.quote(str(model)) + '\nexec ' + shlex.quote(str(model / 'bin/gfsim')) + ' -s ' + ' '.join(overrides) + ' "$@"\n')
    wrapper.chmod(0o755)
    run_name = 'algo' + a.algorithm + (('_dblQ' + ('_constKV' if a.d_const_kv_address else '') + '_preload' + str(a.preload_count)) if a.d_variant == 'dblQ_preload' else
               ('_preload' + str(a.preload_count) if d_preload else
                ('_dblQ' + ('_constKV' if a.d_const_kv_address else '') if d_dblq else '')))
    subprocess.run([sys.executable, str(bm / 'test/common/run_swimlane.py'), '--gfsim', str(wrapper), '--elf', str(saved), '--outdir', str(run), '--name', run_name, '--conf', 'fourpe'], cwd=model, check=True)
