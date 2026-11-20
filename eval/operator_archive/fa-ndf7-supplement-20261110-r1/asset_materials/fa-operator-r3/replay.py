#!/usr/bin/env python3
"""Verify and replay a zero-input FA case; qualify rebuilt artifacts explicitly."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
sys.dont_write_bytecode = True

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for flag in ['operator-root', 'model-root', 'outdir']:
        ap.add_argument('--' + flag, required=True, type=Path)
    ap.add_argument('--case', required=True)
    ap.add_argument('--timeout', type=int, default=3600)
    ap.add_argument('--memory-limit-gib', type=int, default=10)
    ap.add_argument('--preflight', action='store_true', help='verify materials only; no replay acceptance')
    ap.add_argument('--qualification-mode', action='store_true', help='explicitly qualify a rebuilt runtime and/or ELF against fixed timing evidence')
    ap.add_argument('--rebuilt-runtime', type=Path)
    ap.add_argument('--rebuilt-elf', type=Path)
    a = ap.parse_args()
    operator, model, out = a.operator_root.resolve(), a.model_root.resolve(), a.outdir.resolve()
    if out.exists() or any(out == x or out.is_relative_to(x) for x in [operator, model]):
        ap.error('choose a new output directory outside payloads')
    if not 1 <= a.memory_limit_gib <= 64:
        ap.error('memory-limit-gib must be 1..64')
    if bool(a.rebuilt_runtime or a.rebuilt_elf) != a.qualification_mode:
        ap.error('rebuilt artifact overrides require explicit --qualification-mode')
    manifest = json.loads((operator / 'manifest.json').read_text())
    cases = [c for c in manifest['cases'] if c['id'] == a.case]
    if len(cases) != 1:
        ap.error('unknown or duplicate case')
    case = cases[0]
    allowed = {'PATH', 'LANG', 'LC_ALL', 'TZ'}
    env = {'PATH':'/usr/bin:/bin','LANG':'C','LC_ALL':'C','TZ':'UTC'}
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    for root in [operator, model]:
        cmd = [sys.executable, str(root / 'verify.py')]
        if root == model and a.rebuilt_runtime:
            cmd += ['--data-only']
        subprocess.run(cmd, env=env, check=True)
    elf = a.rebuilt_elf.resolve() if a.rebuilt_elf else operator / case['elf']
    runtime = a.rebuilt_runtime.resolve() if a.rebuilt_runtime else model / 'bin/gfsim'
    if not elf.is_file() or not runtime.is_file():
        raise SystemExit('FAIL: ELF/runtime missing')
    if not a.rebuilt_elf and sha(elf) != case['elf_sha256']:
        raise SystemExit('FAIL: ELF identity differs')
    if not a.rebuilt_runtime and sha(runtime) != manifest['runtime_sha256']:
        raise SystemExit('FAIL: runtime identity differs')
    # Never silently accept a default-config run.
    for name, digest in manifest['configs_sha256'].items():
        path = model / 'configs' / name
        if not path.is_file() or sha(path) != digest:
            raise SystemExit('FAIL: missing/mismatched configuration: ' + name)
    if a.preflight:
        print(json.dumps({'status':'pass','scope':'preflight only; simulation not run','case':a.case}))
        return
    out.mkdir(parents=True)
    (out / 'tmp').mkdir()
    env['TMPDIR'] = str(out / 'tmp')
    shutil.copytree(model / 'configs', out / 'configs')
    executor = runtime if a.rebuilt_runtime else model / 'run_model.py'
    command = [sys.executable, str(operator / 'tools/run_swimlane.py'), '--gfsim', str(executor),
               '--elf', str(elf), '--outdir', str(out / 'perf_run'), '--name', a.case,
               '--gzip-full', '--gzip-nocounters', '--conf', 'perf.conf',
               '--timeout', str(a.timeout), '--memory-limit-gib', str(a.memory_limit_gib)]
    result = {'command':command,'cwd':str(out),'environment':env,
              'removed_inherited_environment_names':sorted(set(os.environ)-allowed),
              'input_mode':'static zero input; no RES_CHECK or file inputs',
              'numeric_status':'not_checked; complete numerical failures unresolved',
              'qualification_mode':a.qualification_mode,'runtime_sha256':sha(runtime),
              'elf_sha256':sha(elf),'historical_elf_bytes_equal':sha(elf)==case['elf_sha256'],
              'host_memory_limit_gib':a.memory_limit_gib}
    (out / 'invocation.json').write_text(json.dumps(result,indent=2)+'\n')
    start = time.monotonic()
    proc = subprocess.run(command,cwd=out,env=env)
    result['elapsed_seconds'] = time.monotonic()-start
    perf = out / 'perf_run'
    log = perf / (a.case+'_swim.log')
    matches = re.findall(r'^Total Cycles[.\s]*:\s*(\d+)\s*$',log.read_text() if log.exists() else '',re.M)
    cycles = int(matches[-1]) if matches else None
    files = {}
    for suffix in ['_swim.log','.json.gz','_nocounters.json.gz']:
        path = perf / (a.case+suffix)
        if path.is_file(): files[path.name]={'sha256':sha(path),'bytes':path.stat().st_size}
    identity_path = perf / (a.case+'_full_identity.json')
    identity = json.loads(identity_path.read_text()) if identity_path.exists() else {}
    expected = case['full_trace']
    trace_match = identity.get('sha256')==expected['sha256'] and identity.get('bytes')==expected['uncompressed_bytes']
    passed = proc.returncode==0 and cycles==case['cycles'] and len(files)==3 and trace_match
    result.update(status='pass' if passed else 'fail',returncode=proc.returncode,total_cycles=cycles,
                  expected_cycles=case['cycles'],full_trace=identity,full_trace_matches=trace_match,
                  files=files,scope='case timing and exact full trace only; not numerical or universal equivalence')
    (out / 'replay_result.json').write_text(json.dumps(result,indent=2)+'\n')
    if not passed:
        raise SystemExit('FAIL: simulator exit, cycles, full trace identity or output artifacts differ')
    print(json.dumps({'status':'pass','case':a.case,'total_cycles':cycles,'full_trace_matches':trace_match}))

if __name__=='__main__':
    main()
