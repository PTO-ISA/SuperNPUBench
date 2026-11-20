#!/usr/bin/env python3
"""Optional runtime rebuild from a fixed upstream Git commit and shipped patch."""
import argparse
import json
import hashlib
import shutil
import os
from pathlib import Path
import subprocess
import sys

BASE = 'dd86f8a20b9161bdb1f836dc234c86d187d561e4'
ROOT = Path(__file__).resolve().parent

def verify_postimages(source):
    expected = json.loads((ROOT / 'source/postimages.json').read_text())
    if hashlib.sha256((ROOT / 'source/portable-model.patch').read_bytes()).hexdigest() != expected['patch_sha256']:
        raise SystemExit('FAIL: model patch identity differs')
    for name, digest in expected['sha256'].items():
        path = source / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise SystemExit('FAIL: model postimage differs: ' + name)
    return expected

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model-repo', type=Path, required=True, help='Git repository containing the fixed base commit')
    ap.add_argument('--source-out', type=Path, required=True)
    ap.add_argument('--build-out', type=Path, required=True)
    ap.add_argument('--cxx', default='g++', help='GCC C++ compiler with libstdc++; clang is unqualified')
    ap.add_argument('--cc', default='gcc')
    ap.add_argument('--recover-only', action='store_true', help='restore and verify source only; no build acceptance')
    args = ap.parse_args()
    source = args.source_out.resolve()
    build = args.build_out.resolve()
    if source.exists() or build.exists() or source == build or build.is_relative_to(source) or source.is_relative_to(build) or any(x.is_relative_to(ROOT) for x in [source,build]):
        ap.error('choose distinct new source/build directories, build outside source')
    subprocess.run([sys.executable,'-B',str(ROOT/'verify.py'),'--data-only'],check=True)
    source.mkdir(parents=True)
    archive = subprocess.Popen(['git', '-C', str(args.model_repo.resolve()), 'archive', BASE], stdout=subprocess.PIPE)
    subprocess.run(['tar', '-xf', '-', '-C', str(source)], stdin=archive.stdout, check=True)
    archive.stdout.close()
    if archive.wait() != 0:
        raise SystemExit('FAIL: fixed source commit unavailable')
    if not shutil.which('patch'):
        raise SystemExit('FAIL: POSIX patch is required')
    # patch -p1 resolves against source even inside an enclosing Git worktree.
    subprocess.run(['patch', '--batch', '--forward', '-p1', '-i', str(ROOT / 'source/portable-model.patch')], cwd=source, check=True)
    recovered = verify_postimages(source)
    if args.recover_only:
        print(json.dumps({'status': 'pass', 'scope': 'source postimages only; model build not checked', 'files': len(recovered['sha256'])}))
        return
    cxx = shutil.which(args.cxx)
    cc = shutil.which(args.cc)
    if not cxx or not cc:
        raise SystemExit('FAIL: GCC/g++ unavailable; pass --cxx and --cc explicitly')
    macros = subprocess.check_output([cxx, '-dM', '-E', '-x', 'c++', '-'], input='#include <version>\n', text=True)
    if '__clang__' in macros or '__GLIBCXX__' not in macros or '__GNUC__' not in macros:
        raise SystemExit('FAIL: rebuild qualification requires GCC with libstdc++; do not treat clang/libc++ cycles as equivalent')
    cc_macros = subprocess.check_output([cc,'-dM','-E','-x','c','-'],input='',text=True)
    if '__clang__' in cc_macros or '__GNUC__' not in cc_macros:
        raise SystemExit('FAIL: C compiler must also be GCC')
    build.mkdir(parents=True)
    scratch = build / 'tmp'
    scratch.mkdir()
    env = {'PATH':'/usr/bin:/bin','LANG':'C','LC_ALL':'C','TZ':'UTC','TMPDIR':str(scratch),'CCACHE_DISABLE':'1','PYTHONDONTWRITEBYTECODE':'1'}
    flags = ' '.join(f'-{option}-prefix-map={directory}={label}'
                     for directory, label in [(source, 'SuperScalarModel'), (build, 'build')]
                     for option in ['ffile', 'fdebug'])
    # The fixed softfloat subproject resets CMAKE_C_FLAGS after project().
    # Its project include hook preserves mappings as compile options without editing source.
    hook = build / 'softfloat_prefix_maps.cmake'
    hook.write_text('add_compile_options(\n' + '\n'.join('  "' + flag.replace('\\', '\\\\').replace('"', '\\"') + '"'
                    for flag in flags.split()) + '\n)\n')
    configure = ['cmake', '-S', str(source), '-B', str(build), '-DCMAKE_CXX_COMPILER=' + cxx, '-DCMAKE_C_COMPILER=' + cc,
                 '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON', '-DOPT_LEVEL=O3', '-DDISABLE_DEBUG_SYMBOLS=ON',
                 '-DBUILD_TESTS=OFF', '-DBUILD_DPI=OFF', '-DGENERIC_SOC=OFF', '-DGENERIC_SOC_NEW=OFF',
                 '-DPYTHON_EXECUTABLE=' + sys.executable, '-DCMAKE_CXX_FLAGS=' + flags,
                 '-DCMAKE_C_FLAGS=' + flags,
                 '-DCMAKE_PROJECT_softfloat_INCLUDE=' + str(hook),
                 '-DCMAKE_RUNTIME_OUTPUT_DIRECTORY=' + str(build / 'bin'),
                 '-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY=' + str(build / 'lib'),
                 '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + str(build / 'lib')]
    compile_command = ['cmake', '--build', str(build), '--target', 'gfsim', '-j4']
    (build / 'commands.json').write_text(json.dumps({'configure': configure, 'build': compile_command}, indent=2) + '\n')
    subprocess.run(configure, env=env, check=True)
    subprocess.run(compile_command, env=env, check=True)
    identity = {'runtime_sha256': hashlib.sha256((build / 'bin/gfsim').read_bytes()).hexdigest(),
                'cxx_version': subprocess.check_output([cxx, '--version'], text=True),
                'cc_version': subprocess.check_output([cc, '--version'], text=True),
                'source_postimages_verified': recovered, 'qualification': 'not_checked; run qualify_rebuilt.py with the operator asset'}
    (build / 'rebuilt_identity.json').write_text(json.dumps(identity, indent=2) + '\n')
    print('Built ' + str(build / 'bin/gfsim') + '; run qualify_rebuilt.py before claiming cycle/trace equivalence.')

if __name__ == '__main__':
    main()
