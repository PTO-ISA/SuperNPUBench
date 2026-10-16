#!/usr/bin/env python3
"""Fault-injection checks for runner evidence; these do not qualify kernels."""
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent


class RunnerFailureTest(unittest.TestCase):
    def environment(self, root):
        return dict(os.environ, COMPILER_DIR=str(root / 'bin'), SSM=str(root / 'model'),
                    LINX_RUNTIME_ROOT=str(root / 'runtime'), API_INCLUDE=str(root / 'api/include'),
                    RESOURCE_DIR=str(root / 'resource'), SYSROOT=str(root / 'sysroot'),
                    MODELS='gfrun', ARTIFACT_DIR=str(root / 'artifacts'),
                    GFRUN=str(root / 'gfrun'), CXX=shutil.which('c++') or 'c++')

    def fake_prerequisites(self, root):
        for name in ['bin/clang++', 'bin/llvm-objdump', 'bin/ld.lld', 'gfrun',
                     'sysroot/usr/lib/crt1.o', 'sysroot/usr/lib/libc.a',
                     'sysroot/usr/lib/libc++.a', 'sysroot/usr/lib/libc++abi.a',
                     'sysroot/usr/lib/libunwind.a', 'api/include/common/pto_tileop.hpp',
                     'resource/include/tileop-api/common/pto_tileop.hpp']:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('fixture')
        (root / 'api/include/common/pto_tileop.hpp').write_text('selected API')
        (root / 'resource/include/tileop-api/common/pto_tileop.hpp').write_text('stale API')
        (root / 'bin/clang++').write_text("#!/bin/sh\nprintf 'clang fixture\\n'\n")
        (root / 'bin/clang++').chmod(0o755)
        shim = root / 'shim'
        shim.mkdir()
        # Simulate successful API installation, then fail the actual build.
        (shim / 'make').write_text('''#!/bin/sh
case " $* " in
  *" install "*)
    for arg in "$@"; do
      case "$arg" in INSTALL_DIR=*) dst=${arg#INSTALL_DIR=};; esac
    done
    mkdir -p "$dst"
    cp -R "$API_INCLUDE/." "$dst/"
    exit 0;;
esac
exit 42
''')
        (shim / 'make').chmod(0o755)
        # The runner's unit-test subprocesses are separate from the fault
        # being tested. Keep real golden generation and provenance collection.
        (shim / 'python3').write_text('''#!/bin/sh
case "$1" in */test_*.py) exit 0;; esac
exec ''' + shlex.quote(sys.executable) + ' "$@"\n')
        (shim / 'python3').chmod(0o755)
        return str(shim) + os.pathsep + os.environ.get('PATH', '')

    def execute(self, name, args, env):
        return subprocess.run(['/bin/bash', str(HERE / name), *args], env=env,
                              capture_output=True, text=True, timeout=60)

    def run_finalizer(self, root, runner, snapshot, writer_status):
        # Exercise the actual finalizer definitions without mocking a compiler.
        primary = runner == 'run_element_atomic_topk.sh'
        name = 'write_final_provenance' if primary else 'write_active_provenance'
        source = (HERE / runner).read_text()
        start = source.index(name + '() {')
        function = source[start:source.index('\n}\n', start) + 3]
        if snapshot:
            (root / 'input-provenance.json').write_text('{}')
        globals_ = dict(artifact_dir=root, active_artifact=root, case_dir=root,
                        bench_root=root, elf=root / 'kernel.elf', active_elf=root / 'kernel.elf',
                        active_golden=root, COMPILER_DIR=root, SSM=root,
                        LINX_RUNTIME_ROOT=root, API_INCLUDE=root, api_source_include=root,
                        TARGET_TRIPLE='linx64v5', SYSROOT=root, RESOURCE_DIR=root,
                        MODELS='gfrun', gfrun=root / 'gfrun', gfsim=root / 'gfsim',
                        dump_range='', active_dump_range='', active_pe_count='1')
        script = 'set -euo pipefail\n' + '\n'.join(
            key + '=' + shlex.quote(str(value)) for key, value in globals_.items())
        script += '\nprovenance_source_args=()\nprovenance_model_args=(--models gfrun)\n'
        script += r"""
python3() {
    local snapshot=''
    while (( $# )); do
        if [[ $1 == --verify-inputs-from ]]; then snapshot=$2; shift; fi
        shift
    done
    [[ "$snapshot" == "$artifact_dir/input-provenance.json" ]] || return 91
    echo finalized >> "$artifact_dir/events"
    [[ -f "$snapshot" ]] || return 92
    return """ + str(writer_status) + '\n}\n'
        script += function
        if primary:
            script += '\ntrap write_final_provenance EXIT\ntrue\n'
        else:
            script += '\nwrite_active_provenance 0\necho suite-PASS\n'
        return subprocess.run(['/bin/bash', '-c', script], capture_output=True,
                              text=True, timeout=10)

    def test_success_requires_snapshot_in_both_runners(self):
        for runner in ['run_element_atomic_topk.sh', 'run_tile_element_suite.sh']:
            with self.subTest(runner=runner), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                result = self.run_finalizer(root, runner, False, 0)
                self.assertEqual(result.returncode, 92, result.stdout + result.stderr)
                self.assertNotIn('PASS', result.stdout)
                self.assertEqual((root / 'events').read_text(), 'finalized\n')

    def test_primary_pass_requires_successful_finalization(self):
        for status in [0, 73]:
            with self.subTest(status=status), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                result = self.run_finalizer(root, 'run_element_atomic_topk.sh', True, status)
                self.assertEqual(result.returncode, status, result.stdout + result.stderr)
                self.assertEqual((root / 'events').read_text(), 'finalized\n')
                self.assertEqual('element_atomic_topk: PASS' in result.stdout, status == 0)
                source = (HERE / 'run_element_atomic_topk.sh').read_text()
                self.assertEqual(source.count('element_atomic_topk: PASS'), 1)
                self.assertLess(source.index('local provenance_status=$?'),
                                source.index('element_atomic_topk: PASS'))

    def test_setup_failures_have_partial_identity(self):
        for runner, args in [('run_element_atomic_topk.sh', []),
                             ('run_tile_element_suite.sh', ['--case', 'histogram_tile_element'])]:
            with self.subTest(runner=runner), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                result = self.execute(runner, args, self.environment(root))
                self.assertNotEqual(result.returncode, 0)
                proof = json.loads((root / 'artifacts/provenance.json').read_text())
                self.assertEqual(proof['run_exit_status'], result.returncode)
                self.assertEqual(proof['provenance_status'], 'incomplete')
                self.assertEqual(len(proof['content_id']), 64)
                self.assertIn('benchmark', proof['repositories'])
                self.assertFalse((root / 'artifacts/gfrun.mem').exists())

    def test_primary_installs_selected_api_without_mutating_base(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            env = self.environment(root)
            env['PATH'] = self.fake_prerequisites(root)
            result = self.execute('run_element_atomic_topk.sh', [], env)
            self.assertEqual(result.returncode, 42, result.stdout + result.stderr)
            proof = json.loads((root / 'artifacts/provenance.json').read_text())
            self.assertEqual(proof['run_exit_status'], 42)
            self.assertEqual((root / 'artifacts/clang-resource/include/tileop-api/common/pto_tileop.hpp').read_text(), 'selected API')
            self.assertEqual((root / 'resource/include/tileop-api/common/pto_tileop.hpp').read_text(), 'stale API')
            self.assertEqual(proof['target']['resource_dir'], str(root / 'artifacts/clang-resource'))

    def test_suite_build_failure_records_current_case(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            env = self.environment(root)
            env['PATH'] = self.fake_prerequisites(root)
            (root / 'api/Makefile').write_text('# installation handled by the test shim\n')
            result = self.execute('run_tile_element_suite.sh', ['--case', 'histogram_tile_element'], env)
            self.assertEqual(result.returncode, 42, result.stdout + result.stderr)
            case = root / 'artifacts/histogram_tile_element'
            proof = json.loads((case / 'provenance.json').read_text())
            self.assertEqual(proof['run_exit_status'], 42)
            self.assertTrue(proof['artifacts']['goldens'])
            self.assertIsNone(proof['artifacts']['elf']['sha256'])
            self.assertFalse((case / 'gfrun.mem').exists())


if __name__ == '__main__':
    unittest.main()
