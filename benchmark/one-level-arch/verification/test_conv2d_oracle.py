#!/usr/bin/env python3
"""Reject false completion and malformed observations in both conv2d gates."""
import contextlib
import importlib.util
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

SOURCES = Path(__file__).resolve().parents[1] / 'test/solution/conv2d/src'


class Conv2DOracleTest(unittest.TestCase):
    def test_rejects_false_success_and_missing_observation(self):
        for name in ('golden_conv2d_img2col', 'golden_conv2d_img2col_dyn'):
            spec = importlib.util.spec_from_file_location(name, SOURCES / (name + '.py'))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            for failure in ('process_exit', 'r2', 'unwritten', 'extra_byte'):
                with self.subTest(oracle=name, failure=failure), tempfile.TemporaryDirectory() as temp:
                    root = Path(temp)
                    elf = root / 'test.elf'
                    elf.write_bytes(b'fixture')
                    source = root / 'test/solution/conv2d/src' / (name + '.py')
                    output = root / 'compare/test/output.bin'

                    def fake_model(command, **kwargs):
                        self.assertEqual(kwargs['timeout'], 120)
                        self.assertNotIn('-t', command)
                        if failure == 'extra_byte':
                            output.write_bytes(output.read_bytes() + b'\x00')
                        r2 = 1 if failure == 'r2' else 0
                        return subprocess.CompletedProcess(command,
                            1 if failure == 'process_exit' else 0,
                            stdout=f'Success to Reach the End of Benchmark! R2 = {r2}\n',
                            stderr='')

                    with patch.object(module, '__file__', str(source)), \
                         patch.object(sys, 'argv', [name, '--elf', str(elf)]), \
                         patch.object(module.subprocess, 'run', side_effect=fake_model), \
                         contextlib.redirect_stdout(io.StringIO()), \
                         self.assertRaises(SystemExit):
                        module.main()


if __name__ == '__main__':
    unittest.main()
