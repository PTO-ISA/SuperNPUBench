#!/usr/bin/env python3
"""Linked test inputs must not preload the expected concatenation result."""
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parent


class ConcatGatherF16SeedTest(unittest.TestCase):
    def test_full_capacity_initial_state_and_alias_snapshot(self):
        for alias in (False, True):
            with self.subTest(alias=alias), tempfile.TemporaryDirectory() as work:
                out = Path(work)
                name = 'concat_gather_f16_' + ('alias_probe' if alias else 'original')
                subprocess.run([sys.executable, str(ROOT / 'generate_case.py'),
                                '--case', name, '--out', str(out)], check=True)
                seed = (out / 'input_seed_u16.bin').read_bytes()
                result = (out / 'input_u16.bin').read_bytes()
                self.assertEqual(len(seed), 8192000 * 2)
                for index in (0, 31, 2048, 2049, 2559, 2561, 65535, 65536,
                              2047999, 2048000, 8191999):
                    self.assertEqual(struct.unpack_from('<H', seed, index * 2)[0],
                                     (index * 40503 + 0x35a7) & 65535)
                if alias:
                    self.assertNotEqual(seed[2049 * 2:2561 * 2], result[2049 * 2:2561 * 2])
                    self.assertEqual(seed[:2049 * 2], result[:2049 * 2])
                    self.assertEqual(seed[2561 * 2:], result[2561 * 2:])
                else:
                    self.assertEqual(seed, result)
                count = 512 if alias else 2048000
                output_seed = (out / 'output_seed_u16.bin').read_bytes()
                self.assertEqual(output_seed, b'\x5a\x5a' * count)
                if not alias:
                    self.assertNotEqual(output_seed, (out / 'output_u16.bin').read_bytes())
                inc = (out / 'concat_gather_f16_input.inc').read_text()
                self.assertIn('input_seed_u16.bin', inc)
                self.assertIn('output_seed_u16.bin', inc)
                self.assertNotIn('input_u16.bin"', inc)
                self.assertNotIn('output_u16.bin"', inc)
                self.assertNotIn('status_u32.bin', inc)

    def test_scatter_seeds_preserve_full_input_before_evolving_alias(self):
        for alias in (False, True):
            with self.subTest(alias=alias), tempfile.TemporaryDirectory() as work:
                out = Path(work)
                name = 'concat_scatter_f16_' + ('alias_probe' if alias else 'original')
                subprocess.run([sys.executable, str(ROOT / 'generate_case.py'),
                                '--case', name, '--out', str(out)], check=True)
                seed = (out / 'input_seed.bin').read_bytes()
                result = (out / 'input.bin').read_bytes()
                self.assertEqual(len(seed), 8192000 * 2)
                for index in (0, 508, 509, 512, 1023, 2048000, 8191999):
                    self.assertEqual(struct.unpack_from('<H', seed, index * 2)[0],
                                     (index * 40503 + 0x35a7) & 65535)
                if alias:
                    self.assertNotEqual(seed[509 * 2:517 * 2], result[509 * 2:517 * 2])
                    self.assertEqual(seed[:509 * 2], result[:509 * 2])
                else:
                    self.assertEqual(seed, result)
                self.assertEqual((out / 'output_seed.bin').read_bytes(), b'\x5a\x5a' * 2048000)
                inc = (out / 'concat_scatter_f16_input.inc').read_text()
                self.assertIn('input_seed.bin', inc)
                self.assertIn('output_seed.bin', inc)
                self.assertNotIn('/input.bin"', inc)
                self.assertNotIn('/output.bin"', inc)
                self.assertNotIn('status.bin', inc)


if __name__ == '__main__':
    unittest.main()
