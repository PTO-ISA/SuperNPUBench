#!/usr/bin/env python3
"""Original transpose extents and alias-sensitive per-tile oracle checks."""
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent


class TransposeGoldenTest(unittest.TestCase):
    def test_original_profiles_keep_full_buffers(self):
        for profile, count in enumerate((1476 * 32, 8 * 4096 * 3, 32 * 8 * 32, 8 * 64 * 4 * 16 * 7)):
            with self.subTest(profile=profile), tempfile.TemporaryDirectory() as directory:
                out = Path(directory)
                subprocess.run([sys.executable, str(ROOT / 'generate_case.py'), '--case', f'transpose_f16_{profile}_original', '--out', str(out)], check=True)
                source = (out / 'input.bin').read_bytes()
                actual = (out / 'output.bin').read_bytes()
                self.assertEqual(len(source), 2 * count)
                self.assertEqual(len(actual), 2 * count)
                self.assertEqual(len((out / 'in_shape.bin').read_bytes()), 128 * 4)
                self.assertEqual(len((out / 'out_shape.bin').read_bytes()), 128 * 4)
                self.assertEqual(sorted(struct.unpack(f'<{count}H', source)), sorted(struct.unpack(f'<{count}H', actual)))
                if profile == 0:
                    values = struct.unpack(f'<{count}H', source)
                    expected = [values[row * 32 + col] for col in range(32) for row in range(1476)]
                    self.assertEqual(list(struct.unpack(f'<{count}H', actual)), expected)

    def test_alias_requires_full_tile_but_not_whole_call_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            subprocess.run([sys.executable, str(ROOT / 'generate_case.py'), '--case', 'transpose_f16_0_alias_probe', '--out', str(out)], check=True)
            count = 1476 * 32
            initial = [(i * 40503 + 0x35a7) & 65535 for i in range(count)]
            expected, sequential = initial.copy(), initial.copy()
            for begin in range(0, count, 512):
                end = min(count, begin + 512)
                snapshot = [expected[(i % 1476) * 32 + i // 1476] for i in range(begin, end)]
                expected[begin:end] = snapshot
            for i in range(count):
                sequential[i] = sequential[(i % 1476) * 32 + i // 1476]
            whole_call = [initial[(i % 1476) * 32 + i // 1476] for i in range(count)]
            actual = list(struct.unpack(f'<{count}H', (out / 'input.bin').read_bytes()))
            self.assertEqual(actual, expected)
            self.assertNotEqual(actual, sequential)
            self.assertNotEqual(actual, whole_call)
            self.assertEqual((out / 'output.bin').read_bytes(), b'\x5a' * (2 * count))
            for name in ('input_guard_before', 'input_guard_after', 'output_guard_before', 'output_guard_after'):
                self.assertEqual((out / (name + '.bin')).read_bytes(), b'\x5a' * 64)


if __name__ == '__main__':
    unittest.main()
