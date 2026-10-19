#!/usr/bin/env python3
"""Sparse host oracle keeps original allocations and per-MxN alias snapshots."""
import mmap
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent
CONFIGS = [(200000, 202, 97, 128, 128), (875000, 8192, 1, 128, 128),
           (754, 640, 17, 128, 128), (131072, 32, 256, 32, 64)]
EDGE = [0, 0x80000000, 0x7f800000, 0xff800000, 0x7fc12345, 0x7f800123, 1, 0x80000001]


def initial_value(i):
    return EDGE[i % 32] if i % 32 < 8 else (i * 104729 + 0x80003039) & 0xffffffff


class GatherGoldenTest(unittest.TestCase):
    def test_full_profiles_and_evolving_alias_snapshots(self):
        for profile, (k, m, n, tm, tn) in enumerate(CONFIGS):
            indices = [k - 1 if i % 17 == 0 else (i * 7 + 3) % 64 for i in range(m)]
            initial = {11 + row * n + col: initial_value(row * n + col)
                       for row in indices for col in range(n)}
            initial.update({i: 0x5a5a5a5a for i in range(11)})
            for alias in (False, True):
                with self.subTest(profile=profile, alias=alias), tempfile.TemporaryDirectory() as directory:
                    out = Path(directory)
                    name = f'gather_f32_{profile}_' + ('alias_probe' if alias else 'original')
                    subprocess.run([sys.executable, str(ROOT / 'generate_case.py'), '--case', name, '--out', str(out)], check=True)
                    memory = initial.copy()
                    output = [0x5a5a5a5a] * (m * n + 17)
                    for row_start in range(0, m, tm):
                        for col_start in range(0, n, tn):
                            pending = {}
                            for row in range(row_start, min(m, row_start + tm)):
                                for col in range(col_start, min(n, col_start + tn)):
                                    pending[17 + row * n + col] = memory.get(11 + indices[row] * n + col, 0)
                            if alias:
                                memory.update(pending)
                            else:
                                for address, value in pending.items():
                                    output[address] = value
                    self.assertEqual((out / 'input.bin').stat().st_size, 4 * (k * n + 11))
                    self.assertEqual((out / 'output.bin').read_bytes(), struct.pack(f'<{len(output)}I', *output))
                    self.assertEqual((out / 'indices.bin').read_bytes(), struct.pack(f'<{m + 11}I', *([0x5a5a5a5a] * 11 + indices)))
                    with (out / 'input.bin').open('rb') as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as observed:
                        for address, value in memory.items():
                            self.assertEqual(struct.unpack_from('<I', observed, 4 * address)[0], value)
                        for address in range(11, k * n + 11, max(1, k * n // 101)):
                            self.assertEqual(struct.unpack_from('<I', observed, 4 * address)[0], memory.get(address, 0))
                    if alias:
                        whole_call = [initial.get(11 + indices[row] * n + col, 0)
                                      for row in range(m) for col in range(n)]
                        self.assertNotEqual([memory.get(17 + i, 0) for i in range(m * n)], whole_call,
                                            'fixture must expose an incorrectly hoisted full-call snapshot')


    def test_index_output_alias_reloads_each_column_tile(self):
        k, m, n, tm, tn = CONFIGS[3]
        initial_indices = [k - 1 if row % 17 == 0 else (row * 7 + 3) % 64 for row in range(m)]
        source = {11 + row * n + col: ((row * n + col) * 13 + 5) % k
                  for row in initial_indices for col in range(n)}
        backing = [0x5a5a5a5a] * (m * n + 17)
        backing[11:11 + m] = initial_indices
        hoisted = backing.copy()
        for start in range(0, n, tn):
            picked = backing[11:11 + m]
            pending = {(17 + row * n + col): source.get(11 + picked[row] * n + col, 0)
                       for row in range(m) for col in range(start, start + tn)}
            for address, value in pending.items():
                backing[address] = value
            for row in range(m):
                for col in range(start, start + tn):
                    hoisted[17 + row * n + col] = source.get(11 + initial_indices[row] * n + col, 0)
        self.assertNotEqual(backing, hoisted, "must expose index snapshot hoisted across N tiles")
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            subprocess.run([sys.executable, str(ROOT / 'generate_case.py'), '--case',
                            'gather_f32_3_index_alias_probe', '--out', str(out)], check=True)
            self.assertEqual((out / 'indices.bin').read_bytes(), struct.pack(f'<{len(backing)}I', *backing))
            self.assertEqual((out / 'input.bin').stat().st_size, 4 * (k * n + 11))
            self.assertEqual((out / 'output.bin').read_bytes(), struct.pack(f'<{m * n + 17}I', *([0x5a5a5a5a] * (m * n + 17))))


if __name__ == '__main__':
    unittest.main()
