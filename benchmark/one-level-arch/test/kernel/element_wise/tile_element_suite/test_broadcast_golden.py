#!/usr/bin/env python3
"""Check snapshot scope independently of the coordinate-product generator."""
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent


class BroadcastGoldenTest(unittest.TestCase):
    def test_inline_s32_original_and_alias(self):
        source = [(i * 104729 + 0x80003039) & 0xffffffff for i in range(64 * 49)]
        mapping = [row * 49 + column for row in range(64)
                   for repeat in range(8) for column in range(49)]
        for alias in (False, True):
            with self.subTest(alias=alias), tempfile.TemporaryDirectory() as directory:
                out = Path(directory)
                case = "broadcast_inline_s32_" + ("alias_probe" if alias else "original")
                subprocess.run([sys.executable, str(ROOT / 'generate_case.py'), '--case', case, '--out', str(out)], check=True)
                expected = [0x5a5a5a5a] * (len(mapping) + 17)
                if alias:
                    expected[11:11 + len(source)] = source
                sequential = expected.copy()
                for begin in range(0, len(mapping), 512):
                    snapshot = [expected[11 + index] if alias else source[index]
                                for index in mapping[begin:begin + 512]]
                    expected[17 + begin:17 + begin + 512] = snapshot
                actual = list(struct.unpack(f'<{len(expected)}I', (out / 'output.bin').read_bytes()))
                self.assertEqual(actual, expected)
                self.assertEqual((out / 'input.bin').read_bytes(), struct.pack(f'<{len(source) + 11}I', *([0x5a5a5a5a] * 11 + source)))
                self.assertEqual((out / 'output_guard_before.bin').read_bytes(), b'\x5a' * 64)
                self.assertEqual((out / 'output_guard_after.bin').read_bytes(), b'\x5a' * 64)
                if alias:
                    for i, index in enumerate(mapping):
                        sequential[17 + i] = sequential[11 + index]
                    self.assertNotEqual(actual, sequential)
                    self.assertNotEqual(actual[17:], [source[index] for index in mapping])

    def test_original_shapes_and_alias_snapshot_scope(self):
        for rank, count_in, count_out in ((2, 1042, 134418), (3, 62720, 501760), (5, 8320, 58240)):
            for alias in (False, True):
                with self.subTest(rank=rank, alias=alias), tempfile.TemporaryDirectory() as directory:
                    out = Path(directory)
                    case = f"broadcast_f16_{rank}d_" + ("alias_probe" if alias else "original")
                    subprocess.run([sys.executable, str(ROOT / 'generate_case.py'), '--case', case, '--out', str(out)], check=True)
                    initial = [(i * 40503 + 0x35a7) & 65535 for i in range(count_in)]
                    expected = [0x5a5a] * (count_out + 17)
                    if alias:
                        expected[11:11 + count_in] = initial
                    # Three explicit tensor identities, independent of rank/stride decomposition.
                    def source_index(i):
                        if rank == 2:
                            return i // 129
                        if rank == 3:
                            return (i // 392) * 49 + i % 49
                        return i % 8320
                    sequential = expected.copy()
                    whole_call = expected.copy()
                    for begin in range(0, count_out, 2048):
                        end = min(count_out, begin + 2048)
                        snapshot = [(expected[11 + source_index(i)] if alias else initial[source_index(i)]) for i in range(begin, end)]
                        expected[17 + begin:17 + end] = snapshot
                    actual = list(struct.unpack(f'<{count_out + 17}H', (out / 'output.bin').read_bytes()))
                    self.assertEqual(actual, expected)
                    self.assertEqual(list(struct.unpack(f'<{count_in + 11}H', (out / 'input.bin').read_bytes())), [0x5a5a] * 11 + initial)
                    self.assertEqual((out / 'output_guard_before.bin').read_bytes(), b'\x5a' * 64)
                    self.assertEqual((out / 'output_guard_after.bin').read_bytes(), b'\x5a' * 64)
                    manifest = json.loads((out / 'manifest.json').read_text())
                    self.assertEqual(manifest['tile_elements'], 2048)
                    if alias:
                        for i in range(count_out):
                            sequential[17 + i] = sequential[11 + source_index(i)]
                            whole_call[17 + i] = initial[source_index(i)]
                        self.assertNotEqual(actual, sequential, 'must detect writes before full tile snapshot')
                        self.assertNotEqual(actual, whole_call, 'must detect incorrectly hoisted whole-call snapshot')


if __name__ == '__main__':
    unittest.main()
