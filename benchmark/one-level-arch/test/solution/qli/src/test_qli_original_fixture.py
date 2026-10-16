#!/usr/bin/env python3
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np
import qli_original_fixture as fixture


class OriginalQLIFixtureTest(unittest.TestCase):
    def test_all_original_shapes_and_exact_formula(self):
        self.assertEqual(fixture.profiles(), (
            ('static', 64, 128, 128, 1), ('static', 4, 2048, 512, 1),
            ('static', 4, 8192, 512, 1), ('static', 4, 2080, 512, 1),
            ('dynamic', 4, 128, 8, 1)))
        for _, sq, skv, _, _ in fixture.profiles():
            inputs = fixture.make_inputs(sq, skv)
            self.assertEqual(len(inputs['srcq']), sq * 64 * 128)
            self.assertEqual(len(inputs['srck']), skv * 128)
            scores = fixture.score_oracle(inputs, sq, skv)
            x = np.arange(1, skv + 1)
            for row in range(sq):
                expected = ((10 * x if row % 2 == 0 else 8 * x + 65535) *
                            2.0 ** -21).astype('<f4')
                np.testing.assert_array_equal(scores[row], expected)

    def test_complete_workspace_and_unordered_indices(self):
        for profile in fixture.profiles():
            with self.subTest(profile=profile), tempfile.TemporaryDirectory() as work:
                directory = Path(work) / 'fixture'
                fixture.generate(directory, profile)
                expected = (directory / 'qli_workspace.golden.bin').read_bytes()
                fixture.compare_workspace(expected, expected, profile)
                _, sq, skv, topk, _ = profile
                stride = skv if profile[0] == 'static' else (skv + 2047) // 2048 * 2048
                begin = sq * stride * 4
                reordered = bytearray(expected)
                row = struct.unpack(f'<{topk}I', reordered[begin:begin + topk * 4])
                reordered[begin:begin + topk * 4] = struct.pack(f'<{topk}I', *row[::-1])
                fixture.compare_workspace(reordered, expected, profile)
                corruptions = [expected[:-4], expected + b'\0\0\0\0']
                for offset in (0, sq * stride * 4 + sq * topk * 4, len(expected) - 1):
                    bad = bytearray(expected)
                    bad[offset] ^= 1
                    corruptions.append(bad)
                duplicate = bytearray(expected)
                duplicate[begin + 4:begin + 8] = duplicate[begin:begin + 4]
                corruptions.append(duplicate)
                for actual in corruptions:
                    with self.assertRaises(ValueError):
                        fixture.compare_workspace(actual, expected, profile)


if __name__ == '__main__':
    unittest.main()
