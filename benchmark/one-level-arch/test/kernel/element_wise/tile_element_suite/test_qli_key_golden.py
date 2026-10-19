#!/usr/bin/env python3
"""Check independent key ordering and detect premature per-column writes."""
import unittest
from qli_key_fixture import BITS, expected, inputs, ordered_bits, profiles


class QLIKeyGoldenTest(unittest.TestCase):
    def test_special_bits(self):
        self.assertEqual(list(map(ordered_bits, BITS[:9])), [
            0xbf800001, 0x80000000, 0x7fffffff, 0xff800000,
            0x007fffff, 0xffc12345, 0x003edcba, 0xff812345, 0x007edcba])

    def test_zero_length_untouched(self):
        for alias in ('normal', 'exact', 'shifted'):
            self.assertEqual(expected(0, alias), (inputs(0, alias), [0xdeadbeef]))

    def test_source_extent(self):
        for _, valid, alias in profiles():
            self.assertEqual(len(inputs(valid, alias)),
                             max(1, valid + (alias == 'shifted')))

    def test_shifted_alias_rejects_premature_stores(self):
        for capacity, valid in ((128, 33), (128, 128), (2048, 128), (2048, 2048)):
            with self.subTest(capacity=capacity, valid=valid):
                premature = inputs(valid, 'shifted')
                parts = capacity // 32
                for part in range(parts):
                    indices = list(range(part, valid, parts))
                    leaf = [ordered_bits(premature[i]) for i in indices]
                    for i, value in zip(indices, leaf):
                        premature[i + 1] = value
                self.assertNotEqual(premature, expected(valid, 'shifted')[0])


if __name__ == '__main__':
    unittest.main()
