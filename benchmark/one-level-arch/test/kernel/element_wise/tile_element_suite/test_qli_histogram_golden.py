#!/usr/bin/env python3
import unittest
from qli_histogram_fixture import keys, oracle, profiles


class QLIHistogramGoldenTest(unittest.TestCase):
    def test_selected_bytes_and_cumulative_counts(self):
        source = [0x12345600, 0x123456ff, 0x123456ff, 0x12345700]
        low = oracle(source, 0, 0x12345678)
        self.assertEqual(low[:255], [1] * 255)
        self.assertEqual(low[255], 3)
        second = oracle(source, 1, 0x12345678)
        self.assertEqual(second[0x55:0x58], [0, 3, 4])
        self.assertEqual(oracle(source, 3, 0xffffffff)[0x11:0x13], [0, 4])

    def test_empty_and_prefix_miss(self):
        self.assertEqual(oracle([], 0, 0), [0] * 256)
        self.assertEqual(oracle(keys(128), 0, 0x31415900), [0] * 256)

    def test_alias_clear_before_snapshot_is_detected(self):
        original = keys(2048)
        broken = list(original)
        broken[1:257] = [0] * 256
        for byte in range(4):
            self.assertNotEqual(oracle(original, byte, 0x12345678),
                                oracle(broken, byte, 0x12345678))

    def test_coverage(self):
        cases = list(profiles())
        self.assertEqual(len(cases), 72)
        self.assertEqual({byte for _, _, byte, _ in cases}, set(range(4)))


if __name__ == '__main__':
    unittest.main()
