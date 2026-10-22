#!/usr/bin/env python3
import unittest
from qli_norm_fixture import expected, profiles


class QLINormGoldenTest(unittest.TestCase):
    def test_unsigned_high_bit_and_ties(self):
        result = expected(128, 0)
        self.assertEqual(result['gt_count'][:3], [0, 1, 0])
        self.assertEqual(result['eq_count'][:3], [1, 0, 0])
        self.assertEqual(result['eq_value'][31:33], [0x80000000, 0x80000000])
        self.assertEqual(result['eq_cleared'][0], 0)
        self.assertEqual(result['eq_cleared'][31:33], [0x80000000, 0x80000000])
        self.assertEqual(result['sums'][1], 4)

    def test_outside_valid_domain_does_not_clear(self):
        result = expected(32, 32)
        self.assertEqual(result['gt_value'], result['gt_cleared'])
        self.assertEqual(result['eq_value'], result['eq_cleared'])
        self.assertEqual(len(list(profiles())), 13)


if __name__ == '__main__':
    unittest.main()
