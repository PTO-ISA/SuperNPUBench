#!/usr/bin/env python3

import unittest

from report_fixp import parse_bundle


OLD = [
    "100: 00000000 BSTART.CUBE TMATMUL, FP32",
    "104: 00002023 B.FPATR 0, 0, 0, 0, 0, 0, 0, 0, 0, 0",
    "108: 00000000 B.IOT t#1, mask=1111",
]
NEW = [
    "100: 00000000 BSTART.TMATMUL FP32",
    "104: 00002023 B.FPATR 0, 0, 0, 0, 0, 0, 0, 0, 0, 0",
    "108: 00000000 B.IOT t#1, mask=1111",
]


class FixpNativeHeaderTest(unittest.TestCase):
    def test_old_and_new_headers_decode_identically(self):
        self.assertEqual(parse_bundle(OLD), parse_bundle(NEW))
        self.assertEqual(parse_bundle(NEW)[0], "TMATMUL")


if __name__ == "__main__":
    unittest.main()
