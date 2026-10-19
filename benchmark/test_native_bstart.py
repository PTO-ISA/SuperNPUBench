#!/usr/bin/env python3

import unittest

from native_bstart import BStart, contains, normalize_disassembly, parse


class NativeBStartTest(unittest.TestCase):
    def test_tlsu_old_and_new(self):
        expected = BStart("TLSU", "MGATHER.ADD", "U32")
        self.assertEqual(parse("BSTART.TLSU", "MGATHER.ADD, U32"), expected)
        self.assertEqual(parse("BSTART.MGATHER.ADD", "U32"), expected)
        self.assertTrue(contains("BSTART.TLSU MGATHER.ADD, U32",
                                 "TLSU", "MGATHER.ADD", "U32"))
        self.assertTrue(contains("BSTART.MGATHER.ADD U32",
                                 "TLSU", "MGATHER.ADD", "U32"))

    def test_cube_old_and_new(self):
        expected = BStart("CUBE", "TMATMULMX.ACC", "FP32")
        self.assertEqual(parse("BSTART.CUBE", "TMATMULMX.ACC, FP32"), expected)
        self.assertEqual(parse("BSTART.TMATMULMX.ACC", "FP32"), expected)
        self.assertTrue(contains("BSTART.CUBE TMATMULMX.ACC, FP32",
                                 "CUBE", "TMATMULMX.ACC", "FP32"))
        self.assertTrue(contains("BSTART.TMATMULMX.ACC FP32",
                                 "CUBE", "TMATMULMX.ACC", "FP32"))

    def test_near_matches_rejected(self):
        for text in ("BSTART.TLSU TSTORE, FP32", "BSTART.TLOAD FP16",
                     "BSTART.CUBE TLOAD, FP32", "BSTART.TLOADX FP32"):
            self.assertFalse(contains(text, "TLSU", "TLOAD", "FP32"), text)
        self.assertIsNone(parse("BSTART.TLSU", "TLOADX, FP32"))
        self.assertIsNone(parse("BSTART.TMATMUL", "FP32, EXTRA"))
        self.assertIsNone(parse("BSTART.CUBE", "TLOAD, FP32"))
        self.assertFalse(contains("BSTART.TLOAD FP16", "TLSU", "TLOAD", "FP32"))

    def test_disassembly_canaries_preserve_scope(self):
        old = " 100: 00000000 BSTART.TLSU TLOAD, FP32\n"
        new = " 100: 00000000 BSTART.TLOAD FP32\n"
        expected = old
        self.assertEqual(normalize_disassembly(old), expected)
        self.assertEqual(normalize_disassembly(new), expected)
        unrelated = " 104: 00000000 BSTART.TLOADX FP32\n"
        self.assertEqual(normalize_disassembly(unrelated), unrelated)
        self.assertEqual(normalize_disassembly("\tBSTART.TLOAD FP32\n"),
                         "\tBSTART.TLSU TLOAD, FP32\n")
        masked = (" 100: 00000000 BSTART.TLOAD FP32\n"
                  " 104: 00000000 B.IOT mask=1000, ->t<128B>\n")
        self.assertIn("B.IOT mask=1000", normalize_disassembly(masked))

    def test_timg2col_numeric_old_and_named_new(self):
        expected = " 100: 00000000 BSTART.TLSU TIMG2COL, FP32\n"
        for source in (" 100: 00000000 BSTART.TLSU 28, FP32\n",
                       " 100: 00000000 BSTART.TIMG2COL FP32\n"):
            self.assertEqual(normalize_disassembly(source), expected)


if __name__ == "__main__":
    unittest.main()
