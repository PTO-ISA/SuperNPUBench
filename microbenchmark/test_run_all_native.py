#!/usr/bin/env python3

import unittest

from run_all import canonical_dtype, canonical_operation


class MicrobenchmarkNativeInventoryTest(unittest.TestCase):
    def test_signed_inventory_dtypes(self):
        self.assertEqual(canonical_dtype("i32"), "S32")
        self.assertEqual(canonical_dtype("i8"), "S8")
        self.assertEqual(canonical_dtype("fp32"), "FP32")

    def test_unknown_dtype_is_not_widened(self):
        with self.assertRaises(ValueError):
            canonical_dtype("fp32.bad")

    def test_mx_operation_spelling(self):
        self.assertEqual(canonical_operation("TMATMUL_MX_ACC"),
                         "TMATMULMX.ACC")


if __name__ == "__main__":
    unittest.main()
