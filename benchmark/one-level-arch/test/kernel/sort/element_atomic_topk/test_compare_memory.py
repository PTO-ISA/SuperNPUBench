#!/usr/bin/env python3

import importlib.util
import struct
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("compare_memory.py")
SPEC = importlib.util.spec_from_file_location("topk_compare_memory", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class RelaxedAtomicPairTest(unittest.TestCase):
    def setUp(self) -> None:
        self.item = {"comparison": "relaxed_atomic_pair_7_8"}
        self.expected = struct.pack("<128I", 7, 8, *([0] * 126))

    def test_accepts_both_legal_orders(self) -> None:
        self.assertTrue(
            MODULE.matches_segment(self.item, self.expected, self.expected)
        )
        reversed_pair = struct.pack("<128I", 8, 7, *([0] * 126))
        self.assertTrue(
            MODULE.matches_segment(self.item, reversed_pair, self.expected)
        )

    def test_rejects_duplicate_or_inactive_lane_corruption(self) -> None:
        duplicate = struct.pack("<128I", 7, 7, *([0] * 126))
        self.assertFalse(
            MODULE.matches_segment(self.item, duplicate, self.expected)
        )
        corrupt_tail = bytearray(self.expected)
        struct.pack_into("<I", corrupt_tail, 2 * 4, 1)
        self.assertFalse(
            MODULE.matches_segment(self.item, bytes(corrupt_tail), self.expected)
        )


if __name__ == "__main__":
    unittest.main()
