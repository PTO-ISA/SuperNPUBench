#!/usr/bin/env python3
import struct
from pathlib import Path
import tempfile
import unittest

from qli_dynamic_topk_fixture import (CASES, PAD_SCORE, POISON, compare,
                                      generate, layout, oracle, sortable_key)


class DynamicTopKFixtureTest(unittest.TestCase):
    def test_raw_sortable_total_order_endpoints(self):
        self.assertEqual(sortable_key(0xFFFFFFFF), 0)
        self.assertEqual(sortable_key(0x80000000), 0x7FFFFFFF)
        self.assertEqual(sortable_key(0), 0x80000000)
        self.assertEqual(sortable_key(0x7FFFFFFF), 0xFFFFFFFF)

    def test_all_valid_zero_keys_exclude_high_key_padding(self):
        case = CASES[0]
        initial, expected, rows, info = oracle(case)
        self.assertEqual(set(initial[:case.skv]), {0xFFFFFFFF})
        self.assertEqual(set(initial[case.skv:info["stride"]]), {PAD_SCORE})
        key_row = expected[info["key_start"]:info["key_start"] + info["stride"]]
        self.assertEqual(set(key_row[:case.skv]), {0})
        self.assertEqual(set(key_row[case.skv:]), {0})
        self.assertEqual(rows[0]["quota"], case.topk)
        actual = list(expected)
        actual[info["index_start"]:info["index_start"] + case.topk] = range(
            case.topk - 1, -1, -1)
        raw = struct.pack(f"<{len(actual)}I", *actual)
        compare(raw, struct.pack(f"<{len(expected)}I", *expected), case)

    def test_mixed_tied_cutoff_accepts_any_exact_quota(self):
        case = CASES[1]
        _initial, expected, rows, info = oracle(case)
        actual = list(expected)
        for row, contract in enumerate(rows):
            self.assertGreater(len(contract["equal"]), contract["quota"])
            chosen = sorted(contract["strict"]) + sorted(
                contract["equal"], reverse=True)[:contract["quota"]]
            begin = info["index_start"] + row * case.topk
            actual[begin:begin + case.topk] = chosen
        pack = lambda words: struct.pack(f"<{len(words)}I", *words)
        compare(pack(actual), pack(expected), case)
        actual[info["index_start"]] = case.skv
        with self.assertRaisesRegex(ValueError, "outside runtime Skv"):
            compare(pack(actual), pack(expected), case)

    def test_topk_and_sq_zero_are_complete_noops(self):
        for case in CASES[2:]:
            initial, expected, rows, _info = oracle(case)
            self.assertEqual(initial, expected)
            self.assertEqual(rows, [])
            raw = struct.pack(f"<{len(initial)}I", *initial)
            compare(raw, raw, case)
            changed = list(initial)
            changed[-1] = POISON ^ 1
            with self.assertRaisesRegex(ValueError, "no-op mutated"):
                compare(struct.pack(f"<{len(changed)}I", *changed), raw, case)

    def test_original_reserved_gap_is_preserved(self):
        for case in CASES:
            info = layout(case)
            self.assertEqual(info["gap_words"],
                             2048 + case.sq * info["stride"] + 512)

    def test_elf_seed_contains_only_initial_workspace_and_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "fixture"
            generate(root, CASES[1])
            assembly = (root / "data.s").read_text()
            self.assertIn("workspace.initial.bin", assembly)
            self.assertNotIn("workspace.expected.bin", assembly)
            self.assertIn(".fill 64,1,0xa5", assembly)
            self.assertIn(".fill 64,1,0x3c", assembly)


if __name__ == "__main__":
    unittest.main()
