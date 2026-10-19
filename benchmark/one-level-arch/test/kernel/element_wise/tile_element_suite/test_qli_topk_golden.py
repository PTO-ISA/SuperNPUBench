#!/usr/bin/env python3
import struct
import unittest
from qli_topk_fixture import compare_storage, profiles, scores, storage_golden


class QLITopKGoldenTest(unittest.TestCase):
    def test_original_dimensions_and_positive_unique_scores(self):
        self.assertEqual(profiles()[:4], ((64, 128, 128), (4, 2048, 512),
                                      (4, 8192, 512), (4, 2080, 512)))
        for sq, skv, k in profiles()[:4]:
            bits = scores(sq, skv)
            for row in range(sq):
                values = bits[row * skv:(row + 1) * skv]
                self.assertEqual(len(set(values)), skv)
                self.assertTrue(all(0x3f000000 <= value < 0x3f800000 for value in values))
            self.assertEqual(len(storage_golden(sq, skv, k)),
                             sq * k + 2048 + sq * skv + 256 + 32)

    def test_unordered_indices_accepted(self):
        expected = storage_golden(1, 128, 8)
        actual = expected[:8][::-1] + expected[8:]
        pack = lambda words: struct.pack(f'<{len(words)}I', *words)
        compare_storage(pack(actual), pack(expected), 1, 128, 8)

    def test_duplicate_index_and_scratch_corruption_rejected(self):
        expected = storage_golden(1, 128, 8)
        pack = lambda words: struct.pack(f'<{len(words)}I', *words)
        for changed in (1, 8, len(expected) - 1):
            actual = list(expected)
            actual[changed] = actual[0] if changed == 1 else actual[changed] ^ 1
            with self.subTest(changed=changed), self.assertRaises(AssertionError):
                compare_storage(pack(actual), pack(expected), 1, 128, 8)

    def test_special_values_zero_key_and_any_tied_cutoff(self):
        sq, skv, k = 2, 128, 73
        expected = storage_golden(sq, skv, k)
        actual = list(expected)
        keys = expected[sq * k + 2048:sq * k + 2048 + sq * skv]
        self.assertIn(0, keys)
        for row in range(sq):
            row_keys = keys[row * skv:(row + 1) * skv]
            selected = expected[row * k:(row + 1) * k]
            threshold = min(row_keys[index] for index in selected)
            strict = [i for i, value in enumerate(row_keys) if value > threshold]
            equal = [i for i, value in enumerate(row_keys) if value == threshold]
            self.assertGreater(len(equal), k - len(strict))
            actual[row * k:(row + 1) * k] = strict + equal[::-1][:k - len(strict)]
        pack = lambda words: struct.pack(f'<{len(words)}I', *words)
        compare_storage(pack(actual), pack(expected), sq, skv, k)
        actual[0] = next(i for i, value in enumerate(keys[:skv]) if value == 0)
        with self.assertRaises(AssertionError):
            compare_storage(pack(actual), pack(expected), sq, skv, k)

    def test_all_indices_include_valid_zero_key(self):
        expected = storage_golden(1, 128, 128)
        self.assertEqual(set(expected[:128]), set(range(128)))
        self.assertIn(0, expected[128 + 2048:128 + 2048 + 128])


if __name__ == '__main__':
    unittest.main()
