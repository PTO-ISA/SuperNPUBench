#!/usr/bin/env python3
import struct
import unittest
from qli_popn_fixture import CHUNK_BASE, compare_indices, compare_selection, profiles, values


class QLIPopNGoldenTest(unittest.TestCase):
    def test_ties_and_valid_boundaries(self):
        for _, valid, ties, _ in profiles():
            source, positions = values(valid, ties)
            self.assertEqual(positions, sorted(set(positions)))
            self.assertEqual({i for i, x in enumerate(source) if x}, set(positions))
            self.assertIn(0, positions)
            self.assertIn(valid - 1, positions)
            if ties:
                self.assertEqual({source[i] for i in positions}, {0xffffffff})

    def test_unordered_unique_indices(self):
        pack = lambda words: struct.pack(f'<{len(words)}i', *words)
        wanted = [CHUNK_BASE + i for i in (0, 31, 32, 127)]
        compare_indices(pack(wanted[::-1]), pack(wanted))
        for wrong in ([wanted[0]] * 4, wanted[:3] + [CHUNK_BASE + 128]):
            with self.assertRaises(AssertionError):
                compare_indices(pack(wrong), pack(wanted))


    def test_counts_and_observable_inout_updates(self):
        self.assertEqual(len(list(profiles())), 18)
        for _, valid, ties, count in profiles():
            source, positions = values(valid, ties)
            selected = sorted(positions, key=lambda i: source[i], reverse=True)[:count]
            remaining = list(source)
            for i in selected:
                remaining[i] = 0
            indices = [CHUNK_BASE + i for i in selected] if count else [0xdeadbeef]
            pack = lambda words: struct.pack(f'<{len(words)}I', *words)
            compare_selection(pack(indices), pack(remaining), valid, ties, count)
            if count:
                with self.assertRaises(AssertionError):
                    compare_selection(pack(indices), pack(source), valid, ties, count)


if __name__ == '__main__':
    unittest.main()
