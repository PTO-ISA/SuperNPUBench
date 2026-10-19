#!/usr/bin/env python3
"""Independent byte-oriented Murmur3 and fixed-probe hashtable oracle."""
import os
from pathlib import Path
import random
import shutil
import struct
from functools import lru_cache
import subprocess
import tempfile
import unittest

MASK32 = (1 << 32) - 1
MASK64 = (1 << 64) - 1
CAPACITY = 2550000
QUERY_COUNT = 6144
MAX_PROBE = 512


def rotate_left(value, amount):
    bits = f'{value & MASK32:032b}'
    return int(bits[amount:] + bits[:amount], 2)


def murmur3_key(key):
    """Reference consumes little-endian bytes, independent of the C++ split."""
    message = (key & MASK64).to_bytes(8, 'little')
    result = 0
    for offset in range(0, len(message), 4):
        word = int.from_bytes(message[offset:offset + 4], 'little')
        word = (word * 0xcc9e2d51) & MASK32
        word = rotate_left(word, 15)
        word = (word * 0x1b873593) & MASK32
        result ^= word
        result = (rotate_left(result, 13) * 5 + 0xe6546b64) & MASK32
    result ^= len(message)
    for shift, multiplier in ((16, 0x85ebca6b), (13, 0xc2b2ae35)):
        result ^= result >> shift
        result = (result * multiplier) & MASK32
    return result ^ (result >> 16)


def lookup(table, key, probes=MAX_PROBE):
    start = murmur3_key(key) % len(table)
    value = -1
    for probe in range(probes):
        candidate, candidate_value = table[(start + probe) % len(table)]
        if (candidate & MASK64) == (key & MASK64):
            value = candidate_value
    return value


class HashTableOracleTest(unittest.TestCase):
    def test_known_vectors(self):
        self.assertEqual(murmur3_key(0), 0x63852afc)
        self.assertEqual(murmur3_key(1), 0x53075d44)

    def test_last_duplicate_and_exact_probe_boundary(self):
        key = 0x1122334455667788
        table = [(key ^ 1, 99)] * 1021
        start = murmur3_key(key) % len(table)
        table[(start + 3) % len(table)] = (key, 41)
        table[(start + 511) % len(table)] = (key, -27)
        table[(start + 512) % len(table)] = (key, 63)
        self.assertEqual(lookup(table, key), -27)
        self.assertEqual(lookup(table, key, 511), 41)
        self.assertEqual(lookup(table, key, 513), 63)

    def test_wrap_and_absent(self):
        key = next(key for key in range(10000) if murmur3_key(key) % 1031 > 1010)
        table = [(-777, 0)] * 1031
        table[4] = (key, 123)
        self.assertEqual(lookup(table, key), 123)
        self.assertEqual(lookup(table, key ^ (1 << 63)), -1)

    def test_high_half_is_part_of_equality_and_hash(self):
        key = 0x1122334455667788
        other = key ^ (1 << 40)
        table = [(other, 17)] * 1031
        self.assertEqual(lookup(table, key), -1)
        self.assertNotEqual(murmur3_key(key), murmur3_key(other))

    def test_query_alias_requires_full_original_batch_snapshot(self):
        from hashtable_fixture import fixture, ENTRY
        table, queries, outputs = fixture()
        expected = bytearray(queries)
        expected[4:4 + len(outputs)] = outputs

        @lru_cache(None)
        def find(key):
            position = murmur3_key(key) % CAPACITY
            value = -1
            for probe in range(MAX_PROBE):
                candidate, result, _ = ENTRY.unpack_from(
                    table, ((position + probe) % CAPACITY) * ENTRY.size)
                if candidate == key:
                    value = result
            return value

        for count in (256, 512, 1024):
            correct = bytearray(queries)
            premature = bytearray(queries)
            for base in range(0, QUERY_COUNT, count):
                batch = struct.unpack_from(f'<{count}Q', correct, base * 8)
                values = [find(key) for key in batch]
                struct.pack_into(f'<{count}i', correct, 4 + base * 4, *values)
                # Negative control: store each column before reading the next.
                # Those stores overwrite later query values in the same batch.
                for column in range(count // 32):
                    indices = [base + row * (count // 32) + column for row in range(32)]
                    leaf = [struct.unpack_from('<Q', premature, index * 8)[0]
                            for index in indices]
                    for index, key in zip(indices, leaf):
                        struct.pack_into('<i', premature, 4 + index * 4, find(key))
            self.assertEqual(correct, expected)
            self.assertNotEqual(premature, expected)

    def test_cpp_hash_matches_independent_byte_oracle(self):
        root = Path(__file__).resolve().parents[4]
        header = (root / 'kernels/basic_op/control/hashtable_lookup_simd.hpp').read_text()
        start = header.index('inline uint32_t hashTableKey(')
        end = header.index('\ntemplate <', start)
        scalar_hash = header[start:end]
        compiler = os.environ.get('CXX') or shutil.which('c++')
        self.assertIsNotNone(compiler)
        rng = random.Random(0x50544f)
        keys = [0, 1, MASK64, 1 << 63, 0xffffffff00000000,
                0xffffffff, 0x1122334455667788, 0x5566778811223344]
        keys += [rng.getrandbits(64) for _ in range(128)]
        with tempfile.TemporaryDirectory(prefix='pto-hash-oracle-') as directory:
            source = Path(directory) / 'hash.cpp'
            binary = Path(directory) / 'hash'
            source.write_text(
                '#include <cstdint>\n#include <cstdio>\n#include <bit>\n' +
                scalar_hash + '\nint main(){\n' +
                ''.join(f'printf("%u\\n",hashTableKey(std::bit_cast<int64_t>'
                        f'(UINT64_C({key}))));\n' for key in keys) + '}\n')
            compiled = subprocess.run([compiler, '-std=c++20',
                            str(source), '-o', str(binary)],
                           capture_output=True, text=True)
            self.assertEqual(compiled.returncode, 0, compiled.stderr)
            actual = subprocess.check_output([str(binary)], text=True)
        self.assertEqual([int(line) for line in actual.splitlines()],
                         [murmur3_key(key) for key in keys])


if __name__ == '__main__':
    unittest.main()
