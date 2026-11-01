#!/usr/bin/env python3
"""User's 70%-loaded dataset; golden_data is hash slots, lookup_golden is values.

Seed 0 intentionally preserves the supplied script's unseeded RNG behavior.
Saved initial keys/values allow exact replay with --reuse-init.
"""
import argparse
import hashlib
import json
import struct
from pathlib import Path
import numpy as np

EMPTY_KEY = 0xFFFFFFFFFFFFFFFF
ENTRY_FORMAT = '<Qi4x'
ENTRY_SIZE = struct.calcsize(ENTRY_FORMAT)
ENTRY_DTYPE = np.dtype([('key', '<u8'), ('value', '<i4'), ('padding', '<u4')])


def generate_key_bin(filename, src_keys, elements, seed=0):
    if seed:
        np.random.seed(seed)
    keys = np.random.choice(src_keys, size=elements, replace=False)
    keys.astype('<u8').tofile(filename)
    return keys


def generate_init_keys(elements, seed=0):
    if seed:
        np.random.seed(seed)
    return np.random.randint(0, 2**64, size=elements, dtype=np.uint64)


def generate_init_vals(elements, seed=0):
    if seed:
        np.random.seed(seed)
    return np.random.randint(-2**31, 2**31-1, size=elements, dtype=np.int32)


def murmurhash3(key_val, seed=0):
    # Preserve supplied algorithm: initial hash=0 (seed argument unused).
    data = struct.pack('<Q', int(key_val))
    h = 0
    for i in range(0, 8, 4):
        k = struct.unpack('<I', data[i:i+4])[0]
        k = (k * 0xcc9e2d51) & 0xFFFFFFFF
        k = ((k << 15) | (k >> 17)) & 0xFFFFFFFF
        k = (k * 0x1b873593) & 0xFFFFFFFF
        h ^= k
        h = ((h << 13) | (h >> 19)) & 0xFFFFFFFF
        h = (h * 5 + 0xe6546b64) & 0xFFFFFFFF
    h ^= 8
    h ^= h >> 16
    h = (h * 0x85ebca6b) & 0xFFFFFFFF
    h ^= h >> 13
    h = (h * 0xc2b2ae35) & 0xFFFFFFFF
    h ^= h >> 16
    return h


def generate_table_bin(filename, table_size, input_keys, input_vals):
    if len(input_keys) != len(input_vals) or len(input_keys) >= table_size:
        raise ValueError('Require matching inputs and at least one empty slot')
    if np.any(input_keys == EMPTY_KEY):
        raise ValueError('Generated key collides with reserved EMPTY_KEY')
    table = np.zeros(table_size, dtype=ENTRY_DTYPE)
    table['key'] = EMPTY_KEY
    for idx, (key, val) in enumerate(zip(input_keys, input_vals)):
        slot = murmurhash3(key) % table_size
        while int(table['key'][slot]) not in (EMPTY_KEY, int(key)):
            slot = (slot + 1) % table_size
        table[slot] = (key, val, 0)
        if (idx + 1) % 500000 == 0:
            print(f'Inserted {idx + 1}/{len(input_keys)}', flush=True)
    table.tofile(filename)
    return table


def generate_golden_bin(filename, input_keys, input_vals, elements, table_size=2550000):
    slots = np.array([murmurhash3(k) % table_size for k in input_keys[:elements]], dtype='<u4')
    slots.tofile(filename)
    return slots


def const_mem_bin(filename, total_lookup_cnt):
    params = [0,0,0,0,0x10000000,0,0,0,0x20000000,0,0,0,0x40000000,0,0,0,total_lookup_cnt,0]
    Path(filename).write_bytes(struct.pack('<18I', *params))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent/'data')
    parser.add_argument('--reuse-init', action='store_true')
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    capacity, count = 2550000, 6144
    inserts = int(0.7 * capacity)
    if args.reuse_init:
        keys = np.fromfile(out/'init_keys.bin', dtype='<u8')
        vals = np.fromfile(out/'init_vals.bin', dtype='<i4')
        assert len(keys) == len(vals) == inserts
    else:
        keys = generate_init_keys(inserts, seed=0)
        vals = generate_init_vals(inserts, seed=0)
        keys.astype('<u8').tofile(out/'init_keys.bin')
        vals.astype('<i4').tofile(out/'init_vals.bin')
    table = generate_table_bin(out/'table_data_70pct.bin', capacity, keys, vals)
    queries = generate_key_bin(out/'input_keys_data.bin', keys, count, seed=1)
    slots = generate_golden_bin(out/'golden_data.bin', queries, vals, count, capacity)
    const_mem_bin(out/'const_mem.bin', count)
    # Verify serialized table, all requested keys, probe lengths and values.
    disk = np.memmap(out/'table_data_70pct.bin', mode='r', dtype=ENTRY_DTYPE)
    assert np.array_equal(disk, table)
    assert ENTRY_SIZE == ENTRY_DTYPE.itemsize == 16
    assert np.all(disk['padding'] == 0)
    lookup, probes = [], []
    for key, first in zip(queries, slots):
        slot, n = int(first), 1
        while int(disk['key'][slot]) != int(key):
            assert int(disk['key'][slot]) != EMPTY_KEY and n < capacity
            slot, n = (slot + 1) % capacity, n + 1
        lookup.append(int(disk['value'][slot]))
        probes.append(n)
    # Independent key/value oracle, honoring last-write-wins for duplicate inserts.
    selected = {int(k) for k in queries}
    oracle = {int(k): int(v) for k, v in zip(keys, vals) if int(k) in selected}
    assert lookup == [oracle[int(k)] for k in queries]
    np.asarray(lookup, dtype='<i4').tofile(out/'lookup_golden.bin')
    np.asarray(probes, dtype='<u4').tofile(out/'lookup_probe_counts.bin')
    manifest = dict(capacity=capacity, insert_count=inserts,
                    occupied_slots=int(np.count_nonzero(disk['key'] != EMPTY_KEY)),
                    query_count=count, hit_count=count, miss_count=0,
                    entry_format=ENTRY_FORMAT, seed_init=0, seed_query=1,
                    seed_note='0 does not seed RNG; use saved init files with --reuse-init',
                    golden_data='uint32 initial hash slot, not lookup value',
                    lookup_golden='int32 value, verified against original key/value pairs',
                    probe_count_includes_first_read=True,
                    probe_min=min(probes), probe_max=max(probes), probe_mean=float(np.mean(probes)),
                    numpy_version=np.__version__)
    manifest['files'] = {p.name:dict(bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                         for p in sorted(out.glob('*.bin'))}
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
