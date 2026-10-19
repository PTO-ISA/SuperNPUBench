#!/usr/bin/env python3
"""Build full original hashtable inputs and independent final-memory goldens."""
import argparse
from pathlib import Path
import struct

from test_hashtable_golden import CAPACITY, QUERY_COUNT, MAX_PROBE, MASK64, murmur3_key

EMPTY = 0x5555aaaa77778888
ENTRY = struct.Struct('<Qii')


def fixture():
    keys = [0, 1, MASK64, 1 << 63, 0xffffffff00000000,
            0xffffffff, 0x1122334455667788, 0x5566778811223344]
    keys += [(0xd6e8feb86659fd93 * i) & MASK64 for i in range(1, 23)]
    wrap = next(key for key in range(1, 100000)
                if murmur3_key(key) % CAPACITY >= CAPACITY - 128)
    keys += [wrap, EMPTY]
    assert len(keys) == len(set(keys)) == 32
    table = bytearray(ENTRY.pack(EMPTY, 0x13572468, -559038737) * CAPACITY)
    for index, key in enumerate(keys[:-1]):
        # Every fourth key is absent. Others have an early and a last-probe
        # match; the slot just beyond the probe bound must not be observed.
        if index % 4 == 0:
            continue
        initial = murmur3_key(key) % CAPACITY
        for displacement, value in ((3, index * 13), (511, -1000 - index),
                                    (512, 100000 + index)):
            slot = (initial + displacement) % CAPACITY
            ENTRY.pack_into(table, slot * ENTRY.size, key, value, -559038737)
    queries = [keys[(index * 13 + (index // 17) * 7) % len(keys)]
               for index in range(QUERY_COUNT)]
    outputs = []
    for key in keys:
        initial = murmur3_key(key) % CAPACITY
        found = -1
        for probe in range(MAX_PROBE):
            candidate, value, _ = ENTRY.unpack_from(
                table, ((initial + probe) % CAPACITY) * ENTRY.size)
            if candidate == key:
                found = value
        outputs.append(found)
    assert -1 in outputs and any(value <= -1000 for value in outputs)
    assert outputs[-1] == 0x13572468
    lookup_values = dict(zip(keys, outputs))
    return bytes(table), struct.pack(f'<{QUERY_COUNT}Q', *queries), struct.pack(
        f'<{QUERY_COUNT}i', *(lookup_values[key] for key in queries))


def generate(directory, query_alias=False):
    directory.mkdir(parents=True, exist_ok=True)
    table, queries, outputs = fixture()
    data = {'hashtable_table': table, 'hashtable_queries': queries,
            'hashtable_output': struct.pack('<i', -559038737) * QUERY_COUNT}
    assembly = ['.pushsection .data,"aw",@progbits', '.balign 64']
    for name, raw in data.items():
        file = (directory / (name + '.bin')).resolve()
        file.write_bytes(raw)
        for symbol, payload in ((name + '_before', '.fill 64,1,0x5a'),
                                (name, f'.incbin "{file}"'),
                                (name + '_after', '.fill 64,1,0xa5')):
            assembly += [f'.global {symbol}', f'.type {symbol},@object',
                         symbol + ':', payload, f'.size {symbol},.-{symbol}']
    assembly += ['.popsection']
    (directory / 'hashtable_data.inc').write_text(
        'asm(R"pto(\n' + '\n'.join(assembly) + '\n)pto");\n')
    golden = directory / 'golden'
    golden.mkdir(exist_ok=True)
    expected = {**data, 'hashtable_output': outputs}
    if query_alias:
        # Every original batch snapshots all its keys before the final store.
        # Output starts one i32 into query backing and remains behind unread
        # future batches for each original 256/512/1024 configuration.
        evolved = bytearray(queries)
        evolved[4:4 + len(outputs)] = outputs
        expected['hashtable_queries'] = bytes(evolved)
        expected['hashtable_output'] = data['hashtable_output']
    for name, raw in expected.items():
        (golden / (name + '.bin')).write_bytes(raw)
        (golden / (name + '_before.bin')).write_bytes(b'\x5a' * 64)
        (golden / (name + '_after.bin')).write_bytes(b'\xa5' * 64)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--query-alias', action='store_true')
    args = parser.parse_args()
    generate(args.out, args.query_alias)
