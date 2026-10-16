#!/usr/bin/env python3
"""Original QLI TopK dimensions with independent scores and unordered-set oracle."""
import json
from pathlib import Path
import struct
from qli_histogram_fixture import oracle as cumulative_histogram


def original_profiles():
    return ((64, 128, 128), (4, 2048, 512), (4, 8192, 512), (4, 2080, 512))


def profiles():
    return original_profiles() + ((2, 128, 73), (1, 128, 128), (2, 32, 17))


def scores(sq, skv):
    if skv == 128 and sq in (1, 2):
        # Raw-bit ordering composes signed values, zeros, infinities, NaN
        # payloads, a tied cutoff, and the valid key zero (0xffffffff input).
        bits = (0xffffffff, 0xffc00001, 0xff800000, 0xc0000000,
                0x80000000, 0, 0x3f800000, 0x7f800000, 0x7fc00001)
        return [bits[(row * 3 + i) % len(bits)]
                for row in range(sq) for i in range(skv)]
    # Unique positive finite FP32 scores; integer bit order equals numeric order.
    return [0x3f000000 + ((row * 1009 + i * 8191) % 0x7fffff)
            for row in range(sq) for i in range(skv)]


def storage_golden(sq, skv, k):
    raw = scores(sq, skv)
    keys = [0xffffffff - value if value >= 2**31 else value + 2**31
            for value in raw]
    selected = [sorted(range(skv), key=lambda i: keys[row * skv + i], reverse=True)[:k]
                for row in range(sq)]
    threshold = keys[(sq - 1) * skv + selected[-1][-1]]
    last_chunk = keys[(sq - 1) * skv + ((skv - 1) // 2048) * 2048:]
    hist = cumulative_histogram(last_chunk, 0, threshold)
    prefix = sum(([threshold >> shift & 255] * 8 for shift in (24, 16, 8)), []) + [0] * 8
    return sum(selected, []) + [0xdeadbeef] * 2048 + keys + hist + prefix


def compare_storage(actual, expected, sq, skv, k):
    count = sq * k
    actual_words = struct.unpack(f'<{len(actual)//4}I', actual)
    expected_words = struct.unpack(f'<{len(expected)//4}I', expected)
    for row in range(sq):
        indices = actual_words[row * k:(row + 1) * k]
        wanted = expected_words[row * k:(row + 1) * k]
        key_offset = count + 2048 + row * skv
        keys = expected_words[key_offset:key_offset + skv]
        threshold = min(keys[index] for index in wanted)
        strict = {i for i, value in enumerate(keys) if value > threshold}
        if (len(set(indices)) != k or any(index >= skv for index in indices) or
                not strict <= set(indices) or any(keys[index] < threshold for index in indices)):
            raise AssertionError(f'row {row}: wrong unordered TopK indices')
    if actual[count * 4:] != expected[count * 4:]:
        raise AssertionError('QLI key/histogram/prefix scratch or reserved gap mismatch')


def generate(directory):
    directory.mkdir(parents=True, exist_ok=True)
    golden = directory / 'golden'
    golden.mkdir(exist_ok=True)
    declarations, calls, names, storage_profiles = [], [], [], {}
    assembly = ['.pushsection .data,"aw",@progbits', '.balign 64']
    for sq, skv, k in profiles():
        stem = f'qli_topk_{sq}_{skv}_{k}'
        raw_scores = scores(sq, skv)
        want = storage_golden(sq, skv, k)
        storage_profiles[stem + '_storage'] = [sq, skv, k]
        for suffix, ctype, words, expected in (
            ('scores', 'float', raw_scores, raw_scores),
            ('storage', 'int32_t', [0xdeadbeef] * len(want), want)):
            name = stem + '_' + suffix
            names.append(name)
            declarations.append(f'extern "C" {ctype} {name}[{len(words)}];')
            data = (directory / (name + '.bin')).resolve()
            data.write_bytes(struct.pack(f'<{len(words)}I', *words))
            for symbol, payload, result in (
                (name + '_before', '.fill 64,1,0x5a', b'\x5a' * 64),
                (name, f'.incbin "{data}"', struct.pack(f'<{len(expected)}I', *expected)),
                (name + '_after', '.fill 64,1,0xa5', b'\xa5' * 64)):
                assembly += [f'.global {symbol}', f'.type {symbol},@object',
                             symbol + ':', payload, f'.size {symbol},.-{symbol}']
                (golden / (symbol + '.bin')).write_bytes(result)
        calls.append(f'qli_topk_radix<{sq}, {skv}, {k}>({stem}_scores, {stem}_storage);')
    assembly += ['.popsection']
    (directory / 'qli_topk_data.inc').write_text('\n'.join(declarations) +
        '\nasm(R"pto(\n' + '\n'.join(assembly) + '\n)pto");\n')
    (directory / 'qli_topk_calls.inc').write_text('\n'.join(calls) + '\n')
    (directory / 'buffers.json').write_text(json.dumps(names, indent=2) + '\n')
    (directory / 'storage_profiles.json').write_text(json.dumps(storage_profiles, indent=2) + '\n')
