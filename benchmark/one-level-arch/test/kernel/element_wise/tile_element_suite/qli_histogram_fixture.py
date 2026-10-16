#!/usr/bin/env python3
"""QLI prefix histogram inputs and independent cumulative-count oracle."""
import argparse
import json
from pathlib import Path
import struct


def profiles():
    for capacity, lengths in ((128, (0, 1, 31, 32, 33, 127, 128)),
                              (2048, (0, 1, 31, 32, 33, 127, 128, 2047, 2048))):
        for valid in lengths:
            for byte in range(4):
                yield capacity, valid, byte, 'normal'
    for capacity in (128, 2048):
        for byte in range(4):
            yield capacity, capacity, byte, 'shifted'


def keys(count):
    # Mixed prefixes, repeated buckets, and unsigned endpoints.
    base = (0x12345678, 0x12345678, 0x12345600, 0x123456ff,
            0x12340000, 0x12ffffff, 0xffffffff, 0, 0x80000000)
    return [base[i % len(base)] for i in range(count)]


def oracle(source, byte, prefix):
    key_bytes = [word.to_bytes(4, 'little') for word in source]
    prefix_bytes = prefix.to_bytes(4, 'little')
    return [sum(value[byte] <= bucket and value[byte + 1:] == prefix_bytes[byte + 1:]
                for value in key_bytes) for bucket in range(256)]


def generate(directory):
    directory.mkdir(parents=True, exist_ok=True)
    golden = directory / 'golden'
    golden.mkdir(exist_ok=True)
    declarations, calls, names = [], [], []
    assembly = ['.pushsection .data,"aw",@progbits', '.balign 64']
    for capacity, valid, byte, alias in profiles():
        stem = f'qli_hist_{capacity}_{valid}_{byte}_{alias}'
        src = keys(max(1, valid, 257 if alias == 'shifted' else 0))
        dst = [0xdeadbeef] * 256
        counts = oracle(src[:valid], byte, 0x12345678)
        want_src, want_dst = list(src), list(dst)
        if alias == 'normal':
            want_dst = counts
        else:
            want_src[1:257] = counts
        for suffix, words, want in (('src', src, want_src), ('dst', dst, want_dst)):
            name = stem + '_' + suffix
            names.append(name)
            declarations.append(f'extern "C" uint32_t {name}[{len(words)}];')
            data = (directory / (name + '.bin')).resolve()
            data.write_bytes(struct.pack(f'<{len(words)}I', *words))
            for symbol, payload, result in (
                (name + '_before', '.fill 64,1,0x5a', b'\x5a' * 64),
                (name, f'.incbin "{data}"', struct.pack(f'<{len(want)}I', *want)),
                (name + '_after', '.fill 64,1,0xa5', b'\xa5' * 64)):
                assembly += [f'.global {symbol}', f'.type {symbol},@object',
                             symbol + ':', payload, f'.size {symbol},.-{symbol}']
                (golden / (symbol + '.bin')).write_bytes(result)
        target = stem + ('_dst' if alias == 'normal' else '_src + 1')
        calls.append(f'qli_element::histogram<{capacity}, {valid}>('
                     f'{target}, {stem}_src, {byte}, 0x12345678u);')
    assembly += ['.popsection']
    (directory / 'qli_histogram_data.inc').write_text('\n'.join(declarations) +
        '\nasm(R"pto(\n' + '\n'.join(assembly) + '\n)pto");\n')
    (directory / 'qli_histogram_calls.inc').write_text('\n'.join(calls) + '\n')
    (directory / 'buffers.json').write_text(json.dumps(names, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    generate(parser.parse_args().out)
