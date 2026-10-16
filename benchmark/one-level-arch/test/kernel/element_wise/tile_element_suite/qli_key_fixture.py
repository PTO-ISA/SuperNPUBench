#!/usr/bin/env python3
"""Independent raw-bit QLI key fixtures, including full-chunk alias snapshots."""
import argparse
import json
from pathlib import Path
import struct

# Deliberately retain signed zero, infinities and both NaN payload classes.
BITS = (0x3f800001, 0, 0x80000000, 0x7f800000, 0xff800000,
        0x7fc12345, 0xffc12345, 0x7f812345, 0xff812345, 0xffffffff,
        1, 0x80000001, 0x7f7fffff, 0xff7fffff)


def ordered_bits(word):
    # Arithmetic oracle independent of the kernel's bitwise selection.
    return 0xffffffff - word if word >= 2**31 else word + 2**31


def profiles():
    for capacity, lengths in ((128, (0, 1, 31, 32, 33, 127, 128)),
                              (2048, (0, 1, 31, 32, 33, 127, 128, 2047, 2048))):
        for valid in lengths:
            for alias in ('normal', 'exact', 'shifted'):
                yield capacity, valid, alias


def inputs(valid, alias):
    count = max(1, valid + (alias == 'shifted'))
    return [BITS[i % len(BITS)] if i < len(BITS)
            else ((i * 0x9e3779b9) ^ 0x87654321) & 0xffffffff
            for i in range(count)]


def expected(valid, alias):
    source = inputs(valid, alias)
    output = [0xdeadbeef] * max(valid, 1)
    snapshot = list(map(ordered_bits, source[:valid]))
    if alias == 'normal':
        output[:valid] = snapshot
    else:
        shift = int(alias == 'shifted')
        source[shift:shift + valid] = snapshot
    return source, output


def generate(directory):
    directory.mkdir(parents=True, exist_ok=True)
    golden = directory / 'golden'
    golden.mkdir(exist_ok=True)
    declarations, calls, names = [], [], []
    assembly = ['.pushsection .data,"aw",@progbits', '.balign 64']
    for capacity, valid, alias in profiles():
        stem = f'qli_{capacity}_{valid}_{alias}'
        src, dst = inputs(valid, alias), [0xdeadbeef] * max(valid, 1)
        want_src, want_dst = expected(valid, alias)
        for suffix, words, want in (('src', src, want_src), ('dst', dst, want_dst)):
            name = stem + '_' + suffix
            names.append(name)
            declarations.append(f'extern "C" uint32_t {name}[{len(words)}];')
            raw = struct.pack(f'<{len(words)}I', *words)
            data = (directory / (name + '.bin')).resolve()
            data.write_bytes(raw)
            for symbol, payload, result in (
                (name + '_before', '.fill 64,1,0x5a', b'\x5a' * 64),
                (name, f'.incbin "{data}"', struct.pack(f'<{len(want)}I', *want)),
                (name + '_after', '.fill 64,1,0xa5', b'\xa5' * 64)):
                assembly += [f'.global {symbol}', f'.type {symbol},@object',
                             symbol + ':', payload, f'.size {symbol},.-{symbol}']
                (golden / (symbol + '.bin')).write_bytes(result)
        target = stem + ('_dst' if alias == 'normal' else '_src')
        if alias == 'shifted':
            target += ' + 1'
        calls.append(f'qli_element::make_key<{capacity}, {valid}>({target}, {stem}_src);')
    assembly += ['.popsection']
    (directory / 'qli_key_data.inc').write_text('\n'.join(declarations) +
        '\nasm(R"pto(\n' + '\n'.join(assembly) + '\n)pto");\n')
    (directory / 'qli_key_calls.inc').write_text('\n'.join(calls) + '\n')
    (directory / 'buffers.json').write_text(json.dumps(names, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    generate(parser.parse_args().out)
