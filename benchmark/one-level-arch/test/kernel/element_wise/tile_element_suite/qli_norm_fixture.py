#!/usr/bin/env python3
"""Unsigned comparisons, in-place lane clearing, and reduction on original NORM rows."""
import json
import struct


def profiles():
    for capacity, valid in ((128, 128), (128, 32), (2048, 2048)):
        for selected in sorted({0, 31, 32, valid - 1, valid}):
            yield capacity, valid, selected


def words(valid):
    values = [((i * 0x9e3779b9) ^ 0x87654321) & 0xffffffff for i in range(valid)]
    for i in (0, 31, 32, valid - 1):
        if i < valid:
            values[i] = 0x80000000
    values[1] = 0xffffffff
    values[2] = 0
    return values


def expected(valid, selected):
    source = words(valid)
    result = {'input': source}
    sums = []
    for label, comparison in (('gt', lambda x: x > 2**31), ('eq', lambda x: x == 2**31)):
        take = list(map(comparison, source))
        result[label + '_count'] = list(map(int, take))
        candidates = [x if keep else 0 for x, keep in zip(source, take)]
        result[label + '_value'] = candidates
        result[label + '_cleared'] = [0 if i == selected else x
                                     for i, x in enumerate(candidates)]
        sums.append(sum(take))
    result['sums'] = sums
    return result


def generate(directory, selected_profiles=None):
    directory.mkdir(parents=True, exist_ok=True)
    golden = directory / 'golden'
    golden.mkdir(exist_ok=True)
    names, declarations, calls = [], [], []
    assembly = ['.pushsection .data,"aw",@progbits', '.balign 64']
    for capacity, valid, selected in (profiles() if selected_profiles is None else selected_profiles):
        stem = f'qli_norm_{capacity}_{valid}_{selected}'
        results = expected(valid, selected)
        for suffix, want in results.items():
            name = stem + '_' + suffix
            names.append(name)
            declarations.append(f'extern "C" uint32_t {name}[{len(want)}];')
            initial = want if suffix == 'input' else [0xdeadbeef] * len(want)
            data = (directory / (name + '.bin')).resolve()
            data.write_bytes(struct.pack(f'<{len(initial)}I', *initial))
            for symbol, payload, result in (
                (name + '_before', '.fill 64,1,0x5a', b'\x5a' * 64),
                (name, f'.incbin "{data}"', struct.pack(f'<{len(want)}I', *want)),
                (name + '_after', '.fill 64,1,0xa5', b'\xa5' * 64)):
                assembly += [f'.global {symbol}', f'.type {symbol},@object',
                             symbol + ':', payload, f'.size {symbol},.-{symbol}']
                (golden / (symbol + '.bin')).write_bytes(result)
        args = ', '.join(stem + '_' + suffix for suffix in results)
        calls.append(f'qli_norm_stage<{capacity},{valid}>({args}, 0x80000000u, {selected});')
    assembly += ['.popsection']
    (directory / 'qli_norm_data.inc').write_text('\n'.join(declarations) +
        '\nasm(R"pto(\n' + '\n'.join(assembly) + '\n)pto");\n')
    (directory / 'qli_norm_calls.inc').write_text('\n'.join(calls) + '\n')
    (directory / 'buffers.json').write_text(json.dumps(names, indent=2) + '\n')
