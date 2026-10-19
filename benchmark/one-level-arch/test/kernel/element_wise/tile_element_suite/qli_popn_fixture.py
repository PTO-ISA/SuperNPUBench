#!/usr/bin/env python3
"""Repeated argmax through Tile references with ties and boundary-lane clearing."""
import json
import struct

CHUNK_BASE = 4096


def profiles():
    for capacity, valid in ((128, 128), (128, 32), (2048, 2048)):
        for ties in (False, True):
            for count in (0, 1, len(values(valid, ties)[1])):
                yield capacity, valid, ties, count


def values(valid, ties):
    source = [0] * valid
    positions = sorted({i for i in (0, 31, 32, valid - 1) if i < valid})
    for i, position in enumerate(positions):
        source[position] = 0xffffffff if ties else 0x80000000 + i
    return source, positions


def compare_indices(actual, expected):
    count = len(actual) // 4
    actual = struct.unpack(f'<{count}i', actual)
    expected = struct.unpack(f'<{count}i', expected)
    if len(set(actual)) != count or set(actual) != set(expected):
        raise AssertionError('PopN duplicated a lane or selected outside the candidate set')


def compare_selection(actual_indices, actual_remaining, valid, ties, count):
    source, positions = values(valid, ties)
    remaining = list(source)
    selected = struct.unpack(f'<{max(1, count)}I', actual_indices)
    if count == 0:
        if selected != (0xdeadbeef,):
            raise AssertionError('zero-count PopN wrote output')
    else:
        seen = set()
        for absolute in selected:
            index = absolute - CHUNK_BASE
            maximum = max(remaining)
            if index not in positions or index in seen or remaining[index] != maximum or maximum == 0:
                raise AssertionError('PopN did not select a remaining maximum')
            seen.add(index)
            remaining[index] = 0
    if actual_remaining != struct.pack(f'<{valid}I', *remaining):
        raise AssertionError('caller did not observe the exact selected-lane updates')

def generate(directory):
    directory.mkdir(parents=True, exist_ok=True)
    golden = directory / 'golden'
    golden.mkdir(exist_ok=True)
    names, declarations, calls, selection_profiles = [], [], [], {}
    assembly = ['.pushsection .data,"aw",@progbits', '.balign 64']
    for capacity, valid, ties, count in profiles():
        stem = f'qli_popn_{capacity}_{valid}_{int(ties)}_{count}'
        source, positions = values(valid, ties)
        selection_profiles[stem] = [valid, ties, count]
        remaining = list(source)
        selected = sorted(positions, key=lambda i: source[i], reverse=True)[:count]
        for index in selected:
            remaining[index] = 0
        for suffix, ctype, initial, want in (
            ('input', 'uint32_t', source, source),
            ('remaining', 'uint32_t', [0xdeadbeef] * valid, remaining),
            ('indices', 'int32_t', [0xdeadbeef] * max(1, count),
             [CHUNK_BASE + i for i in selected] if count else [0xdeadbeef])):
            name = stem + '_' + suffix
            names.append(name)
            declarations.append(f'extern "C" {ctype} {name}[{len(want)}];')
            data = (directory / (name + '.bin')).resolve()
            data.write_bytes(struct.pack(f'<{len(initial)}I', *initial))
            for symbol, payload, result in (
                (name + '_before', '.fill 64,1,0x5a', b'\x5a' * 64),
                (name, f'.incbin "{data}"', struct.pack(f'<{len(want)}I', *want)),
                (name + '_after', '.fill 64,1,0xa5', b'\xa5' * 64)):
                assembly += [f'.global {symbol}', f'.type {symbol},@object',
                             symbol + ':', payload, f'.size {symbol},.-{symbol}']
                (golden / (symbol + '.bin')).write_bytes(result)
        calls.append(f'qli_popn_stage<{capacity},{valid}>({stem}_input, '
                     f'{stem}_remaining, {stem}_indices, {count}, {CHUNK_BASE});')
    assembly += ['.popsection']
    (directory / 'qli_popn_data.inc').write_text('\n'.join(declarations) +
        '\nasm(R"pto(\n' + '\n'.join(assembly) + '\n)pto");\n')
    (directory / 'qli_popn_calls.inc').write_text('\n'.join(calls) + '\n')
    (directory / 'buffers.json').write_text(json.dumps(names, indent=2) + '\n')
    (directory / 'selection_profiles.json').write_text(json.dumps(selection_profiles, indent=2) + '\n')
