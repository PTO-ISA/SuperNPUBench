#!/usr/bin/env python3
"""Exact dyadic FP8/FP32 inputs for the original QLI driver configurations."""
import json
from pathlib import Path
import struct
import sys

import numpy as np

SUITE = Path(__file__).resolve().parents[3] / 'kernel/element_wise/tile_element_suite'
sys.path.insert(0, str(SUITE))
from qli_topk_fixture import original_profiles as static_profiles
from qli_histogram_fixture import oracle as cumulative_histogram

SPEC_PIN = '8f589b70085a37c70f210898b168aae634da5ac1'


def profiles():
    return tuple(('static', *shape, 1) for shape in static_profiles()) + (
        ('dynamic', 4, 128, 8, 1),)


def supplemental_four_pe_profiles():
    return (('static', 32, 8192, 512, 4),
            ('static', 32, 8192, 1024, 4))


def make_inputs(sq, skv):
    # Each head selects one coordinate. K encodes the token index in sixteen
    # +/-1 coordinates, while powers-of-two weights make every score unique.
    # The odd-token first head group exercises negative QK and ReLU.
    q = np.zeros((sq, 64, 128), dtype=np.uint8)
    for token in range(sq):
        for head in range(64):
            q[token, head, head % 16] = 0xb8 if token % 2 and head < 16 else 0x38
    k = np.zeros((skv, 128), dtype=np.uint8)
    for bit in range(16):
        k[:, bit] = np.where((np.arange(skv) + 1) & (1 << bit), 0x38, 0xb8)
    w = np.tile(np.array([(head // 16 + 1) * 2.0 ** (head % 16 - 18)
                          for head in range(64)], dtype='<f4'), sq)
    scale_q = np.full(sq * 64, 0.5, dtype='<f4')
    scale_k = np.full(skv, 0.25, dtype='<f4')
    return dict(srcq=q.tobytes(), srck=k.tobytes(), srcw=w.tobytes(),
                srcsq=scale_q.tobytes(), srcsk=scale_k.tobytes())


def score_oracle(inputs, sq, skv):
    # Evaluate the formula from the encoded inputs, with FP32 rounding after
    # each operation and the original four groups of sixteen heads.
    q = np.frombuffer(inputs['srcq'], dtype=np.uint8).reshape(sq, 64, 128)
    k = np.frombuffer(inputs['srck'], dtype=np.uint8).reshape(skv, 128)
    weights = (np.frombuffer(inputs['srcw'], dtype='<f4') *
               np.frombuffer(inputs['srcsq'], dtype='<f4')).reshape(sq, 64)
    scales = np.frombuffer(inputs['srcsk'], dtype='<f4')
    result = np.zeros((sq, skv), dtype='<f4')
    for token in range(sq):
        for group in range(4):
            partial = np.zeros(skv, dtype='<f4')
            for head in range(group * 16, (group + 1) * 16):
                coordinate = head % 16
                same_sign = q[token, head, coordinate] == k[:, coordinate]
                partial += np.where(same_sign, weights[token, head], np.float32(0))
            result[token] += partial
        result[token] *= scales
    return result


def generate(directory, profile):
    kind, sq, skv, topk, npe = profile
    directory.mkdir(parents=True, exist_ok=False)
    inputs = make_inputs(sq, skv)
    scores = score_oracle(inputs, sq, skv)
    stride = skv if kind == 'static' else (skv + 2047) // 2048 * 2048
    score_words = sq * stride
    index_words = sq * topk
    # Preserve both original drivers' scratch offsets, including their
    # existing reserved ranges. Only test guards sit outside that workspace.
    gap_words = 2048 if kind == 'static' else 2048 + sq * stride + 512
    key_start = score_words + index_words + gap_words
    hist_start = key_start + sq * stride
    words = hist_start + npe * 288
    initial = struct.pack('<I', 0xdeadbeef) * words
    expected = np.full(words, 0xdeadbeef, dtype='<u4')
    score_bits = scores.view('<u4')
    keys = np.where(score_bits & 0x80000000, ~score_bits,
                    score_bits | 0x80000000).astype('<u4')
    selected = np.argsort(scores, axis=1)[:, ::-1][:, :topk].astype('<u4')
    for row in range(sq):
        expected[row * stride:row * stride + skv] = score_bits[row]
        expected[score_words + row * topk:score_words + (row + 1) * topk] = selected[row]
        expected[key_start + row * stride:key_start + row * stride + skv] = keys[row]
        if kind == 'dynamic':
            expected[key_start + row * stride + skv:key_start + (row + 1) * stride] = 0
    threshold = int(keys[-1, selected[-1, -1]])
    last = keys[-1, ((skv - 1) // 2048) * 2048:].tolist()
    if kind == 'dynamic':
        last += [0] * (stride - skv)
    hist = cumulative_histogram(last, 0, threshold)
    prefix = sum(([threshold >> shift & 255] * 8 for shift in (24, 16, 8)), []) + [0] * 8
    expected[hist_start:hist_start + 288] = hist + prefix
    golden = {f'_binary_{name}_data_start': raw for name, raw in inputs.items()}
    golden['qli_workspace'] = expected.tobytes()
    assembly = ['.section .data', '.balign 128']
    for name, raw in golden.items():
        data = directory / (name + '.input.bin')
        data.write_bytes(initial if name == 'qli_workspace' else raw)
        if '"' in str(data.resolve()) or '\n' in str(data.resolve()):
            raise ValueError('unsupported fixture path')
        for symbol, payload, want in (
            (name + '_before', '.fill 128,1,0x5a', b'\x5a' * 128),
            (name, f'.incbin "{data.resolve()}"', raw),
            (name + '_after', '.fill 128,1,0xa5', b'\xa5' * 128)):
            assembly += [f'.global {symbol}', f'.type {symbol},@object',
                         symbol + ':', payload, f'.size {symbol},.-{symbol}']
            (directory / (symbol + '.golden.bin')).write_bytes(want)
        assembly.append('.balign 128')
    (directory / 'data.s').write_text('\n'.join(assembly) + '\n')
    (directory / 'manifest.json').write_text(json.dumps(dict(
        profile=profile, stride=stride, buffers=list(golden),
        score_words=score_words, index_words=index_words, words=words), indent=2) + '\n')
    return scores, selected


def compare_workspace(actual, expected, profile):
    kind, sq, skv, topk, npe = profile
    stride = skv if kind == 'static' else (skv + 2047) // 2048 * 2048
    if len(actual) != len(expected):
        raise ValueError('incomplete or oversized QLI workspace dump')
    begin, end = sq * stride * 4, (sq * stride + sq * topk) * 4
    if actual[:begin] != expected[:begin] or actual[end:] != expected[end:]:
        raise ValueError('QLI scores, padding, reserved range or scratch mismatch')
    for row in range(sq):
        offset = begin + row * topk * 4
        got, want = (struct.unpack(f'<{topk}I', raw[offset:offset + topk * 4])
                     for raw in (actual, expected))
        if len(set(got)) != topk or set(got) != set(want):
            raise ValueError(f'QLI row {row}: wrong unordered TopK indices')
