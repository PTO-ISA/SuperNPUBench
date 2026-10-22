#!/usr/bin/env python3
"""Independent numerical inputs for the seven original active low-precision cases.

Semantics pin: pto-spec 8f589b70085a37c70f210898b168aae634da5ac1,
formats/{hif4x2,hif4-scale,e2m1x2,bf16}.asl. Values are dyadic and bounded
so every product and partial sum in these fixtures is exactly representable in
FP32 (and the HiF4 matrix accumulation precision). No model implementation is
imported. Extra A16W4 weight/scale storage keeps the original driver capacity.
"""
import json
import re
from pathlib import Path

import numpy as np

SPEC_PIN = '8f589b70085a37c70f210898b168aae634da5ac1'


def profiles():
    return [
        ('HIF4_HIF4', 'MX_NOGATHER', 256, 2048, 2048, 32, 64, 1024),
        ('HIF4_HIF4', 'MX_NOGATHER', 512, 1280, 4096, 32, 64, 1024),
        ('HIF4_HIF4', 'MX_NOGATHER_REUSEA', 256, 2048, 2048, 32, 64, 1024),
        ('HIF4_HIF4', 'MX_NOGATHER_REUSEA', 256, 1280, 2048, 32, 64, 1024),
        ('A16W4', '', 256, 2048, 2048, 16, 16, 128),
        ('A16W4', '', 512, 1280, 2048, 16, 16, 128),
        ('A16W4', '', 512, 512, 4096, 16, 16, 128),
    ]


def active_commands(path):
    commands = []
    path = Path(path)
    lines = (json.loads(path.read_text())["original_commands"]
             if path.suffix == ".json" else path.read_text().splitlines())
    for line in lines:
        if not re.match(r'^make\s', line):
            continue
        fields = dict(re.findall(r'(\w+)=(\w+)', line))
        if fields.get('TYPE') in ('HIF4_HIF4', 'A16W4'):
            commands.append((fields['TYPE'], fields.get('VER', ''),
                *(int(fields[k]) for k in ('M', 'N', 'K', 'tM', 'tN', 'tK'))))
    return commands


def unpack(raw):
    raw = (np.frombuffer(raw, dtype=np.uint8) if isinstance(raw, (bytes, bytearray))
           else np.asarray(raw, dtype=np.uint8).reshape(-1))
    lanes = np.empty(raw.size * 2, dtype=np.uint8)
    lanes[0::2] = raw & 15
    lanes[1::2] = raw >> 4
    return lanes


def pack(lanes):
    lanes = np.asarray(lanes, dtype=np.uint8).reshape(-1)
    if lanes.size % 2 or np.any(lanes > 15):
        raise ValueError('FP4 packing requires pairs of four-bit lanes')
    return (lanes[0::2] | (lanes[1::2] << 4)).tobytes()


def decode_fp4(lanes, hif4):
    # HiF4 E1M2 magnitudes are 0, 1/4, ..., 7/4.
    table = np.arange(8, dtype=np.float32) / 4 if hif4 else np.array(
        [0, .5, 1, 1.5, 2, 3, 4, 6], dtype=np.float32)
    lanes = np.asarray(lanes, dtype=np.uint8)
    return table[lanes & 7] * np.where(lanes & 8, -1., 1.).astype(np.float32)


def hif4_scale(words, q):
    words = np.asarray(words, dtype=np.uint32)
    base = words & 255
    if np.any(base == 255):
        raise ValueError('fixture scales must be finite')
    exp = (base >> 2).astype(np.int32) - 48
    fraction = 1. + (base & 3).astype(np.float32) / 4
    increment = ((words >> (8 + q // 8)) & 1) + ((words >> (16 + q // 4)) & 1)
    return np.ldexp(fraction, exp + increment.astype(np.int32)).astype(np.float32)


def expand_hif4(lanes, scales, left):
    values = decode_fp4(lanes, True)
    for q in range(64):
        if left:
            values[:, q::64] *= hif4_scale(scales, q)
        else:
            values[q::64, :] *= hif4_scale(scales, q)
    return values


def reference_a16w4(a, b, scale):
    m, k = a.shape
    if k % 128 or b.shape[0] != k or scale.shape != (k // 128, b.shape[1]):
        raise ValueError('A16W4 requires one per-column scale for each 128 K lanes')
    result = np.zeros((m, b.shape[1]), dtype=np.float32)
    for group in range(k // 128):
        part = a[:, group*128:(group+1)*128] @ b[group*128:(group+1)*128]
        result += part * scale[group]
    return result


def original_capacity(rows, cols):
    # Preserve the source expression's left-to-right integer evaluation.
    return ((rows+1)//2*(cols+1)//2)*4


def inputs(profile):
    kind, _, m, n, k, *_ = profile
    rng = np.random.default_rng(13579)
    if kind == 'HIF4_HIF4':
        a_lanes = rng.integers(0, 16, (m, k), dtype=np.uint8)
        b_lanes = rng.integers(0, 16, (k, n), dtype=np.uint8)
        # E6M2 base .25; both exponent bit planes vary independently.
        sa = (rng.integers(0, 1 << 24, (m, k//64), dtype=np.uint32) << 8) | 0xb8
        sb = (rng.integers(0, 1 << 24, (k//64, n), dtype=np.uint32) << 8) | 0xb8
        a = expand_hif4(a_lanes, sa, True)
        b = expand_hif4(b_lanes, sb, False)
        raw = {'src0': pack(a_lanes), 'src1': pack(b_lanes),
               'src0_mx': sa.astype('<u4').tobytes(), 'src1_mx': sb.astype('<u4').tobytes()}
        result = (a @ b).astype('<f4').tobytes()
    elif kind == 'A16W4':
        a = rng.integers(-4, 5, (m, k)).astype(np.float32) / 8
        b_lanes = rng.integers(0, 16, (k, n), dtype=np.uint8)
        scale = rng.integers(1, 5, (k//128, n)).astype(np.float32) / 8
        bf16 = (a.astype('<f4').view('<u4') >> 16).astype('<u2')
        raw = {'src0': bf16.tobytes(), 'src1': pack(b_lanes), 'src2': scale.astype('<f4').tobytes()}
        raw['src0'] += b'\x6d' * (original_capacity(m,k)*2-len(raw['src0']))
        # Match the existing arrays, including their unused packed-weight and scale tails.
        raw['src1'] += b'\x6d' * (original_capacity(k,n) - len(raw['src1']))
        raw['src2'] += b'\x6d' * (original_capacity(k,n)*4 - len(raw['src2']))
        result = reference_a16w4(a, decode_fp4(b_lanes, False), scale).astype('<f4').tobytes()
        result += b'\xef\xbe\xad\xde'*(original_capacity(m,n)-m*n)
    else:
        raise ValueError(kind)
    return raw, result


def generate(directory, profile):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    raw, result = inputs(profile)
    raw['dst'] = b'\xef\xbe\xad\xde' * (len(result)//4)
    wants = {**raw, 'dst': result, 'status': (0x600d600d).to_bytes(4, 'little')}
    raw['status'] = (0xdeadbeef).to_bytes(4, 'little')
    golden = directory / 'golden'
    golden.mkdir()
    assembly, declarations = ['.pushsection .data,"aw",@progbits'], []
    names = []
    alignment = 4096 if profile[0] == 'HIF4_HIF4' else 64
    for suffix, data in raw.items():
        name = 'matmul_' + suffix
        names.append(name)
        declarations.append(f'extern "C" unsigned char {name}[{len(data)}];' if suffix != 'status'
                            else 'extern "C" volatile uint32_t matmul_status;')
        path = (directory / (name + '.bin')).resolve()
        path.write_bytes(data)
        assembly.append(f'.balign {alignment}')
        for symbol, payload, expected in (
            (name+'_before', f'.fill {alignment},1,0x5a', b'\x5a'*alignment),
            (name, f'.incbin "{path}"', wants[suffix]),
            (name+'_after', f'.fill {alignment},1,0xa5', b'\xa5'*alignment)):
            assembly += [f'.global {symbol}', f'.type {symbol},@object', symbol+':',
                         payload, f'.size {symbol},.-{symbol}']
            (golden / (symbol+'.bin')).write_bytes(expected)
    assembly += ['.popsection']
    (directory/'matmul_data.inc').write_text('\n'.join(declarations)+
        '\nasm(R"pto(\n'+'\n'.join(assembly)+'\n)pto");\n')
    (directory/'buffers.json').write_text(json.dumps(names, indent=2)+'\n')
    (directory/'profile.json').write_text(json.dumps({'profile': profile, 'spec_pin': SPEC_PIN,
        'comparison': 'exact FP32 values, including signed-zero equivalence; all other bytes exact'}, indent=2)+'\n')
