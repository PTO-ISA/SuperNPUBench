#!/usr/bin/env python3
"""Check an FP32 B1/M128/N64/K256 L1 ELF built with res_check=on CHK_DIR=.

Uses a NumPy matmul and nearest-value E6M2/S1P2 oracles. The gfrun file
service confines relative guest paths to --output. No simulator edits.
"""
import argparse
from pathlib import Path
import subprocess
import numpy as np

MODEL = Path('/Users/blacktraker/Programming/gitproj/DV4/SuperScalarModel')


def bf16(x):
    x = np.asarray(x, dtype=np.float32)
    bits = x.view(np.uint32)
    rounded = ((bits.astype(np.uint64) + 0x7fff + ((bits >> 16) & 1)) >> 16).astype(np.uint32)
    rounded = np.where((bits & 0x7fffffff) > 0x7f800000, (bits >> 16) | 0x40, rounded)
    return (rounded.astype(np.uint32) << 16).view(np.float32)


def reference(a, b):
    c = bf16(np.einsum("mk,nk->mn", a.astype(np.float64), b.astype(np.float64))).reshape(128, 1, 64)
    base_hp = bf16(np.max(np.abs(c), axis=-1) * bf16(np.float32(4 / 7)))
    # Enumerate all finite E6M2 values; on a tie prefer an even code.
    code = np.arange(255)
    levels = np.ldexp(1 + (code % 4) / 4, code // 4 - 48)
    distance = np.abs(base_hp.astype(np.float64)[..., None] - levels)
    minimum = distance.min(axis=-1, keepdims=True)
    candidates = np.where(distance == minimum, code, 256)
    even = np.where((candidates % 2) == 0, candidates, 256).min(axis=-1)
    scales = np.where(even < 255, even, candidates.min(axis=-1)).astype(np.uint8)
    special = ~np.isfinite(base_hp)
    scales[special] = 255
    base = levels[np.minimum(scales, 254)].astype(np.float32)
    base[special] = np.nan
    normalized = bf16(c * bf16(1 / base)[..., None])
    # S1P2 is a sign and a uniformly spaced magnitude, not E2M1 FP4.
    magnitude = np.rint(np.minimum(np.abs(normalized), 1.75) * 4)
    payload = (np.nan_to_num(magnitude).astype(np.uint8) |
               (np.signbit(normalized).astype(np.uint8) << 3))
    payload[np.isnan(normalized)] = 0
    packed = payload[..., 0::2] | (payload[..., 1::2] << 4)
    return packed.ravel(), scales.ravel()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--elf', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(298)
    a = rng.integers(-4, 5, (128, 256)).astype(np.float32) / 8
    b = rng.integers(-4, 5, (64, 256)).astype(np.float32) / 8
    boundary = np.zeros_like(a)
    lane = np.array([0, .125, .25, .375, .5, .625, .75, .875, 1, 1.125,
                     1.25, 1.375, 1.5, 1.625, 1.75, -1.75], dtype=np.float32)
    for row in range(128):
        boundary[row, :64] = np.tile(lane, 4) * np.float32(2.0 ** ((row % 24) - 12))
        if row % 2: boundary[row, :64] *= -1
    # Zero rows and E6M2 underflow boundaries.
    boundary[0] = 0
    boundary[1, :64] = np.float32(2.0 ** -50)
    boundary[2, :64] = np.float32(2.0 ** -48)
    identity = np.eye(64, 256, dtype=np.float32)
    for name, aa, bb in [('random', a, b), ('boundaries', boundary, identity),
                          ('zero', np.zeros_like(a), np.zeros_like(b))]:
        aa.tofile(args.output / 'src0.bin'); bb.tofile(args.output / 'src1.bin')
        for filename in ('data.bin', 'scale.bin'):
            (args.output / filename).unlink(missing_ok=True)
        r = subprocess.run([str(MODEL / 'bin/gfrun'), '--hosted-file-root',
                            str(args.output.resolve()), '-s', 'softcore.multiThreadNum=4',
                            '-f', str(args.elf.resolve())], cwd=MODEL,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=45)
        (args.output / f'{name}.log').write_bytes(r.stdout)
        if r.returncode or b'formal_End=1 formal_Abort=0 status=0' not in r.stdout:
            raise RuntimeError(f'{name}: gfrun failed; see {(args.output / (name + ".log")).resolve()}')
        expected_data, expected_scale = reference(aa, bb)
        for filename, expected in [('data.bin', expected_data), ('scale.bin', expected_scale)]:
            actual = np.fromfile(args.output / filename, dtype=np.uint8)
            np.testing.assert_array_equal(actual, expected, err_msg=f'{name}/{filename}')
        print(f'PASS {name}: 4096 payload bytes + 128 scale bytes match exactly')


if __name__ == '__main__':
    main()
