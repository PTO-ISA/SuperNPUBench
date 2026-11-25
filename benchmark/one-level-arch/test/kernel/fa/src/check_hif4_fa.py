#!/usr/bin/env python3
"""Verify native HiF4 FA with nonuniform U32 scales and an FP64 softmax oracle.
Build the ELF with res_check=on CHK_DIR=. and pass its absolute path.
"""
import argparse
from pathlib import Path
import re
import subprocess
import numpy as np

MODEL = Path('/Users/blacktraker/Programming/gitproj/DV4/SuperScalarModel')


def inputs(rows, cols, rng, base_code):
    codes = rng.integers(0, 16, (rows, cols), dtype=np.uint8)
    packed = codes[:, ::2] | (codes[:, 1::2] << 4)
    shape = (rows, cols // 64)
    base = (base_code + np.indices(shape).sum(axis=0) % 4).astype(np.uint32)
    words = base | (rng.integers(0, 256, shape, dtype=np.uint32) << 8) | (rng.integers(0, 65536, shape, dtype=np.uint32) << 16)
    values = ((codes & 7).astype(np.float64) / 4) * np.where(codes & 8, -1, 1)
    for lane in range(cols):
        word = words[:, lane // 64]
        bit8 = (word >> (8 + (lane % 64) // 8)) & 1
        bit16 = (word >> (16 + (lane % 64) // 4)) & 1
        scale = np.ldexp(1 + (word & 3) / 4, ((word & 255) >> 2).astype(int) - 48 + bit8.astype(int) + bit16.astype(int))
        values[:, lane] *= scale
    return packed, words.astype('<u4'), values


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--elf', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--zero-query', action='store_true', help='PV control: softmax must be uniform')
    args = p.parse_args()
    sq, skv = map(int, re.search(r'_Sq(\d+)_Skv(\d+)', args.elf.name).groups())
    dims = re.search(r'_qD(\d+)_vD(\d+)', args.elf.name)
    qd, vd = map(int, dims.groups()) if dims else (128, 128)
    rng = np.random.default_rng(298)
    q, qs, qf = inputs(sq, qd, rng, 0xb4)
    k, ks, kf = inputs(skv, qd, rng, 0xb4)
    v, vs, vf = inputs(vd, skv, rng, 0xbc)
    if args.zero_query:
        q.fill(0); qf.fill(0)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, data in [('q', q), ('k', k), ('v', v), ('q_scale', qs), ('k_scale', ks), ('v_scale', vs)]:
        data.tofile(args.output / f'src{name}.bin')
    result = args.output / 'res.bin'
    result.unlink(missing_ok=True)
    run = subprocess.run([str(MODEL / 'bin/gfrun'), '--hosted-file-root', str(args.output.resolve()),
                          '-s', 'softcore.multiThreadNum=4', '-f', str(args.elf.resolve())],
                         cwd=MODEL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
    (args.output / 'gfrun.log').write_bytes(run.stdout)
    if run.returncode or b'formal_End=1 formal_Abort=0 status=0' not in run.stdout:
        raise RuntimeError(f'gfrun failed: {(args.output / "gfrun.log").resolve()}')
    logits = np.einsum('md,nd->mn', qf, kf) / np.sqrt(qd)
    probability = np.exp(logits - logits.max(axis=1, keepdims=True))
    probability /= probability.sum(axis=1, keepdims=True)
    golden = np.einsum('mn,dn->md', probability, vf)
    actual = (np.fromfile(result, dtype='<u2').astype(np.uint32) << 16).view(np.float32).reshape(sq, vd)
    if not np.isfinite(actual).all(): raise AssertionError('nonfinite output')
    # BF16 scores/exp/sums and final BF16 output introduce quantization error.
    # An independent full-softmax oracle also catches scale/layout mistakes.
    np.testing.assert_allclose(actual, golden, atol=.002, rtol=.01)
    print(f'PASS {args.elf.name}: max_abs={np.max(np.abs(actual-golden)):.6g}, rmse={np.sqrt(np.mean((actual-golden)**2)):.6g}')


if __name__ == '__main__':
    main()
