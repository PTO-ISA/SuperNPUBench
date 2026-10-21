#!/usr/bin/env python3
"""Check AlgoD's unscaled MXFP4 attention, including its BF16 online arithmetic.

NumPy is the only host dependency. The reference decodes exactly the generated
payloads, emulates each BF16 operation, E2M1 ties-to-even quantization, E8M0
flooring and FP32 PV accumulation. Exact FP32 softmax error is reported separately.
"""
import argparse
import json
import re
import subprocess
from pathlib import Path

import numpy as np

E2M1 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6,
                0, -.5, -1, -1.5, -2, -3, -4, -6], np.float32)
INPUTS = ("srcq", "srck", "srcv", "srcq_scale", "srck_scale", "srcv_scale")


def bf16(x):
    bits = np.asarray(x, np.float32).view(np.uint32)
    return ((bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000).view(np.float32)


def expdif(left, right):
    return bf16(np.exp(bf16(left - right)))


def quantize(values):
    # Stable sort keeps even codes first on a half-way value.
    order = np.array([0, 2, 4, 6, 1, 3, 5, 7])
    index = np.argmin(np.abs(values[..., None] - E2M1[order]), axis=-1)
    return E2M1[order[index]]


def generate(folder, sq, skv, seed):
    rng = np.random.default_rng(seed)
    matrices = []
    for name, rows, coefficients in (("q", sq, (3, 1)), ("k", skv, (1, 3)),
                                      ("v", skv, (1, 3))):
        codes = rng.choice(np.array([1, 2, 3, 9, 10, 11], np.uint8), (rows, 128))
        shape = (128, skv // 32) if name == "v" else (rows, 4)
        row, col = np.indices(shape)
        scales = (123 + (coefficients[0] * row + coefficients[1] * col) % 4).astype(np.uint8)
        (codes[:, ::2] | (codes[:, 1::2] << 4)).tofile(folder / f"src{name}.bin")
        scales.tofile(folder / f"src{name}_scale.bin")
        decoded = np.exp2(scales.astype(np.float32) - 127)
        expanded = np.repeat(decoded.T, 32, axis=0) if name == "v" else np.repeat(decoded, 32, axis=1)
        matrices.append(E2M1[codes] * expanded)
    return matrices


def reference(q, k, v):
    sq, skv = q.shape[0], k.shape[0]
    ln2, invln2, ln4 = bf16(np.log(2)), bf16(1 / np.log(2)), bf16(np.log(4))
    pv = np.zeros((sq, 128), np.float32)
    for start in range(0, skv, 256):
        score = bf16(q @ k[start:start + 256].T)
        groups = score.reshape(sq, 8, 32)
        log_reference = bf16(groups.max(2) - ln4)
        grid = bf16(bf16(np.floor(bf16(score.max(1) * invln2))) * ln2)
        if start == 0:
            running = grid
        else:
            next_grid = np.maximum(running, grid)
            # TCVT float-to-int with RNONE truncates toward zero.
            exponent = np.trunc(bf16(bf16(next_grid - running) * invln2))
            pv *= np.exp2(-exponent[:, None])
            running = next_grid
        scale = expdif(log_reference, running[:, None])
        values = expdif(groups, log_reference[:, :, None])
        sums = np.zeros((sq, 8), np.float32)
        for column in range(32):
            sums = bf16(sums + values[:, :, column])
        if start == 0:
            group_sums = sums
        else:
            group_sums = bf16(bf16(group_sums * expdif(previous, log_reference)) + sums)
        previous = log_reference
        quantized = quantize(values) * np.exp2(np.floor(np.log2(scale)))[:, :, None]
        pv += quantized.reshape(sq, 256) @ v[start:start + 256]
    denominator = np.zeros(sq, np.float32)
    for group in range(8):
        denominator = bf16(denominator + bf16(group_sums[:, group] * scale[:, group]))
    matched = bf16(pv / denominator[:, None])
    scores = q @ k.T
    probabilities = np.exp(scores - scores.max(1, keepdims=True))
    exact = (probabilities / probabilities.sum(1, keepdims=True)) @ v
    return matched, exact


def compare(actual, golden):
    # All input products and FP32 sums in the declared test distribution are
    # exactly representable. The reference reproduces every BF16 publication;
    # require an exact result, including near-zero and cancellation cases.
    return bool(np.all(np.isfinite(actual)) and np.array_equal(actual, golden))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--elf", type=Path, required=True)
    parser.add_argument("--gfrun", type=Path, required=True)
    parser.add_argument("--case-dir", type=Path, required=True,
                        help="directory named fa_lowp_algD_Sq<Sq>_Skv<Skv> for this ELF")
    parser.add_argument("--sq", type=int, default=256)
    parser.add_argument("--skv", type=int, default=512)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    if args.sq < 256 or args.sq % 256 or args.skv < 256 or args.skv % 256:
        parser.error("Sq and Skv must be positive multiples of 256")
    elf, model = args.elf.resolve(), args.gfrun.resolve()
    if not elf.is_file() or not model.is_file():
        parser.error("ELF and gfrun must exist")
    folder = args.case_dir.resolve()
    if folder.name != f"fa_lowp_algD_Sq{args.sq}_Skv{args.skv}":
        parser.error("case-dir basename must match the ELF shape-specific input path")
    folder.mkdir(parents=True, exist_ok=True)
    for name in ("res",) + tuple(x + "_readback" for x in INPUTS):
        (folder / (name + ".bin")).unlink(missing_ok=True)
    q, k, v = generate(folder, args.sq, args.skv, args.seed)
    golden, exact = reference(q, k, v)
    if not np.isfinite(golden).all() or not np.any(golden):
        raise RuntimeError("non-finite or vacuous golden")
    golden.tofile(folder / "algorithm_golden.bin")
    exact.tofile(folder / "softmax_golden.bin")
    command = [str(model), "-s", "softcore.multiThreadNum=4", "-f", str(elf)]
    result = subprocess.run(command, cwd=folder.parent, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, timeout=args.timeout)
    (folder / "gfrun.log").write_text(result.stdout)
    if result.returncode or re.search(r"R2\s*=\s*0\b", result.stdout) is None:
        raise RuntimeError(f"gfrun failed: rc={result.returncode}")
    sizes = {}
    for name in INPUTS:
        source, readback = folder / (name + ".bin"), folder / (name + "_readback.bin")
        if source.read_bytes() != readback.read_bytes():
            raise RuntimeError(f"input readback mismatch: {name}")
        sizes[name] = source.stat().st_size
    payload = (folder / "res.bin").read_bytes()
    expected_size = args.sq * 128 * 2
    if len(payload) != expected_size:
        raise RuntimeError(f"output byte count {len(payload)} != {expected_size}")
    actual = (np.frombuffer(payload, np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(golden.shape)
    passed = compare(actual, golden)
    perturbed = golden.copy()
    perturbed.flat[0] += max(1.0, abs(float(perturbed.flat[0])))
    negative_rejected = passed and not compare(actual, perturbed)
    report = dict(passed=passed, negative_rejected=negative_rejected,
                  input_sizes=sizes, output_bytes=len(payload),
                  mismatches=int(np.count_nonzero(actual != golden)),
                  algorithm_max_abs=float(np.max(np.abs(actual - golden))),
                  softmax_max_abs=float(np.max(np.abs(actual - exact))),
                  softmax_rmse=float(np.sqrt(np.mean((actual - exact) ** 2))))
    (folder / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if passed and negative_rejected else 1


if __name__ == "__main__":
    raise SystemExit(main())
