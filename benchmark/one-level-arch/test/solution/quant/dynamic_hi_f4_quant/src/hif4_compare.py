#!/usr/bin/env python3
"""Compare DynamicHiF4Quant kernel output vs host golden (reconstruction MSE).

The hi_f4 scale is a bit-packed U32 word (E6M2 | E1_8<<8 | E1_16<<16), so a raw
scale-byte MSE is meaningless. Instead we DECODE both kernel and golden (y nibble
-> hif4 lane value; scale word -> per-lane scale S[q]) and reconstruct
  x_hat[j] = hif4_lane(y[j]) * S[block, q]
then compare kernel reconstruction vs golden reconstruction (primary, should be
~0 since same algorithm) and vs the original input (absolute quant quality).

Files in CHK_DIR:
  output.bin       kernel fp4 packed   (M*N/2 bytes)
  scale_output.bin kernel scale        (M*numKb uint32)
  golden.bin       golden fp4 packed
  scale_golden.bin golden scale
  input.bin        M*N bf16
Pass: MSE(kernel_xhat, golden_xhat) < THRESH (default 0.1).
"""

import argparse
import os
import struct
from pathlib import Path

import numpy as np

THRESH = 0.1
HIF4_LANE_MAG = np.array([0.25 * c for c in range(8)], dtype=np.float64)


def bf16_to_f32(u16: np.ndarray) -> np.ndarray:
    return (u16.astype(np.uint32) << 16).view(np.float32).astype(np.float64)


def unpack_nibbles(byte_arr: np.ndarray, n: int) -> np.ndarray:
    lo = byte_arr & 0xF
    hi = (byte_arr >> 4) & 0xF
    out = np.empty(byte_arr.size * 2, dtype=np.uint8)
    out[0::2] = lo
    out[1::2] = hi
    return out[:n]


def lane_value(nib: np.ndarray) -> np.ndarray:
    sign = np.where((nib >> 3) & 1, -1.0, 1.0)
    return sign * HIF4_LANE_MAG[nib & 0x7]


def e6m2_value(byte: int) -> float:
    if byte == 0xFF:
        return float("nan")
    exp6 = (byte >> 2) & 0x3F
    m2 = byte & 0x3
    return (1.0 + m2 / 4.0) * (2.0 ** (exp6 - 48))


def reconstruct(y_bytes: np.ndarray, scale_words: np.ndarray, M: int, N: int) -> np.ndarray:
    numKb = N // 64
    nib = unpack_nibbles(y_bytes, M * N)
    lanes = lane_value(nib).reshape(M, N)
    xhat = np.zeros((M, N), dtype=np.float64)
    for m in range(M):
        for kb in range(numKb):
            w = int(scale_words[m * numKb + kb])
            base = e6m2_value(w & 0xFF)
            for q in range(64):
                inc = ((w >> (8 + q // 8)) & 1) + ((w >> (16 + q // 4)) & 1)
                xhat[m, kb * 64 + q] = lanes[m, kb * 64 + q] * base * (2.0 ** inc)
    return xhat.reshape(-1)


def scale_field_match(k: np.ndarray, g: np.ndarray) -> str:
    e6_k, e6_g = k & 0xFF, g & 0xFF
    e6_match = float(np.mean(e6_k == e6_g))
    e1_8_match = float(np.mean(((k >> 8) & 0xFF) == ((g >> 8) & 0xFF)))
    e1_16_match = float(np.mean(((k >> 16) & 0xFFFF) == ((g >> 16) & 0xFFFF)))
    return f"e6m2={e6_match:.3f} E1_8={e1_8_match:.3f} E1_16={e1_16_match:.3f}"


def compare(cmp_dir: str, M: int, N: int) -> dict:
    p = Path(cmp_dir)
    yk = np.frombuffer((p / "output.bin").read_bytes(), dtype=np.uint8)
    yg = np.frombuffer((p / "golden.bin").read_bytes(), dtype=np.uint8)
    sk = np.frombuffer((p / "scale_output.bin").read_bytes(), dtype=np.uint32)
    sg = np.frombuffer((p / "scale_golden.bin").read_bytes(), dtype=np.uint32)
    xin = bf16_to_f32(np.frombuffer((p / "input.bin").read_bytes(), dtype=np.uint16))

    res = {}
    if yk.size != yg.size:
        return {"status": "size_mismatch_y", "yk": yk.size, "yg": yg.size}
    if sk.size != sg.size:
        return {"status": "size_mismatch_scale", "sk": sk.size, "sg": sg.size}

    xhat_k = reconstruct(yk, sk, M, N)
    xhat_g = reconstruct(yg, sg, M, N)

    diff_kg = xhat_k - xhat_g
    mse_kg = float(np.mean(diff_kg ** 2))
    mae_kg = float(np.max(np.abs(diff_kg)))
    mse_kin = float(np.mean((xhat_k - xin) ** 2))
    mse_gin = float(np.mean((xhat_g - xin) ** 2))

    ymatch = float(np.mean(yk == yg))
    res.update({
        "status": "pass" if mse_kg < THRESH else "fail",
        "mse_kernel_vs_golden": mse_kg,
        "maxae_kernel_vs_golden": mae_kg,
        "mse_kernel_vs_input": mse_kin,
        "mse_golden_vs_input": mse_gin,
        "y_byte_match": ymatch,
        "scale_field_match": scale_field_match(sk, sg),
    })
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cmp-dir", required=True)
    ap.add_argument("--M", type=int, default=32)
    ap.add_argument("--N", type=int, default=128)
    args = ap.parse_args()
    r = compare(args.cmp_dir, args.M, args.N)
    print(f"[hif4] status={r.get('status')}")
    for k, v in r.items():
        if k == "status":
            continue
        print(f"    {k} = {v}")


if __name__ == "__main__":
    main()
