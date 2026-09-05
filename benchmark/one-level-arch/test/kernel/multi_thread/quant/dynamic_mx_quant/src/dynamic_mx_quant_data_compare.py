#!/usr/bin/env python3
"""Compare DynamicMxQuant output vs golden.

Reads output.bin / scale_output.bin and golden.bin / scale_golden.bin
from the compare directory, computes MSE and max absolute error.
"""

import argparse
import os
import struct
import sys
from pathlib import Path

import numpy as np


def read_fp8_e4m3_bin(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        data = f.read()
    return np.frombuffer(data, dtype=np.uint8)


def read_bf16_bin(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        data = f.read()
    raw = np.frombuffer(data, dtype=np.uint16)
    f32 = raw.astype(np.uint32) << 16
    return f32.view(np.float32)


def read_uint16_bin(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        data = f.read()
    return np.frombuffer(data, dtype=np.uint16)


def fp8_e4m3_to_f32(byte_val: int) -> float:
    sign = (byte_val >> 7) & 1
    exp = (byte_val >> 3) & 0xF
    mant = byte_val & 0x7
    if exp == 0:
        if mant == 0:
            return -0.0 if sign else 0.0
        val = (mant / 8.0) * (2**-6)
    elif exp == 0xF and mant == 0x7:
        return float("nan")
    else:
        val = (1.0 + mant / 8.0) * (2 ** (exp - 7))
    return -val if sign else val


# E2M1 representable magnitudes indexed by 3-bit code (sign is bit3). Mirrors
# gen_dynamic_mx_quant_data.FP4_E2M1_MAG.
FP4_E2M1_MAG = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def fp4_e2m1_nibble_to_f32(nibble: int) -> float:
    sign = (nibble >> 3) & 1
    code = nibble & 0x7
    val = FP4_E2M1_MAG[code]
    return -val if sign else val


def unpack_fp4_nibbles(byte_arr: np.ndarray) -> np.ndarray:
    """2 nibbles/byte, low nibble = even element (matches gen pack_output)."""
    lo = byte_arr & 0xF
    hi = (byte_arr >> 4) & 0xF
    out = np.empty(byte_arr.size * 2, dtype=np.uint8)
    out[0::2] = lo
    out[1::2] = hi
    return out


def compare_fp8(output_path: str, golden_path: str) -> tuple[str, float, float]:
    out_bytes = read_fp8_e4m3_bin(output_path)
    gold_bytes = read_fp8_e4m3_bin(golden_path)
    if len(out_bytes) != len(gold_bytes):
        return "size_mismatch", float("nan"), float("nan")
    out_f32 = np.array([fp8_e4m3_to_f32(int(b)) for b in out_bytes], dtype=np.float32)
    gold_f32 = np.array([fp8_e4m3_to_f32(int(b)) for b in gold_bytes], dtype=np.float32)
    diff = out_f32 - gold_f32
    mse = float(np.mean(diff**2))
    max_ae = float(np.max(np.abs(diff)))
    status = "pass" if mse < 0.1 else "fail"
    return status, mse, max_ae


def compare_fp4(output_path: str, golden_path: str) -> tuple[str, float, float]:
    out_bytes = read_fp8_e4m3_bin(output_path)   # raw uint8 reader
    gold_bytes = read_fp8_e4m3_bin(golden_path)
    if len(out_bytes) != len(gold_bytes):
        return "size_mismatch", float("nan"), float("nan")
    out_nib = unpack_fp4_nibbles(out_bytes)
    gold_nib = unpack_fp4_nibbles(gold_bytes)
    out_f32 = np.array([fp4_e2m1_nibble_to_f32(int(n)) for n in out_nib], dtype=np.float32)
    gold_f32 = np.array([fp4_e2m1_nibble_to_f32(int(n)) for n in gold_nib], dtype=np.float32)
    diff = out_f32 - gold_f32
    mse = float(np.mean(diff**2))
    max_ae = float(np.max(np.abs(diff)))
    status = "pass" if mse < 0.1 else "fail"
    return status, mse, max_ae


def compare_scale(output_path: str, golden_path: str) -> tuple[str, float, float]:
    out_u16 = read_uint16_bin(output_path)
    gold_u16 = read_uint16_bin(golden_path)
    if len(out_u16) != len(gold_u16):
        return "size_mismatch", float("nan"), float("nan")
    diff = out_u16.astype(np.int32) - gold_u16.astype(np.int32)
    mse = float(np.mean(diff.astype(np.float64)**2))
    max_ae = float(np.max(np.abs(diff)))
    status = "pass" if mse < 0.1 else "fail"
    return status, mse, max_ae


def compare_scale_compact(output_path: str, golden_path: str) -> tuple[str, float, float]:
    """AscendC-aligned compact uint8 E8M0 scale. Padding columns are zero in both
    golden and (zero-inited) output buffers, so a direct byte compare is exact."""
    out_u8 = read_fp8_e4m3_bin(output_path)   # raw uint8 reader
    gold_u8 = read_fp8_e4m3_bin(golden_path)
    if len(out_u8) != len(gold_u8):
        return "size_mismatch", float("nan"), float("nan")
    diff = out_u8.astype(np.int32) - gold_u8.astype(np.int32)
    mse = float(np.mean(diff.astype(np.float64)**2))
    max_ae = float(np.max(np.abs(diff)))
    status = "pass" if mse < 0.1 else "fail"
    return status, mse, max_ae


def check_elf(elf_path: str, cmp_root: str, dtype: str = "FP8",
              scale_layout: str = "broadcast") -> dict:
    elf_name = os.path.basename(elf_path).replace(".elf", "").strip()
    cmp_dir = os.path.join(cmp_root, elf_name)
    result = {"elf": elf_name, "output": "N/A", "scale": "N/A"}
    out_path = os.path.join(cmp_dir, "output.bin")
    gold_path = os.path.join(cmp_dir, "golden.bin")
    scale_out_path = os.path.join(cmp_dir, "scale_output.bin")
    scale_gold_path = os.path.join(cmp_dir, "scale_golden.bin")
    # dtype defaults to FP8 but is inferred from the ELF/driver name when unset.
    compare_out = compare_fp4 if dtype.upper() == "FP4" or elf_name.endswith("fp4") else compare_fp8
    compare_sc = compare_scale_compact if scale_layout == "compact" else compare_scale
    if os.path.exists(out_path) and os.path.exists(gold_path):
        status, mse, max_ae = compare_out(out_path, gold_path)
        result["output"] = f"{status} (MSE={mse:.6f}, MaxAE={max_ae:.6f})"
    else:
        result["output"] = "missing files"
    if os.path.exists(scale_out_path) and os.path.exists(scale_gold_path):
        status, mse, max_ae = compare_sc(scale_out_path, scale_gold_path)
        result["scale"] = f"{status} (MSE={mse:.6f}, MaxAE={max_ae:.6f})"
    else:
        result["scale"] = "missing files"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-d", dest="dbg_elf", default=None,
                        help="check single ELF")
    parser.add_argument("-l", dest="elf_list", default=None,
                        help="ELF list file")
    parser.add_argument("--cmp-root", dest="cmp_root",
                        default=os.path.abspath(os.path.dirname(__file__) + "/../../../compare"),
                        help="compare root directory")
    parser.add_argument("--dtype", default="FP8", choices=["FP8", "FP4"],
                        help="output element dtype (selects fp8 vs fp4 decode)")
    parser.add_argument("--scale-layout", default="broadcast",
                        choices=["broadcast", "compact"],
                        help="broadcast=legacy uint16 per-elem; compact=AscendC uint8 [M,scaleCols]")
    parser.add_argument("-o", dest="res_log", default="result_check.log",
                        help="result log name")
    args = parser.parse_args()

    if args.dbg_elf:
        elf_paths = [args.dbg_elf]
    elif args.elf_list:
        with open(args.elf_list) as f:
            elf_paths = [l.strip() for l in f if l.strip()]
    else:
        print("provide -d or -l", file=sys.stderr)
        sys.exit(1)

    results = []
    for elf in elf_paths:
        r = check_elf(elf, args.cmp_root, args.dtype, args.scale_layout)
        results.append(r)
        print(f"{r['elf']}: output={r['output']}, scale={r['scale']}")

    if not args.dbg_elf:
        log_path = os.path.join(args.cmp_root, args.res_log)
        with open(log_path, "w") as f:
            f.write("DynamicMxQuant Precision Check\n")
            f.write("=" * 60 + "\n")
            for r in results:
                f.write(f"{r['elf']}: output={r['output']}, scale={r['scale']}\n")


if __name__ == "__main__":
    main()
