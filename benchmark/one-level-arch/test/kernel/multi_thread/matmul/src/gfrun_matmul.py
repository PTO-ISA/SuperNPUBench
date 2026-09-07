#!/usr/bin/python3
"""Numerical verification for the matmul operators, modeled on
gfrun_flashMLA.py.

Covers the two matmul operators under benchmark/one-level-arch/test/kernel:

  * multi_thread/matmul  -> four-PE GMMA matmul (C = A * B, fp32)
  * matmul (TYPE=MASK)   -> single-thread tiled matmul (MASK_FP32 / _FP16 / _FP8)

Both compute C = A * B with A:[B,M,K], B:[B,K,N], C:[B,M,N] row-major.
The multi_thread variant splits the M rows across four PEs
(get_thread_idx), but the host-visible storage and the golden reference
are the full [B,M,N] result. The output (res.bin) is always float32.

The golden reference is computed with **pytorch** (torch.matmul). If torch
is not installed the script falls back to numpy and prints a warning; install
torch (`pip install torch`) for the pytorch reference.

Per ELF the script:
  1. parses B/M/N/K/tM/tN/tK, the input dtype and the thread count from the
     ELF name;
  2. generates random input (src0.bin = A, src1.bin = B) and golden.bin
     (= A @ B in fp32, with the input dtype's quantization applied first);
  3. pre-allocates res.bin (the guest cannot reliably create files);
  4. runs gfrun on the ELF (the multi_thread variant gets
     `-s softcore.multiThreadNum=4`; single-threaded MASK uses `gfrun -f`);
  5. compares res.bin vs golden.bin with np.allclose (atol/rtol) and
     reports mse / max-abs, flagging an all-zero res.bin (RES_CHECK off?).

Note: the kernel only performs file I/O when built with `res_check=on`
(see test/common/Makefile.common). ELFs built without it leave res.bin at
zero; the script reports that as a "no output / RES_CHECK off?" failure.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

# ---------------------------------------------------------------------------
# Optional pytorch golden reference (preferred).
# ---------------------------------------------------------------------------
try:
    import torch  # type: ignore

    _HAS_TORCH = True
except Exception:  # pragma: no cover - environment dependent
    torch = None
    _HAS_TORCH = False

if not _HAS_TORCH:
    print(
        "[gfrun_matmul] torch not found -> using numpy golden reference.\n"
        "                Install pytorch (`pip install torch`) for the "
        "pytorch reference path.",
        file=sys.stderr,
    )

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))


def _find_root(start):
    """Walk up from *start* until a directory contains both test/ and
    kernels/ (i.e. the one-level-arch root)."""
    d = start
    for _ in range(10):
        if os.path.isdir(os.path.join(d, "test")) and os.path.isdir(
            os.path.join(d, "kernels")
        ):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return None


ROOT = _find_root(SCRIPT_DIR)
if ROOT is None:
    # Fallback: ../../../.. relative to this src dir.
    ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../../.."))
CMP_ROOT = os.path.join(ROOT, "compare")

DEFAULT_GFRUN = "/Users/blacktraker/Programming/gitproj/DV4/SuperScalarModel/bin/gfrun"
# NOTE: gfrun's "-t" flag is a LOG LEVEL (1: basic per-instruction trace,
# 2: tile detail), NOT a thread/PE count. Passing it makes gfrun print every
# instruction, which is fatal for non-trivial shapes (millions of lines).
# Follow the rms_norm golden example: run with just "-f <elf>" and let gfrun
# use its default PE/SIMT config.  (model: test/kernel/norm/src/rms_norm_golden.py)
DEFAULT_GFRUN_ARGS = " -f "
DEFAULT_TIMEOUT = 1200
MAX_WORKERS = min(8, (os.cpu_count() or 4))

# Input-dtype -> numpy dtype and element size used for the .bin layout.
_DTYPE_MAP = {
    "fp32": (np.float32, 4),
    "fp16": (np.float16, 2),
    "bf16": (np.uint16, 2),  # bfloat16 stored as raw uint16 bits
    "fp8": (np.uint8, 1),  # e4m3 stored as raw bytes
    "hif8": (np.uint8, 1),
    "mxfp8": (np.uint8, 1),
    "hif4x2": (np.uint8, 1),  # two logical values per byte
    "mxfp4": (np.uint8, 1),   # two logical values per byte
}

# atol/rtol defaults. fp8 accumulates more error; allow more slack.
_TOL = {
    "fp32": (2e-2, 2e-2),
    "fp16": (2e-2, 2e-2),
    "bf16": (2e-2, 2e-2),
    "fp8": (5e-2, 5e-2),
    "hif8": (5e-2, 5e-2),
    "mxfp8": (5e-2, 5e-2),
    "hif4x2": (5e-2, 5e-2),
    "mxfp4": (5e-2, 5e-2),
}

statics = {"pass": [], "fail": []}


# ---------------------------------------------------------------------------
# Shape / variant parsing from the ELF name
# ---------------------------------------------------------------------------
def extract_int(name, elf_name, default=None):
    m = re.search(rf"{name}(\d+)", elf_name)
    return int(m.group(1)) if m else default


def parse_matmul_shape(elf_name, args):
    """Parse the matmul shape / variant out of an ELF basename (no .elf).

    Recognised ELF name patterns (see the two Makefiles):

      multi_thread:  ..._multi_thread_matmul_B{B}_M{M}_N{N}_K{K}_tM_tN_tK
      kernel MASK:   matmul_MASK_{MODE}_M{M}_N{N}_K{K}_tM_tN_tK

    Returns a dict or None if the ELF is not one we handle.
    """
    base = elf_name.replace(".elf", "").strip()

    # Skip the legacy/deleted matmul_local / matmul_partial / matmul_gmma
    # variants - they are not produced by the two target cpp files.
    if any(
        tok in base
        for tok in ("matmul_local", "matmul_partial", "matmul_gmma", "matmul_fixp")
    ):
        return None

    multi_thread = "multi_thread_matmul" in base

    if multi_thread:
        lowp_match = re.search(
            r"matmul_lowp_(FP8|HIFP8|MXFP8|HIF4X2|MXFP4)_B\d", base
        )
        mode = lowp_match.group(1) if lowp_match else None
        lowp_dtypes = {
            "FP8": ("fp8", 1, False),
            "HIFP8": ("hif8", 1, False),
            "MXFP8": ("mxfp8", 1, True),
            "HIF4X2": ("hif4x2", 2, True),
            "MXFP4": ("mxfp4", 2, True),
        }
        if mode is not None:
            dtype, packed_factor, use_mx = lowp_dtypes[mode]
        elif "_DType__bf16" in base:
            dtype = "bf16"
            packed_factor, use_mx = 1, False
        elif "_DType__half" in base:
            dtype = "fp16"
            packed_factor, use_mx = 1, False
        else:
            dtype = "fp32"
            packed_factor, use_mx = 1, False
        threads = 4  # kPeNum = 4
        b = extract_int("B", base, default=1)
    else:
        # kernel/matmul TYPE=MASK: name is "matmul_MASK_<MODE>_M..". The MODE
        # token (e.g. MASK_FP32, MASK_FP16_REUSEA, MX_FP8) may itself contain
        # underscores, so anchor on the trailing shape marker "_M<digits>".
        m = re.search(r"matmul_MASK_(.+?)_M\d", base)
        mode = m.group(1) if m else None
        if mode is None:
            return None
        if mode.startswith("MASK_FP16"):
            dtype = "fp16"
        elif mode.startswith("MASK_FP8") or mode == "MX_FP8":
            dtype = "fp8"
        elif mode.startswith("MASK_FP32"):
            dtype = "fp32"
        else:
            return None  # unsupported MASK variant
        threads = 1
        b = 1  # MASK Makefile does not pass -DBatch; defaults to 1.
        packed_factor, use_mx = 1, mode == "MX_FP8"

    shape = {
        "B": b,
        "M": extract_int("M", base),
        "N": extract_int("N", base),
        "K": extract_int("K", base),
        "tM": extract_int("tM", base),
        "tN": extract_int("tN", base),
        "tK": extract_int("tK", base),
        "dtype": dtype,
        "multi_thread": multi_thread,
        "mode": mode,
        "threads": threads,
        "packed_factor": packed_factor,
        "use_mx": use_mx,
    }
    if any(shape[k] is None for k in ("M", "N", "K", "tM", "tN", "tK")):
        return None
    return shape


# ---------------------------------------------------------------------------
# fp8 e4m3 quantization (numpy fallback only; torch uses torch.float8_e4m3fn)
# ---------------------------------------------------------------------------
def _dequant_fp8_e4m3_np(bits):
    """Dequantize fp8 e4m3 raw bytes (uint8) -> float32 numpy (numpy path)."""
    try:
        import ml_dtypes  # type: ignore

        return bits.view(ml_dtypes.float8_e4m3).astype(np.float32)
    except Exception:
        bits = bits.astype(np.int64)
        sign = (bits >> 7) & 1
        exp = (bits >> 3) & 0xF
        mant = bits & 0x7
        normal = exp != 0
        val_normal = (1.0 + mant.astype(np.float32) / 8.0) * np.where(
            normal, np.power(2.0, (exp - 7).astype(np.float32)), 1.0
        )
        val_sub = (mant.astype(np.float32) / 8.0) * np.power(2.0, -6.0)
        val = np.where(normal, val_normal, val_sub)
        val = np.where(sign.astype(bool), -val, val)
        return val.astype(np.float32)


def _quantize_fp8_e4m3_np(x):
    """fp8 e4m3 quantization via ml_dtypes if available, else manual RTE."""
    try:
        import ml_dtypes  # type: ignore

        return x.astype(ml_dtypes.float8_e4m3).view(np.uint8)
    except Exception:
        pass
    # Manual e4m3 (round-to-nearest-even, saturate to +-448). Only used when
    # neither torch nor ml_dtypes is available.
    x = np.clip(np.asarray(x, dtype=np.float32), -448.0, 448.0)
    sign = (x < 0)
    ax = np.abs(x).astype(np.float32)
    bits = ax.view(np.uint32).astype(np.int64)
    exp32 = ((bits >> 23) & 0xFF) - 127
    mant32 = bits & 0x7FFFFF
    is_sub = exp32 == -127
    sig_n = (np.int64(1) << 23) | mant32
    sig_s = mant32
    sig = np.where(is_sub, sig_s, sig_n)
    e_eff = np.where(is_sub, np.int64(-126), exp32) + 7
    drop = 20
    half = np.int64(1) << (drop - 1)
    mask_low = (np.int64(1) << drop) - 1
    low = sig & mask_low
    kept_lsb = (sig >> drop) & 1
    round_up = (low > half) | ((low == half) & (kept_lsb == 1))
    q = (sig >> drop) + round_up.astype(np.int64)
    carry = q >> 4
    mant4 = q & 0x7
    e2 = e_eff + carry
    sat = e2 >= 16
    e2 = np.where(sat, 15, e2)
    mant4 = np.where(sat, 7, mant4)
    sub8 = e2 <= 0
    sh = (7 - e_eff) + (23 - 3)
    sh = np.maximum(sh, 0)
    sig_u = sig.astype(np.uint64)
    half_s = np.uint64(1) << np.where(sub8 & (sh > 0), sh - 1, 0)
    low_s = sig_u & ((np.uint64(1) << sh) - 1)
    kept = sig_u >> sh
    kept_lsb_s = kept & 1
    ru = (low_s > half_s) | ((low_s == half_s) & (kept_lsb_s == 1))
    m_sub = (kept + ru.astype(np.uint64)) & 0x7
    mant4 = np.where(sub8, m_sub.astype(np.int64), mant4)
    e2 = np.where(sub8, 0, e2)
    out = ((sign.astype(np.int64) << 7) | (e2 << 3) | mant4).astype(np.uint8)
    return out


def _quantize_fp8_e4m3(x_np):
    """fp8 e4m3 quantization -> uint8 raw bytes (torch preferred)."""
    if _HAS_TORCH:
        t = torch.from_numpy(x_np.astype(np.float32))
        q = t.to(torch.float8_e4m3fn)
        return q.view(torch.uint8).numpy().copy()
    return _quantize_fp8_e4m3_np(x_np)


# ---------------------------------------------------------------------------
# Golden reference (pytorch preferred)
# ---------------------------------------------------------------------------
def matmul_reference(a_np, b_np, dtype):
    """Compute the golden C = A @ B in float32.

    Inputs are first quantized to the operator's input dtype (matching what
    the kernel loads), then the matmul is performed in float32 - this is what
    both kernels do (fp32 accumulator).
    """
    if _HAS_TORCH:
        a = torch.from_numpy(a_np)  # float32 [B,M,K]
        b = torch.from_numpy(b_np)  # float32 [B,K,N]
        if dtype == "fp16":
            a = a.to(torch.float16).to(torch.float32)
            b = b.to(torch.float16).to(torch.float32)
        elif dtype == "bf16":
            a = a.to(torch.bfloat16).to(torch.float32)
            b = b.to(torch.bfloat16).to(torch.float32)
        elif dtype == "fp8":
            a = a.to(torch.float8_e4m3fn).to(torch.float32)
            b = b.to(torch.float8_e4m3fn).to(torch.float32)
        golden = torch.matmul(a, b)  # [B,M,N] float32
        return golden.numpy()

    # numpy fallback
    a = a_np
    b = b_np
    if dtype == "fp16":
        a = a.astype(np.float16).astype(np.float32)
        b = b.astype(np.float16).astype(np.float32)
    elif dtype == "bf16":
        a = _dequantize_bf16_np(_quantize_bf16_np(a))
        b = _dequantize_bf16_np(_quantize_bf16_np(b))
    elif dtype == "fp8":
        a = _dequant_fp8_e4m3_np(_quantize_fp8_e4m3_np(a)).astype(np.float32)
        b = _dequant_fp8_e4m3_np(_quantize_fp8_e4m3_np(b)).astype(np.float32)
    return np.matmul(a, b).astype(np.float32)


def _quantize_bf16_np(x):
    """Round float32 to bfloat16 and return the raw uint16 payload."""
    bits = np.asarray(x, dtype=np.float32).view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)
    return (rounded >> 16).astype(np.uint16)


def _dequantize_bf16_np(bits):
    """Expand raw bfloat16 uint16 payload to float32."""
    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << 16).view(
        np.float32
    )


# ---------------------------------------------------------------------------
# Input generation
# ---------------------------------------------------------------------------
def gen_input_and_golden(elf_name, path, args):
    """Write src0.bin, src1.bin and golden.bin into *path*, and pre-allocate
    res.bin (zero-filled) so the guest can write without O_CREAT."""
    shape = parse_matmul_shape(elf_name, args)
    if shape is None:
        raise ValueError(f"cannot parse matmul shape from ELF name: {elf_name}")

    B, M, N, K = shape["B"], shape["M"], shape["N"], shape["K"]
    dtype = shape["dtype"]

    rng = np.random.default_rng(args.seed)
    if dtype in ("fp8", "hif8", "mxfp8", "hif4x2", "mxfp4"):
        # Use non-zero values that are represented exactly by every tested
        # low-precision format. This exercises sign decoding, packed-lane
        # ordering and MX scales without making the reference depend on a
        # separate approximation of each format's rounding rules.
        a = np.where(
            rng.integers(0, 2, size=(B, M, K), dtype=np.uint8), 1.0, -1.0
        ).astype(np.float32)
        b = np.where(
            rng.integers(0, 2, size=(B, K, N), dtype=np.uint8), 1.0, -1.0
        ).astype(np.float32)
        golden = np.matmul(a, b).astype(np.float32)
    else:
        # Same distribution as gfrun_flashMLA: N(0,1)/10 clipped to [-1,1].
        a = (rng.standard_normal((B, M, K), dtype=np.float32) / 10.0).clip(-1.0, 1.0)
        b = (rng.standard_normal((B, K, N), dtype=np.float32) / 10.0).clip(-1.0, 1.0)
        golden = matmul_reference(a, b, dtype)  # [B,M,N] float32

    # Inputs written in the operator's input-dtype layout.
    if dtype == "fp16":
        a_q = a.astype(np.float16)
        b_q = b.astype(np.float16)
    elif dtype == "bf16":
        a_q = _quantize_bf16_np(a)
        b_q = _quantize_bf16_np(b)
    elif dtype in ("fp8", "mxfp8"):
        a_q = _quantize_fp8_e4m3(a)
        b_q = _quantize_fp8_e4m3(b)
    elif dtype == "hif8":
        # PTO HiF8: D0 payload 0x08 is +1.0; setting the sign bit gives -1.0.
        a_q = np.where(a > 0, 0x08, 0x88).astype(np.uint8)
        b_q = np.where(b > 0, 0x08, 0x88).astype(np.uint8)
    elif dtype in ("hif4x2", "mxfp4"):
        # Finite-only nibble tables used by CubeEngine:
        # HiF4 index 4 == 1.0; E2M1 index 2 == 1.0; bit 3 is the sign.
        one = 0x04 if dtype == "hif4x2" else 0x02
        a_lane = np.where(a > 0, one, one | 0x08).astype(np.uint8)
        b_lane = np.where(b > 0, one, one | 0x08).astype(np.uint8)
        # Packed storage is [M,K/2] and [K/2,N]: low nibble is logical
        # K=2k, high nibble is logical K=2k+1.
        a_q = a_lane[:, :, 0::2] | (a_lane[:, :, 1::2] << 4)
        b_q = b_lane[:, 0::2, :] | (b_lane[:, 1::2, :] << 4)
    else:
        a_q = a.astype(np.float32)
        b_q = b.astype(np.float32)

    a_q.tofile(os.path.join(path, "src0.bin"))
    b_q.tofile(os.path.join(path, "src1.bin"))
    if shape["use_mx"]:
        if dtype == "hif4x2":
            # HiF4X2 is Matrix-MX-only. One raw U32 word scales 64 logical
            # lanes. 0x000000c0 encodes E6M2 1.0 with all E1 bits clear.
            np.full((B, M, K // 64), 0x000000C0, dtype=np.uint32).tofile(
                os.path.join(path, "src0_scale.bin")
            )
            np.full((B, K // 64, N), 0x000000C0, dtype=np.uint32).tofile(
                os.path.join(path, "src1_scale.bin")
            )
        else:
            # E8M0 0x7f decodes to 2^(127-127) == 1.0. Scale tensors follow
            # [B,M,K/32] for A and [B,K/32,N] for B.
            np.full((B, M, K // 32), 0x7F, dtype=np.uint8).tofile(
                os.path.join(path, "src0_scale.bin")
            )
            np.full((B, K // 32, N), 0x7F, dtype=np.uint8).tofile(
                os.path.join(path, "src1_scale.bin")
            )
    golden.tofile(os.path.join(path, "golden.bin"))

    # Pre-allocate res.bin (zero). Guest cannot reliably create files.
    res = np.zeros((B, M, N), dtype=np.float32)
    res.tofile(os.path.join(path, "res.bin"))
    return shape


# ---------------------------------------------------------------------------
# Run / compare
# ---------------------------------------------------------------------------
def run_qemu(elf, args):
    elf_name = os.path.basename(elf).replace(".elf", "").strip()
    shape = parse_matmul_shape(elf_name, args)

    # gfrun's PE/SIMT count is a softcore config value, NOT a CLI flag like
    # "-t" ("-t" is a LOG LEVEL -- see the note on DEFAULT_GFRUN_ARGS). The
    # multi_thread matmul splits M across kPeNum=4 PEs and uses a cooperative
    # TMATMUL that requires ALL 4 PEs to publish (definedMask == 0xf). With the
    # default 1 PE gfrun aborts: "cooperative TMATMUL requires a fully-defined
    # Shared Right". So for the multi_thread variant we inject
    # `-s softcore.multiThreadNum=4` (skipped if the caller already supplied
    # it via --gfrun-args). Single-threaded MASK matmul runs with plain
    # `gfrun -f <elf>`.
    gfrun_args = args.gfrun_args
    if shape is not None and shape.get("multi_thread"):
        pe = shape.get("threads", 4)
        if "softcore.multiThreadNum" not in gfrun_args:
            gfrun_args = f" -s softcore.multiThreadNum={pe} {gfrun_args} "
    cmd = f"{args.gfrun} {gfrun_args} {elf}"
    print(f"[run] {cmd}")
    try:
        proc = subprocess.Popen(
            cmd, shell=True, preexec_fn=os.setsid,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
    except Exception as e:
        return elf, "fail", f"Popen error: {e}"

    try:
        out, _ = proc.communicate(timeout=args.timeout)
        output = out.decode(errors="replace")
        status = "pass" if proc.returncode == 0 else "fail"
    except subprocess.TimeoutExpired:
        # Drain whatever gfrun printed before we kill it, so a hang leaves a
        # useful breadcrumb (where it stopped) instead of a bare "timeout".
        partial = ""
        try:
            os.killpg(os.getpgid(proc.pid), 9)
        except Exception:
            pass
        try:
            out2, _ = proc.communicate(timeout=5)
            partial = out2.decode(errors="replace") if out2 else ""
        except Exception:
            pass
        tail = "\n".join(partial.splitlines()[-20:]) if partial.strip() \
            else "killed after timeout (no output)"
        return elf, "timeout", tail

    tail = "\n".join(output.splitlines()[-8:])
    return elf, status, tail


def compare_array(cmp_data, golden_data, dtype, atol, rtol):
    res = np.fromfile(cmp_data, dtype=np.float32)
    ref = np.fromfile(golden_data, dtype=np.float32)
    if res.size != ref.size:
        return "shape mismatch", {
            "res_size": int(res.size),
            "golden_size": int(ref.size),
            "mse": float("nan"),
            "max_abs": float("nan"),
        }
    if not np.any(res):
        # All-zero result: the kernel did not write (RES_CHECK off / crash).
        return "no output (res.bin all zero; RES_CHECK off or run crashed?)", {
            "mse": float("nan"),
            "max_abs": float("nan"),
        }
    diff = res - ref
    mse = float(np.mean(diff * diff))
    max_abs = float(np.max(np.abs(diff)))
    close = bool(np.allclose(res, ref, atol=atol, rtol=rtol))
    chk = "pass" if close else "fail"
    return chk, {"mse": mse, "max_abs": max_abs}


def result_compare(cmp_data_path, args):
    res_path = os.path.join(cmp_data_path, "res.bin")
    golden_path = os.path.join(cmp_data_path, "golden.bin")
    if not os.path.exists(res_path):
        return "res.bin not exist", {"mse": float("nan"), "max_abs": float("nan")}

    elf_name = os.path.basename(cmp_data_path)
    shape = parse_matmul_shape(elf_name, args)
    dtype = shape["dtype"] if shape else "fp32"
    atol, rtol = _TOL.get(dtype, (2e-2, 2e-2))
    return compare_array(res_path, golden_path, dtype, atol, rtol)


def dump_debug_report(cmp_data_path, elf_name, args):
    res_path = os.path.join(cmp_data_path, "res.bin")
    golden_path = os.path.join(cmp_data_path, "golden.bin")
    if not (os.path.exists(res_path) and os.path.exists(golden_path)):
        return None
    res = np.fromfile(res_path, dtype=np.float32)
    ref = np.fromfile(golden_path, dtype=np.float32)
    n = min(8, res.size, ref.size)
    lines = [f"[debug] {elf_name} first {n} elements:"]
    for i in range(n):
        lines.append(f"  res[{i}]={res[i]:.6f}  golden[{i}]={ref[i]:.6f}")
    debug_log = os.path.join(cmp_data_path, "debug_compare.log")
    with open(debug_log, "w") as f:
        f.write("\n".join(lines) + "\n")
    return debug_log


def check_elf(elf, args):
    print("Start to check elf----------------")
    elf_name = os.path.basename(elf).replace(".elf", "").strip()
    cmp_data_path = os.path.join(CMP_ROOT, elf_name)

    if os.path.isdir(cmp_data_path):
        shutil.rmtree(cmp_data_path)
    os.makedirs(cmp_data_path, exist_ok=True)

    try:
        shape = gen_input_and_golden(elf_name, cmp_data_path, args)
    except ValueError as e:
        return elf_name, "skip", f"not a matmul ELF: {e}", {"mse": float("nan"), "max_abs": float("nan")}
    print(f"[shape] {shape}")

    _, status, output = run_qemu(elf, args)
    if status != "pass":
        return elf_name, status, "not chk", {"mse": float("nan"), "max_abs": float("nan"), "run_output": output}

    res_path = os.path.join(cmp_data_path, "res.bin")
    if not os.path.exists(res_path):
        return elf_name, status, "output not exist", {"mse": float("nan"), "max_abs": float("nan")}

    chk_status, metric = result_compare(cmp_data_path, args)
    debug_log = dump_debug_report(cmp_data_path, elf_name, args)
    if debug_log is not None:
        print("debug compare log:", debug_log)
    return elf_name, status, chk_status, metric


def log_result(elf, status, chk_status, metric):
    item = f"{elf} -> run_status: {status.upper()} chk_status: {chk_status.upper()} metric:{metric}"
    if status == "pass" and chk_status == "pass":
        statics["pass"].append(item)
    else:
        statics["fail"].append(item)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run matmul gfrun result check")
    parser.add_argument("-l", dest="elf_list", default="./tmp.list", type=str,
                        help="file listing ELF paths, one per line")
    parser.add_argument("-d", dest="dbg_elf", default=None, type=str,
                        help="single ELF to check (debug)")
    parser.add_argument("-o", dest="res_log", default="matmul_result_check.log", type=str)
    parser.add_argument("--gfrun", default=DEFAULT_GFRUN, type=str)
    parser.add_argument("--gfrun-args", default=DEFAULT_GFRUN_ARGS, type=str,
                        help="extra gfrun flags (default ' -f '); for the "
                             "multi_thread variant -s softcore.multiThreadNum=4 "
                             "is injected automatically")
    parser.add_argument("--timeout", default=DEFAULT_TIMEOUT, type=int)
    parser.add_argument("--seed", default=123, type=int)
    parser.add_argument("--out-atol", default=None, type=float)
    parser.add_argument("--out-rtol", default=None, type=float)
    parser.add_argument("--workers", default=MAX_WORKERS, type=int)
    args = parser.parse_args()

    # Allow CLI atol/rtol override for fp32 (other dtypes keep their table).
    if args.out_atol is not None:
        _TOL["fp32"] = (args.out_atol, _TOL["fp32"][1])
    if args.out_rtol is not None:
        _TOL["fp32"] = (_TOL["fp32"][0], args.out_rtol)

    if args.dbg_elf is not None:
        elf_paths = [args.dbg_elf]
    else:
        with open(args.elf_list, "r") as f:
            elf_paths = [line.strip() for line in f if line.strip()]

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(check_elf, elf, args): elf for elf in elf_paths}
        for future in as_completed(futures):
            elf, status, chk_status, metric = future.result()
            print(f"{elf} -> run_status: {status.upper()} chk_status: {chk_status.upper()} metric:{metric}")
            log_result(elf, status, chk_status, metric)

    os.makedirs(CMP_ROOT, exist_ok=True)
    with open(os.path.join(CMP_ROOT, args.res_log), "w") as f:
        f.write("\nResult_Check Summary:\n")
        f.write(f"\npass : {len(statics['pass'])}\n")
        f.write(f"\nfail : {len(statics['fail'])}\n")
        f.write("\npass list:\n")
        for item in statics["pass"]:
            f.write(f"{item}\n")
        f.write("\n\nfail list:\n")
        for item in statics["fail"]:
            f.write(f"{item}\n")
    print(f"\n[summary] pass={len(statics['pass'])} fail={len(statics['fail'])} "
          f"log={os.path.join(CMP_ROOT, args.res_log)}")
    sys.exit(1 if statics["fail"] else 0)
