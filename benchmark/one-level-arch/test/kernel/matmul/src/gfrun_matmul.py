#!/usr/bin/python3
"""Numerical verification for the kernel/matmul operators, modeled on
gfrun_flashMLA.py.

Covers the TYPE=MASK matmul variants under
benchmark/one-level-arch/test/kernel/matmul (matmul.cpp):

  * MASK_FP32*        -> float       in, float out
  * MASK_FP16*        -> __half      in, float out
  * MASK_FP8*         -> __fp8_e4m3  in, float out
  (MX_FP8 uses microscaling factors not supplied through the file interface,
   so a plain torch.matmul golden would not match; it is skipped.)

All variants compute C = A * B with A:[M,K], B:[K,N], C:[M,N] row-major
(Batch=1). The output (res.bin) is always float32 [M,N]. The inputs
(src0.bin = A, src1.bin = B) are written in the operator's input-dtype
layout (fp32 / fp16 / fp8-e4m3 raw bytes).

The golden reference is computed with **pytorch** (torch.matmul); if torch is
not installed the script falls back to numpy (np.matmul) and prints a warning.
Inputs are first quantized to the operator's input dtype -- exactly what the
kernel loads -- then the matmul runs in float32 (fp32 accumulator), matching
the kernel.

Per ELF the script (same flow as gfrun_flashMLA.py):
  1. parses M/N/K/tM/tN/tK, the MASK mode and input dtype from the ELF name;
  2. generates random input (src0.bin = A, src1.bin = B) and golden.bin
     (= A @ B in fp32, with the input dtype's quantization applied first);
  3. pre-allocates res.bin (the guest cannot reliably create files);
  4. runs gfrun on the ELF (`-t 1 -f`, matching gfrun_flashMLA.py -- the
     res_check=on (hosted-libc) build needs `-t` for gfrun to service the
     guest open/read/write syscalls);
  5. compares res.bin vs golden.bin with np.allclose (atol/rtol) and reports
     mse / max-abs, flagging an all-zero res.bin (RES_CHECK off / crash).

Note: the kernel only performs file I/O when built with `res_check=on`
(see test/common/Makefile.common). ELFs built without it leave res.bin at
zero; the script reports that as a "no output (res.bin all zero; RES_CHECK
off or run crashed?)" failure.

Build (toolchain at .../linx_blockisa_llvm_musl/bin):

    make TESTCASE=matmul TYPE=MASK MODE=MASK_FP32 \\
         M=64 N=64 K=64 tM=16 tN=16 tK=16 res_check=on
"""

import argparse
import os
import re
import signal
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    print("WARNING: torch not found; falling back to numpy for the golden "
          "reference. `pip install torch` for the pytorch reference.")


MAX_WORKERS = min(8, (os.cpu_count() or 4))
DEFAULT_GFRUN = "/Users/blacktraker/Programming/gitproj/DV4/SuperScalarModel/bin/gfrun"
# Match gfrun_flashMLA.py exactly. gfrun's "-t" is a LOG LEVEL, not a thread
# count: "-t 1" enables the basic per-instruction trace. The res_check=on
# (hosted-libc) build needs it: gfrun only services the guest open/read/write
# syscalls on the trace path, so without "-t" the binary stalls at the first
# libc syscall (it never reaches main). Do NOT switch to bare "-f" for
# res_check builds.
DEFAULT_GFRUN_ARGS = " -t 1 -f "

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../../.."))
CMP_ROOT = os.path.join(ROOT, "compare")

# input dtype -> (numpy dtype for the .bin layout, element size)
_DTYPE_MAP = {
    "fp32": (np.float32, 4),
    "fp16": (np.float16, 2),
    "fp8":  (np.uint8, 1),   # e4m3 stored as raw bytes
}

# atol/rtol defaults. fp8 accumulates more error; allow more slack.
_TOL = {
    "fp32": (2e-2, 2e-2),
    "fp16": (2e-2, 2e-2),
    "fp8":  (5e-2, 5e-2),
}

statics = {"pass": [], "fail": []}


# ---------------------------------------------------------------------------
# Shape / variant parsing from the ELF name
# ---------------------------------------------------------------------------
def extract_int(name, elf_name, default=None):
    match = re.search(rf"{name}(\d+)", elf_name)
    if match:
        return int(match.group(1))
    if default is not None:
        return default
    raise ValueError(f"Cannot extract {name} from {elf_name}")


def parse_matmul_shape(elf_name, args):
    """Parse the matmul shape / MASK mode out of an ELF basename (no .elf).

    Recognised ELF name pattern (see test/kernel/matmul/Makefile, TYPE=MASK):

        matmul_MASK_<MODE>_M{M}_N{N}_K{K}_tM{tM}_tN{tN}_tK{tK}

    The MODE token (e.g. MASK_FP32, MASK_FP16_REUSEA, MASK_FP8_MULTI4_AB) may
    itself contain underscores, so anchor it on the trailing shape marker
    "_M<digits>". Returns a dict, or None if the ELF is not a supported
    MASK matmul variant.
    """
    base = elf_name.replace(".elf", "").strip()
    m = re.search(r"matmul_MASK_(.+?)_M\d", base)
    if not m:
        return None
    mode = m.group(1)

    if mode.startswith("MASK_FP32"):
        dtype = "fp32"
    elif mode.startswith("MASK_FP16"):
        dtype = "fp16"
    elif mode.startswith("MASK_FP8"):
        dtype = "fp8"
    elif mode == "MX_FP8":
        # Microscaling: matmul_mx uses per-tile scale factors (src0_mx /
        # src1_mx) that are NOT read from the file interface, so a plain
        # torch.matmul golden would not match the kernel. Skip it.
        return None
    else:
        return None

    shape = {
        "mode": mode,
        "dtype": dtype,
        "M": extract_int("M", base),
        "N": extract_int("N", base),
        "K": extract_int("K", base),
        "tM": extract_int("tM", base),
        "tN": extract_int("tN", base),
        "tK": extract_int("tK", base),
    }
    if any(shape[k] is None for k in ("M", "N", "K", "tM", "tN", "tK")):
        return None
    return shape


# ---------------------------------------------------------------------------
# fp8 (e4m3) quantization helper
# ---------------------------------------------------------------------------
def quantize_fp8_e4m3(a_np):
    """Quantize a float32 numpy array to fp8 e4m3fn, returning the raw uint8
    bytes (the on-disk layout __fp8_e4m3 uses). Requires torch."""
    if not _HAS_TORCH:
        raise RuntimeError(
            "fp8 (e4m3) quantization needs torch (torch.float8_e4m3fn). "
            "Install it with `pip install torch`."
        )
    t = torch.from_numpy(a_np.astype(np.float32))
    t8 = t.to(torch.float8_e4m3fn)
    return t8.view(torch.uint8).numpy().copy()


# ---------------------------------------------------------------------------
# Golden reference (pytorch preferred)
# ---------------------------------------------------------------------------
def matmul_reference(a_np, b_np, dtype):
    """Compute the golden C = A @ B in float32.

    Inputs are first quantized to the operator's input dtype (matching what
    the kernel loads), then the matmul is performed in float32 -- the kernel
    uses an fp32 accumulator.
    """
    if _HAS_TORCH:
        a = torch.from_numpy(a_np)  # float32 [M,K]
        b = torch.from_numpy(b_np)  # float32 [K,N]
        if dtype == "fp16":
            a = a.to(torch.float16).to(torch.float32)
            b = b.to(torch.float16).to(torch.float32)
        elif dtype == "fp8":
            a = a.to(torch.float8_e4m3fn).to(torch.float32)
            b = b.to(torch.float8_e4m3fn).to(torch.float32)
        golden = torch.matmul(a, b)  # [M,N] float32
        return golden.numpy()

    # numpy fallback (fp32 / fp16 only; fp8 raises in quantize_fp8_e4m3)
    a = a_np
    b = b_np
    if dtype == "fp16":
        a = a.astype(np.float16).astype(np.float32)
        b = b.astype(np.float16).astype(np.float32)
    return np.matmul(a, b).astype(np.float32)


# ---------------------------------------------------------------------------
# Input generation
# ---------------------------------------------------------------------------
def gen_input_and_golden(elf_name, path, args):
    print("Start to gen input data & golden data:", path, elf_name)
    shape = parse_matmul_shape(elf_name, args)
    if shape is None:
        raise ValueError(f"not a supported MASK matmul ELF: {elf_name}")
    print("shape:", shape)

    M, N, K = shape["M"], shape["N"], shape["K"]
    dtype = shape["dtype"]

    rng = np.random.default_rng(args.seed)
    # Same distribution as gfrun_flashMLA: N(0,1)/10 clipped to [-1,1].
    a = (rng.standard_normal((M, K), dtype=np.float32) / 10.0)
    a = np.clip(a, -1.0, 1.0).astype(np.float32)
    b = (rng.standard_normal((K, N), dtype=np.float32) / 10.0)
    b = np.clip(b, -1.0, 1.0).astype(np.float32)

    golden = matmul_reference(a, b, dtype)  # [M,N] float32

    # Inputs written in the operator's input-dtype layout.
    if dtype == "fp16":
        a_q = a.astype(np.float16)
        b_q = b.astype(np.float16)
    elif dtype == "fp8":
        a_q = quantize_fp8_e4m3(a)
        b_q = quantize_fp8_e4m3(b)
    else:
        a_q = a.astype(np.float32)
        b_q = b.astype(np.float32)

    a_q.tofile(os.path.join(path, "src0.bin"))
    b_q.tofile(os.path.join(path, "src1.bin"))
    golden.tofile(os.path.join(path, "golden.bin"))

    # The local SuperScalarModel gfrun can open existing files but does not
    # reliably create missing output files through the guest O_CREAT path.
    # (mirrors gfrun_flashMLA.py)
    np.zeros((M, N), dtype=np.float32).tofile(os.path.join(path, "res.bin"))
    return shape


# ---------------------------------------------------------------------------
# Run / compare
# ---------------------------------------------------------------------------
def run_qemu(elf, args):
    print("Start to run gfrun----------")
    try:
        if not os.path.exists(elf):
            print("elf not exist")
            return elf, "not exist", ""

        if args.plat == "cpu":
            cmd = elf
        else:
            cmd = args.gfrun + args.gfrun_args + elf
        print(f"[run] {cmd}")
        proc = subprocess.Popen(
            [cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            shell=True,
            preexec_fn=os.setsid,
        )
        stdout, stderr = proc.communicate(timeout=args.timeout)
        output = stdout + stderr
        status = "pass" if proc.returncode == 0 else "fail"
        print("gfrun status:", status)
        return elf, status, output
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.communicate()
        return elf, "timeout", "Timeout expired"
    except Exception as exc:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.communicate()
        return elf, "error", str(exc)


def compare_array(cmp_data, golden_data, dtype, atol, rtol):
    res = np.fromfile(cmp_data, dtype=np.float32)
    ref = np.fromfile(golden_data, dtype=np.float32)
    if res.shape != ref.shape:
        return "fail", {
            "reason": f"shape mismatch: got {res.shape}, ref {ref.shape}",
            "mse": float("nan"), "max_abs": float("nan"),
        }
    if not np.any(res):
        # All-zero result: the kernel did not write (RES_CHECK off / crash).
        return "fail", {
            "reason": "res.bin all zero (RES_CHECK off or run crashed?)",
            "mse": float("nan"), "max_abs": float("nan"),
        }
    diff = res - ref
    mse = float(np.mean(diff * diff)) if diff.size else 0.0
    max_abs = float(np.max(np.abs(diff))) if diff.size else 0.0
    close = np.allclose(res, ref, atol=atol, rtol=rtol)
    return ("pass" if close else "fail"), {"mse": mse, "max_abs": max_abs}


def result_compare(cmp_path, args):
    elf_name = os.path.basename(cmp_path)
    shape = parse_matmul_shape(elf_name, args)
    dtype = shape["dtype"] if shape else "fp32"
    atol, rtol = _TOL.get(dtype, (2e-2, 2e-2))
    return compare_array(
        os.path.join(cmp_path, "res.bin"),
        os.path.join(cmp_path, "golden.bin"),
        dtype,
        atol,
        rtol,
    )


def _format_matrix_head(name, data, rows=4, cols=8):
    out = [f"\n[{name}] shape={data.shape}"]
    out.append(
        "  min/max/mean/nonzero: "
        f"{float(np.min(data)):.8g} / {float(np.max(data)):.8g} / "
        f"{float(np.mean(data)):.8g} / {int(np.count_nonzero(data))}"
    )
    out.append("  head:")
    head = data[:rows, :cols] if data.ndim == 2 else data[:rows]
    out.append(np.array2string(head, precision=8, suppress_small=False))
    return "\n".join(out)


def dump_debug_report(cmp_path, elf_name, args):
    """Write a short res-vs-golden diff report (modeled on gfrun_flashMLA's
    debug_compare.log, but without the flashMLA-specific tile dumps)."""
    res_path = os.path.join(cmp_path, "res.bin")
    golden_path = os.path.join(cmp_path, "golden.bin")
    if not (os.path.exists(res_path) and os.path.exists(golden_path)):
        return None

    shape = parse_matmul_shape(elf_name, args)
    if shape is None:
        return None
    M, N = shape["M"], shape["N"]
    res = np.fromfile(res_path, dtype=np.float32).reshape(M, N)
    ref = np.fromfile(golden_path, dtype=np.float32).reshape(M, N)
    diff = res - ref
    max_idx = np.unravel_index(np.argmax(np.abs(diff)), diff.shape) if diff.size else (0, 0)

    lines = [
        "matmul debug compare report",
        f"elf: {elf_name}",
        f"compare dir: {cmp_path}",
        f"shape: {shape}",
        "",
        _format_matrix_head("res.bin", res, 4, 8),
        "",
        _format_matrix_head("golden.bin", ref, 4, 8),
        "",
        f"diff max_abs={float(np.max(np.abs(diff))) if diff.size else 0.0:.8g} "
        f"mse={float(np.mean(diff * diff)) if diff.size else 0.0:.8g} "
        f"idx={max_idx} got={float(res[max_idx]):.8g} ref={float(ref[max_idx]):.8g}",
    ]
    log_path = os.path.join(cmp_path, "debug_compare.log")
    with open(log_path, "w") as f:
        f.write("\n".join(lines))
        f.write("\n")
    return log_path


def check_elf(elf, args):
    print("Start to check elf----------------")
    elf_name = os.path.basename(elf).replace(".elf", "").strip()
    cmp_data_path = os.path.join(CMP_ROOT, elf_name)

    os.system(f"rm -rf {cmp_data_path}; mkdir -p {cmp_data_path}")
    os.makedirs(cmp_data_path, exist_ok=True)

    try:
        gen_input_and_golden(elf_name, cmp_data_path, args)
    except ValueError as e:
        return elf_name, "skip", "not chk", str(e)

    _, status, output = run_qemu(elf, args)
    if status != "pass":
        return elf_name, status, "not chk", output

    cmp_data = os.path.join(cmp_data_path, "res.bin")
    if not os.path.exists(cmp_data):
        return elf_name, status, "output not exist", "NaN"

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run kernel/matmul gfrun result check")
    parser.add_argument("-l", dest="elf_list", default="./tmp.list", type=str,
                        help="file listing ELF paths, one per line")
    parser.add_argument("-d", dest="dbg_elf", default=None, type=str,
                        help="single ELF to check (debug)")
    parser.add_argument("-o", dest="res_log", default="matmul_result_check.log", type=str)
    parser.add_argument("-plat", dest="plat", default="linx", type=str,
                        help="linx (gfrun) or cpu (run the ELF directly)")
    parser.add_argument("--gfrun", default=DEFAULT_GFRUN, type=str)
    parser.add_argument("--gfrun-args", default=DEFAULT_GFRUN_ARGS, type=str)
    parser.add_argument("--timeout", default=1200, type=int)
    parser.add_argument("--seed", default=123, type=int)
    parser.add_argument("--out-atol", default=None, type=float,
                        help="override output atol (default per-dtype)")
    parser.add_argument("--out-rtol", default=None, type=float,
                        help="override output rtol (default per-dtype)")
    parser.add_argument("--workers", default=MAX_WORKERS, type=int)
    args = parser.parse_args()

    if args.dbg_elf is not None:
        elf_paths = [args.dbg_elf]
    else:
        with open(args.elf_list, "r") as f:
            elf_paths = [line.strip() for line in f if line.strip()]

    # Allow CLI atol/rtol to override the per-dtype defaults.
    if args.out_atol is not None:
        _TOL["fp32"] = (args.out_atol, _TOL["fp32"][1])
        _TOL["fp16"] = (args.out_atol, _TOL["fp16"][1])
        _TOL["fp8"] = (args.out_atol, _TOL["fp8"][1])
    if args.out_rtol is not None:
        _TOL["fp32"] = (_TOL["fp32"][0], args.out_rtol)
        _TOL["fp16"] = (_TOL["fp16"][0], args.out_rtol)
        _TOL["fp8"] = (_TOL["fp8"][0], args.out_rtol)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(check_elf, elf, args): elf for elf in elf_paths}
        for future in as_completed(futures):
            elf, status, chk_status, metric = future.result()
            print(f"{elf} -> run_status: {status.upper()} chk_status: {chk_status.upper()} metric:{metric}")
            log_result(elf, status, chk_status, metric)

    if args.dbg_elf is None:
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
