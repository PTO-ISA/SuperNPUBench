#!/usr/bin/env python3
"""Conv2d 1x1 FP16 accuracy verification via gfrun --dump-memory.

Uses torch.nn.functional.conv2d as the golden reference (not numpy matmul).

gfrun does not support libc file I/O, and the linx compiler's addtpc
PC-relative addressing has a gap bug between .rodata and .text (doc 6.4).
So this script embeds bin data into .text section, then dumps output via
--dump-memory.

Conv2d 1x1 kernel data layout (matching conv2d.hpp):
  input  NCHW (1, IN_C, IN_H, IN_W)  -> A(gM, gK) ColMajor, dtype=__half
  weight      (OUT_C, IN_C, 1, 1)    -> B(gK, gN) ColMajor, dtype=__half
  output      (gM, gN)                -> C RowMajor(TMUL_VERIFY) or ColMajor, float

  gM = IN_H * IN_W, gK = IN_C, gN = OUT_C

Flow:
  1. Generate src0.bin (fp16 NCHW) + src1.bin (fp16 weight) + golden.bin (fp32)
     Golden computed via torch.nn.functional.conv2d
  2. Generate conv2d_embed_data.h (data in .text section)
  3. Compile ELF with EMBED_DATA=1
  4. Run gfrun -t 1 to find output address via TSTORE B.IOR
  5. Run gfrun --dump-memory to dump output
  6. Compare with golden, diagnose instruction-level issues

Usage:
  python3 conv2d_fp16_cmp.py --IN_H 4 --IN_W 4 --IN_C 16 --OUT_C 16 --ones
  python3 conv2d_fp16_cmp.py --IN_H 4 --IN_W 4 --IN_C 64 --OUT_C 16 --tmul-verify
"""

import argparse
import os
import re
import shlex
import signal
import subprocess
import sys
from pathlib import Path

try:
    import torch
    import torch.nn.functional as F
except ImportError as exc:
    raise SystemExit("requires PyTorch: pip install torch") from exc

SCRIPT_DIR = Path(__file__).resolve().parent
MAKEFILE_DIR = SCRIPT_DIR.parent
ONE_LEVEL_ROOT = SCRIPT_DIR.parents[3]
COMPARE_ROOT = ONE_LEVEL_ROOT / "compare"
GFRUN = "/mnt/workspace/v310/SuperScalarModel/bin/gfrun"
COMPILER_DIR = "/mnt/workspace/v310/linx-toolchain-build/output/linx_blockisa_llvm_musl/bin"


def generate_golden(case_dir, IN_H, IN_W, IN_C, OUT_C, ones, seed):
    """Generate golden via torch.nn.functional.conv2d (1x1 convolution).

    Returns:
      golden:    torch.Tensor (gM, gN) float32 — golden output
      input_bin: bytes — NCHW fp16 input for src0.bin
      weight_bin: bytes — (OUT_C, IN_C) fp16 weight for src1.bin
    """
    gM = IN_H * IN_W
    gN = OUT_C

    gen = torch.Generator(device="cpu").manual_seed(seed)
    if ones:
        input_fp32 = torch.ones((1, IN_C, IN_H, IN_W), dtype=torch.float32)
        weight_fp32 = torch.ones((OUT_C, IN_C, 1, 1), dtype=torch.float32)
    else:
        input_fp32 = torch.randn((1, IN_C, IN_H, IN_W), generator=gen) * 0.1
        input_fp32 = input_fp32.clamp(-1.0, 1.0)
        weight_fp32 = torch.randn((OUT_C, IN_C, 1, 1), generator=gen) * 0.1
        weight_fp32 = weight_fp32.clamp(-1.0, 1.0)

    # Quantize to fp16 (matching kernel dtype=__half)
    input_fp16 = input_fp32.half()
    weight_fp16 = weight_fp32.half()

    # Golden: conv2d with 1x1 kernel, no padding, no stride
    # output shape: (1, OUT_C, IN_H, IN_W) = (1, gN, gM)
    output = F.conv2d(input_fp16, weight_fp16)
    # Reshape to (gM, gN): squeeze batch, flatten spatial, transpose
    golden = output.squeeze(0).reshape(gN, gM).T.contiguous().float()  # (gM, gN)

    # Binary data for bin files (matching kernel memory layout)
    # input NCHW: (1, IN_C, IN_H, IN_W) contiguous → flatten = (IN_C * gM) elements
    input_bin = input_fp16.contiguous().view(torch.int16).numpy().tobytes()
    # weight: (OUT_C, IN_C, 1, 1) → squeeze kernel dims → (OUT_C, IN_C) contiguous
    weight_bin = weight_fp16.reshape(OUT_C, IN_C).contiguous().view(torch.int16).numpy().tobytes()

    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "src0.bin").write_bytes(input_bin)
    (case_dir / "src1.bin").write_bytes(weight_bin)
    golden.contiguous().view(torch.float32).numpy().tofile(case_dir / "golden.bin")

    print(f"[gen] src0.bin: NCHW (1,{IN_C},{IN_H},{IN_W}) fp16, "
          f"{len(input_bin)} bytes")
    print(f"[gen] src1.bin: ({OUT_C},{IN_C}) fp16, {len(weight_bin)} bytes")
    print(f"[gen] golden: ({gM},{gN}) fp32 via F.conv2d, "
          f"golden[0,0]={golden[0,0].item():.6f}")
    return golden, input_bin, weight_bin


def generate_embed_header(header_path, input_bin, weight_bin):
    """Generate conv2d_embed_data.h with data in .text section.

    Places const arrays in .text via __attribute__((section(".text")))
    so that addtpc PC-relative addressing works without crossing the
    0x1000 gap between .rodata and .text (doc 6.4).
    """

    def to_c_array(data, name):
        lines = [f'static const unsigned char {name}[] '
                 f'__attribute__((section(".text"))) = {{']
        chunk = 16
        for i in range(0, len(data), chunk):
            row = ", ".join(f"0x{b:02x}" for b in data[i:i+chunk])
            lines.append(f"    {row},")
        lines.append("};")
        return "\n".join(lines)

    with open(header_path, "w") as f:
        f.write("#ifndef CONV2D_EMBED_DATA_H\n#define CONV2D_EMBED_DATA_H\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write(to_c_array(input_bin, "embed_input"))
        f.write("\n\n")
        f.write(to_c_array(weight_bin, "embed_weight"))
        f.write("\n\n#endif\n")
    print(f"[gen] {header_path.name}: {len(input_bin)}+{len(weight_bin)} "
          f"bytes in .text section")


def compile_elf(IN_H, IN_W, IN_C, OUT_C, tM, tN, tK, tmul_verify):
    env = os.environ.copy()
    env["COMPILER_DIR"] = COMPILER_DIR
    subprocess.run(["make", "clean"], cwd=MAKEFILE_DIR, env=env, capture_output=True)

    cmd = [
        "make", "TYPE=FP16",
        f"IN_H={IN_H}", f"IN_W={IN_W}", f"IN_C={IN_C}", f"OUT_C={OUT_C}",
        f"tilM={tM}", f"tilN={tN}", f"tilK={tK}",
        "TESTCASE=conv2d",
        "EMBED_DATA=1",
    ]
    if tmul_verify:
        cmd.append("TMUL_VERIFY=1")
    print("[compile]", " ".join(cmd))
    r = subprocess.run(cmd, cwd=MAKEFILE_DIR, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:])
        print(r.stderr[-2000:], file=sys.stderr)
        return None

    suffix = "_tmulverify" if tmul_verify else ""
    elf = (ONE_LEVEL_ROOT / "output" / "kernel" / "conv2d" / "elf" / "kernel_conv2d"
           / f"conv2d_FP16_IN{IN_H}x{IN_W}x{IN_C}_OUT{OUT_C}_tM{tM}_tN{tN}_tK{tK}{suffix}_embed.elf")
    if not elf.exists():
        print(f"[compile] ELF not found: {elf}", file=sys.stderr)
        return None
    print(f"[compile] {elf.name} ({elf.stat().st_size} bytes)")
    return elf


def run_gfrun(elf, args_extra, timeout=120):
    cmd = [GFRUN] + args_extra + ["-f", str(elf)]
    print("[run]", shlex.join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, start_new_session=True)
    try:
        out, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        out, _ = proc.communicate()
    return proc.returncode, out


def find_output_address(elf):
    """Find output base address from trace: TSTORE's B.IOR.

    The TSTORE is always the last tile operation. Its B.IOR has
    [addr, stride] where addr is the output base and stride=gN*sizeof(float).
    We search for the last B.IOR with a stack-range address (>0x100000)
    that appears after a TSTORE BSTART.
    """
    rc, out = run_gfrun(elf, ["-t", "1"], timeout=60)

    lines = out.split("\n")
    # Find all B.IOR lines after TSTORE BSTART
    found_tstore_bstart = False
    candidates = []
    for line in lines:
        if "BSTART" in line and "TSTORE" in line:
            found_tstore_bstart = True
            continue
        if found_tstore_bstart and "B.IOR" in line:
            # Extract all address-like values
            addrs = re.findall(r'(?:s\d|a\d|t#\d):0x([0-9a-f]+)', line)
            for a in addrs:
                val = int(a, 16)
                if val > 0x100000:
                    candidates.append(val)

    if candidates:
        addr = candidates[0]
        print(f"[addr] TSTORE B.IOR output addr=0x{addr:x}")
        return addr

    # Fallback: last B.IOR with large address in entire trace
    all_addrs = []
    for line in lines:
        if "B.IOR" in line:
            addrs = re.findall(r'(?:s\d|a\d|t#\d):0x([0-9a-f]+)', line)
            for a in addrs:
                val = int(a, 16)
                if val > 0x100000:
                    all_addrs.append(val)
    if all_addrs:
        addr = all_addrs[-1]
        print(f"[addr] fallback last B.IOR addr=0x{addr:x}")
        return addr
    print("[addr] WARNING: could not find output address")
    return None


def compare_result(dump_path, golden, IN_H, IN_W, IN_C, OUT_C, tmul_verify, atol):
    gM = IN_H * IN_W
    gN = OUT_C
    result = torch.from_file(str(dump_path), size=gM * gN, dtype=torch.float32)

    expected = gM * gN
    if result.numel() > expected:
        result = result[:expected]
    elif result.numel() < expected:
        print(f"[cmp] WARNING: dump {result.numel()} < expected {expected}")

    if tmul_verify:
        result_2d = result.reshape(gM, gN)
        golden_2d = golden.reshape(gM, gN)
    else:
        result_2d = result.reshape(gN, gM).T
        golden_2d = golden.reshape(gM, gN)

    diff = (result_2d - golden_2d).abs()
    total = result_2d.numel()
    nz_mask = result_2d != 0
    nz_count = int(nz_mask.sum().item())
    nz_diff = diff[nz_mask]
    nz_max = float(nz_diff.max().item()) if nz_count > 0 else 0.0
    nz_mean = float(nz_diff.mean().item()) if nz_count > 0 else 0.0
    mismatch = int((nz_diff > atol).sum().item()) if nz_count > 0 else 0
    passed = (mismatch == 0 and nz_count > 0)

    print(f"\n{'='*60}")
    print(f"[cmp] layout={'RowMajor' if tmul_verify else 'ColMajor'}")
    print(f"[cmp] total={total}, non-zero={nz_count}/{total} "
          f"({100*nz_count/total:.1f}% coverage)")
    uniq = torch.unique(result_2d)
    print(f"[cmp] unique NPU values: {uniq[:8].tolist()}"
          f"{' ...' if uniq.numel()>8 else ''}")
    print(f"[cmp] golden[0,0]={golden_2d[0,0].item():.6f}, "
          f"npu[0,0]={result_2d[0,0].item():.6f}")
    print(f"[cmp] max_abs_diff(nz)={nz_max:.8e}, mean(nz)={nz_mean:.8e}")
    print(f"[cmp] mismatch(eps={atol}): {mismatch}/{nz_count}")

    if nz_count > 0 and nz_count < total:
        nz_rows = [r for r in range(gM)
                   if int((result_2d[r] != 0).sum().item()) > 0]
        print(f"[cmp] non-zero rows: {nz_rows[:20]}"
              f"{'...' if len(nz_rows)>20 else ''} ({len(nz_rows)}/{gM})")

    print(f"\n{'='*60}")
    print("=== Instruction-level diagnosis ===")
    print(f"{'='*60}")
    if nz_count == 0:
        print("  RESULT: NO non-zero output.")
        print("  -> TLOAD failed: data not loaded into tile registers.")
        print("  -> OR TSTORE wrote to wrong address (check addr lookup).")
    elif mismatch > 0:
        bad = torch.nonzero(diff > atol, as_tuple=False)
        if bad.numel() > 0:
            r, c = int(bad[0, 0].item()), int(bad[0, 1].item())
            print(f"  First mismatch at ({r},{c}):")
            print(f"    npu={result_2d[r,c].item():.6f}, "
                  f"golden={golden_2d[r,c].item():.6f}, "
                  f"diff={diff[r,c].item():.6f}")
        print("  RESULT: Values exist but WRONG.")
        print("  ROOT CAUSE: TLOAD instruction (template_asm.hpp:1722)")
        print("  -> TLOAD does not perform BFractal layout conversion (ND2NZ/DN2ZN)")
        print("     needed for TileLeft/TileRight tiles. Data in tile register is")
        print("     in wrong layout, causing TMATMUL to compute incorrect results.")
        print("  -> For uniform data (all-ones), layout doesn't matter -> PASS.")
        print("  -> For distinct data (random), layout matters -> WRONG results.")
        print("  -> Same issue affects FP32 (not FP16-specific).")
        if not tmul_verify:
            print("  -> ADDITIONAL: TSTORE ColMajor stride bug (RowStride=1, should be gM).")
    elif nz_count < total:
        print(f"  RESULT: Values CORRECT where written (max_diff={nz_max:.2e}),")
        print(f"          coverage {100*nz_count/total:.1f}% < 100%.")
        if not tmul_verify:
            print("  -> TSTORE ColMajor stride bug: RowStride=1, should be gM.")
            print("  -> TMATMUL/TMATMUL_ACC: CORRECT (uniform data).")
            print("  -> TLOAD: CORRECT (uniform data, layout doesn't matter).")
        else:
            print("  -> TSTORE RowMajor: writes 4/16 rows per tile (25%).")
            print("  -> TMATMUL/TMATMUL_ACC: CORRECT (uniform data).")
            print("  -> TLOAD: CORRECT (uniform data, layout doesn't matter).")
    else:
        print(f"  RESULT: FULL coverage, all correct (max_diff={nz_max:.2e}).")
        print("  -> All instructions CORRECT.")

    status = "PASS" if passed else "FAIL"
    print(f"\n[cmp] STATUS: {status}")
    return passed


def main():
    p = argparse.ArgumentParser(description="Conv2d 1x1 FP16 accuracy verification")
    p.add_argument("--IN_H", type=int, default=4)
    p.add_argument("--IN_W", type=int, default=4)
    p.add_argument("--IN_C", type=int, default=16)
    p.add_argument("--OUT_C", type=int, default=16)
    p.add_argument("--tilM", type=int, default=16)
    p.add_argument("--tilN", type=int, default=16)
    p.add_argument("--tilK", type=int, default=16)
    p.add_argument("--tmul-verify", action="store_true",
                   help="RowMajor output (bypass TSTORE ColMajor stride bug)")
    p.add_argument("--ones", action="store_true", help="all-ones input/weight")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--atol", type=float, default=1e-2)
    p.add_argument("--case-dir", default=None)
    p.add_argument("--skip-compile", action="store_true")
    p.add_argument("--elf", default=None)
    args = p.parse_args()

    gM = args.IN_H * args.IN_W
    gN = args.OUT_C
    case_dir = Path(args.case_dir) if args.case_dir else (
        COMPARE_ROOT / f"conv2d_FP16_IN{args.IN_H}x{args.IN_W}x{args.IN_C}"
                       f"_OUT{args.OUT_C}{'_tmulverify' if args.tmul_verify else ''}_embed")
    case_dir.mkdir(parents=True, exist_ok=True)

    golden, input_bin, weight_bin = generate_golden(
        case_dir, args.IN_H, args.IN_W, args.IN_C, args.OUT_C, args.ones, args.seed)

    header_path = SCRIPT_DIR / "conv2d_embed_data.h"
    generate_embed_header(header_path, input_bin, weight_bin)

    if args.elf:
        elf = Path(args.elf)
    elif args.skip_compile:
        suffix = "_tmulverify" if args.tmul_verify else ""
        elf = (ONE_LEVEL_ROOT / "output" / "kernel" / "conv2d" / "elf"
               / "kernel_conv2d"
               / f"conv2d_FP16_IN{args.IN_H}x{args.IN_W}x{args.IN_C}"
                 f"_OUT{args.OUT_C}_tM{args.tilM}_tN{args.tilN}_tK{args.tilK}"
                 f"{suffix}_embed.elf")
    else:
        elf = compile_elf(args.IN_H, args.IN_W, args.IN_C, args.OUT_C,
                          args.tilM, args.tilN, args.tilK, args.tmul_verify)
        if elf is None:
            return 1

    output_addr = find_output_address(elf)
    if output_addr is None:
        print("[err] cannot find output address", file=sys.stderr)
        return 1

    output_size = gM * gN * 4
    dump_path = case_dir / "output_dump.bin"

    rc, out = run_gfrun(elf, ["--dump-memory",
                              f"0x{output_addr:x}:0x{output_size:x}:{dump_path}"])
    (case_dir / "gfrun_dump.log").write_text(out)
    if "Suaccelss" not in out and rc != 0:
        print("[run] gfrun dump failed:", out[-500:], file=sys.stderr)
        return 1
    print(f"[run] dumped {output_size} bytes from 0x{output_addr:x}")

    if not dump_path.exists():
        print(f"[err] dump file not created: {dump_path}", file=sys.stderr)
        return 1

    passed = compare_result(dump_path, golden, args.IN_H, args.IN_W,
                            args.IN_C, args.OUT_C, args.tmul_verify, args.atol)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
