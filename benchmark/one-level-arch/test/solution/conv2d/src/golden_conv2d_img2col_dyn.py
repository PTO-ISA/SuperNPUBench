#!/usr/bin/env python3
"""Run the conv2d_img2col_dyn RES_CHECK ELF and compare with torch.conv2d.

The ELF must be built with ``res_check=on``.  The script writes the NCHW
input and OIHW weight binaries into the ELF's CHK_DIR, runs gfrun with four
PEs, and checks the exported NHWC output against
``torch.nn.functional.conv2d``.

The kernel receives every shape number through the runtime tiling
descriptor; the constants below only generate the data and must match the
Makefile defaults of test/solution/conv2d (TESTCASE=conv2d_img2col_dyn).
The default shape deliberately differs from the static conv2d_img2col test
while sharing the same KPhys = 32 bucket.

Usage:
  python3 golden_conv2d_img2col_dyn.py --elf /path/to/solution_conv2d_conv2d_img2col_dyn_PE4.elf
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

# Must match the Makefile defaults of test/solution/conv2d (dyn testcase).
BATCH, CIN, HIN, WIN = 2, 16, 32, 64
COUT, KH, KW = 48, 1, 2
SH, SW = 2, 2
PT, PB, PL, PR = 2, 1, 0, 1
DH, DW = 1, 1

C0 = 256 // 32
C1 = (CIN + C0 - 1) // C0
K = KH * KW * C1 * C0

OUT_H = (HIN + PT + PB - (KH - 1) * DH - 1) // SH + 1
OUT_W = (WIN + PL + PR - (KW - 1) * DW - 1) // SW + 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--elf", required=True, type=Path)
    parser.add_argument("--gfrun", default="/mnt/workspace/v310/SuperScalarModel/bin/gfrun",
                        type=Path)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    elf = args.elf.resolve()
    if not elf.is_file():
        raise SystemExit(f"ELF not found: {elf}")
    compare_dir = Path(__file__).resolve().parents[4] / "compare" / elf.stem
    compare_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    x = torch.empty(BATCH, CIN, HIN, WIN).uniform_(-0.5, 0.5)
    w = torch.empty(COUT, CIN, KH, KW).uniform_(-0.5, 0.5)

    # Asymmetric padding is applied explicitly for the golden computation
    # only; conv2d itself runs unpadded.
    xp = x
    if PT or PB or PL or PR:
        xp = torch.nn.functional.pad(x, (PL, PR, PT, PB))
    golden = torch.nn.functional.conv2d(xp, w, stride=(SH, SW),
                                        dilation=(DH, DW))
    # NCHW -> NHWC; the flattened row-major [Ho*Wo, Co] planes match the
    # kernel's output buffer layout exactly.
    golden = golden.permute(0, 2, 3, 1).contiguous().reshape(-1).numpy()

    # The TIMG2COL SharedND form consumes the feature map with DN
    # (channel-major) source addressing (B.DATR DN2ND), so the input
    # binary stays NCHW -- and UNPADDED: the padding/stride/dilation
    # window contract lives in the hardware, so the kernel must receive
    # the raw [Cin][H][W] planes.
    x.numpy().tofile(compare_dir / "input.bin")
    w.numpy().tofile(compare_dir / "weight.bin")
    # Pre-create the output file so a silent ELF failure cannot pass unnoticed.
    np.zeros(BATCH * OUT_H * OUT_W * COUT, dtype=np.float32).tofile(
        compare_dir / "output.bin")

    print(f"runtime tiling: Cin={CIN} H={HIN} W={WIN} Co={COUT} "
          f"K={KH}x{KW} S={SH}x{SW} P=(t{PT},b{PB},l{PL},r{PR}) "
          f"D={DH}x{DW}")
    print(f"derived:        C1={C1} K={K} (KPhys bucket) Ho={OUT_H} Wo={OUT_W} "
          f"M={OUT_H * OUT_W} Mb={OUT_H * OUT_W // 64} Nb={COUT // 16}")

    cmd = [str(args.gfrun), "-t", "1", "-s", "softcore.multiThreadNum=4",
           "-f", str(elf)]
    print("+", " ".join(cmd))
    completed = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if completed.returncode != 0:
        print(completed.stdout[-4000:])
        print(completed.stderr[-4000:])
        raise SystemExit(f"gfrun failed with rc={completed.returncode}")
    if "Reach the End of Benchmark" not in completed.stdout:
        raise SystemExit("gfrun did not reach the end of the benchmark")

    result = np.fromfile(compare_dir / "output.bin", dtype=np.float32)
    if result.size != golden.size:
        raise SystemExit(f"output size {result.size} != expected {golden.size}")

    diff = np.abs(result.astype(np.float64) - golden.astype(np.float64))
    tol = args.atol + args.rtol * np.abs(golden.astype(np.float64))
    bad = diff > tol
    print(f"elements={result.size}  max_abs_diff={diff.max():.3e}  "
          f"mismatches={int(bad.sum())}")
    if bad.any():
        idx = int(np.flatnonzero(bad)[0])
        print(f"first mismatch at [{idx}]: got {result[idx]}, "
              f"expected {golden[idx]}")
        return 1
    print("PASS: conv2d_img2col_dyn matches torch.nn.functional.conv2d")
    return 0


if __name__ == "__main__":
    sys.exit(main())
