#!/usr/bin/env python3
"""ArgMaxV2 200-case fuzz accuracy validation (gfrun, four PEs, exact-match oracle).

Pipeline: fixed 200-entry config table -> constraint validation (mirror of the
kernel's static_asserts) -> per-case .cpp codegen -> `make res_check=on` ->
NumPy input/golden generation -> `gfrun -s softcore.multiThreadNum=4` ->
exact integer compare (atol=rtol=0) -> TSV/JSON report.

Golden standard: np.argmax over the stored bit-exact values. Semantics shared
with CANN/PyTorch and this kernel: first-occurrence ties, any-NaN (positive or
negative bit pattern) wins as maximum, +-0.0 equal.

Layout produced by this script (all untracked build/validation artifacts):
  src/<id>.cpp                 generated test sources
  ../../../../output/kernel/argmaxv2/elf/kernel_argmaxv2_f*.elf   built ELFs
  ../../../../compare/kernel_argmaxv2_f<id>_PE4/        input.bin/output.bin/gfrun.log
  <repo>/.validation/argmaxv2_fuzz_200/               build logs, report.tsv, results.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TEST_KERNEL = HERE.parent                      # .../test/kernel
ARCH = TEST_KERNEL.parent.parent               # .../one-level-arch
REPO = ARCH.parents[1]                         # SuperNPUBench worktree root
OUTPUT = ARCH / "output/kernel/argmaxv2"
COMPARE = ARCH / "compare"
# Optional CLI env override: FUZZ_VALIDATION_DIR points the report at a new
# directory to preserve a previous run for A/B comparison.
VALIDATION = (Path(os.environ["FUZZ_VALIDATION_DIR"]).resolve()
              if os.environ.get("FUZZ_VALIDATION_DIR")
              else REPO / ".validation/argmaxv2_fuzz_200")


def _req_env(name: str, hint: str) -> Path:
    """仓库约定（compile.all）：外部工具路径经环境变量注入，禁止硬编码。"""
    v = os.environ.get(name)
    if not v:
        sys.exit(f"error: 环境变量 {name} 未设置（{hint}）")
    return Path(v)

GFRUN_DEFAULT = _req_env("SUPERSCALAR_MODEL_DIR", "SuperScalarModel 仓库根目录（含 bin/gfsim、bin/gfrun）") / "bin/gfrun"
COMPILER_DEFAULT = _req_env("COMPILER_DIR", "Linx 编译器 bin 目录（linx_blockisa_llvm_musl/bin）")

MAX_ELEMS = 2_000_000
GFRUN_TIMEOUT = 120  # seconds


# ---------------------------------------------------------------------------
# dtype traits (mirror of ArgMaxTraits defaults in kernels/.../detail/argmaxv2.hpp)
# ---------------------------------------------------------------------------
DTYPE_INFO = {
    #                ctype       np-dtype    key-bytes  row tile  col tile
    "fp32": dict(ctype="float",    np=np.float32, key=4, row=(32, 16), col=(16, 32)),
    "fp16": dict(ctype="__half",   np=np.float16, key=2, row=(64, 16), col=(16, 64)),
    "bf16": dict(ctype="__bf16",   np=None,       key=2, row=(64, 16), col=(16, 64)),
    "i32":  dict(ctype="int32_t",  np=np.int32,   key=4, row=(32, 16), col=(16, 32)),
}


@dataclass
class CaseCfg:
    cid: str
    dtype: str          # fp32 | fp16 | bf16 | i32
    idtype: str         # i32 | i64
    axis: int           # may be negative
    dims: tuple
    tile: tuple | None  # (tM, tN) override, None = kernel default
    seed: int
    pattern: str

    @property
    def elems(self) -> int:
        p = 1
        for d in self.dims:
            p *= d
        return p

    @property
    def axis_norm(self) -> int:
        return self.axis % len(self.dims)

    @property
    def desc(self) -> str:
        tile = "def" if self.tile is None else "x".join(map(str, self.tile))
        return (f"{self.dtype}/{self.idtype} ax{self.axis} "
                f"[{','.join(map(str, self.dims))}] t{tile} {self.pattern}")


# ---------------------------------------------------------------------------
# constraint validation: mirror the kernel wrapper + detail static_asserts
# ---------------------------------------------------------------------------
def validate(cfg: CaseCfg) -> None:
    info = DTYPE_INFO[cfg.dtype]
    rank = len(cfg.dims)
    assert rank >= 1 and all(d >= 1 for d in cfg.dims), "dims must be positive"
    assert 0 <= cfg.axis_norm < rank, "axis out of range"
    assert cfg.elems <= MAX_ELEMS, f"too many elements: {cfg.elems}"
    assert cfg.dtype != "i64", "int64 input unsupported"

    ax = cfg.axis_norm
    outer = int(np.prod(cfg.dims[:ax])) if ax > 0 else 1
    axis_dim = cfg.dims[ax]
    inner = int(np.prod(cfg.dims[ax + 1:])) if ax < rank - 1 else 1

    if ax == rank - 1:  # row reduction
        tM, tN = cfg.tile if cfg.tile else info["row"]
        assert tM * tN * info["key"] <= 2048, "tile exceeds 2048B capacity"
        assert tM * info["key"] >= 128, "key-state tile below 128B"
        assert tM * 4 >= 128, "index-state tile below 128B"
    else:               # column reduction
        tM, tN = cfg.tile if cfg.tile else info["col"]
        assert tM * tN * info["key"] <= 2048, "tile exceeds 2048B capacity"
        assert tN * info["key"] >= 128, "key-state tile below 128B"
        assert tN * 4 >= 128, "index-state tile below 128B"
        assert inner % 4 == 0, "kInner not divisible by PE count"
        assert inner % tN == 0, "kInner not divisible by tN"
        assert (inner // 4) % tN == 0, "per-PE column window not tile aligned"
    assert outer * axis_dim * inner == cfg.elems


# ---------------------------------------------------------------------------
# 200-entry config table: 50 per input dtype, fixed and reproducible
# ---------------------------------------------------------------------------
def build_configs() -> list[CaseCfg]:
    configs: list[CaseCfg] = []
    pat_cycle = ["random", "boundary", "x_all_equal", "random", "boundary",
                 "x_spike_last", "random", "boundary", "x_spike_first", "random",
                 "boundary", "x_alt", "random", "boundary", "x_const_min",
                 "random", "boundary", "x_single_max"]

    for dtype, info in DTYPE_INFO.items():
        rM = info["row"][0]
        if dtype in ("fp32", "i32"):
            K1, K2, K3, K4 = 128, 256, 384, 512
        else:  # fp16 / bf16
            K1, K2, K3, K4 = 256, 512, 1024, 1024
        cM = info["col"][0]

        specs: list[tuple[tuple, int, tuple | None]] = [
            # ---- row reduction (axis == last), 30 cases ----
            ((1,), 0, None),
            ((7,), 0, None),
            ((16,), -1, None),
            ((999,), 0, None),
            ((65537,), -1, None),
            ((4, 32), 1, None),
            ((4, 17), -1, None),
            ((8, 33), 1, None),
            ((12, rM + 1), 1, None),
            ((132, 1000), 1, None),
            ((128, 1024), -1, None),
            ((260, 96), 1, None),
            ((5, 999), 1, None),
            ((6, 64), 1, None),
            ((2, 1), -1, None),
            ((64, 1), 1, None),
            ((2, 8, 300), 2, None),
            ((3, 5, 77), -1, None),
            ((2, 3, 130), 2, None),
            ((4, 33, 256), 2, None),
            ((2, 4, 5000), -1, None),
            ((2, 3, 4, 200), 3, None),
            ((1, 1, 1, 1), -1, None),
            ((1, 2, 3, 97), 3, None),
            ((2, 2, 2, 2, 50), 4, None),
            ((1, 3, 1, 5, 129), -1, None),
            ((4 * rM + 1, 16), 1, None),
            ((512, 63), 1, None),
            ((16, 16, 2048), -1, None),
            ((4, 65536), 1, None),
            # ---- row reduction, extra default-tile edge shapes, 4 cases ----
            # NOTE: the kernel entry takes no tile parameters (tile shapes
            # are fixed from the dtype traits, normalization static
            # convention), so every case runs the default tiles.
            ((rM * 4, 64), 1, None),
            ((rM * 4 + 2, 63), -1, None),
            ((3, rM + 1), 1, None),
            ((2, 5, 63), -1, None),
            # ---- column reduction (axis first/middle), 14 cases ----
            ((200, K1), 0, None),
            ((132, K1), -2, None),
            ((33, K2), 0, None),
            ((1000, K1), 0, None),
            ((128, K2), 0, None),
            ((17, K3), -2, None),
            ((1, K1), 0, None),
            ((65, K2), 0, None),
            ((2048, K1), 0, None),
            ((4, 300, K1), 1, None),
            ((2, 77, K2), -2, None),
            ((5, 64, K3), 1, None),
            ((2, 513, K1), 1, None),
            ((3, 4, 5, K4), -3, None),
            # ---- column reduction, extra default-tile edge shapes, 2 cases ----
            ((cM * 8 + 1, K1), 0, None),
            ((5, 33, K1), 1, None),
        ]
        assert len(specs) == 50, f"{dtype}: expected 50 specs, got {len(specs)}"

        for i, (dims, axis, tile) in enumerate(specs):
            pattern = pat_cycle[(len(configs)) % len(pat_cycle)]
            cfg = CaseCfg(
                cid=f"f{len(configs):03d}",
                dtype=dtype,
                idtype="i64" if i % 5 == 2 else "i32",
                axis=axis,
                dims=tuple(int(d) for d in dims),
                tile=tile,
                seed=1000 + len(configs),
                pattern=pattern,
            )
            # weak "extremes" oracles on very large inputs: use random instead
            if cfg.elems > 262144 and cfg.pattern.startswith("x_"):
                cfg.pattern = "random"
            validate(cfg)
            configs.append(cfg)

    assert len(configs) == 200, f"expected 200 configs, got {len(configs)}"
    return configs


# ---------------------------------------------------------------------------
# value generation (numpy, deterministic per seed)
# ---------------------------------------------------------------------------
NEG_NAN_F32 = np.array([0xFFC00000], dtype=np.uint32).view(np.float32)[0]


def _line_indices(dims: tuple, ax: int, j: int):
    """Index tuple selecting reduction line j: non-axis coords from
    unravel_index(j), axis dim swept."""
    rank = len(dims)
    nonax_shape = tuple(d for d in range(rank) if d != ax)
    if not nonax_shape:
        return tuple(slice(None) for _ in range(rank)) if j == 0 else None
    p = int(np.prod([dims[d] for d in nonax_shape]))
    if j >= p:
        return None
    midx = np.unravel_index(j, tuple(dims[d] for d in nonax_shape))
    it = iter(midx)
    return tuple(next(it) if d != ax else slice(None) for d in range(rank))


def gen_float_values(cfg: CaseCfg) -> np.ndarray:
    """Float-domain values shaped cfg.dims (fp32 domain; cast happens later)."""
    ax, dims, n = cfg.axis_norm, cfg.dims, cfg.dims[cfg.axis_norm]
    rng = np.random.default_rng(cfg.seed)
    # per-target-dtype finite minimum: fp32 min overflows fp16 on cast
    dtype_min = (np.finfo(np.float16).min if cfg.dtype == "fp16"
                 else np.finfo(np.float32).min)
    pat = cfg.pattern
    if pat == "random":
        return rng.uniform(-1.0, 1.0, dims).astype(np.float32)
    if pat == "boundary":
        x = rng.uniform(-1.0, 1.0, dims).astype(np.float32)
        l0 = _line_indices(dims, ax, 0)
        x[l0] = np.float32(-0.5)
        if n > 5:
            x[l0][5] = np.float32(2.0)          # tie with mid+5 below
        if n > n // 2 + 5:
            x[l0][n // 2 + 5] = np.float32(2.0)
        if n > 3:
            x[l0][3] = np.float32(-0.0)         # +-0 equal
        if n > 7:
            x[l0][7] = NEG_NAN_F32              # negative-bit NaN still max
        l1 = _line_indices(dims, ax, 1)
        if l1 is not None and n > 41:
            x[l1][40] = np.inf
            x[l1][41] = np.nan                  # NaN beats +inf
        l2 = _line_indices(dims, ax, 2)
        if l2 is not None:
            x[l2] = np.float32(0.0)
            if n > 3:
                x[l2][3] = np.float32(-0.0)
        l3 = _line_indices(dims, ax, 3)
        if l3 is not None:
            x[l3] = np.float32(0.25)
            if n > 7:
                x[l3][7] = NEG_NAN_F32
        l4 = _line_indices(dims, ax, 4)
        if l4 is not None:
            x[l4] = dtype_min                       # all-min line -> index 0
        return x
    if pat == "x_all_equal":
        return np.full(dims, 0.375, dtype=np.float32)
    if pat == "x_spike_last":
        x = np.full(dims, 0.1, dtype=np.float32)
        idx = [slice(None)] * len(dims)
        idx[ax] = -1
        x[tuple(idx)] = np.float32(0.9)
        return x
    if pat == "x_spike_first":
        x = np.full(dims, 0.1, dtype=np.float32)
        idx = [slice(None)] * len(dims)
        idx[ax] = 0
        x[tuple(idx)] = np.float32(0.9)
        return x
    if pat == "x_alt":
        pos = np.arange(n)
        shape = [1] * len(dims)
        shape[ax] = n
        line = np.where(pos % 2 == 0, np.float32(0.9), np.float32(-0.9))
        return np.broadcast_to(line.reshape(shape), dims).copy().astype(np.float32)
    if pat == "x_const_min":
        return np.full(dims, dtype_min, dtype=np.float32)
    if pat == "x_single_max":
        x = rng.uniform(-1.0, 1.0, dims).astype(np.float32)
        flat = x.reshape(-1)
        flat[int(rng.integers(0, flat.size))] = np.float32(3.0)
        return x
    raise ValueError(f"unknown pattern {pat}")


def gen_int_values(cfg: CaseCfg) -> np.ndarray:
    ax, dims, n = cfg.axis_norm, cfg.dims, cfg.dims[cfg.axis_norm]
    rng = np.random.default_rng(cfg.seed)
    pat = cfg.pattern
    if pat == "random":
        return rng.integers(-1_000_000, 1_000_000, dims).astype(np.int32)
    if pat == "boundary":
        x = rng.integers(-1_000_000, 1_000_000, dims).astype(np.int32)
        l0 = _line_indices(dims, ax, 0)
        x[l0] = -7
        if n > 5:
            x[l0][5] = 42
        if n > n // 2 + 5:
            x[l0][n // 2 + 5] = 42              # tie -> first
        l1 = _line_indices(dims, ax, 1)
        if l1 is not None:
            x[l1] = np.iinfo(np.int32).min      # all-min -> 0
        l2 = _line_indices(dims, ax, 2)
        if l2 is not None:
            x[l2] = 0
            if n > 0:
                x[l2][n - 1] = np.iinfo(np.int32).max
        return x
    if pat == "x_all_equal":
        return np.full(dims, 7, dtype=np.int32)
    if pat == "x_spike_last":
        x = np.full(dims, 5, dtype=np.int32)
        idx = [slice(None)] * len(dims)
        idx[ax] = -1
        x[tuple(idx)] = 9
        return x
    if pat == "x_spike_first":
        x = np.full(dims, 5, dtype=np.int32)
        idx = [slice(None)] * len(dims)
        idx[ax] = 0
        x[tuple(idx)] = 9
        return x
    if pat == "x_alt":
        pos = np.arange(n)
        shape = [1] * len(dims)
        shape[ax] = n
        line = np.where(pos % 2 == 0, 9, -9).astype(np.int32)
        return np.broadcast_to(line.reshape(shape), dims).copy()
    if pat == "x_const_min":
        return np.full(dims, np.iinfo(np.int32).min, dtype=np.int32)
    if pat == "x_single_max":
        x = rng.integers(-1_000_000, 1_000_000, dims).astype(np.int32)
        flat = x.reshape(-1)
        flat[int(rng.integers(0, flat.size))] = np.iinfo(np.int32).max
        return x
    raise ValueError(f"unknown pattern {pat}")


def prepare_data(cfg: CaseCfg, case_dir: Path):
    """Write input.bin; pre-fill output.bin with -1; return golden (1-D)."""
    case_dir.mkdir(parents=True, exist_ok=True)
    out_np = np.int64 if cfg.idtype == "i64" else np.int32
    if cfg.dtype == "i32":
        x = gen_int_values(cfg)
        np.ascontiguousarray(x).tofile(case_dir / "input.bin")
        golden = np.argmax(x, axis=cfg.axis_norm)
    else:
        f32 = gen_float_values(cfg)
        if cfg.dtype == "fp32":
            np.ascontiguousarray(f32).tofile(case_dir / "input.bin")
            golden = np.argmax(f32, axis=cfg.axis_norm)
        elif cfg.dtype == "fp16":
            f16 = f32.astype(np.float16)
            np.ascontiguousarray(f16).tofile(case_dir / "input.bin")
            golden = np.argmax(f16, axis=cfg.axis_norm)
        else:  # bf16: store top 16 bits; compare on exact fp32 reconstruction
            bits = (f32.view(np.uint32) >> 16).astype(np.uint16)
            bits.tofile(case_dir / "input.bin")
            recon = (bits.astype(np.uint32) << 16).view(np.float32)
            golden = np.argmax(recon.reshape(f32.shape), axis=cfg.axis_norm)
    golden = np.asarray(golden).reshape(-1).astype(out_np)
    # pre-fill output with -1 so an unwritten result cannot false-PASS
    np.full(golden.size, -1, dtype=out_np).tofile(case_dir / "output.bin")
    return golden


# ---------------------------------------------------------------------------
# codegen
# ---------------------------------------------------------------------------
CPP_TEMPLATE = """// Generated by run_fuzz.py — ArgMaxV2 fuzz case {cid}: {desc}
#include "benchmark.h"
#include "fileop.h"
#include "multi_thread_res_check.h"
#include "basic_op/argmaxv2/argmaxv2.hpp"

namespace {{
constexpr int kRank = {rank};
{dim_defs}
alignas(4096) {ctype} input[{total}];
alignas(4096) {idtype} output[{out_total}];
{size_asserts}
#ifdef RES_CHECK
MultiThreadResCheckSync res_check_sync{{}};
#endif
}}  // namespace

int main() {{
    const std::uint32_t tid = get_thread_idx();
#ifdef RES_CHECK
    if (tid == 0) {{
        readBinaryFile(CHK_DIR "/input.bin", reinterpret_cast<uint8_t *>(input),
                       sizeof(input));
    }}
    res_check_publish_inputs(res_check_sync, tid);
#endif
    BENCHSTART;
    supernpu::multi_thread::argmaxv2<{ctype}, {idtype}, {axis}{dim_args}>(
        input, output);
    BENCHEND;
#ifdef RES_CHECK
    res_check_wait_for_all(res_check_sync, tid);
    if (tid == 0) {{
        writeBinaryFile(CHK_DIR "/output.bin",
                        reinterpret_cast<uint8_t *>(output), sizeof(output));
    }}
#endif
    return 0;
}}
"""


def gen_cpp(cfg: CaseCfg) -> str:
    info = DTYPE_INFO[cfg.dtype]
    assert cfg.tile is None, (
        "explicit TileM/TileN template args are swallowed by the Dims... pack "
        "(compile-time verified); tile-override cases are not expressible")
    rank = len(cfg.dims)
    dim_defs = "\n".join(
        f"constexpr int kD{i} = {d};" for i, d in enumerate(cfg.dims))
    dim_args = "".join(f", kD{i}" for i in range(rank))
    out_total = cfg.elems // cfg.dims[cfg.axis_norm]
    size_asserts = ""
    if cfg.dtype in ("fp16", "bf16"):
        size_asserts = f'static_assert(sizeof({info["ctype"]}) == 2, "16-bit dtype");'
    return CPP_TEMPLATE.format(
        cid=cfg.cid, desc=cfg.desc, rank=rank, dim_defs=dim_defs,
        ctype=info["ctype"], idtype="int64_t" if cfg.idtype == "i64" else "int32_t",
        total=cfg.elems, out_total=out_total, size_asserts=size_asserts,
        axis=cfg.axis, dim_args=dim_args)


# ---------------------------------------------------------------------------
# build / run / compare
# ---------------------------------------------------------------------------
def build_one(cfg: CaseCfg, compiler_dir: Path, log_dir: Path) -> tuple[str, str]:
    log_path = log_dir / f"{cfg.cid}.build.log"
    cmd = ["make", f"TESTCASE={cfg.cid}", f"COMPILER_DIR={compiler_dir}",
           "res_check=on"]
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, cwd=HERE, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, timeout=600)
        log_path.write_text(proc.stdout, encoding="utf-8")
        if proc.returncode != 0:
            return "BUILD_FAIL", f"make rc={proc.returncode}; log={log_path}"
    except subprocess.TimeoutExpired:
        log_path.write_text("build timeout", encoding="utf-8")
        return "BUILD_FAIL", f"build timeout; log={log_path}"
    elf = OUTPUT / "elf" / f"kernel_argmaxv2_{cfg.cid}_PE4.elf"
    if not elf.is_file():
        return "BUILD_FAIL", f"ELF missing after build: {elf}"
    return "PASS", f"built in {time.time() - t0:.1f}s"


def run_one(cfg: CaseCfg, gfrun: Path, golden_by_cid: dict) -> tuple[str, str, dict]:
    elf = OUTPUT / "elf" / f"kernel_argmaxv2_{cfg.cid}_PE4.elf"
    case_dir = COMPARE / f"kernel_argmaxv2_{cfg.cid}_PE4"
    if not elf.is_file():
        return "TEST_INFRA_FAIL", "missing ELF", {}
    golden = golden_by_cid[cfg.cid]
    prepare_data(cfg, case_dir)
    cmd = [str(gfrun), "-s", "softcore.multiThreadNum=4", "-f", str(elf)]
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, timeout=GFRUN_TIMEOUT)
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout or ""
        if isinstance(out, bytes):
            out = out.decode("utf-8", errors="replace")
        (case_dir / "gfrun.log").write_text(out, encoding="utf-8")
        return "TIMEOUT", f"gfrun timeout after {GFRUN_TIMEOUT}s", {}
    log = proc.stdout
    (case_dir / "gfrun.log").write_text(log, encoding="utf-8")
    elapsed = time.time() - t0
    if proc.returncode != 0:
        return "MODEL_FAIL", f"gfrun rc={proc.returncode}; log={case_dir / 'gfrun.log'}", {}
    if "Reach the End of Benchmark" not in log or "R2 = 0" not in log:
        return "MODEL_FAIL", "missing end marker or R2 != 0", {}
    if "data read to file done" not in log or "data write to file done" not in log:
        return "MODEL_FAIL", "res_check file I/O markers missing", {}
    out_np = np.int64 if cfg.idtype == "i64" else np.int32
    actual = np.fromfile(case_dir / "output.bin", dtype=out_np)
    if actual.size != golden.size:
        return "ACCURACY_FAIL", (
            f"output size actual={actual.size} golden={golden.size}"), {}
    bad = np.nonzero(actual != golden)[0]
    stats = dict(elapsed=round(elapsed, 2),
                 mismatches=int(bad.size),
                 max_abs=int(np.max(np.abs(actual.astype(np.int64) -
                                           golden.astype(np.int64)))) if bad.size else 0)
    if bad.size:
        shown = ", ".join(f"[{i}] got={actual[i]} want={golden[i]}"
                          for i in bad[:8])
        return "ACCURACY_FAIL", f"{bad.size} mismatched indices: {shown}", stats
    return "PASS", "exact match", stats


# ---------------------------------------------------------------------------
# environment identity (immutable source tuple)
# ---------------------------------------------------------------------------
def sh(cmd: list, cwd=None) -> str:
    try:
        return subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE,
                              stderr=subprocess.DEVNULL, text=True,
                              timeout=60).stdout.strip()
    except Exception:
        return "unavailable"


def gather_tuple(compiler_dir: Path, gfrun: Path) -> dict:
    return {
        "bench_worktree": str(REPO),
        "bench_commit": sh(["git", "-C", str(REPO), "rev-parse", "HEAD"]),
        "bench_branch": sh(["git", "-C", str(REPO), "rev-parse", "--abbrev-ref", "HEAD"]),
        "bench_dirty": sh(["git", "-C", str(REPO), "status", "--short"]),
        "argmaxv2_impl_state": "untracked local implementation "
            "(kernels/basic_op/argmaxv2 + test/kernel/argmaxv2, not yet committed)",
        "compiler": sh([str(compiler_dir / "clang"), "--version"]).splitlines()[0]
            if (compiler_dir / "clang").exists() else "unavailable",
        "toolchain_repo_commit": sh(["git", "-C", str(compiler_dir.parents[2]),
                                     "rev-parse", "HEAD"]),
        "gfrun_path": str(gfrun),
        "gfrun_md5": sh(["md5sum", str(gfrun)]).split()[0]
            if gfrun.exists() else "missing",
        "model_commit": sh(["git", "-C", str(gfrun.parents[1]), "rev-parse", "HEAD"]),
        "model_branch": sh(["git", "-C", str(gfrun.parents[1]),
                            "rev-parse", "--abbrev-ref", "HEAD"]),
        "pe_config": "gfrun -s softcore.multiThreadNum=4 (4 PE)",
        "oracle": "np.argmax on stored bit-exact values, atol=rtol=0",
        "gfrun_timeout_s": GFRUN_TIMEOUT,
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gfrun", type=Path, default=GFRUN_DEFAULT)
    ap.add_argument("--compiler-dir", type=Path, default=COMPILER_DEFAULT)
    ap.add_argument("--build-jobs", type=int, default=8)
    ap.add_argument("--run-jobs", type=int, default=6)
    ap.add_argument("--cases", nargs="*", help="restrict to case ids")
    ap.add_argument("--skip-build", action="store_true")
    args = ap.parse_args()

    configs = build_configs()
    if args.cases:
        want = set(args.cases)
        unknown = sorted(want - {c.cid for c in configs})
        if unknown:
            ap.error(f"unknown case ids: {unknown}")
        configs = [c for c in configs if c.cid in want]

    VALIDATION.mkdir(parents=True, exist_ok=True)
    (VALIDATION / "build").mkdir(exist_ok=True)
    tuple_info = gather_tuple(args.compiler_dir, args.gfrun)
    (VALIDATION / "tuple.json").write_text(
        json.dumps(tuple_info, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- phase 1: codegen ----
    src_dir = HERE / "src"
    src_dir.mkdir(exist_ok=True)
    for cfg in configs:
        (src_dir / f"{cfg.cid}.cpp").write_text(gen_cpp(cfg), encoding="utf-8")
    print(f"[GEN] {len(configs)} cases -> {src_dir}", flush=True)

    # ---- phase 2: build (gate first, then batch) ----
    if not args.skip_build:
        gate = min(configs, key=lambda c: c.elems)
        print(f"[GATE] building smallest case {gate.cid} ({gate.desc}) first",
              flush=True)
        status, detail = build_one(gate, args.compiler_dir, VALIDATION / "build")
        print(f"[GATE] {gate.cid}: {status} {detail}", flush=True)
        if status != "PASS":
            print("[GATE] environment gate failed; aborting batch", flush=True)
            return 2
        rest = [c for c in configs if c.cid != gate.cid]
        print(f"[BUILD] {len(rest)} cases with {args.build_jobs} jobs", flush=True)
        t0 = time.time()
        build_status = {}
        with ThreadPoolExecutor(max_workers=args.build_jobs) as pool:
            for cfg, (st, dt) in zip(
                    rest, pool.map(lambda c: build_one(c, args.compiler_dir,
                                                       VALIDATION / "build"), rest)):
                build_status[cfg.cid] = (st, dt)
                if st != "PASS":
                    print(f"[BUILD] {cfg.cid}: {st} {dt}", flush=True)
        ok = sum(1 for st, _ in build_status.values() if st == "PASS") + 1
        print(f"[BUILD] done: {ok}/{len(configs)} built in {time.time()-t0:.0f}s",
              flush=True)
        failed_builds = [c for c in configs if build_status.get(c.cid, ("PASS",))[0] != "PASS"]
        if failed_builds:
            print(f"[BUILD] {len(failed_builds)} build failures: "
                  f"{[c.cid for c in failed_builds]}", flush=True)
    else:
        print("[BUILD] skipped (--skip-build)", flush=True)

    # ---- phase 3: data prep + gfrun + compare ----
    golden_by_cid = {}
    for cfg in configs:
        case_dir = COMPARE / f"kernel_argmaxv2_{cfg.cid}_PE4"
        golden_by_cid[cfg.cid] = prepare_data(cfg, case_dir)

    runnable = [c for c in configs
                if (OUTPUT / "elf" / f"kernel_argmaxv2_{c.cid}_PE4.elf").is_file()]
    print(f"[RUN] {len(runnable)} cases with {args.run_jobs} gfrun workers "
          f"(timeout {GFRUN_TIMEOUT}s each)", flush=True)
    results = {}
    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=args.run_jobs) as pool:
        for cfg, (st, dt, stats) in zip(
                runnable, pool.map(lambda c: run_one(c, args.gfrun, golden_by_cid),
                                   runnable)):
            results[cfg.cid] = dict(status=st, detail=dt, **stats)
            done += 1
            if st != "PASS":
                print(f"[RUN {done:03d}/{len(runnable)}] {cfg.cid} {st}: {dt}",
                      flush=True)
            elif done % 25 == 0:
                print(f"[RUN {done:03d}/{len(runnable)}] ... running "
                      f"({time.time()-t0:.0f}s elapsed)", flush=True)

    for cfg in configs:
        if cfg.cid not in results:
            results[cfg.cid] = dict(status="BUILD_FAIL",
                                    detail=build_status.get(cfg.cid, ("", "no ELF"))[1]
                                    if not args.skip_build else "no ELF")

    # ---- phase 4: report ----
    order = {c.cid: i for i, c in enumerate(configs)}
    rows = sorted(results.items(), key=lambda kv: order[kv[0]])
    tsv = VALIDATION / "report.tsv"
    with tsv.open("w", encoding="utf-8") as f:
        f.write("cid\tdtype\tidtype\taxis\tdims\ttile\tpattern\telems\t"
                "status\tdetail\telapsed_s\tmismatches\n")
        cfg_by_id = {c.cid: c for c in configs}
        for cid, r in rows:
            c = cfg_by_id[cid]
            tile = "def" if c.tile is None else "x".join(map(str, c.tile))
            f.write(f"{cid}\t{c.dtype}\t{c.idtype}\t{c.axis}\t"
                    f"{','.join(map(str, c.dims))}\t{tile}\t{c.pattern}\t{c.elems}\t"
                    f"{r['status']}\t{r['detail']}\t{r.get('elapsed', '')}\t"
                    f"{r.get('mismatches', '')}\n")

    (VALIDATION / "results.json").write_text(
        json.dumps(dict(tuple=tuple_info, results=results,
                        cases={c.cid: asdict(c) for c in configs}),
                   indent=2, ensure_ascii=False), encoding="utf-8")

    # refresh manifest with ELF hashes after build
    manifest = {
        "tuple": tuple_info,
        "cases": [
            dict(asdict(c), elems=cfg.elems,
                 elf_sha256=hashlib.sha256(
                     (OUTPUT / "elf" / f"kernel_argmaxv2_{cfg.cid}_PE4.elf")
                     .read_bytes()).hexdigest()
                 if (OUTPUT / "elf" / f"kernel_argmaxv2_{cfg.cid}_PE4.elf").exists()
                 else None)
            for cfg in configs],
    }
    (VALIDATION / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    counts = {}
    for r in results.values():
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print("\n===== SUMMARY =====")
    for k in sorted(counts):
        print(f"{k}: {counts[k]}")
    print(f"report: {tsv}")
    print(f"results: {VALIDATION / 'results.json'}")
    return 0 if all(r["status"] == "PASS" for r in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
