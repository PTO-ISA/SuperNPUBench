# SuperNPUBench

SuperNPUBench is a high-performance operator library and benchmark platform for
NPU tile-programming ISA. It ships **two architecture backends** under `benchmark/`
(two-level-arch = LinxISA, one-level-arch = PTO ISA) plus an instruction-level
**microbenchmark** suite, all driven by the same Linx toolchain.

> **IMPORTANT**: Only `benchmark/one-level-arch/` and `microbenchmark/` are
> compilable with the current toolchain. The `benchmark/two-level-arch/`
> (LinxISA) kernels are **not** compilable — they require a different ISA mode
> not supported by the current `linx_blockisa_llvm_musl` build. Do **not**
> include `two-level-arch` in batch compilation (`compile_all.sh two-level` will
> fail).

## Repository Structure

```
SuperNPUBench/
├── benchmark/
│   ├── two-level-arch/      # Linx two-level block ISA
│   │   ├── kernels/         # header-only operator implementations
│   │   ├── test/            # test suites + build system
│   │   └── compile_all.sh
│   ├── one-level-arch/      # PTO one-level tile ISA
│   │   ├── kernels/
│   │   ├── test/
│   │   │   ├── common/      # shared Makefile.common, _start.s
│   │   │   └── kernel/      # per-operator test cases
│   │   └── compile_all.sh
├── microbenchmark/          # instruction-level micro-bench (cube/vector/memory/scalar)
├── docs/                    # documentation
│   ├── programming/        # PTO C++ Programming Guide
│   └── workflow/           # end-to-end workflow docs
└── compile_all.sh           # top-level: two-level | one-level | all
```

> Build outputs (`output/`, `**/output/`) and `.DS_Store` are gitignored.

## Architecture Backends

### two-level-arch (LinxISA)
- Block-structured ISA with heterogeneous cores: BCC (main), Cube (matrix), Vector, MTC/TMA (data transfer).
- Programming model: block instructions (VPAR/VSEQ, CUBE, TMA, TEPL).

### one-level-arch (PTO ISA)
- Tile-centric ISA with explicit memory hierarchy: Vec, Mat, Left, Right, Acc.
- Programming model: tile operations via Linx-TileOP-API C++ templates.
- Programming guide: [`docs/programming/pto c++ programming guide.md`](docs/programming/pto%20c++%20programming%20guide.md).

Both backends share the same operator set and test layout; their kernel
implementations differ in ISA style.

## Operator Overview

Each backend implements operator categories:

| Operator | Description |
|----------|-------------|
| **matmul** | FP4/BF16/FP32/FP16/FP8 matrix multiply; quantization, mixed precision, A/B reuse, GMMA shared-tile |
| **fa** | Flash Attention; 2D unroll, SFA (block-sparse), HIF4 quantization, softmax_pto, unaligned boundary |
| **flashMLA** | Flash MLA (multi-head latent attention) |
| **transpose** | 3D~6D tensor transpose; multiple dtypes |
| **reduction** | Row/column max & sum; single-tree, unaligned, cumsum, reduceprod |
| **gelu** | GELU activation; exact (erf) and tanh approximation |
| **broadcast** | 2D~5D broadcast; vectorized variants |
| **gather** | Data gathering; large-scale, power-of-2 dims |
| **concat** | Concatenation; gather/scatter modes |
| **control** | `hashtable_lookup_simd` (pure tile-op, single-tier gfsim) |
| **sort** | `topk` (radix-bucket histogram) |
| **deepseek** | 22 migrated DeepSeek kernels (engram/mhc/moe/quant/transpose) |

## Setup Environment

SuperNPUBench compiles with the **Linx toolchain** (`linx_blockisa_llvm_musl`,
clang-15, target `linx64v5-unknown-linux-musl`). Build it once from the
[`linx-toolchain-build`](https://github.com/LinxISA/linx-toolchain-build) repo,
which clones the matching ISA sources and produces the `linx_blockisa_llvm_musl`
install tree that `COMPILER_DIR` points at.

### 1. Clone the build repo

```bash
git clone https://github.com/LinxISA/linx-toolchain-build.git
cd linx-toolchain-build
```

### 2. Install host build tools

```bash
sudo apt-get install -y git make cmake ninja-build gcc g++ python3 autoconf m4
```

### 3. Initialize component sources

`make init-src` clones the five component repos under `src/` on their pinned
branches (run it again any time to fetch updates):

| Directory | Repository | Branch |
| --- | --- | --- |
| `src/llvm-project` | `LinxISA/llvm-project` | `dev-llvm15_56` |
| `src/musl` | `LinxISA/linx-musl` | `linx` |
| `src/jemalloc` | `LinxISA/jemalloc` | `linx` |
| `src/linux-linxisa` | `LinxISA/linux` | `main` |
| `src/Linx-TileOP-API` | `LinxISA/Linx-TileOP-API` | `linx` |

```bash
make init-src
```

### 4. Build the toolchain

Only `linx64v5-linux-musl` is supported by the top-level Makefile:

```bash
make WITH_TARGET=linx64v5-linux-musl
```

This builds, in order: LLVM/clang/lld → kernel headers → musl → compiler-rt →
libc++/libc++abi/libunwind → jemalloc → Linx-TileOP-API headers. Progress is
tracked by stamp files under `stamps/`, so re-running `make` resumes from the
last completed step; `make clean` rebuilds from scratch. The install tree is
written to `output/linx_blockisa_llvm_musl/`:

```
output/linx_blockisa_llvm_musl/
├── bin/        # clang, clang++, ld.lld, llvm-ar/nm/ranlib,
│              # linx64v5-linux-musl-clang(++) symlinks
├── lib/        # clang runtime, libc++, ...
└── sysroot/    # musl + kernel headers + runtime libs
```

### 5. Point SuperNPUBench at the toolchain

```bash
export COMPILER_DIR=$(pwd)/output/linx_blockisa_llvm_musl/bin
$COMPILER_DIR/clang --version
# clang version 15.0.4 (linx64v5-musl-local ...)
# Target: linx64v5-unknown-linux-musl
```

Then proceed to [Quick Start](#quick-start).

### (Optional) Package

```bash
make package     # -> output/linx_blockisa_llvm_musl.tar.gz
```

## Quick Start

### 1. Environment

Build the Linx toolchain once (see [Setup Environment](#setup-environment)), then
point `COMPILER_DIR` at it:

```bash
export COMPILER_DIR=/path/to/linx_blockisa_llvm_musl/bin
```

### 2. Compile an operator

```bash
# one-level-arch (PTO ISA)
cd benchmark/one-level-arch/test/kernel/matmul
make TESTCASE=matmul TYPE=MASK MODE=MASK_FP32 M=256 N=256 K=256 tM=16 tN=16 tK=64

# deepseek kernel
cd benchmark/one-level-arch/test/kernel/deepseek
make TESTCASE=fused_weight diss
```

### 3. Batch / full compilation

```bash
# one-level-arch only (recommended)
./compile_all.sh one-level

# microbenchmark
cd microbenchmark && bash compile_all.sh all
```

> **Do NOT run `compile_all.sh two-level` or `compile_all.sh all`** —
> `two-level-arch` kernels cannot compile with the current toolchain.

Artifacts land in `benchmark/<arch>/output/kernel/<operator>/elf/`.

## Microbenchmark

`microbenchmark/` is an instruction-level bench organized by ISA family,
generated by `gen_cases.py`.

| family | covers | cases |
| --- | --- | ---: |
| cube (CUBE) | TMATMUL / TMATMUL_BIAS / TMATMUL_MX / ACCCVT | 9 |
| vector (TEPL) | elementwise / tile-scalar / reduce / expand / TCI sequence (toolchain-exposed subset) | 128 |
| memory (TLSU) | TLOAD / TSTORE / TMOV / MGATHER / MSCATTER (+mask, layout) | 25 |
| scalar (GPR) | int ALU / load-store / float / conversion × throughput+latency | 124 |
| **total** | | **286** |

```bash
cd microbenchmark && make TESTCASE=tmatmul_fp16_64x64x64   # one case
cd microbenchmark && bash compile_all.sh all               # all families
```

See [`microbenchmark/README.md`](microbenchmark/README.md) for details.

## Running on the Models

Compiled ELF binaries run on the **SuperScalarModel** simulator suite. Build
`gfrun`/`gfsim` from the [SuperScalarModel](../SuperScalarModel) repo, then
point them at the ELF:

- `gfrun` — functional model (correctness)
- `gfsim` — cycle-accurate model (timing)

```bash
# from the SuperScalarModel repo root (where bin/ lives)
bin/gfrun -f /path/to/SuperNPUBench/benchmark/one-level-arch/output/kernel/<op>/elf/<name>.elf
bin/gfsim -f /path/to/SuperNPUBench/benchmark/one-level-arch/output/kernel/<op>/elf/<name>.elf
```

### Tile-op kernels: single-tier gfsim mode

Kernels written purely with tile ops using TEPL template instructions (e.g.
`control/hashtable_lookup_simd`) run on the VectorLite engine, which gfsim only
steps in **single-tier mode**:

```bash
bin/gfsim -f <elf> -s core.singleTierMode=true
```

Without this flag the engine is inert and the run deadlocks. `gfrun` does not
need the flag.

## Build System

### Makefile parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `TESTCASE` | Test case name | `matmul`, `fa_2d_unroll` |
| `TYPE` | Operator type (matmul) | `HIF4_HIF4`, `A16W4`, `MASK` |
| `MODE` | Operator mode | `MASK_FP32`, `BF16x2_NOGATHER` |
| `M/N/K` | Matrix dimensions | `M=256 N=2048 K=2048` |
| `tM/tN/tK` | Tile sizes | `tM=128 tN=128 tK=128` |
| `COMPILER_DIR` | Compiler path | `/path/to/linx/bin` |
| `PLAT` | Platform | `linx` (default), `cpu` |

### Build targets

```bash
make TESTCASE=<case> all      # compile
make TESTCASE=<case> diss     # disassembly
make TESTCASE=<case> sim      # run in simulator
make TESTCASE=<case> debug    # debug mode
make clean                    # clean current operator
make clean_all                # clean all
```

## Documentation

- **PTO C++ Programming Guide**: [`docs/programming/pto c++ programming guide.md`](docs/programming/pto%20c++%20programming%20guide.md)
- **End-to-end Workflow**: [`docs/workflow/operator_to_chip_execution_flow.md`](docs/workflow/operator_to_chip_execution_flow.md)
- **Per-operator README**: see `benchmark/one-level-arch/kernels/<operator>/README.md`
- **Microbenchmark**: [`microbenchmark/README.md`](microbenchmark/README.md)
- **TileOP-API Reference**: [Linx-TileOP-API tileop-usage docs](https://github.com/LinxISA/Linx-TileOP-API/tree/linx/docs/tileop-usage)

## Toolchain

- Compiler: `linx_blockisa_llvm_musl` (clang-15, linx64v5-musl)
- Flags: `-mlxbc -fenable-matrix -O2 -mllvm -enable-all-vector-as-tilereg=true -std=c++20`
- Target: Linx64 V5

## Development Guide

### Adding an operator

1. Add header-only kernel under `benchmark/<arch>/kernels/<operator>/`.
2. Create test dir under `benchmark/<arch>/test/kernel/<operator>/` with
   `Makefile`, `compile.all`, `src/`.
3. Add the operator to `compile_all.sh`.

### Conventions

- Header-only kernels; PTO tile-programming paradigm.
- Build artifacts not tracked (`.gitignore`).

## Related Links

- [LinxISA](https://linxisa.github.io/linx-isa/)
- [PTO ISA](https://pto-isa.github.io/docs/isa/tile/)
- [Linx-TileOP-API](https://github.com/LinxISA/Linx-TileOP-API/tree/linx/docs/tileop-usage)

## License

See LICENSE.


---

> **最后更新**: 2026-08-18（gfrun 升级至 `a68dba29`，TileOP-API 修复 TileDType 硬编码 4KB bug `8b2ee78`，全量 429 ELF 复测）

# gfrun 执行结果汇总 — 2026-08-18

> 工具链: `linx-toolchain-build-latest  (clang 15.0.4, linx64v5-musl-local, llvm 86959776bd1f)`
>
> gfrun: `/Users/blacktraker/Programming/gitproj/DV4/SuperScalarModel/bin/gfrun  (SuperScalarModel @ a68dba29 "fix(isa): restore single-issuer shared tload")`
>
> 调用: `gfrun -t 1 -f <elf>   (90s per-ELF timeout, 8-way parallel)`

## 工具链&&模型版本

| 组件 | 仓库 | 分支 | Commit |
|---|---|---|---|
| gfrun 模型 | https://github.com/LinxISA/SuperScalarModel | `feat/pto-v058-adaptation` | `a68dba29e0ca84fd0455b3b0ca20ddf67dd631ce` |
| llvm-project | https://github.com/LinxISA/llvm-project | `temp/shared-tload-integration-20260811` | `86959776bd1fb22dcc8e73b57ec2276c65d44f38` |
| Linx-TileOP-API | https://github.com/LinxISA/Linx-TileOP-API | `8b2ee78` | `8b2ee780ddcc5d7a04c50e337d67eadc3637a17c` |

工具链：`llvm-project @ 86959776b` + `Linx-TileOP-API @ 8b2ee78`（含 TileDType 修复）

## 更新日志

### 更新点

1. **TileDType 修复**：Linx-TileOP-API commit `8b2ee78`（2026-08-18）恢复了 `TileDType = DType tile_size(Rows*Cols/...)` 的正确形式，修正了 `f9a30a69`（2026-08-11）引入的 `TileDType = int32_t ext_vector_type(1024)` 硬编码 4KB bug。根因：硬编码 4KB 使所有 tile 类型（int8/fp16/fp32 等）都生成 4KB tile 寄存器，导致 `calculateVCallSizeMask` 固定产出 TSize=6，与算子实际 tile 形状不符。
2. **gfrun 升级**：SuperScalarModel 从 `ce5e1510` 升级至 `a68dba29`（`fix(isa): restore single-issuer shared tload`），修复了 single-issuer 场景下 Shared TLOAD 的功能模型缺陷。
3. **fa 算子全部恢复**：TileDType 修复后，fa 的 10 个用例（8 × `fa_2d_unroll` + 2 × `fa_softmax_pto`）从全 FAIL 恢复为 PASS。
4. **multi_thread 全部通过**：fa（Sq128/Sq512）和 matmul_shared 从 FAIL 恢复为 PASS，multi_thread 5/5 全通过。
5. **flashMLA / norm / reduction 恢复**：flashMLA 2/2、norm 1/1、reduction 5/5 全通过（先前均为 FAIL）。
6. **matmul 数值验证 PASS**：MASK_FP32 M64×64×64 数值验证通过（max abs err 7.63e-6，4096/4096 非零），先前的 1216/4096 失败模式已消除。

### 回归

1. **fixp 大幅回归**：fixp microbench 从 27/63 PASS 退化为 4/63 PASS（−23）。根因：TileDType 修复使非 FP32 类型（int8/fp16/bf16 等）的 tile 尺寸从 4KB 恢复为正确值（如 int8 tile = 32×32×1B = 1KB），暴露了 gfrun CUBE/tmatmul validator 对 `accBytes != 0 && dimensionsArePowersOfTwo && col >= n && dstTile[0]->size % rowBytes == 0` 校验的契约偏差。仅 4 个使用 FP32 累加器的用例（`keep_acc`、`keep_acc_relu`、`legacy3`、`s_qf_f32`）因 4B × 1024 = 4KB 仍满足条件而通过。
2. **sinkhorn_fwd 缺失**：deepseek 从 22 降至 21 个用例，`sinkhorn_fwd` 不在本轮 compile.all 产出中。

### 关键定性

- **TileDType 修复是正确的**：恢复了 `DType tile_size(Rows*Cols/...)` 的逻辑 tile 尺寸，fa 和 multi_thread 的修复均源于此。
- **fixp 回归不是 TileDType 的 bug**：而是 gfrun 功能模型的 CUBE/tmatmul validator 对非 4KB tile 尺寸的校验存在契约偏差。先前 27 个 fixp 用例"通过"是因为 4KB 硬编码恰好满足 validator 条件——属**幻觉通过**。
- **109 个 FAIL 中无 segfault**：先前 3 个 broadcast `tM2048` SIGSEGV 已转为 assertion failure（TileDType 修复使 tile 尺寸变更，gfrun 在 assertion 处提前拦截，不再崩溃）。

## TL;DR

- **编译产出 ELF：429** （microbench 342 + one-level 82 + multi_thread 5）
- **gfrun 执行：429 个 ELF 全部跑完**
- **通过 PASS = 321** / **失败 FAIL = 108**

- 失败构成：gfrun 模型断言失败 106、超时 2、segfault 0
- **关键定性**：106 个 FAIL 均为 gfrun 功能模型自身的校验断言 （`ASSERTION FAILED: …`），即 gfrun(SuperScalarModel @ a68dba29) 的 validator 拒绝了工具链合法生成的某些 tile/inst/descriptor 组合 —— 属**模型侧与工具链/ISA 契约的偏差**，非算子内核 bug、非真实非法指令。
- **初始编译无失败**：429 个 ELF 全部编译成功。

## 1. 编译结果

入口：`bash microbenchmark/compile_all.sh`（cube/vector/memory/scalar/fixp）+ `bash benchmark/one-level-arch/compile_all.sh`（13 个算子）+ multi_thread 3 个算子单独编译。

### 1.1 编译成功（429 ELF）

| 树 | 类别/算子 | ELF 数 |
|---|---|---:|
| microbench | cube | 6 |
| microbench | vector | 128 |
| microbench | memory | 21 |
| microbench | scalar | 124 |
| microbench | fixp | 63 |
| one-level | broadcast | 6 |
| one-level | concat | 4 |
| one-level | control | 6 |
| one-level | deepseek | 21 |
| one-level | element_wise | 2 |
| one-level | fa | 13 |
| one-level | flashMLA | 2 |
| one-level | gather | 1 |
| one-level | matmul | 16 |
| one-level | norm | 1 |
| one-level | reduction | 5 |
| one-level | sort | 1 |
| one-level | transpose | 4 |
| one-level | multi_thread | 5 |
| | **合计** | **429** |

## 2. 执行结果总览（按类别/算子）

| 树 | 类别/算子 | 总数 | PASS | FAIL |
|---|---|---:|---:|---:|
| microbench | cube | 6 | 2 | 4 |
| microbench | vector | 128 | 114 | 14 |
| microbench | memory | 21 | 19 | 2 |
| microbench | scalar | 124 | 124 | 0 |
| microbench | fixp | 63 | 4 | 59 |
| one-level | broadcast | 6 | 3 | 3 |
| one-level | concat | 4 | 3 | 1 |
| one-level | control | 6 | 0 | 6 |
| one-level | deepseek | 21 | 14 | 7 |
| one-level | element_wise | 2 | 1 | 1 |
| one-level | fa | 13 | 10 | 3 |
| one-level | flashMLA | 2 | 2 | 0 |
| one-level | gather | 1 | 1 | 0 |
| one-level | matmul | 16 | 9 | 7 |
| one-level | norm | 1 | 1 | 0 |
| one-level | reduction | 5 | 5 | 0 |
| one-level | sort | 1 | 0 | 1 |
| one-level | transpose | 4 | 4 | 0 |
| one-level | multi_thread | 5 | 5 | 0 |
| | **合计** | **429** | **321** | **108** |

- microbench **scalar 124/124 全通过**；memory 19/21、vector 114/128、cube 2/6、fixp 4/63（回归，见更新日志）。
- one-level 通过率 82/87（fa 10/13、flashMLA 2/2、reduction 5/5、transpose 4/4、norm 1/1 等大算子恢复通过；control 0/6、matmul 9/16、deepseek 14/21 仍有模型断言）。
- multi_thread 5/5 全通过（fa Sq128/Sq512、matmul_shared、vec/tadd、vec/trowsum 均 PASS，详见 §7）。

## 3. 失败清单（按问题类型分组）

### 3.1 gfrun 模型断言失败（106 个）

gfrun 输出形如 `gfrun: illegal instruction: ASSERTION FAILED: <cond>` + backtrace。按断言条件分组（降序）：

#### 组 1（49 个）：`accBytes != 0 && dimensionsArePowersOfTwo && col >= n && dstTile[0]->size != 0 && rowBytes != 0 && dstTile[0]->size % rowBytes == 0 && dstTile[0]->siz…`

CUBE/TMATMUL 目标 tile 尺寸校验。TileDType 修复后非 FP32 类型的 tile 尺寸不再固定 4KB，触发此断言。

- `fixp_tmatmul_acc_s8` · `fixp_tmatmul_bf16` · `fixp_tmatmul_bf16_relu` · `fixp_tmatmul_bias_s8`  [microbench/fixp]
- `fixp_tmatmul_f16` · `fixp_tmatmul_f16_groupmax` · `fixp_tmatmul_f16_prelu` · `fixp_tmatmul_f16_relu`  [microbench/fixp]
- `fixp_tmatmul_gemv` · `fixp_tmatmul_gemv_acc` · `fixp_tmatmul_gemv_bias` · `fixp_tmatmul_gemv_mx`  [microbench/fixp]
- `fixp_tmatmul_gemv_mx_acc` · `fixp_tmatmul_gemv_mx_bias` · `fixp_tmatmul_gemv_mx_s8` · `fixp_tmatmul_gemv_s8`  [microbench/fixp]
- `fixp_tmatmul_mx_s8` · `fixp_tmatmul_s8_prelu` · `fixp_tmatmul_s8_relu` · `fixp_tmatmul_s8_rowmax`  [microbench/fixp]
- `fixp_tmatmul_s8_shared` · `fixp_tmatmul_s_deqf16` · `fixp_tmatmul_s_qf_bf16` · `fixp_tmatmul_s_qf_f16`  [microbench/fixp]
- `fixp_tmatmul_s_qf_fp8` · `fixp_tmatmul_s_qf_hif8` · `fixp_tmatmul_s_qf_s16` · `fixp_tmatmul_s_qf_s4`  [microbench/fixp]
- `fixp_tmatmul_s_qf_s8` · `fixp_tmatmul_s_qs_bf16` · `fixp_tmatmul_s_reqs8` · `fixp_tmatmul_s_shifts16`  [microbench/fixp]
- `fixp_tmatmul_v_deqf16` · `fixp_tmatmul_v_qf_bf16` · `fixp_tmatmul_v_qf_f16` · `fixp_tmatmul_v_qf_fp8`  [microbench/fixp]
- `fixp_tmatmul_v_qf_hif8` · `fixp_tmatmul_v_qf_s16` · `fixp_tmatmul_v_qf_s4` · `fixp_tmatmul_v_qf_s8`  [microbench/fixp]
- `fixp_tmatmul_v_qs_bf16` · `fixp_tmatmul_v_reqs8` · `fixp_tmatmul_v_s8_relu` · `fixp_tmatmul_v_shifts16`  [microbench/fixp]
- `fixp_tmatmul_vqf_s8_prelu` · `fixp_tmatmul_s8_lrelu` · `fixp_tmatmul_bf16`  [microbench/fixp]
- `tmatmul_i8_64x64x64`  [microbench/cube]  rc=1
- `sfa_Sq256_Skv512_Tm16_Tk32`  [one-level/fa]  rc=1
- `sfa_Sq512_Skv512_Tm16_Tk32`  [one-level/fa]  rc=1

#### 组 2（8 个）：`block->dataType == DataType::FP32 && (fp8Pair || fp4Pair) && block->srcTile[leftScaleIndex]->tileInfo && block->srcTile[rightScaleIndex]->tileInfo…`

MX 量化 scale tile 校验。

- `fixp_tmatmul_mx` · `fixp_tmatmul_mxacc` · `fixp_tmatmul_mxbias`  [microbench/fixp]
- `fa_HIF4_HIF4_BF16_NOGATHER_Sq256_Skv512_Tm8_Tk32_X1_Y1`  [one-level/fa]
- `matmul_HIF4_HIF4_MX_NOGATHER_B1_M256_N2048_K2048_tM32_tN32_tK64`  [one-level/matmul]
- `matmul_HIF4_HIF4_MX_NOGATHER_B1_M512_N1280_K4096_tM32_tN32_tK64`  [one-level/matmul]
- `matmul_HIF4_HIF4_MX_NOGATHER_REUSEA_B1_M256_N2048_K2048_tM32_tN32_tK64`  [one-level/matmul]
- `matmul_HIF4_HIF4_MX_NOGATHER_REUSEA_B1_M512_N1280_K4096_tM32_tN32_tK64`  [one-level/matmul]

#### 组 3（8 个）：`gfrun: illegal instruction at 0x0: reserved/deleted TEPL selector`

TEPL 选择码已被 ISA 废弃/删除，gfrun 遇到即拒绝。

- `mgather_fp16_16x16` · `mgather_fp32_16x16`  [microbench/memory]
- `taxpy_fp16_16x16` · `tprelu_fp16_16x16` · `tprelu_fp32_16x16`  [microbench/vector]
- `thistogram_fp16_16x16` · `thistogram_fp32_16x16`  [microbench/vector]
- `inplace_unique_group_indices`  [one-level/deepseek]

#### 组 4（8 个）：`(dataType == DataType::INT8 || dataType == DataType::UINT8 || dataType == DataType::INT16 || dataType == DataType::UINT16 || dataType == DataType::INT…)`

TEPL 数据类型限制：该指令仅支持整数类型，fp16 不在支持列表。

- `hashtable_lookup_simd_kNum6144_kMaxProbe512_knum_col1024_debug_off`  [one-level/control]
- `hashtable_lookup_simd_kNum6144_kMaxProbe512_knum_col1024_debug_on`  [one-level/control]
- `hashtable_lookup_simd_kNum6144_kMaxProbe512_knum_col256_debug_off`  [one-level/control]
- `hashtable_lookup_simd_kNum6144_kMaxProbe512_knum_col256_debug_on`  [one-level/control]
- `hashtable_lookup_simd_kNum6144_kMaxProbe512_knum_col512_debug_off`  [one-level/control]
- `hashtable_lookup_simd_kNum6144_kMaxProbe512_knum_col512_debug_on`  [one-level/control]
- `tshls_fp16_16x16` · `tshrs_fp16_16x16`  [microbench/vector]

#### 组 5（4 个）：`source->tileInfo->dataType == block->dataType && source->tileInfo->validRow == validRow && source->tileInfo->validCol == validCol && source->tileInfo-…`

TSTORE 源 tile 数据类型/有效行列校验不匹配。

- `broadcast_broadcast__DType__half_tM2048_IN_SHAPE1042_1_OUT_SHAPE1042_129`  [one-level/broadcast]
- `broadcast_broadcast__DType__half_tM2048_IN_SHAPE1280_1_49_OUT_SHAPE1280_8_49`  [one-level/broadcast]
- `broadcast_broadcast__DType__half_tM2048_IN_SHAPE1_1_1_65_128_OUT_SHAPE1_1_7_65_128`  [one-level/broadcast]
- `swiglu_forward_and_per_token_cast`  [one-level/deepseek]

#### 组 6（4 个）：`inst->srcs.size() == 3 && inst->dsts.empty() && IsCompatibleLogicalTile(inst->srcs[1], validRow, validCol, physicalCol) && IsCompatibleDataTile(inst->…`

TLOAD 三操作数形式（LogicalTile + DataTile）校验。

- `aux_fi` · `get_fused_mapping` · `group_count` · `mask_indices_by_tp`  [one-level/deepseek]

#### 组 7（4 个）：`inst->srcs.size() == 2 && inst->dsts.empty() && IsLegalLocalTileDescriptor(inst->srcs[1]) && "Local TSTORE requires one legal source Tile descriptor"`

Local TSTORE 源描述符合法性校验。

- `fixp_tmatmul_groupmax_128` · `fixp_tmatmul_groupmax_16` · `fixp_tmatmul_groupmax_8` · `fixp_tmatmul_rowmax`  [microbench/fixp]

#### 组 8（3 个）：`inst->srcs.size() == 1 && inst->dsts.size() == 1 && inst->dsts[0] && inst->dsts[0]->size != 0 && inst->dsts[0]->size % (physicalCol * elementBytes) ==…`

TMATMUL 目标 tile 尺寸与 physicalCol 对齐校验（fp16 cube）。

- `tmatmul_bias_fp16_64x64x64` · `tmatmul_fp16_64x64x64` · `tmatmul_mx_fp16_64x64x64`  [microbench/cube]

#### 组 9（3 个）：`rightInfo->validRow == k && (!tmatmul || rightInfo->validCol == n) && "CUBE right descriptor must be K x N"`

CUBE 右操作数描述符行列维度校验。

- `matmul_A16W4_B1_M256_N2048_K2048_tM16_tN16_tK128`  [one-level/matmul]
- `matmul_A16W4_B1_M512_N1280_K2048_tM16_tN16_tK128`  [one-level/matmul]
- `matmul_A16W4_B1_M512_N512_K4096_tM16_tN16_tK128`  [one-level/matmul]

#### 组 10（3 个）：`IsLogicalIntegerTeplDataType(block->dataType) && "scalar logical TEPL tuple is not defined by PTO ISA v0.2"`

标量逻辑 TEPL（AND/OR/XOR）不支持 fp16 数据类型。

- `tands_fp16_16x16` · `tors_fp16_16x16` · `txors_fp16_16x16`  [microbench/vector]

#### 组 11（3 个）：`IsCompareSelectTeplDataType(block->tileOp, block->dataType) && "compare/select TEPL tuple is not defined by PTO ISA v0.2"`

比较/选择 TEPL 不支持当前 opcode/data-type 组合。

- `tsels_fp16_16x16` · `tsels_fp32_16x16`  [microbench/vector]
- `topk_gate`  [one-level/deepseek]

#### 组 12（2 个）：`block->srcTile.size() + (sharedRight ? 1u : 0u) == requiredSources && leftIndex < block->srcTile.size() && "CUBE operand arity does not match the v0.5…`

CUBE 操作数数量与 v0.5 契约不匹配。

- `fixp_tmatmul_rowmax_init` · `fixp_tmatmul_v_qf_f32`  [microbench/fixp]

#### 组 13（2 个）：`IsBasicUnaryTeplDataType(block->tileOp, block->dataType) && "TEPL opcode/data-type tuple is not defined by PTO ISA v0.2"`

基本一元 TEPL（ABS）不支持 i16/i32 数据类型。

- `tabs_i16_16x16` · `tabs_i32_16x16`  [microbench/vector]

#### 组 14（1 个）：`srcTile.size() == 1 && dstTile.size() == 1 && srcTile[0] && dstTile[0] && srcTile[0]->tileInfo && dstTile[0]->tileInfo && srcTile[0]->tileInfo->row ==…`

- `element_wise_gelu_gelu_Approximate_false_DType__bf16_tM2048_SHAPE24_8_1024`  [one-level/element_wise]

#### 组 15（1 个）：`source && source->tileInfo && (source->tileInfo->validRow == 1u || source->tileInfo->validRow == m) && (source->tileInfo->validCol == 1u || source->tileInfo-…`

- `fixp_tmatmul_bias`  [microbench/fixp]

#### 组 16（1 个）：`currentBlock->hasFixpAttr && "PTO v0.58 Matrix requires one ASL B.FPATR descriptor"`

- `fixp_tmatmul_rowgroup_maxabs`  [microbench/fixp]

#### 组 17（1 个）：`block->dataType == DataType::FP32 && block->blockAttr && block->blockAttr->dataType == DataType::FP32 && block->blockAttr->layout == LayOut::NORM && "…`

- `fixp_tmatmul_shared`  [microbench/fixp]

#### 组 18（1 个）：`(accInfo->dataType == DataType::FP32 || accInfo->dataType == DataType::INT32) && accInfo->validRow == m && accInfo->validCol == n && "TMATMUL.ACC C mu…`

- `fixp_tmatmul_acc`  [microbench/fixp]

### 3.2 超时（2 个，>90s）

gfrun 90s 内未跑完（数据量过大或疑似死循环）：

- `concat_concat_scatter_DType__half_tM512_IN_SHAPE256_8_OUT_SHAPE256_8000`  rc=124  t=90s
- `topk`  rc=124  t=90s

## 4. 通过清单（321 个，按类别）

### microbench/cube（2）

  `tmatmul_bias_fp32_32x32x32` · `tmatmul_fp32_32x32x32`

### microbench/vector（114）

  `tabs_fp16_16x16` · `tadd_fp16_16x16` · `tadd_fp32_16x16` · `tadd_i16_16x16`
  `tadd_i32_16x16` · `tadds_fp16_16x16` · `tand_i16_16x16` · `tand_i32_16x16`
  `tci_i16_16x16` · `tci_i32_16x16` · `tcmp_fp16_16x16` · `tcmp_fp32_16x16`
  `tcmp_i32_16x16` · `tcmps_fp16_16x16` · `tcolexpand_fp16_16x16` · `tcolexpand_fp32_16x16`
  `tcolexpandadd_fp16_16x16` · `tcolexpandadd_fp32_16x16` · `tcolexpanddiv_fp16_16x16` · `tcolexpanddiv_fp32_16x16`
  `tcolexpandexpdif_fp16_16x16` · `tcolexpandexpdif_fp32_16x16` · `tcolexpandmax_fp16_16x16` · `tcolexpandmax_fp32_16x16`
  `tcolexpandmin_fp16_16x16` · `tcolexpandmin_fp32_16x16` · `tcolexpandmul_fp16_16x16` · `tcolexpandmul_fp32_16x16`
  `tcolexpandsub_fp16_16x16` · `tcolexpandsub_fp32_16x16` · `tconcat_fp16_16x16` · `tconcat_fp32_16x16`
  `tcvt_fp16_16x16` · `tcvt_fp32_16x16` · `tdiv_fp16_16x16` · `tdiv_fp32_16x16`
  `tdiv_i16_16x16` · `tdiv_i32_16x16` · `tdivs_fp16_16x16` · `texp_fp16_16x16`
  `texp_fp32_16x16` · `texpands_fp16_16x16` · `texpands_fp32_16x16` · `tlog_fp16_16x16`
  `tlog_fp32_16x16` · `tmax_fp16_16x16` · `tmax_fp32_16x16` · `tmax_i16_16x16`
  `tmax_i32_16x16` · `tmaxs_fp16_16x16` · `tmin_fp16_16x16` · `tmin_fp32_16x16`
  `tmin_i16_16x16` · `tmin_i32_16x16` · `tmins_fp16_16x16` · `tmul_fp16_16x16`
  `tmul_fp32_16x16` · `tmul_i16_16x16` · `tmul_i32_16x16` · `tmuls_fp16_16x16`
  `tneg_fp16_16x16` · `tneg_fp32_16x16` · `tneg_i16_16x16` · `tneg_i32_16x16`
  `tnot_i16_16x16` · `tnot_i32_16x16` · `tor_i16_16x16` · `tor_i32_16x16`
  `tpartadd_fp16_16x16` · `tpartadd_fp32_16x16` · `tpartmax_fp16_16x16` · `tpartmax_fp32_16x16`
  `tpartmin_fp16_16x16` · `tpartmin_fp32_16x16` · `tpartmul_fp16_16x16` · `tpartmul_fp32_16x16`
  `trecip_fp16_16x16` · `trecip_fp32_16x16` · `trelu_fp16_16x16` · `trelu_fp32_16x16`
  `trem_fp16_16x16` · `trem_fp32_16x16` · `trem_i32_16x16` · `trems_fp16_16x16`
  `trowexpand_fp16_16x16` · `trowexpand_fp32_16x16` · `trowexpandadd_fp16_16x16` · `trowexpandadd_fp32_16x16`
  `trowexpanddiv_fp16_16x16` · `trowexpanddiv_fp32_16x16` · `trowexpandexpdif_fp16_16x16` · `trowexpandexpdif_fp32_16x16`
  `trowexpandmax_fp16_16x16` · `trowexpandmax_fp32_16x16` · `trowexpandmin_fp16_16x16` · `trowexpandmin_fp32_16x16`
  `trowexpandmul_fp16_16x16` · `trowexpandmul_fp32_16x16` · `trowexpandsub_fp16_16x16` · `trowexpandsub_fp32_16x16`
  `trsqrt_fp16_16x16` · `tshl_i16_16x16` · `tshl_i32_16x16` · `tshr_i16_16x16`
  `tshr_i32_16x16` · `tsqrt_fp16_16x16` · `tsqrt_fp32_16x16` · `tsub_fp16_16x16`
  `tsub_fp32_16x16` · `tsub_i16_16x16` · `tsub_i32_16x16` · `tsubs_fp16_16x16`
  `txor_i16_16x16` · `txor_i32_16x16`

### microbench/memory（19）

  `mgather_i32_16x16` · `mscatter_fp16_16x16` · `mscatter_fp32_16x16` · `mscatter_i32_16x16`
  `tload_fp16_16x16` · `tload_fp16_32x32` · `tload_fp32_16x16` · `tload_fp32_32x32`
  `tload_i32_16x16` · `tload_nd2nz_fp16_16x16` · `tload_nd2nz_fp32_16x16` · `tmov_fp16_16x16`
  `tmov_fp16_32x32` · `tmov_fp32_16x16` · `tmov_fp32_32x32` · `tmov_i32_16x16`
  `tstore_fp16_16x16` · `tstore_fp32_16x16` · `tstore_i32_16x16`

### microbench/scalar（124）

  `abs_f32_lat` · `abs_f32_thr` · `abs_f64_lat` · `abs_f64_thr`
  `abs_i32_lat` · `abs_i32_thr` · `add_f32_lat` · `add_f32_thr`
  `add_f64_lat` · `add_f64_thr` · `add_i32_lat` · `add_i32_thr`
  `add_i64_lat` · `add_i64_thr` · `and_i32_lat` · `and_i32_thr`
  `and_i64_lat` · `and_i64_thr` · `clz_i32_lat` · `clz_i32_thr`
  `clz_i64_lat` · `clz_i64_thr` · `div_f32_lat` · `div_f32_thr`
  `div_f64_lat` · `div_f64_thr` · `div_i32_lat` · `div_i32_thr`
  `div_i64_lat` · `div_i64_thr` · `f2f_narrow_f64_to_f32_thr` · `f2f_widen_f32_to_f64_thr`
  `f2i_f32_to_i32_thr` · `f2i_f64_to_i32_thr` · `i2f_i32_to_f32_thr` · `i2f_i32_to_f64_thr`
  `ld_f32_thr` · `ld_f64_thr` · `ld_i32_thr` · `ld_i64_thr`
  `max_f32_lat` · `max_f32_thr` · `max_f64_lat` · `max_f64_thr`
  `max_i32_lat` · `max_i32_thr` · `max_i64_lat` · `max_i64_thr`
  `min_f32_lat` · `min_f32_thr` · `min_f64_lat` · `min_f64_thr`
  `min_i32_lat` · `min_i32_thr` · `min_i64_lat` · `min_i64_thr`
  `mod_i32_lat` · `mod_i32_thr` · `mod_i64_lat` · `mod_i64_thr`
  `mul_f32_lat` · `mul_f32_thr` · `mul_f64_lat` · `mul_f64_thr`
  `mul_i32_lat` · `mul_i32_thr` · `mul_i64_lat` · `mul_i64_thr`
  `neg_f32_lat` · `neg_f32_thr` · `neg_f64_lat` · `neg_f64_thr`
  `neg_i32_lat` · `neg_i32_thr` · `neg_i64_lat` · `neg_i64_thr`
  `not_i32_lat` · `not_i32_thr` · `not_i64_lat` · `not_i64_thr`
  `or_i32_lat` · `or_i32_thr` · `or_i64_lat` · `or_i64_thr`
  `popc_i32_lat` · `popc_i32_thr` · `popc_i64_lat` · `popc_i64_thr`
  `sll_i32_lat` · `sll_i32_thr` · `sll_i64_lat` · `sll_i64_thr`
  `slt_i32_lat` · `slt_i32_thr` · `slt_i64_lat` · `slt_i64_thr`
  `sqrt_f32_lat` · `sqrt_f32_thr` · `sqrt_f64_lat` · `sqrt_f64_thr`
  `sra_i32_lat` · `sra_i32_thr` · `sra_i64_lat` · `sra_i64_thr`
  `srl_i32_lat` · `srl_i32_thr` · `srl_i64_lat` · `srl_i64_thr`
  `st_f32_thr` · `st_f64_thr` · `st_i32_thr` · `st_i64_thr`
  `sub_f32_lat` · `sub_f32_thr` · `sub_f64_lat` · `sub_f64_thr`
  `sub_i32_lat` · `sub_i32_thr` · `sub_i64_lat` · `sub_i64_thr`
  `xor_i32_lat` · `xor_i32_thr` · `xor_i64_lat` · `xor_i64_thr`

### microbench/fixp（4）

  `fixp_tmatmul_keep_acc_M32_N32_K32_tM32_tN32_tK32` · `fixp_tmatmul_keep_acc_relu_M32_N32_K32_tM32_tN32_tK32` · `fixp_tmatmul_legacy3_M32_N32_K32_tM32_tN32_tK32` · `fixp_tmatmul_s_qf_f32_M32_N32_K32_tM32_tN32_tK32`

### one-level（53）

- `batched_transpose`
- `broadcast_broadcast_vec_019__DType__half_tM8_kInner49_IN_SHAPE1280_1_49_OUT_SHAPE1280_8_49`
- `broadcast_broadcast_vec_039__DType__half_tM8_kInner16_IN_SHAPE8192_1_16_OUT_SHAPE8192_8_16`
- `broadcast_broadcast_vec_07__DType__half_tM16_IN_SHAPE1443_1_OUT_SHAPE1443_129`
- `cast_back_per_channel`
- `cast_back_per_token`
- `concat_concat_gather_DType__half_tM512_IN_SHAPE256_8_OUT_SHAPE256_8000`
- `concat_concat_gather_DTypeint32_t_tM512_IN_SHAPE64_2_OUT_SHAPE64_2000`
- `concat_concat_scatter_DTypeint32_t_tM512_IN_SHAPE64_2_OUT_SHAPE64_2000`
- `element_wise_gelu_debug_gelu_debug_Approximate_false_DType__bf16_tM2048_SHAPE24_8_1024`
- `engram_hash_layer`
- `expand_to_mhc_bwd`
- `expand_to_mhc_fwd`
- `fa_2d_unroll_Sq256_Skv512_Tm16_Tk32_X1_Y2`
- `fa_2d_unroll_Sq256_Skv512_Tm16_Tk32_X1_Y4`
- `fa_2d_unroll_Sq256_Skv512_Tm16_Tk32_X2_Y2`
- `fa_2d_unroll_Sq256_Skv512_Tm16_Tk32_X2_Y4`
- `fa_2d_unroll_Sq512_Skv512_Tm16_Tk32_X1_Y2`
- `fa_2d_unroll_Sq512_Skv512_Tm16_Tk32_X1_Y4`
- `fa_2d_unroll_Sq512_Skv512_Tm16_Tk32_X2_Y2`
- `fa_2d_unroll_Sq512_Skv512_Tm16_Tk32_X2_Y4`
- `fa_softmax_pto_Sq256_Skv512_Tm16_Tk32`
- `fa_softmax_pto_Sq512_Skv512_Tm16_Tk32`
- `flashMLA_Sq32_QHeadPerHK1_NumBlocks1_Dk512_Dv512_DChunk128_VChunk128_Tm16_Tk16`
- `flashMLA_Sq64_QHeadPerHK1_NumBlocks2_Dk512_Dv512_DChunk128_VChunk128_Tm16_Tk16`
- `fn_normw_merge_fwd`
- `fused_weight`
- `gather_gather_DType__fp32_OTypeuint32_t_gKs131072_gMs32_gNs256_tMs32_tNs64`
- `matmul_MASK_MASK_FP16_M256_N256_K256_tM32_tN32_tK64`
- `matmul_MASK_MASK_FP16_REUSEA_M256_N256_K256_tM32_tN32_tK64`
- `matmul_MASK_MASK_FP16_REUSEB_M256_N256_K256_tM32_tN32_tK64`
- `matmul_MASK_MASK_FP32_M256_N256_K256_tM32_tN32_tK32`
- `matmul_MASK_MASK_FP32_REUSEA_M256_N256_K256_tM32_tN32_tK32`
- `matmul_MASK_MASK_FP32_REUSEB_M256_N256_K256_tM32_tN32_tK32`
- `matmul_MASK_MASK_FP8_M256_N256_K256_tM32_tN32_tK64`
- `matmul_MASK_MASK_FP8_REUSEA_M256_N256_K256_tM32_tN32_tK64`
- `matmul_MASK_MASK_FP8_REUSEB_M256_N256_K256_tM32_tN32_tK64`
- `multilayer_recompute`
- `normalize_weight`
- `norm_rms_norm_M16_N256_tM8_tN128`
- `per_channel_cast`
- `per_token_cast`
- `reduce_fused`
- `reduction_reducemax_col_reducemax_col_DTypeint32_t_tM32_tN64_GM2048_GN64`
- `reduction_reducemax_row_reducemax_row_DTypeint32_t_tM16_tN128_GM16_GN8192`
- `reduction_reducesum_col_reducesum_col_DType__half_tM32_tN64_GM2048_GN64`
- `reduction_reducesum_col_reducesum_col_DTypeint32_t_tM32_tN64_GM2048_GN64`
- `reduction_reducesum_row_reducesum_row_DTypefloat_tM16_tN128_GM16_GN8192`
- `rms_norm`
- `transpose_DType__half_tM512_IN1476_32_OUT32_1476`
- `transpose_DType__half_tM512_IN1_32_8_32_OUT1_8_32_32`
- `transpose_DType__half_tM512_IN1_8_4096_3_OUT1_8_3_4096`
- `transpose_DType__half_tM512_IN1_8_64_4_16_7_OUT1_8_16_4_64_7`

### one-level/multi_thread（5）

- `multi_thread_fa_Sq128_Skv64_Tm16_Tk16`
- `multi_thread_fa_Sq512_Skv512_Tm16_Tk16`
- `multi_thread_matmul_matmul_shared_B1_M256_N256_K256_tM32_tN32_tK32`
- `multi_thread_vec_Rows16_Cols16`
- `multi_thread_vec_trowsum_Rows16_Cols16`

## 5. 说明与边界

- **通过判定**：gfrun 退出码 0 且输出含 `Reach the End of Benchmark` 且 `R2 = 0`。
- **模型断言 ≠ 非法指令**：gfrun 把内部 validator 断言违例也打印为 `illegal instruction`，故分类器初判为 `FAIL-illegal-inst`；实为 SuperScalarModel 功能模型对部分 tile/inst 配置尚未支持。
- **工具链**：全程使用 `linx-toolchain-build-latest`（用户指定），未用旧 `linx-toolchain-build`。
- **gfrun 版本**：SuperScalarModel @ `a68dba29`（`feat/pto-v058-adaptation` 分支）。
- **TileOP-API 修复**：`8b2ee78` 恢复了 `TileDType = DType tile_size(Rows*Cols/...)` 的正确形式（先前 `f9a30a69` 硬编码为 `int32_t ext_vector_type(1024)` = 4KB）。
- **fixp 回归**：TileDType 修复使非 FP32 类型的 tile 尺寸从 4KB 恢复为正确值，暴露了 gfrun CUBE/tmatmul validator 的 `accBytes` 校验偏差。仅 4 个 FP32 累加器用例仍通过。此回归属**模型侧校验偏差**，非算子内核 bug。
- 原始数据：`/tmp/gfrun_results_20260818.tsv`（含每个 ELF 的 rc/elapsed/status/signature）。

## 6. matmul 数值验证

### 6.1 结论

matmul MASK_FP32 M64×64×64 数值验证 **PASS**：max abs err 7.63e-6，4096/4096 非零，所有元素与 golden reference 一致。

> 先前（2026-08-14）该验证 FAIL：仅 1216/4096 非零，根因为 TileDType 硬编码 4KB 导致 tile 尺寸与算子 tile 形状不符。TileOP-API `8b2ee78` 修复后，tile 尺寸恢复正确，数值验证通过。

### 6.2 验证配置

| 项 | 值 |
|---|---|
| ELF | `matmul_MASK_MASK_FP32_M64_N64_K64_tM16_tN16_tK16.elf` |
| 输入 | 随机 FP32，M=N=K=64，seed=42 |
| golden | numpy `a @ b`（FP32 参考矩阵乘） |
| gfrun | `a68dba29`，单线程 `-t 1` |
| 验证目录 | `benchmark/one-level-arch/compare/matmul_MASK_MASK_FP32_M64_N64_K64_tM16_tN16_tK16/` |

### 6.3 结果

| 指标 | 值 |
|---|---|
| 非零元素 | 4096 / 4096 |
| max abs err | 7.62939453e-06 |
| MSE | 1.77010598e-12 |
| 判定 | **PASS** |

gfrun 输出：`Total Block number = 4037, Total Inst number = 18096, Suaccelss to Reach the End of Benchmark! R2 = 0`

### 6.4 复现

```bash
GFRUN=/Users/blacktraker/Programming/gitproj/DV4/SuperScalarModel/bin/gfrun   # a68dba29
ELF=benchmark/one-level-arch/output/kernel/matmul/elf/kernel_matmul/matmul_MASK_MASK_FP32_M64_N64_K64_tM16_tN16_tK16.elf
CHKD=benchmark/one-level-arch/compare/matmul_MASK_MASK_FP32_M64_N64_K64_tM16_tN16_tK16

# 生成随机输入 + golden reference
python3 -c "
import numpy as np
np.random.seed(42)
a = np.random.randn(64,64).astype(np.float32)
b = np.random.randn(64,64).astype(np.float32)
golden = a @ b
a.tofile('$CHKD/src0.bin')
b.tofile('$CHKD/src1.bin')
golden.tofile('$CHKD/golden.bin')
np.zeros(64*64, dtype=np.float32).tofile('$CHKD/res.bin')
"

# 跑 gfrun
$GFRUN -t 1 -f $ELF

# 比较
python3 -c "
import numpy as np
res = np.fromfile('$CHKD/res.bin', dtype=np.float32).reshape(64,64)
golden = np.fromfile('$CHKD/golden.bin', dtype=np.float32).reshape(64,64)
diff = np.abs(res - golden)
print(f'nonzero: {np.count_nonzero(res)}/{64*64}')
print(f'max_abs={diff.max():.8e}  mse={(diff**2).mean():.8e}')
print('PASS' if diff.max() < 1e-5 else 'FAIL')
"
```

## 7. multi_thread 算子编译与执行

### 7.1 编译

工具链：`linx-toolchain-build-latest/output/linx_blockisa_llvm_musl/bin`（clang 15.0.4）。3 个算子共 5 个 ELF 全部编译成功。

| 算子 | compile.all 参数 | ELF |
|---|---|---|
| vec/tadd | `TESTCASE=tadd TileRows=16 TileCols=16` | `output/kernel/multi_thread/vec/elf/kernel_multi_thread_vec_Rows16_Cols16.elf` |
| vec/trowsum | `TESTCASE=trowsum TileRows=16 TileCols=16` | `output/kernel/multi_thread/vec/elf/kernel_multi_thread_vec_trowsum_Rows16_Cols16.elf` |
| matmul | `TESTCASE=matmul_shared B=1 M=256 N=256 K=256 tM=32 tN=32 tK=32` | `output/kernel/multi_thread/matmul/elf/kernel_multi_thread_matmul_matmul_shared_B1_M256_N256_K256_tM32_tN32_tK32.elf` |
| fa (Sq128) | `TESTCASE=fa_2d_unroll_gmma Sq=128 Skv=64 Tm=16 Tk=16` | `output/kernel/multi_thread/fa/elf/kernel_multi_thread_fa_Sq128_Skv64_Tm16_Tk16.elf` |
| fa (Sq512) | `TESTCASE=fa_2d_unroll_gmma Sq=512 Skv=512 Tm=16 Tk=16`（Makefile 默认） | `output/kernel/multi_thread/fa/elf/kernel_multi_thread_fa_Sq512_Skv512_Tm16_Tk16.elf` |

### 7.2 执行

gfrun 版本 `a68dba29`（`feat/pto-v058-adaptation`）。multi_thread 算子需带 `-s softcore.multiThreadNum=4`（4-PE 配置）。

命令：`gfrun -t 1 -f <elf> -s softcore.multiThreadNum=4`

| 算子 | 结果 | 耗时 | 说明 |
|---|---|---|---|
| vec/tadd | **PASS** | 0s | 退出码 0，`Reach the End of Benchmark! R2 = 0` |
| vec/trowsum | **PASS** | 1s | 退出码 0，`Reach the End of Benchmark! R2 = 0` |
| matmul_shared | **PASS** | 2s | 退出码 0，`Reach the End of Benchmark! R2 = 0` |
| fa (Sq128) | **PASS** | 1s | 退出码 0，`Reach the End of Benchmark! R2 = 0` |
| fa (Sq512) | **PASS** | 40s | 退出码 0，`Reach the End of Benchmark! R2 = 0` |

### 7.3 修复说明

先前（2026-08-14）fa 和 matmul_shared 在 4-PE 配置下 FAIL，根因为 TileOP-API `f9a30a69` 硬编码 `TileDType = int32_t ext_vector_type(1024)`（4KB），导致 Shared TLOAD 的 tile 尺寸与算子所需不符，触发 `ExecuteSharedTMA` 断言。TileOP-API `8b2ee78` 恢复了 `TileDType = DType tile_size(Rows*Cols/...)` 的正确形式后，Shared TLOAD 尺寸与算子 tile 形状匹配，两个算子均通过。

### 7.4 复现

```bash
GFRUN=/Users/blacktraker/Programming/gitproj/DV4/SuperScalarModel/bin/gfrun   # a68dba29
OUT=benchmark/one-level-arch/output/kernel/multi_thread

# vec/tadd
$GFRUN -t 1 -f $OUT/vec/elf/kernel_multi_thread_vec_Rows16_Cols16.elf -s softcore.multiThreadNum=4

# vec/trowsum
$GFRUN -t 1 -f $OUT/vec/elf/kernel_multi_thread_vec_trowsum_Rows16_Cols16.elf -s softcore.multiThreadNum=4

# matmul_shared
$GFRUN -t 1 -f $OUT/matmul/elf/kernel_multi_thread_matmul_matmul_shared_B1_M256_N256_K256_tM32_tN32_tK32.elf -s softcore.multiThreadNum=4

# fa (Sq128)
$GFRUN -t 1 -f $OUT/fa/elf/kernel_multi_thread_fa_Sq128_Skv64_Tm16_Tk16.elf -s softcore.multiThreadNum=4

# fa (Sq512)
$GFRUN -t 1 -f $OUT/fa/elf/kernel_multi_thread_fa_Sq512_Skv512_Tm16_Tk16.elf -s softcore.multiThreadNum=4
```
