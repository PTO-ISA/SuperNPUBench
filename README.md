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

> **当前验证基线**：2026-08-27（427 个已编译 ELF 全量 gfrun 复测；编译器按 AGENTS.md 用主
> linx-toolchain-build worktree（llvm `adcb8794` + TileOP-API `f94bc12`，CUBE cell-layout 强制）、
> gfrun 用 SuperScalarModel `codex/pr-0.58.4-shared-model` `d8903938`（含 reduce/expand dtype 门控
> 对齐修复 + 未提交 SoftCore/SysCall 改动）；总 PASS 349，通过率 81.7%——环境较 08-23 大改：编译器
> 分支 blessed-latest→main、gfrun 分支 exp→codex、fa kernel 有 WIP 改动，差异非单一变量）

# gfrun 执行结果汇总 — 2026-08-27

## 验证环境

| 组件 | 分支/版本 | Commit |
|---|---|---|
| gfrun / SuperScalarModel | `codex/pr-0.58.4-shared-model` | `d8903938`（08-27 21:47 构建，含未提交 `SoftCore.cpp`/`SysCall.cpp/.h` 改动） |
| llvm-project | detached（dev-llvm15_56 谱系） | `adcb87948` |
| Linx-TileOP-API | `linx` | `f94bc12` |

编译器严格按 **AGENTS.md** 指定用主 `linx-toolchain-build` worktree（非 08-23 的 blessed-latest）：`COMPILER_DIR=…/linx-toolchain-build/output/linx_blockisa_llvm_musl/bin`，clang 15.0.4，target `linx64v5-unknown-linux-musl`。本版 TileOP-API `f94bc12` 新增 **CUBE cell-layout 强制**（`IsCubeLayout` 静态断言：TMATMUL 的 A/D 必须 `CUBE_M16/M32`、B 必须 `CUBE_N8`），导致仍用 `TileLeft/TileRight/TileAcc` 的 kernel 编译失败（matmul/deepseek 回归，见下"编译覆盖"）。gfrun 用 `codex/pr-0.58.4-shared-model` `d8903938`，含本轮 reduce/expand dtype 门控对齐修复（`IsReduceAndExpandTeplDataType`）及未提交的 `SoftCore.cpp`/`SysCall.cpp/.h` 改动。执行：`gfrun -t 1 -f <elf>`，multi_thread 加 `-s softcore.multiThreadNum=4`，单 ELF 90s 超时。PASS = 退出码 0 + `Reach the End of Benchmark` + `R2 = 0`。

## 本次新增特性（工具链 / 模型，PTO 0.58.4 全栈对齐）

三个组件 08-23→08-27 同步演进到 PTO 0.58.4（TileOP-API `a795b97→f94bc12` 19 commits、llvm `611105f→adcb8794` 11 commits、gfrun `a5dca25a→d8903938` ~40 commits）。按类归档，commit 均为各仓库 HEAD 范围内：

**1. SizeCode（代码沿用旧名 TSize）容量扩展 —— 128/256 KiB Local tile**
- llvm `d9dcf68e8` "Allow Local B.IOT SizeCode 11 and 12"：B.IOT 目的 SizeCode（4-bit 容量字段，tablegen `B_IOT_TSize_Op`）扩到 **1..12**，新增 11=128 KiB、12=256 KiB（每 PE）；0=仅源，13..15 reserved。
- TileOP `1e2f130` "Allow 128 KiB and 256 KiB Local tiles"：TMOV/MGATHER/MSCATTER/TLOAD/TSTORE/TMATMUL/GMOV 等 Local tile 容量上限提到 256 KiB，新增 `__tilesize_128KB/256KB`。Local tile 不再卡 ~64 KiB，大 tile 不必再依赖 32 KiB shared 工具链。

**2. CUBE cell layout 规范化（本轮最大 layout 改变）**
三侧对齐 canonical CUBE transport：TileOP `dacedc2`"Use canonical CUBE transport selectors" + `bd1ecca`"zero CUBE compute padding" + llvm `76044f436`"Add canonical CUBE layout transport selectors" + gfrun `6858e274`"add 0.58.4 cube cell layout support" / `1c55f5c5` / `5d370a1f`。强制点 `IsCubeLayout` 静态断言——TMATMUL 的 A/D 必须 `CUBE_M16/M32`、B 必须 `CUBE_N8`（template_asm.hpp:2573/2579/2589）。→ matmul/deepseek 编译回归（待 `TileLeft/Right/Acc`→`CubeTileM16/M32/N8` 迁移）；同时 cube 运行 2→11 全过（目的容量断言消除 + `9c5840b0` 补 CUBE TLOAD 物理列）。

**3. 新指令 / tile op**
- **B.SUBVIEW / B.ASSEMBLE range modifier**（三侧）：TileOP `cdfbadb`/`495f01a`/`4053fb5` + llvm `3f3938427` + gfrun `f7337b9d`"implement PTO 0.58.4 range modifiers" —— B.IOT 子视图/拼装范围修饰。
- **TGEMV 完整 MC 展开**：llvm `e0762147b`"complete TGEMV MC expansion" + `3434ea3ab` reject CScale for TGEMV。
- TEPL selector 打包（TileOP `83b9903`）、TMATMUL M 由输入 A 推导（`87608f0`）、TCVT 维度先于源绑定（`f94bc12`，HEAD）、CScale for matrix ACC（`cf4c053`）+ llvm B.FPATR CScale 编码（`82e69a818`）、HiF4X2 MX contracts（llvm `e8242e962`）。

**4. gfrun 模型侧新功能（直接驱动 PASS 变化）**
- **Cooperative PTO MX matmul**（`0ab6593e`）+ cooperative MX carrier（`b4b51993`）→ multi_thread/matmul lowp 4→9 全过。
- **BF16 / HiF8 cube profiles**（`e3288a1a`）→ cube 支持 BF16/HiF8。
- **Reduce/expand dtype gates**（`d8903938` + `9d683f91`"restore reduce and expand legality gates"）→ FP8_VECBF16 FAIL→PASS（TROWEXPANDSUB BF16 解锁）；同一改动新触发 `dataType==block->dataType` 断言 → fa/flashMLA/reduction −16 PASS（回归）。
- **Hosted SMT4 runtime**（`accc09b9`，`multiThreadNum=4` 多线程路径）；**TCI legality gate / physical-shape/padding**（`dab5a63a`/`5ab7e3fb`/`d83bab52`）→ tci 编译修复（编出 `tci_1x64`）；**PEMode/SizeCode 模型采纳**（PR #333 `9bbe1448` feat/gfrun-pemode-sizecode-118）。
- 其他：CUBE FPATR post-processing（`1837e2d8`）、single-issuer shared matmul（`da8e8ac9`）、row reduction contract（`66b0ba4c`）、E4M3 overflow saturation（`7e270310`）。未提交：`SoftCore.cpp`/`SysCall.cpp/.h` dirty。

**5. PTO 0.58.4 契约对齐（横切）**
TileOP `943311c`（canonical RMode 语法）/`2e4d695`/`9745ebf`；llvm `8a4bb5bda`（B.IOT/RMode assembly 0.58.4）/`118d038ab`/`9ea74798f`；gfrun `9ca9368b`/`7ab161ed`/`47e2000a`。

> 提取方法：对三个仓库分别跑 `git -C <repo> log --oneline <上轮 commit>..<本轮 commit>`（`linx-toolchain-build/src/Linx-TileOP-API`、`…/src/llvm-project`、`SuperScalarModel`），按 ISA/SizeCode/layout/模型分类。

## 总体结果

| 范围 | ELF 数 | PASS | FAIL | TIMEOUT | 通过率 |
|---|---:|---:|---:|---:|---:|
| microbenchmark | 341 | 298 | 43 | 0 | 87.4% |
| one-level | 86 | 51 | 35 | 0 | 59.3% |
| **合计** | **427** | **349** | **78** | **0** | **81.7%** |

## 算子通过率

按算子族列出当前通过率（427 ELF 全量 gfrun，349 PASS / 78 FAIL / 0 TIMEOUT，通过率 81.7%）：

| 算子族 | 编译成功 | PASS | FAIL | 通过率 | 说明 |
|---|---:|---:|---:|---:|---|
| micro/scalar | 124 | 124 | 0 | 100% | 全过 |
| micro/vector | 129 | 129 | 0 | 100% | 全过（`tci`/`sinkhorn_fwd`/`topk` 编译已修） |
| micro/memory | 14 | 14 | 0 | 100% | `mgather` 16×16 不再失败 |
| micro/cube | 11 | 11 | 0 | 100% | CUBE 目的容量断言消除，+9 全过 |
| micro/fixp | 63 | 20 | 43 | 31.7% | fixp tmatmul 模型侧断言（quant/outputBytes/accumulator/scale…） |
| one-level/broadcast | 6 | 5 | 1 | 83.3% | `vec_07 half` COPY 扩展断言（新） |
| one-level/concat | 4 | 3 | 1 | 75.0% | `scatter half` 缺结束标记 |
| one-level/transpose | 4 | 4 | 0 | 100% | 全过 |
| one-level/gather | 1 | 1 | 0 | 100% | 全过 |
| one-level/element_wise | 1 | 1 | 0 | 100% | gelu 全过 |
| one-level/reduction | 5 | 3 | 2 | 60.0% | `dataType` 断言回归（未改 kernel） |
| one-level/control | 6 | 0 | 6 | 0% | INT8/16 dtype 元组未定义 |
| one-level/fa | 10 | 0 | 10 | 0% | `dataType` 断言回归（fa kernel WIP 亦有影响） |
| one-level/flashMLA | 2 | 0 | 2 | 0% | 同 `dataType` 断言（未改 kernel） |
| one-level/matmul | 3 | 3 | 0 | 100% | 仅 3/16 编译成功（CUBE layout），幸存全过 |
| one-level/sort | 1 | 0 | 1 | 0% | `topk` 编译已修，运行 `R2=1` |
| one-level/multi_thread/fa | 16 | 10 | 6 | 62.5% | `FP8_VECBF16` 过；HIF4/MXFP4/HIF8 仍挂 |
| one-level/multi_thread/matmul | 9 | 9 | 0 | 100% | 全过（含 5 lowp，cooperative 已建模） |
| one-level/multi_thread/vec | 2 | 2 | 0 | 100% | 全过 |
| one-level/deepseek | 16 | 10 | 6 | 62.5% | 5 CUBE 编译失败 + cast/normalize 运行失败 |

> 编译成功合计 427（micro 341 + one-level 86），编译失败 23 个未计入上表（见下「编译覆盖」）。`conv2d`/`norm`/`normalization` 未接入 `compile_all.sh`、无 ELF；`two-level-arch` 不支持当前 ISA 模式，未编译未跑。

## 编译覆盖

成功生成 ELF：427 个（microbenchmark 341 + one-level 86）。编译失败、未进入 gfrun，分两类：

**新增编译回归 —— TileOP-API `f94bc12` CUBE cell-layout 强制**（`IsCubeLayout` 静态断言要求 A/D=`CUBE_M16/M32`、B=`CUBE_N8`；仍用 `TileLeft/TileRight/TileAcc` 的 kernel 不再通过）：

| # | 用例 | 报错位置 | 根因 |
|---|---|---|---|
| 1–13 | matmul 13 个变体（除 MASK_MASK FP32/FP8/FP16 外全部） | template_asm.hpp `IsCubeLayout` | `CUBE destination D must use CUBE_M16/M32`、`Local matrix A/B must use CUBE_N8` 等 |
| 14 | deepseek/aux_fi | 同上 | CUBE layout（08-23 为运行时 FAIL，本轮前置为编译失败） |
| 15 | deepseek/get_fused_mapping | 同上 | 同上 |
| 16 | deepseek/group_count | 同上 | 同上 |
| 17 | deepseek/inplace_unique_group_indices | 同上 | 同上 |
| 18 | deepseek/mask_indices_by_tp | 同上 | 同上 |

**历史已知失败（与 08-23 一致）**：

| # | 用例 | 根因 |
|---|---|---|
| 19 | fa/sfa Sq=256 | TMATMUL `output shape must be A.Rows x B.Cols` |
| 20 | fa/sfa Sq=512 | 同上 |
| 21 | fa/fa_hif4 | `QuantType` 未声明 + fp4 tile 行/对齐断言 |
| 22 | deepseek/topk_gate | TCI `ValidRow==1` + clang exit 134 |
| 23 | deepseek/expand_to_fused | clang 前端 SIGABRT |

**本轮编译修复（08-23 FAIL→本轮编译成功）**：`sort/topk`（`no matching TLOAD` 已修，运行 `R2=1`）、`deepseek/sinkhorn_fwd`（B.IOT mask 已修，运行通过）、microbench `tci_i16/i32`（`ValidRow==1` 已修，现编 `tci_i{16,32}_1x64`）、microbench `mgather_mask`/`mscatter_mask`（不再失败/移出构建集）。

## 运行失败清单（78 FAIL，全部模型侧）

### one-level/fa — 10 FAIL（新断言，08-23 全 PASS）★★回归
`fa_2d_unroll`(8) + `fa_softmax_pto`(2) —— `source->tileInfo->dataType == block->dataType && …validRow==validRow…` 断言。

### one-level/flashMLA — 2 FAIL（同断言，08-23 全 PASS）★回归
两个 flashMLA 变体 —— 同上 `dataType==block->dataType` 断言。

### one-level/reduction — 2 FAIL（同断言，08-23 全 PASS）★回归
`reducemax_row_int32`、`reducesum_row_float` —— 同上断言。

> 上述 fa/flashMLA/reduction 共 14 个 PASS 损失同源于一个新触发的 `source->tileInfo->dataType == block->dataType` 断言；flashMLA/reduction 本轮未改 kernel，故为环境驱动（gfrun `d8903938` dtype 门控改动或 TileOP CUBE 代码生成），待定位根因。

### one-level/deepseek — 6 FAIL
- `cast_back_per_token`/`normalize_weight`/`rms_norm`/`sinkhorn_fwd` —— `IsCompatibleDataTile`（elemBytes/validCol/physicalCol）。
- `per_token_cast`/`swiglu_forward_and_per_token_cast` —— 同 `dataType==block->dataType` 断言。

### one-level/multi_thread/fa — 6 FAIL（fa kernel + gfrun 双侧）
- `HIF4_VECBF16`(2)、`MXFP4_VECBF16`(2) —— `fa_tcvt_packed_x2`（BF16→FP4 打包 TCVT）触发 `ValidateOperandContract` 形状断言（packed-x2 dst 列数=src/2，gfrun 不识别打包转换）。
- `HIF8_VECFP32`(2) —— `FloatPointUtils.cpp:1719` `.fs→.hifb` convert 未注册。

### one-level/control — 6 FAIL（不变）
`hashtable_lookup_simd_*` —— `dataType==INT8||UINT8||INT16…` 元组未定义。

### microbenchmark/fixp — 43 FAIL（+7 vs 08-23）
fixp tmatmul 系列模型侧断言：quant(13)、outputBytes!=0(10)、srcs.size()==2(6)、scale(3)、accumulator(3)、cooperative-PE-count(2，原 `shared`/`s8_shared` 超时→现快速 FAIL)、source/relu(4)、rowMax(1)、hasFixpAttr(1)。

### 其余零散
- one-level/broadcast 1：`broadcast_vec_07 half` —— `COPY expansion` 断言（新）。
- one-level/concat 1：`concat_scatter half` —— 缺结束标记（不变）。
- one-level/sort 1：`topk` —— 编译已修，运行 `R2=1`（结果错误）。

## 本次更新要点

- **环境大改（非单一变量）**：编译器由 08-23 的 blessed-latest（`611105f2b`/`a795b973020d`）切到 AGENTS.md 主 worktree（`adcb8794`/`f94bc12`）；gfrun 由 exp `a5dca25a` 切到 codex `d8903938`（含 reduce/expand dtype 门控修复 + 未提交 SoftCore/SysCall）；fa kernel 有 WIP 改动（`fa_2d_unroll_gmma.hpp` state-tile/TCVT、`fa_2d_unroll.hpp` CubeTile 迁移、`compile.all` 新配置）。下述差异为以上变化的合效应。
- **编译回归（CUBE cell-layout 强制）**：TileOP `f94bc12` 强制 TMATMUL 操作数用 `CUBE_M16/M32/N8`，未迁移的 matmul(13)/deepseek(5) 编译失败；matmul 16→3、deepseek 20→16 ELF。需把 `TileLeft/TileRight/TileAcc` 迁到 `CubeTileM16/M32/N8`（`fa_2d_unroll.hpp` 已示范）。
- **运行回归（dataType 断言）**：新触发的 `source->tileInfo->dataType == block->dataType` 使 fa(10)/flashMLA(2)/reduction(2)/deepseek(2) 共 −16 PASS，flashMLA/reduction 未改 kernel → 环境驱动，待定位。
- **正向变化**：cube 2→11 全过（+9）、multi_thread/matmul lowp 4→9（+5，cooperative 低精度 profile 已建模）、multi_thread/fa `FP8_VECBF16` 由 FAIL→PASS（gfrun reduce/expand dtype 修复 + fa state-tile 物理行修复）、microbench `tci`/`sinkhorn_fwd`/`topk` 编译修复。
- **净 366→349 PASS（−17）**：回归 −38（fa-10、matmul-6、fixp-5、flashMLA-2、reduction-2、deepseek-4、vector-5、memory-3、broadcast-1）大于改善 +21（cube+9、mt/matmul+5、mt/fa+7）。0 TIMEOUT（08-23 的 2 个 fixp shared 超时现快速 FAIL 于 cooperative-PE-count 断言）。

## 与 2026-08-23 基线的差异

> ⚠ 本轮与 08-23 非单一变量对比：编译器（blessed-latest→main worktree）、gfrun（exp→codex 分支 + WIP）、TileOP（`a795b97`→`f94bc12` CUBE 强制）、fa kernel（WIP）均变。差异为合效应，不归因单一组件。

| 类别 | 08-23 (ELF/P/F/T) | 08-27 (ELF/P/F/T) | 变化 |
|---|---|---|---|
| micro/scalar | 124/124/0/0 | 124/124/0/0 | — |
| micro/cube | 6/2/4/0 | 11/11/0/0 | PASS +9 / FAIL −4（cube 目的容量断言消除 + 新增 5 全过） |
| micro/fixp | 63/25/36/2 | 63/20/43/0 | PASS −5 / FAIL +7 / TIMEOUT −2（shared 超时→cooperative 快速 FAIL + 新断言） |
| micro/memory | 19/17/2/0 | 14/14/0/0 | ELF −5 / FAIL −2（mgather 16x16 不再失败/移出） |
| micro/vector | 136/134/2/0 | 129/129/0/0 | ELF −7 / FAIL −2（thistogram 不再失败） |
| one-level/broadcast | 6/6/0/0 | 6/5/1/0 | PASS −1（vec_07 half COPY 扩展断言，新） |
| one-level/concat | 4/3/1/0 | 4/3/1/0 | — |
| one-level/control | 6/0/6/0 | 6/0/6/0 | — |
| one-level/deepseek | 20/14/6/0 | 16/10/6/0 | ELF −4 / PASS −4（5 个 CUBE 编译失败；runtime FAIL 集合变化） |
| one-level/element_wise | 1/1/0/0 | 1/1/0/0 | — |
| one-level/fa | 10/10/0/0 | 10/0/10/0 | PASS −10（dataType 断言回归，fa kernel WIP 亦有影响） |
| one-level/flashMLA | 2/2/0/0 | 2/0/2/0 | PASS −2（同 dataType 断言，未改 kernel） |
| one-level/gather | 1/1/0/0 | 1/1/0/0 | — |
| one-level/matmul | 16/9/7/0 | 3/3/0/0 | ELF −13 / PASS −6 / FAIL −7（CUBE 编译回归；幸存 3 个全过） |
| one-level/multi_thread/fa | 3/3/0/0 | 16/10/6/0 | ELF +13 / PASS +7 / FAIL +6（新配置；FP8_VECBF16 过，HIF4/MXFP4/HIF8 仍挂） |
| one-level/multi_thread/matmul | 9/4/5/0 | 9/9/0/0 | PASS +5 / FAIL −5（cooperative 低精度 profile 已建模） |
| one-level/multi_thread/vec | 2/2/0/0 | 2/2/0/0 | — |
| one-level/reduction | 5/5/0/0 | 5/3/2/0 | PASS −2（dataType 断言，未改 kernel） |
| one-level/sort | 0/0/0/0 | 1/0/1/0 | 编译修复（topk），运行 R2=1 |
| one-level/transpose | 4/4/0/0 | 4/4/0/0 | — |

> 改善 +21（cube+9、mt/matmul+5、mt/fa+7）vs 回归 −38（fa-10、matmul-6、fixp-5、flashMLA-2、reduction-2、deepseek-4、vector-5、memory-3、broadcast-1），净 −17 PASS。两大回归——CUBE cell-layout 编译失败（matmul/deepseek）与 `dataType==block->dataType` 运行断言（fa/flashMLA/reduction）——为本轮重点排查项。

---

# 历史验证记录

> 早期每日基线仅保留环境版本与总量，供复现与趋势对比；完整分类/失败明细已归档。

| 日期 | gfrun (SuperScalarModel) | llvm / TileOP-API | 工具链 | ELF | PASS | FAIL | T/O | 通过率 | 关键变化 |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| 08-23 | exp `a5dca25a` | `611105f2b` / `a795b973020d` | blessed-latest（ADR 0069） | 437 | 366 | 69 | 2 | 83.8% | blessed 编译器+exp 模型正确配对；+27 PASS（fa 全过、mt/matmul lowp 部分）；13 编译失败 |
| 08-21 | main `7b691d4d` | `a84c4d10a` / `ffa257738f` | toolchain-build（32KB shared） | 437 | 339 | 98 | 0 | 77.6% | 老 compiler+main 模型；multi_thread 大 tile 首编（bf16/fp16/lowp 运行 FAIL） |
| 08-20 | exp `5a64c34d` | `b945a5d0` / `c02dae65` | blessed-latest | 433 | 344 | 89 | 0 | 79.4% | 零步幅 raw tile spill 修复 4 例；fa `sfa`×2 编译回归（TMATMUL 形状契约） |
| 08-19 | `01f9ec10` | `86959776b` / `8b2ee78` | — | 434 | 341 | 92 | 1 | 78.6% | 4 例 broadcast/GELU FAIL→PASS；mt/matmul 1 PASS→FAIL |
| 08-18 | `a68dba29` | — / `8b2ee78`（TileDType 修复） | — | 429 | 321 | 108 | 2 | 74.8% | 首次全量基线；fa/flashMLA/reduction 由 FAIL 恢复；fixp 27→4（TileDType 暴露契约偏差） |

**跨版本要点**：

- **ADR 0069 编码配对**（08-21 ↔ 08-23）：版本匹配则高 PASS，错位则骤降。08-21 老 compiler+main 模型（均无 ADR 0069）= 匹配 → 339P；08-23 blessed+exp（均有 ADR 0069）= 匹配 → 366P；而 blessed compiler+main 模型（编译器领先、模型落后）= 错位 → 仅 124P（262 个 `reserved/deleted TEPL selector`：store 的 SizeCode=0 被旧模型误读为 0B 目的）。
- **08-27 切回 AGENTS.md 主 worktree**（非 blessed-latest）：gfrun codex `d8903938`、TileOP `f94bc12`（CUBE cell-layout 强制）。与 08-23 非单一变量对比，详见上文「与 2026-08-23 基线的差异」。
- **持续模型侧限制**（跨基线不变）：fixp tmatmul 系列、control INT8/16 dtype 元组、MGATHER/THISTOGRAM reserved TEPL selector；cube 目的容量断言在 08-27 已消除。
