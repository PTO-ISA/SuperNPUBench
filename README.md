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
| vector (TEPL) | elementwise / tile-scalar / reduce / expand (toolchain-exposed subset) | 126 |
| memory (TLSU) | TLOAD / TSTORE / TMOV / MGATHER / MSCATTER (+mask, layout) | 25 |
| scalar (GPR) | int ALU / load-store / float / conversion × throughput+latency | 124 |
| **total** | | **284** |

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

# release_ver0812 gfrun 验证报告

本节记录临时稳定分支 `release_ver0812` 已验证的算子、验证环境及当前已知问题。

## 验证仓库版本

| 组件 | 仓库 | 分支 | Commit |
|---|---|---|---|
| gfrun 模型 | https://github.com/LinxISA/SuperScalarModel | `feat/pto-v058-adaptation` | `319294ffdde0304e12c530746b7f63b5ce4083d9` |
| llvm-project | https://github.com/LinxISA/llvm-project | `temp/shared-tload-integration-20260811` | `eb64de8afcbda043aec7e56dae346905dc982039` |
| Linx-TileOP-API | https://github.com/LinxISA/Linx-TileOP-API | `temp/shared-tload-integration-20260811` | `72f8255ca610eae1542dfb633709ce2b18b49955` |

工具链：`llvm-project temp/shared-tload-integration-20260811 @ eb64de8af` + `Linx-TileOP-API temp/shared-tload-integration-20260811 @ 72f8255c`

## 验证方式

- 串行执行（非并行，避免并行负载导致的假阳性）
- 每配置 120s 超时上限
- 平台：macOS arm64，clang

## 验证统计

| 状态 | 数量 | 占比 |
|---|---|---|
| **通过** | **20** | **69%** |
| **失败** | **9** | **31%** |
| 超时 | 0 | 0% |

## 全量验证结果

| # | 类别 | 配置 | dtype | 结果 | blocks/insts | 断言位置 |
|---|---|---|---|---|---|---|
| 1 | broadcast | vec_07 | half | **FAIL** | — | AccumulateBlockInfo.cpp:622 |
| 2 | broadcast | vec_019 | half | **FAIL** | — | AccumulateBlockInfo.cpp:64 |
| 3 | element_wise | gelu | bf16 | **FAIL** | — | AccumulateBlockInfo.cpp:64 |
| 4 | gather | gather | fp32 | **FAIL** | — | AccumulateBlockInfo.cpp:64 |
| 5 | fa | 2d_unroll Tm16 Tk16 | float | **PASS** | 9583 / 48350 | — |
| 6 | concat | concat_gather | int32 | **PASS** | 14511 | — |
| 7 | concat | concat_scatter | int32 | **FAIL** | — | TMAEngine.cpp:381 |
| 8 | control | hashtable_lookup | — | **FAIL** | — | AccumulateBlockInfo.cpp:64 |
| 9 | norm | rms_norm | — | **PASS** | 4424 | — |
| 10 | reduction | reducesum_row | float | **PASS** | 269 | — |
| 11 | reduction | reducesum_col | float | **PASS** | 268 | — |
| 12 | reduction | reducesum_col | int32 | **PASS** | 268 | — |
| 13 | reduction | reducesum_col | half | **PASS** | 268 | — |
| 14 | reduction | reducemax_row | float | **PASS** | 269 | — |
| 15 | reduction | reducemax_row | int32 | **PASS** | 269 | — |
| 16 | reduction | reducemax_col | float | **PASS** | 153 | — |
| 17 | reduction | reducemax_col | int32 | **PASS** | 153 | — |
| 18 | sort | topk | — | **FAIL** | — | AaccelssMemoryEngine.cpp:12 |
| 19 | transpose | unroll | half | **PASS** | 2902 | — |
| 20 | deepseek | fused_weight | — | **PASS** | 20 | — |
| 21 | deepseek | rms_norm | — | **FAIL** | — | Memory.cpp:336 |
| 22 | matmul | 256² tK32 | float | **PASS** | 2269 | — |
| 23 | matmul | 256² tK64 | float | **PASS** | 1245 | — |
| 24 | matmul | 512² tK64 | float | **PASS** | 9005 | — |
| 25 | matmul | 256×2048² tK64 | float | **PASS** | 67101 | — |
| 26 | multi_thread | vec/tadd | — | **PASS** | 269 | — |
| 27 | multi_thread | vec/trowsum | — | **PASS** | 267 | — |
| 28 | flashMLA | Sq64 Dk512 | — | **PASS** | 5174 | — |

> 注：multi_thread/matmul（`kernel_multi_thread_matmul_B1_M256_N256_K256_tM32_tN32_tK32.elf`）未在 `kernel_elf_list.md` 中列出，单独执行结果为 **FAIL**（AccumulateBlockInfo.cpp:711，implicit-ACC CUBE 目标编码），不计入上表 28 项。

## 失败分类（9 个）

| 失败类型 | 数量 | 断言 | 影响算子 |
|---|---|---|---|
| TSTORE source 契约不匹配 | 4 | `ValidateLocalTlsu` AccumulateBlockInfo.cpp:64 | gelu, gather, control, broadcast vec_019 |
| Text-store 被拒绝 | 2 | `AssertNotTextStore` AaccelssMemoryEngine.cpp:12 / Memory.cpp:336 | topk, deepseek rms_norm |
| COPY expansion 源契约 | 1 | `ValidateReduceAndExpandTepl` AccumulateBlockInfo.cpp:622 | broadcast vec_07 |
| MSCATTER index dtype | 1 | `ExecuteMSCATTER` TMAEngine.cpp:381 | concat_scatter |
| implicit-ACC CUBE 目标 | 1 | `AccumulateBlockInfo` AccumulateBlockInfo.cpp:711 | multi_thread matmul（表外） |

## 失败断言详情

### 1. TSTORE source 契约不匹配（4 个）

```text
ASSERTION FAILED: inst->srcs.size() == 2 && inst->dsts.empty() &&
IsCompatibleDataTile(inst->srcs[1], ...) &&
"Local TSTORE requires one compatible source Tile"
```

位置：`emulator/engine/AccumulateBlockInfo.cpp:64`，`ValidateLocalTlsu`

影响：gelu(bf16)、gather(fp32)、control、broadcast vec_019

### 2. Text-store 被拒绝（2 个）

```text
// topk:
ASSERTION FAILED: false
位置: AssertNotTextStore, AaccelssMemoryEngine.cpp:12

// deepseek rms_norm:
ASSERTION FAILED: gAllowTextStore || !is_text_region(address, width)
位置: Store, Memory.cpp:336
```

影响：topk、deepseek rms_norm

### 3. COPY expansion 源契约（1 个）

```text
ASSERTION FAILED: inst->srcs.size() == 3 && inst->dsts.size() == 1 &&
inst->srcs[1] && inst->srcs[2] &&
IsCompatibleDataTile(inst->srcs[1], ...) &&
"PTO v0.58 COPY expansion requires source0 and broadcast-source"
```

位置：`emulator/engine/AccumulateBlockInfo.cpp:622`，`ValidateReduceAndExpandTepl`

影响：broadcast vec_07

### 4. MSCATTER index dtype（1 个）

```text
ASSERTION FAILED: block->srcTile.size() >= 2u &&
block->srcTile[1]->tileInfo &&
(block->srcTile[1]->tileInfo->dataType == DataType::INT32 ||
 block->srcTile[1]->tileInfo->dataType == DataType::UINT32) &&
"MSCATTER index must be S32/U32 (MSCATTER.md dtypes)"
```

位置：`emulator/engine/TMAEngine.cpp:381`，`ExecuteMSCATTER`

影响：concat_scatter

### 5. implicit-ACC CUBE 目标（1 个）

```text
ASSERTION FAILED: (inst->dsts.empty() || sharedLocalToShared) &&
"implicit-ACC CUBE operations must not encode an ordinary destination"
```

位置：`emulator/engine/AccumulateBlockInfo.cpp:711`

影响：multi_thread matmul
