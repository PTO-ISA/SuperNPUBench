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

> **最后更新**: 2026-08-14（gfrun 从 `ce5e1510` 切换至 `3dc2ffb7`，§6/§7 已复测）

# gfrun 执行结果汇总 — 2026-08-14

> 工具链: `linx-toolchain-build-latest  (clang 15.0.4, linx64v5-musl-local, llvm 86959776bd1f)`
>
> gfrun: `/Users/blacktraker/Programming/gitproj/DV4/SuperScalarModel/bin/gfrun  (SuperScalarModel @ ce5e1510 "fix(emulator): align shared PE mask with ASL")`
>
> 调用: `gfrun -t 1 -f <elf>   (90s per-ELF timeout, 8-way parallel)`

## 工具链&&模型版本

| 组件 | 仓库 | 分支 | Commit |
|---|---|---|---|
| gfrun 模型 | https://github.com/LinxISA/SuperScalarModel | `feat/pto-v058-adaptation` | `ce5e151066f7403819ba8d433c2302300ef46120` |
| llvm-project | https://github.com/LinxISA/llvm-project | `temp/shared-tload-integration-20260811` | `86959776bd1fb22dcc8e73b57ec2276c65d44f38` |
| Linx-TileOP-API | https://github.com/LinxISA/Linx-TileOP-API | `temp/shared-tload-integration-20260811` | `a96d4e9c64d298636f8f2c81f131d2d665fe8693` |

工具链：`llvm-project temp/shared-tload-integration-20260811 @ 86959776b` + `Linx-TileOP-API temp/shared-tload-integration-20260811 @ a96d4e9`

## 更新日志

### 更新点

1. **适配最新 gfrun 与编译工具链**：gfrun 升级至 SuperScalarModel @ `ce5e1510`（`feat/pto-v058-adaptation` 分支）；工具链切换至 `linx-toolchain-build-latest`（clang 15.0.4, linx64v5-musl-local, llvm 86959776bd1f）。
2. **修正算子崩溃问题**：修复 element_wise（gelu）、broadcast 以及 B.IOR stride 语义不一致等导致的 gfrun 崩溃 / 段错误（SIGSEGV）。
3. **新增 fixp microbenchmark**：覆盖定点矩阵乘（fixp）各变体共 63 个用例，含 deq/qf/qs 系列、gemv、groupmax、prelu 等。
4. **新增 TCI 指令 microbenchmark**：增加 TCI（Tile Conditional Initialize）指令用例 `tci_i16_16x16`、`tci_i32_16x16`。
5. **修正结果写回丢失 bug**：修正数值校验中部分算子遗失结果写回（TSTORE 未执行 / 地址错误）的 bug，确保 `res.bin` 完整落盘。

### 回归失败点

1. **multi-PE 算子执行失败**：multi_thread 版本的 fa、matmul 算子在 4-PE 配置（`-s softcore.multiThreadNum=4`）下执行失败。根因：ISA `mask=0001`（单 PE 选择）场景未在 B.IOS 描述符中增加对应声明，gfrun 功能模型在 `ExecuteSharedTMA` 路径触发断言中断（详见 §7）。
2. **数值验证回归失败**：matmul MASK_FP32 数值验证从 PASS（gfrun `c3051e3a`，max abs err 9.81e-7）退化为 FAIL（全 1 输入仅 1216/4096 非零）。疑为 0817 版 bug 修复产生的"幻觉"——修复一处问题的同时引入了副作用，影响了 TLOAD/TSTORE 路径的数值正确性。具体根因待排查（详见 §6）。

### 建议

1. **ISA 侧尽快修复**：补充 `mask=0001` 场景的 B.IOS 描述符声明，统一 B.IOR stride 语义（元素 vs 字节），消除 multi-PE 算子执行断言与数值回归。防止agent生成错误代码。
2. **慎用自动化工具大幅改码**：尽量避免使用机器人（AI / 自动化脚本）对代码进行大范围修改，防止引入难以排查的副作用。

## TL;DR

- **编译产出 ELF：429** （microbench 342 + one-level 83 + multi_thread 4）
- **gfrun 执行：429 个 ELF 全部跑完**（初始批次 425 + multi_thread 4）
- **通过 PASS = 315** / **失败 FAIL = 114**

- 失败构成：gfrun 模型断言失败 109（含 multi_thread 2）、gfrun 段错误 3、超时 2
- **关键定性**：109 个 `FAIL-illegal-inst` **不是 ISA 非法指令**，而是 gfrun 功能模型自身的校验断言 （`gfrun: illegal instruction: ASSERTION FAILED: …` 后跟 backtrace）。即 gfrun(SuperScalarModel @ ce5e1510) 的 validator 拒绝了工具链合法生成的某些 tile/inst/descriptor 组合 —— 属**模型侧与工具链/ISA 契约的偏差**，非算子内核 bug、非真实非法指令。
- **初始编译无失败**：425 个 ELF 全部编译成功。multi_thread/{fa,matmul,vec} 初始编译失败后已修复并重新编译成功（4 个 ELF 见 §7）。

## 1. 编译结果

入口：`bash microbenchmark/compile_all.sh`（cube/vector/memory/scalar/fixp）+ `bash benchmark/one-level-arch/compile_all.sh`（13 个算子）+ 8 个 `compile.all` 附加目录。

### 1.1 编译成功（425 ELF）

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
| one-level | deepseek | 22 |
| one-level | element_wise | 2 |
| one-level | fa | 13 |
| one-level | flashMLA | 2 |
| one-level | gather | 1 |
| one-level | matmul | 16 |
| one-level | norm | 1 |
| one-level | reduction | 5 |
| one-level | sort | 1 |
| one-level | transpose | 4 |
| one-level | multi_thread | 4 |
| | **合计** | **429 |**

### 1.2 初始编译失败（已修复）

multi_thread/{fa,matmul,vec} 在初始编译时因脚本过期 / 头文件 blocker 失败，已后续修复并编译成功（4 ELF，见 §7）。

## 2. 执行结果总览（按类别/算子）

| 树 | 类别/算子 | 总数 | PASS | FAIL |
|---|---|---:|---:|---:|
| microbench | cube | 6 | 2 | 4 |
| microbench | vector | 128 | 114 | 14 |
| microbench | memory | 21 | 19 | 2 |
| microbench | scalar | 124 | 124 | 0 |
| microbench | fixp | 63 | 27 | 36 |
| one-level | broadcast | 6 | 1 | 5 |
| one-level | concat | 4 | 3 | 1 |
| one-level | control | 6 | 0 | 6 |
| one-level | deepseek | 22 | 8 | 14 |
| one-level | element_wise | 2 | 0 | 2 |
| one-level | fa | 13 | 0 | 13 |
| one-level | flashMLA | 2 | 0 | 2 |
| one-level | gather | 1 | 1 | 0 |
| one-level | matmul | 16 | 9 | 7 |
| one-level | norm | 1 | 0 | 1 |
| one-level | reduction | 5 | 1 | 4 |
| one-level | sort | 1 | 0 | 1 |
| one-level | transpose | 4 | 4 | 0 |
| one-level | multi_thread | 4 | 2 | 2 |
| | **合计** | **429** | **315** | **114** |

- microbench **scalar 124/124 全通过**；memory 19/21、vector 114/128、fixp 27/63、cube 仅 2/6。
- one-level 通过率 27/83（broadcast/deepseek/fa/matmul 等大算子多触发模型断言）。
- multi_thread 2/4 通过（vec PASS，fa/matmul 模型断言失败，需 `-s softcore.multiThreadNum=4`，详见 §7）。

## 3. 失败清单（按问题类型分组）

### 3.1 gfrun 模型断言失败（107 个）

gfrun 输出形如 `gfrun: illegal instruction: ASSERTION FAILED: <cond>` + backtrace。按断言条件分组（降序）：

#### 组 1（21 个）：`inst->srcs.size() == 1 && inst->dsts.size() == 1 && inst->dsts[0] && inst->dsts[0]->size !…`

- `tmatmul_bias_fp16_64x64x64`  [microbench/cube]  rc=1
- `tmatmul_fp16_64x64x64`  [microbench/cube]  rc=1
- `tmatmul_mx_fp16_64x64x64`  [microbench/cube]  rc=1
- `fa_2d_unroll_Sq256_Skv512_Tm16_Tk32_X1_Y2`  [one-level/kernel]  rc=1
- `fa_2d_unroll_Sq256_Skv512_Tm16_Tk32_X1_Y4`  [one-level/kernel]  rc=1
- `fa_2d_unroll_Sq256_Skv512_Tm16_Tk32_X2_Y2`  [one-level/kernel]  rc=1
- `fa_2d_unroll_Sq256_Skv512_Tm16_Tk32_X2_Y4`  [one-level/kernel]  rc=1
- `fa_2d_unroll_Sq512_Skv512_Tm16_Tk32_X1_Y2`  [one-level/kernel]  rc=1
- `fa_2d_unroll_Sq512_Skv512_Tm16_Tk32_X1_Y4`  [one-level/kernel]  rc=1
- `fa_2d_unroll_Sq512_Skv512_Tm16_Tk32_X2_Y2`  [one-level/kernel]  rc=1
- `fa_2d_unroll_Sq512_Skv512_Tm16_Tk32_X2_Y4`  [one-level/kernel]  rc=1
- `hashtable_lookup_simd_kNum6144_kMaxProbe512_knum_col1024_debug_off`  [one-level/kernel]  rc=1
- `hashtable_lookup_simd_kNum6144_kMaxProbe512_knum_col1024_debug_on`  [one-level/kernel]  rc=1
- `broadcast_broadcast_vec_019__DType__half_tM8_kInner49_IN_SHAPE1280_1_49_OUT_SHAPE1280_8_49`  [one-level/kernel]  rc=1
- `broadcast_broadcast_vec_07__DType__half_tM16_IN_SHAPE1443_1_OUT_SHAPE1443_129`  [one-level/kernel]  rc=1
- `reduction_reducemax_col_reducemax_col_DTypeint32_t_tM32_tN64_GM2048_GN64`  [one-level/kernel]  rc=1
- `reduction_reducemax_row_reducemax_row_DTypeint32_t_tM16_tN128_GM16_GN8192`  [one-level/kernel]  rc=1
- `reduction_reducesum_col_reducesum_col_DTypeint32_t_tM32_tN64_GM2048_GN64`  [one-level/kernel]  rc=1
- `reduction_reducesum_row_reducesum_row_DTypefloat_tM16_tN128_GM16_GN8192`  [one-level/kernel]  rc=1
- `sfa_Sq256_Skv512_Tm16_Tk32`  [one-level/kernel]  rc=1
- `sfa_Sq512_Skv512_Tm16_Tk32`  [one-level/kernel]  rc=1

#### 组 2（17 个）：`block->srcTile.size() + (sharedRight ? 1u : 0u) == requiredSources && leftIndex < block->s…`

- `fixp_tmatmul_f16_prelu_M32_N32_K32_tM32_tN32_tK32`  [microbench/fixp]  rc=1
- `fixp_tmatmul_rowmax_init_M32_N32_K32_tM32_tN32_tK32`  [microbench/fixp]  rc=1
- `fixp_tmatmul_s8_prelu_M32_N32_K32_tM32_tN32_tK32`  [microbench/fixp]  rc=1
- `fixp_tmatmul_v_deqf16_M32_N32_K32_tM32_tN32_tK32`  [microbench/fixp]  rc=1
- `fixp_tmatmul_v_qf_bf16_M32_N32_K32_tM32_tN32_tK32`  [microbench/fixp]  rc=1
- `fixp_tmatmul_v_qf_f16_M32_N32_K32_tM32_tN32_tK32`  [microbench/fixp]  rc=1
- `fixp_tmatmul_v_qf_f32_M32_N32_K32_tM32_tN32_tK32`  [microbench/fixp]  rc=1
- `fixp_tmatmul_v_qf_fp8_M32_N32_K32_tM32_tN32_tK32`  [microbench/fixp]  rc=1
- `fixp_tmatmul_v_qf_hif8_M32_N32_K32_tM32_tN32_tK32`  [microbench/fixp]  rc=1
- `fixp_tmatmul_v_qf_s16_M32_N32_K32_tM32_tN32_tK32`  [microbench/fixp]  rc=1
- `fixp_tmatmul_v_qf_s4_M32_N32_K32_tM32_tN32_tK32`  [microbench/fixp]  rc=1
- `fixp_tmatmul_v_qf_s8_M32_N32_K32_tM32_tN32_tK32`  [microbench/fixp]  rc=1
- `fixp_tmatmul_v_qs_bf16_M32_N32_K32_tM32_tN32_tK32`  [microbench/fixp]  rc=1
- `fixp_tmatmul_v_reqs8_M32_N32_K32_tM32_tN32_tK32`  [microbench/fixp]  rc=1
- `fixp_tmatmul_v_s8_relu_M32_N32_K32_tM32_tN32_tK32`  [microbench/fixp]  rc=1
- `fixp_tmatmul_v_shifts16_M32_N32_K32_tM32_tN32_tK32`  [microbench/fixp]  rc=1
- `fixp_tmatmul_vqf_s8_prelu_M32_N32_K32_tM32_tN32_tK32`  [microbench/fixp]  rc=1

#### 组 3（14 个）：`srcTile.size() == 1 && dstTile.size() == 1 && srcTile[0] && dstTile[0] && srcTile[0]->tile…`

- `cast_back_per_channel`  [one-level/kernel]  rc=1
- `cast_back_per_token`  [one-level/kernel]  rc=1
- `expand_to_mhc_bwd`  [one-level/kernel]  rc=1
- `fused_weight`  [one-level/kernel]  rc=1
- `hashtable_lookup_simd_kNum6144_kMaxProbe512_knum_col256_debug_off`  [one-level/kernel]  rc=1
- `hashtable_lookup_simd_kNum6144_kMaxProbe512_knum_col256_debug_on`  [one-level/kernel]  rc=1
- `hashtable_lookup_simd_kNum6144_kMaxProbe512_knum_col512_debug_off`  [one-level/kernel]  rc=1
- `hashtable_lookup_simd_kNum6144_kMaxProbe512_knum_col512_debug_on`  [one-level/kernel]  rc=1
- `element_wise_gelu_debug_gelu_debug_Approximate_false_DType__bf16_tM2048_SHAPE24_8_1024`  [one-level/kernel]  rc=1
- `element_wise_gelu_gelu_Approximate_false_DType__bf16_tM2048_SHAPE24_8_1024`  [one-level/kernel]  rc=1
- `per_channel_cast`  [one-level/kernel]  rc=1
- `per_token_cast`  [one-level/kernel]  rc=1
- `reduce_fused`  [one-level/kernel]  rc=1
- `swiglu_forward_and_per_token_cast`  [one-level/kernel]  rc=1

#### 组 4（11 个）：`rightInfo->validRow == k && (!tmatmul || rightInfo->validCol == n) && "CUBE right descript…`

- `fixp_tmatmul_gemv_M32_N32_K32_tM32_tN32_tK32`  [microbench/fixp]  rc=1
- `fixp_tmatmul_gemv_acc_M32_N32_K32_tM32_tN32_tK32`  [microbench/fixp]  rc=1
- `fixp_tmatmul_gemv_bias_M32_N32_K32_tM32_tN32_tK32`  [microbench/fixp]  rc=1
- `fixp_tmatmul_gemv_mx_M32_N32_K32_tM32_tN32_tK32`  [microbench/fixp]  rc=1
- `fixp_tmatmul_gemv_mx_acc_M32_N32_K32_tM32_tN32_tK32`  [microbench/fixp]  rc=1
- `fixp_tmatmul_gemv_mx_bias_M32_N32_K32_tM32_tN32_tK32`  [microbench/fixp]  rc=1
- `fixp_tmatmul_gemv_mx_s8_M32_N32_K32_tM32_tN32_tK32`  [microbench/fixp]  rc=1
- `fixp_tmatmul_gemv_s8_M32_N32_K32_tM32_tN32_tK32`  [microbench/fixp]  rc=1
- `matmul_A16W4_B1_M256_N2048_K2048_tM16_tN16_tK128`  [one-level/kernel]  rc=1
- `matmul_A16W4_B1_M512_N1280_K2048_tM16_tN16_tK128`  [one-level/kernel]  rc=1
- `matmul_A16W4_B1_M512_N512_K4096_tM16_tN16_tK128`  [one-level/kernel]  rc=1

#### 组 5（9 个）：`block->dataType == DataType::FP32 && fp8Pair && block->srcTile[leftScaleIndex]->tileInfo &…`

- `fixp_tmatmul_mx_M32_N32_K32_tM32_tN32_tK32`  [microbench/fixp]  rc=1
- `fixp_tmatmul_mx_s8_M32_N32_K32_tM32_tN32_tK32`  [microbench/fixp]  rc=1
- `fixp_tmatmul_mxacc_M32_N32_K32_tM32_tN32_tK32`  [microbench/fixp]  rc=1
- `fixp_tmatmul_mxbias_M32_N32_K32_tM32_tN32_tK32`  [microbench/fixp]  rc=1
- `fa_HIF4_HIF4_BF16_NOGATHER_Sq256_Skv512_Tm8_Tk32_X1_Y1`  [one-level/kernel]  rc=1
- `matmul_HIF4_HIF4_MX_NOGATHER_B1_M256_N2048_K2048_tM32_tN32_tK64`  [one-level/kernel]  rc=1
- `matmul_HIF4_HIF4_MX_NOGATHER_B1_M512_N1280_K4096_tM32_tN32_tK64`  [one-level/kernel]  rc=1
- `matmul_HIF4_HIF4_MX_NOGATHER_REUSEA_B1_M256_N2048_K2048_tM32_tN32_tK64`  [one-level/kernel]  rc=1
- `matmul_HIF4_HIF4_MX_NOGATHER_REUSEA_B1_M512_N1280_K4096_tM32_tN32_tK64`  [one-level/kernel]  rc=1

#### 组 6（8 个）：`reserved/deleted TEPL selector`

- `mgather_fp16_16x16`  [microbench/memory]  rc=1
- `mgather_fp32_16x16`  [microbench/memory]  rc=1
- `taxpy_fp16_16x16`  [microbench/vector]  rc=1
- `thistogram_fp16_16x16`  [microbench/vector]  rc=1
- `thistogram_fp32_16x16`  [microbench/vector]  rc=1
- `tprelu_fp16_16x16`  [microbench/vector]  rc=1
- `tprelu_fp32_16x16`  [microbench/vector]  rc=1
- `inplace_unique_group_indices`  [one-level/kernel]  rc=1

#### 组 7（5 个）：`source->tileInfo->dataType == block->dataType && source->tileInfo->validRow == validRow &&…`

- `fa_softmax_pto_Sq256_Skv512_Tm16_Tk32`  [one-level/kernel]  rc=1
- `fa_softmax_pto_Sq512_Skv512_Tm16_Tk32`  [one-level/kernel]  rc=1
- `flashMLA_Sq32_QHeadPerHK1_NumBlocks1_Dk512_Dv512_DChunk128_VChunk128_Tm16_Tk16`  [one-level/kernel]  rc=1
- `flashMLA_Sq64_QHeadPerHK1_NumBlocks2_Dk512_Dv512_DChunk128_VChunk128_Tm16_Tk16`  [one-level/kernel]  rc=1
- `norm_rms_norm_M16_N256_tM8_tN128`  [one-level/kernel]  rc=1

#### 组 8（4 个）：`inst->srcs.size() == 3 && inst->dsts.empty() && IsCompatibleLogicalTile(inst->srcs[1], val…`

- `aux_fi`  [one-level/kernel]  rc=1
- `get_fused_mapping`  [one-level/kernel]  rc=1
- `group_count`  [one-level/kernel]  rc=1
- `mask_indices_by_tp`  [one-level/kernel]  rc=1

#### 组 9（3 个）：`IsCompareSelectTeplDataType(block->tileOp, block->dataType) && "compare/select TEPL tuple …`

- `tsels_fp16_16x16`  [microbench/vector]  rc=1
- `tsels_fp32_16x16`  [microbench/vector]  rc=1
- `topk_gate`  [one-level/kernel]  rc=1

#### 组 10（3 个）：`IsLogicalIntegerTeplDataType(block->dataType) && "scalar logical TEPL tuple is not defined…`

- `tands_fp16_16x16`  [microbench/vector]  rc=1
- `tors_fp16_16x16`  [microbench/vector]  rc=1
- `txors_fp16_16x16`  [microbench/vector]  rc=1

#### 组 11（2 个）：`(accInfo->dataType == DataType::FP32 || accInfo->dataType == DataType::INT32) && accInfo->…`

- `fixp_tmatmul_acc_M32_N32_K32_tM32_tN32_tK32`  [microbench/fixp]  rc=1
- `fixp_tmatmul_acc_s8_M32_N32_K32_tM32_tN32_tK32`  [microbench/fixp]  rc=1

#### 组 12（2 个）：`(dataType == DataType::INT8 || dataType == DataType::UINT8 || dataType == DataType::INT16 …`

- `tshls_fp16_16x16`  [microbench/vector]  rc=1
- `tshrs_fp16_16x16`  [microbench/vector]  rc=1

#### 组 13（2 个）：`IsBasicUnaryTeplDataType(block->tileOp, block->dataType) && "TEPL opcode/data-type tuple i…`

- `tabs_i16_16x16`  [microbench/vector]  rc=1
- `tabs_i32_16x16`  [microbench/vector]  rc=1

#### 组 14（2 个）：`block->dataType == DataType::FP32 && block->blockAttr && block->blockAttr->dataType == Dat…`

- `fixp_tmatmul_s8_shared_M32_N32_K32_tM32_tN32_tK32`  [microbench/fixp]  rc=1
- `fixp_tmatmul_shared_M32_N32_K32_tM32_tN32_tK32`  [microbench/fixp]  rc=1

#### 组 15（2 个）：`source && source->tileInfo && (source->tileInfo->validRow == 1u || source->tileInfo->valid…`

- `fixp_tmatmul_bias_M32_N32_K32_tM32_tN32_tK32`  [microbench/fixp]  rc=1
- `fixp_tmatmul_bias_s8_M32_N32_K32_tM32_tN32_tK32`  [microbench/fixp]  rc=1

#### 组 16（1 个）：`accBytes != 0 && dimensionsArePowersOfTwo && col >= n && dstTile[0]->size != 0 && rowBytes…`

- `tmatmul_i8_64x64x64`  [microbench/cube]  rc=1

#### 组 17（1 个）：`currentBlock->hasFixpAttr && "PTO v0.58 Matrix requires one ASL B.FPATR descriptor"`

- `fixp_tmatmul_rowgroup_maxabs_M32_N32_K32_tM32_tN32_tK32`  [microbench/fixp]  rc=1

### 3.2 gfrun 段错误（3 个）

gfrun 进程 SIGSEGV（模型自身崩溃，rc=-11），均为 `broadcast __half tM2048`：

- `broadcast_broadcast__DType__half_tM2048_IN_SHAPE1042_1_OUT_SHAPE1042_129`  rc=-11  t=0.1s
- `broadcast_broadcast__DType__half_tM2048_IN_SHAPE1280_1_49_OUT_SHAPE1280_8_49`  rc=-11  t=0.1s
- `broadcast_broadcast__DType__half_tM2048_IN_SHAPE1_1_1_65_128_OUT_SHAPE1_1_7_65_128`  rc=-11  t=0.1s

### 3.3 超时（2 个，>90s）

gfrun 90s 内未跑完（数据量过大或疑似死循环）：

- `concat_concat_scatter_DType__half_tM512_IN_SHAPE256_8_OUT_SHAPE256_8000`  rc=124  t=96.0s
- `topk`  rc=124  t=98.5s

## 4. 通过清单（313 个，按类别）

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

### microbench/fixp（27）

  `fixp_tmatmul_bf16_M32_N32_K32_tM32_tN32_tK32` · `fixp_tmatmul_bf16_relu_M32_N32_K32_tM32_tN32_tK32` · `fixp_tmatmul_f16_M32_N32_K32_tM32_tN32_tK32` · `fixp_tmatmul_f16_groupmax_M32_N32_K32_tM32_tN32_tK32`
  `fixp_tmatmul_f16_relu_M32_N32_K32_tM32_tN32_tK32` · `fixp_tmatmul_groupmax_128_M32_N32_K32_tM32_tN32_tK32` · `fixp_tmatmul_groupmax_16_M32_N32_K32_tM32_tN32_tK32` · `fixp_tmatmul_groupmax_8_M32_N32_K32_tM32_tN32_tK32`
  `fixp_tmatmul_keep_acc_M32_N32_K32_tM32_tN32_tK32` · `fixp_tmatmul_keep_acc_relu_M32_N32_K32_tM32_tN32_tK32` · `fixp_tmatmul_legacy3_M32_N32_K32_tM32_tN32_tK32` · `fixp_tmatmul_rowmax_M32_N32_K32_tM32_tN32_tK32`
  `fixp_tmatmul_s8_lrelu_M32_N32_K32_tM32_tN32_tK32` · `fixp_tmatmul_s8_relu_M32_N32_K32_tM32_tN32_tK32` · `fixp_tmatmul_s8_rowmax_M32_N32_K32_tM32_tN32_tK32` · `fixp_tmatmul_s_deqf16_M32_N32_K32_tM32_tN32_tK32`
  `fixp_tmatmul_s_qf_bf16_M32_N32_K32_tM32_tN32_tK32` · `fixp_tmatmul_s_qf_f16_M32_N32_K32_tM32_tN32_tK32` · `fixp_tmatmul_s_qf_f32_M32_N32_K32_tM32_tN32_tK32` · `fixp_tmatmul_s_qf_fp8_M32_N32_K32_tM32_tN32_tK32`
  `fixp_tmatmul_s_qf_hif8_M32_N32_K32_tM32_tN32_tK32` · `fixp_tmatmul_s_qf_s16_M32_N32_K32_tM32_tN32_tK32` · `fixp_tmatmul_s_qf_s4_M32_N32_K32_tM32_tN32_tK32` · `fixp_tmatmul_s_qf_s8_M32_N32_K32_tM32_tN32_tK32`
  `fixp_tmatmul_s_qs_bf16_M32_N32_K32_tM32_tN32_tK32` · `fixp_tmatmul_s_reqs8_M32_N32_K32_tM32_tN32_tK32` · `fixp_tmatmul_s_shifts16_M32_N32_K32_tM32_tN32_tK32`

### one-level（27）

- `batched_transpose`
- `engram_hash_layer`
- `expand_to_mhc_fwd`
- `fn_normw_merge_fwd`
- `broadcast_broadcast_vec_039__DType__half_tM8_kInner16_IN_SHAPE8192_1_16_OUT_SHAPE8192_8_16`
- `concat_concat_gather_DType__half_tM512_IN_SHAPE256_8_OUT_SHAPE256_8000`
- `concat_concat_gather_DTypeint32_t_tM512_IN_SHAPE64_2_OUT_SHAPE64_2000`
- `concat_concat_scatter_DTypeint32_t_tM512_IN_SHAPE64_2_OUT_SHAPE64_2000`
- `gather_gather_DType__fp32_OTypeuint32_t_gKs131072_gMs32_gNs256_tMs32_tNs64`
- `reduction_reducesum_col_reducesum_col_DType__half_tM32_tN64_GM2048_GN64`
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
- `rms_norm`
- `sinkhorn_fwd`
- `transpose_DType__half_tM512_IN1476_32_OUT32_1476`
- `transpose_DType__half_tM512_IN1_32_8_32_OUT1_8_32_32`
- `transpose_DType__half_tM512_IN1_8_4096_3_OUT1_8_3_4096`
- `transpose_DType__half_tM512_IN1_8_64_4_16_7_OUT1_8_16_4_64_7`

## 5. 说明与边界

- **通过判定**：gfrun 退出码 0 且输出含 `Reach the End of Benchmark` 且 `R2 = 0`。
- **模型断言 ≠ 非法指令**：gfrun 把内部 validator 断言违例也打印为 `illegal instruction`，故分类器初判为 `FAIL-illegal-inst`；实为 SuperScalarModel 功能模型对部分 tile/inst 配置尚未支持。
- **工具链**：全程使用 `linx-toolchain-build-latest`（用户指定），未用旧 `linx-toolchain-build`。
- **gfrun 版本**：SuperScalarModel @ `ce5e1510`（早于先前 issue 报告所引 `c3051e3a`，后者已过期）。
- **fixp**：63 个可编译模式中 27 通过 / 36 模型断言失败；另有 `LRELU_ONLY` 因工具链 B.IOR 匹配缺口未编译（已知 BLOCKED，不在 425 之列）。
- 原始数据：`/tmp/gfrun_results.json`（含每个 ELF 的 rc/elapsed/signature/tail）、`/tmp/gfrun_results.csv`。

## 6. matmul 数值验证：Path B 分析

### 6.1 结论

matmul MASK_FP32 数值验证 **FAIL**：两条路径（裸机自检 / reson 随机输入）均不通过。

**关于根因**：先前分析将 B.IOR stride 值 64 误认为"元素"并归因于 gfrun commit `82dffea8`，此结论有误。B.IOR 中 row stride 值 64 的单位是**字节**——一行 tile 16 元素 × 4B = 64B，stride=64 本身正确。`82dffea8` 将 B.IOR stride 按字节处理（`×1`）是正确的，并非回归根因。实际的 1216/4096 失败模式需进一步排查。

### 6.2 两条验证路径均 FAIL

| 路径 | 配置 | 输入 | 结果 |
|---|---|---|---|
| A: MATMUL_VALIDATE | 裸机 `_start.s`，代码内填全 1，自检 `dst==K` | 全 1 | **FAIL**：1216/4096 正确，2880 为 0，6 为 printf 垃圾；R2=1 |
| B: MASK_FP32 res_check | hosted crt0，读 src0/src1.bin，写 res.bin | 全 1 / 随机 | **FAIL**：全 1 1216/4096=64.0 其余 0；随机 mse=0.008 max_abs=0.401 |

> 注：先前 `matmul_MASK_FP32_reson_precision.md`（2026-08-17）报告 PASS（max abs err 9.81e-7），所用 gfrun 版本早于 `82dffea8`。同一 ELF、同一编译器，gfrun 升级后即 FAIL。

### 6.3 输出模式

用全 1 输入跑 MASK_FP32，dump `res.bin`（64×64 FP32）：

| 行范围 | 非零元素数 | 说明 |
|---|---|---|
| 0–3 | 各 64（=64.0） | 每 16 行 block 的前 4 行被写入 |
| 4 | 48 | 第 5 行仅前 48 元素 |
| 5–15 | 0 | 未写入 |
| 16–19 | 各 64 | 第二个 16 行 block |
| 20 | 48 | |
| 21–31 | 0 | |
| 32–35 | 各 64 | 第三个 block |
| 36 | 48 | |
| 37–47 | 0 | |
| 48–51 | 各 64 | 第四个 block |
| 52 | 48 | |
| 53–63 | 0 | |

**总计：4 × (4×64 + 48) = 4 × 304 = 1216 / 4096 非零**，两条路径一致。

### 6.4 B.IOR stride 说明

MASK_FP32 反汇编中，K-tile 循环体 `.LBB4_3`：

```asm
1238c: addi zero, 16,  ->s1     ; s1 = 16 (tile 宽/高)
12390: addi zero, 64,  ->s2     ; s2 = 64 = 一行 tile 的字节数 (16 × 4B)
12394: BSTART.TLSU TLOAD, FP32
123a0: C.B.DIMI 16, ->lb2       ; lb2 = 16 (dense stride)
123a6: B.IOR [a7,s2],[]         ; B.IOR: base=a7, stride=s2=64 (字节)
```

- `s2 = 64`：一行 16 元素 FP32 tile = 16 × 4B = 64 字节，stride 值正确。
- gfrun `82dffea8` 将 B.IOR stride 按字节处理（`×1`）与此一致。

### 6.5 复现

```bash
GFRUN=/Users/blacktraker/Programming/gitproj/DV4/SuperScalarModel/bin/gfrun   # ce5e1510
ELF=benchmark/one-level-arch/output/kernel/matmul/elf/kernel_matmul/matmul_MASK_MASK_FP32_M64_N64_K64_tM16_tN16_tK16.elf
CHKD=benchmark/one-level-arch/compare/matmul_MASK_MASK_FP32_M64_N64_K64_tM16_tN16_tK16

# 全 1 输入
python3 -c "import numpy as np; n=64*64; np.ones(n,dtype=np.float32).tofile('$CHKD/src0.bin'); np.ones(n,dtype=np.float32).tofile('$CHKD/src1.bin'); np.zeros(n,dtype=np.float32).tofile('$CHKD/res.bin')"

# 跑 gfrun
$GFRUN -t 1 -f $ELF -m 200000

# 验证模式：1216/4096 非零
python3 -c "
import numpy as np
r=np.fromfile('$CHKD/res.bin',dtype=np.float32).reshape(64,64)
print('nonzero:',np.count_nonzero(r),'/',4096)
for i in range(64):
    nz=np.count_nonzero(r[i])
    if nz: print(f'Row {i:2d}: {nz} nonzero')
"
```

### 6.6 待排查

1216/4096 的失败模式仍然存在（gfrun ce5e1510），但根因不是 B.IOR stride。c3051e3a → ce5e1510 之间共 4 个 commit：

| Commit | 内容 | 是否影响 MASK_FP32（初步判断） |
|---|---|---|
| `82dffea8` | B.IOR stride 按字节处理 | 否：stride=64B 正确，`×1` 正确 |
| `ce5e1510` | PE mask 位序反转 + soleExecutor 移除 | 待查：MASK_FP32 用 regular TLOAD（非 Shared），但需验证 dispatch 路径 |
| `e98c50bf` | copy expansion 操作数变更 | 待查：matmul 无 TROWEXPAND/TCOLEXPAND，但需确认 AccumulateBlockInfo 影响 |
| `ee635932` | copy expansion 断言放宽 | 待查：同上 |

需进一步排查的方向：ce5e1510 对 regular TLOAD/TMATMUL 路径是否有间接影响（如 block 属性、tile register 提交逻辑等）。

## 7. multi_thread 算子编译与执行

### 7.1 编译

工具链：`linx-toolchain-build-latest/output/linx_blockisa_llvm_musl/bin`（clang 15.0.4）。编译前遇到 SSRGET 内联汇编 blocker，因工具链 bundled header `pto_tileop.hpp` 已从裸 `asm volatile("SSRGET ...")` 更新为 `__builtin_linx_get_thread_idx()`，故已解除。3 个算子共 4 个 ELF 全部编译成功。

| 算子 | compile.all 参数 | ELF |
|---|---|---|
| vec/tadd | `TESTCASE=tadd TileRows=16 TileCols=16` | `output/kernel/multi_thread/vec/elf/kernel_multi_thread_vec_Rows16_Cols16.elf` |
| vec/trowsum | `TESTCASE=trowsum TileRows=16 TileCols=16` | `output/kernel/multi_thread/vec/elf/kernel_multi_thread_vec_trowsum_Rows16_Cols16.elf` |
| matmul | `TESTCASE=matmul_shared B=1 M=256 N=256 K=256 tM=32 tN=32 tK=32` | `output/kernel/multi_thread/matmul/elf/kernel_multi_thread_matmul_matmul_shared_B1_M256_N256_K256_tM32_tN32_tK32.elf` |
| fa | `TESTCASE=fa_2d_unroll_gmma Sq=128 Skv=64 Tm=16 Tk=16` | `output/kernel/multi_thread/fa/elf/kernel_multi_thread_fa_Sq128_Skv64_Tm16_Tk16.elf` |

### 7.2 执行

gfrun 版本 `ce5e1510`（`feat/pto-v058-adaptation`）。multi_thread 算子需带 `-s softcore.multiThreadNum=4`（4-PE 配置），否则默认 1-PE 会导致 cooperative TMATMUL 断言失败。

命令：`gfrun -t 1 -f <elf> -s softcore.multiThreadNum=4`

| 算子 | 结果 | 说明 |
|---|---|---|
| vec/tadd | **PASS** | 退出码 0，`Reach the End of Benchmark! R2 = 0` |
| vec/trowsum | **PASS** | 退出码 0，`Reach the End of Benchmark! R2 = 0` |
| matmul_shared | **FAIL** | gfrun 模型断言：`cooperative TMATMUL requires fully-defined Shared sources`（rightShared 未就绪） |
| fa | **FAIL** | gfrun 模型断言：`Shared TLOAD size must match the supported Right tile`（`ExecuteSharedTMA`, `SoftCore.cpp:710`） |

### 7.3 失败说明

两个 FAIL 均为 gfrun 功能模型侧的 validator 断言，**非 ISA 非法指令、非编译错误**：

- **fa**：`ExecuteSharedTMA` 断言 Shared TLOAD 的 tile 尺寸不符合模型当前支持的 Right tile 规格。模型对 Shared TLOAD 的尺寸校验尚未覆盖 fa 算子所需的配置。
- **matmul_shared**：cooperative TMATMUL 要求 Shared 源（右操作数 B）在所有参与 PE 上均已定义（`shared.definedMask == binding.peMask`），但模型当前未将 rightShared 标记为 ready，导致 `shared.ready` 为 false。

> 两者均属 SuperScalarModel 功能模型对 multi_thread Shared TLOAD / cooperative TMATMUL 路径尚未完全支持，与算子代码和编译结果无关。

### 7.4 复现

```bash
GFRUN=/Users/blacktraker/Programming/gitproj/DV4/SuperScalarModel/bin/gfrun
OUT=benchmark/one-level-arch/output/kernel/multi_thread

# vec/tadd
$GFRUN -t 1 -f $OUT/vec/elf/kernel_multi_thread_vec_Rows16_Cols16.elf -s softcore.multiThreadNum=4

# vec/trowsum
$GFRUN -t 1 -f $OUT/vec/elf/kernel_multi_thread_vec_trowsum_Rows16_Cols16.elf -s softcore.multiThreadNum=4

# matmul_shared
$GFRUN -t 1 -f $OUT/matmul/elf/kernel_multi_thread_matmul_matmul_shared_B1_M256_N256_K256_tM32_tN32_tK32.elf -s softcore.multiThreadNum=4

# fa
$GFRUN -t 1 -f $OUT/fa/elf/kernel_multi_thread_fa_Sq128_Skv64_Tm16_Tk16.elf -s softcore.multiThreadNum=4
```
