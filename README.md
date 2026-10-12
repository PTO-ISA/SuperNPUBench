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
# one-level-arch (PTO ISA) — multi-thread operator
cd benchmark/one-level-arch/test/kernel/matmul
make TESTCASE=matmul COMPILER_DIR="$COMPILER_DIR" B=1 M=256 N=256 K=256 tM=32 tN=32 tK=32
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

# res_check 数值校验结果汇总 — 2026-09-01

与常规 gfrun 功能回归（只验“跑到终点 + R2=0”，不查结果数值）不同，本轮在
`res_check=on` 下重编全部算子并复跑 gfrun，**对每个算子的计算结果做金标准（golden）比对**，
暴露功能性模型在数值层面的保真度差距。编译器仍按 AGENTS.md 用主 `linx-toolchain-build`
worktree（clang 15.0.4 / linx64v5-unknown-linux-musl）；gfrun 用 `SuperScalarModel/bin/gfrun`。

## 校验方法

| 范围 | res_check 机制 | PASS 判据 |
|---|---|---|
| microbenchmark | `res_check=on` → `Makefile.common` 加 `-DRES_CHECK`，产物落到 `output/res_check/`；测试在 `main()` 内用 `bench_utils.hpp::verify()/verify_scalar()` 把算子输出与 host C 参考逐元素比对，失败置 `g_numeric_failure` | gfrun rc=0 且含 `Reach the End of Benchmark` 且 `R2=0`（R2≠0 即数值不匹配） |
| one-level-arch | `res_check=on` → 加 `-DRES_CHECK -DENABLE_BINARY_OUTPUT -DCHK_DIR="compare/<test>"`，`CC_LINK` 置空并链 `group_worker_runtime.o`；运行时把算子二进制输出与 `compare/<test>/` 金标准逐字节比对 | 同上（`R2=0` 表示金标准一致） |

- 固定 `COMPILER_DIR`（主 worktree）。multi_thread 与 fixp 协作（cooperative）模式均加
  `-s softcore.multiThreadNum=4`。单 ELF 90s 看门狗。共 492 个 res_check ELF。
- 常规回归里“PASS”只代表“模型没崩、跑到终点”；res_check 才查“结果对不对”。因此本轮
  通过率（55.1%）显著低于常规回归（~81.7%）——多出的失败全属**数值层**问题。

## 总体结果

| 范围 | ELF 数 | PASS | FAIL | 通过率 |
|---|---|---|---|---|
| microbenchmark | 398 | 195 | 203 | 49.0% |
| one-level-arch | 94 | 76 | 18 | 80.9% |
| **合计** | **492** | **271** | **221** | **55.1%** |

## 算子族通过率

| 算子族 | ELF | PASS | FAIL | 通过率 | 说明 |
|---|---|---|---|---|---|
| micro/vector | 127 | 6 | 121 | 4.7% | 系统性数值不匹配（见下 Bucket A） |
| micro/scalar | 124 | 68 | 56 | 54.8% | 42 数值 + 14 移位/SQRT 无 handler |
| micro/fixp | 122 | 116 | 6 | 95.1% | 4 协作 max-reduction 缺口 + 2 S4 零点 |
| micro/memory | 14 | 4 | 10 | 28.6% | tload/tstore/mgather/mscatter 往返失真 |
| micro/cube | 11 | 1 | 10 | 9.1% | TMATMUL 累加结果偏离 host 参考 |
| one-level/fa | 10 | 10 | 0 | 100% | 金标准全过 |
| one-level/matmul | 3 | 3 | 0 | 100% | 金标准全过 |
| one-level/multi_thread/matmul | 8 | 8 | 0 | 100% | 金标准全过 |
| one-level/multi_thread/normalization | 2 | 2 | 0 | 100% | 金标准全过 |
| one-level/multi_thread/reduction | 4 | 4 | 0 | 100% | 金标准全过 |
| one-level/deepseek | 21 | 16 | 5 | 76.2% | 5 个逻辑 tile 契约断言（模型侧） |
| one-level/multi_thread/fa | 7 | 5 | 2 | 71.4% | HIF8 rc=134 + MXFP4 tile-carrier |
| one-level/multi_thread/broadcast | 1 | 0 | 1 | 0% | raw tile spill 源形状不匹配 carrier |
| one-level/broadcast | 6 | 5 | 1 | 83.3% | 1 个 COPY 广播展开契约 |
| one-level/reduction | 6 | 5 | 1 | 83.3% | 1 个 TROWSUM 操作数契约 |
| one-level/concat | 4 | 3 | 1 | 75.0% | 1 个 scatter 日志截断/超时（待定） |
| one-level/control | 6 | 0 | 6 | 0% | hashtable_lookup tile-carrier 契约 |
| one-level/sort | 1 | 0 | 1 | 0% | topk 日志截断/超时（待定） |
| 其余 one-level 单/多线程族 | 15 | 15 | 0 | 100% | element_wise/gather/transpose/flashMLA + 多线程 concat·conv2d·gather·transpose·vec·element_wise 全过 |

> 合计编译 492 ELF（microbench 398 + one-level 94）。本轮为**数值校验**口径，FAIL 含
> “数值不匹配”与“模型断言中止”两类；与常规功能回归的 FAIL 不可直接对比。

## 失败归因（两大桶）

### Bucket A — 数值不匹配（178，rc=0 / R2=1，跑到终点但比对失败）

算子完整执行并打印 `Reach the End of Benchmark`，但 `verify()`/金标准比对报错。全部集中在
microbench（one-level 金标准比对几乎全过）。**与精度容差无关**：i32/i16 整型（精确算术，eps=0）
与 fp16/fp32 同等失败，证明不是浮点容差问题，而是模型侧结果本身不对。

| 算子族 | 数量 | 根因 |
|---|---|---|
| micro/vector | 114 | 元素级算术结果未落回输出缓冲：`tadd/tsub/tmul`（ref=3/1/2 非零）全失败，而 `tand/trem`（2&1=0、2%1=0）因 ref 恰为 0 与零初值 c 相等而**伪通过**；仅 `tcvt`（拷贝写回）真通过。dtype 无关（fp16/fp32/i32/i16 全失败） |
| micro/scalar | 42 | per-op 算术保真度缺口：同模板下 `and` 通过、`add/sub/mul` 失败，输入为非常量、`verify_scalar` 实比对，模型标量算术结果偏离 host 参考 |
| micro/cube | 10 | TMATMUL（fp16/fp32/bf16/i8/bias/acc 全变体）累加结果偏离 host 参考；唯一通过的是不需累加的变体 |
| micro/memory | 10 | tload/tstore/mgather/mscatter 的 load→tile→store 往返不保数据：加载到 tile 再写回 c 后，c 与源 a 不等 |
| micro/fixp | 2 | `s_qf_s4`/`v_qf_s4`：S4 量化带零点偏移，零输入下 `check_zero_result` 仍检出非零 D（该 smoke-test 不适用于带零点量化的测例，非真 bug） |

### Bucket B — 功能性故障（41，rc≠0，gfrun 模型断言/illegal instruction 中止）

算子未跑到终点，gfrun 在执行中命中模型断言。这部分与常规功能回归的 FAIL 重合。

| 断言/现象 | 数量 | 算子族 | 根因 |
|---|---|---|---|
| `threadStatus.size() >= kCorePeCount`（协作 TMATMUL 需 4 PE） | 4 | micro/fixp | `shared_rowmax_init`/`shared_rowgroup_maxabs`/`shared_f16_groupmax`/`shared_s8_rowmax`——协作+非 keep_acc 预量化+max 归约集合的已知工具链/模型缺口（单 PE 孪生通过），详见 fixp 源码 NOTE |
| `m_handlers.find(grp) != m_handlers.cend()` | 21 | micro/scalar 14 + micro/vector 7 | **模型未注册移位与开方指令 handler**：标量 `sll/sra/srl`（i32/i64）、`sqrt`（f64）与向量 `tshl/tshr/trsqrt/tsqrt` 全部 illegal instruction |
| `srcTile.size()==1 && dstTile.size()==1 && ...TileCarrier` | 5 | one-level/control 4 + multi_thread/fa 1 | TEPL/COPY tile-carrier 契约：hashtable_lookup 的 tile 传送与 MXFP4 fa 的 tile 尺寸不满足契约 |
| `IsCompatibleLogicalTile` / `priorSources` / `IsCompatibleOperationDataTile` | 5 | one-level/deepseek | deepseek group/mapping kernel 用到的 3 源逻辑 tile 形状/数据 tile 契约未满足 |
| `RawTileSourceFits(source, shape)` | 3 | control 2 + multi_thread/broadcast 1 | raw tile spill 源形状不匹配 carrier |
| `broadcastShapeLegal`（COPY 广播展开） | 1 | one-level/broadcast | 广播展开维度契约 |
| `illegal TROWSUM operand or descriptor` | 1 | one-level/reduction | TROWSUM 操作数/descriptor 契约 |
| rc=134（abort） | 1 | multi_thread/fa | `HIF8_VECFP32` 运行时 abort |

### 待定（2）

`concat_scatter`（half, tM512）与 `topk`：日志被 400 行截断且无 `Reach the End` 标记、rc 无法解析，
疑为超时或截断致判据缺失（非数值/非断言）。需以更长日志复跑确认。

## 结论与要点

1. **常规 gfrun 回归“PASS”≠ 结果正确**。`micro/vector` 121/127 在功能回归里全部“PASS”，但
   res_check 下 114 个数值不匹配——功能性模型把指令跑通了，结果却没写回/算错。res_check 是
   唯一能挡住这类“假绿”的关卡，应纳入回归基线。
2. **两大缺口可定位到模型侧**：(a) 元素级算术/访存结果未正确落回内存（vector/cube/memory/scalar
   数值桶）；(b) 移位/SQRT 指令组未注册 handler（scalar/vector 功能桶）。两者均非 kernel 代码
   缺陷——同一模板下 `and/cvt` 通过、`add/mul` 失败即可证。
3. **one-level 金标准保真度高**：94 个里 76 过（80.9%），失败全是少量 kernel 命中 tile 契约断言
   （hashtable/deepseek/broadcast/TROWSUM），属模型对个别 tile op 的支持边界，非数值漂移。
4. **fixp 协作模式**：33 个 cooperative 模式须加 `-s softcore.multiThreadNum=4`（首轮漏配致 33 个
   伪 FAIL，补跑后 29 翻转为 PASS、4 留作已记录缺口）。后续回归脚本对 fixp 协作模式应默认带 4-PE。
5. **数值不匹配的 verify() 不打印逐元素差异**（只置 R2=1），定位需离线比对。建议后续给
   `bench_utils.hpp::verify()` 加一行首个失配元素的 `expected/got` 打印，可大幅缩短排障路径。

> 提取方法：`res_check=on` 全量重编 → gfrun 逐 ELF 跑（multi_thread/fixp 协作加 4-PE）→
> 按rc 与 R2 二分（rc=0&R2=1=数值；rc≠0=断言）→ 失配断言文本取每日志首行聚类。原始明细：
> `/tmp/res_check_run/summary_corrected.tsv`（elf / 类别 / 状态 / rc / note）。


> **当前验证基线**：2026-09-18（566 个已编译 ELF 全量 gfrun 复测，本地代码含 fa_lowp 重构 + 新 fa 测例；fa 结果为陈旧对象事故修正后复测版）；
> TileOP-API `697f5d8`→`3be8652`（+11 提交，prefix-view 重载约束、B.ASSEMBLE WriterSizeCode 等）；pto-spec `9323e466`→`b541bbdd`；
> gfrun 切到 `fix/gfrun-717-pto291-subview-ordering` `ec5ee047`（PTO #291 subview ordering 修复分支，含 PTO #311 分支合并）；
> fa 86 ELF 53 PASS（unroll_gmma 15/15、kchains 32链 18/21 + 单链 18/21、**fa_gmma_opt 1/1、fa_lowp_recip 1/2**）；matmul 30/31；
> **10 个回归全部模型侧同根因**（normalization ×8 + reduction row ×2，`validCol/physicalCol` Local layout 断言，fix 分支 PTO #291 契约变化）；
> matmul_quantize ×2 编译失败（远端 mxquant 重构破坏集成）；fixp MX 家族 ×10 编译失败（TileOP 3be8652 契约加严）。
> 总 PASS 494，通过率 87.3%

# gfrun 执行结果汇总 — 2026-09-18

## 验证环境

| 组件 | 分支/版本 | Commit |
|---|---|---|
| PTO ISA 规范 / pto-spec | `main`（v0.58.6 + 已接受修正案） | `b541bbdd` |
| gfrun / SuperScalarModel-asl | `fix/gfrun-717-pto291-subview-ordering` | `ec5ee047` |
| llvm-project | `dev-llvm15_56` | `1037cc1cd` |
| Linx-TileOP-API | `main` | `3be8652` |
| linx-toolchain-build | `main` | `e6a31ef` |
| SuperNPUBench | `main` | `2f272c2e` |

编译器按 **AGENTS.md** 指定用主 `linx-toolchain-build` worktree：clang 15.0.4（clang hash `1037cc1cd31c80e5493ffd7851f7b35a2f8dd79c`），target `linx64v5-unknown-linux-musl`。ISA 规范 **PTO v0.58.6 + 已接受修正案**（pto-spec `b541bbdd`，`v0.58.6.0-35`，本轮新增 #322 packed FP4/E6M2 TCVT 语义、#325 relaxed-consistency 内存模型、#329 E6M2 TCVT profile 路由）。TileOP-API `697f5d8`→**`3be8652`**（`linxisa-v0.58.0-194`，+11：`3be8652` prefix-view 重载约束 #154、`bcaeddf` B.ASSEMBLE WriterSizeCode #145、`8e7a6b9` 跨 TEPL 源消费 prefix views #147/#150、`205cea8` Shared subview 父恢复 #148、`a5f1136` Vec CUBE tile 父 #141、`977547d` 原子 RMW gather/scatter #149 等）。gfrun 用 **SuperScalarModel-asl worktree**，分支从 `feat/gfrun-pto-311-cube-reduction-geometry`（`b00ed95c`）切到 **`fix/gfrun-717-pto291-subview-ordering` `ec5ee047`**（该分支已合并 PTO #311 分支 #677，含 28 个修复提交：Local layout 契约适配、B.ASSEMBLE parent/RegSrc 建模、行归约物理列保持、PTO #291 前向适配器等；落后模型 main 22 提交——MGATHER_MASK/PMU/swimlane 类，与本轮无关）。**本地代码算子**：fa_lowp 重构（TMULS 直接消费 prefix view、M32 E8M0 scale、recip 路径开关）+ 新增 fa_gmma_opt / fa_gmma_dynamic / fa_lowp_recip 测例（`2f272c2e`）。执行：`gfrun -f <elf>`（**无 `-t`**，ERROR 级日志 ~1–2KB/ELF），kernel 算子加 `-s softcore.multiThreadNum=4`，单 ELF 420s 超时 + 2GB 早停，**零日志落盘**（内存 64KB 滚动窗口分类）。PASS = `Reach the End of Benchmark` + `R2 = 0`。

## 关键变更（09-16→09-18）

**1. TileOP-API `697f5d8`→`3be8652`（+11 提交，本轮核心变化）**
- prefix-view 重载约束到 reduction-prefix views（#154）、跨 TEPL 源族消费 reduction-prefix views（#147/#150）、B.ASSEMBLE reduction 会话 WriterSizeCode（#145）、Shared/unit-correct subview 父恢复（#148）、Vec CUBE tile 作 subview 父（#141）。
- **编译面影响**：fixp MX 家族 ×10（mx/mx_s8/gemv_mx 系列）不再编译（MX scale tile 契约断言：E8M0 Vec tile 载体形状/CUBE_M 布局）；matmul_quantize ×2 编译失败叠加——`mxquant::InputTile`/`BfFullTile` 被远端 mxquant 重构（`2b2a37f`）改名，集成待修。
- matmul compile.all 的 `set -euo pipefail` 使 quantize 失败级联跳过 hif4_l1（单独验证可编译并纳入回归）。

**2. gfrun 切换 `fix/gfrun-717-pto291-subview-ordering`**
- PTO #291 subview ordering 修复分支（已并入 PTO #311）：Local layout 契约、B.ASSEMBLE 建模、行归约物理列等 28 提交。
- **正面**：fa_2d_unroll_gmma 保持 15/15 全过；fa_fixpipe 失败原因从 "PTO #311 destinationShape requiredBytes" 变为 "!SelectsPto291CubeMLayout（PTO #291 direct CUBE-M layout）"——断言类别迁移，反映 fixpipe 走上了新的 CUBE-M 直连路径检查。
- **负面（10 个回归，全部模型侧同根因）**：normalization ×8 + reduction reducesum/reducemax_row ×2，全部 `validCol/physicalCol >= validCol` Local tile descriptor 断言——fix 分支 PTO #291 Local layout 契约变化与 TileOP 3be8652 新编码的组合效应。已核对模型 main 缺失的 22 提交（MGATHER_MASK/PMU/swimlane）与本回归无关。
- mega_moe_sim（非 mt）>2GB FAIL→PASS：无 trace 模式下可在时限内跑完。

**3. 本地 fa 改动验证（`2f272c2e` + 修正提交）**
- fa_lowp 重构（prefix 直接消费、M32 E8M0 scale carrier、recip 开关）编译通过；运行 FAIL（模型 TCVT srcTile descriptor 断言，与 kchains MXFP4 同类）。
- **fa_gmma_opt 新测例 1/1 PASS**；**fa_lowp_recip 1/2**：Sq256/Skv256 **PASS**（TRECIP 路径），Sq128/Skv8192 FAIL（模型 B.ASSEMBLE descriptor 契约——深 KV 循环 512 次 TASSEMBLE，issue #717 修复区）；fa_gmma_dynamic 编译器在 Simple Register Coalescing 崩溃，暂不入矩阵（compile.all 注释记录）。
- **fa_lowp_recip 构建事故与修复**：fa_lowp 与 fa_lowp_recip 曾共用 `fa_lowp.o`，Makefile.common 的 clean 按 TESTCASE 名删对象（`${TESTCASE}.o`），对象名 ≠ TESTCASE 名时 clean 失效、make 又不感知 -D 宏变化 → 首轮回归中的 "fa_lowp_recip" 实为无 recip 宏的 TSUB 路径代码（假 FAIL）。修复：独立编译单元 `src/fa_lowp_recip.cpp`（对象名回归 TESTCASE 名，clean 重新生效），两 shape 干净复测确认。
- kchains 默认 PVChainK 32→Tk（单链/每 KV 块一条链，`fa_gmma_kchains.cpp` + Makefile），32 链形式经 compile.all 显式 PVChainK=32 pass 保留。

**4. fa 矩阵口径（沿 09-16 之后的调整）**
- Sq256/Skv256 三组已移除（`eb37e16a`）；本轮 fa 86 ELF：Sq1024 / Sq128-Skv8192 / Tk256(Sq256-Skv512) 三组 × 5 模式 × {unroll, fixpipe, subview} + kchains（单链 + 32 链两配置）×6 模式 + lowp 1 + lowp_recip 2（Sq256 基线 + Sq128/Skv8192 长序列）+ gmma_opt 1。

## 总体结果

| 范围 | ELF 数 | PASS | FAIL | TIMEOUT | 通过率 |
|---|---:|---:|---:|---:|---:|
| microbenchmark | 414 | 394 | 20 | 0 | 95.2% |
| one-level-arch (kernel) | 128 | 92 | 36 | 0 | 71.9% |
| solution | 24 | 8 | 16 | 0 | 33.3% |
| **合计** | **566** | **494** | **72** | **0** | **87.3%** |

> vs 09-16（脚本化逐名对比）：−14 ELF（580→566），+11 PASS（483→494），−25 FAIL（97→72），通过率 83.3%→87.3%。同名 521：**PASS→FAIL ×10**（normalization ×8 + reduction row ×2，模型侧 validCol 断言）；**FAIL→PASS ×1**（mega_moe_sim 非 mt）。仅 09-16 有 59（P18/F41）：fa Sq256/Skv256 组 47、matmul_quantize ×2（mxquant 重构）、fixp MX ×10（全为 09-16 运行 FAIL）。仅 09-18 有 45（P38/F7）：kchains 单链 21 + 32 链重配 21（P18/F3 各）、fa_gmma_opt 1、fa_lowp_recip 2（P1/F1）。

## 算子通过率

| 算子族 | ELF | PASS | FAIL | 通过率 | 说明 |
|---|---:|---:|---:|---:|---|
| micro/scalar | 124 | 124 | 0 | 100% | 全过 |
| micro/vector | 170 | 152 | 18 | 89.4% | compare/select TSTORE（与 09-16 一致） |
| micro/fixp | 86 | 84 | 2 | 97.7% | MX 家族 ×10 不再编译（TileOP 3be8652 契约）；运行 FAIL 仅余 2（09-16 为 12） |
| micro/memory + cube | 34 | 34 | 0 | 100% | 全过 |
| one-level/fa | 86 | 53 | 33 | 61.6% | **unroll_gmma 15/15**；kchains 单链 18/21 + 32 链 18/21（MXFP4 ×3×2 模型 TCVT）；**gmma_opt 1/1（新）**；**lowp_recip 1/2（Sq256 PASS / 长序列 B.ASSEMBLE）**；fixpipe 0/15（PTO #291 CUBE-M）；subview 0/10（TROWMAX）；lowp 0/1（TCVT） |
| one-level/matmul | 31 | 30 | 1 | 96.8% | hif4_l1 FAIL（RawTileSourceFits，同 09-16）；quantize ×2 编译失败（mxquant 重构集成断裂） |
| one-level/reduction | 4 | 2 | 2 | 50% | **回归**：reducesum/reducemax_row FAIL（validCol，模型侧）；cumsum/reduceprod 过 |
| one-level/{broadcast,concat,conv2d,element_wise,gather,vec} | 7 | 7 | 0 | 100% | 全过 |
| solution/normalization | 8 | 0 | 8 | 0% | **回归**：全灭（validCol/physicalCol Local layout 断言，模型侧） |
| solution/{moe_dispatch,moe_combine} | 4 | 4 | 0 | 100% | 全过 |
| solution/mega_moe | 2 | 2 | 0 | 100% | **修复**：非 mt 变体无 trace 下跑通 |
| solution/{group_token_old,group_token_vec} | 4 | 2 | 2 | 50% | mt 变体 PASS；非 mt rc=0 无结束标记（~1KB 输出即退，行为变化待查） |
| solution/{gather_v2,view_copy} | 6 | 0 | 6 | 0% | R2=1（09-14 起持续） |

## 编译覆盖

编译产出 544 ELF（414 micro + 106 kernel + 24 solution）。编译失败：
- **matmul_quantize ×2**（远端 mxquant 重构 `2b2a37f` 改名 `mxquant::InputTile`/`BfFullTile`，matmul_quantize.hpp:79/182 引用断裂；叠加 TileOP 新增 TCVT CUBE-source 约束）——**SuperNPUBench 侧集成待修**。
- **fixp MX 家族 ×10**（TileOP 3be8652 契约加严：E8M0 scale tile Vec 载体/CUBE_M 布局断言）。
- **fa_gmma_dynamic**（编译器 Simple Register Coalescing 崩溃，已记录，未入矩阵）。
- micro 既有 4 例（Bias CUBE_M ×3、MGATHER_CAS 传输宽度 ×1）、fa HIF4/MXFP4/FP8_VECBF16 低精度模式（TileOP 契约）、subview Tk=256（LinxV5 后端 64KB 帧上限崩溃 `RegSize >= 512 && RegSize <= 64*1024`）——与 09-16 相同。

## 运行失败清单

| 失败原因 | 数量 | 范围 | 说明 |
|---|---:|---|---|
| PTO #291 direct CUBE-M layout | 15 | one-level/fa | fa_fixpipe 全模式（失败类别从 09-16 的 PTO #311 destinationShape 迁移） |
| illegal TROWMAX operand | 10 | one-level/fa | fa_subview（持续） |
| validCol/physicalCol Local descriptor | 10 | solution + kernel | **新增回归**：normalization ×8 + reduction row ×2，模型 fix 分支 PTO #291 契约 |
| vector compare/select TSTORE | 18 | micro/vector | 09-10 起持续 |
| TCVT srcTile descriptor（gfrunPto291Tcvt） | 7 | one-level/fa | kchains MXFP4 ×6（单链 3 + 32 链 3）+ fa_lowp ×1（模型侧；**注意**：U8→E8M0 reinterpret TCVT 类型等价待规范/模型确认，pto-spec #322/#329 区域） |
| illegal B.ASSEMBLE descriptor | 1 | one-level/fa | lowp_recip Sq128/Skv8192（深 KV 循环 512 次 TASSEMBLE，模型侧 issue #717 修复区） |
| R2=1 result mismatch | 6 | solution | gather_v2 ×3 + view_copy ×3（09-14 起持续） |
| RawTileSourceFits | 1 | one-level/matmul | matmul_hif4_l1_quantize（持续） |
| rc=0 无结束标记 | 2 | solution | group_token_old/vec 非 mt（~1KB 输出即退，行为变化，待模型侧确认） |
| fixp 残余 | 2 | micro/fixp | 09-16 为 12，10 个 MX 转为编译失败 |

## 本次更新要点

- **回归主线 = 模型 fix 分支验证**：gfrun `fix/gfrun-717-pto291-subview-ordering` 是为了修 PTO #291 subview ordering；10 个 normalization/reduction 回归全部是其 Local layout 契约变化所致（同根因），fa_fixpipe 断言类别迁移也源于此——这些是修复分支的已知工作面，非算子代码回归。
- **TileOP 3be8652 双刃**：fa_2d_unroll_gmma / kchains 主路径全部通过（prefix-view 新契约验证 OK）；fixp MX ×10 与 matmul_quantize ×2 编译面受损（后者是 mxquant 重构集成断裂，SuperNPUBench 侧需跟进）。
- **本地 fa 代码零回归落地**：unroll_gmma 15/15、kchains 双配置 36/42（仅模型侧 MXFP4）、fa_gmma_opt 新测例即过、**fa_lowp_recip Sq256 PASS**；fa_lowp（TCVT U8→E8M0 reinterpret）与 lowp_recip 长序列（B.ASSEMBLE 深循环会话）的 FAIL 均为模型侧。
- **fa_lowp_recip 陈旧对象事故**：首轮回归的 lowp_recip FAIL 是共享 `fa_lowp.o` 伪影（clean 按 TESTCASE 名删对象失效 + make 不感知 -D 变化），已用独立 TU 修复并干净复测；仅此一个 ELF 受影响，其余多配置构建对象名与 TESTCASE 名一致不受影响。
- mega_moe_sim 非 mt 在无 trace 模式下修复（>2GB 输出问题是 trace 产物放大）。
- group_token 非 mt 双变体出现 rc=0 无标记早退（新行为，需模型侧确认）。
- micro 通过率 92.9%→95.2%（fixp MX ×10 转编译失败，运行 FAIL 30→20）。

## 与 09-16 基线的差异

| 类别 | 09-16 (ELF,P,F) | 09-18 (ELF,P,F) | 变化 |
|---|---|---|---|
| microbenchmark | 424, 394, 30 | 414, 394, 20 | −10 ELF（fixp MX 编译失败，其 09-16 全为运行 FAIL）→ FAIL −10 |
| one-level/fa | 88, 32, 56 | 86, 53, 33 | Sq256/Skv256 组移除 −47（其中 kchains 旧 Sq256 基线 2）；kchains 新增 42（单链 21 + 32 链 21，P36/F6）；gmma_opt +1 P；lowp_recip +2（P1/F1）；同名 41 持平（unroll 15/fixpipe 15/subview 10/lowp 1） |
| one-level/matmul | 33, 31, 2 | 31, 30, 1 | quantize ×2 编译失败（−1 P −1 F） |
| one-level/reduction | 4, 4, 0 | 4, 2, 2 | **回归 ×2**（validCol，模型侧） |
| one-level 其他 kernel | 7, 7, 0 | 7, 7, 0 | 持平 |
| solution | 24, 15, 9 | 24, 8, 16 | **回归 ×8**（normalization，模型侧）+ 修复 ×1（mega_moe_sim）− 行为变化 ×2（group_token 非 mt） |
| **合计** | **580, 483, 97** | **566, 494, 72** | 同名 521：PASS→FAIL ×10（模型侧）；FAIL→PASS ×1（mega_moe_sim）；移除 59（P18/F41）；新增 45（P38/F7） |

---

# 历史验证记录

> 早期每日基线仅保留环境版本与总量，供复现与趋势对比；完整分类/失败明细已归档。

| 日期 | gfrun (SuperScalarModel) | llvm / TileOP-API | 工具链 | ELF | PASS | FAIL | T/O | 通过率 | 关键变化 |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| 09-18 | asl `fix/gfrun-717-pto291-subview-ordering` `ec5ee047`；pto-spec `b541bbdd`（v0.58.6+修正案） | `1037cc1cd` / `3be8652` | AGENTS.md 主 worktree（PTO v0.58.6 + 已接受修正案） | 544 | 475 | 69 | 0 | 87.3% | TileOP 697f5d8→3be8652（+11，prefix-view 契约）；gfrun 切 PTO #291 fix 分支；本地 fa_lowp 重构 + fa_gmma_opt/gmma_dynamic/lowp_recip 新测例（2f272c2e）；fa 34 PASS（unroll 15/15、gmma_opt 1/1 新增即过）；**10 回归全模型侧**（normalization ×8 + reduction row ×2，validCol Local layout 断言）；matmul_quantize ×2 编译失败（mxquant 重构集成断裂）；fixp MX ×10 编译失败（TileOP 契约加严）；mega_moe_sim 非 mt 修复 |
| 09-16 | asl `feat/gfrun-pto-311-cube-reduction-geometry` `b00ed95c`；pto-spec `86f46079`（v0.58.6+修正案） | `1037cc1cd` / `697f5d8` | AGENTS.md 主 worktree（PTO v0.58.6 + 已接受修正案） | 580 | 483 | 97 | 0 | 83.3% | gfrun B.ASSEMBLE 支持（b00ed95c）→ HIF4/HIF8 18 ELF 首次纳入（7 PASS）；fa Shared-B TransB 契约修正（1f4425d）fa 32/32 PASS 零回归；单线程套件退役（−31 ELF，重构 79e492a）；matmul_quantize_FP8_ASM1 被 B.ASSEMBLE 检查拦截（PASS→FAIL）；concat_scatter 420s 直接 PASS |
| 09-15 | asl `feat/gfrun-pto-311-cube-reduction-geometry` `10dd099f` | `1037cc1cd` / `697f5d8` | AGENTS.md 主 worktree（PTO v0.58.6） | 593 | 500 | 93 | 0 | 84.3% | gfrun 分支切换（PTO #311 CUBE reduction geometry + binary reduction-prefix subview 修复）；TileOP fix/issue-138-reduction-prefix-subview；fa_2d_unroll_gmma TCVT→TREDUCEPREFIXVIEW 零拷贝行归约；validCol TSTORE 28→1 FAIL（gfrun 修复）；fa_fixpipe 新增 24 FAIL（PTO #311 destinationShape）+ fa_subview 20 FAIL（TROWMAX）；mt/fa compile.all 扩展 38→71 ELF（+fa_gmma_kchains ×2）；normalization +8（全 PASS）；dynamic_mx_quant 移除（−14）；reduction 4→0 FAIL；fixp 16→12 FAIL |
| 09-14 | asl `fix/gfrun-shared-tmatmul-layout-257` `ad972d21` | `4a3e0bdb5` / `987d034` | AGENTS.md 主 worktree（PTO v0.58.6） | 579 | 486 | 93 | 0 | 83.9% | gfrun 分支切换（Shared TMATMUL layout）；TileOP 加严矩阵 shape 校验→单线程 fa/matmul 19 ELF 编译失败；新增 fa_gmma_kchains(+2)、matmul_blockM/kchains/lowp(+19)、dynamic_mx_quant(+14)；gather_v2/view_copy 回归 R2=1(−6 PASS)；输出截断 500MB→2GB；compile.all set -euo→set -u |
| 09-11 | asl `codex/gfrun-pto-0586-asl` `e7d883c9` | `0a141cbd2` / `e93ef98` | AGENTS.md 主 worktree（PTO v0.58.6） | 569 | 487 | 82 | 0 | 85.6% | 编译器更新（llvm B.DATR CCTRL 消歧 + TileOP reinterpret_tile）；全量重编译 584 ELF；fixp +2 PASS（18→16 FAIL）；新增 solution matmul_test ×3（FAIL）；dynamic_mx_quant compile FAIL；A16W4 matmul ×3 全部独立验证 PASS（1.07–1.36GB）；group_token_old_mt 独立验证 PASS |
| 09-10 | asl `codex/gfrun-pto-0586-asl` `e7d883c9` | `553b08045` / `b8669ce` | AGENTS.md 主 worktree（PTO v0.58.6） | 566 | 484 | 82 | 0 | 85.5% | gfrun 切换 asl worktree；multi_thread/fa 编译扩展（5→36，set -e 移除）；fa_subview 12 PASS；fa_2d_unroll_gmma/fa_fixpipe 24 FAIL（2048B 行归约限制）；vector +20 FAIL（compare/select TSTORE）；HIF4/HIF8 排除；fa_2d_unroll_gmma_subview 新算子；solution 树 16 ELF（11 PASS / 5 FAIL） |
| 09-08 | fix/issue-558 `07e9c661` | `553b08045` / `b8669ce` | AGENTS.md 主 worktree（PTO v0.58.5） | 470 | 438 | 32 | 0 | 93.2% | FA/matmul CUBE 迁移（matmul +13 编译）；HIF8 convert 修复；MX scale 对齐 PTO 暴露 HIF4 断言（fixp +14 FAIL）；deepseek 移出 compile_all；solution 排除；−40 PASS vs 09-04 |
| 09-04 | codex `bc7fae00` | `1ae4ee39` / `804eb03` | AGENTS.md 主 worktree（CUBE 强制） | 496 | 478 | 18 | 0 | 96.4% | CUBE subview 行归约 + fixpipe GroupMax/RowMaxIn 修复（fa/flashMLA/reduction +14、fixp +98）；deepseek CUBE 编译修复；mt 新增 8 算子；+129 PASS / 0 回归 vs 08-27 |
| 08-27 | codex `d8903938` | `adcb8794` / `f94bc12` | AGENTS.md 主 worktree（CUBE 强制） | 427 | 349 | 78 | 0 | 81.7% | CUBE cell-layout 强制（matmul/deepseek 编译回归）；dataType 断言回归（fa/flashMLA/reduction −14）；cube +9、mt/matmul lowp +5；净 −17 vs 08-23 |
| 08-23 | exp `a5dca25a` | `611105f2b` / `a795b973020d` | blessed-latest（ADR 0069） | 437 | 366 | 69 | 2 | 83.8% | blessed 编译器+exp 模型正确配对；+27 PASS（fa 全过、mt/matmul lowp 部分）；13 编译失败 |
| 08-21 | main `7b691d4d` | `a84c4d10a` / `ffa257738f` | toolchain-build（32KB shared） | 437 | 339 | 98 | 0 | 77.6% | 老 compiler+main 模型；multi_thread 大 tile 首编（bf16/fp16/lowp 运行 FAIL） |
| 08-20 | exp `5a64c34d` | `b945a5d0` / `c02dae65` | blessed-latest | 433 | 344 | 89 | 0 | 79.4% | 零步幅 raw tile spill 修复 4 例；fa `sfa`×2 编译回归（TMATMUL 形状契约） |
| 08-19 | `01f9ec10` | `86959776b` / `8b2ee78` | — | 434 | 341 | 92 | 1 | 78.6% | 4 例 broadcast/GELU FAIL→PASS；mt/matmul 1 PASS→FAIL |
| 08-18 | `a68dba29` | — / `8b2ee78`（TileDType 修复） | — | 429 | 321 | 108 | 2 | 74.8% | 首次全量基线；fa/flashMLA/reduction 由 FAIL 恢复；fixp 27→4（TileDType 暴露契约偏差） |

**跨版本要点**：

- **09-18 TileOP 3be8652 + PTO #291 fix 分支验证 + 本地 fa 变体**：TileOP `697f5d8`→`3be8652`（+11：prefix-view 重载约束 #154、B.ASSEMBLE WriterSizeCode #145、跨 TEPL prefix views #147/#150、subview 父修复 #148/#141）。gfrun 切 `fix/gfrun-717-pto291-subview-ordering` `ec5ee047`（已并入 PTO #311 分支）。本地 fa_lowp 重构 + fa_gmma_opt（新测例 1/1 PASS）/fa_gmma_dynamic（编译器崩溃未入矩阵）/fa_lowp_recip（`2f272c2e`）。fa_2d_unroll_gmma 15/15、kchains 18/21（MXFP4 ×3 模型 TCVT）零代码回归；fa_fixpipe 失败类别从 PTO #311 destinationShape 迁移为 PTO #291 CUBE-M。**10 个 PASS→FAIL 全部模型侧同根因**（normalization ×8 + reduction reducesum/reducemax_row ×2，`validCol/physicalCol` Local descriptor 断言——fix 分支 Local layout 契约变化；已排除模型 main 缺失 22 提交的影响）。mega_moe_sim 非 mt 修复（无 trace 模式可跑完）。编译面：matmul_quantize ×2 挂于远端 mxquant 重构的类型改名（集成待修），fixp MX ×10 挂于 TileOP 契约加严（其 09-16 运行时本就 FAIL，micro 通过率 92.9%→95.2%）。
- **09-16 HIF4/HIF8 解锁 + fa TransB 契约修正 + 单线程退役**：gfrun `10dd099f`→`b00ed95c`（model Local B.ASSEMBLE parent references），09-10 起被排除的 18 个 HIF4/HIF8 ELF 首次纳入回归：7 PASS（fa_2d_unroll_gmma HIF8 ×6、matmul_lowp_HIF4X2 ×1）/ 11 FAIL（fa_fixpipe HIF8 ×6、fa_subview HIF8 ×5，失败原因与非 HIF 版本相同）。SuperNPUBench `1f4425d` 修正 fa Shared-B TransB 存储契约（TransB=0 声明物理 [N,K]；QK 不再 transpose_b、kchains PV 改 TransB=0），fa_2d_unroll_gmma 30/30、fa_gmma_kchains 2/2 全 PASS 零回归。单线程套件随 single_thread 退役移出编译（−31 ELF / −23 PASS / −8 FAIL，含 hashtable_lookup ×6、topk、broadcast vec_07 half）。matmul_quantize_FP8_ASM1 被 gfrun 新增 B.ASSEMBLE descriptor 检查拦截（PASS→FAIL）。基线口径变化：通过率 84.3%→83.3% 主要由 −31/+18 口径贡献；可比集合与 09-15 完全一致。
- **09-15 gfrun PTO #311 CUBE reduction + TileOP zero-copy prefix views**：gfrun 从 `fix/gfrun-shared-tmatmul-layout-257`(`ad972d21`) 切换到 `feat/gfrun-pto-311-cube-reduction-geometry`(`10dd099f`，PTO #311 CUBE reduction geometry 适配 + binary reduction-prefix subview 修复)。TileOP-API `987d034`→`697f5d8`（fix/issue-138-reduction-prefix-subview，zero-copy reduction prefix views）。fa_2d_unroll_gmma 从 TCVT tile 拷贝改为 **TREDUCEPREFIXVIEW** 零拷贝行归约 prefix view（统一 TMAX、TMUL+TADD 替代 TFMA），24/24 PASS。gfrun 修复 validCol TSTORE 断言（28→1 FAIL，09-14 fa_2d_unroll_gmma + fa_fixpipe 24 FAIL → 09-15 仅 fa_lowp MXFP4 1 FAIL），reduction validCol/TROWSUM 全修复（4→0 FAIL）。但 gfrun 新增 PTO #311 destinationShape 断言→ fa_fixpipe 24 FAIL（`inst->dsts[0]->size >= destinationShape.requiredBytes`），fa_subview 20 FAIL（`illegal TROWMAX operand`）。multi_thread/fa compile.all 大幅扩展 38→71 ELF（6 config × 8 mode × 3 case + fa_gmma_kchains ×2），fa_2d_unroll_gmma 24 PASS、fa_gmma_kchains 2 PASS（新 config 全过）。新增 normalization ×8（全 PASS）、matmul_quantize ×2（PASS）。dynamic_mx_quant 移除（−14 ELF）。fixp 改善（16→12 FAIL）。net: 486→500 PASS，93→93 FAIL，83.9%→84.3%。
- **09-14 gfrun 分支切换 + TileOP 加严校验**：gfrun 从 `codex/gfrun-pto-0586-asl`(`e7d883c9`) 切换到 `fix/gfrun-shared-tmatmul-layout-257`(`ad972d21`)，Shared TMATMUL layout 适配。TileOP-API `e93ef98`→`987d034` 加严矩阵 shape 校验（`AValidCols==BValidRows`、D valid shape、MX ScaleB shape）→ 单线程 fa_2d_unroll/sfa ×10 + matmul MASK_FP16/FP8/A16W4 ×9 编译失败（09-11 全 PASS，仅 MASK_FP32 ×3 + fa_softmax_pto ×2 存活）。gfrun 分支切换致 solution gather_v2 ×3 + view_copy ×3 回归 R2=1（结果不匹配）。新增 fa_gmma_kchains(+2 PASS)、matmul_blockM/kchains/lowp_blockM(+19 PASS)、dynamic_mx_quant(+14，2 PASS / 12 FAIL)。输出截断 500MB→2GB，A16W4/concat_scatter/group_token_old_mt 不再需独立验证。compile.all `set -euo pipefail`→`set -u`（HIF4 失败不再阻塞后续变体）。net: 487→486 PASS，82→93 FAIL，85.6%→83.9%。
- **09-11 编译器更新**：llvm-project `553b08045` → `0a141cbd2`（B.DATR CCTRL union 消歧，PTO-ISA #236），Linx-TileOP-API `b8669ce` → `e93ef98`（reinterpret_tile 文档 + 回归 fixture）。全量重编译 584 ELF。fixp +2 PASS（MX scale dataType 断言 18→16 FAIL，编译器修复）。新增 solution matmul_test ×3（输出 >500MB FAIL，疑似大输出/死循环）。dynamic_mx_quant 编译失败（09-10 为 FAIL，09-11 排除）。A16W4 matmul ×3 因 500MB 输出截断初始 FAIL/TIMEOUT，独立验证全部 PASS（1.07–1.36GB 输出，09-10 均 PASS）。group_token_old_mt 独立验证 PASS（662MB 输出，500MB 截断误判 FAIL）。gfrun 同 09-10（asl `e7d883c9`），无 gfrun 侧变化。
- **09-10 gfrun 切换 asl worktree**：gfrun 从 `fix/issue-558-fp4-rne-zero`(`07e9c661`) 切换到 `codex/gfrun-pto-0586-asl`(`e7d883c9`)，PTO v0.58.5→v0.58.6。asl worktree 强制 PTO v0.58 行归约 source ≤2048 字节限制 → fa_2d_unroll_gmma/fa_fixpipe 24 FAIL（main gfrun 不强制此限制，09-08 全 PASS）；fa_subview 用 TPARTVIEW 分块规避，12/12 PASS。vector compare/select TSTORE 断言 +20 FAIL（asl gfrun 行为差异）。multi_thread/fa compile.all 移除 set -e → 5→36 ELF。gfsim 不支持 B.SUBVIEW 指令，fa_2d_unroll_gmma_subview 无法获取时序数据。solution 树 16 ELF 纳入验证（11 PASS / 5 FAIL——4 例 gfrun 超时疑似算子无限循环、1 例 validCol 断言）。
- **09-08 MX scale 对齐 PTO 暴露 HIF4 断言**：gfrun `970ce7af` MX scale 处理对齐 PTO 后，HIF4 MX scale dataType 断言在 fixp（+14 FAIL）/ matmul（+4）/ fa（+1）暴露——09-04 这些全 PASS。叠加 fixp shared 15 例编译失败（TileOP B.DATR/PreQuant 契约）→ fixp 118→89 PASS（−29）。FA/matmul CUBE 迁移（`334737e`）正面抵消：matmul +13 编译 / +9 PASS。
- **09-08 HIF8 convert 修复**：gfrun `24d26eeb` 实现 HIF8 tile 转换 → multi_thread/fa HIF8 FAIL→PASS（09-04 `.fs→.hifb` convert 未注册消除）。但 FP8_VECBF16 编译失败（TileOP dtype 断言）+ set -e 级联 → mt/fa 7→5 ELF。
- **09-08 deepseek 移出 compile_all.sh**：MC2 算子重构进 solution 树，deepseek 21 ELF 不在本轮范围（−17 PASS / −4 FAIL）。solution 树 15 ELF 本轮排除（用户指定）。
- **09-04 CUBE subview 行归约 + fixpipe 修复**：gfrun `bc7fae00` 修复 CUBE subview 行归约 → fa/flashMLA/reduction 全恢复（+14）；fixpipe GroupMax/RowMaxIn → fixp 43→4 FAIL（+98 PASS）。与 08-27 单一变量（仅版本更新，编译器 worktree 不变），+129 PASS / 0 回归，通过率 81.7%→96.4%。
- **ADR 0069 编码配对**（08-21 ↔ 08-23）：版本匹配则高 PASS，错位则骤降。08-21 老 compiler+main 模型（均无 ADR 0069）= 匹配 → 339P；08-23 blessed+exp（均有 ADR 0069）= 匹配 → 366P；而 blessed compiler+main 模型（编译器领先、模型落后）= 错位 → 仅 124P（262 个 `reserved/deleted TEPL selector`：store 的 SizeCode=0 被旧模型误读为 0B 目的）。
- **08-27 切回 AGENTS.md 主 worktree**（非 blessed-latest）：gfrun codex `d8903938`、TileOP `f94bc12`（CUBE cell-layout 强制）。与 08-23 非单一变量对比；09-04 在此基础上仅版本更新（无 worktree 切换），+129 PASS / 0 回归。
- **持续模型侧限制**（跨基线不变）：fixp MX scale dataType 断言（09-08 起 18→16→12 FAIL，09-15 部分修复后持平）、vector compare/select TSTORE（18 FAIL，09-10 起）、fa_fixpipe PTO #311 destinationShape（24→30 FAIL，含 HIF8）、fa_subview TROWMAX（20→25 FAIL，含 HIF8）。gather_v2/view_copy R2=1（6 FAIL，09-14 gfrun 分支切换致回归，持续）。09-16 新增：matmul_quantize_FP8_ASM1 B.ASSEMBLE descriptor（1 FAIL）。输出 >2GB（3 ELF：group_token_old/group_token_vec 非 mt + mega_moe_sim，疑似死循环/大输出；单线程 control/sort/topk 已随 single_thread 退役，hashtable_lookup ×6 / broadcast vec_07 half / topk 不再在编译范围）。
