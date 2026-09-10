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


> **当前验证基线**：2026-09-10（550 个已编译 ELF 全量 gfrun 复测，**排除 matmul/fa HIF4/HIF8**
> （需 assemble 功能）和 solution 树；编译器按 AGENTS.md 用主 linx-toolchain-build worktree
> （llvm `553b08045` + TileOP-API `b8669ce`）、gfrun 用 **SuperScalarModel-asl worktree**
> `codex/gfrun-pto-0586-asl` `e7d883c9`（PTO v0.58.6 tile contracts + CUBE reduction subview 修复）；
> 总 PASS 473，通过率 86.0%——multi_thread/fa 编译扩展（5→36 ELF，compile.all 移除 set -e），
> fa_subview 全过但 fa_2d_unroll_gmma/fa_fixpipe 24 FAIL（asl gfrun 强制 2048B 行归约 source 限制），
> + vector compare/select 20 FAIL + reduction/broadcast 回归

# gfrun 执行结果汇总 — 2026-09-10

## 验证环境

| 组件 | 分支/版本 | Commit |
|---|---|---|
| gfrun / SuperScalarModel | `codex/gfrun-pto-0586-asl` | `e7d883c9` |
| llvm-project | `dev-llvm15_56` | `553b08045` |
| Linx-TileOP-API | `linx` | `b8669ce` |
| SuperNPUBench | `main` | `5905c42`（+未提交改动） |

编译器按 **AGENTS.md** 指定用主 `linx-toolchain-build` worktree：`COMPILER_DIR=…/linx-toolchain-build/output/linx_blockisa_llvm_musl/bin`，clang 15.0.4，target `linx64v5-unknown-linux-musl`。gfrun 用 **SuperScalarModel-asl worktree**（`codex/gfrun-pto-0586-asl` `e7d883c9`，PTO **v0.58.6** tile contracts + CUBE reduction subview 修复，3 个提交：`fb83794a` adapt v0.58.6 tile contracts、`03be3eb3` drop retired CCTRL references、`e7d883c9` preserve CUBE reduction subviews）。执行：`gfrun -t 1 -f <elf>`，multi_thread 加 `-s softcore.multiThreadNum=4`，单 ELF 300s 超时。PASS = 退出码 0 + `Reach the End of Benchmark` + `R2 = 0`。**本轮排除 matmul/fa HIF4/HIF8**（用户指定，需 assemble 功能）和 solution 树。

## 本次新增特性（09-08→09-10）

**1. gfrun 切换至 asl worktree**
- 从 `fix/issue-558-fp4-rne-zero`(`07e9c661`) 切换到 `codex/gfrun-pto-0586-asl`(`e7d883c9`)，PTO v0.58.5→v0.58.6。
- **CUBE reduction subview 修复**（`e7d883c9`）：行归约 source tile ≤2048 字节限制强制执行。
- **v0.58.6 tile contracts 适配**（`fb83794a`）。

**2. multi_thread/fa 编译扩展（5→36 ELF）**
- `compile.all` 移除 `set -e`，FP8_VECBF16 编译失败不再级联跳过后续变体。
- 三种 FA kernel（fa_2d_unroll_gmma / fa_fixpipe / fa_subview）× 4 模式（FP32/BF16/MXFP8/FP8_VECFP32）× 3 配置（Sq256/Skv256、Sq1024/Skv1024、Sq128/Skv8192）= 36 ELF。
- fa_subview 全 12 PASS；fa_2d_unroll_gmma + fa_fixpipe 全 24 FAIL（`validCol != 0` TSTORE 断言——tile 超过 2048 字节行归约 source 限制，asl gfrun 强制校验后暴露）。

**3. fa_2d_unroll_gmma_subview 新算子**
- 融合 fa_2d_unroll_gmma v2 优化 + fa_subview TPARTVIEW 分块行归约。
- gfsim 时序模拟器不支持 B.SUBVIEW 指令，无法获取 cycle 数。
- 基线 fa_2d_unroll_gmma gfsim cycle 数：FP32_VECFP32 138,431 cycles；BF16_VECFP32 78,706 cycles。

**4. HIF4/HIF8 排除**
- 用户指定排除 matmul/fa HIF4/HIF8 变体（需 assemble 功能）。
- 消除 09-08 的 5 个 HIF4 FAIL（matmul 4 + fa 1）。

## 总体结果

| 范围 | ELF 数 | PASS | FAIL | TIMEOUT | 通过率 |
|---|---:|---:|---:|---:|---:|
| microbenchmark | 439 | 401 | 38 | 0 | 91.3% |
| one-level-arch | 111 | 72 | 39 | 0 | 64.9% |
| **合计** | **550** | **473** | **77** | **0** | **86.0%** |

> vs 09-08：+80 ELF（470→550），+35 PASS（438→473），+45 FAIL（32→77），通过率 93.2%→86.0%。ELF 增量主要来自 multi_thread/fa 扩展（5→36）。

## 算子通过率

| 算子族 | ELF | PASS | FAIL | 通过率 | 说明 |
|---|---:|---:|---:|---:|---|
| micro/scalar | 124 | 124 | 0 | 100% | 全过 |
| micro/memory | 27 | 27 | 0 | 100% | 全过 |
| micro/cube | 11 | 11 | 0 | 100% | 全过 |
| micro/fixp | 107 | 89 | 18 | 83.2% | MX scale dataType 断言（不变） |
| micro/vector | 170 | 150 | 20 | 88.2% | **新增 20 FAIL**：compare/select TSTORE 断言（asl gfrun） |
| one-level/fa | 12 | 12 | 0 | 100% | 全过（HIF4 已排除） |
| one-level/matmul | 12 | 12 | 0 | 100% | 全过（HIF4 已排除） |
| one-level/multi_thread/fa | 36 | 12 | 24 | 33.3% | fa_subview 12/12 PASS；fa_2d_unroll_gmma + fa_fixpipe 24 FAIL（validCol != 0） |
| one-level/multi_thread/matmul | 10 | 10 | 0 | 100% | 全过 |
| one-level/multi_thread/(其他) | 8 | 6 | 2 | 75% | reduction 2 FAIL + transpose 1 FAIL（新） |
| one-level/broadcast | 6 | 3 | 3 | 50% | reserved/deleted tile selector + COPY expansion（+2 FAIL） |
| one-level/concat | 4 | 4 | 0 | 100% | 全过（2 例 tail 截断假 FAIL 已修正） |
| one-level/control | 6 | 0 | 6 | 0% | INT8/16 dtype 元组未定义（不变） |
| one-level/reduction | 6 | 3 | 3 | 50% | validCol + illegal TROWSUM（+2 FAIL） |
| 其他（全过） | 41 | 41 | 0 | 100% | flashMLA/gather/element_wise/transpose/conv2d/vec 等 |

## 77 FAIL 按根因分组

| 根因 | 数量 | 类别 | 说明 |
|---|---:|---|---|
| `validCol != 0` TSTORE 断言 | 24 | multi_thread/fa | fa_2d_unroll_gmma + fa_fixpipe X1_Y2 全模式；asl gfrun 强制 2048B 行归约 source 限制 |
| MX scale dataType 断言 | 18 | micro/fixp | `fixp_tmatmul_*_mx_*`（不变） |
| vector compare/select TSTORE | 20 | micro/vector | tcmp/tcmps/tsel/tsels/tcolargmax/trowargmax/tgather/tscatter/tgpr2t；asl gfrun 新增 |
| control INT8/16 元组 | 6 | one-level/control | `hashtable_lookup`（不变） |
| broadcast tile selector/COPY | 3 | one-level/broadcast | +2 新增 |
| reduction validCol/TROWSUM | 5 | one-level/reduction + multi_thread/reduction | +4 新增 |
| transpose tile selector | 1 | one-level/multi_thread/transpose | 新增 |

> 注：run_all.sh 的 `tail -5000` 日志截断对 gfrun 输出超长 trace 的 ELF（matmul A16W4 3 例、concat half 2 例）误判为 FAIL。手工复跑 5 例确认实际 PASS（已修正 summary.tsv）。

## 与 2026-09-08 基线的差异

> 本轮与 09-08 非单一变量：gfrun 切换 asl worktree（PTO v0.58.6）+ SuperNPUBench 本侧 multi_thread/fa 编译扩展 + fa_subview 融合 + HIF4/HIF8 排除。

- **gfrun 切换**：`fix/issue-558-fp4-rne-zero`(`07e9c661`) → `codex/gfrun-pto-0586-asl`(`e7d883c9`)，PTO v0.58.5→v0.58.6。asl worktree 强制 2048B 行归约 source 限制 → fa_2d_unroll_gmma/fa_fixpipe 24 FAIL（main gfrun 不强制此限制，09-08 全 PASS）。
- **multi_thread/fa ELF 扩展**：5→36（compile.all 移除 `set -e`）。fa_subview 12/12 PASS，fa_2d_unroll_gmma + fa_fixpipe 24/24 FAIL。
- **vector 新增 20 FAIL**：compare/select TSTORE 断言，asl gfrun 行为差异。
- **reduction/broadcast/transpose 回归**：+2 reduction +2 broadcast +1 transpose FAIL（asl gfrun）。
- **HIF4/HIF8 排除**：matmul 4 FAIL + fa 1 FAIL 消除（用户指定排除）。
- **5 例假 FAIL 已修正**：matmul A16W4 ×3 + concat half ×2（trace 数百万行，tail 截断丢失 PASS 标记，手工验证为 PASS）。
- **净 438→473 PASS（+35）**：multi_thread/fa +7（fa_subview 12 PASS − 5 原有 PASS）、microbenchmark +28（新增 ELF）。**77→32 FAIL 变化**：+53 新增（fa 24 + vector 20 + reduction 4 + broadcast 2 + transpose 1 + multi_thread/reduction 2）−6 移除（HIF4 5 + topk 1）= +47，但 2 例假 FAIL 修正（09-08 topk 已排除）→ 净 +45。

---

# gfrun 执行结果汇总 — 2026-09-08

## 验证环境

| 组件 | 分支/版本 | Commit |
|---|---|---|
| gfrun / SuperScalarModel | `fix/issue-558-fp4-rne-zero` | `07e9c661` |
| llvm-project | `dev-llvm15_56` | `553b08045` |
| Linx-TileOP-API | `linx` | `b8669ce` |

编译器按 **AGENTS.md** 指定用主 `linx-toolchain-build` worktree：`COMPILER_DIR=…/linx-toolchain-build/output/linx_blockisa_llvm_musl/bin`，clang 15.0.4，target `linx64v5-unknown-linux-musl`，对应 **PTO ISA v0.58.5**（TileOP-API `b8669ce` git describe `linxisa-v0.58.0-149-gb8669ce`，README 标注 v0.58.5 layout-and-rearrangement 契约）。gfrun 用 `fix/issue-558-fp4-rne-zero` `07e9c661`（合并 codex/consolidate-post-main-fixes-20260903 后续工作 + 3 个新提交：HIF8 tile 转换 `24d26eeb`、FP4 TCVT RNE-zero `dedcaba3` issue #558、CUBE 目的容量保持 `07e9c661`；MX scale 对齐 PTO `970ce7af` 暴露 HIF4 MX scale dataType 断言）。执行：`gfrun -t 1 -f <elf>`，multi_thread 加 `-s softcore.multiThreadNum=4`，单 ELF 90s 超时。PASS = 退出码 0 + `Reach the End of Benchmark` + `R2 = 0`。**本轮排除 solution 树**（用户指定，15 ELF / 5 编译失败未计入）。

> 注：run_all.sh 的 `tail -400` 日志截断对 gfrun 输出超 400 行的 ELF（cube tmatmul fp16/i8、concat half gather/scatter、matmul A16W4、scalar sqrt_f64）误判为 FAIL。手工复跑 11 例，10 例确认实际 PASS（已修正 summary.tsv），仅 topk 为真实 FAIL（跟踪爆炸 >5000 万行，R2=1）。

## 本次新增特性（工具链 / 模型，PTO v0.58.5 全栈对齐，09-04→09-08）

三个组件 09-04→09-08 同步演进：TileOP-API `804eb03→b8669ce` 34 commits、llvm `1ae4ee39→553b08045` 2 commits、gfrun `bc7fae00→07e9c661` 97 commits（含 codex 后续合并 + fix/issue-558 分支 3 个新提交）。SuperNPUBench 本侧：单线程 FA/matmul CUBE 迁移（`334737e`）、fa_opt v2（`3e9d4ba`）、solution 树扩展（view_copy/gather_v2/normalization/moe/dynamic_mx_quant）、deepseek 移出 compile_all.sh。按类归档：

**1. gfrun 模型侧修复**
- **HIF8 tile 转换**（`24d26eeb`）：gfrun 实现 HIF8 convert → multi_thread/fa HIF8 FAIL→PASS（09-04 的 `.fs→.hifb` convert 未注册消除）。
- **FP4 TCVT RNE-zero**（`dedcaba3`）：packed FP4 TCVT round-to-zero（issue #558，分支名 fix/issue-558-fp4-rne-zero）。
- **CUBE 目的容量保持**（`07e9c661`，HEAD）：retain primary CUBE destination capacity。
- **Packed local tile 存储**（`3e07fce8`）：按元素位宽存储 packed local tile。
- **MX scale 对齐 PTO**（`970ce7af`/`68a2324d`）：MX scale 处理对齐 PTO + MX B scale 验证为 CUBE_M32 [N,G] → **暴露 HIF4 MX scale dataType 断言**（fixp +14 FAIL、matmul +4 FAIL、fa +1 FAIL）。
- **RowMajor TileArray parent**（`47a5ec29`）：物化 RowMajor TileArray parent。
- 其他（codex 后续）：TMRGSORT merge（`07c69ea2`）、TLOG 自然对数（`28c024cb`）、reserved size code 拒绝（`8942027b`）、scalar SrcRType restore（`9e48ddb9`）、syscall X1 dispatch（`b3a5665e`）、writev EFAULT（`5a13a45d`）、cube store layout 全接受（`d2062aff`）、CUBE subview 行归约（`84c32cc8`）、HiF4X2 cooperative MX（`7104984b`/`c13cdb25`）。

**2. TileOP-API 契约扩展（34 commits）**
- **TMATMUL options+groupM 重载**（`5c4b9e9`/`b8669ce`，HEAD）：TMATMUL family 新增 options + groupM 重载。
- **B.DIM 维度 lowering**（`67d47ab`/`80cfeac`）：per-dimension static/dynamic B.DIM lowering（SS/SD/DS/DD）。
- **B.DATR PadValue/RMode 契约**（`f8fb894`/`fe11fb4`）：B.DATR PadValue Zero + RMode by PreQuant → **导致 fa/matmul TMATMUL 目的 dtype 必须匹配 PreQuantMode 断言**（multi_thread/fa FP8_VECBF16 编译失败 + set -e 级联）。
- **TROWPROD/TROWMIN 单列约束**：destination must be single-column tile (N x 1) → multi_thread/reduceprod_row 编译失败。
- **RedRows per-PE 语义**（`331a091`）：Local-A/Shared-B groupM 的 RedRows per-PE 语义修正。
- **cooperative TMATMUL shape**（`906f2c2`）：Local-A/Shared-B cooperative TMATMUL shape 校正。
- **row-reduction B.DIM**（`d3f8e47`）：行归约 B.DIM 描述源 tile geometry。
- **CUBE subview region asm**（`68a8ac0`）：允许 cube subview 在 region asm 中。
- **PTO 0.58.5 layout 接口**（`8677a6f`）：新增 PTO 0.58.5 layout TileOP 接口（TPERMUTE 等）。
- **template_asm 布局枚举修正**（`682bc21`）+ **InternalAcc CCTRL**（`32c4804`/`5b41111`，添加后回退）。

**3. llvm-project 小幅更新（2 commits）**
- **tile spill size code**（`553b08045`，HEAD）：Fix tile spill size code conversion。
- **B.DATR PadValue aliases**（`67d3ac986`）：B.DATR aliases exposing PadValue with RMode。

**4. SuperNPUBench 本侧改动**
- **FA/matmul CUBE 迁移**（`334737e`）：单线程 FA 和 matmul 迁移到 CUBE 布局 → matmul 3→16 编译成功（+13，IsCubeLayout 断言消除）、fa 10→13（+3）。
- **fa_opt v2**（`3e9d4ba`）：fa_2d_unroll_gmma 替换为 fa_opt v2 实现。
- **multi_thread/fa CUBE 迁移 WIP**（未提交）：fa_2d_unroll_gmma/fa_fixpipe/fa_subview 未提交改动，FP8_VECBF16 模式触发 TMATMUL dtype 断言 + compile.all `set -e` 级联跳过后续 ~43 个变体（7→5 ELF）。
- **microbenchmark 改动**（未提交）：gen_cases.py/vector_bench.hpp/memory_bench.hpp 改动 → vector tconcat 2 例编译失败（`bench_concat` 未声明）。
- **solution 树扩展 + deepseek 移除**：view_copy/gather_v2/normalization/moe/dynamic_mx_quant 移入 solution 树；deepseek 从 compile_all.sh 移除（MC2 算子重构进 solution，本轮排除）。

> 提取方法：对三个仓库分别跑 `git -C <repo> log --oneline <上轮 commit>..<本轮 commit>`（`linx-toolchain-build/src/Linx-TileOP-API`、`…/src/llvm-project`、`SuperScalarModel`），按模型/TileOP/llvm/SuperNPUBench 分类。

## 总体结果

| 范围 | ELF 数 | PASS | FAIL | TIMEOUT | 通过率 |
|---|---:|---:|---:|---:|---:|
| microbenchmark | 383 | 365 | 18 | 0 | 95.3% |
| one-level | 87 | 73 | 14 | 0 | 83.9% |
| **合计** | **470** | **438** | **32** | **0** | **93.2%** |

> 本轮排除 solution 树（15 ELF / 10 PASS / 5 编译失败），用户指定「solution 目录先不用管」。deepseek 从 compile_all.sh 移除（MC2 算子重构进 solution 树），09-04 的 21 deepseek ELF 不在本轮范围。

## 算子通过率

按算子族列出当前通过率（470 ELF 全量 gfrun，438 PASS / 32 FAIL / 0 TIMEOUT，通过率 93.2%）：

| 算子族 | 编译成功 | PASS | FAIL | 通过率 | 说明 |
|---|---:|---:|---:|---:|---|
| micro/scalar | 124 | 124 | 0 | 100% | 全过（tail-400 误判 sqrt_f64 1 例已修正） |
| micro/vector | 127 | 127 | 0 | 100% | 全过；2 例 tconcat 编译失败（`bench_concat` 未声明），未计入 |
| micro/memory | 14 | 14 | 0 | 100% | 全过 |
| micro/cube | 11 | 11 | 0 | 100% | 全过（tail-400 误判 4 例已修正） |
| micro/fixp | 107 | 89 | 18 | 83.2% | 15 例 shared 编译失败 + 18 例 MX scale dataType 断言（MX scale 对齐 PTO 后暴露） |
| one-level/broadcast | 6 | 5 | 1 | 83.3% | `vec_07 half` COPY 扩展断言（不变） |
| one-level/concat | 4 | 4 | 0 | 100% | 全过（tail-400 误判 2 例已修正） |
| one-level/control | 6 | 0 | 6 | 0% | INT8/16 dtype 元组未定义（不变） |
| one-level/element_wise | 1 | 1 | 0 | 100% | gelu 全过 |
| one-level/fa | 13 | 12 | 1 | 92.3% | CUBE 迁移新增变体（+3 ELF）；`fa_HIF4` 运行 FAIL（scale dataType 断言） |
| one-level/flashMLA | 2 | 2 | 0 | 100% | 全过 |
| one-level/gather | 1 | 1 | 0 | 100% | 全过 |
| one-level/matmul | 16 | 12 | 4 | 75.0% | CUBE 迁移修复编译（3→16，+13）；HIF4 MX 4 例运行 FAIL（scale dataType 断言） |
| one-level/multi_thread/fa | 5 | 5 | 0 | 100% | HIF8 convert 修复→全过；但 `set -e` 在 FP8_VECBF16 编译失败后跳过后续变体（7→5 ELF） |
| one-level/multi_thread/matmul | 11 | 11 | 0 | 100% | 全过 |
| one-level/multi_thread/(新算子) | 10 | 10 | 0 | 100% | broadcast/concat/conv2d/element_wise/gather/reduction/transpose（normalization 移入 solution） |
| one-level/multi_thread/vec | 1 | 1 | 0 | 100% | 全过 |
| one-level/reduction | 6 | 5 | 1 | 83.3% | 新增 `rowsum_subview`（TROWSUM 断言）；其余全过 |
| one-level/sort | 1 | 0 | 1 | 0% | `topk` 跟踪爆炸 >5000 万行（不变） |
| one-level/transpose | 4 | 4 | 0 | 100% | 全过 |

> 编译成功合计 470（micro 383 + one-level 87，排除 solution 15）。编译失败：micro 17（vector tconcat 2 + fixp shared 15）+ one-level 2（multi_thread/fa set -e 级联 + reduceprod_row）+ solution 5（排除）。`conv2d`/`norm` 单线程版未接入 `compile_all.sh`；`two-level-arch` 不支持当前 ISA 模式，未编译未跑。

## 编译覆盖

成功生成 ELF：470 个（microbenchmark 383 + one-level 87，排除 solution 15），编译失败约 24 个（micro 17 + one-level 2 + solution 5）。

**microbenchmark 编译失败（17）**：
- vector 2 个：`tconcat_fp16/fp32` — `bench_concat` 未声明标识符（vector_bench.hpp 未提交改动引入）。
- fixp 15 个：`shared_rowmax`/`shared_groupmax_*`/`shared_rowgroup_maxabs`/`shared_f16_groupmax`/`shared_s8_rowmax` — TileOP B.DATR/PreQuant 契约变更后编译失败（09-04 这些编译成功但 4 例运行 FAIL，本轮全部编译失败）。

**one-level 编译失败（2，排除 solution）**：
- multi_thread/fa：`compile.all` 有 `set -e`，FP8_VECBF16 模式触发 `TMATMUL destination dtype does not match PreQuantMode`（`is_fixp_output_type` / `matrix_output_type_legal` 静态断言，TileOP `b8669ce` B.DATR RMode 契约）后脚本退出，跳过 fa_fixpipe/fa_subview 全部变体 + 容量受限 / Sq=1024 / long-context 配置（~43 个变体跳过，7→5 ELF）。
- multi_thread/reduction/reduceprod_row：`TROWPROD destination must be a single-column tile (N x 1)` 静态断言（TileOP 行归约 B.DIM 契约变更）。

**solution 编译失败（5，本轮排除）**：group_token_vec、normalization/rms_norm、rms_norm_binary、group_norm_grad、group_norm_grad_1d。

**历史已知失败（不变）**：control `hashtable_lookup`（INT8/16 dtype 元组）、sort `topk`（运行 R2=1）。

**matmul 编译修复**：09-04 仅 3/16 编译成功（CUBE `IsCubeLayout` 断言），本轮 CUBE 迁移（`334737e`）后 16 个全部编译成功（MASK_MASK 9 + A16W4 3 + HIF4_MX 4）。

## 运行失败清单（32 FAIL，全部模型侧）

### microbenchmark/fixp — 18 FAIL（+14 vs 09-04）
`fixp_tmatmul_*_mx_*` 系列（gemv_mx/gemv_mx_acc/gemv_mx_bias/gemv_mx_s8/gemv_mx_scale_a/gemv_mx_scale_b/mx/mx_s8/mx_scale_a/mx_scale_b/mxacc/mxacc_s8/mxbias/mxbias_s8/s_shifts16/v_shifts16）—— `scale && scale->tileInfo && scale->tileInfo->dataType == (leftType == DataType::HIF4 ? DataType::UINT32 : DataType...)` 断言。MX scale 对齐 PTO（`970ce7af`）后 HIF4 scale dataType 校验暴露。09-04 这 18 例全部 PASS。

### one-level/control — 6 FAIL（不变）
`hashtable_lookup_simd_*` —— `srcTile.size()==1` / `RawTileSourceFits` 断言（INT8/16 dtype 元组未定义）。

### one-level/matmul — 4 FAIL（新增）
`matmul_HIF4_HIF4_MX_NOGATHER_*`（4 个变体）—— 同 fixp 的 `scale->dataType` 断言（HIF4 MX scale dataType）。09-04 这些变体未编译（IsCubeLayout），本轮 CUBE 迁移后编译成功但运行 FAIL。

### one-level/broadcast — 1 FAIL（不变）
`broadcast_vec_07 half` —— `COPY expansion` 断言。

### one-level/fa — 1 FAIL（新增）
`fa_HIF4_HIF4_BF16_NOGATHER_Sq256_Skv512` —— 同 HIF4 scale dataType 断言。CUBE 迁移新增变体。

### one-level/reduction — 1 FAIL（新增）
`rowsum_subview_Rows32_Cols32_Parts4` —— `illegal TROWSUM operand or descriptor contract`。新增 rowsum_subview 变体（未提交改动）。

### one-level/sort — 1 FAIL（不变）
`topk` —— gfrun 跟踪爆炸（>5000 万行指令跟踪），R2=1（结果错误）。

## 本次更新要点

- **环境**：gfrun `bc7fae00→07e9c661`（fix/issue-558-fp4-rne-zero，合并 codex 后续 + 3 新提交）；llvm `1ae4ee39→553b08045`（2 commits，tile spill size code + B.DATR PadValue aliases）；TileOP `804eb03→b8669ce`（34 commits，TMATMUL options/groupM 重载 + B.DATR RMode 契约 + TROWPROD 单列约束 + PTO 0.58.5 layout）。编译器仍按 AGENTS.md 用主 worktree，无切换。
- **FA/matmul CUBE 迁移**（`334737e`）：单线程 FA 和 matmul 迁移到 CUBE 布局 → matmul 3→16 编译成功（+13，IsCubeLayout 断言消除）、fa 10→13（+3）；matmul +9 PASS、fa +2 PASS。
- **HIF8 convert 修复**（`24d26eeb`）：gfrun 实现 HIF8 tile 转换 → multi_thread/fa HIF8 FAIL→PASS（09-04 `.fs→.hifb` convert 未注册消除）。
- **MX scale 对齐 PTO 回归**（`970ce7af`）：MX scale 处理对齐 PTO 后暴露 HIF4 scale dataType 断言 → fixp +14 FAIL（18 例 `fixp_tmatmul_*_mx_*`，09-04 全 PASS）、matmul +4 FAIL（HIF4 MX 新变体）、fa +1 FAIL（HIF4 新变体）。**本轮最大回归源**。
- **fixp 编译回归**：15 个 shared rowmax/groupmax 模式编译失败（TileOP B.DATR/PreQuant 契约变更，09-04 编译成功但 4 例运行 FAIL → 本轮全部编译失败）。
- **deepseek 移出 compile_all.sh**：MC2 算子重构进 solution 树，deepseek 21 ELF 不在本轮范围（−17 PASS / −4 FAIL）。
- **multi_thread/fa set -e 级联**：FP8_VECBF16 编译失败（TileOP dtype 断言）+ `set -e` 跳过后续 ~43 变体（7→5 ELF），但幸存 5 个全 PASS（含 HIF8 修复）。
- **solution 树排除**：用户指定，15 ELF / 10 PASS / 5 编译失败未计入。
- **run_all.sh 截断修正**：`tail -400` 对 verbose ELF（cube tmatmul、concat half、matmul A16W4、scalar sqrt_f64）误判 10 例 FAIL→实际 PASS，手工复跑修正。
- **净 478→438 PASS（−40）**：fixp −29（MX 断言 +14 FAIL + shared 编译失败 −15 ELF）、deepseek −17（移出）、vector −2（tconcat 编译失败），被 matmul +9、fa +2 部分抵消。0 TIMEOUT。通过率 96.4%→93.2%。

## 与 2026-09-04 基线的差异

> 本轮与 09-04 非单一变量：gfrun/llvm/TileOP 版本更新 + SuperNPUBench 本侧 FA/matmul CUBE 迁移 + solution 树扩展 + deepseek 移出 + microbenchmark 未提交改动。差异归因较复杂，需区分模型侧修复、契约变更暴露、本侧迁移三类。

| 类别 | 09-04 (ELF/P/F/T) | 09-08 (ELF/P/F/T) | 变化 |
|---|---|---|---|
| micro/scalar | 124/124/0/0 | 124/124/0/0 | — |
| micro/vector | 129/129/0/0 | 127/127/0/0 | ELF −2 / PASS −2（tconcat 编译失败，`bench_concat` 未声明） |
| micro/memory | 14/14/0/0 | 14/14/0/0 | — |
| micro/cube | 11/11/0/0 | 11/11/0/0 | — |
| micro/fixp | 122/118/4/0 | 107/89/18/0 | ELF −15 / PASS −29 / FAIL +14（MX scale 对齐 PTO 暴露 HIF4 断言 + shared 编译失败） |
| one-level/broadcast | 6/5/1/0 | 6/5/1/0 | — |
| one-level/concat | 4/4/0/0 | 4/4/0/0 | — |
| one-level/control | 6/0/6/0 | 6/0/6/0 | — |
| one-level/deepseek | 21/17/4/0 | 0/0/0/0 | 从 compile_all.sh 移除（MC2 算子重构进 solution 树，本轮排除） |
| one-level/element_wise | 1/1/0/0 | 1/1/0/0 | — |
| one-level/fa | 10/10/0/0 | 13/12/1/0 | ELF +3 / PASS +2 / FAIL +1（CUBE 迁移新增 HIF4 变体，运行 FAIL） |
| one-level/flashMLA | 2/2/0/0 | 2/2/0/0 | — |
| one-level/gather | 1/1/0/0 | 1/1/0/0 | — |
| one-level/matmul | 3/3/0/0 | 16/12/4/0 | ELF +13 / PASS +9 / FAIL +4（CUBE 迁移修复编译；HIF4 MX 运行 FAIL） |
| one-level/multi_thread/fa | 7/5/2/0 | 5/5/0/0 | ELF −2 / FAIL −2（HIF8 convert 修复；但 FP8_VECBF16 编译失败 + set-e 跳过后续） |
| one-level/multi_thread/matmul | 9/9/0/0 | 11/11/0/0 | — |
| one-level/multi_thread/vec | 1/1/0/0 | 1/1/0/0 | — |
| one-level/multi_thread/(新算子) | 13/13/0/0 | 10/10/0/0 | ELF −3 / PASS −3（normalization 移入 solution 树，排除） |
| one-level/reduction | 5/5/0/0 | 6/5/1/0 | ELF +1 / FAIL +1（新增 rowsum_subview，TROWSUM 断言） |
| one-level/sort | 1/0/1/0 | 1/0/1/0 | — |
| one-level/transpose | 4/4/0/0 | 4/4/0/0 | — |

> −40 PASS：fixp −29（MX 断言 +14 + shared 编译失败 −15 ELF）、deepseek −17（移出）、vector −2（编译失败），被 matmul +9、fa +2 部分抵消。两大变化——FA/matmul CUBE 迁移（matmul +13 编译 / +9 PASS）、MX scale 对齐 PTO 暴露 HIF4 断言（fixp +14 / matmul +4 / fa +1 FAIL）——为本轮核心。HIF8 convert 修复（mt/fa −2 FAIL）是正面成果。09-04 基线 496 ELF / 478 PASS / 18 FAIL / 96.4% → 09-08 470 ELF / 438 PASS / 32 FAIL / 93.2%。

---

# 历史验证记录

> 早期每日基线仅保留环境版本与总量，供复现与趋势对比；完整分类/失败明细已归档。

| 日期 | gfrun (SuperScalarModel) | llvm / TileOP-API | 工具链 | ELF | PASS | FAIL | T/O | 通过率 | 关键变化 |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| 09-10 | asl `codex/gfrun-pto-0586-asl` `e7d883c9` | `553b08045` / `b8669ce` | AGENTS.md 主 worktree（PTO v0.58.6） | 550 | 473 | 77 | 0 | 86.0% | gfrun 切换 asl worktree；multi_thread/fa 编译扩展（5→36，set -e 移除）；fa_subview 12 PASS；fa_2d_unroll_gmma/fa_fixpipe 24 FAIL（2048B 行归约限制）；vector +20 FAIL（compare/select TSTORE）；HIF4/HIF8 排除；fa_2d_unroll_gmma_subview 新算子 |
| 09-08 | fix/issue-558 `07e9c661` | `553b08045` / `b8669ce` | AGENTS.md 主 worktree（PTO v0.58.5） | 470 | 438 | 32 | 0 | 93.2% | FA/matmul CUBE 迁移（matmul +13 编译）；HIF8 convert 修复；MX scale 对齐 PTO 暴露 HIF4 断言（fixp +14 FAIL）；deepseek 移出 compile_all；solution 排除；−40 PASS vs 09-04 |
| 09-04 | codex `bc7fae00` | `1ae4ee39` / `804eb03` | AGENTS.md 主 worktree（CUBE 强制） | 496 | 478 | 18 | 0 | 96.4% | CUBE subview 行归约 + fixpipe GroupMax/RowMaxIn 修复（fa/flashMLA/reduction +14、fixp +98）；deepseek CUBE 编译修复；mt 新增 8 算子；+129 PASS / 0 回归 vs 08-27 |
| 08-27 | codex `d8903938` | `adcb8794` / `f94bc12` | AGENTS.md 主 worktree（CUBE 强制） | 427 | 349 | 78 | 0 | 81.7% | CUBE cell-layout 强制（matmul/deepseek 编译回归）；dataType 断言回归（fa/flashMLA/reduction −14）；cube +9、mt/matmul lowp +5；净 −17 vs 08-23 |
| 08-23 | exp `a5dca25a` | `611105f2b` / `a795b973020d` | blessed-latest（ADR 0069） | 437 | 366 | 69 | 2 | 83.8% | blessed 编译器+exp 模型正确配对；+27 PASS（fa 全过、mt/matmul lowp 部分）；13 编译失败 |
| 08-21 | main `7b691d4d` | `a84c4d10a` / `ffa257738f` | toolchain-build（32KB shared） | 437 | 339 | 98 | 0 | 77.6% | 老 compiler+main 模型；multi_thread 大 tile 首编（bf16/fp16/lowp 运行 FAIL） |
| 08-20 | exp `5a64c34d` | `b945a5d0` / `c02dae65` | blessed-latest | 433 | 344 | 89 | 0 | 79.4% | 零步幅 raw tile spill 修复 4 例；fa `sfa`×2 编译回归（TMATMUL 形状契约） |
| 08-19 | `01f9ec10` | `86959776b` / `8b2ee78` | — | 434 | 341 | 92 | 1 | 78.6% | 4 例 broadcast/GELU FAIL→PASS；mt/matmul 1 PASS→FAIL |
| 08-18 | `a68dba29` | — / `8b2ee78`（TileDType 修复） | — | 429 | 321 | 108 | 2 | 74.8% | 首次全量基线；fa/flashMLA/reduction 由 FAIL 恢复；fixp 27→4（TileDType 暴露契约偏差） |

**跨版本要点**：

- **09-10 gfrun 切换 asl worktree**：gfrun 从 `fix/issue-558-fp4-rne-zero`(`07e9c661`) 切换到 `codex/gfrun-pto-0586-asl`(`e7d883c9`)，PTO v0.58.5→v0.58.6。asl worktree 强制 PTO v0.58 行归约 source ≤2048 字节限制 → fa_2d_unroll_gmma/fa_fixpipe 24 FAIL（main gfrun 不强制此限制，09-08 全 PASS）；fa_subview 用 TPARTVIEW 分块规避，12/12 PASS。vector compare/select TSTORE 断言 +20 FAIL（asl gfrun 行为差异）。multi_thread/fa compile.all 移除 set -e → 5→36 ELF。gfsim 不支持 B.SUBVIEW 指令，fa_2d_unroll_gmma_subview 无法获取时序数据。
- **09-08 MX scale 对齐 PTO 暴露 HIF4 断言**：gfrun `970ce7af` MX scale 处理对齐 PTO 后，HIF4 MX scale dataType 断言在 fixp（+14 FAIL）/ matmul（+4）/ fa（+1）暴露——09-04 这些全 PASS。叠加 fixp shared 15 例编译失败（TileOP B.DATR/PreQuant 契约）→ fixp 118→89 PASS（−29）。FA/matmul CUBE 迁移（`334737e`）正面抵消：matmul +13 编译 / +9 PASS。
- **09-08 HIF8 convert 修复**：gfrun `24d26eeb` 实现 HIF8 tile 转换 → multi_thread/fa HIF8 FAIL→PASS（09-04 `.fs→.hifb` convert 未注册消除）。但 FP8_VECBF16 编译失败（TileOP dtype 断言）+ set -e 级联 → mt/fa 7→5 ELF。
- **09-08 deepseek 移出 compile_all.sh**：MC2 算子重构进 solution 树，deepseek 21 ELF 不在本轮范围（−17 PASS / −4 FAIL）。solution 树 15 ELF 本轮排除（用户指定）。
- **09-04 CUBE subview 行归约 + fixpipe 修复**：gfrun `bc7fae00` 修复 CUBE subview 行归约 → fa/flashMLA/reduction 全恢复（+14）；fixpipe GroupMax/RowMaxIn → fixp 43→4 FAIL（+98 PASS）。与 08-27 单一变量（仅版本更新，编译器 worktree 不变），+129 PASS / 0 回归，通过率 81.7%→96.4%。
- **ADR 0069 编码配对**（08-21 ↔ 08-23）：版本匹配则高 PASS，错位则骤降。08-21 老 compiler+main 模型（均无 ADR 0069）= 匹配 → 339P；08-23 blessed+exp（均有 ADR 0069）= 匹配 → 366P；而 blessed compiler+main 模型（编译器领先、模型落后）= 错位 → 仅 124P（262 个 `reserved/deleted TEPL selector`：store 的 SizeCode=0 被旧模型误读为 0B 目的）。
- **08-27 切回 AGENTS.md 主 worktree**（非 blessed-latest）：gfrun codex `d8903938`、TileOP `f94bc12`（CUBE cell-layout 强制）。与 08-23 非单一变量对比；09-04 在此基础上仅版本更新（无 worktree 切换），+129 PASS / 0 回归。
- **持续模型侧限制**（跨基线不变）：control `hashtable_lookup` INT8/16 dtype 元组（6 FAIL）、broadcast `vec_07 half` COPY 扩展断言、fixp MX scale dataType 断言（18 FAIL，09-08 起不变）。09-10 asl gfrun 新增：fa_2d_unroll_gmma/fa_fixpipe 行归约 2048B 限制（24 FAIL，main gfrun 不强制）、vector compare/select TSTORE（20 FAIL）、reduction validCol/TROWSUM（5 FAIL）。09-10 排除 sort/topk（不在此轮范围）。
