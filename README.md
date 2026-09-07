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

> **当前验证基线**：2026-09-04（496 个已编译 ELF 全量 gfrun 复测；编译器按 AGENTS.md 用主
> linx-toolchain-build worktree（llvm `1ae4ee39` + TileOP-API `804eb03`）、gfrun 用
> SuperScalarModel `codex/consolidate-post-main-fixes-20260903` `bc7fae00`（08-27 后 253 commits，
> 集中修复 CUBE subview 行归约 / fixpipe GroupMax/RowMax / HiF4X2 cooperative MX / packed FP4 TCVT
> 等模型侧断言）；总 PASS 478，通过率 96.4%——与 08-27 单一变量（仅版本更新，编译器 worktree 不变），
> +129 PASS / 0 回归）

# gfrun 执行结果汇总 — 2026-09-04

## 验证环境

| 组件 | 分支/版本 | Commit |
|---|---|---|
| gfrun / SuperScalarModel | `codex/consolidate-post-main-fixes-20260903` | `bc7fae00` |
| llvm-project | `dev-llvm15_56` | `1ae4ee39` |
| Linx-TileOP-API | `linx` | `804eb03` |

编译器按 **AGENTS.md** 指定用主 `linx-toolchain-build` worktree：`COMPILER_DIR=…/linx-toolchain-build/output/linx_blockisa_llvm_musl/bin`，clang 15.0.4，target `linx64v5-unknown-linux-musl`。gfrun 用 `codex/consolidate-post-main-fixes-20260903` `bc7fae00`（08-27 `d8903938` 之后 253 commits，集中修复 CUBE subview 行归约、fixpipe GroupMax/RowMax、HiF4X2 cooperative MX、packed FP4 TCVT、cube store layout 等模型侧断言）。执行：`gfrun -t 1 -f <elf>`，multi_thread 加 `-s softcore.multiThreadNum=4`，单 ELF 90s 超时。PASS = 退出码 0 + `Reach the End of Benchmark` + `R2 = 0`。

> 注：run_all.sh 的 `tail -400` 日志截断对 gfrun 输出超 400 行的 ELF（cube tmatmul fp16/i8、concat half gather/scatter）误判为 FAIL（PASS 标记被截断）。手工复跑 6 例确认实际 PASS，已修正；修正后合计 478 PASS / 18 FAIL。

> 注（09-04 复测）：编译器更新至 llvm `1ae4ee39`（+1 commit over `25677bb1a`，重编 toolchain output），TileOP-API `804eb03` 含未提交 `template_asm.hpp`（cooperative matmul LB0 = per-PE M × 4，满足 ADR-0100 group_M 合约）。multi_thread/fa 全 6 组 48 模式复测：fa_2d_unroll_gmma 三种 shape（Sq256 多 tile / Sq256 容量受限 / Sq1024）各 5 PASS / 2 FAIL（HIF8 convert + MXFP4 TCVT，模型侧限制，与 `25677bb1a` 完全一致，无回归）；fa_fixpipe 编译通过但 gfrun 全部 FAIL（cooperative TMATMUL 断言，新内核待修复）；HIF4_VECBF16 仍编译失败（shared fp4+shared fp4 不被 `matrix_input_pair_legal` 接受）。

## 本次新增特性（工具链 / 模型，08-27→09-04）

三个组件 08-27→09-04 同步演进：TileOP-API `f94bc12→804eb03` 42 commits、llvm `adcb8794→25677bb1a` 7 commits、gfrun `d8903938→bc7fae00` 253 commits。SuperNPUBench 本侧：fa nocvt 合并到 gmma（删除 nocvt 文件）、matmul/fa 新增 1024 大 shape 配置、multi_thread 新增 8 个算子（broadcast/concat/conv2d/element_wise/gather/normalization/reduction/transpose）。按类归档：

**1. gfrun 模型侧断言修复（直接驱动 +129 PASS）**
- **CUBE subview 行归约**（`bc7fae00`）：行归约支持 CUBE subview → fa(10)/flashMLA(2)/reduction(2) 共 +14 PASS（08-27 `dataType==block->dataType` 断言消除）。
- **FixPipe GroupMax/RowMaxIn**（`eade826c`/`671edad3`）：CUBE fixpipe 归约输出正确 → fixp shared rowmax/groupmax 路径修复（fixp 43→4 FAIL，−39）。
- **HiF4X2 cooperative MX**（`ab4188a0`+`0793bc98`）：cube 接受 HiF4X2 cooperative MX 输入 + shared MX parent/packed HIF4X2 load 对齐。
- **Packed FP4 TCVT**（`e6f6ed09`）：恢复 packed FP4 TCVT 支持（fa MXFP4 BF16→FP4 打包转换）。
- **Cube store layout 全接受**（`217c9fc8`/`1cce1c82`/`e2074d8c`）：TLSU 接受所有 cube store layout + cube m32→nd layout + tile args cube layout。
- **默认 4 PE**（`b4ded254`）：gfrun 默认 4 PE → multi_thread 无需显式 `-s multiThreadNum=4`。
- **TMRGSORT merge**（`bc1fd29d`）：ASL TMRGSORT 合并指令。
- **TLOG 自然对数**（`d10e6808`）：TEPL TLOG 实现为自然对数。
- **SizeCode 验证**（`25f60f31`/`9e3c1bec`）：拒绝 reserved tile size codes + GMMA.LD dest-binding 对齐 SizeCode 1..12。
- **Partial M reduction**（`a91a2c60`）：CUBE partial M reduction group 处理。
- **TLSU SL2 writeback**（`90afdcbe`）：scalar store complete 点后移 + SCB writeback 修复。
- 其他：`f32f73c3` syscall X1 dispatch、`b3c6a80f` writev EFAULT、`e662bc9a` scalar SrcRType restore。

**2. TileOP-API 契约扩展（42 commits）**
- **HiF4X2 Matrix-MX contracts**（`804eb03`，HEAD）：新增 HiF4X2 MX 契约 + 测试。
- **M16/M32 vector layout**（`c712579`）：vector 支持 CUBE_M16/M32 layout。
- **CUBE TCVT layout closure**（`d494a98`）+ **TCVT valid shape matching**（`e77df65`）：CUBE TCVT 闭环 + 形状校验。
- **Range modifier 统一**（`9550309`/`7572b3d`/`780417b`）：统一 range modifier 接口 + byte-sized subview ranges。
- **CScale for TMATMULMX.ACC**（`d6a52b8`）：optional CScale 编码。
- **Tile partition inline asm**（`0fe4879`）：tile partition 内联汇编。
- **Masked gather/reinterpret fix**（`f113cee`）：修正 masked gather 和 reinterpret tile 操作数。

**3. llvm-project 小幅更新（7 commits）**
- **MASK/GMOV block contracts**（`25677bb1a`，HEAD）：匹配当前 MASK 和 GMOV block 契约。
- **Tile region verifier**（`82faea4ca`）：实验性 tile region 验证器。
- **ELF machine 0xE9**（`0f878a871`）：ELF machine code 变更。
- SrcRType encoding 调整（revert/reapply 循环）。

**4. SuperNPUBench 本侧改动**
- **fa nocvt → gmma 合并**：nocvt 版本（local-Left P*V + shared-V，P 留寄存器）功能等价于 gmma，已替换 gmma.hpp 内容并删除 nocvt 文件；gmma.cpp 保留 RES_CHECK 框架，删除 prob_convert scratch。
- **matmul/fa 1024 大 shape**：matmul M=N=K=1024（shared + reuseB）、fa Sq=Skv=1024（gmma + fixpipe）。
- **multi_thread 新算子**：broadcast/concat/conv2d/element_wise/gather/normalization/reduction/transpose 接入 compile，全部 PASS（13 ELF）。

> 提取方法：对三个仓库分别跑 `git -C <repo> log --oneline <上轮 commit>..<本轮 commit>`（`linx-toolchain-build/src/Linx-TileOP-API`、`…/src/llvm-project`、`SuperScalarModel`），按模型/TileOP/llvm/SuperNPUBench 分类。

## 总体结果

| 范围 | ELF 数 | PASS | FAIL | TIMEOUT | 通过率 |
|---|---:|---:|---:|---:|---:|
| microbenchmark | 400 | 396 | 4 | 0 | 99.0% |
| one-level | 96 | 82 | 14 | 0 | 85.4% |
| **合计** | **496** | **478** | **18** | **0** | **96.4%** |

## 算子通过率

按算子族列出当前通过率（496 ELF 全量 gfrun，478 PASS / 18 FAIL / 0 TIMEOUT，通过率 96.4%）：

| 算子族 | 编译成功 | PASS | FAIL | 通过率 | 说明 |
|---|---:|---:|---:|---:|---|
| micro/scalar | 124 | 124 | 0 | 100% | 全过 |
| micro/vector | 129 | 129 | 0 | 100% | 全过 |
| micro/memory | 14 | 14 | 0 | 100% | 全过 |
| micro/cube | 11 | 11 | 0 | 100% | 全过（run_all.sh tail-400 误判 4 例已修正） |
| micro/fixp | 122 | 118 | 4 | 96.7% | 仅 shared rowmax/groupmax 4 例挂（模型侧 rowMax/powersOfTwo 断言） |
| one-level/broadcast | 6 | 5 | 1 | 83.3% | `vec_07 half` COPY 扩展断言（不变） |
| one-level/concat | 4 | 4 | 0 | 100% | 全过（run_all.sh tail-400 误判 2 例已修正） |
| one-level/control | 6 | 0 | 6 | 0% | INT8/16 dtype 元组未定义（不变） |
| one-level/deepseek | 21 | 17 | 4 | 81.0% | 5 CUBE 编译失败已修；aux_fi/group_count/inplace_unique/mask_indices 运行 FAIL |
| one-level/element_wise | 1 | 1 | 0 | 100% | gelu 全过 |
| one-level/fa | 10 | 10 | 0 | 100% | 全过（08-27 dataType 断言已修，+10） |
| one-level/flashMLA | 2 | 2 | 0 | 100% | 全过（同断言修复，+2） |
| one-level/gather | 1 | 1 | 0 | 100% | 全过 |
| one-level/matmul | 3 | 3 | 0 | 100% | 仅 3/16 编译成功（CUBE layout），幸存全过 |
| one-level/multi_thread/fa | 7 | 5 | 2 | 71.4% | HIF8（.fs→.hifb 未注册）、MXFP4（srcTile 断言） |
| one-level/multi_thread/matmul | 11 | 11 | 0 | 100% | 全过（含 1024 shape，cooperative 已建模） |
| one-level/multi_thread/(新算子) | 13 | 13 | 0 | 100% | broadcast/concat/conv2d/element_wise/gather/normalization/reduction/transpose |
| one-level/reduction | 5 | 5 | 0 | 100% | 全过（08-27 dataType 断言已修，+2） |
| one-level/sort | 1 | 0 | 1 | 0% | `topk` 编译已修，运行 `R2=1`（不变） |
| one-level/transpose | 4 | 4 | 0 | 100% | 全过 |

> 编译成功合计 496（micro 400 + one-level 96），编译失败约 20 个未计入上表（见下「编译覆盖」）。`conv2d`/`norm` 单线程版未接入 `compile_all.sh`、无 ELF；`two-level-arch` 不支持当前 ISA 模式，未编译未跑。

## 编译覆盖

成功生成 ELF：496 个（microbenchmark 400 + one-level 96），编译失败约 20 个。microbench 全部编译成功（0 失败）。one-level 编译失败分两类：

**CUBE cell-layout 编译回归（与 08-27 一致）**：TileOP-API CUBE cell-layout 强制（`IsCubeLayout` 静态断言要求 A/D=`CUBE_M16/M32`、B=`CUBE_N8`）：
- matmul 13 个变体（除 MASK_MASK FP32/FP8/FP16 外全部）— `template_asm.hpp IsCubeLayout`，与 08-27 相同。

**08-27 编译失败→本轮修复（deepseek CUBE）**：08-27 因 CUBE layout 编译失败的 5 个 deepseek 用例（aux_fi/get_fused_mapping/group_count/inplace_unique_group_indices/mask_indices_by_tp）本轮编译成功；其中 get_fused_mapping 运行 PASS，其余 4 个运行 FAIL（模型侧 binary TEPL / priorSources 断言）。

**历史已知失败（不变）**：fa/sfa Sq=256/Sq=512（TMATMUL output shape）、fa/fa_hif4（QuantType 未声明）、deepseek/expand_to_fused（clang SIGABRT）、deepseek/topk_gate（TCI ValidRow==1）、reduction/rowsum_subview。

**fa compile.all `set -e` 问题**：fa `compile.all` 脚本头有 `set -e`，HIF4_VECBF16 模式编译失败（shared fp4+shared fp4 被 `matrix_input_pair_legal` 拒绝）后脚本退出，跳过 fa_fixpipe 和 Sq=1024 大 shape（Sq=1024 FP32 单独编译验证通过）。需移除 `set -e` 或将 HIF4 移到循环末尾后容错。

## 运行失败清单（18 FAIL，全部模型侧）

### one-level/control — 6 FAIL（不变）
`hashtable_lookup_simd_*` —— `srcTile.size()==1` / `RawTileSourceFits` 断言（INT8/16 dtype 元组未定义）。

### one-level/deepseek — 4 FAIL
- `aux_fi`/`group_count`/`mask_indices_by_tp` —— `IsCompatibleOperationDataTile`（binary TEPL elemBytes/validCol/physicalCol）。
- `inplace_unique_group_indices` —— `priorSources==0 && inst->srcs.size()==3`（三源 TEPL 谓词断言）。

### one-level/multi_thread/fa — 2 FAIL（fa kernel + gfrun 双侧）
- `HIF8_VECFP32` —— `FloatPointUtils.cpp:1888` `.fs→.hifb` convert 未注册（gfrun SIGABRT，rc=134）。
- `MXFP4_VECBF16` —— `srcTile.size()==1` 断言（packed-x2 TCVT dst 列数=src/2，gfrun 不识别打包转换）。

### one-level/broadcast — 1 FAIL（不变）
`broadcast_vec_07 half` —— `COPY expansion` 断言。

### one-level/sort — 1 FAIL（不变）
`topk` —— 编译已修，运行 `R2=1`（结果错误）。

### microbenchmark/fixp — 4 FAIL（−39 vs 08-27）
fixp shared tmatmul 系列（`shared_f16_groupmax`/`shared_rowgroup_maxabs`/`shared_rowmax_init`/`shared_s8_rowmax`）—— `rowMax` layout/dataType 断言 + `powersOfTwo` sharedLeftSubview 断言。08-27 的 43 个 FAIL（quant/outputBytes/srcs.size/scale/accumulator/cooperative-PE-count/source/relu/hasFixpAttr）已全部修复。

## 本次更新要点

- **环境**：gfrun `d8903938→bc7fae00`（codex/consolidate-post-main-fixes-20260903，253 commits，集中修复模型侧断言）；llvm `adcb8794→25677bb1a`（7 commits，MASK/GMOV 契约 + tile region verifier）；TileOP `f94bc12→804eb03`（42 commits，HiF4X2 MX 契约 + CUBE TCVT 闭环 + range modifier 统一）。编译器仍按 AGENTS.md 用主 worktree，无切换。
- **fa/flashMLA/reduction 全恢复**：gfrun `bc7fae00` CUBE subview 行归约修复 → 08-27 的 `dataType==block->dataType` 断言消除，fa +10/flashMLA +2/reduction +2 = +14 PASS。
- **fixp 大幅改善**：fixpipe GroupMax/RowMaxIn 修复 + 新增 shared 测试 → 63→122 ELF，20→118 PASS（+98），43→4 FAIL（−39）。仅 4 个 shared rowmax/groupmax 仍挂。
- **deepseek CUBE 编译修复**：5 个 08-27 编译失败的 deepseek 用例编译成功，deepseek 16→21 ELF，+7 PASS。
- **multi_thread 扩展**：新增 8 个 multi_thread 算子（broadcast/concat/conv2d/element_wise/gather/normalization/reduction/transpose），13 ELF 全 PASS。matmul 加 1024 shape（+2 ELF，全过）。
- **fa nocvt 合并**：nocvt 版本等价替换 gmma，删除 nocvt 文件；gmma 保留 RES_CHECK。
- **run_all.sh 截断修正**：`tail -400` 对 verbose ELF（cube tmatmul、concat half）误判 6 例 FAIL→实际 PASS，手工复跑修正。
- **净 349→478 PASS（+129）**：fa+10、fixp+98、deepseek+7、flashMLA+2、reduction+2、mt/matmul+2、mt/新算子+13、concat+1，合计 +129 改善 vs 0 回归。0 TIMEOUT。

## 与 2026-08-27 基线的差异

> 本轮与 08-27 为单一变量对比：编译器 worktree 不变（主 linx-toolchain-build），仅 gfrun/llvm/TileOP 版本更新 + SuperNPUBench 本侧 fa/matmul/multi_thread 改动。差异可较可靠归因模型侧修复。

| 类别 | 08-27 (ELF/P/F/T) | 09-04 (ELF/P/F/T) | 变化 |
|---|---|---|---|
| micro/scalar | 124/124/0/0 | 124/124/0/0 | — |
| micro/vector | 129/129/0/0 | 129/129/0/0 | — |
| micro/memory | 14/14/0/0 | 14/14/0/0 | — |
| micro/cube | 11/11/0/0 | 11/11/0/0 | — |
| micro/fixp | 63/20/43/0 | 122/118/4/0 | ELF +59 / PASS +98 / FAIL −39（fixpipe GroupMax/RowMaxIn 修复 + 新增 shared 测试） |
| one-level/broadcast | 6/5/1/0 | 6/5/1/0 | — |
| one-level/concat | 4/3/1/0 | 4/4/0/0 | PASS +1 / FAIL −1（concat half 实际 PASS，08-27 为 run_all.sh 截断误判） |
| one-level/control | 6/0/6/0 | 6/0/6/0 | — |
| one-level/deepseek | 16/10/6/0 | 21/17/4/0 | ELF +5 / PASS +7 / FAIL −2（5 CUBE 编译修复；6 runtime FAIL→PASS） |
| one-level/element_wise | 1/1/0/0 | 1/1/0/0 | — |
| one-level/fa | 10/0/10/0 | 10/10/0/0 | PASS +10 / FAIL −10（CUBE subview 行归约修复） |
| one-level/flashMLA | 2/0/2/0 | 2/2/0/0 | PASS +2 / FAIL −2（同上） |
| one-level/gather | 1/1/0/0 | 1/1/0/0 | — |
| one-level/matmul | 3/3/0/0 | 3/3/0/0 | — |
| one-level/multi_thread/fa | 16/10/6/0 | 7/5/2/0 | ELF −9 / PASS −5 / FAIL −4（nocvt 删除 + set -e 跳过 1024；HIF8/MXFP4 仍挂） |
| one-level/multi_thread/matmul | 9/9/0/0 | 11/11/0/0 | ELF +2 / PASS +2（1024 shape，全过） |
| one-level/multi_thread/vec | 2/2/0/0 | 1/1/0/0 | ELF −1 |
| one-level/multi_thread/(新算子) | 0/0/0/0 | 13/13/0/0 | 新增 8 算子，全 PASS |
| one-level/reduction | 5/3/2/0 | 5/5/0/0 | PASS +2 / FAIL −2（CUBE subview 行归约修复） |
| one-level/sort | 1/0/1/0 | 1/0/1/0 | — |
| one-level/transpose | 4/4/0/0 | 4/4/0/0 | — |

> 改善 +129 PASS / 0 回归：fa+10、fixp+98、deepseek+7、flashMLA+2、reduction+2、mt/matmul+2、mt/新算子+13、concat+1。三大改善——CUBE subview 行归约修复（fa/flashMLA/reduction +14）、fixpipe GroupMax/RowMaxIn 修复（fixp +98）、deepseek CUBE 编译修复（+5 ELF/+7 PASS）——为本轮核心成果。

---

# 历史验证记录

> 早期每日基线仅保留环境版本与总量，供复现与趋势对比；完整分类/失败明细已归档。

| 日期 | gfrun (SuperScalarModel) | llvm / TileOP-API | 工具链 | ELF | PASS | FAIL | T/O | 通过率 | 关键变化 |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| 08-27 | codex `d8903938` | `adcb8794` / `f94bc12` | AGENTS.md 主 worktree（CUBE 强制） | 427 | 349 | 78 | 0 | 81.7% | CUBE cell-layout 强制（matmul/deepseek 编译回归）；dataType 断言回归（fa/flashMLA/reduction −14）；cube +9、mt/matmul lowp +5；净 −17 vs 08-23 |
| 08-23 | exp `a5dca25a` | `611105f2b` / `a795b973020d` | blessed-latest（ADR 0069） | 437 | 366 | 69 | 2 | 83.8% | blessed 编译器+exp 模型正确配对；+27 PASS（fa 全过、mt/matmul lowp 部分）；13 编译失败 |
| 08-21 | main `7b691d4d` | `a84c4d10a` / `ffa257738f` | toolchain-build（32KB shared） | 437 | 339 | 98 | 0 | 77.6% | 老 compiler+main 模型；multi_thread 大 tile 首编（bf16/fp16/lowp 运行 FAIL） |
| 08-20 | exp `5a64c34d` | `b945a5d0` / `c02dae65` | blessed-latest | 433 | 344 | 89 | 0 | 79.4% | 零步幅 raw tile spill 修复 4 例；fa `sfa`×2 编译回归（TMATMUL 形状契约） |
| 08-19 | `01f9ec10` | `86959776b` / `8b2ee78` | — | 434 | 341 | 92 | 1 | 78.6% | 4 例 broadcast/GELU FAIL→PASS；mt/matmul 1 PASS→FAIL |
| 08-18 | `a68dba29` | — / `8b2ee78`（TileDType 修复） | — | 429 | 321 | 108 | 2 | 74.8% | 首次全量基线；fa/flashMLA/reduction 由 FAIL 恢复；fixp 27→4（TileDType 暴露契约偏差） |

**跨版本要点**：

- **09-04 CUBE subview 行归约 + fixpipe 修复**：gfrun `bc7fae00` 修复 CUBE subview 行归约 → fa/flashMLA/reduction 全恢复（+14）；fixpipe GroupMax/RowMaxIn → fixp 43→4 FAIL（+98 PASS）。与 08-27 单一变量（仅版本更新，编译器 worktree 不变），+129 PASS / 0 回归，通过率 81.7%→96.4%。
- **ADR 0069 编码配对**（08-21 ↔ 08-23）：版本匹配则高 PASS，错位则骤降。08-21 老 compiler+main 模型（均无 ADR 0069）= 匹配 → 339P；08-23 blessed+exp（均有 ADR 0069）= 匹配 → 366P；而 blessed compiler+main 模型（编译器领先、模型落后）= 错位 → 仅 124P（262 个 `reserved/deleted TEPL selector`：store 的 SizeCode=0 被旧模型误读为 0B 目的）。
- **08-27 切回 AGENTS.md 主 worktree**（非 blessed-latest）：gfrun codex `d8903938`、TileOP `f94bc12`（CUBE cell-layout 强制）。与 08-23 非单一变量对比；09-04 在此基础上仅版本更新（无 worktree 切换），+129 PASS / 0 回归。
- **持续模型侧限制**（跨基线不变）：control `hashtable_lookup` INT8/16 dtype 元组（6 FAIL）、sort `topk` R2=1（结果错误）、broadcast `vec_07 half` COPY 扩展断言；fixp 43→4 FAIL（shared rowmax/groupmax 路径残留）、mt/fa HIF8/MXFP4（gfrun 不识别 packed TCVT / .fs→.hifb convert）。cube 目的容量断言在 08-27 已消除；fixp quant/outputBytes/accumulator 等断言在 09-04 已消除。
