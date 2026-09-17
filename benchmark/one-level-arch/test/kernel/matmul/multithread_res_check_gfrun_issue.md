# [gfrun] 4 PE 算子启用 `res_check=on` 后在 musl 退出路径触发 `Block BARG target` 断言

## 问题摘要

`multi_thread/matmul` 使用裸机启动方式编译时，可以在 gfrun 的 4 PE 配置下正常执行结束；同一份算子代码启用 `res_check=on` 后，编译和链接仍然成功，但 gfrun 在执行 musl 文件关闭/退出路径时失败：

```text
ecall warnning: close function failed! errno: 9 Bad file descriptor
ecall warnning: close function failed! errno: 9 Bad file descriptor
ecall warnning: close function failed! errno: 9 Bad file descriptor
gfrun: illegal instruction: ASSERTION FAILED:
currentBlock->barg.target != 0 && "Block BARG target"
```

对照实验说明 cooperative `TMATMUL` 本身可以正常执行。问题只在 4 PE 与 hosted musl `res_check` 路径组合时出现，尚未进入 `res.bin` 与 PyTorch golden 的比较阶段。

## 环境信息

### SuperNPUBench

```text
仓库: https://github.com/PTO-ISA/SuperNPUBench
分支: main
commit: 4fadccb7a33e562dece91d256684e9fb39ae412d
```

测试时工作区包含本地修改，本文涉及的主要文件为：

```text
benchmark/one-level-arch/test/kernel/multi_thread/matmul/src/matmul_shared.cpp
benchmark/one-level-arch/test/kernel/multi_thread/matmul/src/golden_cmp.py
benchmark/one-level-arch/test/kernel/multi_thread/matmul/Makefile
benchmark/one-level-arch/test/common/Makefile.common
```

### 编译器

```text
路径:
/Users/blacktraker/Programming/gitproj/DV4/linx-toolchain-build-latest/output/linx_blockisa_llvm_musl/bin/clang++

clang version 15.0.4
linx64v5-musl-local 611105f2be11fab9a8ef20bd02b740f2c5d786b3
Target: linx64v5-unknown-linux-musl

toolchain worktree commit:
e6a31efb4cfb17f1f1c33265cbf6dbb61bbba156

llvm-project branch:
dev-llvm15_56

llvm-project commit:
611105f2be11fab9a8ef20bd02b740f2c5d786b3
```

### gfrun 模型

```text
二进制:
/Users/blacktraker/Programming/gitproj/DV4/SuperScalarModel/bin/gfrun

源码分支:
feat/gfrun-cooperative-tmatmul-fp16-bf16-rerun

源码 commit:
a5dca25a5a6802d047573ae71cf39e2615d5356b
```

模型源码工作区存在 `CubeEngine.cpp`、`TEPLEngine.cpp` 的本地修改，因此以上 commit 描述的是源码 checkout 基线，不能单独证明现有 `bin/gfrun` 的精确构建身份。

## 相关算子代码

算子在 `RES_CHECK` 下由每个 PE 执行相同的文件读取和写入代码：

```cpp
int main() {
    using dtype = DTYPE;

    dtype src0p[Batch * globM * globK + 2 * ALIGN];
    dtype src1p[Batch * globK * globN + 2 * ALIGN];
    float dstp[Batch * globM * globN + 2 * ALIGN];

    dtype *src0 = (dtype *)(((uint64_t)src0p & ALIGN_MASK) + ALIGN);
    dtype *src1 = (dtype *)(((uint64_t)src1p & ALIGN_MASK) + ALIGN);
    float *dst = (float *)(((uint64_t)dstp & ALIGN_MASK) + ALIGN);

#ifdef RES_CHECK
    readBinaryFile(CHK_DIR "/src0.bin", (uint8_t *)src0,
                   Batch * globM * globK * sizeof(dtype));
    readBinaryFile(CHK_DIR "/src1.bin", (uint8_t *)src1,
                   Batch * globK * globN * sizeof(dtype));
#endif

    BENCHSTART;
    for (int b = 0; b < Batch; ++b) {
        matmul_shared<dtype, globM, globN, globK,
                      tilM, tilN, tilK>(
            dst + b * globM * globN,
            src0 + b * globM * globK,
            src1 + b * globK * globN);
    }
    BENCHEND;

#ifdef RES_CHECK
    writeBinaryFile(CHK_DIR "/res.bin", (uint8_t *)dst,
                    Batch * globM * globN * sizeof(float));
#endif

    return 0;
}
```

当前代码没有让 PE0 独占文件 I/O，也没有在 I/O 与 cooperative kernel 之间设置多 PE 同步。更重要的是，即使在 `main()` 内增加 `tid == 0` 判断，四个 PE 在进入 `main()` 之前仍然都会从 hosted musl `_start` 开始执行，因此仅修改 `main()` 还不足以解决完整问题。

## 构建行为差异

`Makefile.common` 的默认非 bare-metal 链接路径显式使用项目自带的 `_start.s`：

```make
CC_LINK += -nostartfiles $(ROOT)/test/common/_start.s
```

启用 `res_check=on` 后会清空该链接选项：

```make
ifeq ($(res_check), on)
DEFINES += -DRES_CHECK -DENABLE_BINARY_OUTPUT
DEFINES += -DCHK_DIR="$(ROOT)/compare/$(notdir $(basename $(TARGET)))"
CC_LINK =
endif
```

这使最终 ELF 改用完整 musl 启动和退出流程。

### 普通 ELF

```text
Entry point: 0x1129c
```

核心反汇编：

```asm
000000000001129c <_start>:
    C.BSTART.STD DIRECT, main
    c.setret _end, ->ra

00000000000112a0 <_end>:
    addi zero, 94, ->x1
    acrc 1
    C.BSTOP
```

### `res_check=on` ELF

```text
Entry point: 0x1207c
```

符号及调用路径：

```text
0x1207c  _start
0x120a4  _start_c
0x1269c  __libc_start_main
0x122d2  main
```

核心反汇编：

```asm
000000000001207c <_start>:
    ...
    C.BSTART.STD DIRECT, 0x120a4 <_start_c>

00000000000120a4 <_start_c>:
    C.BSTART.STD DIRECT, 0x1269c <__libc_start_main>
```

## 复现步骤

以下命令均在目录中执行：

```bash
cd /Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench/benchmark/one-level-arch/test/kernel/multi_thread/matmul

export COMPILER_DIR=/Users/blacktraker/Programming/gitproj/DV4/linx-toolchain-build-latest/output/linx_blockisa_llvm_musl/bin
export GFRUN=/Users/blacktraker/Programming/gitproj/DV4/SuperScalarModel/bin/gfrun
```

### 对照组：普通 4 PE ELF

```bash
make -B \
  TESTCASE=matmul_shared DTYPE=float \
  B=1 M=64 N=16 K=16 \
  tM=64 tN=16 tK=16 \
  OBJ_ROOT=/tmp/issue_mt_matmul_plain \
  COMPILER_DIR="$COMPILER_DIR" diss

"$GFRUN" -s softcore.multiThreadNum=4 -f \
  /tmp/issue_mt_matmul_plain/kernel/multi_thread/matmul/elf/kernel_multi_thread_matmul_matmul_shared_B1_M64_N16_K16_tM64_tN16_tK16.elf
```

实际结果：

```text
Thread:0 Total Block number = 12
Thread:1 Total Block number = 12
Thread:2 Total Block number = 12
Thread:3 Total Block number = 12
Total Block number = 48
Total Inst number = 272
Success to Reach the End of Benchmark! R2 = 0
```

### 实验组：`res_check=on` 4 PE ELF

```bash
make -B \
  TESTCASE=matmul_shared DTYPE=float \
  B=1 M=64 N=16 K=16 \
  tM=64 tN=16 tK=16 \
  res_check=on \
  OBJ_ROOT=/tmp/issue_mt_matmul_rescheck \
  COMPILER_DIR="$COMPILER_DIR" diss

cd /Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench

python3 -B \
  benchmark/one-level-arch/test/kernel/multi_thread/matmul/src/golden_cmp.py \
  --ones \
  -d /tmp/issue_mt_matmul_rescheck/kernel/multi_thread/matmul/elf/kernel_multi_thread_matmul_matmul_shared_B1_M64_N16_K16_tM64_tN16_tK16.elf
```

注意：如果复制命令，请确认 ELF 文件名中的结尾为当前 Makefile 生成的：

```text
..._tM64_tN16_tK16.elf
```

实际 gfrun 命令为：

```bash
/Users/blacktraker/Programming/gitproj/DV4/SuperScalarModel/bin/gfrun \
  -s softcore.multiThreadNum=4 -f \
  /tmp/issue_mt_matmul_rescheck/kernel/multi_thread/matmul/elf/kernel_multi_thread_matmul_matmul_shared_B1_M64_N16_K16_tM64_tN16_tK16.elf
```

实际结果：

```text
Starting from 0x1207c
ecall warnning: close function failed! errno: 9 Bad file descriptor
ecall warnning: close function failed! errno: 9 Bad file descriptor
ecall warnning: close function failed! errno: 9 Bad file descriptor
gfrun: illegal instruction: ASSERTION FAILED:
currentBlock->barg.target != 0 && "Block BARG target"
, func UpdateNextPC,
file .../SuperScalarModel/emulator/SoftCore.cpp:1038
```

## 预期结果

4 PE 的 `res_check=on` ELF 应当：

1. 正确读取一次 `src0.bin` 和 `src1.bin`；
2. 四个 PE cooperative 执行 `matmul_shared`；
3. 正确写出一次完整 `res.bin`；
4. 正常退出，不出现文件描述符错误或控制流断言；
5. `golden_cmp.py` 继续执行 PyTorch 数值比较。全 1 输入下，每个输出元素应为 `K=16`。

## 实际结果

1. 编译、链接成功；
2. gfrun 正确加载 ELF；
3. 运行过程中出现三次 `close()` 的 `EBADF`；
4. 某个 PE 的返回块目标为 0；
5. `SoftCore::UpdateNextPC()` 在检查 `currentBlock->barg.target` 时断言；
6. 模型返回码为 1，未进行数值比较。

对应模型代码：

```cpp
case BranchType::BLK_BR_DIRECT:
case BranchType::BLK_BR_CALL:
case BranchType::BLK_BR_IND:
case BranchType::BLK_BR_ICALL:
case BranchType::BLK_BR_RET:
    ASSERT(currentBlock->barg.target != 0 && "Block BARG target");
    return currentBlock->barg.target;
```

## 根因分析

### 已确认事实

1. 普通 ELF 和 `res_check=on` ELF 使用相同的 `matmul_shared` cooperative kernel；普通 ELF 在 4 PE 下执行成功。
2. `res_check=on` 会清空 `CC_LINK`，从而移除项目自带 `_start.s`，切换到 musl `_start -> __libc_start_main -> main -> exit` 路径。
3. `softcore.multiThreadNum=4` 下四个 PE 都从同一个 ELF 入口开始执行，而不是仅 PE0 执行进程启动、其余 PE 作为受 libc 管理的工作线程加入。
4. 当前 `RES_CHECK` 代码没有按 PE 区分文件 I/O，四个 PE 都会读写和关闭相同路径的文件。
5. 失败发生在 musl `close`/退出附近，先出现三个 `EBADF`，随后出现返回目标为 0 的断言。

### 根因判断

直接触发条件是模型执行返回类 Block 时得到 `barg.target == 0`。上游原因是当前 gfrun 多 PE 启动语义与 hosted musl 的单进程启动/退出语义不兼容：四个 PE 重复执行进程级 libc 初始化、文件描述符操作和退出流程，导致文件状态及控制流返回状态失效。

因此问题不属于 `TMATMUL` 数值逻辑，也没有证据表明是编译器错误。它是以下两部分的接口问题：

- gfrun 目前不能直接让四个 PE 同时执行同一套 hosted musl 启动与退出流程；
- benchmark 的 `RES_CHECK` harness 也不应让所有 PE 无同步地执行进程级文件 I/O。

问题链路可概括为：

```text
res_check=on
  -> 清除自定义 _start.s
  -> 链接 hosted musl runtime
  -> 4 PE 同时从 musl _start 执行
  -> 4 PE 重复 read/write/close/exit
  -> close(fd) 出现 EBADF，返回控制流状态异常
  -> RET/IND Block 的 barg.target 变为 0
  -> SoftCore::UpdateNextPC 断言
```

## 问题归属建议

建议主要在 gfrun/多线程运行时接口中跟踪，同时在 SuperNPUBench 补充适合 multi-thread 算子的数值验证机制。

不建议通过删除 `SoftCore.cpp` 中的非零断言来掩盖问题。`target == 0` 已经表示返回地址或 Block 控制流状态无效，简单跳过断言可能产生静默错误或跳转到零地址。

## 修复建议

### 方案一：为多 PE ELF 提供明确的启动协议（推荐）

- 仅 PE0 执行 musl `_start`、文件 I/O 和进程退出；
- PE1～PE3 从专用 worker 入口启动，或者在 PE0 初始化完成后由模型激活；
- 在进入和离开 cooperative kernel 前后提供 core-level barrier；
- 仅 PE0 最终执行 libc 退出流程。

这需要 gfrun 明确区分 process leader 与 worker PE。

### 方案二：数值验证继续使用裸机入口

- 保留项目自带 `_start.s`；
- 通过 gfrun 的预加载能力把 `src0/src1` 写入已知的全局地址；
- 将输入和输出声明为具有稳定符号地址的全局缓冲区；
- 算子执行完成后由模型 dump 输出地址；
- host Python 脚本读取 dump 并与 PyTorch golden 比较。

该方案绕开多 PE hosted libc，改动范围相对可控。

### 方案三：增加 multi-thread 专用 RES_CHECK launcher

- PE0 负责读写文件；
- 输入、输出使用所有 PE 可见的全局/共享缓冲区，而不是 PE0 私有栈；
- 文件读取后、kernel 前以及 kernel 后、文件写出前分别同步；
- worker PE 不参与文件系统调用和 libc exit。

但如果四个 PE 仍然都从 musl `_start` 进入，仅在 `main()` 内添加 `if (tid == 0)` 仍不能完全避免多 PE 重复执行 libc 启动与退出，因此该方案需要与方案一配合，或者使用定制启动代码。

## 建议增加的回归用例

建议建立一个不含 Tile 指令的最小测试，用于单独验证 hosted runtime：

1. `softcore.multiThreadNum=1`，单次 open/read/write/close/exit；
2. `softcore.multiThreadNum=4`，所有 PE 执行同一 hosted `main`；
3. `softcore.multiThreadNum=4`，仅 leader 执行 I/O，worker 走独立入口；
4. 验证每个 PE 的 `ra`、Block `barg.target`、文件描述符生命周期以及最终退出状态；
5. 在运行 cooperative Tile 算子前，先保证以上 runtime-only 用例稳定通过。

## 附件与日志位置

本地复现日志：

```text
/Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench/benchmark/one-level-arch/compare/kernel_multi_thread_matmul_matmul_shared_B1_M64_N16_K16_tM64_tN16_tK16/gfrun.log
```

普通 ELF 与反汇编：

```text
/tmp/issue_mt_matmul_plain/kernel/multi_thread/matmul/elf/kernel_multi_thread_matmul_matmul_shared_B1_M64_N16_K16_tM64_tN16_tK16.elf
/tmp/issue_mt_matmul_plain/kernel/multi_thread/matmul/elf/kernel_multi_thread_matmul_matmul_shared_B1_M64_N16_K16_tM64_tN16_tK16.elf.diss
```

`res_check=on` ELF 与反汇编：

```text
/tmp/issue_mt_matmul_rescheck/kernel/multi_thread/matmul/elf/kernel_multi_thread_matmul_matmul_shared_B1_M64_N16_K16_tM64_tN16_tK16.elf
/tmp/issue_mt_matmul_rescheck/kernel/multi_thread/matmul/elf/kernel_multi_thread_matmul_matmul_shared_B1_M64_N16_K16_tM64_tN16_tK16.elf.diss
```
