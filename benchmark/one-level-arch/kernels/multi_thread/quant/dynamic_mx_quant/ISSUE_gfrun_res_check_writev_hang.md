# ISSUE: gfrun res_check 精度流程在 stdout writev 上无限挂起(guest iov 落到未映射高地址)

- **归属仓库**: SuperScalarModel(gfrun / emulator)
- **状态**: 未修复,已定位根因
- **暴露组合(new combo)**:
  - model: SuperScalarModel `codex/pr-0.58.4-shared-model`(工作目录 `dmxq-ops-20260828` 分支)
  - 工具链: Linx-TileOP-API `d6a52b8` + llvm `0f878a8`(新 clang / 新 musl)
- **触发 kernel**: `dynamic_mx_quant_tail_ocp_fp8`(RES_CHECK build,4PE);`tail_ocp_fp4` 同理
- **不影响**: kernel 计算路径本身。plain build(非 res_check)compute 已验 `R2=0`。

---

## 一、现象

res_check 精度流程(读 input.bin → 计算 → 写 output.bin,并向 stdout 打印状态)在**向 stdout 打印时无限挂起**,gfrun 不退出。musl 对 fd=1 的 `writev` 反复用同一 iov 重试。

## 二、复现

```bash
# 用 new combo 工具链编好 res_check ELF 后:
GFRUN_FORCE_DIRECTBOOT_ABI=1 \
  bin/gfrun -f dynamic_mx_quant_tail_ocp_fp8.elf     # 挂起,不退出
```
> `GFRUN_FORCE_DIRECTBOOT_ABI=1` 是问题23 的既有本地绕过(强制读 X1 取 syscall 号),
> 用于让 res_check ELF 越过 Bad Syscall、跑到本问题暴露的位置。它**不是本问题的成因**(见 §五)。

## 三、诊断证据

### 3.1 writev 每次都拿到全零 iov、返回 0

在 `emulator/SysCall.h` 的 writev handler 内 host `sys_writev` 之后临时打印(已撤销):

```
[WV-DIAG] a1=fffffffff7f741e0 iovcnt=2 base0=0 len0=0 ret=0
[WV-DIAG] a1=fffffffff7f741e0 iovcnt=2 base0=0 len0=0 ret=0
[WV-DIAG] a1=fffffffff7f741e0 iovcnt=2 base0=0 len0=0 ret=0   ← 同一 iov 无限重复
```

链条:
1. guest musl 把 stdout 的 `struct iovec[2]` 放在地址 **`A1 = 0xfffffffff7f741e0`**;
2. handler 执行 `aaccelssor.Load(A1 + i*16, ...)` 读 `iov_base`/`iov_len`,**均读回 0**;
3. → host `writev(1, iov, 2)` 实际写 0 字节 → 返回 `ret=0`;
4. → 返回值 0 写回 guest A0 → musl 认为"未写完",拿**同一 iov** 再次 `writev` → 无限循环。

### 3.2 该 iov 地址不在任何映射段内

同一次运行,gfrun 启动时打印的内存映射:

```
Memory: 0x19000   - 0x591cf     [.data]
Memory: 0x5a000   - 0x7c67f     [.bss]
Memory: 0x7c680   - 0x808c680   [stack mem]   (~131 MB)
Memory: 0x4000802000 - 0x4008803000 [map mem] (mmap 区)
```

而 iov 指针 `0xfffffffff7f741e0`(高 32 位全 1,符号扩展的负地址):
- 不在 stack 段(上界 `0x808c680`)内;
- 不在 mmap 段(`0x40_00802000`..)内;

→ 对该地址的 `Load` 落到未映射区,返回 0。**这是 writev 拿到全零 iov 的直接原因。**

## 四、根因分析

guest 的 stdout iov 指针解析为一个**未映射的高/符号扩展地址** `0xfffffffff7f741e0`,codex 内存模型无法为其提供后备存储,`Load` 静默返回 0。两个待确认的候选源头:

- **(A) 栈/缓冲布局错配**:LoadElf 的栈初始化(`InitializeThreadStacks` / `InitialStackPointer`)给新 musl 设的 guest SP,与实际映射的 `[0x7c680, 0x808c680]` 栈段不一致;新 musl 把 stdio 缓冲/iov 放到了映射段之外。
- **(B) 新 clang 地址生成**:llvm `0f878a8` 对该指针做了 32→64 符号扩展(`0xf7f741e0 → 0xfffffffff7f741e0`),而 codex 的 `Load` 未对地址做规范化/掩码,直接按未映射处理。

> 待确认动作(只读诊断,不改逻辑):打印 `InitialStackPointer` 返回的 guest SP,与栈段边界 `[0x7c680, 0x808c680]` 比对。SP 若落在 `0xfffffff...` 高段,则 (A) 成立,属 LoadElf/栈初始化对新 musl 布局的适配缺陷。

## 五、排除项:与问题23 的 ABI 移植无关

`main.cpp` 里 `GFRUN_FORCE_DIRECTBOOT_ABI` 只覆写 `core->directBootSyscallAbi` 一个 bool。经 grep,该 flag 的**全部逻辑用处仅 3 处**,都在"选 syscall **号**从 X1 还是 A7 读":
`SysCall.cpp:105`、`SoftCore.cpp:2716`、`SysCall.h:650`。它**不出现在任何内存/栈路径**(`LoadElf` / `InitializeThreadStacks` / `InitialStackPointer` / `Load` 均不引用)。

铁证:syscall **参数**是无条件从 A0–A5 读的(`SysCall.h:653`):
```cpp
arguments{gpr[xA0], gpr[xA1], gpr[xA2], gpr[xA3], gpr[xA4], gpr[xA5]},
```
即 iov 指针 `A1 = 0xfffffffff7f741e0` 无论 flag 取值都从同一寄存器读出。ABI flag 只让执行**走到**这里(否则先 Bad Syscall 死掉),并不**产生**这个坏地址。

## 六、影响范围

- fp8 / fp4 的 **res_check 精度 harness** 在 new combo 上全部挂起,无法产出 output.bin 精度对比。
- kernel **计算正确性不受影响**(plain build compute `R2=0`)。

## 七、建议修法(按代价排序,待定夺)

1. **先做 §四 的只读诊断**,确认 (A) 还是 (B),据此定位到 LoadElf/栈初始化 或 `Load` 地址规范化,提 SuperScalarModel 修复。
2. **harness 侧规避**:res_check 精度只依赖对 output.bin 的文件 `write`(fd 来自 openat),stdout 的状态 printf 非必需;静音该 printf 即可绕过这条 writev。
3. **compute-only 过渡**:精度 harness 修好前,先用 plain build 验证 fp4 计算正确性。

> 注:不建议在 codex 内存模型/writev handler 上直接硬改(易引新问题)。本 issue 只定位根因,修复走上游。
