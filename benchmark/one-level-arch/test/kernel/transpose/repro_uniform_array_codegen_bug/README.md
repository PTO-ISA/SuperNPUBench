# 均匀常量数组物化错误 — 最小复现包

关联 issue：<https://github.com/LinxISA/SuperScalarModel/issues/1038>

## 现象

`uint32` 数组若**所有元素相同且非 0**（uniform），物化/写出后**奇数下标的元素被放大为 2×**（`gfrun` 与 `gfsim` 均复现）：

| 源 `constexpr uint32_t[4]` | 期望 | 实测 |
|---|---|---|
| `{8,8,8,8}` | `[8,8,8,8]` | **`[8,16,8,16]`** |
| `{16,16,16,16}` | `[16,16,16,16]` | **`[16,32,16,32]`** |
| `{1,1,1,1}` | `[1,1,1,1]` | **`[1,2,1,2]`** |
| 运行时构造 `for(i) r[i]=9` | `[9,9,9,9]` | **`[9,18,9,18]`** |
| `{1,2,3,4}` / `{5,6,7,8}` / `{0,0,0,0}` | 原值 | 正确（对照组） |

- **触发条件**：数组元素**全相同且非 0**；`-O0` 与 `-O2` 均复现；2/4/8 元素均复现；`constexpr` 与运行时构造均复现。
- **影响**：会污染"均匀的 shape 数组"，导致 `transpose` 测试在小 shape（如 `[8,8,8,8]`）上得到错误结果（看似"恒等置换"）。

## 目录内容

| 文件 | 说明 |
|---|---|
| `../src/transpose_minrepro.cpp` | 最小复现源码（随本 PR 一起提交） |
| `prebuilt/kernel_transpose_transpose_minrepro_PE4.elf` | 预编译 ELF（`gfrun`/`gfsim` 可直接跑） |
| `prebuilt/kernel_transpose_transpose_minrepro_PE4.elf.diss` | ELF 反汇编 |
| `evidence/uniform_const_store.cpp` / `uniform.s` / `uniform_O0.s` | 均匀常量 codegen 证据（`-O2` / `-O0`） |
| `evidence/nonuniform_const_store.cpp` / `nonuniform.s` | 非均匀常量对照（正确） |
| `evidence/observed_output.txt` | 实测输出 |
| `evidence/toolchain_commit.txt` | 工具链版本 |

## 复现命令

### 方式 A：直接跑预编译 ELF（最快）

```bash
SUPERNPU=<本仓库根目录>
MODEL=<SuperScalarModel 根目录>
OUT=$SUPERNPU/benchmark/one-level-arch/compare/kernel_transpose_transpose_minrepro_PE4
mkdir -p "$OUT"

# 功能模型
$MODEL/bin/gfrun -s softcore.multiThreadNum=1 -f \
    benchmark/one-level-arch/test/kernel/transpose/repro_uniform_array_codegen_bug/prebuilt/kernel_transpose_transpose_minrepro_PE4.elf

# 或 soc 模型
$MODEL/bin/gfsim -s softcore.multiThreadNum=1 -f \
    benchmark/one-level-arch/test/kernel/transpose/repro_uniform_array_codegen_bug/prebuilt/kernel_transpose_transpose_minrepro_PE4.elf

# 查看结果
od -An -tu4 "$OUT/uniform_out.bin"      # 期望 [8 8 8 8]，实测 [8 16 8 16]
od -An -tu4 "$OUT/nonuniform_out.bin"   # [1 2 3 4]（对照，正确）
```

### 方式 B：从源码构建再跑

```bash
SUPERNPU=<本仓库根目录>
export COMPILER_DIR=<linx_blockisa_llvm_musl>/bin   # clang/clang++ 所在目录

make -C "$SUPERNPU/benchmark/one-level-arch/test/kernel/transpose" \
     TESTCASE=transpose_minrepro res_check=on diss

# 生成：benchmark/one-level-arch/output/kernel/transpose/elf/kernel_transpose_transpose_minrepro_PE4.elf
# 之后同方式 A 的 gfrun/gfsim/od 命令
```

## 观测输出

```
uniform_out.bin    = [8, 16, 8, 16]   ← 错误（奇数下标 ×2）
nonuniform_out.bin = [1, 2, 3, 4]     ← 正确
```

## 环境

- 工具链：`linx_blockisa_llvm_musl`，`clang version 15.0.4 (linx64v5-musl-local 7a120731d46c3a35ce0a174586404a60302c74e8)`，target `linx64v5-unknown-linux-musl`（`-O0`/`-O2` 均复现）。
- 模型：`gfrun`（功能）与 `gfsim`（soc）均复现。
- 前端无误：LLVM IR 为 4 条独立 `store volatile i32 8`；问题在后端"均匀常量"物化路径（`hl.bfi` 广播 / 常量复用寄存器）。
