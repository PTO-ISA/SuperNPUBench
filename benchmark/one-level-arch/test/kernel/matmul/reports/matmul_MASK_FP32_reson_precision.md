# Matmul (MASK_FP32) reson 精度验证报告

- 日期: 2026-08-17
- 算子: `matmul.cpp` → `MASK_FP32` 分支，内核 `matmul_mask_tileop<float,...>`
- 配置: `TYPE=MASK MODE=MASK_FP32 res_check=on`（reson，读/写文件）
- 尺寸: M=N=K=64, tM=tN=tK=16（4 tile/dim，64 个输出 tile，每 tile 4 次 K-累加，tile 对齐无 partial-tile 掩码）

## 1. 环境

| 项 | 值 |
|---|---|
| 工具链 | latest worktree: `linx-toolchain-build-latest/output/linx_blockisa_llvm_musl/bin` |
| 编译器 | clang 15.0.4（llvm-project 86959776b） |
| 功能模型 | gfrun（PTO-ISA BlockISA functional model），`SuperScalarModel/bin/gfrun` |
| 链接 | hosted（`res_check=on` 触发 `CC_LINK=` 清空，默认 crt0，非裸机 `_start.s`） |
| 编译宏 | `-DRES_CHECK -DENABLE_BINARY_OUTPUT -DCHK_DIR="..." -DMASK_FP32` |
| ELF | `output/kernel/matmul/elf/kernel_matmul/matmul_MASK_MASK_FP32_M64_N64_K64_tM16_tN16_tK16.elf` |
| 输入/输出 | `compare/matmul_MASK_MASK_FP32_M64_N64_K64_tM16_tN16_tK16/{src0,src1,res}.bin`（FP32，4096 元素/16 KiB 每个） |

## 2. 方法

1. **编译**: `make TESTCASE=matmul TYPE=MASK MODE=MASK_FP32 M=64 N=64 K=64 tM=16 tN=16 tK=16 res_check=on COMPILER_DIR=<latest>`。
2. **执行**: `gfrun -f <elf> -m 200000`（block 预算充足，避免假挂死）。gfrun 接管 `open/read/write/close` syscall，将 host 文件读到 guest 内存、写出 `res.bin`。
3. **取证（printf 无关）**: 直接 dump `src0.bin/src1.bin/res.bin`，用 numpy 以 **float64** 计算 `golden = A@B`，与 gfrun 写出的 FP32 `res.bin` 逐元素比对。
4. **两组输入**:
   - **全 1**（correctness 基线）：黄金 `C[i][j]=Σ1=K=64`，FP32 下精确可表。
   - **随机 uniform[-1,1]**（精度主体，seed=42）：逼出 FP32 累加舍入误差。

> 注：该裸机环境 `printf` 对 `%d/%s` 处理有 bug（write 的 "data write done" 显示错文件名），但**底层 open/write/close 与算子计算正确**；所有结论均基于文件 dump 直接比对，与 printf 无关。

## 3. 结果

### 3.1 执行（两组一致）

| 指标 | 值 |
|---|---|
| 完成 block 数 | 4037 |
| 完成 inst 数 | 18096 |
| 退出码 `R2` | 0（正常完成） |
| 文件 I/O | read src0 ✓ / read src1 ✓ / write res ✓ |
| 是否挂死 | **否** —— 4037 blocks 内正常结束 |

### 3.2 全 1 输入（correctness）

| 指标 | 值 |
|---|---|
| 黄金 | `C[i][j] = 64`（处处相等） |
| `res.bin` 实测 | 处处 `64.0`（`0x42800000`） |
| max abs diff | **0.0** |
| 命中/总数 | 4096 / 4096 |
| 结论 | **PASS**（bit-exact） |

### 3.3 随机输入（精度）

`golden = float64(A@B)`，与 gfrun FP32 `res.bin` 比对：

```
C[0,:4]      = [-3.23280382 -2.81466198  2.88366103 -0.40265819]   # gfrun FP32
golden[0,:4] = [-3.23280345 -2.81466245  2.88366135 -0.40265838]   # float64 参考
```

| 指标 | 值 |
|---|---|
| max\|C\| / max\|golden\| | 9.7836 / 9.7836 |
| **max abs err** | **9.81e-07** |
| mean abs err | 1.86e-07 |
| mean rel err | 1.39e-06 |
| max rel err | 2.62e-03（出现在 golden≈0 的抵消格，abs err 仍 ~1e-7，属 FP32 正常） |
| 容差判定 (`atol=rtol=1e-3`) | **PASS** |

## 4. 分析

### 4.1 精度解读

随机输入下 FP32 结果与 float64 参考的**绝对误差 ~1e-6**，恰为 FP32 精度预期量级：float32 机器 ε≈1.19e-7，64 项累加（√K≈8 放大）→ 期望 ~1e-6，实测 9.81e-7，吻合。说明 `TMATMUL FP32` + 跨 4 个 K-tile 的 `TMATMUL.ACC` 累加路径计算正确，仅含 FP32 本征舍入，无模型/算子缺陷。

`max rel err=2.62e-3` 出现在 `golden≈0` 的格（catastrophic cancellation），此时绝对误差仍 ~1e-7，相对误差被近零分母放大，属 FP32 正常现象，不计为缺陷。

### 4.2 "挂死" 澄清

先前用 `-m 3050` 跑 reson 时呈现"挂死"，经 `-m 200000` 复测证伪：reson 需 ~4037 blocks（hosted crt0 启动 + 3 次文件 syscall + matmul + write 的 printf 格式化开销），`-m<3725` 会在 write 的 printf 字符串处理中途被 block 上限截断，造成假挂死。给足预算即正常结束（R2=0）。issue 文档里"reson FAIL"应为另一配置（M=256 无预置 bin 或旧 gfrun 版本），与本次 M=64 + 最新 gfrun 实测不符。

### 4.3 与裸机 validate 版对比

| 维度 | 本报告（reson / 普通 matmul） | matmul_validate（裸机） |
|---|---|---|
| 链接 | hosted crt0 | 裸机 `_start.s`（nostartfiles） |
| 文件 I/O | 有（open/read/write syscall） | 无（代码内填全 1） |
| 输入 | src0.bin/src1.bin | 编译期常量 |
| 取证 | dump res.bin 比对 | dump 全局 g_dst 比对 |
| M=64 完成 block | 4037 | 18918（含标量填值/校验循环） |
| 全 1 数值 | PASS，diff 0.0 | PASS，diff 0.0 |

两者用同一 `matmul_mask_tileop` 内核，数值结论一致。

## 5. 结论

**普通 matmul（MASK_FP32）在 reson 配置（M=N=K=64, tM=tN=tK=16）下：**

- **不挂死**：4037 blocks / 18096 insts / R2=0 正常完成。
- **正确性**：全 1 输入 bit-exact（4096 格全 == 64，diff 0.0）。
- **精度**：随机输入下 FP32 结果 vs float64 参考，max abs err 9.81e-7、mean rel err 1.4e-6，达 FP32 本征精度，无算子/模型缺陷。
- **综合判定：PASS**。

## 6. 复现

```bash
# 工具链
TC=/Users/blacktraker/Programming/gitproj/DV4/linx-toolchain-build-latest/output/linx_blockisa_llvm_musl/bin
GFRUN=/Users/blacktraker/Programming/gitproj/DV4/SuperScalarModel/bin/gfrun
ROOT=/Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench/benchmark/one-level-arch
CHKD=$ROOT/compare/matmul_MASK_MASK_FP32_M64_N64_K64_tM16_tN16_tK16

# 1. 生成全 1 输入 bin
mkdir -p $CHKD
python3 -c "import struct;n=64*64;open('$CHKD/src0.bin','wb').write(struct.pack('<%df'%n,*([1.0]*n)));open('$CHKD/src1.bin','wb').write(struct.pack('<%df'%n,*([1.0]*n)))"

# 2. 编译（普通 matmul，reson）
cd $ROOT/test/kernel/matmul
make TESTCASE=matmul TYPE=MASK MODE=MASK_FP32 M=64 N=64 K=64 tM=16 tN=16 tK=16 \
  res_check=on COMPILER_DIR=$TC

# 3. 跑 gfrun（全 1）
$GFRUN -f $ROOT/output/kernel/matmul/elf/kernel_matmul/matmul_MASK_MASK_FP32_M64_N64_K64_tM16_tN16_tK16.elf -m 200000

# 4. 数值比对（全 1 → 64.0）
python3 - <<'PY'
import struct,numpy as np
D="CHKD_PLACEHOLDER"  # 替换为上面的 $CHKD
def rd(f):
    b=open(D+"/"+f,"rb").read();n=len(b)//4
    return np.array(struct.unpack("<%df"%n,b),dtype=np.float32).reshape(64,64)
C=rd("res.bin");golden=np.full((64,64),64.0)
print("max diff:",np.abs(C-golden).max())
PY

# 5. 随机精度（同二进制，换 bin 重跑）
python3 -c "
import numpy as np,os;np.random.seed(42);n=64*64
a=(np.random.rand(n)*2-1).astype('<f4');b=(np.random.rand(n)*2-1).astype('<f4')
open(os.environ['CHKD']+'/src0.bin','wb').write(a.tobytes())
open(os.environ['CHKD']+'/src1.bin','wb').write(b.tobytes())"
$GFRUN -f $ROOT/output/kernel/matmul/elf/kernel_matmul/matmul_MASK_MASK_FP32_M64_N64_K64_tM16_tN16_tK16.elf -m 200000
# 再用上面 python 比对，golden 改为 A@B（float64）
```

> 复现后如需恢复全 1 默认 bin：见步骤 1 重新生成。
