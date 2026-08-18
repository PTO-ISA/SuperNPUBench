# Issue #61 复现报告：matmul res_check 精度通路

> 对应 issue：<https://github.com/PTO-ISA/SuperNPUBench/issues/61>
> 标题：matmul: res_check 精度通路在 SoftCore 上不通（功能 R2=0 PASS，golden_cmp 无法比对）
> 复现日期：2026-08-17

## 结论

**部分复现**——issue 报的 `AssertNotTextStore`（CRT/text-store，读 bin 前崩）**已不复现**，该根因已被 `0e9bff9b` 修复；但 `res_check` 精度通路在 `M=256 tM=32 tK=64` 下仍跑不通，卡点已变（见下）。

---

## 环境与分支

各仓当时分支/HEAD（issue 用的 vs 我本地）：

| 项 | issue 用的 | 我本地 |
|---|---|---|
| SuperNPUBench | `release_ver0812` @ `ba1acfb` | `release_ver0812` @ `e13d54d`（2026-08-13 22:28，工作树含未提交的 FP16 `writeBinaryFile` 补丁）|
| SuperScalarModel | `feat/pto-v058-adaptation` @ `5a4d2774`（2026-08-13 12:51）| `feat/pto-v058-adaptation` @ `c3051e3a`（2026-08-17 11:49）|
| Linx-TileOP-API | `temp/shared-tload-integration-20260811` @ `abe8411`（2026-08-13 09:43）+ `-resource-dir` overlay | detached @ `6a4378428e4f`（2026-08-14 11:05，用工具链 install 内置头，无 overlay）|
| llvm-project | `temp/shared-tload-integration-20260811` @ `eb64de8`（2026-08-11 21:32）| detached @ `86959776bd1f`（2026-08-14 09:58，`clang++ --version` 内嵌一致）|
| linx-toolchain-build | `main` @ `e6a31ef`（2026-06-26）| detached @ `e6a31ef`（2026-06-26，同 issue）|
| pto-spec | `main` @ `2c663ce` | 本机未用 |
| gfrun 二进制 mtime | 2026-08-17 10:27（建于 `5a4d2774`）| 2026-08-17 14:31（建于 `c3051e3a`）|

差异要点：

- **SuperScalarModel**：我本地 `c3051e3a` 含 `0e9bff9b`，issue `5a4d2774` 不含——`AssertNotTextStore` 不复现的直接原因。
- **工具链**：本地用 latest 工具链（llvm `86959776b` / TileOP `6a43784`，无 `-resource-dir` overlay），非 issue 的 shared-tload 工具链（llvm `eb64de8` / TileOP `abe8411` + overlay）；按"用本地"原则未另配 shared-tload。
- gfrun 运行：`SuperScalarModel/bin/gfrun -f <elf>`，单次，无 `-t 1`，timeout 90s。

---

## 一、复现结果

### 版本对照

| 项 | issue 用的 | 我本地 |
|---|---|---|
| SuperScalarModel | `5a4d2774`（2026-08-13）| `c3051e3a`（2026-08-17 11:49）|
| gfrun 二进制 mtime | 2026-08-17 10:27 | 2026-08-17 14:31 |
| `AssertNotTextStore`（`AaccelssMemoryEngine.cpp`）| 在 | 已删 |

关键：issue 报的崩溃源 `AssertNotTextStore` 被 **`0e9bff9b fix(tlsu): align memory and scatter semantics with PTO ASL`（2026-08-15）** 整个移除（commit 原文："Remove gfrun-only text-store write rejection; ASL memory model only bounds-checks the address space and does not forbid text writes."）。`0e9bff9b` 不是 `5a4d2774` 的祖先、但是 `c3051e3a` 的祖先——所以 issue 那版 gfrun 还带着这条断言、我本地已没有。

按 issue 的命令（`MASK M=256 N=256 K=256 tM=32 tN=32 tK=64 res_check=on` + `golden_cmp.py --ones`）实跑：

| 路径 | dtype | gfrun 退出 | 现象 |
|---|---|---|---|
| 精度 reson | FP16 | 0（R2=0）| 读 src0/src1 成功，跑完 3746 块，但 stdout 无 "data write done"，res.bin 全 0 |
| 精度 reson | FP32 | 1（非法指令）| 读完两 bin 后，matmul 中撞 `ValidateLocalTlsu`：`AccumulateBlockInfo.cpp:74 "Local TLOAD requires one fitting destination Tile"` |
| 功能 resoff（对照）| FP32 | 1 | 同一条 TLOAD 断言——**非 reson 专有** |

观察：

- reson **已能正常读 `src0.bin`/`src1.bin`**（hosted musl libc + 文件 I/O 在 gfrun 上通了），issue 报的"读 bin 之前就崩"不复现。
- FP16 不撞 TLOAD（`[32,64]` fp16 = 4KB 放得下），跑通 R2=0，纯粹败在没写 res.bin。
- FP32 撞的 TLOAD 断言（`a29c5a4a4`，2026-08-11 引入）在 issue 的 `5a4d2774` 里**就已存在**，只是当时被 `AssertNotTextStore` 挡在 CRT、没跑到 matmul，所以 issue 没提到。`0e9bff9b` 移除第一层后，第二层才露出来。

---

## 二、错误原因

### 1. `AssertNotTextStore`（issue 报的根因）→ 已修复

`0e9bff9b`（2026-08-15）删除了 `AaccelssMemoryEngine.cpp` 里的 `AssertNotTextStore` 函数及全部调用点。reson 链 musl libc，libc 启动用 `addtpc` 做 PC 相对寻址会写 `.text`（运行时重定位），按 PTO ASL 内存模型这是合法的（只做地址越界检查），旧的 gfrun-only 断言过严。修复后 reson 能过 CRT、正常 `open`/`read`/`write` 文件。

### 2. FP16 res.bin 全 0 → bench 源码漏写

`test/kernel/matmul/src/matmul.cpp` 的 MASK_FP16 分支在 `BENCHEND` 后**没有 `writeBinaryFile`**，而 MASK_FP32 分支（`:100`）和 MASK_FP8 分支（`:194`）都有。即 FP16 算了 `dst` 却从不写 res.bin，`golden_cmp` 看到的是它自己预写的全 0 → compare FAIL。一行可补的 bench 侧 bug，与 gfrun 无关。

### 3. FP32 TLOAD 断言 → gfrun 侧（非 reson 专有）

`ValidateLocalTlsu`（`AccumulateBlockInfo.cpp:74`）拒绝 `[tM=32, tK=64]` 的 fp32 tile（8KB）：断言要求

```
inst->dsts[0]->size % (physicalCol * elementBytes) == 0
&& inst->dsts[0]->size >= validRow * physicalCol * elementBytes
&& "Local TLOAD requires one fitting destination Tile"
```

该断言在 resoff（功能路径）同样复现，所以与 `res_check` 无关，是当前 gfrun 对该 tile 形状的装配校验不通过（是 tile 真非法 per ASL，还是 gfrun 的 `physicalCol`/`validRow` 计算有偏差，需 gfrun 侧进一步定位）。

---

## 三、本地复现步骤

环境：bench 当前 main；工具链 `linx_blockisa_llvm_musl`（clang 15.0.4）；gfrun 用本地 `SuperScalarModel/bin/gfrun`（`c3051e3a`）。

```bash
cd benchmark/one-level-arch/test/kernel/matmul
export COMPILER_DIR=/path/to/linx_blockisa_llvm_musl/bin

# 1) 编译精度 ELF（FP16；FP32 把 MODE 换成 MASK_FP32）
make TESTCASE=matmul TYPE=MASK MODE=MASK_FP16 \
  M=256 N=256 K=256 tM=32 tN=32 tK=64 res_check=on

# 2) golden_cmp（自动生成 src0/src1/golden.bin 到 CHK_DIR，跑 gfrun，比 res.bin）
python3 src/golden_cmp.py \
  -d output/kernel/matmul/elf/kernel_matmul/matmul_MASK_MASK_FP16_M256_N256_K256_tM32_tN32_tK64.elf \
  --ones --gfrun-args=-f --timeout 90 --workers 1 --atol 2e-2 --rtol 2e-2

# 3) 或直接裸跑 gfrun 看 stdout
/path/to/SuperScalarModel/bin/gfrun -f \
  output/kernel/matmul/elf/kernel_matmul/matmul_MASK_MASK_FP16_M256_N256_K256_tM32_tN32_tK64.elf
```

`golden_cmp.py` 会把 `src0.bin`/`src1.bin`/`golden.bin`/`res.bin`（预写全 0）写到
`benchmark/one-level-arch/compare/matmul_MASK_MASK_FP16_M256_N256_K256_tM32_tN32_tK64/`，正好等于 kernel 的 `CHK_DIR`。

---

## 四、本地复现结果

### FP16（修复前 → 修复后）

```
# 修复前（原 bench）：golden_cmp FAIL
run_status='pass'  compare_status='fail'
mse=65536.0  max_abs=256.0  actual_min=0.0 actual_max=0.0  golden=256.0
# stdout 只有 "data read to file done: src0.bin / src1.bin"，无 "data write done"
```

给 FP16 分支补 `writeBinaryFile` 后（见下 diff）：

```
# golden_cmp PASS
run_status='pass'  compare_status='pass'
mse=0.0  max_abs=0.0  actual_min=256.0 actual_max=256.0  golden=256.0
res.bin: 65536 elements, uniq=[256.0]   summary: pass=1, fail=0
```

即 gfrun 的 FP16 matmul 数值正确（256 = K 精确），FP16 唯一问题就是漏写。

FP16 一行修复 diff（`matmul.cpp` FP16 分支 `BENCHEND` 后补，仿 FP32:100 / FP8:194）：

```diff
     BENCHEND;

+    #ifdef RES_CHECK
+    #define RES_PATH CHK_DIR "/res.bin"
+    writeBinaryFile(RES_PATH, (uint8_t*)dst, globM*globN*sizeof(float));
+    #endif
+
   #elif defined(MASK_FP8) || ...
```

### FP32

```
golden_cmp FAIL: run_status='fail' returncode=1
gfrun: illegal instruction: ASSERTION FAILED ... "Local TLOAD requires one fitting destination Tile"
  at ValidateLocalTlsu, emulator/engine/AccumulateBlockInfo.cpp:74
（读完 src0/src1 后、matmul 中触发；resoff 同样触发）
```

gfrun stdout 关键行：

```
data read to file done: .../src0.bin
data read to file done: .../src1.bin
gfrun: illegal instruction: ASSERTION FAILED: ... && "Local TLOAD requires one fitting destination Tile"
, func ValidateLocalTlsu, file .../emulator/engine/AccumulateBlockInfo.cpp:74
```

---

## 小结

- issue #61 报的 `AssertNotTextStore` 根因 → **已修（`0e9bff9b`）**，不复现；reson 已能正常读输入 bin。
- 精度闭环在 `M=256 tM=32 tK=64` 仍不通，卡点变了：
  - **FP16 = bench 漏写 res.bin**（一行修复即 PASS，附 diff）；
  - **FP32 = gfrun `ValidateLocalTlsu` 对 8KB tile 装配校验失败**（resoff 亦复现，非 reson 专有，建议 gfrun 侧定位是 ASL 非法 tile 还是 `physicalCol`/`validRow` 计算偏差）。
