以 `hashtable_lookup_simd` 为例，完整流程如下：

---

## 公共目录

`{WORKSPACE}` 为公共目录，下含两个子项目：

```
{WORKSPACE}/
  LinxBlockModel/          ← QEMU（LinxBlockModel）
  JanusCoreBench/          ← Benchmark
```

在以下命令中，将 `{WORKSPACE}` 替换为实际路径，例如：

```bash
WORKSPACE=/remote/lms01/j00827727/jcore
# 或
WORKSPACE=$HOME/jcore
```

---


---

## 1. 编译 QEMU（LinxBlockModel）

```bash
cd {WORKSPACE}/LinxBlockModel/build

# Step 1：configure（cmake）
../configure --target-list=linx-linux-user,linx-softmmu

# Step 2：ninja 编译（-j32 并行 32 线程）
ninja -j 32
```

> 产物: `build/qemu-linx`

> 后续增量编译只需再执行 `ninja -j 32`，无需重新 configure。

---

## 2. 编译 hashtable_lookup_simd benchmark

### 2.1 `compile.all` 展开

`compile.all` 本质是对 Makefile 的多次调用：

```bash
cd {WORKSPACE}/JanusCoreBench/test/kernel/control

# hashtable_lookup_simd
make TESTCASE=hashtable_lookup_simd SUFFIX=_kNum409600 EXTRA_DEFINES="-DkNum=409600"
make TESTCASE=hashtable_lookup_simd SUFFIX=_kNum256 EXTRA_DEFINES="-DkNum=256"
```

每条 `make` 做了什么（以 `TESTCASE=hashtable_lookup_simd` 为例）：

---

#### Step 1 — 准备输出目录

```bash
# 自动创建，Makefile 会执行：
mkdir -p output/kernel/control/src/
mkdir -p output/kernel/control/elf/
mkdir -p output/kernel/control/hashtable_lookup_simd/data_obj/
```

---

#### Step 2 — 把数据 .data 文件转为 .o（hashtable_lookup_simd 专用）

```bash
cd {WORKSPACE}/JanusCoreBench/test/kernel/control/hashtable_lookup_simd
COMPILER_DIR=<your_compiler_dir> bash data_obj/build_data_obj.sh data_obj ../../../output/kernel/control/hashtable_lookup_simd/data_obj
```

`build_data_obj.sh` 遍历 `data_obj/` 下所有 `.data` 文件，调用：

```bash
# 对每个 *.data 文件执行：
$COMPILER_DIR/clang++ -target linx64-linx-none-elf -c *.s -o output/.../*.o
```

生成的 `.o` 文件：
```
output/kernel/control/hashtable_lookup_simd/data_obj/inserted_slot.o     # hash 表
output/kernel/control/hashtable_lookup_simd/data_obj/lookup_keys.o      # 查询 key
output/kernel/control/hashtable_lookup_simd/data_obj/lookup_values.o    # 期望 value
```

---

#### Step 3 — 编译 hashtable_lookup_simd.cpp → .o

```bash
# 实际执行（make 自动推导）：
cd {WORKSPACE}/JanusCoreBench/test/kernel/control
$COMPILER_DIR/clang++ \
  -c -target linx64-linx-none-elf -fenable-matrix -O2 \
  -std=c++20 \
  -I{WORKSPACE}/JanusCoreBench/include \
  -I{WORKSPACE}/JanusCoreBench/test/common \
  -D__linx -DENABLE_TENSOR_INSTR \
  hashtable_lookup_simd/hashtable_lookup_simd.cpp \
  -o output/kernel/control/src/hashtable_lookup_simd.o
```

---

#### Step 4 — 链接所有 .o → ELF

```bash
$COMPILER_DIR/clang++ \
  -nostartfiles \
  output/kernel/control/src/hashtable_lookup_simd.o \
  output/kernel/control/hashtable_lookup_simd/data_obj/inserted_slot.o \
  output/kernel/control/hashtable_lookup_simd/data_obj/lookup_keys.o \
  output/kernel/control/hashtable_lookup_simd/data_obj/lookup_values.o \
  -o output/kernel/control/elf/kernel_control_hashtable_lookup_simd_kNum409600.elf
```

---

### 2.2 已有 ELF 汇总

| ELF 名称 | 编译命令 |
|----------|----------|
| `kernel_control_hashtable_lookup_simd_kNum409600.elf` | `make TESTCASE=hashtable_lookup_simd SUFFIX=_kNum409600 EXTRA_DEFINES="-DkNum=409600"` |
| `kernel_control_hashtable_lookup_simd_kNum256.elf` | `make TESTCASE=hashtable_lookup_simd SUFFIX=_kNum256 EXTRA_DEFINES="-DkNum=256"` |

---

## 3. 在 QEMU 中运行仿真

```bash
cd {WORKSPACE}/LinxBlockModel
./build/qemu-linx \
  ../JanusCoreBench/output/kernel/control/elf/kernel_control_hashtable_lookup_simd_kNum256.elf \
  2>&1 | tee ../JanusCoreBench/test/kernel/control/hashtable_lookup_simd/run.log
```

**说明：**
- `2>&1`：printf 输出在 stderr，需要一起重定向
- `tee`：同时输出到屏幕和文件
- QEMU 把 ELF 路径直接当作参数，不需要 `-f`

---

## 4. 编译器版本说明

编译器路径通过环境变量 `COMPILER_DIR` 传入，不在 Makefile 或 compile.all 中写死：

```bash
export COMPILER_DIR=${WORKSPACE}/compiler/llvm/build-linxisa-clang/bin
cd {WORKSPACE}/JanusCoreBench/test/kernel/control
./compile.all
```

---

## 文件说明

路径前缀约定：`{WORKSPACE}/JanusCoreBench/test/kernel/control/hashtable_lookup_simd/`

| 文件 | 说明 |
|------|------|
| `hashtable_lookup_simd.cpp` | 源码（tile-based SIMD gather + linear probing） |
| `compute_offsets.py` | 计算 hash 偏移量的辅助脚本 |
| `gen_data.py` | 生成数据集的脚本 |
| `data_obj/` | 嵌入式数据文件（.data）及构建脚本 |
| `run_hashtable_lookup_simd.md` | 本文档 |
