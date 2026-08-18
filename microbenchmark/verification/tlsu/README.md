# TLSU 功能验证用例

**这不是 benchmark。** 本目录下的用例只做 TLSU（`BSTART.TLSU`）搬运语义的
正确性验证：把结果 dump 出来，用独立判据（不依赖被测模型）判对错，并在不同
模型/工具链之间做差分对比。它们不测 cycle，不参与 benchmark 的用例统计，也
不被 `microbenchmark/compile_all.sh` 编译。

指令级的 TLSU **性能**微基准在 `microbenchmark/memory/`（25 个用例），与本目录
无关，不要混淆。

## 用例分组

| 组 | 用例 | 考察点 |
| --- | --- | --- |
| S | `s1_copy_i32_32x32` / `s2_strided_i32_8x128` / `s2b_stride_elem_i32_8x128` / `s3_two_dst_i32` | 单条搬运的形状、stride（按元素表达）、多目的地 |
| C | `c1_store_load_i32` / `c2_load_store_i32` / `c3_load_store_load_i32` | 组合序列下的 RAW/WAR 可见性 |
| R | `r1_random_seq_i32` | 200 块随机 TLOAD/TSTORE 序列，覆盖交叉重叠 |
| — | `copy_i32_8x128` | 最初的端到端 copy 骨架 |

公共骨架：`tlsu_bench.hpp`（pattern 生成、结果缓冲区、区域划分）、
`tlsu_finish.h`（收尾与 dump）。

## 构建

```bash
export COMPILER_DIR=/path/to/linx_blockisa_llvm_musl/bin
cd microbenchmark/verification/tlsu
make TESTCASE=s1_copy_i32_32x32
```

产物：`output/microbenchmark/verification/elf/verification/<case>.elf`。

## 运行与判定

ELF 跑在 SuperScalarModel 的 `gfsim`/`gfrun` 上，把结果缓冲区 dump 成二进制，
再用判据脚本独立核对：

```bash
# 随机用例：生成 / 可视化 / 校验 dump
python3 gen_random_case.py -o src/r1_random_seq_i32.cpp
python3 viz_random_case.py src/r1_random_seq_i32.cpp -o r1_flow.html
python3 viz_random_case.py src/r1_random_seq_i32.cpp --check dump.bin
```

`--check` 由用例源码重新推导期望结果，不读被测模型的任何中间状态，因此可以
用同一份判据比对不同模型的 dump。生成的 `*.html` 已 gitignore，随时可重新生成。
