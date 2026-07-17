# DeepSeek 迁移算子（PTO 一层 tile 版）

本目录收录从 [TileKernels](https://github.com/deepseek-ai/TileKernels)（基于 TileLang DSL，
NVIDIA SM90/SM100）迁移到 SuperNPUBench one-level-arch（PTO Tile ISA）的算子。

## 总览

- **说明文档**：[`TileKernels迁移说明.md`](TileKernels迁移说明.md) — 19 个算子总表、分阶段说明、
  核心迁移模式、工具链约束速查、验证方法。
- **编译验证**：`_compile_test.cpp` 实例化全部算子，经 `linx_blockisa_llvm_musl` 编译+链接为 ELF。
  ```
  export COMPILER_DIR=/path/to/linx_blockisa_llvm_musl/bin
  $COMPILER_DIR/clang++ -c -mlxbc -fenable-matrix -O2 -mllvm -enable-all-vector-as-tilereg=true \
      -std=c++20 -D__linx -DENABLE_TENSOR_INSTR -Ikernels -Itest/common -Itest/common/src \
      -Iinclude -Imodels kernels/deepseek/_compile_test.cpp -o /tmp/tk.o
  $COMPILER_DIR/clang++ -nostartfiles test/common/_start.s /tmp/tk.o -o /tmp/tk.elf
  ```

## 模块

| 模块 | 算子数 | 说明 |
|------|:---:|------|
| [engram/](engram/README.md) | 2 | 权重融合、n-gram 哈希 |
| [mhc/](mhc/README.md) | 5 | MHC 扩展(fwd/bwd)、Sinkhorn、RMSNorm、多层重算 |
| [moe/](moe/README.md) | 8 | top-k 门控、权重归一化、负载统计、扩展/归约/去重/映射 |
| [quant/](quant/README.md) | 3 | 反量化、行/列级量化、融合 SwiGLU+量化 |
| [transpose/](transpose/README.md) | 1 | 批量 3D 转置 |

## 重要说明

- 工具链实际暴露的指令集**窄于** Linx-TileOP-API 文档全集。不可用指令（如 `TROWARGMAX`/
  `THISTOGRAM`/`TFMA`/`TABS`/`TRSQRT`/`TROWEXPANDDIV`）已用可用指令模拟，详见各 `.hpp`
  头部注释与 `TileKernels迁移说明.md` §七"核心迁移模式"。
- 全部算子 tile 版（非标量回退）。
