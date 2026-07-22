# Transpose — 批量转置算子（PTO 一层 tile 版）

迁移自 `TileKernels/tile_kernels/transpose/`。tile 版，经 linx 工具链编译通过。
`36d9e45d38e204ebb87e6f6e833821eee0482fe5` 的 transpose family 只有一个
standalone module，当前 `batched_transpose_pto.hpp` 已表示。

## 已迁移

| 文件 | 对应 TileKernels | 功能 | 主指令（实际编译版） |
|------|------------------|------|---------------------|
| `batched_transpose_pto.hpp` | `transpose/batched_transpose_kernel.py` | (B,M,N)→(B,N,M) 批量转置 | `TLOAD`→`TTRANS`→`TSTORE`（硬件原生转置，替代源端 swizzle shared memory） |

## 注

- 2D 转置已由 `kernels/transpose/transpose_pto.hpp`（预存）实现，本目录只补 batch 维。
- global_iterator 仅 2 参，3D(B,M,N) 展平为 2D(B*M, N) / (B*N, M)。
- TileRows/Cols 须使 dtype 32B 对齐。
- 本轮已复核：transpose family 无其他 standalone TileKernels 模块需要新增端口。
