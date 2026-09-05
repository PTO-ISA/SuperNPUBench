# Matmul (PTO one-level-arch)

## 功能

矩阵乘法 `C = A × B`，支持多种精度（FP32/FP16/FP8/BF16/HiF4）和混合精度（A16W4、MXFP4）。
提供多个优化变体：基础分块、A-tile 复用、K 轴分块、Vec 寄存器累加。

## 输入输出

| 参数 | 类型 | 说明 |
|------|------|------|
| `c_ptr` / `dst` | `float*` | 输出矩阵 C (gM×gN, FP32, RowMajor) |
| `a_ptr` / `src0` | `dtype*` | 输入矩阵 A (gM×gK) |
| `b_ptr` / `src1` | `dtype*` | 输入矩阵 B (gK×gN) |
| `src0_mx` | `uint8_t*` | (MX 变体) A 的 per-tile 缩放因子 |
| `src1_mx` | `uint8_t*` | (MX 变体) B 的 per-tile 缩放因子 |

## Tile 类型

| Tile | 类型 | 用途 |
|------|------|------|
| `TileLeft<dtype, tM, tK>` | L0A, boxed 512B | A 操作数 |
| `TileRight<dtype, tK, tN>` | L0B, boxed 512B | B 操作数 |
| `TileAcc<float, tM, tN>` | L0C, boxed 1024B | C 累加器 (FP32) |
| `Tile<Vec, float, tM, tN>` | Vec | Acc→Vec 转换后存储 |

Tail 变体使用 `ValidRow`/`ValidCol` 处理非整除维度。

## 调用的 TileOp

| 操作 | 说明 |
|------|------|
| `TLOAD` / `TCOPYIN` | 加载 A、B tile |
| `TMATMUL` | 首个 K 块: `C = A × B` |
| `TMATMUL_ACC` | 后续 K 块: `C += A × B` |
| `MATMULMX` / `MATMACCMX` | MX 混合精度 GEMM (带缩放因子) |
| `ACCCVT` / `TCVT` | Acc → Vec 类型转换 |
| `TADD` | OPT2 变体: Vec 寄存器累加 |
| `TSTORE` / `TCOPYOUT` | 存储 C 到 global memory |
| `MGATHER` | MX 变体: 缩放因子 gather |

## 实现方式

### 基础变体 (`matmul_mask`)
三重 M/N/K 循环。首个 K 块用 `TMATMUL`（初始化），后续用 `TMATMUL_ACC`（累加）。
四象限 tail 处理：full / col-tail / row-tail / corner，每个使用独立的 `ValidRow`/`ValidCol` tile 类型。

### A-tile 复用 (`matmul_mask_reuseA`)
预加载 `tA[R.m][R.k]` 寄存器数组，A tile 在 N 列方向复用，减少 TLOAD 次数。
`R.m × R.k` 由 `constexpr find_reuseA()` 在 `MAX_TILE_NUM` 约束下搜索最优。

### K 分块优化 (`matmul_mask_reuseA_OPT`)
当 `Kb > R.k` 时，将剩余 K 轴分块处理。

### Vec 寄存器累加 (`matmul_mask_reuseA_OPT2`)
部分和驻留在 Vec tile (`tC_main[Mb][Nb]`) 中，用 `TCVT + TADD` 跨 K 块累加，
减少 Acc tile 占用，输出改为 ColMajor。

### MX 混合精度 (`matmul_mxfp`)
使用 `MATMULMX`/`MATMACCMX`，额外加载 `Tile<Scaling, uint8_t>` 缩放因子 tile。
缩放因子通过 `gen_ND2ZZ_offset` + `MGATHER` 获取。

## 源文件

| 文件 | 说明 |
|------|------|
| `matmul.hpp` | _tileop 后缀变体 |
| `matmul.hpp` | PTO 命名变体 |
| `matmul_mx.hpp` | MX 混合精度 (_tileop) |
| `matmul_mx.hpp` | MX 混合精度 (PTO) |
| `matmul_shared.hpp` | matmul的A，B矩阵均从gm加载到sharedtreg版本 |
