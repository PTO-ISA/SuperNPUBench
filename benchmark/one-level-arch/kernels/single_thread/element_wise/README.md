# GELU (PTO one-level-arch)

## 功能

GELU 激活函数。使用多项式拟合实现: `GELU(x) = x / (1 + exp(t·P(t²)))`，
其中 P 是 7 阶 Horner 多项式。FP16 输入输出，FP32 中间计算。纯 tile-op 实现。

## 输入输出

| 参数 | 类型 | 说明 |
|------|------|------|
| `in_ptr` | `dtype*` | 输入 (1×gM, FP16) |
| `out_ptr` | `dtype*` | 输出 (1×gM, FP16) |
| `approximate` | `bool` | (保留) 近似模式选择 |

## Tile 类型

| Tile | 类型 | 用途 |
|------|------|------|
| 数据 tile | `Tile<Vec, dtype, 1, tM, RowMajor>` | FP16 输入/输出 |
| FP32 tile | `Tile<Vec, float, 1, tM, RowMajor>` | 中间计算 |

Tail 变体使用 `ValidCol` 处理非整除列。

## 调用的 TileOp

| 操作 | 说明 |
|------|------|
| `TLOAD` | 加载 FP16 输入 tile |
| `TCVT` | FP16 → FP32 |
| `TMAXS` / `TMINS` | clamp 到 ±5.75 (多项式收敛域) |
| `TMUL` | 计算 t² = x·x |
| `TMULS` + `TADDS` | Horner 多项式求值 ×6 步 |
| `TMUL` | t·P(t²) |
| `TEXP` | exp(t·P) |
| `TADDS` | 1 + exp(...) |
| `TRECIP` | 1 / (1 + exp) |
| `TMUL` | x · (1/denom) = GELU(x) |
| `TCVT` | FP32 → FP16 |
| `TSTORE` | 存储输出 |

## 实现方式

对每个 tile (Mb + tail): `TLOAD` FP16 → `gelu_impl` 内部:
1. `TCVT` FP16→FP32
2. clamp 到 ±5.75
3. 计算 t²
4. 7 阶 Horner 多项式 (6 步 `TMULS`+`TADDS`)
5. 计算 t·P(t²)
6. `TEXP` + `TADDS` + `TRECIP` = 1/(1+exp)
7. `TMUL` x·(1/denom) = y
8. `TCVT` FP32→FP16
→ `TSTORE`。

## 源文件

| 文件 | 说明 |
|------|------|
| `gelu.hpp` | GELU 多项式拟合实现 |
