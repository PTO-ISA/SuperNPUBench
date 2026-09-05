# Sort / TopK (PTO one-level-arch)

## 功能

TopK (k=2048, 从 131072 个 uint16 元素中选出前 2048 大的)。
使用两阶段 radix-bucket 直方图法:
1. high8 直方图 → 确定 kth_bin
2. low8 直方图 → 确定 low8_boundary
3. 标量掩码 scatter 收集结果

## 输入输出

| 参数 | 类型 | 说明 |
|------|------|------|
| `src` | `const uint16_t*` | 输入 (131072 元素) |
| 输出 | `uint16_t*` | TopK 结果 (2048 元素, 降序) |

## Tile 类型

| Tile | 类型 | 用途 |
|------|------|------|
| DataTile | `Tile<Vec, uint16_t, 1, 256, RowMajor>` | 输入数据 tile |
| HistTile | `Tile<Vec, uint32_t, 1, 256, RowMajor>` | 256-bucket 直方图 |

## 调用的 TileOp

| 操作 | 说明 |
|------|------|
| `TEXPANDSCALAR_Impl` | 初始化直方图 tile 为 0 |
| `TLOAD` | 加载输入数据 tile |
| `TSHRS` | 提取 high8 = val >> 8 |
| `TANDS` | 提取 low8 = val & 0xFF |
| `TEXPANDS` | 广播 bucket 常量 |
| `TCOPYOUT` | 存储直方图到 global memory |

## 实现方式

### Phase 1: high8 直方图 (`ExtractHigh8Hist_Impl`)
- `TEXPANDSCALAR_Impl(hist, 0)` 初始化
- 循环 kNumTiles 次: `TLOAD` 256 元素 → `TSHRS` 提取 high8 →
  per-bucket 比较 (当前为部分 tile-op，标量 fallback 补充)

### Phase 2: 标量前缀扫描 (`find_kth_bin`)
- 从 bin 255 向下累加，找到 cumsum ≥ k 的 bin
- 返回 kth_bin 和 need_from_kth_bin (该 bin 内还需多少)

### Phase 3: low8 直方图 (`ExtractLow8HistForKthBin_Impl`)
- 仅对 high8 == kth_bin 的元素计算 low8 直方图

### Phase 4-5: 标量掩码 scatter
- high8 > kth_bin: 直接收集
- high8 == kth_bin 且 low8 >= boundary: 收集

## 注意事项

- 使用 `TEXPANDSCALAR_Impl` (非 `TEXPANDSCALAR`)，需 include `jcore/TExpandScalar.hpp`
- 已适配最新 tileop-api（移除了对本地 `template_asm.h` 的依赖）

## 源文件

| 文件 | 说明 |
|------|------|
| `topk.hpp` | PTO tile-op 直方图提取 |
| `topk.hpp` | topk 完整实现 |
