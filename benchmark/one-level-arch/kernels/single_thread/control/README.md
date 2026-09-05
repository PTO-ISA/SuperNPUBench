# Control (PTO one-level-arch)

## 功能

纯 tile-op 哈希表查找 (`hashtable_lookup_simd`)，使用 MurmurHash3_x86_32 哈希函数，
开放寻址 + 线性探测。无 `__vec__`/SIMT，无数据依赖的提前退出
（运行所有 `kMaxProbe` 次迭代，`TSELECT` 保持已找到的结果不受影响）。

## 输入输出

| 参数 | 类型 | 说明 |
|------|------|------|
| `out` | `int32_t*` | 查询结果值 (未找到 = -1) |
| `table` | `TableEntry*` | 哈希表 (int64 key + int32 value) |
| `queries` | `int64_t*` | 查询键 |

## Tile 类型

| Tile | 类型 | 用途 |
|------|------|------|
| Query tile | `Tile<Vec, int64_t, kTileRows, kTileCols, RowMajor>` | 查询键 |
| Key/Val tile | `Tile<Vec, uint32_t, kTileRows, kTileCols, RowMajor>` | 哈希计算中间值 |
| Offset tile | `Tile<Vec, uint32_t, ...>` | 探测偏移 |

## 调用的 TileOp

| 操作 | 说明 |
|------|------|
| `TSHLS` / `TSHRS` / `TOR` | 32 位循环左移 (rotl32) |
| `TCVT` / `TMULS` / `TXOR` / `TADDS` | MurmurHash3 混合 |
| `TREM` / `TMULS` / `TADD` | unsigned mod (hash % capacity) |
| `TLOAD` | 加载查询键 |
| `TEXPANDS` | 初始化输出为 -1 |
| `MGATHER` | 按探测偏移从表中 gather key+value |
| `TCMP` | 比较 hi/lo 32 位键 |
| `TSEL` | 选择匹配结果到输出 |
| `TSTORE` | 存储结果 |

## 实现方式

1. **哈希计算** (`computeProbeOffsets`): 64 位 key 拆分为两个 32 位块，
   分别 MurmurHash3 混合 → 合并 → `% kCap` → 乘 `sizeof(TableEntry)` 得字节偏移
2. **探测循环** (`runHashFind`): `kMaxProbe` 次迭代，无提前退出
   - `MGATHER` 从 table[probe_off] 读取 key+value
   - `TCMP` 比较 hi/lo 32 位
   - `TAND` 合并匹配掩码
   - `TSEL` 将匹配的 value 选择到输出 (已匹配的 lane 不再更新)
   - 推进 probe_off: `(probe_off + entrySize) % (kCap * entrySize)`
3. `TSTORE` 输出

## 注意事项

- 纯 tile-op 算子，gfsim 需 `-s core.singleTierMode=true`
- `.data` 文件由 `gen_data.py` 在首次编译时生成

## 源文件

| 文件 | 说明 |
|------|------|
| `hashtable_lookup_simd.hpp` | MurmurHash3 哈希查找实现 |
