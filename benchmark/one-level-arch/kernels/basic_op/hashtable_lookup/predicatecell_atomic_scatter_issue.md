# [LinxV5] PredicateCell atomic-add / scatter legalization 缺失（forced element-tile-mask 路径）

- 状态：OPEN
- Issue：https://github.com/LinxISA/llvm-project/issues/153
- 证据 gist：https://gist.github.com/lvhao7896/6cbe1fe479da6db66237f0528c0581ca
- 归档日期：2026-11-05
- 编译器：LinxISA/llvm-project `a75987bcc1d9cf5fcd98c1c64c6ca03badbb6919`
  （PR #146 `a5f03b38` + #151 修复 `71351b61`，clang 15.0.4）
- PTO API：PR #303 `decac4f`

## 现象

带 `-mllvm -linxv5-force-element-tile-mask=true` 时：

- 带 mask 的 per-element 原子加（结果保留）：
  `slots[e] = __atomic_fetch_add(&count, 1, ...)`
  → `Tile legalization: PredicateCell atomic-add legalization is not implemented`
- per-element scatter（每 region 一条输出流）：
  `qkey[slot[e]] = key[e]`
  → `Tile legalization: PredicateCell unordered scatter legalization is not implemented`

同样的源码**去掉该 flag** 可以编译通过（exit 0）。

## A/B 证据（同一命令行，只差 flag）

| 源文件 | 默认路径 | forced mask 路径 |
|---|---|---|
| `atomic_tile_plain.cpp` | exit 0 | atomic-add legalization 未实现 |
| `split_streams_control.cpp`（#134 控制用例） | exit 0 | 同上 atomic-add 错误 |
| `scatter_regions_control.cpp`（三 region 各一条 scatter） | exit 0 | unordered-scatter legalization 未实现 |
| `two_streams.cpp`（#134 失败用例，单 region 两条流） | one isolated output stream | 同左（与 flag 无关） |

## 用途与影响

256-lane hashtable 探测 kernel 想把每批未命中的 lane 压缩成稠密 GM 队列：
每条存活 lane 用原子取号，再用三条 4 B scatter 写 key/offset/out_index。
这两条 legalization 缺失导致队列无法生成，稀疏 lane 汇聚方案被阻断。

与 #134（单 region 多 stream）正交——我们的设计是"每个 region 一条 stream"。
相关：#133（atomic 区域的 uniform base）、#132/#151（region ownership）。

## 重要发现（变通）

同一个 256-lane kernel 在**有/无该 flag** 下编译产物**逐字节相同**
（SHA256 `08f58b96…`，130,114 cycles 相同），所以对本 kernel 而言去掉 flag
是可行变通；但真正需要 forced mask 路径的 kernel 没有这个变通。

## 期望

在 forced-mask 路径补齐两条 legalization（inactive lane 不访问内存、atomic
返回旧值、scatter 每 region 单输出流），或明确文档化该限制和推荐写法。

## 本地证据

`/tmp/iss151b/`（源文件、`*_forced.log`、`*_plain.log`、`reproduce.sh`、
`EVIDENCE_digest.md`、`control_results.txt`）；远程备份见上面的 gist。

## 2026-11-05 复测更新（0.59 分支）

- 用 0.59 分支 `f5a45639c` 重建编译器（`/Users/blacktraker/Programming/gitproj/DV4/llvm-project-059/build-059`）。
- gist 四份源码在 O2/O3 全部生成 object（8/8）；`-linxv5-force-element-tile-mask` 在 0.59 上是 unknown option。
- 反汇编确认：`alloc_slots` → `MGATHER.ADD`，`scatter_regions` → `MSCATTER`。issue #153 的两个 PredicateCell legalization 缺失在 0.59 上已不存在。
- 但 256-lane 本 kernel 在 0.59 上仍被 `unsupported logical view contract '...rows=32;cols=8;layout=cube_m32'` 拒绝（PR146 的 wide-view 边界）。
- 已在 issue 下留言复测结果并询问 256-element grain 写法：https://github.com/LinxISA/llvm-project/issues/153#issuecomment-6655682260
