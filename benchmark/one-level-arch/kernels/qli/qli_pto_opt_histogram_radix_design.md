# QLI `qli_pto_opt` 直方图 Radix-Select TopK 方案说明

> 文档状态：新增方案说明
>
> 分析对象：`SuperNPUBench/benchmark/one-level-arch/kernels/qli/qli_pto_opt.hpp`
>
> 适用版本：P3 `THISTOGRAM` 直方图实现，基于 float32 score 与 uint32 sortable key
>
> 记录日期：2026-08-24

---

## 1. 文档目的与结论

本文档对当前 `qli_pto_opt.hpp` 中的 QLI 直方图 TopK 实现做完整拆解，说明：

1. QLI Step1–6 如何生成 score；
2. Step7 如何把 float score 转换成可进行整数排序的 `uint32` key；
3. `THISTOGRAM` API 的输入、输出和 `Idx` 语义；
4. 当前代码中 histogram、桶边界判断和 TopK 提取分别由哪一部分完成；
5. 当前实现是否属于标准桶排序/radix 排序；
6. 当前精度验证能证明什么，以及还没有证明什么；
7. 后续如何演进为真正面向 gsim 性能的多轮 radix-select。

### 1.1 核心结论

当前实现是一个 **MSD radix-select hybrid**：

```text
float score
    ↓
uint32 IEEE-754 sortable key
    ↓
最高字节 Byte3 的 THISTOGRAM
    ↓
找到 TopK 所在的最高字节边界桶
    ↓
高于边界的元素直接确定入选
    ↓
边界桶内 scalar 扫描 + insertion sort
    ↓
输出 TopK index
```

它已经具备 radix/bucket 算法的关键思想：

- 将排序关键字拆成固定宽度 digit；
- 先统计高位 digit 的分布；
- 从高桶向低桶选择 TopK 边界；
- 只对边界候选做进一步精确选择。

但它**不是完整的多轮 radix sort，也不是完整的多轮 radix-select**：

- 当前只使用 `Byte3` 一轮 histogram；
- 没有继续使用 `Byte2/Byte1/Byte0` 递归收窄；
- 没有做 radix sort 所需的 scatter/reorder；
- 高于边界的元素按原始 index 扫描顺序写出，不是严格的 score 降序；
- 边界桶使用 scalar insertion sort。

因此，当前实现可以准确地称为：

> **最高字节桶选择 + 边界桶精排的 TopK 选择算法**，而不是完整 radix 排序。

---

## 2. 当前代码路径与相关实现

### 2.1 当前实际使用的路径

当前优化 driver 使用：

```cpp
qli_pto<dtype, Sq, Skv, D, g, kTm, kTk>(...);
qli_topk_histogram<Sq, Skv, topK>(...);
```

调用位置为：

- `SuperNPUBench/benchmark/one-level-arch/test/kernel/qli/src/qli_check_opt.cpp:78-96`

其中：

- `BENCHSTART/BENCHEND` 当前只包围 `qli_pto` 的 Step1–6；
- `qli_topk_histogram` 在 `BENCHEND` 之后调用；
- 因此现有 benchmark marker 并没有独立包围 Step7。

这是分析 gsim 性能时必须注意的测量边界。

### 2.2 `qli_pto_opt.hpp` 中的三个 TopK 版本

`qli_pto_opt.hpp` 同时保留了多个实验版本：

| 函数 | 类型 | 作用 | 当前定位 |
|---|---|---|---|
| `qli_topk_npu` | `TROWARGMAX` + scalar 消零 | 每轮 argmax 一个元素 | P2 对照基线 |
| `qli_topk_extract` | scalar 扫描 + insertion sort | 对 histogram 结果做精确提取 | P3 的后处理部分 |
| `qli_topk_histogram` | sortable key + `THISTOGRAM` | 统计最高字节并调用 extract | P3 当前直方图入口 |

相关代码位置：

- `qli_topk_npu`：`qli_pto_opt.hpp:536-673`
- `qli_topk_extract`：`qli_pto_opt.hpp:675-712`
- `qli_topk_histogram`：`qli_pto_opt.hpp:714-804`

旧文件 `qli_bucket_pto.hpp`、`qli_bucket_pto_histnpu.hpp` 和
`qli_bucket_pto_radixnpu.hpp` 是独立的 bucket/radix 实验路径，不能直接视为
当前 `qli_pto_opt` 的实际执行代码。

---

## 3. QLI 的整体计算流程

QLI 的目标是对每个 query token，计算它与所有 KV token 的 score，并选出 TopK
个 KV index。

### 3.1 输入输出数据

当前 QLI 的主要形状为：

```text
Q        : [Sq * g, D]
K        : [Skv, D]
W        : [Sq, g]
scale_q  : [Sq, g]
scale_k  : [Skv]
scores   : [Sq, Skv]
indices  : [Sq, topK]
```

其中：

- `Sq`：query token 数；
- `Skv`：KV token 数，也就是 TopK 候选数量；
- `g`：head 数；
- `D`：head dimension；
- `topK`：每个 query 要输出的 index 数。

### 3.2 Score 公式

当前实现计算：

```text
score[s1, s2]
  = scale_k[s2]
    * Σ_g(
        W[s1, g]
        * scale_q[s1, g]
        * ReLU(Q[s1, g] · K[s2])
      )
```

### 3.3 Step1–6 的 tile 流程

`qli_pto` 位于 `qli_pto_opt.hpp:141-294`，主要过程如下：

```text
for each query token s1:
    for each K block s2:
        TLOAD K block
        TLOAD scale_k

        for each head block g:
            TLOAD Q block
            TLOAD W
            TLOAD scale_q

            TMUL(W, W, scale_q)
            TMATMUL(S, Q, K)
            TMAX(S, S, 0)
            TROWEXPANDMUL(S, S, W)
            TCVT / reduction layout conversion
            TCOLSUM(partial, S)
            TADD(sum, sum, partial)

        TMUL(sum, sum, scale_k)
        TSTORE(score[s1, s2])
```

这一阶段的输出是 FP32 `scores`，TopK 直方图只消费这些 score，不改变 score
计算公式。

---

## 4. 为什么不能直接把 float bit 当作 uint32 排序

IEEE-754 float 的原始 bit pattern 并不是对所有数值都保持无符号整数单调性。

### 4.1 正数

对正数而言，float 的 bit pattern 大致与数值递增一致：

```text
0.0 < 1.0 < 2.0
bits(0.0) < bits(1.0) < bits(2.0)
```

但这只对非负数成立。

### 4.2 负数

负数的 sign bit 为 1，且数值越接近零，原始 bit pattern 的整数关系并不能直接
用于从小到大排序。例如负数如果直接按 `uint32` 比较，会被整体排到正数之后。

### 4.3 单调 key 映射

当前实现使用下面的映射：

```cpp
bits = reinterpret_cast<uint32_t>(score)

if (is_nan(bits))
    key0 = 0xffffffffu
else
    key0 = bits

if (key0 has sign bit)
    key = ~key0
else
    key = key0 | 0x80000000u
```

代码位于 `qli_pto_opt.hpp:761-777`。

对应的数学关系是：

```text
key(score) =
    ~bits                  , score < 0
    bits | 0x80000000      , score >= 0
```

其效果是：

```text
更大的 float score  <=>  更大的 uint32 key
```

典型值如下：

| float | 原始 bits | sortable key |
|---|---:|---:|
| `-inf` | `0xff800000` | `0x007fffff` |
| `-1.0` | `0xbf800000` | `0x407fffff` |
| `-0.0` | `0x80000000` | `0x7fffffff` |
| `+0.0` | `0x00000000` | `0x80000000` |
| `+1.0` | `0x3f800000` | `0xbf800000` |
| `+inf` | `0x7f800000` | `0xff800000` |

所以 `TROWMAX`、`TCMP` 等整数域操作可以等价地完成 float 的最大值选择。

### 4.4 NaN 处理

当前意图是：

```text
NaN → key0 = 0xffffffff → key = 0
```

这样 NaN 会被放到最低，不会被 TopK 选中。

需要注意，当前 tile 路径使用的是规范 NaN 位型比较：

```cpp
FP32_NAN_BITS = 0x7FC00000u
TCMP(tNan01, bits, tNan)
```

而 `qli_topk_extract` 的 scalar 路径会检查完整的 exponent/mantissa 条件，能识别
更多非规范 NaN 位型。两条路径的 NaN 判定并不完全一致，后续应统一。

---

## 5. 当前 `qli_topk_histogram` 的逐步过程

### 5.1 Step7 输入约束

函数入口为：

```cpp
template <int Sq, int Skv, int topK>
void qli_topk_histogram(float* scores_gm, int32_t* indices_gm)
```

代码有以下静态约束：

```cpp
static_assert(Skv % 8 == 0);
static_assert(topK <= Skv);
```

当前实现定义：

```cpp
MaxTileCol = 2048
NumChunks  = ceil(Skv / 2048)
```

原因是 FP32/UINT32 下 2048 个元素正好对应约 8KB tile。

代码位置：`qli_pto_opt.hpp:726-739`。

### 5.2 Step7.1：按 chunk 加载 score

对于每个 query row 和每个 chunk：

```cpp
uint32_t* chunk_ptr = reinterpret_cast<uint32_t*>(scores_gm + row_offset);
TLOAD(bits, gin);
```

这不是数值转换，而是**位重解释**：

- GM 中仍然保存 FP32 score；
- tile 中以 UINT32 类型读取同样的 32 位数据；
- 目的是让后续位操作和整数比较合法。

`TLOAD` 只负责搬运 bit，不负责改变数值。

### 5.3 Step7.2：在 tile 中构造 sortable key

当前使用整数域完成以下操作：

```cpp
TCMP   // 判断规范 NaN
TNOT   // 取反
TANDS  // 取 sign mask
TORS   // 对正数置 sign bit
TSHRS  // 提取 sign bit
TSUB   // 构造两路结果差值
TMUL   // 用 0/1 mask 做整数 select
TADD   // 合成最终 key
```

关键代码对应 `qli_pto_opt.hpp:761-777`。

之所以使用整数域，是因为当前仿真器存在一个重要限制：

```text
TCMP 在 FP32 tile 中写入 raw bits 0/1，
0x00000001 作为 FP32 是 1.4e-45，而不是 1.0。
```

因此不能简单使用：

```text
FP32 value * FP32 TCMP mask
```

而 UINT32 中的 `0/1` 可以安全用于 `TMUL` 作为算术 mask。

### 5.4 Step7.3：取最高 byte

当前 FP32 key 的最高字节为 `Byte3`：

```cpp
TSHRS(tByte, key, 24);
```

逻辑上等价于：

```cpp
tByte[i] = (key[i] >> 24) & 0xff
```

最高字节包含 sign、exponent 的高位以及部分 mantissa 信息，是 MSD radix 的第一
个 digit。

### 5.5 Step7.4：调用 `THISTOGRAM`

当前代码：

```cpp
using tile_hist = Tile<Location::Vec, uint32_t, 1, 256,
                        BLayout::RowMajor>;

tile_hist hist;
tile_key idx_tile;
TEXPANDS(idx_tile, 0u);

THISTOGRAM(hist, key, idx_tile, 3);
```

代码位置：`qli_pto_opt.hpp:779-787`。

这一条 API 完成：

```text
读取 key 的 Byte3
统计 0..255 每个 byte value 出现多少次
输出 256 个 UINT32 bucket count
```

与旧版手工直方图相比，它把下面的 256 次循环压缩为一条 tile instruction：

```text
for b in 0..255:
    TEXPANDS(bucket, b)
    TCMP(mask, byte, bucket)
    TROWSUM(count, mask)
    TSTORE(hist[b], count)
```

旧版实现可在 `qli_bucket_pto.hpp:379-401` 看到。

### 5.6 Step7.5：将 histogram 写回 GM

`THISTOGRAM` 输出 tile 不能直接由 scalar C++ 变量读取，因此当前代码先构造：

```cpp
using gm_hist = global_tensor<uint32_t, RowMajor<1, 256>>;
gm_hist histGm(temp_hist);
TSTORE(histGm, hist);
```

此时数据流是：

```text
tile register histogram
        ↓ TSTORE
GM temp_hist[256]
        ↓ scalar load
host/compiler-visible array
```

### 5.7 Step7.6：累计前缀和与原始桶计数

`THISTOGRAM` 输出的不是：

```text
hist[b] = count[b]
```

而是：

```text
hist[b] = Σ(count[0..b])
```

也就是升序累计前缀和。

例如原始桶计数为：

```text
count = [2, 0, 3, 1]
```

API 输出为：

```text
prefix = [2, 2, 5, 6]
```

恢复原始计数的方法是：

```text
count[0] = prefix[0]
count[b] = prefix[b] - prefix[b-1]
```

当前代码在 `qli_pto_opt.hpp:792-797` 对每个 chunk 做差分，然后累加到
`per_bin[256]`。这是必要的，因为 `Skv` 可能被拆成多个 chunk，每个 chunk 都有
自己的一份累计前缀和。

### 5.8 Step7.7：确定 TopK 边界桶

TopK 选择是从大 key 到小 key，因此从 bucket 255 向下累计：

```cpp
cumsum = 0;
for (b = 255; b >= 0; --b) {
    cumsum += per_bin[b];
    if (cumsum >= topK) {
        kth_bin = b;
        break;
    }
}
```

其含义是：

- `b > kth_bin` 的元素数量小于 `topK`，全部确定入选；
- `b == kth_bin` 的元素补足剩余名额；
- `b < kth_bin` 的元素不可能进入 TopK。

如果使用 API 返回的累计前缀和 `prefix`，也可以直接计算：

```text
count_ge(b) = prefix[255] - prefix[b-1]
```

当前 `qli_pto_opt` 先把 chunk histogram 转成全局 `per_bin`，再做从高到低扫描。

### 5.9 Step7.8：提取高于边界的元素

`qli_topk_extract` 的第一段代码：

```cpp
for (j = 0; j < Skv && found < above_count; ++j) {
    key = map(score[j]);
    if (high_byte(key) > kth_bin)
        indices[found++] = j;
}
```

这一段的算法意义是正确的：高于边界桶的元素一定属于 TopK。

但是输出顺序是按 `j=0..Skv-1` 的原始 index 顺序，并不是按 score/key 降序。

因此当前实现的精度结论应区分：

- **TopK 集合正确**：可以由高字节边界保证；
- **TopK 输出顺序正确**：当前代码不能仅凭这一段保证；
- 若下游只需要无序 TopK set，则没有问题；
- 若下游要求按 score 降序排列，则必须继续做 tile radix extraction 或最终 merge sort。

### 5.10 Step7.9：边界桶 scalar 精排

如果 `found < topK`，当前代码会再次扫描整行：

```cpp
for (j = 0; j < Skv; ++j) {
    if (high_byte(key[j]) == kth_bin) {
        ci[nc] = j;
        cv[nc] = score[j];
        ++nc;
    }
}
```

之后执行 insertion sort：

```cpp
for (a = 1; a < nc; ++a) {
    while (b >= 0 && cv[b] < kv) {
        cv[b + 1] = cv[b];
        ci[b + 1] = ci[b];
        --b;
    }
}
```

这个步骤保证边界桶内可以按照 FP32 score 从大到小选取，并且由于输入是按 index
递增扫描、相等值不触发移动，通常可以保持较小 index 优先。

但它有明显性能代价：

- 重新 scalar 读取完整 `Skv` 行；
- 重新做一次 float bit mapping；
- `nc` 较大时 insertion sort 是 O(`nc²`)；
- `-O0` 是为规避 LinxV5 后端问题而采用的 workaround，不是最终算法形态。

代码位置：`qli_pto_opt.hpp:689-709`。

---

## 6. `THISTOGRAM` API 详细语义

### 6.1 API 形式

TileOP-API 形式为：

```cpp
THISTOGRAM(dst, src, Idx, ByteId);
```

当前实现中的实际调用为：

```cpp
THISTOGRAM(hist, key, idx_tile, 3);
```

### 6.2 参数说明

| 参数 | 当前对象 | 作用 |
|---|---|---|
| `dst` | `Tile<uint32_t, 1, 256>` | 输出 256 个桶的累计计数 |
| `src` | `Tile<uint32_t, 1, 2048>` | 待统计的 sortable key |
| `Idx` | 当前为 zero tile | 多轮 radix 时提供已确定的高字节前缀 |
| `ByteId` | 当前为 `3` | 选择统计哪个 byte，`3` 是 uint32 最高字节 |

### 6.3 `ByteId` 的含义

对 UINT32 source：

```text
ByteId = 0 → bits[7:0]
ByteId = 1 → bits[15:8]
ByteId = 2 → bits[23:16]
ByteId = 3 → bits[31:24]
```

仿真器实现使用：

```cpp
(value >> (selectedByte * 8)) & 0xff
```

实现位置：`SuperScalarModel/emulator/engine/TEPLEngine.cpp:1411-1413`。

因此当前 FP32 radix 的第一轮使用 `Byte3` 是正确的。

### 6.4 `dst` 的形状与 dtype

`THISTOGRAM` 的输出固定为每行 256 个 UINT32：

```text
dst shape = [sourceRows, 256]
dst dtype = uint32
```

即使 source 是 UINT16，histogram count 仍然使用 UINT32，以避免计数溢出并统一
API 语义。

当前 QLI 每次只统计一个 score row，因此：

```text
source shape = [1, 2048]
hist shape   = [1, 256]
```

### 6.5 `Idx` 的作用

`Idx` 不是最终输出的 TopK index，也不是普通的 index vector。它的用途是支持
多轮 radix 的高位前缀约束。

例如 UINT32 进行四轮 radix-select：

```text
第一轮：Byte3，统计全部 key
第二轮：Byte2，只统计 Byte3 == selected_high
第三轮：Byte1，只统计 Byte3/Byte2 前缀均相等
第四轮：Byte0，只统计前三个 byte 前缀均相等
```

仿真器中，`selectedByte` 小于 3 时，会读取 `Idx` 的前序行，并检查 source 的高位
byte 是否与前缀相等：

```cpp
for (level = selectedByte + 1; level <= 3; ++level) {
    idx = 3 - level;
    selected = source_byte(level) == Idx[idx * indexStride];
}
```

对应实现：`TEPLEngine.cpp:1404-1409`。

这意味着：

- 当前 `Byte3` 最高字节统计不需要前缀条件；
- 当前使用 zero `idx_tile` 是可接受的过渡写法；
- 如果未来统计 `Byte2/Byte1/Byte0`，必须正确构造 `Idx` 的 prefix rows；
- 不能简单地把 `Idx` 当作最终 index 输出地址。

### 6.6 输出是累计前缀和，不是 raw count

仿真器最后执行：

```cpp
uint32_t cumulative = 0;
for (bin = 0; bin < 256; ++bin) {
    cumulative += counts[bin];
    dst[bin] = cumulative;
}
```

实现位置：`TEPLEngine.cpp:1415-1418`。

因此调用方必须明确区分：

```text
raw_count[b] = prefix[b] - prefix[b-1]
count_ge[b]  = prefix[255] - prefix[b-1]
```

如果误把 `prefix[b]` 当成 `count[b]`，多 chunk 合并和 kth bucket 判断都会错误。

### 6.7 当前 `idx_tile` 为什么可以填零

当前使用：

```cpp
TEXPANDS(idx_tile, 0u);
THISTOGRAM(hist, key, idx_tile, 3);
```

因为 `ByteId=3` 已经是最高 byte，不存在更高字节前缀约束，`Idx` 在这轮实际上
不会筛掉 source 元素。

这只是最高字节第一轮的合法简化。进入低字节 radix 后，zero `Idx` 会导致错误的
前缀筛选，必须替换为真实 prefix tile。

---

## 7. 当前方案是否符合桶排序/radix 算法

### 7.1 标准 radix sort 的完整流程

以基数 256、MSD radix 为例，标准完整排序通常包括：

```text
1. 取当前 digit
2. 统计 256 个 bucket count
3. 计算 bucket offset/prefix
4. 按 bucket 将元素 scatter/reorder
5. 对每个未完成 bucket 递归处理下一 digit
6. 输出完整有序序列
```

### 7.2 当前实现与标准流程的对应关系

| 标准 radix 步骤 | 当前 `qli_pto_opt` 状态 | 说明 |
|---|---|---|
| 生成整数排序 key | 已实现 | float→uint32 monotonic mapping |
| 取 digit | 已实现 | `TSHRS(key, 24)` 取 Byte3 |
| 统计 bucket | 已实现 | `THISTOGRAM(..., ByteId=3)` |
| 计算边界 | 已实现 | scalar 从高桶向下扫描 |
| 递归处理低 digit | 未实现 | 没有 Byte2/Byte1/Byte0 的连续 radix pass |
| scatter/reorder | 未实现 | TopK 不需要全量排序，因此被省略 |
| 选择边界桶内元素 | 已实现 | scalar 扫描 + insertion sort |
| 输出完整有序序列 | 部分实现 | 高于边界的元素不是 score 降序输出 |

### 7.3 准确定义

当前实现符合以下定义：

> **单轮 MSD radix-select 的桶边界筛选算法，后接 scalar 精确 TopK 提取。**

它不是以下定义：

- 不是完整 radix sort；
- 不是完整多轮 radix-select；
- 不是稳定的全序排序；
- 不是完全 NPU/tile 化的 TopK。

### 7.4 为什么不需要 scatter

QLI 只需要 TopK index，不需要输出所有 `Skv` 个 score 的排序结果。

因此可以省略完整 radix sort 的 scatter/reorder，采用 selection：

```text
只确定第 topK 个元素所在的 bucket
只保留可能进入 TopK 的候选
对候选做精确提取
```

省略 scatter 是合理优化，但前提是候选提取仍然需要满足下游的输出契约：

- 如果下游只使用 TopK set，当前高桶顺序问题可以接受；
- 如果下游要求 TopK 按 score 降序，则需要追加 tile radix extraction 或最终候选排序。

---

## 8. 精度正确性分析

### 8.1 score 计算精度与 TopK 精度是两个层次

应分别验证：

```text
Step1–6：score cosine / NaN
Step7：TopK exact / TopK set / 排序顺序
```

直方图不会重新计算 score，只对已经生成的 FP32 score 做 key 映射和选择。

### 8.2 key 映射的单调性

对于有限 FP32：

```text
score_a > score_b
    ⇒ sortable_key(score_a) > sortable_key(score_b)
```

因此在 key 域确定 TopK 与在 float 域确定 TopK 等价。

### 8.3 边界桶为什么不会漏选

设 `kth_bin` 是从高到低累计后首次达到 `topK` 的桶：

```text
N(high > kth_bin) < topK
N(high >= kth_bin) >= topK
```

所以：

- `high > kth_bin` 的元素一定入选；
- `high < kth_bin` 的元素一定不入选；
- 只有 `high == kth_bin` 需要进一步比较完整 key/float。

这就是 bucket selection 的正确性基础。

### 8.4 当前精度验证的实际含义

P3 记录中已确认：

- `cosine=1.0`；
- `TopK set=4/4`；
- `NaN=0`。

这证明当前测试数据上：

- Step1–6 score 数值正确；
- TopK 集合没有漏选或误选；
- NaN 测试场景没有进入最终结果。

但还应补充验证：

1. 输出 index 是否严格按 score 降序；
2. 完整 NaN 位型是否统一被排除；
3. `Skv` 不是 2048 整数倍时，tail chunk 是否安全；
4. 大量相同 high byte 时，边界候选是否覆盖完整；
5. 重复 score 时是否稳定选择较小 index。

---

## 9. 当前实现的性能结构

### 9.1 理想收益

相对于 P2 的每轮 `TROWARGMAX`：

```text
P2：topK 轮 argmax + 每轮 scalar 消零
P3：每个 chunk 一次高字节 THISTOGRAM + 一次候选提取
```

直方图本身把 256 次：

```text
TCMP + TROWSUM + TSTORE + barrier
```

压缩为：

```text
THISTOGRAM + TSTORE
```

这部分理论上可以显著减少 tile instruction 和 block 数。

### 9.2 当前实际瓶颈

P3 当前仍有：

```text
256 个 bucket 的 scalar 控制扫描
+ Skv 次 scalar score 读取
+ Skv 次 scalar key 重建
+ 边界候选 insertion sort
+ indices scalar 写出
```

其中最重的是 `qli_topk_extract`，不是 `THISTOGRAM`。

因此出现了：

```text
THISTOGRAM 统计变快
但整个 P3 gfrun block 数反而上升
```

已有记录显示：

- P2：约 `84,513 gfrun blocks`；
- P3：约 `4.64M gfrun blocks`；
- P3 的 `-O0` 是为绕开栈数组 vectorization 生成错误而使用的 workaround。

### 9.3 scratch memory

当前 histogram scratch 地址通过下面的方式计算：

```cpp
temp_hist = reinterpret_cast<uint32_t*>(
    reinterpret_cast<uint8_t*>(indices_gm)
    + Sq * topK * 4
    + 4096);
```

代码位置：`qli_pto_opt.hpp:741-743`。

这意味着：

- scratch 紧跟在 `indices` 输出区域之后；
- 需要调用者保证该区域可写且足够大；
- 该内存契约目前是隐式的；
- 多 PE 并行时不能让多个 PE 共用同一 `temp_hist`。

后续建议显式传入 `scratch_ptr` 和 `scratch_bytes`，不要依赖输出 buffer 后方的
隐式空间。

### 9.4 tail chunk 风险

代码计算了：

```cpp
validCol = min(MaxTileCol, Skv - offset)
```

但当前 `validCol` 没有进一步写入 tile descriptor，`TLOAD` 仍然以固定
`[1, MaxTileCol]` tile 处理。

因此当前实现对 `Skv=8192` 这类完整 2048 分块配置最明确；对于非 2048 整数倍的
`Skv`，需要单独验证 tail tile 是否会访问边界外数据。

---

## 10. 推荐的后续完整直方图方案

目标是把当前方案从：

```text
单轮 Byte3 histogram
+ scalar 全行 extraction
```

升级为：

```text
分块多轮 MSD radix-select
+ tile candidate mask
+ tile/local TopK
+ 小规模候选 merge
```

### 10.1 第一阶段：保留当前 float32/uint32 方案

不立即切换 BF16，先保持：

```text
score: float32
key  : uint32
radix digit: 4 × 8-bit
```

原因：

- float32 key 路径已有精度结果；
- UINT32 `TCMP/TMUL` mask 行为已经验证；
- BF16→UINT16 当前存在仿真器跨 dtype 读取问题；
- BF16 会引入新的量化 tie-breaking。

### 10.2 第二阶段：全局 Byte3 选择

对每个 chunk：

```text
TLOAD score bits
tile key mapping
THISTOGRAM(Byte3)
TSTORE local prefix histogram
```

host/scalar 只做：

```text
local prefix → local count
local count → global count
global count → kth Byte3
```

256 个桶的控制扫描是固定小开销，可以保留在 scalar；真正需要消除的是对完整
`Skv` 行的逐元素 scalar 扫描。

### 10.3 第三阶段：低字节递归选择

确定 `Byte3` 后，设置：

```text
done   = Byte3 > high_boundary
active = Byte3 == high_boundary
remaining = topK - count(done)
```

然后逐轮处理：

```text
Byte2
Byte1
Byte0
```

每一轮：

1. 只统计 `active` 集合；
2. 使用 `Idx` tile 传递已确定的高字节 prefix；
3. 统计当前 byte 的 256 个桶；
4. 找到新的 boundary；
5. 将当前 byte 高于 boundary 的元素加入 `done`；
6. 将当前 byte 等于 boundary 的元素保留在 `active`；
7. 更新 `remaining`。

伪代码：

```text
done = empty
active = all elements
remaining = topK

for byte in [3, 2, 1, 0]:
    hist = histogram(active, byte, prefix)
    boundary = select_descending_boundary(hist, remaining)
    fixed = count(active and digit > boundary)
    done = done union (active and digit > boundary)
    active = active and (digit == boundary)
    remaining -= fixed
    if remaining == 0:
        break
```

这才是完整的多轮 radix-select 语义。

### 10.4 第四阶段：候选输出

当 radix pass 完成后，候选集合为：

```text
candidates = done union active
```

建议不要再回到 `float* scores_gm` 做 scalar insertion sort，而是：

1. 保留 uint32 key tile；
2. 在 key 域屏蔽非候选元素；
3. 用 tile max/argmax 提取 chunk-local TopK；
4. 将最多 `NumChunks * topK` 个候选合并；
5. 在最终 merge 中显式处理全局 index tie-breaking。

### 10.5 输出顺序契约

需要先确定 QLI 的下游契约：

#### 方案 A：只要求 TopK set

当前高桶元素按原始 index 顺序输出是可以接受的，文档和 golden 应明确使用
`TopK set` 校验。

#### 方案 B：要求 score 降序

必须保证：

```text
scores[idx[0]] >= scores[idx[1]] >= ...
```

并在 score 相同的情况下：

```text
idx[i] < idx[i+1]
```

这时必须追加 tile radix extraction 或最终候选排序，不能只做高字节 bucket
membership 判断。

---

## 11. 与 BF16/UINT16 方案的关系

BF16 方案的目标是：

```text
float32 score
    ↓ TCVT
bf16 score
    ↓ order-preserving mapping
uint16 key
    ↓
Byte1 + Byte0 两轮 radix
```

理论优势：

- radix digit 从 4 轮减少到 2 轮；
- 每个 8KB tile 可以容纳更多元素；
- score/key 的 GM 流量减半。

但目前有两个阻塞：

1. `__bf16 TSTORE → uint16 TLOAD` 读取结果为 0；
2. scalar store 到 heap 后，`uint16 TLOAD` 读取结果为 0。

见根目录 `qli_bf16_bugs.md:11-48`。

此外，BF16 只有 7-bit mantissa，golden 必须基于 BF16 量化后的 score 重新生成，
否则 TopK tie-breaking 不可比。

因此 BF16/UINT16 是当前 float32/UINT32 多轮 radix-select 稳定后再评估的第二阶段
方案，不是当前 P3 的直接替换方案。

---

## 12. 验证计划

### 12.1 功能验证

至少覆盖以下配置：

| 场景 | 目的 |
|---|---|
| `Sq=4, Skv=8192, topK=512` | 当前 P3 大规模基线 |
| `Sq=4/16/32/64, Skv=128, topK=8` | 多 `Sq` 回归 |
| `Skv=2048` | 单 tile 边界 |
| `Skv=2056` 等非 2048 倍数 | tail chunk |
| 大量相同 Byte3 | 边界桶压力 |
| 大量相同 score | tie-breaking |
| 负 score / `-0/+0` | sortable key 单调性 |
| `NaN` / `±inf` | 特殊值策略 |

每个用例输出：

```text
score cosine
TopK exact
TopK set
TopK order
NaN count
duplicate/tie result
```

### 12.2 THISTOGRAM API 单测

需要单独验证：

1. `ByteId=3` 对 UINT32 最高字节的统计；
2. `ByteId=2/1/0` 的低字节统计；
3. `Idx` prefix 约束是否符合预期；
4. 输出是否为 cumulative prefix；
5. source tile 的 `validCol` 与 tail chunk 是否正确；
6. 多 chunk histogram 差分合并是否等价于单个完整 row 的 histogram。

### 12.3 gsim 性能验证

Step1–6 与 Step7 应分别设置 trace marker：

```text
TRACE.begin score
    qli_pto
TRACE.end score

TRACE.begin topk
    qli_topk_histogram
TRACE.end topk
```

重点对比：

| 指标 | 目标 |
|---|---|
| Total cycles | 新方案低于 P2 基线，或至少不因 histogram 恶化 |
| STD block 占比 | 明显低于当前 P3 |
| BRob Full Stall | 随 scalar extraction 消除而下降 |
| Idle fraction | 低于当前 P3 |
| BPC | 高于当前 P2/P3 scalar-heavy 路径 |
| Vector/Cube overlap | 不因 TopK 控制流破坏 |

当前 TimingSim 已知限制：

- `TMA Busy=0` 不能直接解释为真实硬件 TMA 未工作；
- scalar store 与 tile load 混合可能触发末尾死锁；
- 完整 run 与 `-m` 截断 run 必须使用一致的比较口径。

---

## 13. 当前方案的最终评价

### 已完成

- Step1–6 score 计算保持原 QLI 语义；
- float32→uint32 sortable key 映射已实现；
- `THISTOGRAM(Byte3)` 已接入当前 P3；
- histogram 的 cumulative prefix 语义已在仿真器实现；
- 当前测试配置 TopK set 精度已经通过；
- 直方图统计阶段已经从 256 次手工 bucket 操作压缩为单条 API 指令。

### 尚未完成

- Byte2/Byte1/Byte0 的多轮 radix-select；
- 完全 tile 化的候选提取；
- 高于边界元素的 score 降序输出；
- tail chunk 的通用安全处理；
- 显式 scratch memory contract；
- Step7 独立 gsim cycle 数据；
- BF16/UINT16 方案的仿真器支持。

### 推荐实现顺序

```text
1. 固定 benchmark marker，单独测 Step7
2. 保留 float32/uint32 key 路径
3. 消除 qli_topk_extract 的全行 scalar 扫描
4. 使用 THISTOGRAM + Idx 实现 Byte3→Byte0 多轮 radix-select
5. 使用 tile mask + local TopK + compact merge 输出 index
6. 补齐 tail/tie/NaN/order 验证
7. 最后再评估 BF16/UINT16 两级 radix
```

当前方案的准确定位是：

> **精度已通过的单轮高字节 histogram TopK 原型，下一步应演进为完全 tile 化的分块多轮 radix-select，而不是继续扩大 scalar extraction。**

---

## 14. 实施记录：分块多轮 MSD radix-select（2026-08-24）✅

> 本轮完成 §10.2–§10.4 的全部落地：`qli_topk_radix` 已在
> `qli_pto_opt.hpp` 实现并通过精度/性能验证。文档状态从
> "方案说明"升级为"已实现"。

### 14.1 实现摘要

| 组件 | 内容 |
|---|---|
| 算法 | Byte3→Byte0 逐字节 THISTOGRAM（`THISTOGRAMX` 扩展 Idx）→ kth_value → tile pop-argmax（内部 TMUL 消零，无标量 store、无重 TLOAD） |
| Idx 前缀 | 仿真器已支持 ByteId<3 前缀约束（TEPLEngine 按各 tile tileInfo 读 row0/1/2），实测正确 |
| 提取 | `QLI_RADIX_POP_N` / `QLI_RADIX_POP_EQN`（宏内联，tile 作函数参数触发 LinxV5 编码 bug 已规避） |
| tail | `TailPhy/SinglePhy` 物理列宽 ≥128（512B 下限）+ ValidCol 参数化 |
| 输出契约 | TopK set（无序集合）；并列取最小索引（rev 技巧） |

### 14.2 精度结果（gfrun, set-match 主判据）

| Sq | Skv | topK | chunks | cosine | TopK set | exact(参考) | NaN |
|---|---|---|---|---|---|---|---|
| 4 | 128 | 8 | 1 | 1.000000 | 4/4 | 100% | 0 |
| 4 | 2048 | 8 | 1 | 1.000000 | 4/4 | 100% | 0 |
| 4 | 2048 | 256 | 1 | 1.000000 | 4/4 | 100% | 0 |
| 4 | 4096 | 512 | 2 | 1.000000 | 4/4 | 0.29%* | 0 |
| 4 | 8192 | 512 | 4 | 1.000000 | 4/4 | 0.05%* | 0 |
| 16 | 8192 | 512 | 4 | 1.000000 | 16/16 | 0.39%* | 0 |
| 16/32/64 | 128 | 8 | 1 | 1.000000 | 100% | 100% | 0 |
| 4 | 2056(尾) | 8 | 1+tail | 0.998680 | 4/4 | 100% | 0 |

> *`exact` 仅为参考：set 契约下输出无序，相同元素位置 swap 是预期行为。
> `Skv=2056` 的 cosine=0.998 源于 Step1-6 ql_pto 的 `Skv%kTk` 限制
> （2056/32 非整除，仅计算 2048 列），与 TopK 无关。

### 14.3 特殊值单测（`/tmp/opencode/test_special_radix.cpp`）

构造含 +inf/-inf/NaN/-0/+0/负数的 1×128 scores 行验证 sortable key 单调性：

| 元素 | key 映射 | TopK 行为 |
|---|---|---|
| NaN (0x7FC00000) | →0（最低） | 排除 ✅（arch35 FloatToSortableKey 一致） |
| -inf | ~bits=0x007FFFFF | 排除 ✅ |
| -2.0 / -1.0 | ~bits | 排除 ✅ |
| -0.0 / +0.0 | 0x7FFFFFFF / 0x80000000 | 单调 ✅ |
| +inf | 最高 | 入选 ✅ |

topK 结果 {+inf, 10000, 5,4,3,2,1,0.5} 与按 key 降序的期望完全一致。

### 14.4 性能（gfsim, Sq=4 Skv=8192 topK=512, kTk=32）

| 指标 | P2 基线(记录) | P3(记录) | Radix（本轮） |
|---|---|---|---|
| gfrun blocks | 84,513 | 4,639,927 | **137,456** |
| Total cycles | 1,819,116 | — | **657,097** |
| BRob Full Stall | 31.0% | — | **14.9%** |
| STD block 占比 | 92.8% | — | **43.4%** |
| All Cores Idle | 78.96% | — | 20.2% (132,794) |
| 精度 | cosine 1.0 | set 100% | **cosine 1.0, set 100%** |

已消除 92.8% 标量 block 中的 TopK 消零部分（P2 的 `row_u32[idx]=...` +
重 TLOAD 全部去掉）；STD 剩余来自 copy/loop 控制。TopK 的
tile op 提取（TROWMAX/TCMP/TMUL 链）已无标量参与。

### 14.5 过程中发现/修复的仿真器与工具链问题

1. **THISTOGRAM ByteId decode 错误**（ISA 层 bug，本地修复）：
   LLVM 工具链把 ByteId 编码在 `bits[19:18]`（LinxV5InstrInfo.td
   `Inst{19-18}=ByteId`），而 `Block::HandleBDATR` 从
   `srcs[SRC5_IDX]`（decode 表 `%padValue_27_28`，bits[28:27]）读
   selectedByte → 恒为 3（Byte3）。P3 只用 Byte3 故未暴露。修复：
   `Block.cpp:716` 改读 `(inst.binary >> 18) & 0x3`。
2. **B8 校验门禁**（`AccumulateBlockInfo.cpp`）：
   TROWSUM/TROWMAX/TROWARGMAX 对 UINT32/UINT16 放宽——
   radix 在 UINT32 域做掩码求和/取最大（0/1 算术 mask、sortable key pop）。
3. **TROWEXPAND 广播 tileInfo 与 TCMP 不兼容**（编译器规避）：
   `TROWEXPAND(mxbc,mx)` 输出 tileInfo 取自 src([1,32])，TCMP 校验按
   [1,2048] 要求 → 失败。改用 `TSTORE→标量→TEXPANDS` 广播（同 Spike C）。
4. **tile 作函数参数触发 LinxV5 错误编码**：`RadixPopN(mv)` 参数
   引用被编译成 S64 [1,1024] TLOAD。规避：pop 逻辑内联为宏
   `QLI_RADIX_POP_N`，所有 tile 均为局部变量。
5. **驱动布局 bug**：`qli_check_opt.cpp` 的 OUT_INDICES 固定
   `0x4000822000`，Sq*Skv*4 > 0x20000（如 Sq=16,Skv=8192=0x80000）
   时与 scores 重叠 → 改 `OUT_SCORES + Sq*Skv*4`。
6. **Tail tile 容量下限**：Skv%2048 ∈ (0,128) 时 TailCols tile <512B，
   LLVM static_assert 拒绝。TailPhy=128 物理列宽 + ValidCol 解决。

### 14.6 遗留与后续

- `qli_topk_radix` 基准配置与 P2 同等（kTk=32）；换 kTk=64 可与记录中
  P2(84K blocks) 直接对比。
- 候选提取用 tile pop（O(topK)×tile-op）；进一步可做 chunk-local TopK
  + 合并（§10.4），减少 pop 次数。
- 输出契约为 set；若下游需要降序，需补最终排序（§10.5）。
- BF16/UINT16 两级 radix：待 #342/#343 仿真器跨 dtype bug 修复后接入。

*Generated 2026-08-24 · Base: a68dba29 + 既有补丁 + 本轮 2 项仿真器修复（ByteId decode、
B8 门禁放宽）+ qli_pto_opt.hpp 增量（THISTOGRAMX/qli_topk_radix）*
