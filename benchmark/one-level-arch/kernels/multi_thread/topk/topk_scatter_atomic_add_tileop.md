# Top-K radix-select：FP16 粗筛 + FP32 四字节精筛的 TileOP 数据流

本文把 `examples/deepseek_v32/topk_selector.py` 的 `tl_topk_impl` 翻译为
TileOP 级伪代码。目标是描述数据流、形状、循环域、同步和原子 scatter，
不是给出可直接汇编的二进制 bundle。

## 1. 接口、布局和共享状态

```text
function TOPK_RADIX_SELECT(
    input  : FP32[B,S] in GM,
    starts : S32[B]    in GM,
    ends   : S32[B]    in GM,
    topk   : X0<S32>,
    bx     : X1<S32>
) -> index : S32[B,topk] in GM
```

每个 block 处理一行 `input[bx,:]`：

```text
X2<S32> = max(0, starts[bx])你可以r
X3<S32> = min(S, ends[bx])
X4<S32> = max(0, X3 - X2)       // 有效长度 N
X5<S32>  = 0                    // integer zero / false value
X5F<FP32> = 0.0                 // FP32 zero for x < 0 comparisons
X6<S32>  = 1                    // integer one / true value
X7<S32> = 4                     // sizeof(S32)，字节
X8<S32> = 0x000000ff            // byte mask
X9<S32> = 0x00008000            // FP16 sign mask
X10<S32> = 0x80000000           // FP32 sign mask
X11<S32> = topk                 // 剩余需求 new_topk
```

TileOP 采用 32-lane 的 `M32` 逻辑向量。源码在 CUDA 中每个线程处理 4 个
元素；在 TileOP listing 中等价地按 32 个元素一个 tile 表示：

```text
chunk = 0 .. ceil(X4 / 32)-1
valid_col = min(32, X4 - 32*chunk)
base_idx  = X2 + 32*chunk
```

| 逻辑对象 | Tile 形状 | 逻辑 payload | M32 CellReg 分配 |
|---|---|---:|---:|
| 32 个 S32 索引/计数 | `<1,32,1,S32,M32>` | 128B | 128B |
| 32 个 FP32 | `<1,32,1,FP32,M32>` | 128B | 128B |
| 32 个 U16（一个逻辑列） | `<1,32,2,U16,M32>` | 64B | 128B |
| 32 个 predicate bit | `<1,32,1,PRED,M32,PredicateCell>` | 4B | 128B |
| 256 个 S32 histogram bin | `<8,32,8,S32,M32>` | 1KB | 1KB |

全局内存（L2）对象，每个 `bx` 使用互不重叠区域：

```text
H    : S32[257]       in GM     // H[0:255] histogram，H[256] 为 sentinel 槽
Num  : S32[2]         in GM     // ping-pong 候选数
Cand : S32[2,4096]    in GM     // ping-pong 候选索引
```

`H`/`Num`/`Cand` 全部放 GM，是为了用 TLSU 的 GM atom/red 和 indexed transfer
（没有 SMEM atomic，也没有 indexed Shared scatter）。`Cand` 的 4096 是源码的固定
容量假设；每一轮必须保证 `Num[r] <= 4096`。

下文统一用 `GM[H_gm]`/`GM[Num_gm]`/`GM[Cand_gm]` 表示这些对象所在的 GM 区，
不再出现 `SMEM[...]`。

## 2. 通用 FP16 sortable-key 数据流

第一阶段的每次输入扫描都执行以下 TileOP 序列。`TCVT FP32 -> FP16`
负责数值转换；转换结果的 16-bit payload 随后直接以 `U16` 类型视图供
位运算消费，从而表达源代码中的 `reinterpret`，不再单独插入位重解释操作。
这里的 `U16` 只是同一 payload 的整数类型视图，不是一次数值转换。

```text
TLOAD <1,32,1,FP32,M32> GM[input[bx,base_idx]],
       valid_col=valid_col, valid_row=1, pad=0
    -> T0<1,32,1,FP32,M32><128B>

TCVT <1,32,1,FP32,M32> T0 -> T1<1,32,2,FP16,M32><128B>, round=RN
TCMPS CMode=LT <1,32,1,FP32,M32> T0, X5F<FP32>
    -> T3<1,32,1,PRED,M32,PredicateCell><128B>
// T1 的同一 16-bit payload 以 U16 类型视图消费，表达 reinterpret；不产生新 tile
TNOT <1,32,2,U16,M32> T1 -> T4<1,32,2,U16,M32><128B>
TANDS <1,32,2,U16,M32> T4, 0xffff -> T5<1,32,2,U16,M32><128B>
TORS <1,32,2,U16,M32> T1, 0x8000 -> T6<1,32,2,U16,M32><128B>
TSELS <1,32,2,U16,M32> T3, T5, T6
    -> T7<1,32,2,U16,M32><128B>              // FP16 sortable key
TSHRS <1,32,2,U16,M32> T7, 8 -> T8<1,32,2,U16,M32><128B>
TCVT <1,32,2,U16,M32> T8 -> T9<1,32,1,S32,M32><128B> // U16 数值转换为 S32，bin 0..255
```

对应数学式：

```text
raw16 = bits(cast_FP16(x))
key16 = (x < 0) ? (~raw16 & 0xffff) : (raw16 | 0x8000)
bin16 = key16 >> 8
```

这里 `T8 -> T9` 的 `TCVT` 是数值类型转换，不是 bit reinterpret：`T8`
中的值已经是 `0..255` 的 U16 桶号，转换为 S32 后才能作为 histogram 的
索引/字节位移参与后续 `TMULS` 和 scatter。tail pass 中的 `T49 -> T50`
同理，是 `U32` 当前字节桶号到 `S32` 的数值转换。

必须先 `FP32 -> FP16`，再对 FP16 bit 做符号排序变换；不能把 FP32 排序
key 当成普通浮点数再 cast 成 FP16。

## 3. 第一阶段：FP16 高 8 bit 粗筛

### 3.1 初始化

```text
T10 = broadcast(X5, 256)          // 逻辑零 tile；不是 PTO TCONST_ZERO 指令
TSTORE <8,32,8,S32,M32> T10 -> GM[H_gm+0], valid_col=256, valid_row=32
T11 = broadcast(X5, 1)            // 逻辑零 tile；不是 PTO TCONST_ZERO 指令
TSTORE <1,32,1,S32,M32> T11 -> GM[Num_gm+0], valid_col=1, valid_row=1
TSTORE <1,32,1,S32,M32> T11 -> GM[H_gm+256], valid_col=1, valid_row=1
X11<S32> = topk
```

### 3.2 扫描完整输入并建立 histogram

histogram 放在 GM（L2）：`H_gm` 为当前 `bx` 私有的 257×S32 区域，各 `bx` 不
重叠。这一遍的原子加用 **`MGATHER_ADD`**（TLSU function 12 = `GMAtomic_ADD`，
atom 形式，handler `GM_ATOM_VALUE`）：它既是 GM 原子 RMW，又把 old value 发布到
destination Tile，可同时覆盖 3.2 的 `H[bin] += 1` 和后续 `return_prev=True` 的
位置分配。listing-level 注意：`lane = 0..31` 是静态逻辑坐标、不是 opcode；用正式
的 `TEXPANDS` 把 GPR 常量 1 扩成目标 Tile，替代不存在的 `TCONST_ONE`。

```text
// [需要屏障] H_gm 的 3.1 清零必须全部完成且可见后，才能开始原子累加
for chunk = 0 .. ceil(X4/32)-1:
    valid_col = min(32, X4 - 32*chunk)
    base_idx  = X2 + 32*chunk

    // 展开第 2 节，得到 T9 = bin16
    TEXPANDS <1,32,1,S32,M32> scalar=X6
        -> T16<1,32,1,S32,M32><128B>     // 每 lane 的加 1 数据
    TMULS <1,32,1,S32,M32> T9, X7
        -> T17<1,32,1,S32,M32><128B>     // bin 的字节位移
    MGATHER_ADD <S32, PE_MASK> [GM[H_gm]]
        index=T17, value=T16
        -> T18<1,32,1,S32,M32><128B>     // 本遍 old counter 丢弃
```

形式为 `MGATHER_ADD <DataType, PEMask>, [BaseGPR], IndexTile, ValueTile, ->DstTile`：
对每个 valid element 做 `GM[BaseGPR + index*width] = old + value`，并把 `old` 写入
destination 的对应元素。`S32` 在 ADD 的合法类型内，`T17 = bin*4` 正是 S32 counter
的字节位移；参与元素由 Index/Value/Dst 的 valid region 决定，尾部
`lane >= valid_col` 靠有效区域裁剪排除，不需要 per-lane predicate。注意 `PE_MASK`
只是 4-bit 的 **per-PE** 参与掩码，不是 per-lane mask。

若只关心计数、不需要 old value，当前 ISA 也有 red 形式 `MSCATTER_ADD`（TLSU
function 21 = `GMReduction_ADD`，只绑 IndexTile + ValueTile、无 destination），
本文未采用；统一用 `MGATHER_ADD` 以保留 old-value 语义。

`MGATHER_ADD` 没有独立 per-lane predicate。若只想对某些谓词为真的 lane 更新
（如 3.5 的 `T28`/`T29`），必须先把 active lane 压缩，或走逐 lane atomic
lowering；不能把它写成带 `MaskTile` 的原子指令——带 per-lane `MaskTile` 的是
**非原子**的 `MSCATTER_MASK`（TLSU function 7），不是它的 mask 变体。

关于 lane-index tile（`T13[lane] = base_idx + lane`）：正式的 `TCI` 目前仍只生成
RowMajor、`ValidRow=1` 的单行 Local Tile，无法直接产出 M32
`<ValidCol=1,ValidRow=32,PhysicalCol=1>` 形状，ISA 里也没有 `TIDX`。3.2 这一遍只
消费 `bin`、不消费 `T13`；`T13` 是 3.5/4.2 写回 `index[bx,pos]` 时用的，需要预物化
索引 tile 或明确的后端 lowering。

注意：H 既落在 GM，3.1 的清零、3.3 的 suffix cumsum、3.4 与 4.1 的 threshold 读取
以及 4.1 的清零/扫描都必须是针对 GM 的 `TLOAD/TSTORE`（`H` 是 CUBE 布局 tile，
可用 Function 1 的 Local CUBE→GM 形式写回），这不是只替换 atomic mnemonic 就完成的。

### 3.3 256-bin suffix cumsum

```text
// [需要屏障] 进入 cumsum 前：所有 lane 的 histogram 累加必须完成且对读可见
for i = 0 .. 7:
    offset = 1 << i                 // 1,2,4,8,16,32,64,128
    // [需要屏障] 每步 doubling 之间：H[bin] 与 H[bin+offset] 分属不同 lane，
    // 必须等上一步所有写完成后才能读（PTO 无 radix 子组屏障）
    for bin = 0 .. 255-offset:
        // 逻辑 slice；不是新的 ISA VIEW 指令
        TADD <1,32,1,S32,M32> H[bin] + H[bin+offset] -> H[bin]
    // [需要屏障] 本步结果对下一步可见
// 完成后 H[bin] = sum_{j=bin..255} original_hist[j]
```

也可以把该固定子流程写成算法级组合点：

```text
CALL HISTOGRAM_SUFFIX_CUMSUM_M32(H)
```

这个 CALL 不是已确认存在的 ISA opcode。

### 3.4 找到粗筛阈值桶

`REDUCE_FIRST_TRUE` 不是当前 PTO ISA 指令。这里不把 PredicateCell
转换成不存在的“first-true”操作，而是利用当前 ISA 已有的 CUBE GPR
predicate 形式：对 S32/M32 的 8 个逻辑列按每 2 列分成 4 组；每组的
`TCMPS` 输出一个 64-bit predicate GPR，随后用标量 `AND` 合并条件，
再用 `CTZ` 找到该组最低的 true bit。M32 的 bit 编号按
`bit = row + column * 32`，所以组内结果加上 `64*pair` 就是全局 bin。

```text
T19 = broadcast(X5, 1)            // 逻辑零 tile；不是 PTO TCONST_ZERO 指令
TSTORE <1,32,1,S32,M32> T19 -> GM[threshold_bin_gm], valid_col=1, valid_row=1

for pair = 0 .. 3:
    bin_base = 64*pair
    TLOAD <2,32,2,S32,M32> GM[H_gm + bin_base], valid_col=2, valid_row=32
        -> T20<2,32,2,S32,M32><256B>
    TLOAD <2,32,2,S32,M32> GM[H_gm + bin_base + 1], valid_col=2, valid_row=32
        -> T21<2,32,2,S32,M32><256B>  // 逻辑上取 H[bin+1]

    // CUBE_M32/S32 GPR predicate form；每个结果覆盖 2*32=64 个位置
    TCMPS CMode=GE <2,32,2,S32,M32> T20, X11 -> X20<U64 predicate mask>
    TCMPS CMode=LT <2,32,2,S32,M32> T21, X11 -> X21<U64 predicate mask>
    AND X20, X21 -> X22<U64 predicate mask>

    if X22 != 0:
        CTZ X22, 0, 64 -> X23<S32>
        X12<S32> = bin_base + X23
        break

if every pair had X22 == 0:
    X12<S32> = 0                   // 无 crossing：保留源码默认 threshold_bin_id = 0
                                   // 不能用 256，否则 X13=257 越界

X13<S32> = X12 + X6
// 标量读 H_gm[threshold+1]：H 在 GM，直接用 lw，不经过 tile
// [此处需要 fence.d] 产者是 TLSU/tile 的 TSTORE/atomic，lw 是 relaxed load，
// 需先 fence.d 才能读到最新值
lw [H_base, X13<<2] -> X14<S32>   // X14 = H_gm[threshold+1]
X11<S32> = X11 - X14
```

判断条件对应源码：

```text
H[bin] >= new_topk && H[bin+1] < new_topk
```

读取 `H[threshold+1]` 不再用 tile：`lw [SrcL, SrcR<<shamt]` 的地址是
`H_base + X13*4`（`SrcL`=当前 bx 的 counter 区 GM 基址，`SrcR`=X13，`shamt`=2），
4 字节对齐、符号扩展（`asl/scalar/agu/LW.asl`）。三点约束：

1. **对齐**：`H` 为 S32 数组，天然 4 字节对齐；否则 `Fault_DataAlignment`。
2. **排序**：`lw` 是 relaxed load，读的是 TLSU/tile 写进 GM 的值，必须在产者之后加
   `fence.d`（`asl/scalar/sys/FENCE.D.asl`）；普通屏障只保证 lane 到达，不能替代
   跨 agent 的 fence。
3. **范围**：`X12` 必须落在 `0..255`，使 `X13∈1..256` 命中 `H[0..256]`（含 sentinel）；
   无 crossing 时按源码保持默认 0。

`H_base` 是标量地址算术（`H_gm + bx*stride`，用 `lui/addi/slli/add`），不需要 tile。
`TCI` 生成 lane-index 的限制在此仍不涉及（这一步只用标量索引）。

### 3.5 第二次扫描：输出或保存阈值候选

```text
for chunk = 0 .. ceil(X4/32)-1:
    valid_col = min(32, X4 - 32*chunk)
    base_idx  = X2 + 32*chunk
    // 重新 TLOAD input，并展开第 2 节得到 T9 = bin16

    TCMPS CMode=GT <1,32,1,S32,M32> T9, X12
        -> T28<1,32,1,PRED,M32,PredicateCell><128B>
    TCMPS CMode=EQ <1,32,1,S32,M32> T9, X12
        -> T29<1,32,1,PRED,M32,PredicateCell><128B>

    TADDS <1,32,1,S32,M32> T9, X6 -> T33<1,32,1,S32,M32><128B> // bin+1
    TMULS <1,32,1,S32,M32> T33, X7 -> T34<1,32,1,S32,M32><128B> // 字节位移 = (bin+1)*4
    // MGATHER_ADD 的 index 是字节位移（TileMemoryByteDisplacementAddress），故 *4
    MGATHER_ADD <S32, PE_MASK> [GM[H_gm]]
        index=T34, value=T16
        -> T35<1,32,1,S32,M32><128B>   // old = 输出元素位置（元素索引，非字节）
        // mask=T28：MGATHER_ADD 无 per-lane predicate，需 active-lane 压缩
    // MSCATTER_MASK 的 IndexTile 是逻辑线性“元素索引”，硬件按元素大小自动换算，不再 *4
    MSCATTER_MASK base=GM[index[bx,0]], stride=1, index=T35, data=T13, mask=T28

    // 谓词转 0/1：非 T29 lane 加 0，不推进计数器（绕开缺失的 per-lane atomic mask）
    TEXPANDS <1,32,1,S32,M32> scalar=X5 -> Tzero<1,32,1,S32,M32><128B>
    TSELS <1,32,1,S32,M32> T29, T16, Tzero
        -> Tval<1,32,1,S32,M32><128B>   // Tval = T29 ? 1 : 0
    // 真指令 MGATHER_ADD：GM 计数器，index=0 是字节位移，所有 lane 命中 GM[Num_gm+0]
    MGATHER_ADD <S32, PE_MASK> [GM[Num_gm+0]]
        index=0, value=Tval
        -> T37<1,32,1,S32,M32><128B>   // T37 = old 槽位（T29 lane 唯一）
    // 真指令 MSCATTER_MASK：GM 上带 per-lane MaskTile 的 indexed scatter
    MSCATTER_MASK base=GM[Cand_gm], stride=1, index=T37,
                  data=T13, mask=T29
```

`T13` 是原始 `input_idx`，不是排序 key。H_gm 上的 old-value 用 `MGATHER_ADD`
取得，用于给每个 lane 分配紧凑位置；同一桶内的输出顺序不稳定。

候选 append（源码 `atomic_add(s_num_input, 1, return_prev=True)` + `s_input_idx[pos]=idx`）
不再用不存在的 `MSCATTER_ATOMIC_ADD_MASKED`/`SHARED_SCATTER_MASK`，而是拆成当前
ISA 的真指令：

- `TSELS` 把谓词 `T29` 转成 0/1，`MGATHER_ADD` 对 `GM[Num]` 做原子加：`value=1`
  的 lane 得到唯一槽位并让计数器 +1，`value=0` 的 lane 加 0、计数器不变。这样即使
  `MGATHER_ADD` 没有 per-lane predicate，也只让 `T29` lane 推进计数器（代价是非
  `T29` lane 也发一次 +0 的 atomic RMW）。
- `MSCATTER_MASK` 是 GM 上真正带 per-lane `MaskTile` 的 indexed scatter，用来
  `Cand[old] = input_idx`。

前提是 `Num`/`Cand` 放在 GM（见第 1 节）；否则当前 ISA 确实无对应指令。

注意两种 index 语义：`MGATHER_ADD`/`MSCATTER_ADD`（GM atom/red）的 index 是
**字节位移**，所以要 `*4`；`MGATHER`/`MSCATTER`/`MSCATTER_MASK`（indexed TLSU）
的 index 是**逻辑线性元素索引**，由硬件按元素大小和 row stride 换算，**不能
再 `*4`**。B.IOR 的 `stride` 是 GM row stride（以元素为单位，≥ ValidCol）；
这里数据是扁平数组、`ValidCol=1`，故 `stride=1`。

## 4. 第二阶段：FP32 四轮 tail pass

### 4.1 外层 `round` 循环和当前候选 byte

```text
for round = 0 .. 3:
    if X11 <= 0: break
    r      = round % 2
    next_r = r ^ 1
    X15<S32> = topk - X11       // Out 已占用的前缀长度

    // [需要屏障] 复用 H/Num 前：上一轮对这些对象的读必须已结束（反依赖）
    TSTORE <8,32,8,S32,M32> T10 -> GM[H_gm+0], valid_col=256, valid_row=32
    // Num 在 GM：清零/读取都是单个计数器，用标量 sw/lw（TSTORE 不能写普通 Local->GM）
    sw [Num_base, next_r<<2], X5      // Num[next_r] = 0
    // [需要屏障] 清零必须对后续 MGATHER_ADD 可见后才能开始累加

    // [此处需要 fence.d] Num 由 TLSU 的 MGATHER_ADD 写、标量 lw 读，需排序
    lw [Num_base, r<<2] -> X16<S32>   // X16 = C_r（候选数）

    for tile = 0 .. ceil(X16/32)-1:
        valid_col = min(32, X16 - 32*tile)
        TLOAD <1,32,1,S32,M32> GM[Cand_gm + r*4096 + 32*tile],
               valid_col=valid_col, valid_row=1, pad=0
            -> T40<1,32,1,S32,M32><128B>
        // MGATHER 的 IndexTile 是逻辑元素索引，硬件按 FP32 元素大小换算，不 *4
        MGATHER <1,32,1,FP32,M32> base=GM[input[bx,0]], stride=1, index=T40,
                valid_col=valid_col, valid_row=1, pad=0
            -> T42<1,32,1,FP32,M32><128B>

        TCMPS CMode=LT <1,32,1,FP32,M32> T42, X5F
            -> T44<1,32,1,PRED,M32,PredicateCell><128B>
        // T42 的同一 32-bit payload 以 U32 类型视图消费，表达 reinterpret；不产生新 tile
        TNOT <1,32,1,U32,M32> T42 -> T45<1,32,1,U32,M32><128B>
        TORS <1,32,1,U32,M32> T42, X10 -> T46<1,32,1,U32,M32><128B>
        TSELS <1,32,1,U32,M32> T44, T45, T46
            -> T47<1,32,1,U32,M32><128B>       // FP32 sortable key
        X17<S32> = 24 - 8*round
        TSHRS <1,32,1,U32,M32> T47, X17 -> T48<1,32,1,U32,M32><128B>
        TANDS <1,32,1,U32,M32> T48, X8 -> T49<1,32,1,U32,M32><128B>
        TCVT <1,32,1,U32,M32> T49 -> T50<1,32,1,S32,M32><128B> // U32 数值转换为 S32 桶号
        TMULS <1,32,1,S32,M32> T50, X7 -> T51<1,32,1,S32,M32><128B>
        // 当前 tile 的所有候选 lane 都有效；尾部由 valid_col 裁剪，不需要 TIDX
        MGATHER_ADD <S32, PE_MASK> [GM[H_gm]]
            index=T51, value=T16
            -> T55<1,32,1,S32,M32><128B>   // old counter 丢弃
            // 尾部由 valid_col 裁剪；PE_MASK 是 per-PE，不是 per-lane

    // [需要屏障] 本 tile 的 H 累加必须全部完成后才能 cumsum
    // H 上再做同样的 8 轮 suffix cumsum
    for i = 0 .. 7:
        offset = 1 << i
        // [需要屏障] 每步 doubling 之间（H[bin] 与 H[bin+offset] 跨 lane）
        for bin = 0 .. 255-offset:
            TADD <1,32,1,S32,M32> H[bin] + H[bin+offset] -> H[bin]
        // [需要屏障] 本步结果对下一步可见

    // threshold byte：按 3.4 的现有 ISA GPR lowering 分成 4 组
    for pair = 0 .. 3:
        bin_base = 64*pair
        TLOAD <2,32,2,S32,M32> GM[H_gm + bin_base], valid_col=2, valid_row=32
            -> T56<2,32,2,S32,M32><256B>
        TLOAD <2,32,2,S32,M32> GM[H_gm + bin_base + 1], valid_col=2, valid_row=32
            -> T57<2,32,2,S32,M32><256B>
        TCMPS CMode=GE <2,32,2,S32,M32> T56, X11 -> X24<U64 predicate mask>
        TCMPS CMode=LT <2,32,2,S32,M32> T57, X11 -> X25<U64 predicate mask>
        AND X24, X25 -> X26<U64 predicate mask>
        if X26 != 0:
            CTZ X26, 0, 64 -> X27<S32>
            X18<S32> = bin_base + X27
            break
    if every pair had X26 == 0:
        X18<S32> = 0                   // 无 crossing：默认 0（避免 X18+1 越界）

    // 同 3.4：H 在 GM，threshold+1 用标量 lw 读，不用 tile 提取
    X13<S32> = X18 + X6                // threshold + 1
    // [此处需要 fence.d] H 由 TLSU 的 TSTORE/cumsum 写，标量 lw 需排序
    lw [H_base, X13<<2] -> X19<S32>    // X19 = H_gm[threshold+1]
    X11<S32> = X11 - X19
```

### 4.2 当前轮候选的输出/继续缓存

```text
for tile = 0 .. ceil(X16/32)-1:
    valid_col = min(32, X16 - 32*tile)
    // 重新加载 Cand，并重复 4.1 的 MGATHER -> FP32 key -> 当前 byte，得到 T50

    TCMPS CMode=GT <1,32,1,S32,M32> T50, X18
        -> T62<1,32,1,PRED,M32,PredicateCell><128B>
    TCMPS CMode=EQ <1,32,1,S32,M32> T50, X18
        -> T63<1,32,1,PRED,M32,PredicateCell><128B>

    TADDS <1,32,1,S32,M32> T50, X6 -> T66<1,32,1,S32,M32><128B> // bin+1
    TMULS <1,32,1,S32,M32> T66, X7 -> T67<1,32,1,S32,M32><128B> // 字节位移
    // MGATHER_ADD index 是字节位移，故 *4
    MGATHER_ADD <S32, PE_MASK> [GM[H_gm]]
        index=T67, value=T16
        -> T68<1,32,1,S32,M32><128B>   // old = 元素位置
        // mask=T62：MGATHER_ADD 无 per-lane predicate，需 active-lane 压缩
    TADDS <1,32,1,S32,M32> T68, X15 -> T69<1,32,1,S32,M32><128B> // old + 前缀（元素索引）
    // MSCATTER_MASK index 是元素索引，不 *4
    MSCATTER_MASK base=GM[index[bx,0]], stride=1, index=T69,
                  data=Cand[r,32*tile], mask=T62

    if round < 3:
        // 同 3.5：谓词转 0/1 + MGATHER_ADD 做 GM 计数器（value=0 不推进）
        TEXPANDS <1,32,1,S32,M32> scalar=X5 -> Tzero<1,32,1,S32,M32><128B>
        TSELS <1,32,1,S32,M32> T63, T16, Tzero
            -> Tval<1,32,1,S32,M32><128B>   // Tval = T63 ? 1 : 0
        MGATHER_ADD <S32, PE_MASK> [GM[Num_gm+next_r]]
            index=0, value=Tval
            -> T71<1,32,1,S32,M32><128B>   // T71 = old 槽位（T63 lane 唯一）
        MSCATTER_MASK base=GM[Cand_gm+next_r*4096], stride=1, index=T71,
                      data=Cand[r,32*tile], mask=T63
    else:
        // round==3：最低 byte 已处理完，阈值桶内候选直接写 Out
        TCMPS CMode=LT <1,32,1,S32,M32> T69, Xtopk<S32>
            -> T73<1,32,1,PRED,M32,PredicateCell><128B>
        MSCATTER_MASK base=GM[index[bx,0]], stride=1, index=T69,
                      data=Cand[r,32*tile],
                      mask=(T63 AND T73)  // provisional: current ISA has no PredicateTile-AND
```

最后的边界判断使用独立的 `Xtopk<S32>` GPR 表示 `topk`。`T69` 是
`old_counter + output_prefix`，因此实现的条件是 `T69 < Xtopk`。

## 5. Tile/GPR 数据流表

| 名称 | 用途 | 形状/类型 | 生命周期 |
|---|---|---|---|
| `T0` | 输入 FP32 tile | `<1,32,1,FP32,M32>` | 每个 chunk/候选 tile |
| `T1..T8` | FP16 payload 的 U16 类型视图、bit key、bin | `<1,32,2,U16,M32>` | 第一阶段扫描 |
| `T9` | FP16 高 8-bit bin | `<1,32,1,S32,M32>` | 第一阶段扫描 |
| `T10` | zero histogram tile | `<8,32,8,S32,M32>` | 初始化/清空 |
| `T13` | 原始输入索引 | `<1,32,1,S32,M32>` | 扫描和写回 |
| `T20/T21` | 每次 threshold pair 的 `H[bin]`、`H[bin+1]` | `<2,32,2,S32,M32>` | threshold 查找，4 组复用 |
| `X20..X22` | 一组 `GE`/`LT` GPR mask 及其 AND 结果 | 64-bit GPR | threshold pair，循环复用 |
| `X24..X26` | 下一阶段一组 `GE`/`LT` GPR mask 及其 AND 结果 | 64-bit GPR | threshold pair，循环复用 |
| `X23/X27` | `CTZ` 组内下标 | GPR `S32` | 命中 pair 时短暂存在 |
| `T28/T29` | GT/EQ predicate | `<1,32,1,PRED,...>` | 第一阶段收集 |
| `T35/T37` | atomic old counter | `<1,32,1,S32,M32>` | 输出/候选位置 |
| `T40` | Cand 索引 tile | `<1,32,1,S32,M32>` | tail pass |
| `T42` | gather 得到 FP32 | `<1,32,1,FP32,M32>` | tail pass |
| `T42,T45..T50` | FP32 payload 的 U32 类型视图、sortable key/current byte | `<1,32,1,U32/S32,M32>` | 每轮 radix |
| `T62/T63` | 当前 byte 的 GT/EQ PredicateCell | `<1,32,1,PRED,...>` | tail 写回 |
| `H` | 直方图/位置计数器 | `S32[257]` in GM | block 全生命周期 |
| `Num[0:2]` | 候选数量 ping-pong | `S32[2]` in GM | stage 1 + tail |
| `Cand[0:2]` | 候选索引 ping-pong | `S32[2,4096]` in GM | stage 1 + tail |
| `X11` | 剩余 topk | GPR `S32` | block 全生命周期 |
| `X18` | 当前 radix threshold | GPR `S32` | 每轮 tail |

Tile 是逻辑目的地，物理 T/U/M queue 由编译器分配；`X#` 是私有 GPR。
单元素标量读（如 `H[threshold+1]`、`Num[r]`）用标量 `lw`（配 `fence.d`），
不依赖不存在的数值 tile→GPR 提取。

## 6. 算法步骤与 TileOP 对应

| Python 步骤 | TileOP 级对应 |
|---|---|
| 读取输入 | `TLOAD`（stage 1）或 `MGATHER`（tail） |
| FP32→FP16 | `TCVT FP32 -> FP16` |
| 读取 bit pattern | 后续位运算以 `U16`/`U32` 类型视图消费同一 payload |
| 浮点 sortable key | `TCMPS + TNOT/TANDS/TORS + TSELS` |
| 取高 8 bit/当前 byte | `TSHRS + TANDS + TCVT` |
| histogram | GM atomic-add；用 `MGATHER_ADD`（TLSU function 12，GM 原子 RMW + 发布 old value），`PE_MASK` 为 per-PE、非 per-lane |
| suffix cumsum | 8 轮 `TADD`，步骤间需全局屏障（PTO 无对应指令） |
| threshold | 4 组 `TCMPS(GPR) + AND + CTZ`，每组覆盖 `<2,32,2,S32,M32>` |
| 紧凑输出 | GM 写回用 `MSCATTER_MASK`（带 per-lane `MaskTile`） |
| 候选 append | 计数：`TSELS`(谓词→0/1) + `MGATHER_ADD`(GM 原子加，value=0 不推进)；写入：`MSCATTER_MASK` |
| 候选 ping-pong | `Num[r]` 和 `Cand[r]` 的 `r^1` 交替写入 |
| 最后 byte | 直接写 `Out`，检查 `out_pos < topk` |

## 7. ISA、同步和边界假设

1. `TCI` 是当前 PTO ISA 的正式指令，可生成递增/递减整数序列，但仅限
   RowMajor、`ValidRow=1` 的单行 Local Tile；当前 M32 vector 采用
   `<ValidCol=1,ValidRow=32,PhysicalCol=1>`，因此不能直接用 TCI 生成
   同形状的 lane-index tile。`TIDX` 仍不可作为 ISA mnemonic；若需要
   M32 lane index，必须使用预先物化/加载的索引 Tile 或明确的后端 lowering。
2. `TEXPANDS` 是当前 PTO ISA 的正式指令，可把 GPR 标量的 raw encoding
   广播到新建 Local Tile，因此可以替代 `TCONST_ONE`/`TCONST_ZERO` 的
   listing-level 意图；目标 Tile 的 dtype、layout、valid shape 必须满足
   `TEXPANDS` contract。
3. PTO ISA 的 Tile/ASL contract 已定义并测试了 `MSCATTER_MASK`（TLSU function 7，
   非原子 GM scatter）：它带独立的 `MaskTile`，mask=0 的 lane 不生成地址、不做
   权限检查、不写内存、不产生 memory event。这里的 `MaskTile` 是 per-lane
   数据 mask；B.IOT 上的 `PE_MASK` 只是参与执行的 PE mask，不能替代它。它也
   不是 `MSCATTER_ADD` 的 mask 变体。当前 Streaming-L2 的 as-implemented 文档
   仍将 `MSCATTER_MASK` 列为延期项，因此不能据此声称整条后端路径已经闭环。
4. 当前 ISA 的 atomic-add 采用 `MGATHER_ADD`（TLSU function 12 = `GMAtomic_ADD`，
   atom 形式，带 destination 并发布 old value）；另有 red 形式 `MSCATTER_ADD`
   （function 21 = `GMReduction_ADD`，无 destination），本文未采用。两者 bundle
   都只有 `IndexTile + ValueTile + PE_MASK`，该 `PE_MASK` 是 4-bit **per-PE**
   掩码，不能替代 per-lane predicate；带 per-lane `MaskTile` 的是非原子的
   `MSCATTER_MASK`（function 7）。因此“只对谓词为真的 lane 做 atomic_add”在当前
   ISA 用 `TSELS`(谓词→0/1) + `MGATHER_ADD` 表达：加 0 的 lane 不改变计数器，
   再由 `MSCATTER_MASK` 按 mask 写回。文档不再使用不存在的
   `MSCATTER_ATOMIC_ADD_MASKED`/`SHARED_SCATTER_MASK`。
5. 无 per-lane mask 的 atomic 计数只能用“`value=0` 不推进计数器”来近似：它语义
   正确（加 0 不改变 counter），但非目标 lane 仍会发起 +0 的 atomic RMW，产生地址
   生成、冲突/排序和访存流量。因此它只适合计数器这种“加 0 即无操作”的对象；对写
   数据的 scatter 必须用真正带 per-lane `MaskTile` 的 `MSCATTER_MASK`，不能靠写 0
   顶替。对于有 `valid_col` 的尾部 tile，仍直接裁剪有效区域。
6. 通过 `U16`/`U32` 类型视图表达 reinterpret 的写法依赖 TileOP listing 的
   类型视图约定；predicate-AND 和逻辑 Tile slice 仍需映射到目标 ISA 的
   predicate carrier、range `Subview/Assemble` 或等价序列。本文的 threshold
   first-true 已改写为当前 ISA 可表达的 `TCMPS(GPR) + AND + CTZ`，不再使用
   不存在的 `REDUCE_FIRST_TRUE`。当前 ISA 没有通用数值 tile→GPR 提取（只有
   `TCMP/TCMPS` 的 predicate-GPR carrier），因此 3.4 读 `H[threshold+1]` 改为
   标量 `lw`（配合 `fence.d`），不再依赖不存在的 scalar extraction。
7. `TLOAD/MGATHER/TSTORE` 的 `valid_col` 描述逻辑有效元素；尾部 lane 必须
   补零，且不能读取下一个 batch 行。
8. listing 里的 `// [需要屏障]` 表示该处需要一次全局同步（PTO 没有对应的屏障
   指令，必须 lowered 到目标后端的同步/Shared 发布/atomic 组合）：histogram
   清零与累加之间、累加与 suffix cumsum 之间、cumsum 每个 doubling 步骤之间、
   以及每轮 tail 复用 `H`/`Num` 之前。listing 不再写出 `sync_threads()`。
9. CUDA 源码使用 1024 threads，HIP 使用 256 threads。这里用 32-lane M32
   描述逻辑 tile，实际线程分块/barrier group 由后端展开。
10. 第一阶段只使用 FP16 sortable key 的高 8 bit；第二阶段重新读取原始
   FP32，依次处理 key 的 `31:24`、`23:16`、`15:8`、`7:0`。
11. 输出是 Top-K 索引集合，不保证同一桶内的顺序，也不等同于
   `torch.topk(..., sorted=True)` 的排序结果。
12. `Q=4096` 是候选 workspace 的正确性边界；每次 append 都必须证明
   `Num[r] < Q`。
13. indexed TLSU 的 index 语义分两类，不能混用：
   - **GM atom/red**（`MGATHER_ADD`/`MSCATTER_ADD`，function 12/21）用
     `TileMemoryByteDisplacementAddress`，index 是**字节位移**
     （`asl/tile/model/memory/addressing.asl:113`），所以 bin 相关偏移要 `*4`。
   - **indexed transfer**（`MGATHER`/`MSCATTER`/`MSCATTER_MASK`，function 4/6/7）
     用 `TileMemoryIndexedStridedAddress`，index 是**逻辑线性元素索引**，硬件按
     transfer 元素大小和 B.IOR row stride 换算（`addressing.asl:128`），
     **不能再 `*4`**。
   本文 listing 已按此区分：atom/red 保留 `*4`；indexed transfer 直接传元素索引，
   并显式写出 `stride=1`（扁平数组、`ValidCol=1`，故 row stride 取 1）。

## 8. 复杂度和访存量

设有效长度为 `N`，第一阶段候选数为 `C0`，tail 各轮候选数为 `C1..C4`：

```text
O(N + C0 + C1 + C2 + C3 + C4)
```

第一阶段约两次扫描输入（建 histogram、收集候选）；tail 每轮对当前候选
做一次 `MGATHER + histogram` 和一次分类写回。直方图只有 256 个逻辑桶，
每次 suffix cumsum 固定执行 8 个 doubling 步骤。
