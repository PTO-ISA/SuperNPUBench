#ifndef QLI_PTO_OPT_HPP
#define QLI_PTO_OPT_HPP

// =============================================================================
// qli_pto.hpp — Quant Lightning Indexer (PTO tile-op variant)
// =============================================================================
//
// 【算子功能】
//   QuantLightningIndexer (QLI) 是 SparseFlashAttention (SFA) 的前处理算子。
//   从全量 K/V 序列中选出最关键的 token 索引，供后续稀疏注意力使用。
//
// 【计算公式】（与参考实现 quant_lightning_indexer_v2 一致）
//   score[s1, s2] = scale_k[s2] * Σ_g ( W[s1, g] * scale_q[s1, g] * ReLU(QK[g, s1, s2]) )
//
//   展开为 7 个步骤:
//     Step 1: S = Q @ K^T                       → [g, Skv]   量化矩阵乘
//     Step 2: S = ReLU(S)                        → [g, Skv]   激活过滤负信号
//     Step 3: load scale_q, fuse W *= scale_q    → [g, 1]    反量化 scale 折入权重
//     Step 4: S *= W*scale_q (broadcast)         → [g, Skv]   权重加权
//     Step 5: out = [1]_g @ S (ReduceG)          → [1, Skv]   沿 head 维度求和
//     Step 6: out *= scale_k                     → [1, Skv]   反量化 scale_k 后乘
//     Step 7: indices = TopK(out)                → [topK]     选取关键索引
//
// 【与参考实现的一致性】
//   参考实现 (quant_lightning_indexer_v2) 的 golden 公式:
//     score[s1,s2] = k_scale[s2] * Σ_g ( w[s1,g] * q_scale[s1,g] * ReLU(QK[g,s1,s2]) )
//   - scale_q[s1, g] 折入 weights，在 g-reduction 之内、ReLU 之后
//   - scale_k[s2] 在 g-reduction 之外作为后乘因子
//   本实现严格遵循此两阶段应用顺序。
//
// 【数据布局 — BSND】
//   Q:        [B, S, N, D] → 展平为 [S*N, D]，同一 token 的 g 个 head 连续存放
//   K:        [Skv, D]      → Key 不分组，所有 head 共享
//   W:        [S, N]        → 与 query 前 2 维同形，float
//   scale_q:  [S, N]        → 与 W 同形，per-token per-head，float
//   scale_k:  [Skv]         → per-token（N2=1），float
//   scores:   [S, Skv]      → ReduceG 后的输出
//   indices:  [S, topK]     → TopK 索引输出
//
// 【数据类型】
//   dtype: Q/K 的数据类型，支持 __half / int8_t
//   W / scale_q / scale_k: 统一使用 float（与参考实现 arch35 一致）
//
// 【与 CANN 原版的对应关系】
//   CANN 代码位置: op_kernel/arch22/quant_lightning_indexer_v2_*.h
//
//   Step 1 (Q@K^T):  CANN ComputeQk (Mmad)              → QLI TMATMUL(tSacc, tQ, tK)
//   Step 2 (ReLU):   CANN FixpSToL1 reluPre=1 (Fixpipe) → QLI TMAX(tS, tS, tZero)
//   Step 3 (W*Sq):   CANN ProcessVec0 Mul(W, QScale)    → QLI TMUL(tWf, tWf, tSq)
//   Step 4 (W广播):  CANN Brcb(W*QScale)                → QLI TCOLEXPANDMUL(tS, tS, tWf)
//   Step 5 (ReduceG):CANN ComputeWs Mmad([1,g]×[g,Skv]) → QLI TCOLSUM(tPartial, tRed)
//   Step 6 (ScK):    CANN ProcessVec1 Mul(mmIn, kScale) → QLI TMUL(tPartial, tPartial, tSk)
//   Step 7 (TopK):   CANN SortAll+MergeSort+LD           → QLI qli_topk_npu (NPU tile op)
//
//   CANN ProcessVec0 中先计算 W *= Scale_Q（Mul(inWeights, inQScale)），再 Brcb 广播；
//   QLI 在 Step 3 用 TMUL 融合 W *= scale_q，在 Step 4 用 TCOLEXPANDMUL 广播。
//   CANN ProcessVec1 中 Mul(mmIn, kScale) 在 g-reduction 之后应用 scale_k；
//   QLI 在 Step 6 用 TMUL 在 TCOLSUM 之后应用 scale_k。
//
// 【与 FA 的区别】
//   FA:  O = softmax(Q@K^T / √d) @ V    — online softmax, P@V 矩阵乘, 输出 [S, vD]
//   QLI: out = TopK(scale_k * ReduceG(W*scale_q * ReLU(Q@K^T)))  — 无 softmax, 无 V
//
// 【工具链约束】
//   - TCOLEXPANDMUL 使用 TileOP-API 标准算子（src0/src1/dst 均为 float，
//     满足 dtype 断言）。早期用于绕过 dtype 断言 bug 的 _TEPL 内联汇编变体
//     （test/common/template_asm.h）已随 v5 迁移弃用。
//   - TCOLSUM 参考 reduction/reducesum_colvec_pto.hpp 的归约模式。
//   - TCVT 用于布局转换（ColMajor → RowMajor）以满足 TCOLSUM 的输入要求。
//
// 【tile 尺寸约束】
//   每个 tile 活跃尺寸须在 128B..8KB（DavinciOO active PE-local profile）。
//   当前配置: dtype=__half/int8_t, D=128, kTm=16, kTk=32
//     tileQ:   [16, 128] × 2B = 4KB (fp16) / 2KB (int8)  ✅
//     tileK:   [32, 128] × 2B = 8KB (fp16) / 4KB (int8)  ✅
//     tileS:   [16, 32]  × 4B = 2KB (float)               ✅
//     tileWf:  [16, 8]   × 4B = 512B (float) valid[16,1]  ✅
//     tileSq:  [16, 8]   × 4B = 512B (float) valid[16,1]  ✅
//     tileSk:  [1,  32]  × 4B = 128B (float)              ✅ (同 tileSum)
//     tileSum: [1,  32]  × 4B = 128B (float)              ✅ (刚好达到下限)
//
// 【已知简化（相对 CANN 原版）】
//   - 不支持 Causal Mask (CANN sparse_mode=3, rightDownCausal)
//   - 不支持 actual_seq_lengths (变长序列)
//   - 不支持 PageAttention (block_table 间接寻址)
//   - TopK 在 NPU tile op 实现（TROWMAX+TCMP+TSEL argmax 模拟）
//   - g=kTm（未分块循环），g>16 时需扩展为 g/kTm 内层循环
// =============================================================================

#include <common/pto_tileop.hpp>
#include <cstdint>
#include <limits>

using namespace pto;

// -----------------------------------------------------------------------------
// qli_pto — QLI 核心计算（Step 1-6，NPU tile op 实现）
// -----------------------------------------------------------------------------
//
// 模板参数:
//   dtype   : Q/K 的数据类型（__half / int8_t）
//   Sq      : Query 序列长度 S
//   Skv     : Key 序列长度（通常 = Sq）
//   D       : Head 维度（固定 128）
//   g       : Head 数 N，kTm 必须等于 g
//   kTm     : tile M 维度（= g）
//   kTk     : tile K/N 维度（K block 大小，如 32）
//
// 输入:
//   q_ptr       : Q [Sq*g, D]       BSND 布局，dtype
//   k_ptr       : K [Skv, D]        所有 head 共享，dtype
//   w_ptr       : W [Sq, g]         BSND weight，float
//   scale_q_ptr : scale_q [Sq, g]   per-token per-head 反量化 scale，float
//   scale_k_ptr : scale_k [Skv]     per-token 反量化 scale，float
//
// 输出:
//   scores_ptr  : scores [Sq, Skv]  ReduceG + scale_k 后的 score 矩阵，float
//
// 外部循环结构:
//   for i in 0..Sq:           // 逐 token（外层 S 维度）
//     for j in 0..Kb:         // 逐 K-block（外层 Skv 维度，步长 kTk）
//       load K_j [kTk, D]     // K 只加载一次，所有 gi 共享
//       load scale_k_j        // scale_k 只加载一次
//       tSum = 0              // 片上 G 维累加器
//       for gi in 0..Gb:      // 逐 G-block（内层 g 维度，步长 kTm）
//         load Q_gi [kTm, D]  // 加载该 token 第 gi 组 head
//         load W_gi [kTm, 1]
//         load Sq_gi [kTm, 1]
//         W *= scale_q        // 融合：TMUL(tWf, tWf, tSq)
//         Step 1: Q@K^T       // TMATMUL → [kTm, kTk]
//         Step 2: ReLU        // TMAX
//         Step 3: skipped (W*scale_q already fused)
//         Step 4: *= W*scale_q // TCOLEXPANDMUL (broadcast)
//         Step 5: ReduceG     // TCOLSUM → [1, kTk]
//         if gi==0: tSum = tPartial
//         else:      tSum += tPartial  // TADD 累加跨 G-block
//       Step 6: tSum *= scale_k // TMUL
//       store scores[i, j]    // TSTORE [1, kTk]
// -----------------------------------------------------------------------------
template <typename dtype, int Sq, int Skv, int D, int g, int kTm, int kTk>
void qli_pto(float* scores_ptr,
             dtype* q_ptr, dtype* k_ptr,
             float* w_ptr,
             float* scale_q_ptr,
             float* scale_k_ptr)
{
    constexpr int Qb = Sq;              // S 维度 block 数（每 token 一个 block）
    constexpr int Kb = Skv / kTk;       // K 维度 block 数
    constexpr int Gb = g / kTm;         // G 维度 block 数（g 须为 kTm 的倍数）
    static_assert(g % kTm == 0, "g must be multiple of kTm for G-blocking");

    // ---- 全局张量形状和内存布局 ----
    using gmQ   = global_tensor<dtype,  RowMajor<Sq * g, D>>;
    using gmK   = global_tensor<dtype,  ColMajor<D, Skv>>;
    using gmOut = global_tensor<float,  RowMajor<Sq, Skv>>;

    // ---- tile 寄存器形状 ----
    using tileQ    = TileLeft<dtype, kTm, D>;
    using tileK    = TileRight<dtype, D, kTk>;
    using tileS    = Tile<Location::Vec, float, kTm, kTk, BLayout::RowMajor>;

    // W / scale_q: [kTm, 8] RowMajor valid=[kTm, 1], float
    using tileWf   = Tile<Location::Vec, float, kTm, 8, BLayout::RowMajor, kTm, 1>;
    using tileSq   = Tile<Location::Vec, float, kTm, 8, BLayout::RowMajor, kTm, 1>;

    // ReduceG / scale_k: [1, kTk] RowMajor, float
    using tileRed  = Tile<Location::Vec, float, kTm, kTk, BLayout::RowMajor>;
    using tileSum  = Tile<Location::Vec, float, 1, kTk, BLayout::RowMajor>;
    using tileSk   = Tile<Location::Vec, float, 1, kTk, BLayout::RowMajor>;

    // ---- 全局迭代器 ----
    using itQ   = global_iterator<gmQ,   tileQ>;
    using itK   = global_iterator<gmK,   tileK>;
    using itOut = global_iterator<gmOut, tileSum>;

    itQ   gIterQ(q_ptr);
    itK   gIterK(k_ptr);
    itOut gIterOut(scores_ptr);

    // ================================================================
    //  外层循环：逐 token 处理（S 维度）
    // ================================================================
    for (int i = 0; i < Qb; i++) {

        // ================================================================
        //  中层循环：逐 K-block（Skv 维度，步长 kTk）
        //    K 每个 (i,j) 只加载一次，所有 gi 共享
        // ================================================================
        for (int j = 0; j < Kb; j++) {

            // ---- TLOAD: 加载 K block [kTk, D] ----
            tileK tK;
            auto gK = gIterK(0, j);
            TLOAD(tK, gK);

            // ---- TLOAD: 加载 scale_k [1, kTk] ----
            tileSk tSk;
            {
                using gmSkLocal = global_tensor<float, RowMajor<1, Skv>>;
                using itSkLocal = global_iterator<gmSkLocal, tileSk>;
                itSkLocal gIterSk(scale_k_ptr);
                auto gSk = gIterSk(0, j);
                TLOAD(tSk, gSk);
            }

            // ---- 片上 G 维累加器 ----
            tileSum tSum;
            tileSum tZeroSum;
            TEXPANDS(tZeroSum, 0.0f);

            // ================================================================
            //  内层循环：逐 G-block（g 维度，步长 kTm）
            //    Q/W/scale_q 按 gi 分块加载
            //    必须 unroll(full)：不展开时 Linx TReg 分配 pass 崩溃
            //    已知限制：Gb>1 时 gfrun 输出非零元素减少（TReg 重用问题）
            // ================================================================
            #pragma clang loop unroll(full)
            for (int gi = 0; gi < Gb; gi++) {

                // ---- TLOAD: 加载 Q block [kTm, D] ----
                tileQ tQ;
                auto gQ = gIterQ(i * Gb + gi, 0);
                TLOAD(tQ, gQ);

                // ---- TLOAD: 加载 W [kTm, 1] (float) ----
                tileWf tWf;
                {
                    using gmWLocal = global_tensor<float, RowMajor<kTm, 1>>;
                    using itWLocal = global_iterator<gmWLocal, tileWf>;
                    itWLocal gIterW(w_ptr + (uint64_t)i * g + gi * kTm);
                    auto gW = gIterW(0, 0);
                    TLOAD(tWf, gW);
                }

                // ---- TLOAD: 加载 scale_q [kTm, 1] (float) ----
                tileSq tSq;
                {
                    using gmSqLocal = global_tensor<float, RowMajor<kTm, 1>>;
                    using itSqLocal = global_iterator<gmSqLocal, tileSq>;
                    itSqLocal gIterSq(scale_q_ptr + (uint64_t)i * g + gi * kTm);
                    auto gSq = gIterSq(0, 0);
                    TLOAD(tSq, gSq);
                }

                // ---- 融合: W *= scale_q ----
                TMUL(tWf, tWf, tSq);

                // ================================================================
                //  Step 1: S = Q @ K^T   (Cube: TMATMUL)
                // ================================================================
                tileS tS;
                TMATMUL(tS, tQ, tK);

                // ================================================================
                //  Step 2: S = ReLU(S)   (Vector: TEXPANDS + TMAX)
                // ================================================================
                tileS tZero;
                TEXPANDS(tZero, 0.0f);
                TMAX(tS, tS, tZero);

                // ================================================================
                //  Step 3+4: S *= (W * scale_q) (broadcast)   (Vector: TROWEXPANDMUL)
                // ================================================================
                TROWEXPANDMUL(tS, tS, tWf);

                // ================================================================
                //  Step 5: out = [1]_g @ S (ReduceG)   (Vector: TCVT + TCOLSUM)
                // ================================================================
                tileRed tRed;
                TCVT(tRed, tS);

                tileSum tPartial;
                TCOLSUM(tPartial, tRed);

                // ---- 累加跨 G-block 的 partial sum ----
                if (gi == 0) {
                    TADD(tSum, tZeroSum, tPartial);
                } else {
                    TADD(tSum, tSum, tPartial);
                }
            }

            // ================================================================
            //  Step 6: out *= scale_k   (Vector: TMUL)
            // ================================================================
            TMUL(tSum, tSum, tSk);

            // ---- TSTORE: 写回 score [1, kTk] 到全局内存 ----
            auto gOut = gIterOut(i, j);
            TSTORE(gOut, tSum);
        }
    }
}

// -----------------------------------------------------------------------------
// THISTOGRAMX — 自研展开版 THISTOGRAM（允许 Idx 与 src 不同 shape）
// -----------------------------------------------------------------------------
// TileOP-API 的 THISTOGRAM 要求 src/Idx 同 shape；但仿真器 ExecuteTHISTOGRAM
// 按各 tile 自身 tileInfo 读取 Idx 前缀行（validRow >= 3-selectedByte），
// 例如 [4,8] Idx 可承载 byte3/byte2/byte1 前缀。编码与 template_asm 的
// THISTOGRAM 一致（含 BSTOP，P3-D5）。Spike A/B/C 已验证 ByteId=0..3 +
// 前缀约束 + scalar→heap→TLOAD 桥全部正确。
template <typename tile_o, typename tile_s, typename tile_idx>
void THISTOGRAMX(tile_o& dst, tile_s& src, tile_idx& idx, int ByteId) {
#define THISTOGRAMX_ASM(BYTE_NAME)                                    \
  asm volatile(                                                        \
    "BSTART.TEPL 0b1101000, %c1\n"                                     \
    "B.DATR %c2," BYTE_NAME ",Null\n"                                  \
    "B.DIM %3, 0, ->LB0\n"                                             \
    "B.DIM %4, 0, ->LB1\n"                                             \
    "B.DIM zero, %c5, ->LB2\n"                                         \
    "B.IOT %6, %7, mask=15, last, ->%0<%Z8>\n"                         \
    "BSTOP\n"                                                          \
    : "=Tr"(dst.data())                                                \
    : "i"(type_traits<typename tile_s::DType>::TypeCode),              \
      "i"(type_traits<typename tile_o::DType>::TypeCode),              \
      "r"(src.GetValidCol()),                                          \
      "r"(src.GetValidRow()),                                          \
      "i"(tile_s::Cols),                                               \
      "Tr"(src.data()),                                                \
      "Tr"(idx.data()),                                                \
      "i"(tile_type_traits<typename tile_o::TileDType>::TilesizeCode))
  switch (ByteId) {
    case 0: THISTOGRAMX_ASM("Byte0"); break;
    case 1: THISTOGRAMX_ASM("Byte1"); break;
    case 2: THISTOGRAMX_ASM("Byte2"); break;
    default: THISTOGRAMX_ASM("Byte3"); break;
  }
#undef THISTOGRAMX_ASM
}

// -----------------------------------------------------------------------------
// qli_topk_radix — 分块多轮 MSD radix-select TopK（Step 7）
// -----------------------------------------------------------------------------
// 【设计】（对齐 qli_pto_opt_histogram_radix_design.md §10.2-§10.4）
//   1. 逐 chunk: TLOAD float scores as uint32 → tile op IEEE-754 单调映射
//      （NaN→0, 正→bits|MSB, 负→~bits）成 sortable key，TSTORE 原位写回
//   2. 逐字节 MSD radix（Byte3→Byte0）：
//        THISTOGRAMX(ByteId=r, Idx=[已定高位字节前缀]) 统计 active 集合
//        标量合并多 chunk 直方图（累积→差分→累加→重算累积）
//        从高桶向下累计 → kth_byte + above 计数；remaining -= above
//        若 remaining==0 提前停止
//   3. 提取（输出契约为 TopK set，set 语义下无需降序）：
//        Case A（提前停止）: mask = key & keep_mask > kth & keep_mask
//        Case B（4 轮后仍有并列）: %gt mask = key > kth（全部入选）
//          + %eq mask = key == kth（并列取剩余，最小索引优先，rev 技巧）
//        逐 chunk: tile 内 pop-argmax（TROWMAX+TCMP+索引运算+TMUL 消零，
//          全程无标量 store、无重 TLOAD，避免 P2 的 BROB stall 瓶颈）
//          pop 顺序 = key 降序；set 语义直接按序写入 indices
// 【约束】
//   - Skv % 8 == 0；Src dtype uint32→ UINT32 域操作（TCMP mask 0/1 算术可用）
//   - Idx 前缀 tile [4,8]（validRow=4 ≥ 3-selectedByte），scalar→heap→TLOAD 构造
//   - tail chunk（Skv 非 MaxTileCol 倍数）：validCol < 2048 用编译期 TailCols 分支
// -----------------------------------------------------------------------------
namespace qli_radix_detail {

using RU = uint32_t;

template <int CK, int CKV = CK>
using TKey = Tile<Location::Vec, RU, 1, CK, BLayout::RowMajor, 1, CKV>;
template <int CK, int CKV = CK>
using GMKey = global_tensor<RU, RowMajor<1, CKV>>;

// 读 src 位模式 → sortable key → TSTORE 到 dst（独立 scratch，不覆盖 scores）
template <int CK, int CKV = CK>
inline void RadixMakeKey(RU* dst_key, const RU* src_bits)
{
    using tk = TKey<CK, CKV>;
    using gk_s = global_tensor<RU, RowMajor<1, CKV>>;
    gk_s gs(const_cast<RU*>(src_bits));
    tk bits;
    TLOAD(bits, gs);
    constexpr RU FP32_NAN_MASK   = 0x7F800000u;  // exponent 全 1
    constexpr RU FP32_MANT_MASK  = 0x007FFFFFu;  // mantissa
    constexpr RU FP32_SIGN_MASK  = 0x80000000u;
    // NaN 判定：exponent 全 1 && mantissa 非 0（覆盖正/负/quiet/signaling，
    //     以及任意 payload），而非仅匹配单一规范位型
    //     nan01 = (exp==EXP_MASK) AND (mant != 0) → 0/1
    tk tExp;     TANDS(tExp, bits, FP32_NAN_MASK);
    tk tExpA;    TEXPANDS(tExpA, FP32_NAN_MASK);
    tk tExpIs;   TCMP(tExpIs, tExp, tExpA);        // exponent==full → 0/1
    tk tMant;    TANDS(tMant, bits, FP32_MANT_MASK);
    tk tMantZ;   TEXPANDS(tMantZ, static_cast<RU>(0));
    tk tMantIs;  TCMP<CmpMode::NE>(tMantIs, tMant, tMantZ); // mant!=0
    tk tNan01;   TMUL(tNan01, tExpIs, tMantIs);    // NaN 判定 0/1
    tk tNotBits; TNOT(tNotBits, bits);
    tk tNanPart; TMUL(tNanPart, tNotBits, tNan01);
    tk key0;   TADD(key0, bits, tNanPart);
    tk tSign;  TANDS(tSign, key0, FP32_SIGN_MASK);
    tk tNeg;   TNOT(tNeg, key0);
    tk tPos;   TORS(tPos, key0, FP32_SIGN_MASK);
    tk tSign01; TSHRS(tSign01, tSign, static_cast<RU>(31));
    tk tDiff;  TSUB(tDiff, tNeg, tPos);
    TMUL(tDiff, tDiff, tSign01);
    tk key;    TADD(key, tPos, tDiff);
    gk_s gd(dst_key);
    TSTORE(gd, key);
}

// 单 chunk 直方图：THISTOGRAM(ByteId=r, Idx=prefix) → TSTORE 256 累积到 hist_gm
template <int CK, int CKV = CK>
inline void RadixChunkHist(RU* key_ptr, int r, RU* prefix_buf, RU* hist_gm_ptr)
{
    using tk = TKey<CK, CKV>;
    using gk = GMKey<CK, CKV>;
    using tidx  = Tile<Location::Vec, RU, 4, 8, BLayout::RowMajor>;
    using gidx  = global_tensor<RU, RowMajor<4, 8>>;
    using th    = Tile<Location::Vec, RU, 1, 256, BLayout::RowMajor>;
    using gh    = global_tensor<RU, RowMajor<1, 256>>;

    gk g(key_ptr);
    tk key;
    TLOAD(key, g);

    tidx idxTile;
    { gidx gi(prefix_buf); TLOAD(idxTile, gi); }

    th hist;
    THISTOGRAMX(hist, key, idxTile, r);

    gh gout(hist_gm_ptr);
    TSTORE(gout, hist);
}

// 0/1 掩码 t 求和 → cntOut（标量 GM [1,1]）
template <int CK, int CKV = CK>
inline void RadixCountOf(TKey<CK, CKV>& m01, RU* cntOut)
{
    using t1 = Tile<Location::Vec, RU, 1, 32, BLayout::RowMajor, 1, 1>;
    using gi1 = global_tensor<RU, RowMajor<1, 1>>;
    t1 s;
    TROWSUM(s, m01);
    gi1 g(cntOut);
    TSTORE(g, s);
}

// pop n 次（mv 内元素，去重，每 pop 只消所选位置），写入 out
// 注意：必须内联为宏——tile 作函数参数时 LinxV5 后端生成错误编码
// （S64 [1,1024] TLOAD，TCMP 校验失败）。宏内所有 tile 均为局部变量。
// A1：改用 TROWARGMAX 单指令取最大 key 的列索引（省 TROWMAX+TCMP+TMUL+TROWMAX
// 索引反推链），加 CHBASE 得全局索引；set 契约无需降序。
#define QLI_RADIX_POP_N(TK, MV, CHBASE, OUTPTR, N)                       \
    do {                                                               \
        using tkv_ = TK;                                               \
        using t1v_ = Tile<Location::Vec, RU, 1, 32,                    \
                          BLayout::RowMajor, 1, 1>;                    \
        using t1iv_ = Tile<Location::Vec, int32_t, 1, 32,              \
                           BLayout::RowMajor, 1, 1>;                   \
        tkv_ I_;                                                       \
        TCI(I_, (CHBASE));                                             \
        for (int k_ = 0; k_ < (N); k_++) {                             \
            t1v_ best_;                                                \
            TROWARGMAX(best_, (MV));          /* 最大 key 的列索引 */ \
            t1v_ bestg_;                                               \
            TADDS(bestg_, best_, (CHBASE));   /* 全局索引 */          \
            t1iv_ besti_;                                              \
            TCVT(besti_, bestg_);                                      \
            { using gi1_ = global_tensor<int32_t, RowMajor<1, 1>>;     \
              gi1_ gout_((OUTPTR) + k_); TSTORE(gout_, besti_); }      \
            RU bestVal_ = 0;                                           \
            { using gi1_ = global_tensor<RU, RowMajor<1, 1>>;          \
              gi1_ gb_(&bestVal_); TSTORE(gb_, bestg_); }              \
            tkv_ bestbc_;                                              \
            TEXPANDS(bestbc_, bestVal_);                               \
            tkv_ ispick_;                                              \
            TCMP<CmpMode::EQ>(ispick_, I_, bestbc_);                   \
            tkv_ one_;                                                 \
            TEXPANDS(one_, static_cast<RU>(1));                        \
            tkv_ notpick_;                                             \
            TSUB(notpick_, one_, ispick_);                             \
            TMUL((MV), (MV), notpick_);                                \
        }                                                              \
    } while (0)

// pop 并列候选 k 次，最小全局索引优先（rev 技巧），写入 out（同样必须内联为宏）
#define QLI_RADIX_POP_EQN(TK, EQMASK, CHBASE, N1, OUTPTR, N)            \
    do {                                                               \
        using tkq_ = TK;                                               \
        using t1q_ = Tile<Location::Vec, RU, 1, 32,                    \
                          BLayout::RowMajor, 1, 1>;                    \
        using t1iq_ = Tile<Location::Vec, int32_t, 1, 32,              \
                           BLayout::RowMajor, 1, 1>;                   \
        tkq_ I_;                                                       \
        TCI(I_, (CHBASE));                                             \
        tkq_ n1k_;                                                     \
        TEXPANDS(n1k_, (N1));                                          \
        for (int k_ = 0; k_ < (N); k_++) {                             \
            tkq_ rev_;                                                 \
            TSUB(rev_, n1k_, I_);                                      \
            tkq_ rev1_;                                                \
            TADDS(rev1_, rev_, static_cast<RU>(1));                    \
            tkq_ pv_;                                                  \
            TMUL(pv_, rev1_, (EQMASK));                                \
            t1q_ best_;                                                \
            TROWARGMAX(best_, pv_);      /* 最大 rev+1 = 最小索引（列号） */ \
            t1q_ bestg_;                                               \
            TADDS(bestg_, best_, (CHBASE));   /* 全局最小索引 */        \
            t1iq_ besti_;                                              \
            TCVT(besti_, bestg_);                                      \
            { using gi1_ = global_tensor<int32_t, RowMajor<1, 1>>;     \
              gi1_ gout_((OUTPTR) + k_); TSTORE(gout_, besti_); }      \
            RU bestVal_ = 0;                                           \
            { using gi1_ = global_tensor<RU, RowMajor<1, 1>>;          \
              gi1_ gb_(&bestVal_); TSTORE(gb_, bestg_); }              \
            tkq_ bestbc_;                                              \
            TEXPANDS(bestbc_, bestVal_);                               \
            tkq_ ispick_;                                              \
            TCMP<CmpMode::EQ>(ispick_, I_, bestbc_);                   \
            tkq_ one_;                                                 \
            TEXPANDS(one_, static_cast<RU>(1));                        \
            tkq_ notpick_;                                             \
            TSUB(notpick_, one_, ispick_);                             \
            TMUL((EQMASK), (EQMASK), notpick_);                        \
        }                                                              \
    } while (0)

}  // namespace qli_radix_detail

// ==================== qli_topk_radix 主函数 ====================
template <int Sq, int Skv, int topK>
void qli_topk_radix(float* scores_gm, int32_t* indices_gm) {
    using namespace qli_radix_detail;
    static_assert(Skv % 8 == 0, "Skv must be multiple of 8");
    static_assert(topK <= Skv, "topK must be <= Skv");

    constexpr int MaxTileCol = 2048;
    constexpr int NumChunks = (Skv + MaxTileCol - 1) / MaxTileCol;
    constexpr int TailCols = Skv - (NumChunks - 1) * MaxTileCol;
    // 单 chunk/尾 chunk 物理列宽：保证 ≥512B（128 元素×4B）；valid 用实际列数
    constexpr int SinglePhy = (Skv < 128) ? 128 : Skv;
    constexpr int TailPhy = (TailCols < 128) ? 128 : TailCols;

    // scratch 布局（紧随 indices 输出区，调用方须保证可写）：
    //   key_scratch: Sq*Skv*4 字节 可容纳全部 key（不覆盖 scores_gm 供 cosine 验证）
    //   temp_hist:   [1,256] 直方图 TSTORE 缓冲 1KB(=256*4B)
    //   prefix_buf:  [4,8]  Idx 前缀 128B
    uint32_t* key_scratch = reinterpret_cast<uint32_t*>(
        reinterpret_cast<uint8_t*>(indices_gm) + (uint64_t)Sq * topK * 4 + 8192);
    uint32_t* temp_hist = key_scratch + (uint64_t)Sq * Skv;
    uint32_t* prefix_buf = temp_hist + 256;

    for (int i = 0; i < Sq; i++) {
        const RU* row = reinterpret_cast<const RU*>(scores_gm) + (uint64_t)i * Skv;
        RU* krow = key_scratch + (uint64_t)i * Skv;

        // ---- 0) 逐 chunk: scores → sortable key（写 key_scratch）----
        // 普通 chunk: TKey<MaxTileCol>；tail chunk: TKey<MaxTileCol, TailCols>
#define MAKEKEY(CK, CKV)                                                    \
        {                                                                  \
            RadixMakeKey<CK, CKV>(krow + (uint64_t)c * MaxTileCol,        \
                                  row + (uint64_t)c * MaxTileCol);         \
        }
        if constexpr (NumChunks == 1) {
            { constexpr int c = 0; MAKEKEY(SinglePhy, Skv); }
        } else {
            if constexpr (TailCols != MaxTileCol) {  // 有 tail
                for (int c = 0; c < NumChunks - 1; c++) { MAKEKEY(MaxTileCol, MaxTileCol); }
                { constexpr int c = NumChunks - 1; MAKEKEY(TailPhy, TailCols); }
            } else {
                for (int c = 0; c < NumChunks; c++) { MAKEKEY(MaxTileCol, MaxTileCol); }
            }
        }
#undef MAKEKEY

        // ---- 1) 逐字节 MSD radix 找 kth_value（Boundary 直到 Byte0）----
        // 注：单选第 remaining 大元素时，每轮恒有 above < remaining，
        //     因此必然走满 Byte3..Byte0 四轮，不存在"提前停止"。
        RU kth_value = 0;
        int remaining = topK;

        // prefix_buf 初始清零（[4,8] = 32 个 uint32）
        for (int w = 0; w < 32; w++) prefix_buf[w] = 0;

        for (int r = 3; r >= 0; r--) {
            RU per_bin[256];
            for (int b = 0; b < 256; b++) per_bin[b] = 0;

            // 当前轮前缀: 已定高位字节 kth（r<3 时需要 prefix 约束）
            if (r < 3) {
                volatile RU* pb = reinterpret_cast<volatile RU*>(prefix_buf);
                RU b3 = (kth_value >> 24) & 0xFFu;
                RU b2 = (kth_value >> 16) & 0xFFu;
                RU b1 = (kth_value >> 8) & 0xFFu;
                for (int cc = 0; cc < 8; cc++) {
                    pb[0*8+cc] = b3;
                    pb[1*8+cc] = b2;
                    pb[2*8+cc] = b1;
                    pb[3*8+cc] = 0;
                }
            }

#define HCHUNK(CK, CKV)                                                    \
            {                                                              \
                RadixChunkHist<CK, CKV>(krow + (uint64_t)c * MaxTileCol, r,\
                                        prefix_buf, temp_hist);            \
                volatile RU* rh = reinterpret_cast<volatile RU*>(temp_hist); \
                RU prev = 0;                                               \
                for (int b = 0; b < 256; b++) {                            \
                    RU cum = rh[b];                                        \
                    per_bin[b] += (cum - prev);                            \
                    prev = cum;                                            \
                }                                                          \
            }
            if constexpr (NumChunks == 1) {
                { constexpr int c = 0; HCHUNK(SinglePhy, Skv); }
            } else {
                if constexpr (TailCols != MaxTileCol) {
                    for (int c = 0; c < NumChunks - 1; c++) { HCHUNK(MaxTileCol, MaxTileCol); }
                    { constexpr int c = NumChunks - 1; HCHUNK(TailPhy, TailCols); }
                } else {
                    for (int c = 0; c < NumChunks; c++) { HCHUNK(MaxTileCol, MaxTileCol); }
                }
            }
#undef HCHUNK

            // 从高桶向下累计→ kth_byte（首个 cum >= remaining）
            RU cum = 0;
            int kth_byte = 0;
            for (int b = 255; b >= 0; b--) {
                cum += per_bin[b];
                if (cum >= (RU)remaining) { kth_byte = b; break; }
            }
            remaining -= (int)(cum - per_bin[kth_byte]);
            kth_value |= ((RU)kth_byte) << (r * 8);
        }

        // ---- 2) 提取（输出契约为 TopK set）----
        //   pass 1: 逐 chunk 输出全部 key > kth_value（gt）
        //   pass 2: 再逐 chunk 从 key == kth_value（eq，并列）中按全局 remaining
        //           补足最小索引。两遍分离保证 gt（更大 key）始终先于 eq
        //           占据 outPos 前缀，避免 eq 提前占位导致越界写。
        int outPos = 0;
        const RU n1 = (RU)(Skv - 1);
        const RU kthVal = kth_value;
        int needEq = remaining;   // 还需并列补足的数量（radix 结束时 remaining）

#define EXTRACT_GT(CK, CKV)                                                \
        {                                                                  \
            using tk = TKey<CK, CKV>; using gk = GMKey<CK, CKV>;           \
            gk g(krow + (uint64_t)c * MaxTileCol);                         \
            tk key; TLOAD(key, g);                                         \
            tk kthk; TEXPANDS(kthk, kthVal);                               \
            tk isgt; TCMP<CmpMode::GT>(isgt, key, kthk);                   \
            tk cand; TMUL(cand, key, isgt);   /* 仅 >kth 保留 key 用于 pop */\
            RU cntGt = 0;                                                  \
            RadixCountOf<CK, CKV>(isgt, &cntGt);                           \
            QLI_RADIX_POP_N(tk, cand, (RU)c * MaxTileCol,                  \
                            reinterpret_cast<int32_t*>(indices_gm) +       \
                                (uint64_t)i * topK + outPos,               \
                            static_cast<int>(cntGt));                      \
            outPos += (int)cntGt;                                          \
        }

#define EXTRACT_EQ(CK, CKV)                                                \
        {                                                                  \
            if (needEq > 0) {                                              \
                using tk = TKey<CK, CKV>; using gk = GMKey<CK, CKV>;       \
                gk g(krow + (uint64_t)c * MaxTileCol);                     \
                tk key; TLOAD(key, g);                                     \
                tk kthk; TEXPANDS(kthk, kthVal);                           \
                tk iseq; TCMP<CmpMode::EQ>(iseq, key, kthk);               \
                RU cntEq = 0;                                              \
                RadixCountOf<CK, CKV>(iseq, &cntEq);                       \
                int take = static_cast<int>(cntEq);                        \
                if (take > needEq) take = needEq;                          \
                QLI_RADIX_POP_EQN(tk, iseq, (RU)c * MaxTileCol, n1,        \
                                  reinterpret_cast<int32_t*>(indices_gm) + \
                                      (uint64_t)i * topK + outPos, take);  \
                outPos += take;                                            \
                needEq -= take;                                            \
            }                                                              \
        }

        if constexpr (NumChunks == 1) {
            { constexpr int c = 0; EXTRACT_GT(SinglePhy, Skv); }
        } else {
            if constexpr (TailCols != MaxTileCol) {
                for (int c = 0; c < NumChunks - 1; c++) { EXTRACT_GT(MaxTileCol, MaxTileCol); }
                { constexpr int c = NumChunks - 1; EXTRACT_GT(TailPhy, TailCols); }
            } else {
                for (int c = 0; c < NumChunks; c++) { EXTRACT_GT(MaxTileCol, MaxTileCol); }
            }
        }
        if constexpr (NumChunks == 1) {
            { constexpr int c = 0; EXTRACT_EQ(SinglePhy, Skv); }
        } else {
            if constexpr (TailCols != MaxTileCol) {
                for (int c = 0; c < NumChunks - 1; c++) { EXTRACT_EQ(MaxTileCol, MaxTileCol); }
                { constexpr int c = NumChunks - 1; EXTRACT_EQ(TailPhy, TailCols); }
            } else {
                for (int c = 0; c < NumChunks; c++) { EXTRACT_EQ(MaxTileCol, MaxTileCol); }
            }
        }
#undef EXTRACT_GT
#undef EXTRACT_EQ
    }
}

#endif
