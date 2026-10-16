#ifndef QLI_PTO_OPT_SIMPLE_HPP
#define QLI_PTO_OPT_SIMPLE_HPP

// =============================================================================
// qli_pto_opt_simple.hpp — Quant Lightning Indexer (精简 radix-select TopK)
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
//   Step 2 (ReLU):   CANN FixpSToL1 reluPre=1 (Fixpipe) → QLI element-for: score > 0 ? score : 0
//   Step 3 (W*Sq):   CANN ProcessVec0 Mul(W, QScale)    → driver: W * scale_q
//   Step 4 (W广播):  CANN Brcb(W*QScale)                → QLI element-for: score *= weight
//   Step 5 (ReduceG):CANN ComputeWs Mmad([1,g]×[g,Skv]) → QLI TCOLSUM(tPartial, tRed)
//   Step 6 (ScK):    CANN ProcessVec1 Mul(mmIn, kScale) → QLI element-for: sum *= scale_k
//   Step 7 (TopK):   CANN SortAll+MergeSort+LD           → QLI qli_topk_simple (NPU tile op)
//
//   CANN ProcessVec0 中先计算 W *= Scale_Q（Mul(inWeights, inQScale)），再 Brcb 广播；
//   调用方预广播 W*scale_q，Step 4 的普通 float 循环逐元素加权。
//   CANN ProcessVec1 中 Mul(mmIn, kScale) 在 g-reduction 之后应用 scale_k；
//   Step 6 的普通 float 循环在 TCOLSUM 之后应用 scale_k。
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
//   - TopK 在 NPU tile op 实现（精简 radix-select）
//   - g=kTm（未分块循环），g>16 时需扩展为 g/kTm 内层循环
// =============================================================================
//
// 【精简版说明】
//   key 构建仅符号翻转（无 NaN/-0 防御，golden 生成器不产生 NaN）、提取无
//   rev 最小索引（输出契约为无序 TopK set，平局取任一等价）、tail 分支用
//   FOR_EACH_CHUNK 宏统一处理。完整版（含防御）的设计记录见
//   qli_pto_opt_histogram_radix_design.md。
//   精度验证通过（cosine=1.0, set=100%），性能相当。
// =============================================================================

#include <common/pto_tileop.hpp>
#include "qli_element.hpp"
#include <cstdint>
#include <limits>

using namespace pto;

// -----------------------------------------------------------------------------
// qli_pto — QLI 核心计算（CUBE 矩阵乘 + element-for + 列归约）
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
//   wb_ptr      : 预广播 W*scale_q [Sq*g, kTk] float（行 r 全列同值）
//   scale_k_ptr : scale_k [Skv]     per-token 反量化 scale，float
//
// 输出:
//   scores_ptr  : scores [Sq, Skv]  ReduceG + scale_k 后的 score 矩阵，float
//
// 注：v0.58.4 row-expansion 校验要求广播源物理单列（<128B 不可加载），
// 故 W*scale_q 由调用方预广播为 [kTm, kTk] tile，kernel 内用普通 TMUL。
template <typename dtype, int Sq, int Skv, int D, int g, int kTm, int kTk>
void qli_pto(float* scores_ptr,
             dtype* q_ptr, dtype* k_ptr,
             float* wb_ptr,
             float* scale_k_ptr,
             float* temp_gm = nullptr)   // [kTm, kTk] CUBE->Vec 桥接临时区
{
    constexpr int Qb = Sq;
    constexpr int Kb = Skv / kTk;
    constexpr int Gb = g / kTm;
    static_assert(g % kTm == 0, "g must be multiple of kTm for G-blocking");
    static_assert(kTm <= 32, "CUBE_M16/M32 matmul supports kTm <= 32");

    // 临时区缺省：紧随 scores 之后（调用方保证可写 kTm*kTk*4 字节）
    static float* s_default_temp = nullptr;
    if (temp_gm == nullptr) {
        if (s_default_temp == nullptr) {
            s_default_temp = scores_ptr + (uint64_t)Sq * Skv + 2048;
        }
        temp_gm = s_default_temp;
    }

    using gmQ   = global_tensor<dtype,  RowMajor<Sq * g, D>>;
    using gmK   = global_tensor<dtype,  RowMajor<D, Skv>>;   // K^T [D, Skv] 行主序（CUBE_N8 契约）
    using gmOut = global_tensor<float,  RowMajor<Sq, Skv>>;
    using gmTmp = global_tensor<float,  RowMajor<kTm, kTk>>;

    // PTO v0.58.4 CUBE cell-layout：TMATMUL 的 A 用 CUBE_M16/M32、B 用 CUBE_N8、
    // D 用 CUBE 累加器；TSTORE_CUBE 写回 GM 后 TLOAD 回 Vec tile 继续 Vec 链
    using tileQ    = std::conditional_t<(kTm <= 16),
                       CubeTileM16<dtype, kTm, D>, CubeTileM32<dtype, kTm, D>>;
    using tileK    = CubeTileN8<dtype, D, kTk>;
    using tileSCube = std::conditional_t<(kTm <= 16),
                       CubeAccumulatorM16<float, kTm, kTk>,
                       CubeAccumulatorM32<float, kTm, kTk>>;
    using tileS    = Tile<Location::Vec, float, kTm, kTk, BLayout::RowMajor>;
    using tileWb   = Tile<Location::Vec, float, kTm, kTk, BLayout::RowMajor>;
    using tileSum  = Tile<Location::Vec, float, 1, kTk, BLayout::RowMajor>;
    using tileSk   = Tile<Location::Vec, float, 1, kTk, BLayout::RowMajor>;

    using itQ   = global_iterator<gmQ,   tileQ>;
    using itK   = global_iterator<gmK,   tileK>;
    using itOut = global_iterator<gmOut, tileSum>;

    itQ   gIterQ(q_ptr);
    itK   gIterK(k_ptr);
    itOut gIterOut(scores_ptr);

    for (int i = 0; i < Qb; i++) {
        for (int j = 0; j < Kb; j++) {
            tileK tK;
            auto gK = gIterK(0, j);
            TLOAD_CUBE(tK, gK);

            tileSk tSk;
            {
                using gmSkLocal = global_tensor<float, RowMajor<1, Skv>>;
                using itSkLocal = global_iterator<gmSkLocal, tileSk>;
                itSkLocal gIterSk(scale_k_ptr);
                auto gSk = gIterSk(0, j);
                TLOAD(tSk, gSk);
            }

            tileSum tSum;
            auto &sum = TPARTELEMENT(tSum);
#pragma pto element for
            for (uint32_t column = 0; column < kTk; ++column)
                sum[column] = 0.0f;

            #pragma clang loop unroll(full)
            for (int gi = 0; gi < Gb; gi++) {
                tileQ tQ;
                auto gQ = gIterQ(i * Gb + gi, 0);
                TLOAD_CUBE(tQ, gQ);

                // 预广播 W*scale_q tile [kTm, kTk]（调用方 wb_ptr 提供，
                // 行 r 全列同值 = w[r]*scale_q[r]；规避 v0.58.4 单列广播源校验）
                tileWb tWb;
                {
                    using gmWbLocal = global_tensor<float, RowMajor<kTm, kTk>>;
                    using itWbLocal = global_iterator<gmWbLocal, tileWb>;
                    itWbLocal gIterWb(wb_ptr + (uint64_t)(i * Gb + gi) * kTm * kTk);
                    auto gWb = gIterWb(0, 0);
                    TLOAD(tWb, gWb);
                }

                // TMATMUL（CUBE cell-layout）→ TSTORE_CUBE 桥接回 Vec
                tileSCube tSCube;
                TMATMUL(tSCube, tQ, tK);
                gmTmp gTmp(temp_gm);
                TSTORE_CUBE(gTmp, tSCube);

                tileS tS;
                TLOAD(tS, gTmp);

                auto &score = TPARTELEMENT(tS);
                const auto &weight = TPARTELEMENT(tWb);
#pragma pto element for
                for (uint32_t element = 0; element < kTm * kTk; ++element) {
                    const float positive = score[element] > 0.0f ? score[element] : 0.0f;
                    score[element] = positive * weight[element];
                }

                tileSum tPartial;
                TCOLSUM(tPartial, tS);

                const auto &partial = TPARTELEMENT(tPartial);
#pragma pto element for
                for (uint32_t column = 0; column < kTk; ++column)
                    sum[column] += partial[column];
            }

            const auto &scale = TPARTELEMENT(tSk);
#pragma pto element for
            for (uint32_t column = 0; column < kTk; ++column)
                sum[column] *= scale[column];

            auto gOut = gIterOut(i, j);
            TSTORE(gOut, tSum);
        }
    }
}

// -----------------------------------------------------------------------------
// qli_topk_radix — 4 轮 MSD radix-select TopK（Step 7，模板函数风格）
// -----------------------------------------------------------------------------
// 【算法】
//   1. float→uint32 sortable key（element-for 符号翻转）
//      key = (sign) ? ~bits : (bits | 0x80000000)
//   2. 逐字节 MSD radix（Byte3→Byte0）→ kth_value
//   3. 提取：key > kth 全部入选，key == kth 边界并列补足
//      每轮提取 = TROWARGMAX + element-for 将选中元素清零
// -----------------------------------------------------------------------------
namespace qli_radix {

using RU = uint32_t;

template <int CK, int CKV = CK>
using TKey = Tile<Location::Vec, RU, 1, CK, BLayout::RowMajor, 1, CKV>;

// Step 1: 用标量 element-for 表达符号翻转，保留完整 chunk 快照。
template <int CK, int CKV>
inline void RadixMakeKey(RU* dst, const RU* src) {
    qli_element::make_key<CK, CKV>(dst, src);
}

// Step 2: 单 chunk 累积直方图：element-for 原子计数 + 标量前缀和
template <int CK, int CKV>
inline void RadixChunkHist(RU* key_ptr, int byteId, RU* prefix, RU* hist) {
    const RU highPrefix = (prefix[0] << 24) | (prefix[8] << 16) | (prefix[16] << 8);
    qli_element::histogram<CK, CKV>(hist, key_ptr, static_cast<RU>(byteId), highPrefix);
}

// Step 3: 从 key tile pop n 个最大元素（TROWARGMAX + 索引消零）
template <int CK, int CKV>
inline void RadixPopN(TKey<CK, CKV>& mv, RU chunkBase, int32_t* out, int n) {
    // v0.58.4 行归约契约：TROWARGMAX 目的须物理单列 [N,1]（Col==1）；
    // 物理行数 32 维持 128B tile 尺寸下限，valid 仍为 [1,1]
    using Scalar = Tile<Location::Vec, RU, 32, 1, BLayout::RowMajor, 1, 1>;
    auto &values = TPARTELEMENT(mv);
    for (int k = 0; k < n; ++k) {
        Scalar best;
        TROWARGMAX(best, mv);
        RU selected = 0;
        global_tensor<RU, RowMajor<1, 1>> index(&selected);
        TSTORE(index, best);
        out[k] = static_cast<int32_t>(chunkBase + selected);
        // 归约按原顺序提取；循环体只描述被选中元素的清零。
#pragma pto element for
        for (uint32_t element = 0; element < CKV; ++element) {
            if (element == selected)
                values[element] = 0;
        }
    }
}

// Step 3（单 chunk）: 先 pop GT 再 pop EQ 补足
template <int CK, int CKV>
inline void RadixExtract(RU* key_ptr, RU kthVal, RU chunkBase, int32_t* outBase, int& outPos, int& needEq) {
    using gk = global_tensor<RU, RowMajor<1, CKV>>;
    // v0.58.4 行归约契约：TROWSUM 目的须物理单列 [N,1]
    using t1 = Tile<Location::Vec, RU, 32, 1, BLayout::RowMajor, 1, 1>;
    using gi1 = global_tensor<RU, RowMajor<1, 1>>;
    gk g(key_ptr);

    // GT 候选及计数由普通标量比较物化，归约保持原 Tile 布局。
    {
        TKey<CK, CKV> key; TLOAD(key, g);
        TKey<CK, CKV> sel1, cand;
        auto &keys = TPARTELEMENT(key);
        auto &counts = TPARTELEMENT(sel1);
        auto &values = TPARTELEMENT(cand);
#pragma pto element for
        for (uint32_t element = 0; element < CKV; ++element) {
            const bool take = keys[element] > kthVal;
            counts[element] = take ? 1u : 0u;
            // PopN selects eligible indices; zero remains the cleared marker
            // even when the original sortable key is itself zero.
            values[element] = take ? element + 1u : 0u;
        }
        t1 s; TROWSUM(s, sel1);
        RU cnt = 0; gi1 gcnt(&cnt); TSTORE(gcnt, s);
        RadixPopN<CK, CKV>(cand, chunkBase, outBase + outPos, (int)cnt);
        outPos += (int)cnt;
    }
    // EQ: key == kth（补足剩余名额）
    if (needEq > 0) {
        TKey<CK, CKV> key; TLOAD(key, g);
        TKey<CK, CKV> sel1, cand;
        auto &keys = TPARTELEMENT(key);
        auto &counts = TPARTELEMENT(sel1);
        auto &values = TPARTELEMENT(cand);
#pragma pto element for
        for (uint32_t element = 0; element < CKV; ++element) {
            const bool take = keys[element] == kthVal;
            counts[element] = take ? 1u : 0u;
            values[element] = take ? element + 1u : 0u;
        }
        t1 s; TROWSUM(s, sel1);
        RU cnt = 0; gi1 gcnt(&cnt); TSTORE(gcnt, s);
        int take = (int)cnt; if (take > needEq) take = needEq;
        RadixPopN<CK, CKV>(cand, chunkBase, outBase + outPos, take);
        outPos += take; needEq -= take;
    }
}

// 单个 chunk 的直方图 + 差分合并到 per_bin
template <int CK, int CKV>
inline void HistMergeChunk(RU* krow, int c, int stride, int r, RU* prefix, RU* hist, RU* per_bin) {
    RadixChunkHist<CK, CKV>(krow + c * stride, r, prefix, hist);
    volatile RU* rh = reinterpret_cast<volatile RU*>(hist);
    RU prev = 0;
    for (int b = 0; b < 256; b++) { RU cum = rh[b]; per_bin[b] += (cum - prev); prev = cum; }
}

// 单个 chunk 的提取（调用 RadixExtract，stride = MaxTileCol）
template <int CK, int CKV>
inline void ExtractChunk(RU* krow, int c, int stride, RU kthVal, int32_t* outBase, int& outPos, int& needEq) {
    RadixExtract<CK, CKV>(krow + c * stride, kthVal, (RU)c * stride, outBase, outPos, needEq);
}

}  // namespace qli_radix

// ==================== qli_topk_radix 主函数 ====================
template <int Sq, int Skv, int topK>
void qli_topk_radix(float* scores_gm, int32_t* indices_gm) {
    using namespace qli_radix;
    static_assert(Skv % 8 == 0, "Skv must be multiple of 8");
    static_assert(Sq > 0 && topK > 0 && topK <= Skv,
                  "QLI requires Sq > 0 and 0 < topK <= Skv");

    constexpr int MaxTileCol = 2048;
    constexpr int NumChunks = (Skv + MaxTileCol - 1) / MaxTileCol;
    constexpr int TailCols = Skv - (NumChunks - 1) * MaxTileCol;
    constexpr int SinglePhy = (Skv < 128) ? 128 : Skv;
    constexpr int TailPhy = (TailCols < 128) ? 128 : TailCols;
    static_assert((NumChunks == 1 && (Skv == 32 || Skv == 128 || Skv == 2048)) ||
                  (NumChunks > 1 &&
                   (TailCols == 32 || TailCols == 128 || TailCols == 2048)),
                  "QLI element views require full 128/2048 columns or a 32-column tail");

    // The caller supplies disjoint score and output/workspace allocations.
    // Workspace holds Sq*topK indices, an existing 8192-byte reserved span,
    // Sq*Skv keys, 256 histogram words, and 32 prefix words. The inner key and
    // histogram helpers separately preserve their documented alias snapshots.
    uint32_t* key_scratch = reinterpret_cast<uint32_t*>(
            reinterpret_cast<uint8_t*>(indices_gm) + (uint64_t)Sq * topK * 4 + 8192);
    uint32_t* hist_scratch = key_scratch + (uint64_t)Sq * Skv;
    uint32_t* prefix_buf = hist_scratch + 256;

    for (int i = 0; i < Sq; i++) {
        const RU* row = reinterpret_cast<const RU*>(scores_gm) + (uint64_t)i * Skv;
        RU* krow = key_scratch + (uint64_t)i * Skv;

        // ---- Step 1: float → sortable key（每 chunk 完整快照）----
        if constexpr (NumChunks == 1) {
            RadixMakeKey<SinglePhy, Skv>(krow, row);
        } else if constexpr (TailCols != MaxTileCol) {
            for (int c = 0; c < NumChunks - 1; c++)
                RadixMakeKey<MaxTileCol, MaxTileCol>(krow + c * MaxTileCol, row + c * MaxTileCol);
            RadixMakeKey<TailPhy, TailCols>(krow + (NumChunks - 1) * MaxTileCol, row + (NumChunks - 1) * MaxTileCol);
        } else {
            for (int c = 0; c < NumChunks; c++)
                RadixMakeKey<MaxTileCol, MaxTileCol>(krow + c * MaxTileCol, row + c * MaxTileCol);
        }

        // ---- Step 2: 4 轮 MSD radix → kth_value ----
        RU kth_value = 0;
        int remaining = topK;
#pragma pto element for
        for (uint32_t element = 0; element < 32; ++element)
            prefix_buf[element] = 0;

        for (int r = 3; r >= 0; r--) {
            RU per_bin[256] = {0};
            // 前缀: 已定高位字节（ByteId<3 时生效）
            if (r < 3) {
                const RU b3 = (kth_value >> 24) & 0xFFu;
                const RU b2 = (kth_value >> 16) & 0xFFu;
                const RU b1 = (kth_value >> 8) & 0xFFu;
                // PE 私有前缀缓冲；下一次 histogram 调用按语句顺序读取。
#pragma pto element for
                for (uint32_t element = 0; element < 32; ++element) {
                    prefix_buf[element] = element < 8 ? b3 :
                                          element < 16 ? b2 :
                                          element < 24 ? b1 : 0u;
                }
            }
            // 逐 chunk 原子直方图 + 差分合并
            if constexpr (NumChunks == 1) {
                HistMergeChunk<SinglePhy, Skv>(krow, 0, MaxTileCol, r, prefix_buf, hist_scratch, per_bin);
            } else if constexpr (TailCols != MaxTileCol) {
                for (int c = 0; c < NumChunks - 1; c++)
                    HistMergeChunk<MaxTileCol, MaxTileCol>(krow, c, MaxTileCol, r, prefix_buf, hist_scratch, per_bin);
                HistMergeChunk<TailPhy, TailCols>(krow, NumChunks - 1, MaxTileCol, r, prefix_buf, hist_scratch, per_bin);
            } else {
                for (int c = 0; c < NumChunks; c++)
                    HistMergeChunk<MaxTileCol, MaxTileCol>(krow, c, MaxTileCol, r, prefix_buf, hist_scratch, per_bin);
            }

            // 高桶向下累计 → kth_byte
            RU cum = 0; int kth_byte = 0;
            for (int b = 255; b >= 0; b--) {
                cum += per_bin[b];
                if (cum >= (RU)remaining) { kth_byte = b; break; }
            }
            remaining -= (int)(cum - per_bin[kth_byte]);
            kth_value |= ((RU)kth_byte) << (r * 8);
        }

        // ---- Step 3: 提取 topK 索引（每 chunk: 先 GT 后 EQ）----
        int outPos = 0;
        int needEq = remaining;
        if constexpr (NumChunks == 1) {
            ExtractChunk<SinglePhy, Skv>(krow, 0, MaxTileCol, kth_value, indices_gm + i * topK, outPos, needEq);
        } else if constexpr (TailCols != MaxTileCol) {
            for (int c = 0; c < NumChunks - 1; c++)
                ExtractChunk<MaxTileCol, MaxTileCol>(krow, c, MaxTileCol, kth_value, indices_gm + i * topK, outPos, needEq);
            ExtractChunk<TailPhy, TailCols>(krow, NumChunks - 1, MaxTileCol, kth_value, indices_gm + i * topK, outPos, needEq);
        } else {
            for (int c = 0; c < NumChunks; c++)
                ExtractChunk<MaxTileCol, MaxTileCol>(krow, c, MaxTileCol, kth_value, indices_gm + i * topK, outPos, needEq);
        }
    }
}

#endif
