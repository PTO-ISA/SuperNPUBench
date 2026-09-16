#ifndef QUANT_SPARSE_FLASH_MLA_CSA_1PE_HPP
#define QUANT_SPARSE_FLASH_MLA_CSA_1PE_HPP

// =============================================================================
// quant_sparse_flash_mla_csa_1pe.hpp
//   Quant Sparse Flash MLA（CSA 模式）—— 单 PE Local CUBE 实现
//
// 【定位】参考 4-PE 统一实现（quant_sparse_flash_mla.hpp）的 CSA 双源语义，
//   跑在单 PE Local CUBE 路径（机制来自已验证的 quant_sparse_flash_mla_tadd.hpp）。
//   用于单核 CSA 的精度/性能对照（vs 4-PE csa_tadd_4pe）。
//   dtype 双态：FP16（descale=1，P 不量化）或 HIF8（与官方算子/4-PE 对齐：
//   per-tensor descale、P×16 量化 HIF8、PV×kvDescale、输出×1/16、BF16 输出）。
//
// 【CSA 语义】（与 4-PE 逐项对齐）
//   O = softmax(Q·K^T × softmax_scale + mask) · V，双逻辑源拼接：
//   - ORI 源：窗口 [diag-wl, diag+wr+1) ∩ [0, S2) 的连续 token；
//     块对齐裁剪（qsmla_swa_block_range）+ 窗口 mask（对齐边缘屏蔽）
//   - CMP 源：cmp_valid_end = clamp(⌊(CmpS2·ratio−S1+q+1)/ratio⌋, 0, CmpS2)；
//     candidate = cmp_topk_length（缺省 CmpTopK）clamp 到 [0, CmpTopK]，
//     collect 剔除越界/超 valid_end 索引（-1 终止），按逻辑序 gather
//   - 双源顺序：先 ORI 后 CMP；两遍式各自按相同顺序重访
//   - score 缩放：softmax_scale × q_descale × kv_descale（FP16 下三者中
//     descale 均为 1；HIF8 下 per-tensor 反量化，ORI/CMP 各自 kv_descale）
//   - HIF8 量化契约（对齐官方算子）：P×16 → HIF8 → PV × kv_descale →
//     输出 ×1/16（kHif8ProbabilityScale=16 与 hifp8ScaleValue 一致）
//
// 【单 PE 机制】（继承 tadd.hpp 的已验证方案）
//   - TTRANS 已退役 → ORI 全序列一次性预转置 ori_t[D][S2]；CMP 一次性
//     gather（行序 cmp_rows_buf[CmpTopK][D]，未选中行补零防 NaN 污染）
//     + 转置（cmp_t[D][CmpTopK]）
//     HIF8 下 gather 用 uint32 carrier（LinxV5 无法 legalize 分发散标量
//     i8/HIF8 拷贝，4 字节保留原始 payload，D%4==0）
//   - 方阵 kTk == kTd：未打补丁 TileOP 的 Local-B 双约定仅在方阵上一致
//   - 行归约源 tW [kTm, kTk] FP32 ≤ 2048B（pto-spec 0.58.6 契约）
//   - 所有标量循环（转置 / gather / mask 构造）先于一切 tile 指令执行
//     （规避编译器 off-spec spill TMOV）
//   - 平铺四段（pass1/pass2 × ORI/CMP），不用 lambda：泛型 lambda 按引用
//     捕获跨源调用的 tile 变量会触发编译器 raw TSTORE 栈 spill（模型
//     RecordRawTileTransport 断言，RawTileSourceFits 不匹配）
//   - mask 行无关（同一 q_token 的全部 head 共享窗口/索引）→ 每块预构造
//     [kTm, kTk] mask（行复制），tile 段保持无分支的固定指令序列
//   - Acc↔Vec 经 GM scratch（TSTORE_CUBE/TLOAD，Acc 不能直读）
// =============================================================================

#include <common/pto_tileop.hpp>
#include "template_asm.h"
#include "qsmla_config.hpp"
#include "qsmla_mode.hpp"
#include <type_traits>

using namespace pto;

template <typename qdtype, typename kvdtype, typename odttype,
          typename Config, typename ModeConfig>
void quant_sparse_flash_mla_csa_1pe_pto(
    odttype* out_ptr,
    qdtype* q_ptr,
    kvdtype* ori_kv_ptr,
    kvdtype* cmp_kv_ptr,
    const int* cmp_sparse_indices,
    const int* cmp_topk_length,
    float softmax_scale,
    float q_descale,
    float ori_kv_descale,
    float cmp_kv_descale,
    int cmp_ratio,
    int ori_win_left,
    int ori_win_right)
{
    // ---- 编译期常量 ----
    constexpr int kB = Config::B;
    constexpr int s1 = Config::S1;
    constexpr int s2 = Config::S2;             // ORI S2
    constexpr int N1 = Config::N1;
    constexpr int D = Config::D;
    constexpr int kTm = Config::TileM;
    constexpr int kTk = Config::TileK;
    constexpr int kTd = Config::TileD;
    constexpr int Db = D / kTd;
    constexpr int kCmpS2 = ModeConfig::CmpS2;
    constexpr int kCmpTopK = ModeConfig::CmpTopK;
    // 转置/行缓冲的物理容量：向上对齐到 kTk 的倍数，保证 iterator
    // 分块（列宽 kTk）永不越过分配的行距边界。
    constexpr int kCmpCap =
        (kCmpTopK + kTk - 1) / kTk * kTk;
    constexpr int kOriBlkMax = (s2 + kTk - 1) / kTk;
    constexpr int kCmpBlkMax = (kCmpTopK + kTk - 1) / kTk;
    constexpr int kMBlockCount = N1 / kTm;     // M 维 = head（每 q_token）

    // ---- 编译期约束 ----
    static_assert((std::is_same_v<qdtype, __half> &&
                   std::is_same_v<kvdtype, __half> &&
                   std::is_same_v<odttype, __half>) ||
                      (std::is_same_v<qdtype, __hif8> &&
                       std::is_same_v<kvdtype, __hif8> &&
                       std::is_same_v<odttype, __bf16>),
                  "single-PE CSA supports FP16 or HIF8 dtype sets");
    static_assert(!std::is_same_v<kvdtype, __hif8> || D % 4 == 0,
                  "HIF8 scalar loops use four-byte carriers");
    static_assert(Config::N2 == 1,
                  "single-PE CSA requires contiguous N2=1 KV");
    static_assert(kTk == kTd,
                  "unpatched-headers single-PE requires square B tiles "
                  "(Tk == Td_block)");
    static_assert(kTm <= 32, "single-PE Local CUBE supports TileM <= 32");
    static_assert(kTm * kTk * 4 <= 2048,
                  "row-reduction source tW must be <= 2048 bytes "
                  "(pto-spec 0.58.6)");
    static_assert(D % kTd == 0, "D-tail unsupported");
    static_assert(N1 % kTm == 0, "N1 must be a multiple of TileM");
    static_assert(kCmpTopK > 0, "CSA requires positive CmpTopK");

    // HIF8 量化契约（与 4-PE kUseHif8Probability / 官方 hifp8ScaleValue 对齐）
    constexpr bool kUseHif8Probability = std::is_same_v<qdtype, __hif8>;
    constexpr float kHif8ProbabilityScale = 16.0f;
    // score 反量化缩放（FP16 下 descale 均为 1，数值不变）
    const float ori_score_scale = softmax_scale * q_descale * ori_kv_descale;
    const float cmp_score_scale = softmax_scale * q_descale * cmp_kv_descale;

    // ---- Tile 类型（Tm=32, Tk=Td=16 全部合规）----
    using tileQ = CubeTileM32<qdtype, kTm, kTd>;            // A: Q [Tm, Td]
    using tileKRight = CubeTileN8<kvdtype, kTd, kTk>;       // B: K^T [Td, Tk]
    using tileV = CubeTileN8<kvdtype, kTk, kTd>;            // B: V   [Tk, Td]
    using tileWLeft = CubeTileM32<qdtype, kTm, kTk>;        // A: P   [Tm, Tk]
    using tileScoreCube = CubeAccumulatorM32<float, kTm, kTk>;
    using tilePVCube = CubeAccumulatorM32<float, kTm, kTd>;
    using tileW = Tile<Location::Vec, float, kTm, kTk, BLayout::RowMajor>;
    using tileMask = tileW;
    using tileP = Tile<Location::Vec, qdtype, kTm, kTk, BLayout::RowMajor>;
    using tileO = Tile<Location::Vec, float, kTm, kTd, BLayout::RowMajor>;
    using tileOCast =
        Tile<Location::Vec, odttype, kTm, kTd, BLayout::RowMajor>;
    using tileRowState =
        Tile<Location::Vec, float, kTm, 1, BLayout::RowMajor, kTm, 1>;

    // ---- GM 视图与迭代器 ----
    using gmQO = global_tensor<qdtype, RowMajor<N1, D>>;
    using gmO = global_tensor<odttype, RowMajor<N1, D>>;
    using itQ = global_iterator<gmQO, tileQ>;
    using itO = global_iterator<gmO, tileOCast>;
    using gmOriT = global_tensor<kvdtype, RowMajor<D, s2>>;
    using itKtOri = global_iterator<gmOriT, tileKRight>;
    using gmKV = global_tensor<kvdtype, RowMajor<s2, D>>;
    using itVOri = global_iterator<gmKV, tileV>;
    using gmCmpT = global_tensor<kvdtype, RowMajor<D, kCmpCap>>;
    using itKtCmp = global_iterator<gmCmpT, tileKRight>;
    using gmCmpRows = global_tensor<kvdtype, RowMajor<kCmpCap, D>>;
    using itVCmp = global_iterator<gmCmpRows, tileV>;
    using gmMask = global_tensor<float, RowMajor<kTm, kTk>>;
    using gmScore = global_tensor<float, RowMajor<kTm, kTk>>;
    using gmProb = global_tensor<qdtype, RowMajor<kTm, kTk>>;
    using gmPV = global_tensor<float, RowMajor<kTm, kTd>>;

    // ---- 栈 scratch ----
    alignas(64) kvdtype ori_t[D * s2];                       // 128KB (FP16)
    alignas(64) kvdtype cmp_rows_buf[kCmpCap * D];
    alignas(64) kvdtype cmp_t[D * kCmpCap];
    alignas(64) float ori_masks[kOriBlkMax][kTm * kTk];      // 16KB
    alignas(64) float cmp_masks[kCmpBlkMax][kTm * kTk];      // 6KB
    alignas(64) float score_scratch[kTm * kTk];
    alignas(64) qdtype prob_scratch[kTm * kTk];
    alignas(64) float pv_scratch[kTm * kTd];
    int cmp_selected[kCmpTopK];

    gmScore gScore(score_scratch);
    gmProb gProb(prob_scratch);
    gmPV gPV(pv_scratch);

    // ================= batch 循环 =================
    for (int b = 0; b < kB; ++b) {
        kvdtype* batch_ori =
            ori_kv_ptr + static_cast<std::size_t>(b) * s2 * D;
        kvdtype* batch_cmp =
            cmp_kv_ptr + static_cast<std::size_t>(b) * kCmpS2 * D;
        qdtype* batch_q =
            q_ptr + static_cast<std::size_t>(b) * s1 * N1 * D;
        odttype* batch_out =
            out_ptr + static_cast<std::size_t>(b) * s1 * N1 * D;

        // ---- Phase S（纯标量，先于一切 tile）：ORI 全序列预转置 ----
        for (int t = 0; t < s2; ++t)
            for (int d = 0; d < D; ++d)
                ori_t[d * s2 + t] = batch_ori[t * D + d];

        // ================= q_token 循环 =================
        for (int q_token = 0; q_token < s1; ++q_token) {
            // ---- Phase T（纯标量）：CMP 索引收集 / gather / 转置 / mask ----
            const int cmp_valid_end = qsmla_csa_cmp_valid_end(
                kCmpS2, s1, q_token, cmp_ratio);
            int cmp_count = 0;
            {
                int candidate_count = kCmpTopK;
                if (cmp_topk_length != nullptr) {
                    candidate_count =
                        cmp_topk_length[static_cast<std::size_t>(b) * s1 +
                                        q_token];
                }
                candidate_count = qsmla_sparse_clamp(
                    candidate_count, 0, kCmpTopK);
                cmp_count = qsmla_sparse_collect_indices(
                    cmp_selected, kCmpTopK,
                    cmp_sparse_indices +
                        static_cast<std::size_t>(
                            (b * s1 + q_token)) * kCmpTopK,
                    candidate_count, kCmpS2, cmp_valid_end);
            }
            // CMP 行序 gather（逻辑序 i；未选中行补零防垃圾 NaN 污染）。
            // HIF8：LinxV5 无法 legalize 分发散标量 i8/HIF8 拷贝，
            // 按 uint32 carrier 逐字搬运保留原始 payload（D%4==0）。
            if constexpr (std::is_same_v<kvdtype, __hif8>) {
                auto* rows_words =
                    reinterpret_cast<uint32_t*>(cmp_rows_buf);
                for (int i = 0; i < kCmpCap; ++i) {
                    const int row =
                        i < cmp_count ? cmp_selected[i] : 0;
                    const bool valid = i < cmp_count;
                    const auto* source_words =
                        reinterpret_cast<const uint32_t*>(
                            batch_cmp +
                            static_cast<std::size_t>(row) * D);
                    for (int w = 0; w < D / 4; ++w) {
                        rows_words[static_cast<std::size_t>(i) * (D / 4) +
                                   w] =
                            valid ? source_words[w] : 0U;
                    }
                }
            } else {
                for (int i = 0; i < kCmpCap; ++i) {
                    const int row = i < cmp_count ? cmp_selected[i] : 0;
                    const bool valid = i < cmp_count;
                    for (int d = 0; d < D; ++d) {
                        cmp_rows_buf[i * D + d] =
                            valid
                                ? batch_cmp[static_cast<std::size_t>(row) *
                                                D +
                                            d]
                                : static_cast<kvdtype>(0.0f);
                    }
                }
            }
            // CMP 转置 [D, CmpTopK]（列 = 逻辑序，块对齐）
            for (int d = 0; d < D; ++d)
                for (int i = 0; i < kCmpCap; ++i)
                    cmp_t[d * kCmpCap + i] = cmp_rows_buf[i * D + d];


            // ORI 窗口与块范围（块对齐裁剪 + 边缘 mask，tadd 语义）
            const QsmlaSwaRange ori_range = qsmla_swa_range(
                s2, s1, q_token, ori_win_left, ori_win_right);
            const QsmlaSwaRange ori_blocks =
                qsmla_swa_block_range(ori_range, kTk);
            const int ori_blk_begin = ori_blocks.begin;
            const int ori_blk_count = ori_blocks.end - ori_blocks.begin;
            const int cmp_blk_count = (cmp_count + kTk - 1) / kTk;

            // mask 构造（行无关 → 每行同值复制 kTm 份）
            for (int j = 0; j < ori_blk_count; ++j) {
                const int token_base = (ori_blk_begin + j) * kTk;
                for (int r = 0; r < kTm; ++r) {
                    for (int c = 0; c < kTk; ++c) {
                        const int token = token_base + c;
                        ori_masks[j][r * kTk + c] =
                            (token >= ori_range.begin &&
                             token < ori_range.end)
                                ? 0.0f
                                : -1.0e30f;
                    }
                }
            }
            for (int j = 0; j < cmp_blk_count; ++j) {
                const int valid_cols = cmp_count - j * kTk;
                for (int r = 0; r < kTm; ++r) {
                    for (int c = 0; c < kTk; ++c) {
                        cmp_masks[j][r * kTk + c] =
                            c < valid_cols ? 0.0f : -1.0e30f;
                    }
                }
            }

            // Q / O 迭代器（本 q_token 的 [N1, D] 视图）
            itQ gIterQ(batch_q +
                       static_cast<std::size_t>(q_token) * N1 * D);
            itO gIterO(batch_out +
                       static_cast<std::size_t>(q_token) * N1 * D);
            itKtOri itOriT(ori_t);
            itVOri itVOriGm(batch_ori);
            itKtCmp itCmpT(cmp_t);
            itVCmp itVCmpRows(cmp_rows_buf);

            // ================= m_block（head 块）循环 =================
            for (int m_block = 0; m_block < kMBlockCount; ++m_block) {

                // ============ Pass 1：online softmax 归约 (m, l) ============
                tileRowState tMax;
                tileRowState tSum;
                TEXPANDS(tMax, -1e30f);
                TEXPANDS(tSum, 0.0f);

                // ---- Pass 1 / ORI 源 ----
                for (int j = 0; j < ori_blk_count; ++j) {
                    tileW tW;
                    TEXPANDS(tW, 0.0f);
#pragma clang loop unroll(full)
                    for (int dd = 0; dd < Db; ++dd) {
                        tileQ tQ;
                        auto gQ = gIterQ(m_block, dd);
                        TLOAD_CUBE(tQ, gQ);
                        tileKRight tK;
                        auto gK = itOriT(dd, ori_blk_begin + j);
                        TLOAD_CUBE(tK, gK);
                        tileScoreCube tScoreCube;
                        TMATMUL(tScoreCube, tQ, tK);
                        TSTORE_CUBE(gScore, tScoreCube);
                        tileW tWPartial;
                        TLOAD(tWPartial, gScore);
                        TADD(tW, tW, tWPartial);
                    }
                    TMULS(tW, tW, ori_score_scale);
                    tileMask tMask;
                    gmMask gMaskBuf(ori_masks[j]);
                    auto gMask = gMaskBuf;
                    TLOAD(tMask, gMask);
                    TADD(tW, tW, tMask);
                    tileRowState tLocalMax;
                    tileRowState tNewMax;
                    TROWMAX(tLocalMax, tW);
                    TMAX(tNewMax, tMax, tLocalMax);
                    tileRowState tScale;
                    TSUB(tScale, tMax, tNewMax);
                    TEXP(tScale, tScale);
                    tileRowState tScaledOldSum;
                    TMUL(tScaledOldSum, tSum, tScale);
                    TROWEXPANDSUB(tW, tW, tNewMax);
                    TEXP(tW, tW);
                    tileRowState tLocalSum;
                    TROWSUM(tLocalSum, tW);
                    TADD(tSum, tScaledOldSum, tLocalSum);
                    tMax = tNewMax;
                }

                // ---- Pass 1 / CMP 源 ----
                for (int j = 0; j < cmp_blk_count; ++j) {
                    tileW tW;
                    TEXPANDS(tW, 0.0f);
#pragma clang loop unroll(full)
                    for (int dd = 0; dd < Db; ++dd) {
                        tileQ tQ;
                        auto gQ = gIterQ(m_block, dd);
                        TLOAD_CUBE(tQ, gQ);
                        tileKRight tK;
                        auto gK = itCmpT(dd, j);
                        TLOAD_CUBE(tK, gK);
                        tileScoreCube tScoreCube;
                        TMATMUL(tScoreCube, tQ, tK);
                        TSTORE_CUBE(gScore, tScoreCube);
                        tileW tWPartial;
                        TLOAD(tWPartial, gScore);
                        TADD(tW, tW, tWPartial);
                    }
                    TMULS(tW, tW, cmp_score_scale);
                    tileMask tMask;
                    gmMask gMaskBuf(cmp_masks[j]);
                    auto gMask = gMaskBuf;
                    TLOAD(tMask, gMask);
                    TADD(tW, tW, tMask);
                    tileRowState tLocalMax;
                    tileRowState tNewMax;
                    TROWMAX(tLocalMax, tW);
                    TMAX(tNewMax, tMax, tLocalMax);
                    tileRowState tScale;
                    TSUB(tScale, tMax, tNewMax);
                    TEXP(tScale, tScale);
                    tileRowState tScaledOldSum;
                    TMUL(tScaledOldSum, tSum, tScale);
                    TROWEXPANDSUB(tW, tW, tNewMax);
                    TEXP(tW, tW);
                    tileRowState tLocalSum;
                    TROWSUM(tLocalSum, tW);
                    TADD(tSum, tScaledOldSum, tLocalSum);
                    tMax = tNewMax;
                }

                // ============ Pass 2：P 计算与 O = Σ P·V 累加 ============
                tileRowState tInvSum;
                TRECIP(tInvSum, tSum);

                // dd 外层（tO 寿命单 D 块，tadd 结构）
                for (int dd = 0; dd < Db; ++dd) {
                    tileO tO;
                    TEXPANDS(tO, 0.0f);

                    // ---- Pass 2 / ORI 源 ----
                    for (int j = 0; j < ori_blk_count; ++j) {
                        tileW tW;
                        TEXPANDS(tW, 0.0f);
#pragma clang loop unroll(full)
                        for (int dd2 = 0; dd2 < Db; ++dd2) {
                            tileQ tQ;
                            auto gQ = gIterQ(m_block, dd2);
                            TLOAD_CUBE(tQ, gQ);
                            tileKRight tK;
                            auto gK = itOriT(dd2, ori_blk_begin + j);
                            TLOAD_CUBE(tK, gK);
                            tileScoreCube tScoreCube;
                            TMATMUL(tScoreCube, tQ, tK);
                            TSTORE_CUBE(gScore, tScoreCube);
                            tileW tWPartial;
                            TLOAD(tWPartial, gScore);
                            TADD(tW, tW, tWPartial);
                        }
                        TMULS(tW, tW, ori_score_scale);
                        tileMask tMask;
                        gmMask gMaskBuf(ori_masks[j]);
                        auto gMask = gMaskBuf;
                        TLOAD(tMask, gMask);
                        TADD(tW, tW, tMask);
                        TROWEXPANDSUB(tW, tW, tMax);
                        TEXP(tW, tW);
                        TROWEXPANDMUL(tW, tW, tInvSum);
                        if constexpr (kUseHif8Probability) {
                            // 官方 QSMLA 契约：P×16 量化 HIF8 参与 BMM2，
                            // 输出统一 ×1/16 还原；online l 保持未量化
                            TMULS(tW, tW, kHif8ProbabilityScale);
                        }
                        tileP tP;
                        TCVT(tP, tW);
                        TSTORE(gProb, tP);
                        tileWLeft tWLeft;
                        TLOAD_CUBE(tWLeft, gProb);
                        tileV tV;
                        auto gV = itVOriGm(ori_blk_begin + j, dd);
                        TLOAD_CUBE(tV, gV);
                        tilePVCube tPVCube;
                        TMATMUL(tPVCube, tWLeft, tV);
                        TSTORE_CUBE(gPV, tPVCube);
                        tileO tPV;
                        TLOAD(tPV, gPV);
                        if constexpr (kUseHif8Probability) {
                            TMULS(tPV, tPV, ori_kv_descale);
                        }
                        TADD(tO, tO, tPV);
                    }

                    // ---- Pass 2 / CMP 源 ----
                    for (int j = 0; j < cmp_blk_count; ++j) {
                        tileW tW;
                        TEXPANDS(tW, 0.0f);
#pragma clang loop unroll(full)
                        for (int dd2 = 0; dd2 < Db; ++dd2) {
                            tileQ tQ;
                            auto gQ = gIterQ(m_block, dd2);
                            TLOAD_CUBE(tQ, gQ);
                            tileKRight tK;
                            auto gK = itCmpT(dd2, j);
                            TLOAD_CUBE(tK, gK);
                            tileScoreCube tScoreCube;
                            TMATMUL(tScoreCube, tQ, tK);
                            TSTORE_CUBE(gScore, tScoreCube);
                            tileW tWPartial;
                            TLOAD(tWPartial, gScore);
                            TADD(tW, tW, tWPartial);
                        }
                        TMULS(tW, tW, cmp_score_scale);
                        tileMask tMask;
                        gmMask gMaskBuf(cmp_masks[j]);
                        auto gMask = gMaskBuf;
                        TLOAD(tMask, gMask);
                        TADD(tW, tW, tMask);
                        TROWEXPANDSUB(tW, tW, tMax);
                        TEXP(tW, tW);
                        TROWEXPANDMUL(tW, tW, tInvSum);
                        if constexpr (kUseHif8Probability) {
                            TMULS(tW, tW, kHif8ProbabilityScale);
                        }
                        tileP tP;
                        TCVT(tP, tW);
                        TSTORE(gProb, tP);
                        tileWLeft tWLeft;
                        TLOAD_CUBE(tWLeft, gProb);
                        tileV tV;
                        auto gV = itVCmpRows(j, dd);
                        TLOAD_CUBE(tV, gV);
                        tilePVCube tPVCube;
                        TMATMUL(tPVCube, tWLeft, tV);
                        TSTORE_CUBE(gPV, tPVCube);
                        tileO tPV;
                        TLOAD(tPV, gPV);
                        if constexpr (kUseHif8Probability) {
                            TMULS(tPV, tPV, cmp_kv_descale);
                        }
                        TADD(tO, tO, tPV);
                    }

                    // 写回本 D 块输出（HIF8：还原 P 的 ×16 量化系数）
                    if constexpr (kUseHif8Probability) {
                        TMULS(tO, tO, 1.0f / kHif8ProbabilityScale);
                    }
                    tileOCast tOCast;
                    TCVT(tOCast, tO);
                    auto gO = gIterO(m_block, dd);
                    TSTORE(gO, tOCast);
                }
            }
        }
    }
}

#endif // QUANT_SPARSE_FLASH_MLA_CSA_1PE_HPP
