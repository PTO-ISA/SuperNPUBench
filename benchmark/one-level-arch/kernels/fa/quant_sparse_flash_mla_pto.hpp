#ifndef QUANT_SPARSE_FLASH_MLA_PTO_HPP
#define QUANT_SPARSE_FLASH_MLA_PTO_HPP

// =============================================================================
// quant_sparse_flash_mla_pto.hpp
//   Quant Sparse Flash MLA (SWA mode) on PTO Tile-OP
//
// 【计算语义】
//   O = softmax(Q @ K^T * softmax_scale, mask=invalid) @ V
//   Q: [s1, D], KV: [s2, D] (MLA shared K=V=ori_kv), O: [s1, D]
//
// 【SWA 滑动窗口 — kernel 内部 token 级 mask, 使用 TSEL】
//   mask 为 UINT32 位打包格式 (每 32 列 1 个 uint32, bit=1 表示无效/被 mask)
//   TSEL(dst, mask, neg_inf): mask bit=1 → dst=-1e30 (无效), bit=0 → 保持原 score
//   窗口范围 (对第 q 个 Q token, 0-indexed):
//     diagonal = (s2 - s1) + q
//     valid kv: [diagonal - win_left, diagonal + win_right]  (闭区间)
//   mask 在 kernel 函数内部根据 win_left/win_right 计算, 放在 stack 上.
//   与原算子入参完全一致: win_left/win_right 为标量属性, 无额外 mask 入参.
//
// 【入参说明】
//   win_left / win_right : 滑动窗口参数, kernel 内部用于计算 mask
//   ori_sparse_indices   : SWA 模式下不使用 (为 nullptr), 保留入参签名
//   ori_block_table      : 非 PA 场景下不使用, 保留入参签名
//   其余可选入参均为 nullptr 占位, 保留签名方便后续扩展
//
// 【D=512 分块】
//   D 超出单 tile 上限, 沿 D 维切分为 Db 块 (kTd)
//   QK^T: 沿 D 累加 (TMATMUL + TMATMUL_ACC)
//   PV: 每个 D 分块独立计算并存储
//
// 【两遍式】
//   Pass 1: online softmax 归约 (m, l), 含 mask
//   Pass 2: 归一化 P, 计算 P@V, 含 mask
//
// 【mask tile 格式】
//   TSEL 的 mask 必须是 UINT32 位打包格式:
//     maskWordsPerRow = ceil(kTk / 32)
//     mask tile shape: [kTm, maskWordsPerRow], dtype=uint32_t, RowMajor
//     bit (1<<j) = 1 表示第 j 列被 mask (无效), 0 表示有效
//   TCMP 输出天然为此格式; 手动构建的 mask 也需按此格式打包
// =============================================================================

#include <common/pto_tileop.hpp>
#include "template_asm.h"

using namespace pto;

// CPU 侧 mask 预计算 (在 kernel 内部调用, 放在 stack 上)
// 生成 UINT32 位打包格式的 mask:
//   对 [s1, s2] 的每个 (q, kv), 如果 kv 在窗口外则对应 bit=1
//   每 32 个 kv 列打包成 1 个 uint32
//   maskBuf 行大小 = ceil(s2 / 32) 个 uint32
static inline void build_swa_mask_bitpacked(
    uint32_t* maskBuf, int s1, int s2, int win_left, int win_right)
{
    const int wordsPerRow = (s2 + 31) / 32;
    const int causal_offset = s2 - s1;
    for (int q = 0; q < s1; ++q) {
        int diagonal = causal_offset + q;
        int lo = diagonal - win_left;
        int hi = diagonal + win_right;
        uint32_t* row = maskBuf + q * wordsPerRow;
        for (int w = 0; w < wordsPerRow; ++w) row[w] = 0;
        for (int kv = 0; kv < s2; ++kv) {
            bool valid = (kv >= lo) && (kv <= hi);
            if (!valid) {
                row[kv / 32] |= (uint32_t{1} << (kv % 32));
            }
        }
    }
}

template <typename qdtype, typename kvdtype, typename odttype,
          int s1, int s2, int D, int kTm, int kTk, int kTd,
          int scaleD = D>
void quant_sparse_flash_mla_swa_pto(
    odttype* out_ptr,
    qdtype* q_ptr,
    kvdtype* ori_kv_ptr,
    float softmax_scale,
    int ori_win_left,
    int ori_win_right,
    float* q_descale,
    float* ori_kv_descale,
    int* ori_sparse_indices,
    int* ori_block_table,
    int* cu_seqlens_q,
    int* cu_seqlens_ori_kv,
    int* seqused_q,
    int* seqused_ori_kv,
    float* sinks,
    int* metadata,
    float* softmax_lse)
{
    constexpr int Db = D / kTd;

    // === mask 预计算 (UINT32 位打包) ===
    // 全局 mask: [s1, ceil(s2/32)] uint32
    constexpr int maskWordsPerRowGlobal = (s2 + 31) / 32;
    uint32_t maskBuf[s1 * maskWordsPerRowGlobal]; //[64 * 4]
    build_swa_mask_bitpacked(maskBuf, s1, s2, ori_win_left, ori_win_right);

    // 每个 [kTm, kTk] block 对应的 mask tile:
    //   maskWordsPerRowBlock = ceil(kTk / 32)
    //   mask tile: [kTm, maskWordsPerRowBlock], uint32, RowMajor
    constexpr int maskWordsPerRowBlock = (kTk + 31) / 32; // 1

    using gmQ    = global_tensor<qdtype,  RowMajor<s1, D>>;
    using gmKV   = global_tensor<kvdtype, RowMajor<s2, D>>;
    using gmO    = global_tensor<odttype, RowMajor<s1, D>>;

    using tileQ      = TileLeft<qdtype,  kTm, kTd>;
    using tileKV     = TileRight<kvdtype, kTk, kTd>;
    using tileW_out  = TileAcc<float, kTm, kTk>;

    // score tile: RowMajor float
    using tileW      = Tile<Location::Vec, float, kTm, kTk, BLayout::RowMajor>;

    // mask tile: RowMajor uint32, shape [kTm, maskWordsPerRowBlock]
    // TSEL 要求 dst/mask/src 同 tile_shape, 但模拟器内部 mask 按 uint32 位打包读
    // 这里用 float 类型满足编译器约束, 实际数据是 uint32 位掩码
    // tile Cols 设为 kTk (与 score tile 同形状), 模拟器只读前 maskWordsPerRowBlock 个 uint32
    using tileMask   = Tile<Location::Vec, float, kTm, kTk, BLayout::RowMajor>;

    using tileW_cast = Tile<Location::Vec, qdtype, kTm, kTk, BLayout::RowMajor>;
    using tileW_left = TileLeft<qdtype, kTm, kTk>;

    using tileO_out  = TileAcc<float, kTm, kTd>;
    using tileO      = Tile<Location::Vec, float, kTm, kTd, BLayout::RowMajor>;
    using tileO_cast = Tile<Location::Vec, odttype, kTm, kTd, BLayout::RowMajor>;

    using tileV      = TileRight<kvdtype, kTk, kTd>;
    using tileMax    = Tile<Location::Vec, float, kTm, 8, BLayout::RowMajor, kTm, 1>;
    using tileSum    = Tile<Location::Vec, float, kTm, 8, BLayout::RowMajor, kTm, 1>;

    using itQ    = global_iterator<gmQ,  tileQ>;
    using itKV   = global_iterator<gmKV, tileKV>;
    using itV    = global_iterator<gmKV, tileV>;
    using itO    = global_iterator<gmO,  tileO_cast>;

    itQ  gIterQ(q_ptr);
    itKV gIterKV(ori_kv_ptr);
    itV  gIterV(ori_kv_ptr);
    itO  gIterO(out_ptr);

    const int Qb = (s1 + kTm - 1) / kTm;
    const int Kb = (s2 + kTk - 1) / kTk;

    const float scale = softmax_scale;

    for (int i = 0; i < Qb; ++i) {

        // ============================================================
        //  Pass 1: online softmax 归约 row max (m) 与 row sum (l)
        //  遍历全部 KV 块, 用 mask 屏蔽窗口外 token
        // ============================================================
        tileMax tMax;  TEXPANDS(tMax, -1e30f);
        tileSum tSum;  TEXPANDS(tSum, 0.0f);

        // TSEL 的 true-value: -1e30 (mask 命中时取此值)
        tileW tNegInf;  TEXPANDS(tNegInf, -1e30f);

        for (int j = 0; j < Kb; ++j) {

            // QK^T 沿 D 维累加
            tileW_out tW_out;
            bool first_d = true;
            #pragma clang loop unroll(full)
            for (int dd = 0; dd < Db; ++dd) {
                tileQ tQ;
                auto gQ = gIterQ(i, dd);
                TLOAD(tQ, gQ);

                tileKV tK;
                auto gK = gIterKV(j, dd);
                TLOAD(tK, gK);

                if (first_d) {
                    TMATMUL(tW_out, tQ, tK);
                    first_d = false;
                } else {
                    TMATMUL_ACC(tW_out, tQ, tK);
                }
            }

            tileW tW;
            ACCCVT(tW, tW_out);
            TMULS(tW, tW, scale);

            // 应用 token 级 mask: TSEL(score, mask, neg_inf)
            // mask bit=1 → score=-1e30 (无效), bit=0 → 保持原 score
            // mask 数据从全局 maskBuf 中提取当前 [kTm, kTk] block 对应的位打包数据
            // 构建局部 mask tile: 从全局 [s1, maskWordsPerRowGlobal] 中提取
            //   [kTm 行, kTk 列] 对应的 bit, 重新打包为 [kTm, maskWordsPerRowBlock]
            {
                // 构建 block mask: [kTm, maskWordsPerRowBlock] uint32
                // 从全局 mask 中提取列 [j*kTk, (j+1)*kTk) 的 bits
                uint32_t blockMask[kTm * maskWordsPerRowBlock]; // [32 * 1]
                for (int r = 0; r < kTm; ++r) {
                    int q_idx = i * kTm + r;
                    const uint32_t* globalRow = maskBuf + q_idx * maskWordsPerRowGlobal;
                    uint32_t* blockRow = blockMask + r * maskWordsPerRowBlock;
                    for (int w = 0; w < maskWordsPerRowBlock; ++w) blockRow[w] = 0;
                    for (int c = 0; c < kTk; ++c) {
                        int global_col = j * kTk + c;
                        if (global_col >= s2) {
                            blockRow[c / 32] |= (uint32_t{1} << (c % 32));
                            continue;
                        }
                        bool masked = (globalRow[global_col / 32] >> (global_col % 32)) & 1;
                        if (masked) {
                            blockRow[c / 32] |= (uint32_t{1} << (c % 32));
                        }
                    }
                }
                // TLOAD mask tile from blockMask (stack buffer)
                using gmBlockMask = global_tensor<uint32_t, RowMajor<kTm, kTk>>;
                gmBlockMask gBlockMask(blockMask);
                tileMask tMask;
                TLOAD(tMask, gBlockMask);
                TSEL(tW, tMask, tNegInf);
            }

            // m_new = max(m_old, rowmax(score))
            tileMax tLocalMax;
            TROWMAX(tLocalMax, tW);
            tileMax tNewMax;
            TMAX(tNewMax, tMax, tLocalMax);

            // rescale = exp(m_old - m_new); l_old' = l_old * rescale
            tileMax tScale;
            TSUB(tScale, tMax, tNewMax);
            TEXP(tScale, tScale);
            tileSum tScaledOldSum;
            TMUL(tScaledOldSum, tSum, tScale);

            // local_sum = rowsum(exp(score - m_new))
            TROWEXPANDSUB(tW, tW, tNewMax);
            TEXP(tW, tW);
            tileSum tLocalSum;
            TROWSUM(tLocalSum, tW);

            // l_new = l_old' + local_sum
            tileSum tNewSum;
            TADD(tNewSum, tScaledOldSum, tLocalSum);

            tMax = tNewMax;
            tSum = tNewSum;
        }

        // ============================================================
        //  Pass 2: p = exp(QK*scale, mask - m) / l, O = Σ p·V
        //  QK^T 用全 D 累加, PV 按 D 分块计算
        // ============================================================
        tileSum tInvSum;
        TRECIP(tInvSum, tSum);

        for (int dd = 0; dd < Db; ++dd) {
            tileO tO;
            TEXPANDS(tO, 0.0f);

            for (int j = 0; j < Kb; ++j) {

                // 计算完整 QK^T (沿 D 累加, 与 Pass 1 一致)
                tileW_out tW_out;
                bool first_d = true;
                #pragma clang loop unroll(full)
                for (int dd2 = 0; dd2 < Db; ++dd2) {
                    tileQ tQ;
                    auto gQ = gIterQ(i, dd2);
                    TLOAD(tQ, gQ);

                    tileKV tK;
                    auto gK = gIterKV(j, dd2);
                    TLOAD(tK, gK);

                    if (first_d) {
                        TMATMUL(tW_out, tQ, tK);
                        first_d = false;
                    } else {
                        TMATMUL_ACC(tW_out, tQ, tK);
                    }
                }

                tileW tW;
                ACCCVT(tW, tW_out);
                TMULS(tW, tW, scale);

                // 应用 token 级 mask: TSEL(score, mask, neg_inf)
                {
                    uint32_t blockMask[kTm * maskWordsPerRowBlock];
                    for (int r = 0; r < kTm; ++r) {
                        int q_idx = i * kTm + r;
                        const uint32_t* globalRow = maskBuf + q_idx * maskWordsPerRowGlobal;
                        uint32_t* blockRow = blockMask + r * maskWordsPerRowBlock;
                        for (int w = 0; w < maskWordsPerRowBlock; ++w) blockRow[w] = 0;
                        for (int c = 0; c < kTk; ++c) {
                            int global_col = j * kTk + c;
                            if (global_col >= s2) {
                                blockRow[c / 32] |= (uint32_t{1} << (c % 32));
                                continue;
                            }
                            bool masked = (globalRow[global_col / 32] >> (global_col % 32)) & 1;
                            if (masked) {
                                blockRow[c / 32] |= (uint32_t{1} << (c % 32));
                            }
                        }
                    }
                    using gmBlockMask = global_tensor<uint32_t, RowMajor<kTm, kTk>>;
                    gmBlockMask gBlockMask(blockMask);
                    tileMask tMask;
                    TLOAD(tMask, gBlockMask);
                    TSEL(tW, tMask, tNegInf);
                }

                // p = exp(score - m) / l
                TROWEXPANDSUB(tW, tW, tMax);
                TEXP(tW, tW);
                TROWEXPANDMUL(tW, tW, tInvSum);

                // cast p -> qdtype Left tile for TMATMUL
                tileW_cast tExpW;
                TCVT(tExpW, tW);
                tileW_left tW_left;
                TCVT(tW_left, tExpW);

                // PV = p * V (当前 D 分块)
                tileV tV;
                auto gV = gIterV(j, dd);
                TLOAD(tV, gV);

                tileO_out tPV_out;
                TMATMUL(tPV_out, tW_left, tV);
                tileO tPV;
                ACCCVT(tPV, tPV_out);

                TADD(tO, tO, tPV);
            }

            // 写回 O 分块 [kTm, kTd]
            tileO_cast tO_cast;
            TCVT(tO_cast, tO);
            auto gO = gIterO(i, dd);
            TSTORE(gO, tO_cast);
        }
    }
}

#endif
