#ifndef QUANT_SPARSE_FLASH_MLA_ONEPASS_PTO_HPP
#define QUANT_SPARSE_FLASH_MLA_ONEPASS_PTO_HPP

// =============================================================================
// quant_sparse_flash_mla_onepass_pto.hpp
//   Quant Sparse Flash MLA (SWA mode) — One-pass (fused) variant
//
//   一遍式 (fused online softmax) 实现, 将归约 (m,l) 和 P@V 合并为单遍.
//   QK^T 只算 1 次 (两遍式算 2 次).
//
// 【D=512 支持】
//   QK^T 沿 D 累加得到 score [kTm, kTk] (不依赖 D 分块)
//   PV 按 D 分块: O[Db] 数组, 每个 O[dd] = [kTm, kTd]
//   每个 j 迭代中: 算完 score → rescale 所有 O[dd] → 对每个 dd 做 P@V
//   同时存活 tile: O[0..Db-1] + score/mask/m/l/scale/V/P ≈ Db+10
//   Db=8 时峰值约 18 个 tile, 每个 O tile 8KB, 总 64KB
//
// 【与两遍式的差异】
//   - QK^T 只算 1 次, 计算量减半
//   - online softmax 中同时 rescale 旧 O 并累加新 PV
//   - tile 寄存器压力更大 (O[Db] 数组跨 j 循环存活)
//
// 【mask 方式】
//   TADD: mask=float[s1*s2], 0.0(有效)/-1e30(无效), score += mask
//
// 【切换方法】
//   test 文件中:
//     #include "fa/quant_sparse_flash_mla_onepass_pto.hpp"
//     quant_sparse_flash_mla_swa_onepass_pto<...>(...)
// =============================================================================

#include <common/pto_tileop.hpp>
#include "template_asm.h"

using namespace pto;

static inline void build_swa_mask_onepass(
    float* mask, int s1, int s2, int win_left, int win_right)
{
    const int causal_offset = s2 - s1;
    for (int q = 0; q < s1; ++q) {
        int diagonal = causal_offset + q;
        int lo = diagonal - win_left;
        int hi = diagonal + win_right;
        for (int kv = 0; kv < s2; ++kv) {
            bool valid = (kv >= lo) && (kv <= hi);
            mask[q * s2 + kv] = valid ? 0.0f : -1e30f;
        }
    }
}

template <typename qdtype, typename kvdtype, typename odttype,
          int s1, int s2, int D, int kTm, int kTk, int kTd,
          int scaleD = D>
void quant_sparse_flash_mla_swa_onepass_pto(
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

    float mask_buf[s1 * s2];
    build_swa_mask_onepass(mask_buf, s1, s2, ori_win_left, ori_win_right);

    using gmQ    = global_tensor<qdtype,  RowMajor<s1, D>>;
    using gmKV   = global_tensor<kvdtype, RowMajor<s2, D>>;
    // Same storage as gmKV, viewed as K^T.  RowMajor<s2, D> and
    // ColMajor<D, s2> have the same address formula, so this view performs
    // the logical transpose without moving data.  TCOPYIN below then
    // performs the physical DN -> ZN conversion required by Cube SrcR.
    using gmKT   = global_tensor<kvdtype, ColMajor<D, s2>>;
    using gmO    = global_tensor<odttype, RowMajor<s1, D>>;
    using gmMask = global_tensor<float,   RowMajor<s1, s2>>;

    using tileQ      = TileLeft<qdtype, kTm, kTd>;
    using tileKRight = TileRight<kvdtype, kTd, kTk>;
    using tileW_out  = TileAcc<float, kTm, kTk>;
    using tileW      = Tile<Location::Vec, float, kTm, kTk, BLayout::RowMajor>;
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
    using itK    = global_iterator<gmKT, tileKRight>;
    using itV    = global_iterator<gmKV, tileV>;
    using itO    = global_iterator<gmO,  tileO_cast>;
    using itMask = global_iterator<gmMask, tileMask>;

    itQ    gIterQ(q_ptr);
    itK    gIterK(ori_kv_ptr);
    itV    gIterV(ori_kv_ptr);
    itO    gIterO(out_ptr);
    itMask gIterMask(mask_buf);

    const int Qb = (s1 + kTm - 1) / kTm;
    const int Kb = (s2 + kTk - 1) / kTk;

    const float scale = softmax_scale;

    for (int i = 0; i < Qb; ++i) {

        // ============================================================
        //  一遍式 fused online softmax + PV
        //
        //  O[Db] 数组跨 j 循环存活:
        //    O[0..Db-1] 每个 [kTm, kTd] float = 8KB, 共 Db*8KB
        //    Db=8 → 64KB (128 tile tags × 32KB = 4MB, 硬件足够)
        //
        //  每个 j 迭代:
        //    1. QK^T 沿全 D 累加 → score [kTm, kTk]
        //    2. score *= scale + mask
        //    3. online softmax: m_new, scale_old, l_new
        //    4. rescale 所有 O[dd] *= scale_old
        //    5. p = exp(score - m_new)
        //    6. 对每个 dd: PV = p @ V[dd], O[dd] += PV
        //
        //  最终: O[dd] /= l, 写回
        // ============================================================

        // 初始化 m, l
        tileMax tMax;  TEXPANDS(tMax, -1e30f);
        tileSum tSum;  TEXPANDS(tSum, 0.0f);

        // 初始化 O[Db] 数组
        tileO tO[Db];
        #pragma clang loop unroll(full)
        for (int dd = 0; dd < Db; ++dd) {
            TEXPANDS(tO[dd], 0.0f);
        }

        for (int j = 0; j < Kb; ++j) {

            // --- Step 1: QK^T 沿全 D 累加 ---
            // SuperScalarModel main does not preserve the input ACC correctly
            // across TMATMUL_ACC calls.  Convert each independent partial from
            // ACC NZ to Vec ND immediately and accumulate in the Vec tile.
            tileW tW;
            TEXPANDS(tW, 0.0f);
            #pragma clang loop unroll(full)
            for (int dd = 0; dd < Db; ++dd) {
                tileQ tQ;
                auto gQ = gIterQ(i, dd);
                TCOPYIN(tQ, gQ);  // RowMajor Q: ND -> NZ (Cube SrcL)

                tileKRight tK;
                auto gK = gIterK(dd, j);
                TCOPYIN(tK, gK);  // ColumnMajor K^T: DN -> ZN (Cube SrcR)

                tileW_out tW_out;
                TMATMUL(tW_out, tQ, tK);
                tileW tW_partial;
                TCVT_Impl(tW_partial, tW_out);  // ACC NZ -> Vec ND
                TADD(tW, tW, tW_partial);
            }

            // --- Step 2: scale + mask ---
            TMULS(tW, tW, scale);

            tileMask tMask;
            auto gMask = gIterMask(i, j);
            TLOAD(tMask, gMask);
            TADD(tW, tW, tMask);

            // --- Step 3: online softmax ---
            tileMax tLocalMax;
            TROWMAX(tLocalMax, tW);
            tileMax tNewMax;
            TMAX(tNewMax, tMax, tLocalMax);

            // rescale factor: exp(m_old - m_new)
            tileMax tScale;
            TSUB(tScale, tMax, tNewMax);
            TEXP(tScale, tScale);

            // rescale old sum
            tileSum tScaledOldSum;
            TMUL(tScaledOldSum, tSum, tScale);

            // p = exp(score - m_new)
            TROWEXPANDSUB(tW, tW, tNewMax);
            TEXP(tW, tW);

            // local sum
            tileSum tLocalSum;
            TROWSUM(tLocalSum, tW);

            // l_new = l_old' + local_sum
            tileSum tNewSum;
            TADD(tNewSum, tScaledOldSum, tLocalSum);

            // --- Step 4: rescale all O[dd] ---
            #pragma clang loop unroll(full)
            for (int dd = 0; dd < Db; ++dd) {
                TROWEXPANDMUL(tO[dd], tO[dd], tScale);
            }

            // --- Step 5: P@V for each D block ---
            tileW_cast tExpW;
            TCVT(tExpW, tW);
            tileW_left tW_left;
            TMOV_ND2NZ(tW_left, tExpW);  // Vec ND -> Cube SrcL NZ

            #pragma clang loop unroll(full)
            for (int dd = 0; dd < Db; ++dd) {
                tileV tV;
                auto gV = gIterV(j, dd);
                TCOPYIN(tV, gV);  // RowMajor V: ND -> ZN (Cube SrcR)

                tileO_out tPV_out;
                TMATMUL(tPV_out, tW_left, tV);
                tileO tPV;
                TCVT_Impl(tPV, tPV_out);  // ACC NZ -> Vec ND

                TADD(tO[dd], tO[dd], tPV);
            }

            // --- commit state ---
            tMax = tNewMax;
            tSum = tNewSum;
        }

        // --- 最终归一化: O[dd] /= l, 写回 ---
        tileSum tInvSum;
        TRECIP(tInvSum, tSum);

        #pragma clang loop unroll(full)
        for (int dd = 0; dd < Db; ++dd) {
            TROWEXPANDMUL(tO[dd], tO[dd], tInvSum);

            tileO_cast tO_cast;
            TCVT(tO_cast, tO[dd]);
            auto gO = gIterO(i, dd);
            TCOPYOUT(gO, tO_cast);
        }
    }
}

#endif
