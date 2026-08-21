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
//   mask 为 UINT32 条件矩阵 (1 表示无效/被 mask, 0 表示有效)
//   TSELECT_Impl(dst, mask, neg_inf, score): mask=1 → dst=-1e30
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
//   QK^T: 每个 D 块独立 TMATMUL, 转为 Vec 后用 TADD 累加
//   PV: 每个 D 分块独立计算并存储
//
// 【两遍式】
//   Pass 1: online softmax 归约 (m, l), 含 mask
//   Pass 2: 归一化 P, 计算 P@V, 含 mask
//
// 当前 template_asm.hpp 的 packed TSEL 包装与模型三输入契约不一致，
// 因此使用类型安全的四参数 TSELECT_Impl 和完整 UINT32 条件矩阵。
// =============================================================================

#include <common/pto_tileop.hpp>
#include "template_asm.h"

using namespace pto;

// CPU 侧 mask 预计算 (在 kernel 内部调用, 放在 stack 上)
// 生成 UINT32 条件矩阵：窗口外为 1，窗口内为 0。
static inline void build_swa_mask_select(
    uint32_t* maskBuf, int s1, int s2, int win_left, int win_right)
{
    const int causal_offset = s2 - s1;
    for (int q = 0; q < s1; ++q) {
        int diagonal = causal_offset + q;
        int lo = diagonal - win_left;
        int hi = diagonal + win_right;
        for (int kv = 0; kv < s2; ++kv) {
            bool valid = (kv >= lo) && (kv <= hi);
            maskBuf[q * s2 + kv] = valid ? 0u : 1u;
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

    uint32_t maskBuf[s1 * s2];
    build_swa_mask_select(maskBuf, s1, s2, ori_win_left, ori_win_right);

    using gmQ    = global_tensor<qdtype,  RowMajor<s1, D>>;
    using gmKV   = global_tensor<kvdtype, RowMajor<s2, D>>;
    // 与 gmKV 共用同一块内存，逻辑上表示 K^T；TCOPYIN 将 DN 转为 ZN。
    using gmKT   = global_tensor<kvdtype, ColMajor<D, s2>>;
    using gmO    = global_tensor<odttype, RowMajor<s1, D>>;
    using gmMask = global_tensor<uint32_t, RowMajor<s1, s2>>;

    using tileQ      = TileLeft<qdtype,  kTm, kTd>;
    using tileK      = TileRight<kvdtype, kTd, kTk>;
    using tileW_out  = TileAcc<float, kTm, kTk>;

    // score tile: RowMajor float
    using tileW      = Tile<Location::Vec, float, kTm, kTk, BLayout::RowMajor>;

    using tileMask   = Tile<Location::Vec, uint32_t, kTm, kTk,
                            BLayout::RowMajor>;

    using tileW_cast = Tile<Location::Vec, qdtype, kTm, kTk, BLayout::RowMajor>;
    using tileW_left = TileLeft<qdtype, kTm, kTk>;

    using tileO_out  = TileAcc<float, kTm, kTd>;
    using tileO      = Tile<Location::Vec, float, kTm, kTd, BLayout::RowMajor>;
    using tileO_cast = Tile<Location::Vec, odttype, kTm, kTd, BLayout::RowMajor>;

    using tileV      = TileRight<kvdtype, kTk, kTd>;
    using tileMax    = Tile<Location::Vec, float, kTm, 8, BLayout::RowMajor, kTm, 1>;
    using tileSum    = Tile<Location::Vec, float, kTm, 8, BLayout::RowMajor, kTm, 1>;

    using itQ    = global_iterator<gmQ,  tileQ>;
    using itK    = global_iterator<gmKT, tileK>;
    using itV    = global_iterator<gmKV, tileV>;
    using itO    = global_iterator<gmO,  tileO_cast>;
    using itMask = global_iterator<gmMask, tileMask>;

    itQ  gIterQ(q_ptr);
    itK  gIterK(ori_kv_ptr);
    itV  gIterV(ori_kv_ptr);
    itO  gIterO(out_ptr);
    itMask gIterMask(maskBuf);

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
            tileW tW;
            TEXPANDS(tW, 0.0f);
            #pragma clang loop unroll(full)
            for (int dd = 0; dd < Db; ++dd) {
                tileQ tQ;
                auto gQ = gIterQ(i, dd);
                TCOPYIN(tQ, gQ);

                tileK tK;
                auto gK = gIterK(dd, j);
                TCOPYIN(tK, gK);

                tileW_out tW_out;
                TMATMUL(tW_out, tQ, tK);
                tileW tW_partial;
                TCVT_Impl(tW_partial, tW_out);
                TADD(tW, tW, tW_partial);
            }

            TMULS(tW, tW, scale);

            // mask=1 选择 neg_inf，mask=0 保留 score。
            {
                tileMask tMask;
                auto gMask = gIterMask(i, j);
                TLOAD(tMask, gMask);
                tileW tMasked;
                TSELECT_Impl(tMasked, tMask, tNegInf, tW);
                tW = tMasked;
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
                tileW tW;
                TEXPANDS(tW, 0.0f);
                #pragma clang loop unroll(full)
                for (int dd2 = 0; dd2 < Db; ++dd2) {
                    tileQ tQ;
                    auto gQ = gIterQ(i, dd2);
                    TCOPYIN(tQ, gQ);

                    tileK tK;
                    auto gK = gIterK(dd2, j);
                    TCOPYIN(tK, gK);

                    tileW_out tW_out;
                    TMATMUL(tW_out, tQ, tK);
                    tileW tW_partial;
                    TCVT_Impl(tW_partial, tW_out);
                    TADD(tW, tW, tW_partial);
                }

                TMULS(tW, tW, scale);

                // 应用 token 级 mask: TSEL(score, mask, neg_inf)
                {
                    tileMask tMask;
                    auto gMask = gIterMask(i, j);
                    TLOAD(tMask, gMask);
                    tileW tMasked;
                    TSELECT_Impl(tMasked, tMask, tNegInf, tW);
                    tW = tMasked;
                }

                // p = exp(score - m) / l
                TROWEXPANDSUB(tW, tW, tMax);
                TEXP(tW, tW);
                TROWEXPANDMUL(tW, tW, tInvSum);

                // cast p -> qdtype Left tile for TMATMUL
                tileW_cast tExpW;
                TCVT(tExpW, tW);
                tileW_left tW_left;
                TMOV_ND2NZ(tW_left, tExpW);

                // PV = p * V (当前 D 分块)
                tileV tV;
                auto gV = gIterV(j, dd);
                TCOPYIN(tV, gV);

                tileO_out tPV_out;
                TMATMUL(tPV_out, tW_left, tV);
                tileO tPV;
                TCVT_Impl(tPV, tPV_out);

                TADD(tO, tO, tPV);
            }

            // 写回 O 分块 [kTm, kTd]
            tileO_cast tO_cast;
            TCVT(tO_cast, tO);
            auto gO = gIterO(i, dd);
            TCOPYOUT(gO, tO_cast);
        }
    }
}

#endif
