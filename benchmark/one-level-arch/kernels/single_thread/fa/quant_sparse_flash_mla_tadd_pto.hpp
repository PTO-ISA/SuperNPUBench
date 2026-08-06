#ifndef QUANT_SPARSE_FLASH_MLA_TADD_PTO_HPP
#define QUANT_SPARSE_FLASH_MLA_TADD_PTO_HPP

// =============================================================================
// quant_sparse_flash_mla_tadd_pto.hpp
//   Quant Sparse Flash MLA (SWA mode) — TADD mask variant
//
//   本文件是 quant_sparse_flash_mla_pto.hpp (TSEL 版) 的备选实现,
//   使用 TADD 替代 TSEL 施加 mask, 代码更简单, 兼容旧版模拟器.
//
//   切换方法: 在 test 文件中修改 include 和函数名:
//     #include "fa/quant_sparse_flash_mla_tadd_pto.hpp"
//     quant_sparse_flash_mla_swa_tadd_pto<...>(...)
//
// 【与 TSEL 版的差异】
//   | 项目        | TADD 版 (本文件)              | TSEL 版 (quant_sparse_flash_mla_pto.hpp)     |
//   |-------------|------------------------------|----------------------------------------------|
//   | mask 格式   | float[s1*s2], 0.0/-1e30      | uint32_t 位打包[s1*ceil(s2/32)]              |
//   | mask 构建   | build_swa_mask 直接赋值       | build_swa_mask_bitpacked 位操作              |
//   | mask 加载   | global_iterator 直接 TLOAD    | 手动提取 block bit 重打包后 TLOAD             |
//   | mask 应用   | TADD(score, score, mask)      | TSEL(score, mask, neg_inf)                   |
//   | 额外 tile   | 无                            | tNegInf (TEXPANDS -1e30)                     |
//   | 模拟器要求  | 无特殊要求                    | 需更新后模拟器 (o_main 分支, ExecuteTSEL fix)|
//
// 【计算语义】
//   O = softmax(Q @ K^T * softmax_scale + mask) @ V
//   Q: [s1, D], KV: [s2, D] (MLA shared K=V=ori_kv), O: [s1, D]
//
// 【SWA 滑动窗口 — kernel 内部 token 级 mask, 使用 TADD】
//   mask[q_idx, kv_idx] = 0.0f   if kv_idx 在窗口内 (有效, score 不变)
//                       = -1e30f if kv_idx 在窗口外 (无效, score → -1e30)
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
// =============================================================================

#include <common/pto_tileop.hpp>
#include "template_asm.h"

using namespace pto;

// CPU 侧 mask 预计算 (在 kernel 内部调用, 放在 stack 上)
// mask[q_idx * s2 + kv_idx] = 0.0f if valid, -1e30f if invalid
// valid: diagonal - win_left <= kv_idx <= diagonal + win_right
//   where diagonal = (s2 - s1) + q_idx  (causal offset + q position)
// kernel 中用 TADD: score += mask (0 保持原值, -1e30 屏蔽)
static inline void build_swa_mask(
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
void quant_sparse_flash_mla_swa_tadd_pto(
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

    // kernel 内部计算 mask, 放在 stack 上
    // s1*s2*sizeof(float) = 64*128*4 = 32KB (可容纳)
    float mask_buf[s1 * s2];
    build_swa_mask(mask_buf, s1, s2, ori_win_left, ori_win_right);

    using gmQ    = global_tensor<qdtype,  RowMajor<s1, D>>;
    using gmKV   = global_tensor<kvdtype, RowMajor<s2, D>>;
    using gmO    = global_tensor<odttype, RowMajor<s1, D>>;
    using gmMask = global_tensor<float,   RowMajor<s1, s2>>;

    using tileQ      = TileLeft<qdtype,  kTm, kTd>;
    using tileKV     = TileRight<kvdtype, kTk, kTd>;
    using tileW_out  = TileAcc<float, kTm, kTk>;

    // score tile 与 mask tile 都用 RowMajor, 保证 TADD 类型一致
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
    using itKV   = global_iterator<gmKV, tileKV>;
    using itV    = global_iterator<gmKV, tileV>;
    using itO    = global_iterator<gmO,  tileO_cast>;
    using itMask = global_iterator<gmMask, tileMask>;

    itQ    gIterQ(q_ptr);
    itKV   gIterKV(ori_kv_ptr);
    itV    gIterV(ori_kv_ptr);
    itO    gIterO(out_ptr);
    itMask gIterMask(mask_buf);

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

            // 应用 token 级 mask: score += mask (0 保持原值, -1e30 屏蔽)
            tileMask tMask;
            auto gMask = gIterMask(i, j);
            TLOAD(tMask, gMask);
            TADD(tW, tW, tMask);

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
        //  Pass 2: p = exp(QK*scale + mask - m) / l, O = Σ p·V
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

                // 应用 token 级 mask: score += mask (0 保持原值, -1e30 屏蔽)
                tileMask tMask;
                auto gMask = gIterMask(i, j);
                TLOAD(tMask, gMask);
                TADD(tW, tW, tMask);

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
