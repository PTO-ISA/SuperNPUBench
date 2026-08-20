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
#include "qsmla_config_pto.hpp"

using namespace pto;

static inline void build_swa_mask_onepass(
    float* mask, int s1, int s2, int win_left, int win_right,
    int q_position = -1, int q_sequence_length = -1)
{
    for (int q = 0; q < s1; ++q) {
        const int logical_q = q_position >= 0 ? q_position : q;
        const int logical_s1 = q_position >= 0 ? q_sequence_length : s1;
        int diagonal = s2 - logical_s1 + logical_q;
        int lo = diagonal - win_left;
        int hi = diagonal + win_right;
        for (int kv = 0; kv < s2; ++kv) {
            bool valid = (kv >= lo) && (kv <= hi);
            mask[q * s2 + kv] = valid ? 0.0f : -1e30f;
        }
    }
}

template <typename qdtype, typename kvdtype, typename odttype, typename Config>
void quant_sparse_flash_mla_swa_onepass_config_pto(
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
    float* softmax_lse,
    int q_position = -1,
    int q_sequence_length = -1)
{
    constexpr int s1 = Config::S1;
    constexpr int s2 = Config::S2;
    constexpr int D = Config::D;
    constexpr int kTm = Config::TileM;
    constexpr int kTk = Config::TileK;
    constexpr int kTd = Config::TileD;
    static_assert(D % kTd == 0,
                  "one-pass D-tail support is implemented in the next shape-generalization step");
    constexpr int Db = D / kTd;

    float mask_buf[s1 * s2];
    build_swa_mask_onepass(
        mask_buf, s1, s2, ori_win_left, ori_win_right,
        q_position, q_sequence_length);

    using gmQ    = global_tensor<qdtype,  RowMajor<s1, D>>;
    using gmKV   = global_tensor<kvdtype, RowMajor<s2, D>>;
    using gmO    = global_tensor<odttype, RowMajor<s1, D>>;
    using gmMask = global_tensor<float,   RowMajor<s1, s2>>;

    using tileQ      = TileLeft<qdtype, kTm, kTd>;
    using tileKSrc   = Tile<Location::Vec, kvdtype, kTk, kTd, BLayout::RowMajor>;
    using tileKRight = TileRight<kvdtype, kTd, kTk>;
    using tileW      = Tile<Location::Vec, float, kTm, kTk, BLayout::RowMajor>;
    using tileMask   = Tile<Location::Vec, float, kTm, kTk, BLayout::RowMajor>;
    using tileW_left = TileLeft<qdtype, kTm, kTk>;

    using tileO      = Tile<Location::Vec, float, kTm, kTd, BLayout::RowMajor>;
    using tileO_cast = Tile<Location::Vec, odttype, kTm, kTd, BLayout::RowMajor>;

    using tileV      = TileRight<kvdtype, kTk, kTd>;
    using tileMax    = Tile<Location::Vec, float, kTm, 8, BLayout::RowMajor, kTm, 1>;
    using tileSum    = Tile<Location::Vec, float, kTm, 8, BLayout::RowMajor, kTm, 1>;

    using itQ    = global_iterator<gmQ,  tileQ>;
    using itKSrc = global_iterator<gmKV, tileKSrc>;
    using itV    = global_iterator<gmKV, tileV>;
    using itO    = global_iterator<gmO,  tileO_cast>;
    using itMask = global_iterator<gmMask, tileMask>;

    itQ    gIterQ(q_ptr);
    itKSrc gIterKSrc(ori_kv_ptr);
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
                TLOAD(tQ, gQ);  // RowMajor Q -> Left tile

                tileKSrc tKSrc;
                auto gK = gIterKSrc(j, dd);
                TLOAD(tKSrc, gK);  // Original K block [Tk, Td]

                tileKRight tK;
                TTRANS(tK, tKSrc);  // [Tk, Td] -> [Td, Tk] for MM1 SrcR

                tileW tW_partial;
                TMATMUL(tW_partial, tQ, tK);
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
            tileW_left tW_left;
            // PTO v0.58 Local CUBE reads Left payloads as NORM row-major and
            // has no NZ dependency, so convert/copy FP32 probabilities directly
            // into the qdtype Left tile without an intermediate Vec tile.
            TCVT(tW_left, tW);

            #pragma clang loop unroll(full)
            for (int dd = 0; dd < Db; ++dd) {
                tileV tV;
                auto gV = gIterV(j, dd);
                TLOAD(tV, gV);  // RowMajor V -> Right tile

                tileO tPV;
                TMATMUL(tPV, tW_left, tV);

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
            TSTORE(gO, tO_cast);
        }
    }
}

// Compatibility entry for the existing fixed two-dimensional smoke. New BSND
// dispatch code should instantiate QsmlaConfig explicitly and call the Config
// entry above.
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
    using Config = QsmlaConfig<1, s1, s2, 1, 1, D, 0, kTm, kTk, kTd>;
    quant_sparse_flash_mla_swa_onepass_config_pto<qdtype, kvdtype, odttype, Config>(
        out_ptr, q_ptr, ori_kv_ptr, softmax_scale, ori_win_left, ori_win_right,
        q_descale, ori_kv_descale, ori_sparse_indices, ori_block_table,
        cu_seqlens_q, cu_seqlens_ori_kv, seqused_q, seqused_ori_kv,
        sinks, metadata, softmax_lse, -1, -1);
}

// Stage-1 BSND dispatcher. A work item owns all G rows for one
// (batch, qToken, kvHead, gSlice), so its online-softmax state never crosses
// work-item boundaries. Each slice is further divided into full TileM chunks.
// The final short chunk is zero-padded in a kernel-local buffer because the
// current Local TEPL reduction chain cannot consistently consume a partial-M
// ValidRow tile; only its valid rows are copied back. N2>1 needs a strided KV
// view and is intentionally deferred instead of treating non-contiguous
// [S2,N2,D] storage as [S2,D].
template <typename qdtype, typename kvdtype, typename odttype, typename Config>
void quant_sparse_flash_mla_swa_onepass_bsnd_pto(
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
    static_assert(Config::N2 == 1,
                  "Stage-1 BSND dispatcher currently requires contiguous N2=1 KV");

    auto run_full_rows = [&](int row_offset, const QsmlaWorkItem& work) {
        using WorkConfig = QsmlaConfig<
            1, Config::TileM, Config::S2, 1, 1, Config::D, Config::K,
            Config::TileM, Config::TileK, Config::TileD, Config::TileM>;

        quant_sparse_flash_mla_swa_onepass_config_pto<
            qdtype, kvdtype, odttype, WorkConfig>(
            out_ptr + Config::out_work_offset(work) + row_offset * Config::D,
            q_ptr + Config::q_work_offset(work) + row_offset * Config::D,
            ori_kv_ptr + Config::kv_work_offset(work),
            softmax_scale, ori_win_left, ori_win_right,
            q_descale, ori_kv_descale, ori_sparse_indices, ori_block_table,
            cu_seqlens_q, cu_seqlens_ori_kv, seqused_q, seqused_ori_kv,
            sinks, metadata, softmax_lse, work.q_token, Config::S1);
    };

    auto run_tail_rows = [&]<int Rows>(int row_offset, const QsmlaWorkItem& work) {
        static_assert(Rows > 0 && Rows < Config::TileM);
        qdtype padded_q[Config::TileM * Config::D];
        odttype padded_out[Config::TileM * Config::D];
        qdtype* work_q = q_ptr + Config::q_work_offset(work) + row_offset * Config::D;
        odttype* work_out = out_ptr + Config::out_work_offset(work) + row_offset * Config::D;

        for (int row = 0; row < Config::TileM; ++row) {
            for (int dim = 0; dim < Config::D; ++dim) {
                padded_q[row * Config::D + dim] =
                    row < Rows ? work_q[row * Config::D + dim]
                               : static_cast<qdtype>(0.0f);
            }
        }

        using TailConfig = QsmlaConfig<
            1, Config::TileM, Config::S2, 1, 1, Config::D, Config::K,
            Config::TileM, Config::TileK, Config::TileD, Config::TileM>;
        quant_sparse_flash_mla_swa_onepass_config_pto<
            qdtype, kvdtype, odttype, TailConfig>(
            padded_out, padded_q,
            ori_kv_ptr + Config::kv_work_offset(work),
            softmax_scale, ori_win_left, ori_win_right,
            q_descale, ori_kv_descale, ori_sparse_indices, ori_block_table,
            cu_seqlens_q, cu_seqlens_ori_kv, seqused_q, seqused_ori_kv,
            sinks, metadata, softmax_lse, work.q_token, Config::S1);

        for (int row = 0; row < Rows; ++row) {
            for (int dim = 0; dim < Config::D; ++dim) {
                work_out[row * Config::D + dim] =
                    padded_out[row * Config::D + dim];
            }
        }
    };

    constexpr int kFullSliceChunks = Config::GSliceMax / Config::TileM;
    constexpr int kFullSliceTail = Config::GSliceMax % Config::TileM;
    constexpr int kLastSliceRows = Config::G % Config::GSliceMax;
    constexpr int kLastSliceChunks = kLastSliceRows / Config::TileM;
    constexpr int kLastSliceTail = kLastSliceRows % Config::TileM;

    for (int work_id = 0; work_id < Config::WorkCount; ++work_id) {
        const QsmlaWorkItem work = Config::decode_work(work_id);
        if (work.m_real == Config::GSliceMax) {
            for (int chunk = 0; chunk < kFullSliceChunks; ++chunk) {
                run_full_rows(chunk * Config::TileM, work);
            }
            if constexpr (kFullSliceTail != 0) {
                run_tail_rows.template operator()<kFullSliceTail>(
                    kFullSliceChunks * Config::TileM, work);
            }
        } else {
            for (int chunk = 0; chunk < kLastSliceChunks; ++chunk) {
                run_full_rows(chunk * Config::TileM, work);
            }
            if constexpr (kLastSliceTail != 0) {
                run_tail_rows.template operator()<kLastSliceTail>(
                    kLastSliceChunks * Config::TileM, work);
            }
        }
    }
}

#endif
