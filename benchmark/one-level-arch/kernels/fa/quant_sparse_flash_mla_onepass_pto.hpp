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
//   BSND 共享 Q token 路径只预生成 first/last/zero 三个 [TileM,TileK]
//   mask，内部整块直接复用 zero mask；旧 2D 路径保留 [s1,s2] mask。
//   mask 写入仍放在 Tile/CUBE 流程前，避免当前编码问题。
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
        const QsmlaSwaRange range = qsmla_swa_range(
            s2, logical_s1, logical_q, win_left, win_right);
        for (int kv = 0; kv < s2; ++kv) {
            mask[q * s2 + kv] = qsmla_swa_mask_value(kv, range);
        }
    }
}

template <typename qdtype, typename kvdtype, typename odttype, typename Config,
          bool SharedSwaMask = false>
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
    static_assert(!SharedSwaMask || s1 == kTm,
                  "shared BSND SWA mask expects one full M tile");
    constexpr int Db = D / kTd;

    constexpr int MaskTileElements = kTm * kTk;
    constexpr int MaskBufferElements =
        SharedSwaMask ? 3 * MaskTileElements : s1 * s2;
    float mask_buf[MaskBufferElements];
    if constexpr (!SharedSwaMask) {
        build_swa_mask_onepass(
            mask_buf, s1, s2, ori_win_left, ori_win_right,
            q_position, q_sequence_length);
    }

    using gmQ    = global_tensor<qdtype,  RowMajor<s1, D>>;
    using gmKV   = global_tensor<kvdtype, RowMajor<s2, D>>;
    using gmO    = global_tensor<odttype, RowMajor<s1, D>>;

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

    itQ    gIterQ(q_ptr);
    itO    gIterO(out_ptr);

    const int Qb = (s1 + kTm - 1) / kTm;
    const float scale = softmax_scale;

    for (int i = 0; i < Qb; ++i) {
        const int q_row_begin = i * kTm;
        constexpr bool shared_q_position = SharedSwaMask;
        const int logical_q_sequence_length =
            shared_q_position ? q_sequence_length : s1;
        const int first_logical_q =
            shared_q_position ? q_position : q_row_begin;
        const int last_q_row =
            q_row_begin + kTm < s1 ? q_row_begin + kTm - 1 : s1 - 1;
        const int last_logical_q =
            shared_q_position ? q_position : last_q_row;
        const QsmlaSwaRange first_range = qsmla_swa_range(
            s2, logical_q_sequence_length, first_logical_q,
            ori_win_left, ori_win_right);
        const QsmlaSwaRange last_range = qsmla_swa_range(
            s2, logical_q_sequence_length, last_logical_q,
            ori_win_left, ori_win_right);
        const QsmlaSwaRange kv_range = {first_range.begin, last_range.end};
        const QsmlaSwaRange kv_blocks = qsmla_swa_block_range(kv_range, kTk);
        const int kv_block_count = kv_blocks.end - kv_blocks.begin;
        if constexpr (SharedSwaMask) {
            qsmla_build_shared_swa_masks(
                mask_buf,
                mask_buf + MaskTileElements,
                mask_buf + 2 * MaskTileElements,
                kTm, kTk, kv_blocks.begin, kv_block_count,
                s2, q_sequence_length, q_position,
                ori_win_left, ori_win_right);
        }
        kvdtype* clipped_kv_ptr =
            ori_kv_ptr + static_cast<std::size_t>(kv_blocks.begin) * kTk * D;
        itKSrc gIterKSrc(clipped_kv_ptr);
        itV gIterV(clipped_kv_ptr);

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

        for (int j = 0; j < kv_block_count; ++j) {

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
            if constexpr (SharedSwaMask) {
                float* selected_mask = mask_buf + 2 * MaskTileElements;
                if (j == 0) {
                    selected_mask = mask_buf;
                }
                if (j + 1 == kv_block_count) {
                    selected_mask = mask_buf + MaskTileElements;
                }
                using gmSharedMask =
                    global_tensor<float, RowMajor<kTm, kTk>>;
                using itSharedMask = global_iterator<gmSharedMask, tileMask>;
                itSharedMask gIterMask(selected_mask);
                auto gMask = gIterMask(0, 0);
                TLOAD(tMask, gMask);
            } else {
                using gmFullMask = global_tensor<float, RowMajor<s1, s2>>;
                using itFullMask = global_iterator<gmFullMask, tileMask>;
                itFullMask gIterMask(mask_buf);
                auto gMask = gIterMask(i, kv_blocks.begin + j);
                TLOAD(tMask, gMask);
            }
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
            qdtype, kvdtype, odttype, WorkConfig, true>(
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
            qdtype, kvdtype, odttype, TailConfig, true>(
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
