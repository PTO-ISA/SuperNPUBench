#ifndef QUANT_SPARSE_FLASH_MLA_TADD_HPP
#define QUANT_SPARSE_FLASH_MLA_TADD_HPP

// =============================================================================
// quant_sparse_flash_mla_tadd.hpp
//   Quant Sparse Flash MLA (SWA mode) — TADD mask variant
//
//   保留的单 PE FP16 SWA 实现，使用 TADD 施加 mask。
//   正式五模式 / HIF8 / 4-PE 实现在 multi_thread/fa 中，与本文件独立。
//
//   切换方法: 在 test 文件中修改 include 和函数名:
//     #include "solution/quant_sparse_flash_mla/quant_sparse_flash_mla_tadd.hpp"
//     quant_sparse_flash_mla_swa_tadd_pto<...>(...)
//   现有测试入口: make ... TESTCASE=quant_sparse_flash_mla IMPL=tadd
//                  QSMLA_DTYPE=FP16 Tm=32 Tk=32 Td_block=64
//   执行: gfrun -f <ELF> -s softcore.multiThreadNum=1
//
// 【当前 Local CUBE 接口】
//   参考 single_thread/fa/fa_2d_unroll.hpp：Left/Acc 使用 CUBE_M16/M32，
//   Right 使用 CUBE_N8。TLOAD_CUBE/TSTORE_CUBE 显式转换 GM ND 与 CELL 布局；
//   不能把普通 Vec/旧 TileLeft/TileRight 直接交给当前 Local TMATMUL。
//   K 仍按块 TTRANS；经小块 GM 缓冲加载到 CUBE，不分配完整转置 KV。
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
//   QK^T: 每个 D 块独立 TMATMUL, 转为 Vec 后用 TADD 累加
//   PV: 每个 D 分块独立计算并存储
//
// 【两遍式】
//   Pass 1: online softmax 归约 (m, l), 含 mask
//   Pass 2: 归一化 P, 计算 P@V, 含 mask
// =============================================================================

#include <common/pto_tileop.hpp>
#include "template_asm.h"
#include "qsmla_config.hpp"
#include <type_traits>

using namespace pto;

// CPU 侧 mask 预计算 (在 kernel 内部调用, 放在 stack 上)
// mask[q_idx * s2 + kv_idx] = 0.0f if valid, -1e30f if invalid
// valid: diagonal - win_left <= kv_idx <= diagonal + win_right
//   where diagonal = (s2 - s1) + q_idx  (causal offset + q position)
// kernel 中用 TADD: score += mask (0 保持原值, -1e30 屏蔽)
static inline void build_swa_mask_tadd(
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
void quant_sparse_flash_mla_swa_tadd_config_pto(
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
    static_assert(kTm <= 32, "single-PE Local CUBE supports TileM <= 32");
    static_assert(std::is_same_v<qdtype, __half> &&
                      std::is_same_v<kvdtype, __half> &&
                      std::is_same_v<odttype, __half>,
                  "single-PE tadd supports FP16 only; HIF8 uses the four-PE kernel");
    static_assert(D % kTd == 0,
                  "tadd D-tail support is intentionally deferred");
    static_assert(!SharedSwaMask || s1 == kTm,
                  "shared BSND SWA mask expects one full M tile");
    constexpr int Db = D / kTd;

    constexpr int MaskTileElements = kTm * kTk;
    constexpr int MaskBufferElements =
        SharedSwaMask ? 3 * MaskTileElements : s1 * s2;
    // BSND 的 TileM 行属于同一 Q token，因此只需要首块、尾块和
    // 全有效中间块三个小 mask；旧 2D 路径仍需要逐行 mask。
    float mask_buf[MaskBufferElements];
    if constexpr (!SharedSwaMask) {
        build_swa_mask_tadd(
            mask_buf, s1, s2, ori_win_left, ori_win_right,
            q_position, q_sequence_length);
    }

    using gmQ    = global_tensor<qdtype,  RowMajor<s1, D>>;
    using gmKV   = global_tensor<kvdtype, RowMajor<s2, D>>;
    using gmO    = global_tensor<odttype, RowMajor<s1, D>>;

    using tileQ      = std::conditional_t<
        (kTm <= 16), CubeTileM16<qdtype, kTm, kTd>,
        CubeTileM32<qdtype, kTm, kTd>>;
    using tileKSrc   = Tile<Location::Vec, kvdtype, kTk, kTd, BLayout::RowMajor>;
    using tileKTrans = Tile<Location::Vec, kvdtype, kTd, kTk, BLayout::RowMajor>;
    using tileKRight = CubeTileN8<kvdtype, kTd, kTk>;
    using tileScoreCube = std::conditional_t<
        (kTm <= 16), CubeAccumulatorM16<float, kTm, kTk>,
        CubeAccumulatorM32<float, kTm, kTk>>;

    // score tile 与 mask tile 都用 RowMajor, 保证 TADD 类型一致
    using tileW      = Tile<Location::Vec, float, kTm, kTk, BLayout::RowMajor>;
    using tileMask   = Tile<Location::Vec, float, kTm, kTk, BLayout::RowMajor>;
    using tileW_cast = Tile<Location::Vec, qdtype, kTm, kTk, BLayout::RowMajor>;
    using tileW_left = std::conditional_t<
        (kTm <= 16), CubeTileM16<qdtype, kTm, kTk>,
        CubeTileM32<qdtype, kTm, kTk>>;

    using tileO      = Tile<Location::Vec, float, kTm, kTd, BLayout::RowMajor>;
    using tileO_cast = Tile<Location::Vec, odttype, kTm, kTd, BLayout::RowMajor>;

    using tileV      = CubeTileN8<kvdtype, kTk, kTd>;
    using tilePVCube = std::conditional_t<
        (kTm <= 16), CubeAccumulatorM16<float, kTm, kTd>,
        CubeAccumulatorM32<float, kTm, kTd>>;
    // Row reductions produce a dense [M,1] descriptor in the current API.
    using tileMax    = Tile<Location::Vec, float, kTm, 1, BLayout::RowMajor, kTm, 1>;
    using tileSum    = tileMax;

    // Explicit Local CUBE <-> Vec conversion boundaries, reused per block.
    // Single-PE scratch may live on this PE's stack; no shared workspace or
    // whole-sequence K transpose is needed. TCVT alone is not a CELL conversion.
    alignas(64) kvdtype k_trans_scratch[kTd * kTk];
    alignas(64) float score_scratch[kTm * kTk];
    alignas(64) qdtype prob_scratch[kTm * kTk];
    alignas(64) float pv_scratch[kTm * kTd];
    global_tensor<kvdtype, RowMajor<kTd, kTk>> gKTrans(k_trans_scratch);
    global_tensor<float, RowMajor<kTm, kTk>> gScore(score_scratch);
    global_tensor<qdtype, RowMajor<kTm, kTk>> gProb(prob_scratch);
    global_tensor<float, RowMajor<kTm, kTd>> gPV(pv_scratch);

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
        //  Pass 1: online softmax 归约 row max (m) 与 row sum (l)
        //  只遍历与 SWA 有效区间相交的 KV 块
        // ============================================================
        tileMax tMax;  TEXPANDS(tMax, -1e30f);
        tileSum tSum;  TEXPANDS(tSum, 0.0f);

        for (int j = 0; j < kv_block_count; ++j) {

            // QK^T 沿 D 维累加
            tileW tW;
            TEXPANDS(tW, 0.0f);
            #pragma clang loop unroll(full)
            for (int dd = 0; dd < Db; ++dd) {
                tileQ tQ;
                auto gQ = gIterQ(i, dd);
                TLOAD_CUBE(tQ, gQ);
                tileKSrc tKSrc;
                auto gK = gIterKSrc(j, dd);
                TLOAD(tKSrc, gK);
                tileKTrans tKTrans;
                TTRANS(tKTrans, tKSrc);
                TSTORE(gKTrans, tKTrans);
                tileKRight tK;
                TLOAD_CUBE(tK, gKTrans);
                tileScoreCube tScoreCube;
                TMATMUL(tScoreCube, tQ, tK);
                TSTORE_CUBE(gScore, tScoreCube);
                tileW tW_partial;
                TLOAD(tW_partial, gScore);
                TADD(tW, tW, tW_partial);
            }

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

            for (int j = 0; j < kv_block_count; ++j) {

                // 计算完整 QK^T (沿 D 累加, 与 Pass 1 一致)
                tileW tW;
                TEXPANDS(tW, 0.0f);
                #pragma clang loop unroll(full)
                for (int dd2 = 0; dd2 < Db; ++dd2) {
                    tileQ tQ;
                    auto gQ = gIterQ(i, dd2);
                    TLOAD_CUBE(tQ, gQ);
                    tileKSrc tKSrc;
                    auto gK = gIterKSrc(j, dd2);
                    TLOAD(tKSrc, gK);
                    tileKTrans tKTrans;
                    TTRANS(tKTrans, tKSrc);
                    TSTORE(gKTrans, tKTrans);
                    tileKRight tK;
                    TLOAD_CUBE(tK, gKTrans);
                    tileScoreCube tScoreCube;
                    TMATMUL(tScoreCube, tQ, tK);
                    TSTORE_CUBE(gScore, tScoreCube);
                    tileW tW_partial;
                    TLOAD(tW_partial, gScore);
                    TADD(tW, tW, tW_partial);
                }

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

                // p = exp(score - m) / l
                TROWEXPANDSUB(tW, tW, tMax);
                TEXP(tW, tW);
                TROWEXPANDMUL(tW, tW, tInvSum);

                tileW_cast tProb;
                TCVT(tProb, tW);
                TSTORE(gProb, tProb);
                tileW_left tW_left;
                TLOAD_CUBE(tW_left, gProb);

                // PV = p * V (当前 D 分块)
                auto gV = gIterV(j, dd);
                tileV tV;
                TLOAD_CUBE(tV, gV);
                tilePVCube tPVCube;
                TMATMUL(tPVCube, tW_left, tV);
                TSTORE_CUBE(gPV, tPVCube);
                tileO tPV;
                TLOAD(tPV, gPV);

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

// Compatibility entry for the existing fixed two-dimensional smoke.
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
    using Config = QsmlaConfig<1, s1, s2, 1, 1, D, 0, kTm, kTk, kTd>;
    quant_sparse_flash_mla_swa_tadd_config_pto<
        qdtype, kvdtype, odttype, Config>(
        out_ptr, q_ptr, ori_kv_ptr, softmax_scale,
        ori_win_left, ori_win_right,
        q_descale, ori_kv_descale, ori_sparse_indices, ori_block_table,
        cu_seqlens_q, cu_seqlens_ori_kv, seqused_q, seqused_ori_kv,
        sinks, metadata, softmax_lse, -1, -1);
}

// BSND dispatcher using the common QsmlaConfig address model. One work item owns
// all G rows of one (batch, qToken, kvHead, gSlice). N2>1 remains deferred
// because [S2,N2,D] KV storage needs a strided view.
template <typename qdtype, typename kvdtype, typename odttype, typename Config>
void quant_sparse_flash_mla_swa_tadd_bsnd_pto(
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

        quant_sparse_flash_mla_swa_tadd_config_pto<
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
        qdtype* work_q =
            q_ptr + Config::q_work_offset(work) + row_offset * Config::D;
        odttype* work_out =
            out_ptr + Config::out_work_offset(work) + row_offset * Config::D;

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
        quant_sparse_flash_mla_swa_tadd_config_pto<
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
