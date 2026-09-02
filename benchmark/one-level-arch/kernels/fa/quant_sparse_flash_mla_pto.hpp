#ifndef QUANT_SPARSE_FLASH_MLA_PTO_HPP
#define QUANT_SPARSE_FLASH_MLA_PTO_HPP

#include <type_traits>
#include <common/pto_tileop.hpp>
#include "template_asm.h"
#include "qsmla_config_pto.hpp"
#include "qsmla_mode_pto.hpp"

using namespace pto;

// Unified five-mode four-PE TADD implementation. Mode selection only decides
// which logical ORI/CMP RANGE or INDEXED sources participate. Every source is
// then visited by the same QK -> online-softmax -> PV machinery and updates
// one shared (m,l,O) state. Indexed rows use one reusable TileK-by-D staging
// tile; no TopK-by-D or source-length-by-D workspace is allocated.
template <typename qdtype, typename kvdtype, typename odttype,
          typename ModeConfig>
void quant_sparse_flash_mla_tadd_4pe_bsnd_pto(
    odttype* out_ptr,
    qdtype* q_ptr,
    kvdtype* ori_kv_ptr,
    kvdtype* cmp_kv_ptr,
    const int* ori_sparse_indices,
    const int* cmp_sparse_indices,
    const int* ori_topk_length,
    const int* cmp_topk_length,
    float softmax_scale,
    float q_descale,
    float ori_kv_descale,
    float cmp_kv_descale,
    int cmp_ratio,
    int ori_win_left,
    int ori_win_right,
    float* score_scratch,
    qdtype* prob_scratch,
    float* pv_scratch)
{
    using Config = typename ModeConfig::Base;
    constexpr int kPeNum = 4;
    constexpr int kGroupM = Config::TileM;
    constexpr int kTk = Config::TileK;
    constexpr int kTd = Config::TileD;
    constexpr int kDb = Config::D / kTd;
    constexpr int kPeRows = kGroupM / kPeNum;
    constexpr float kHif8ProbabilityScale = 16.0f;
    constexpr bool kUseHif8Probability =
        std::is_same_v<qdtype, __hif8>;
    constexpr int kOriIndexStorage =
        ModeConfig::OriTopK > 0 ? ModeConfig::OriTopK : 1;
    constexpr int kCmpIndexStorage =
        ModeConfig::CmpTopK > 0 ? ModeConfig::CmpTopK : 1;

    static_assert(Config::N2 == 1,
                  "sparse four-PE BSND requires contiguous N2=1 KV");
    static_assert(Config::GSliceMax == 64 && Config::G % 64 == 0,
                  "sparse four-PE requires complete 64-head G slices");
    static_assert(kGroupM == 64 && kTk == 32,
                  "sparse v1 fixes TileM=64 and TileK=32");
    static_assert(Config::D % kTd == 0,
                  "sparse four-PE D-tail support is deferred");
    static_assert(!std::is_same_v<kvdtype, __hif8> || Config::D % 4 == 0,
                  "HIF8 gather uses four-byte carriers");

    const int pe_id = static_cast<int>(get_thread_idx());

    using tileQMatrix = SharedMatrixLeft<qdtype, kGroupM, kTd>;
    using tileKMatrix = SharedMatrixRight<kvdtype, kTd, kTk>;
    using tilePMatrix = SharedMatrixLeft<qdtype, kGroupM, kTk>;
    using tileVMatrix = SharedMatrixRight<kvdtype, kTk, kTd>;
    using tileQShared = SharedTile<tileQMatrix>;
    using tileKShared = SharedTile<tileKMatrix>;
    using tilePShared = SharedTile<tilePMatrix>;
    using tileVShared = SharedTile<tileVMatrix>;
    using tileScoreCube = CubeAccumulatorM16<float, kPeRows, kTk>;
    using tilePVCube = CubeAccumulatorM16<float, kPeRows, kTd>;
    using tileW =
        Tile<Location::Vec, float, kPeRows, kTk, BLayout::RowMajor>;
    using tileMask = tileW;
    using tilePShard =
        Tile<Location::Vec, qdtype, kPeRows, kTk, BLayout::RowMajor>;
    using tileKSrc =
        Tile<Location::Vec, kvdtype, kTk, kTd, BLayout::RowMajor>;
    using tileO =
        Tile<Location::Vec, float, kPeRows, kTd, BLayout::RowMajor>;
    using tileOCast =
        Tile<Location::Vec, odttype, kPeRows, kTd, BLayout::RowMajor>;
    using tileMax =
        Tile<Location::Vec, float, kPeRows, 1, BLayout::RowMajor,
             kPeRows, 1>;
    using tileSum = tileMax;

    using gmQ = global_tensor<qdtype, RowMajor<kGroupM, Config::D>>;
    using gmGatherKV = global_tensor<kvdtype, RowMajor<kTk, Config::D>>;
    using gmO = global_tensor<odttype, RowMajor<kGroupM, Config::D>>;
    using itQ = global_iterator<gmQ, tileQMatrix>;
    using itKSrc = global_iterator<gmGatherKV, tileKSrc>;
    using itV = global_iterator<gmGatherKV, tileVMatrix>;
    using itO = global_iterator<gmO, tileOCast>;

    using gmScoreScratch = global_tensor<float, RowMajor<kPeRows, kTk>>;
    using gmProbScratch = global_tensor<qdtype, RowMajor<kGroupM, kTk>>;
    using gmPVScratch = global_tensor<float, RowMajor<kPeRows, Config::D>>;
    using gmRowState = global_tensor<float, RowMajor<kPeRows, 1>>;
    using itProbShard = global_iterator<gmProbScratch, tilePShard>;
    using itPShared = global_iterator<gmProbScratch, tilePMatrix>;
    using itPVScratch = global_iterator<gmPVScratch, tileO>;
    using itRowState = global_iterator<gmRowState, tileMax>;
    float* pe_score_scratch = score_scratch + pe_id * kPeRows * kTk;
    gmScoreScratch gScore(pe_score_scratch);
    itPVScratch gIterPV(
        pv_scratch +
        qsmla_full_o_scratch_pe_offset(pe_id, kPeRows, Config::D));
    itRowState gIterMax(pe_score_scratch);
    itRowState gIterSum(pe_score_scratch + kPeRows);
    itProbShard gIterProb(prob_scratch);
    itPShared gIterP(prob_scratch);

    constexpr int kMaskElements = kPeRows * kTk;
    float mask_buf[kMaskElements];
    // One bounded staging tile is reused by ORI and CMP and by both passes.
    kvdtype kv_tile_buf[kTk * Config::D];
    int ori_selected[kOriIndexStorage];
    int cmp_selected[kCmpIndexStorage];

    for (int work_id = 0; work_id < Config::WorkCount; ++work_id) {
        const QsmlaWorkItem work = Config::decode_work(work_id);
        qdtype* work_q = q_ptr + Config::q_work_offset(work);
        const std::size_t work_out_offset = Config::out_work_offset(work);
        itQ gIterQ(work_q);

        kvdtype* work_ori = ori_kv_ptr +
            ((static_cast<std::size_t>(work.batch) * ModeConfig::OriS2)
             * Config::N2 + work.kv_head) * Config::D;
        kvdtype* work_cmp = nullptr;
        if constexpr (ModeConfig::HasCmp) {
            work_cmp = cmp_kv_ptr +
                ((static_cast<std::size_t>(work.batch) * ModeConfig::CmpS2)
                 * Config::N2 + work.kv_head) * Config::D;
        }

        int ori_begin = 0;
        int ori_count = 0;
        if constexpr (ModeConfig::Mode == QsmlaMode::SWA ||
                      ModeConfig::Mode == QsmlaMode::HCA ||
                      ModeConfig::Mode == QsmlaMode::CSA) {
            const QsmlaSwaRange range = qsmla_swa_range(
                ModeConfig::OriS2, Config::S1, work.q_token,
                ori_win_left, ori_win_right);
            ori_begin = range.begin;
            ori_count = range.end - range.begin;
        } else if constexpr (ModeConfig::HasIndexedOri) {
            const std::size_t list_offset =
                ((static_cast<std::size_t>(work.batch) * Config::S1
                  + work.q_token) * Config::N2 + work.kv_head)
                * ModeConfig::OriTopK;
            const std::size_t length_offset =
                (static_cast<std::size_t>(work.batch) * Config::S1
                 + work.q_token) * Config::N2 + work.kv_head;
            int candidate_count = ori_topk_length[length_offset];
            candidate_count = qsmla_sparse_clamp(
                candidate_count, 0, ModeConfig::OriTopK);
            ori_count = qsmla_sparse_collect_indices(
                ori_selected, ModeConfig::OriTopK,
                ori_sparse_indices + list_offset, candidate_count,
                ModeConfig::OriS2,
                qsmla_sparse_ori_valid_end(
                    ModeConfig::OriS2, Config::S1, work.q_token));
        }

        int cmp_count = 0;
        if constexpr (ModeConfig::HasCmp) {
            const int cmp_valid_end = qsmla_csa_cmp_valid_end(
                ModeConfig::CmpS2, Config::S1,
                work.q_token, cmp_ratio);
            if constexpr (ModeConfig::Mode == QsmlaMode::HCA) {
                // HCA is continuous compressed attention. Sparse CMP inputs
                // are intentionally ignored; [0,cmp_valid_end) participates.
                cmp_count = cmp_valid_end;
            } else {
                const std::size_t list_offset =
                    ((static_cast<std::size_t>(work.batch) * Config::S1
                      + work.q_token) * Config::N2 + work.kv_head)
                    * ModeConfig::CmpTopK;
                const std::size_t length_offset =
                    (static_cast<std::size_t>(work.batch) * Config::S1
                     + work.q_token) * Config::N2 + work.kv_head;
                int candidate_count = ModeConfig::CmpTopK;
                if (cmp_topk_length != nullptr) {
                    candidate_count = cmp_topk_length[length_offset];
                }
                candidate_count = qsmla_sparse_clamp(
                    candidate_count, 0, ModeConfig::CmpTopK);
                cmp_count = qsmla_sparse_collect_indices(
                    cmp_selected, ModeConfig::CmpTopK,
                    cmp_sparse_indices + list_offset, candidate_count,
                    ModeConfig::CmpS2, cmp_valid_end);
            }
        }

        auto stage_source_tile = [&] (
            kvdtype* source, const int* selected, int range_begin,
            int logical_begin, int valid_rows) {
            for (int row = 0; row < kTk; ++row) {
                const int source_row = row < valid_rows
                    ? (selected == nullptr
                           ? range_begin + logical_begin + row
                           : selected[logical_begin + row])
                    : 0;
                if constexpr (std::is_same_v<kvdtype, __hif8>) {
                    // The LinxV5 backend cannot legalize divergent scalar
                    // i8/HIF8 copies. Preserve the raw HIF8 payload and gather
                    // four elements per uint32 carrier instead.
                    auto* destination = reinterpret_cast<uint32_t*>(
                        kv_tile_buf + row * Config::D);
                    auto* source_words = reinterpret_cast<const uint32_t*>(
                        source + source_row * Config::D);
                    for (int word = 0; word < Config::D / 4; ++word) {
                        destination[word] =
                            row < valid_rows ? source_words[word] : 0U;
                    }
                } else {
                    for (int dim = 0; dim < Config::D; ++dim) {
                        kv_tile_buf[row * Config::D + dim] =
                            row < valid_rows
                                ? source[source_row * Config::D + dim]
                                : static_cast<kvdtype>(0.0f);
                    }
                }
            }
        };
        auto build_source_mask = [&](int valid_rows) {
            for (int row = 0; row < kPeRows; ++row) {
                for (int column = 0; column < kTk; ++column) {
                    mask_buf[row * kTk + column] =
                        column < valid_rows ? 0.0f : -1.0e30f;
                }
            }
        };

        tileMax tMax;
        tileSum tSum;
        TEXPANDS(tMax, -1e30f);
        TEXPANDS(tSum, 0.0f);
        auto gMaxState = gIterMax(0, 0);
        auto gSumState = gIterSum(0, 0);
        // Scalar indexed gather makes implicit Tile spills compiler-dependent.
        // Persist the one shared m/l state explicitly between source blocks;
        // gScore may overwrite these slots only after both states are loaded.
        TSTORE(gMaxState, tMax);
        TSTORE(gSumState, tSum);

        auto visit_source_pass1 = [&](kvdtype* source, const int* selected,
                                      int range_begin, int logical_count,
                                      int allow_direct, float kv_descale) {
            const float score_scale =
                softmax_scale * q_descale * kv_descale;
            const int block_count = (logical_count + kTk - 1) / kTk;
            for (int block = 0; block < block_count; ++block) {
                const int logical_begin = block * kTk;
                const int remaining = logical_count - logical_begin;
                const int valid_rows = remaining < kTk ? remaining : kTk;
                const bool direct_contiguous =
                    qsmla_use_direct_contiguous_tile(
                        allow_direct, selected != nullptr, valid_rows, kTk);
                kvdtype* tile_ptr = kv_tile_buf;
                if (direct_contiguous) {
                    // A full contiguous ORI/HCA tile can be consumed from GM
                    // directly. Indexed tiles and the non-integral tail keep
                    // using the bounded gather buffer to avoid over-reading.
                    tile_ptr = source +
                        (range_begin + logical_begin) * Config::D;
                } else {
                    stage_source_tile(source, selected, range_begin,
                                      logical_begin, valid_rows);
                }
                build_source_mask(valid_rows);
                itKSrc gIterKSrc(tile_ptr);

                tileMax tMax;
                tileSum tSum;
                TLOAD(tMax, gMaxState);
                TLOAD(tSum, gSumState);

                tileScoreCube tScoreCube;
#pragma clang loop unroll(full)
                for (int dd = 0; dd < kDb; ++dd) {
                    tileQShared tQShared;
                    tileKSrc tKSrc;
                    tileKMatrix tKLocal;
                    tileKShared tKShared;
                    auto gQ = gIterQ(0, dd);
                    auto gK = gIterKSrc(0, dd);
                    TLOAD<tileQMatrix, 1>(tQShared, gQ);
                    TLOAD(tKSrc, gK);
                    TTRANS(tKLocal, tKSrc);
                    TMOV_L2S_PUBLISH(tKShared, tKLocal);
                    if (dd == 0) {
                        TMATMUL(tScoreCube, tQShared, tKShared,
                                fixp::keep_acc());
                    } else {
                        TMATMUL_ACC(tScoreCube, tScoreCube,
                                    tQShared, tKShared, fixp::keep_acc());
                    }
                }

                TSTORE_CUBE(gScore, tScoreCube);
                tileW tW;
                TLOAD(tW, gScore);
                TMULS(tW, tW, score_scale);
                using gmMask = global_tensor<float, RowMajor<kPeRows, kTk>>;
                using itMask = global_iterator<gmMask, tileMask>;
                itMask gIterMask(mask_buf);
                tileMask tMask;
                auto gMask = gIterMask(0, 0);
                TLOAD(tMask, gMask);
                TADD(tW, tW, tMask);

                tileMax tLocalMax;
                tileMax tNewMax;
                TROWMAX(tLocalMax, tW);
                TMAX(tNewMax, tMax, tLocalMax);
                tileMax tScale;
                TSUB(tScale, tMax, tNewMax);
                TEXP(tScale, tScale);
                tileSum tScaledOldSum;
                TMUL(tScaledOldSum, tSum, tScale);
                TROWEXPANDSUB(tW, tW, tNewMax);
                TEXP(tW, tW);
                tileSum tLocalSum;
                TROWSUM(tLocalSum, tW);
                TADD(tSum, tScaledOldSum, tLocalSum);
                tMax = tNewMax;
                TSTORE(gMaxState, tMax);
                TSTORE(gSumState, tSum);
            }
        };

        // visit_ori_pass1: ORI is always the first logical source.
        visit_source_pass1(
            work_ori,
            ModeConfig::HasIndexedOri ? ori_selected : nullptr,
            ori_begin, ori_count, true, ori_kv_descale);
        // visit_cmp_pass1: CMP continues the same tMax/tSum state.
        if constexpr (ModeConfig::HasCmp) {
            visit_source_pass1(
                work_cmp,
                ModeConfig::HasIndexedCmp ? cmp_selected : nullptr,
                0, cmp_count, false, cmp_kv_descale);
        }

        tileSum tFinalSum;
        TLOAD(tFinalSum, gSumState);
        tileSum tInvSum;
        TRECIP(tInvSum, tFinalSum);
        TSTORE(gSumState, tInvSum);

        // Keep the complete FP32 O state in GM. Only one [PeRows,Td] tile is
        // live locally, so loop interchange does not recreate one-pass tile
        // pressure.
#pragma clang loop unroll(disable)
        for (int out_dd = 0; out_dd < kDb; ++out_dd) {
            tileO tZeroO;
            TEXPANDS(tZeroO, 0.0f);
            auto gOState = gIterPV(0, out_dd);
            TSTORE(gOState, tZeroO);
        }

        auto visit_source_pass2 = [&](kvdtype* source,
                                      const int* selected,
                                      int range_begin,
                                      int logical_count,
                                      int allow_direct,
                                      float kv_descale) {
            const float score_scale =
                softmax_scale * q_descale * kv_descale;
            const int block_count = (logical_count + kTk - 1) / kTk;
            for (int block = 0; block < block_count; ++block) {
                const int logical_begin = block * kTk;
                const int remaining = logical_count - logical_begin;
                const int valid_rows = remaining < kTk ? remaining : kTk;
                const bool direct_contiguous =
                    qsmla_use_direct_contiguous_tile(
                        allow_direct, selected != nullptr,
                        valid_rows, kTk);
                kvdtype* tile_ptr = kv_tile_buf;
                if (direct_contiguous) {
                    tile_ptr = source +
                        (range_begin + logical_begin) * Config::D;
                } else {
                    stage_source_tile(source, selected, range_begin,
                                      logical_begin, valid_rows);
                }
                build_source_mask(valid_rows);
                itKSrc gIterKSrc(tile_ptr);
                itV gIterV(tile_ptr);

                tileMax tMax;
                tileSum tInvSum;
                TLOAD(tMax, gMaxState);
                TLOAD(tInvSum, gSumState);

                tileScoreCube tScoreCube;
#pragma clang loop unroll(full)
                for (int dd = 0; dd < kDb; ++dd) {
                    tileQShared tQShared;
                    tileKSrc tKSrc;
                    tileKMatrix tKLocal;
                    tileKShared tKShared;
                    auto gQ = gIterQ(0, dd);
                    auto gK = gIterKSrc(0, dd);
                    TLOAD<tileQMatrix, 1>(tQShared, gQ);
                    TLOAD(tKSrc, gK);
                    TTRANS(tKLocal, tKSrc);
                    TMOV_L2S_PUBLISH(tKShared, tKLocal);
                    if (dd == 0) {
                        TMATMUL(tScoreCube, tQShared, tKShared,
                                fixp::keep_acc());
                    } else {
                        TMATMUL_ACC(tScoreCube, tScoreCube,
                                    tQShared, tKShared,
                                    fixp::keep_acc());
                    }
                }

                TSTORE_CUBE(gScore, tScoreCube);
                tileW tW;
                TLOAD(tW, gScore);
                TMULS(tW, tW, score_scale);
                using gmMask =
                    global_tensor<float, RowMajor<kPeRows, kTk>>;
                using itMask = global_iterator<gmMask, tileMask>;
                itMask gIterMask(mask_buf);
                tileMask tMask;
                auto gMask = gIterMask(0, 0);
                TLOAD(tMask, gMask);
                TADD(tW, tW, tMask);
                TROWEXPANDSUB(tW, tW, tMax);
                TEXP(tW, tW);
                TROWEXPANDMUL(tW, tW, tInvSum);
                if constexpr (kUseHif8Probability) {
                    // The original QSMLA contract quantizes P*16 to HIF8,
                    // performs the second CUBE matmul, then removes 16 once
                    // at final output. Online l remains the unquantized sum.
                    TMULS(tW, tW, kHif8ProbabilityScale);
                }

                tilePShard tPShard;
                TCVT(tPShard, tW);
                auto gProbShard = gIterProb(pe_id, 0);
                TSTORE(gProbShard, tPShard);

                // gScore aliases the row-state backing store. Restore it once
                // after probability generation, before the next source block.
                TSTORE(gMaxState, tMax);
                TSTORE(gSumState, tInvSum);

#pragma clang loop unroll(disable)
                for (int out_dd = 0; out_dd < kDb; ++out_dd) {
                    auto gOState = gIterPV(0, out_dd);
                    tileO tO;
                    TLOAD(tO, gOState);
                    // Shared CUBE operands must not remain live across a
                    // runtime loop: reload the already-published P payload
                    // into a short-lived tile for each D block.
                    tilePShared tPShared;
                    auto gP = gIterP(0, 0);
                    TLOAD<tilePMatrix, 1>(tPShared, gP);
                    tileVShared tVShared;
                    auto gV = gIterV(0, out_dd);
                    TLOAD<tileVMatrix, 1>(tVShared, gV);
                    tilePVCube tPVCube;
                    TMATMUL(tPVCube, tPShared, tVShared,
                            fixp::keep_acc());
                    TSTORE_CUBE(gOState, tPVCube);
                    tileO tPV;
                    TLOAD(tPV, gOState);
                    TMULS(tPV, tPV, kv_descale);
                    TADD(tO, tO, tPV);
                    TSTORE(gOState, tO);
                }
            }
        };

        // visit_ori_pass2: preserve the pass-1 ORI then CMP source order.
        visit_source_pass2(
            work_ori,
            ModeConfig::HasIndexedOri ? ori_selected : nullptr,
            ori_begin, ori_count, true, ori_kv_descale);
        if constexpr (ModeConfig::HasCmp) {
            // visit_cmp_pass2: continue into the same FP32 O scratch.
            visit_source_pass2(
                work_cmp,
                ModeConfig::HasIndexedCmp ? cmp_selected : nullptr,
                0, cmp_count, false, cmp_kv_descale);
        }

#pragma clang loop unroll(disable)
        for (int out_dd = 0; out_dd < kDb; ++out_dd) {
            auto gOState = gIterPV(0, out_dd);
            tileO tFinalO;
            TLOAD(tFinalO, gOState);
            if constexpr (kUseHif8Probability) {
                TMULS(tFinalO, tFinalO,
                      1.0f / kHif8ProbabilityScale);
            }
            tileOCast tOCast;
            TCVT(tOCast, tFinalO);
            itO gIterO(out_ptr + work_out_offset
                       + pe_id * kPeRows * Config::D);
            auto gO = gIterO(0, out_dd);
            TSTORE(gO, tOCast);
        }
    }
}

#endif // QUANT_SPARSE_FLASH_MLA_PTO_HPP
