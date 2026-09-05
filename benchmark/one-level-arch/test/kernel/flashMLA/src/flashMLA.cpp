#include <common/pto_tileop.hpp>
#include "benchmark.h"
#include "fileop.h"

#include <cstdint>
#include <cstdio>

using namespace pto;

#ifdef FLASHMLA_DEBUG_TILE
template <typename tile_shape>
void flashmla_dump_tile_head(const char* label, tile_shape& tile, int max_rows = 4, int max_cols = 8) {
    static constexpr size_t tile_size = tile_shape::Rows * tile_shape::Cols;
    typename tile_shape::DType data[tile_size] = {0};
    using dtype = typename tile_shape::DType;
    using shape = Shape<1, 1, 1, 1, 1>;
    using stride = std::conditional_t<
        tile_shape::isRowMajor || tile_shape::isBoxedLayout,
        Stride<1, 1, tile_shape::Rows * tile_shape::Cols, tile_shape::Cols, 1>,
        Stride<1, 1, tile_shape::Rows * tile_shape::Cols, 1, tile_shape::Rows>>;
    using gm_shape = std::conditional_t<
        tile_shape::isRowMajor || tile_shape::isBoxedLayout,
        GlobalTensor<dtype, shape, stride, Layout::ND>,
        GlobalTensor<dtype, shape, stride, Layout::DN>>;

    gm_shape dst(data);
    TCOPYOUT(dst, tile);

    const int rows = tile_shape::Rows < max_rows ? tile_shape::Rows : max_rows;
    const int cols = tile_shape::Cols < max_cols ? tile_shape::Cols : max_cols;
    printf("\n[FLASHMLA_DEBUG] %s rows=%d cols=%d valid=%d,%d\n",
           label, tile_shape::Rows, tile_shape::Cols, tile.GetValidRow(), tile.GetValidCol());
    for (int i = 0; i < rows; ++i) {
        printf("  r%02d:", i);
        for (int j = 0; j < cols; ++j) {
            const int offset = (tile_shape::isRowMajor || tile_shape::isBoxedLayout)
                ? i * tile_shape::Cols + j
                : j * tile_shape::Rows + i;
            printf(" %.6f", static_cast<float>(data[offset]));
        }
        printf("\n");
    }
    fflush(stdout);
}

#define FLASHMLA_DUMP_TILE(label, tile_var) flashmla_dump_tile_head(label, tile_var)
#else
#define FLASHMLA_DUMP_TILE(label, tile_var) \
    do { \
    } while (0)
#endif

#define B 1
#define H_K 1

#ifndef Tsq
#define Sq 64
#else
#define Sq Tsq
#endif

#ifndef QHeadPerHK
#define kQHeadPerHK 1
#else
#define kQHeadPerHK QHeadPerHK
#endif

#ifndef NumBlocks
#define kNumBlocks 2
#else
#define kNumBlocks NumBlocks
#endif

#ifndef MaxBlocksPerSeq
#define kMaxBlocksPerSeq 2
#else
#define kMaxBlocksPerSeq MaxBlocksPerSeq
#endif

#ifndef Dk
#define kDk 512
#else
#define kDk Dk
#endif

#ifndef Dv
#define kDv 512
#else
#define kDv Dv
#endif

#ifndef DChunk
#define kDChunk 128
#else
#define kDChunk DChunk
#endif

#ifndef VChunk
#define kVChunk 128
#else
#define kVChunk VChunk
#endif

#ifndef Tm
#define kTm 16
#else
#define kTm Tm
#endif

#ifndef Tk
#define kTk 16
#else
#define kTk Tk
#endif

#ifndef PageBlockSize
#define kPageBlockSize 64
#else
#define kPageBlockSize PageBlockSize
#endif

#define ALIGN_MASK 0xfffffffffffff000ull
#define ALIGN 4 * 1024

// PTO tileop model of FlashMLA dense_decode_fwd.
//
// FlashMLA dense decode API reshapes Q into:
//   Q: [batch, q_seq_per_hk, h_k, d_k]
// where q_seq_per_hk = seqlen_q * (num_q_heads / num_kv_heads).
//
// This standalone tileop model operates on one batch and one KV head:
//   Q        : [q_seq_per_hk, d_k]
//   KV cache : [num_blocks * page_block_size, d_k]
//   O        : [q_seq_per_hk, d_v]
//
// The original CUDA kernel uses TMA, warpgroup pipelining, splitKV scheduling
// and a combine kernel. Here we keep the dense decode math and paged KV lookup:
//   score = Q * K^T * softmax_scale
//   O     = softmax(score) * V
//
// To fit the local PTO tile constraints, the large FlashMLA dimensions are
// chunked explicitly:
//   QK: reduce d_k through DChunk-sized TMATMUL chunks and ordinary Tile adds.
//   PV: write d_v through VChunk-sized output chunks.
//
// Dense MLA stores K/V in one dense cache. This model treats V as the first
// d_v columns of the selected KV-cache row. Current examples do not expose a
// TLOG tileop, so lse_ptr stores the final softmax denominator l rather than
// log(l) + m. The output O is normalized.
template <
    typename dtype,
    int QSeqPerHK,
    int NumBlocks_,
    int MaxBlocksPerSeq_,
    int Dk_,
    int Dv_,
    int DChunk_,
    int VChunk_,
    int Tm_,
    int Tk_,
    int PageBlockSize_ = 64>
void flash_mla_dense_decode_tileop(
    dtype* out_ptr,
    float* lse_ptr,
    dtype* q_ptr,
    dtype* kv_cache_ptr,
    int* seqlen_k_ptr,
    int* block_table_ptr,
    float softmax_scale
#ifdef FLASHMLA_DEBUG_BIN
    ,
    float* dbg_score_ptr,
    float* dbg_exp_ptr,
    float* dbg_sum_ptr,
    float* dbg_prob_ptr,
    dtype* dbg_v_ptr,
    float* dbg_pv_ptr,
    float* dbg_o_ptr,
    float* dbg_raw_score_sub_ptr,
    float* dbg_score_sub_ptr,
    float* dbg_prob_sub_ptr,
    dtype* dbg_k_sub_ptr,
    dtype* dbg_v_sub_ptr,
    float* dbg_pv_sub_ptr,
    float* dbg_o_sub_ptr
#endif
) {
    static_assert(QSeqPerHK % Tm_ == 0, "QSeqPerHK must be divisible by Tm");
    static_assert(PageBlockSize_ % Tk_ == 0, "PageBlockSize must be divisible by Tk");
    static_assert(Dk_ % DChunk_ == 0, "Dk must be divisible by DChunk");
    static_assert(Dv_ % VChunk_ == 0, "Dv must be divisible by VChunk");
    static_assert(Dv_ <= Dk_, "dense MLA V is read from the first Dv columns of KV cache");
    static_assert(Tm_ * DChunk_ * sizeof(dtype) <= 8 * 1024, "Q chunk tile exceeds 8KB");
    static_assert(Tk_ * DChunk_ * sizeof(dtype) <= 8 * 1024, "K chunk tile exceeds 8KB");
    static_assert(Tk_ * VChunk_ * sizeof(dtype) <= 8 * 1024, "V chunk tile exceeds 8KB");
    static_assert(Tm_ * VChunk_ * sizeof(float) <= 8 * 1024, "O chunk tile exceeds 8KB");

    using gmQ = global_tensor<dtype, RowMajor<QSeqPerHK, Dk_>>;
    using gmKV = global_tensor<dtype, RowMajor<NumBlocks_ * PageBlockSize_, Dk_>>;
    using gmKView = global_tensor<dtype, MatrixLayout<Dk_, NumBlocks_ * PageBlockSize_, 1, Dk_>>;
    using gmO = global_tensor<dtype, RowMajor<QSeqPerHK, Dv_>>;
    using gmLSE = global_tensor<float, RowMajor<QSeqPerHK, 1>>;

    using tileQ = std::conditional_t<
        (Tm_ <= 16), CubeTileM16<dtype, Tm_, DChunk_>,
        CubeTileM32<dtype, Tm_, DChunk_>>;
    using tileK = CubeTileN8<dtype, DChunk_, Tk_>;
    using tileV = CubeTileN8<dtype, Tk_, VChunk_>;
    using tileScoreCube = std::conditional_t<
        (Tm_ <= 16), CubeAccumulatorM16<float, Tm_, Tk_>,
        CubeAccumulatorM32<float, Tm_, Tk_>>;
    using tilePVCube = std::conditional_t<
        (Tm_ <= 16), CubeAccumulatorM16<float, Tm_, VChunk_>,
        CubeAccumulatorM32<float, Tm_, VChunk_>>;

    using tileScore = Tile<Location::Vec, float, Tm_, Tk_, BLayout::ColMajor>;
    using tileScoreCast = Tile<Location::Vec, dtype, Tm_, Tk_, BLayout::ColMajor>;
    using tileScoreLeft = std::conditional_t<
        (Tm_ <= 16), CubeTileM16<dtype, Tm_, Tk_>,
        CubeTileM32<dtype, Tm_, Tk_>>;

    using tileO = Tile<Location::Vec, float, Tm_, VChunk_, BLayout::ColMajor>;
    using tileOCast = Tile<Location::Vec, dtype, Tm_, VChunk_, BLayout::ColMajor>;

    using tileMax = Tile<Location::Vec, float, Tm_, 1, BLayout::ColMajor, Tm_, 1>;
    using tileSum = Tile<Location::Vec, float, Tm_, 1, BLayout::ColMajor, Tm_, 1>;
    using tileScale = Tile<Location::Vec, float, Tm_, 1, BLayout::ColMajor, Tm_, 1>;

    using itQ = global_iterator<gmQ, tileQ>;
    using itK = global_iterator<gmKView, tileK>;
    using itV = global_iterator<gmKV, tileV>;
    using itO = global_iterator<gmO, tileO>;
    using itLSE = global_iterator<gmLSE, tileSum>;

    itQ gQIter(q_ptr);
    itK gKIter(kv_cache_ptr);
    itV gVIter(kv_cache_ptr);
    itO gOIter(out_ptr);
    itLSE gLSEIter(lse_ptr);

    float score_cube_scratch[Tm_ * Tk_];
    dtype prob_cube_scratch[Tm_ * Tk_];
    float pv_cube_scratch[Tm_ * VChunk_];
    global_tensor<float, RowMajor<Tm_, Tk_>> gScoreCubeScratch(
        score_cube_scratch);
    global_tensor<dtype, RowMajor<Tm_, Tk_>> gProbCubeScratch(
        prob_cube_scratch);
    global_tensor<float, RowMajor<Tm_, VChunk_>> gPVCubeScratch(
        pv_cube_scratch);

#ifdef FLASHMLA_DEBUG_BIN
    using gmDbgScore = global_tensor<float, RowMajor<Tm_, Tk_>>;
    using gmDbgSum = global_tensor<float, RowMajor<Tm_, 8>>;
    using gmDbgV = global_tensor<dtype, RowMajor<Tk_, VChunk_>>;
    using gmDbgO = global_tensor<float, RowMajor<Tm_, VChunk_>>;
    using gmDbgStage = global_tensor<float, RowMajor<(PageBlockSize_ / Tk_) * Tm_, VChunk_>>;
    using gmDbgProbStage = global_tensor<float, RowMajor<(PageBlockSize_ / Tk_) * Tm_, Tk_>>;
    using gmDbgKStage = global_tensor<dtype, RowMajor<(PageBlockSize_ / Tk_) * (Dk_ / DChunk_) * DChunk_, Tk_>>;
    using gmDbgVStage = global_tensor<dtype, RowMajor<(PageBlockSize_ / Tk_) * Tk_, VChunk_>>;
    using itDbgScore = global_iterator<gmDbgScore, tileScore>;
    using itDbgSum = global_iterator<gmDbgSum, tileSum>;
    using itDbgV = global_iterator<gmDbgV, tileV>;
    using itDbgO = global_iterator<gmDbgO, tileO>;
    using itDbgStage = global_iterator<gmDbgStage, tileO>;
    using itDbgProbStage = global_iterator<gmDbgProbStage, tileScore>;
    using itDbgKStage = global_iterator<gmDbgKStage, tileK>;
    using itDbgVStage = global_iterator<gmDbgVStage, tileV>;
    itDbgScore gDbgScore(dbg_score_ptr);
    itDbgScore gDbgExp(dbg_exp_ptr);
    itDbgSum gDbgSum(dbg_sum_ptr);
    itDbgScore gDbgProb(dbg_prob_ptr);
    itDbgV gDbgV(dbg_v_ptr);
    itDbgO gDbgPV(dbg_pv_ptr);
    itDbgO gDbgO(dbg_o_ptr);
    itDbgProbStage gDbgRawScoreSub(dbg_raw_score_sub_ptr);
    itDbgProbStage gDbgScoreSub(dbg_score_sub_ptr);
    itDbgProbStage gDbgProbSub(dbg_prob_sub_ptr);
    itDbgKStage gDbgKSub(dbg_k_sub_ptr);
    itDbgVStage gDbgVSub(dbg_v_sub_ptr);
    itDbgStage gDbgPVSub(dbg_pv_sub_ptr);
    itDbgStage gDbgOSub(dbg_o_sub_ptr);
#endif

    constexpr int Qb = QSeqPerHK / Tm_;
    constexpr int Db = Dk_ / DChunk_;
    constexpr int Vb = Dv_ / VChunk_;
    constexpr int subBlocksPerPage = PageBlockSize_ / Tk_;

    const int seqlen_k = seqlen_k_ptr[0];
    const int logical_pages = (seqlen_k + PageBlockSize_ - 1) / PageBlockSize_;

    for (int q_block = 0; q_block < Qb; ++q_block) {
        tileMax tMax;
        tileSum tSum;
        TEXPANDS(tMax, -1e30f);
        TEXPANDS(tSum, 0.0f);

        // Pass 1: compute final row max and denominator over paged KV.
        for (int logical_page = 0; logical_page < logical_pages; ++logical_page) {
            const int physical_page = block_table_ptr[logical_page];

            #pragma clang loop unroll(full)
            for (int sub = 0; sub < subBlocksPerPage; ++sub) {
                const int kv_tile_row = physical_page * subBlocksPerPage + sub;

                tileQ tQ0;
                tileK tK0;
                tileScore tScore;
                auto gQ0 = gQIter(q_block, 0);
                auto gK0 = gKIter(0, kv_tile_row);
                TLOAD_CUBE(tQ0, gQ0);
                TLOAD_CUBE(tK0, gK0);
                if (q_block == 0 && logical_page == 0 && sub == 0) {
                    FLASHMLA_DUMP_TILE("pass1/tQ q_block=0 d_block=0 shape=[Tm,DChunk]", tQ0);
                    FLASHMLA_DUMP_TILE("pass1/tK cube-right kv_tile=0 d_block=0 logical shape=[DChunk,Tk]", tK0);
                }
                tileScoreCube tScoreCube;
                TMATMUL(tScoreCube, tQ0, tK0);
                TSTORE_CUBE(gScoreCubeScratch, tScoreCube);
                TLOAD(tScore, gScoreCubeScratch);

                #pragma clang loop unroll(full)
                for (int d_block = 1; d_block < Db; ++d_block) {
                    tileQ tQ;
                    tileK tK;
                    tileScore tPartial;
                    auto gQ = gQIter(q_block, d_block);
                    auto gK = gKIter(d_block, kv_tile_row);
                    TLOAD_CUBE(tQ, gQ);
                    TLOAD_CUBE(tK, gK);
                    tileScoreCube tPartialCube;
                    TMATMUL(tPartialCube, tQ, tK);
                    TSTORE_CUBE(gScoreCubeScratch, tPartialCube);
                    TLOAD(tPartial, gScoreCubeScratch);
                    TADD(tScore, tScore, tPartial);
                }

                TMULS(tScore, tScore, softmax_scale);
                if (q_block == 0 && logical_page == 0 && sub == 0) {
                    FLASHMLA_DUMP_TILE("pass1/score after QK*scale shape=[Tm,Tk]", tScore);
#ifdef FLASHMLA_DEBUG_BIN
                    auto dbgScore = gDbgScore(0, 0);
                    TSTORE(dbgScore, tScore);
#endif
                }

                tileMax tLocalMax;
                tileMax tNewMax;
                tileScale tScale;
                tileSum tScaledOldSum;
                tileSum tLocalSum;
                tileSum tNewSum;

                TROWMAX(tLocalMax, tScore);
                TMAX(tNewMax, tMax, tLocalMax);
                TSUB(tScale, tMax, tNewMax);
                TEXP(tScale, tScale);
                TMUL(tScaledOldSum, tSum, tScale);

                TROWEXPANDSUB(tScore, tScore, tNewMax);
                TEXP(tScore, tScore);
                TROWSUM(tLocalSum, tScore);
                TADD(tNewSum, tScaledOldSum, tLocalSum);
                if (q_block == 0 && logical_page == 0 && sub == 0) {
                    FLASHMLA_DUMP_TILE("pass1/local rowmax shape=[Tm,1]", tLocalMax);
                    FLASHMLA_DUMP_TILE("pass1/exp(score-rowmax) shape=[Tm,Tk]", tScore);
                    FLASHMLA_DUMP_TILE("pass1/local rowsum shape=[Tm,1]", tLocalSum);
                    FLASHMLA_DUMP_TILE("pass1/new sum after first sub shape=[Tm,1]", tNewSum);
#ifdef FLASHMLA_DEBUG_BIN
                    auto dbgExp = gDbgExp(0, 0);
                    auto dbgSum = gDbgSum(0, 0);
                    TSTORE(dbgExp, tScore);
                    TSTORE(dbgSum, tNewSum);
#endif
                }

                tMax = tNewMax;
                tSum = tNewSum;
            }
        }

        if (lse_ptr != nullptr) {
            if (q_block == 0) {
                FLASHMLA_DUMP_TILE("pass1/final denominator tSum before lse store shape=[Tm,1]", tSum);
#ifdef FLASHMLA_DEBUG_BIN
                auto dbgFinalSum = gDbgSum(0, 0);
                TSTORE(dbgFinalSum, tSum);
#endif
            }
            auto gLSE = gLSEIter(q_block, 0);
            TSTORE(gLSE, tSum);
        }

        tileScale tInvSum;
        TRECIP(tInvSum, tSum);

        // Pass 2: recompute probabilities and accumulate output per V chunk.
        #pragma clang loop unroll(full)
        for (int v_block = 0; v_block < Vb; ++v_block) {
            tileO tO;
            TEXPANDS(tO, 0.0f);

            for (int logical_page = 0; logical_page < logical_pages; ++logical_page) {
                const int physical_page = block_table_ptr[logical_page];

                #pragma clang loop unroll(full)
                for (int sub = 0; sub < subBlocksPerPage; ++sub) {
                    const int kv_tile_row = physical_page * subBlocksPerPage + sub;

                    tileQ tQ0;
                    tileK tK0;
                    tileScore tScore;
                    auto gQ0 = gQIter(q_block, 0);
                    auto gK0 = gKIter(0, kv_tile_row);
                    TLOAD_CUBE(tQ0, gQ0);
                    TLOAD_CUBE(tK0, gK0);
                    if (q_block == 0 && v_block == 0 && logical_page == 0 && sub == 0) {
                        FLASHMLA_DUMP_TILE("pass2/tQ q_block=0 d_block=0 shape=[Tm,DChunk]", tQ0);
                        FLASHMLA_DUMP_TILE("pass2/tK cube-right kv_tile=0 d_block=0 logical shape=[DChunk,Tk]", tK0);
                    }
#ifdef FLASHMLA_DEBUG_BIN
                    if (q_block == 0 && v_block == 0 && logical_page == 0) {
                        auto dbgKSub = gDbgKSub(sub * Db, 0);
                        TSTORE(dbgKSub, tK0);
                    }
#endif
                    tileScoreCube tScoreCube;
                    TMATMUL(tScoreCube, tQ0, tK0);
                    TSTORE_CUBE(gScoreCubeScratch, tScoreCube);
                    TLOAD(tScore, gScoreCubeScratch);

                    #pragma clang loop unroll(full)
                    for (int d_block = 1; d_block < Db; ++d_block) {
                        tileQ tQ;
                        tileK tK;
                        tileScore tPartial;
                        auto gQ = gQIter(q_block, d_block);
                        auto gK = gKIter(d_block, kv_tile_row);
                        TLOAD_CUBE(tQ, gQ);
                        TLOAD_CUBE(tK, gK);
#ifdef FLASHMLA_DEBUG_BIN
                        if (q_block == 0 && v_block == 0 && logical_page == 0) {
                            auto dbgKSub = gDbgKSub(sub * Db + d_block, 0);
                            TSTORE(dbgKSub, tK);
                        }
#endif
                        tileScoreCube tPartialCube;
                        TMATMUL(tPartialCube, tQ, tK);
                        TSTORE_CUBE(gScoreCubeScratch, tPartialCube);
                        TLOAD(tPartial, gScoreCubeScratch);
                        TADD(tScore, tScore, tPartial);
                    }

#ifdef FLASHMLA_DEBUG_BIN
                    if (q_block == 0 && v_block == 0 && logical_page == 0) {
                        auto dbgRawScoreSub = gDbgRawScoreSub(sub, 0);
                        TSTORE(dbgRawScoreSub, tScore);
                    }
#endif
                    TMULS(tScore, tScore, softmax_scale);
#ifdef FLASHMLA_DEBUG_BIN
                    if (q_block == 0 && v_block == 0 && logical_page == 0) {
                        auto dbgScoreSub = gDbgScoreSub(sub, 0);
                        TSTORE(dbgScoreSub, tScore);
                    }
#endif
                    TROWEXPANDSUB(tScore, tScore, tMax);
                    TEXP(tScore, tScore);
                    TROWEXPANDMUL(tScore, tScore, tInvSum);
                    if (q_block == 0 && v_block == 0 && logical_page == 0 && sub == 0) {
                        FLASHMLA_DUMP_TILE("pass2/prob after normalize shape=[Tm,Tk]", tScore);
#ifdef FLASHMLA_DEBUG_BIN
                        auto dbgProb = gDbgProb(0, 0);
                        TSTORE(dbgProb, tScore);
#endif
                    }

                    tileScoreCast tProbCast;
                    tileScoreLeft tProbLeft;
                    TCVT(tProbCast, tScore);
                    TSTORE(gProbCubeScratch, tProbCast);
                    TLOAD_CUBE(tProbLeft, gProbCubeScratch);
                    if (q_block == 0 && v_block == 0 && logical_page == 0 && sub == 0) {
                        FLASHMLA_DUMP_TILE("pass2/prob cast to left shape=[Tm,Tk]", tProbLeft);
                    }

                    tileV tV;
                    auto gV = gVIter(kv_tile_row, v_block);
                    TLOAD_CUBE(tV, gV);
                    if (q_block == 0 && v_block == 0 && logical_page == 0 && sub == 0) {
                        FLASHMLA_DUMP_TILE("pass2/tV v_block=0 shape=[Tk,VChunk]", tV);
#ifdef FLASHMLA_DEBUG_BIN
                        auto dbgV = gDbgV(0, 0);
                        TSTORE(dbgV, tV);
#endif
                    }

                    tileO tPV;
                    tilePVCube tPVCube;
                    TMATMUL(tPVCube, tProbLeft, tV);
                    TSTORE_CUBE(gPVCubeScratch, tPVCube);
                    TLOAD(tPV, gPVCubeScratch);
                    TADD(tO, tO, tPV);
                    if (q_block == 0 && v_block == 0 && logical_page == 0 && sub == 0) {
                        FLASHMLA_DUMP_TILE("pass2/tPV first sub shape=[Tm,VChunk]", tPV);
                        FLASHMLA_DUMP_TILE("pass2/tO after first sub shape=[Tm,VChunk]", tO);
#ifdef FLASHMLA_DEBUG_BIN
                        auto dbgPV = gDbgPV(0, 0);
                        TSTORE(dbgPV, tPV);
#endif
                    }
#ifdef FLASHMLA_DEBUG_BIN
                    if (q_block == 0 && v_block == 0 && logical_page == 0) {
                        auto dbgProbSub = gDbgProbSub(sub, 0);
                        TSTORE(dbgProbSub, tScore);
                    }
                    if (q_block == 0 && v_block == 0 && logical_page == 0) {
                        auto dbgVSub = gDbgVSub(sub, 0);
                        auto dbgPVSub = gDbgPVSub(sub, 0);
                        auto dbgOSub = gDbgOSub(sub, 0);
                        TSTORE(dbgVSub, tV);
                        TSTORE(dbgPVSub, tPV);
                        TSTORE(dbgOSub, tO);
                    }
#endif
                }
            }

            tileOCast tOCast;
            TCVT(tOCast, tO);
            if (q_block == 0 && v_block == 0) {
                FLASHMLA_DUMP_TILE("pass2/final tO before store shape=[Tm,VChunk]", tO);
                FLASHMLA_DUMP_TILE("pass2/final tOCast before store shape=[Tm,VChunk]", tOCast);
#ifdef FLASHMLA_DEBUG_BIN
                auto dbgO = gDbgO(0, 0);
                TSTORE(dbgO, tO);
#endif
            }
            auto gO = gOIter(q_block, v_block);
            TSTORE(gO, tOCast);
        }
    }
}

int main() {
    using dtype = __half;
    constexpr int qSeqPerHK = Sq * kQHeadPerHK;

    dtype qp[B * H_K * qSeqPerHK * kDk + 2 * ALIGN];
    dtype kvp[B * H_K * kNumBlocks * kPageBlockSize * kDk + 2 * ALIGN];
    dtype outp[B * H_K * qSeqPerHK * kDv + 2 * ALIGN];
    float lsep[B * H_K * qSeqPerHK + 2 * ALIGN];
    int seqlens_k[B + 2 * ALIGN];
    int block_table[B * kMaxBlocksPerSeq + 2 * ALIGN];
#ifdef FLASHMLA_DEBUG_BIN
    float dbg_score[kTm * kTk + 2 * ALIGN];
    float dbg_exp[kTm * kTk + 2 * ALIGN];
    float dbg_sum[kTm * 8 + 2 * ALIGN];
    float dbg_prob[kTm * kTk + 2 * ALIGN];
    dtype dbg_v[kTk * kVChunk + 2 * ALIGN];
    float dbg_pv[kTm * kVChunk + 2 * ALIGN];
    float dbg_o[kTm * kVChunk + 2 * ALIGN];
    float dbg_raw_score_sub[(kPageBlockSize / kTk) * kTm * kTk + 2 * ALIGN];
    float dbg_score_sub[(kPageBlockSize / kTk) * kTm * kTk + 2 * ALIGN];
    float dbg_prob_sub[(kPageBlockSize / kTk) * kTm * kTk + 2 * ALIGN];
    dtype dbg_k_sub[(kPageBlockSize / kTk) * (kDk / kDChunk) * kDChunk * kTk + 2 * ALIGN];
    dtype dbg_v_sub[(kPageBlockSize / kTk) * kTk * kVChunk + 2 * ALIGN];
    float dbg_pv_sub[(kPageBlockSize / kTk) * kTm * kVChunk + 2 * ALIGN];
    float dbg_o_sub[(kPageBlockSize / kTk) * kTm * kVChunk + 2 * ALIGN];
#endif

    dtype* q = (dtype*)(((uint64_t)qp & ALIGN_MASK) + ALIGN);
    dtype* kv = (dtype*)(((uint64_t)kvp & ALIGN_MASK) + ALIGN);
    dtype* out = (dtype*)(((uint64_t)outp & ALIGN_MASK) + ALIGN);
    float* lse = (float*)(((uint64_t)lsep & ALIGN_MASK) + ALIGN);
#ifdef FLASHMLA_DEBUG_BIN
    float* dbg_score_p = (float*)(((uint64_t)dbg_score & ALIGN_MASK) + ALIGN);
    float* dbg_exp_p = (float*)(((uint64_t)dbg_exp & ALIGN_MASK) + ALIGN);
    float* dbg_sum_p = (float*)(((uint64_t)dbg_sum & ALIGN_MASK) + ALIGN);
    float* dbg_prob_p = (float*)(((uint64_t)dbg_prob & ALIGN_MASK) + ALIGN);
    dtype* dbg_v_p = (dtype*)(((uint64_t)dbg_v & ALIGN_MASK) + ALIGN);
    float* dbg_pv_p = (float*)(((uint64_t)dbg_pv & ALIGN_MASK) + ALIGN);
    float* dbg_o_p = (float*)(((uint64_t)dbg_o & ALIGN_MASK) + ALIGN);
    float* dbg_raw_score_sub_p = (float*)(((uint64_t)dbg_raw_score_sub & ALIGN_MASK) + ALIGN);
    float* dbg_score_sub_p = (float*)(((uint64_t)dbg_score_sub & ALIGN_MASK) + ALIGN);
    float* dbg_prob_sub_p = (float*)(((uint64_t)dbg_prob_sub & ALIGN_MASK) + ALIGN);
    dtype* dbg_k_sub_p = (dtype*)(((uint64_t)dbg_k_sub & ALIGN_MASK) + ALIGN);
    dtype* dbg_v_sub_p = (dtype*)(((uint64_t)dbg_v_sub & ALIGN_MASK) + ALIGN);
    float* dbg_pv_sub_p = (float*)(((uint64_t)dbg_pv_sub & ALIGN_MASK) + ALIGN);
    float* dbg_o_sub_p = (float*)(((uint64_t)dbg_o_sub & ALIGN_MASK) + ALIGN);
#endif

    seqlens_k[0] = kMaxBlocksPerSeq * kPageBlockSize;
    for (int i = 0; i < kMaxBlocksPerSeq; ++i) {
        block_table[i] = i;
    }

#ifdef RES_CHECK
#define SRCQ_PATH CHK_DIR "/srcq.bin"
#define SRCKV_PATH CHK_DIR "/srckv.bin"
    readBinaryFile(SRCQ_PATH, (uint8_t*)q, B * H_K * qSeqPerHK * kDk * sizeof(dtype));
    readBinaryFile(SRCKV_PATH, (uint8_t*)kv, B * H_K * kNumBlocks * kPageBlockSize * kDk * sizeof(dtype));
#endif

    const float softmax_scale = 1.0f / sqrt((float)kDk);

    BENCHSTART;
    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H_K; ++h) {
            const int q_offset = (b * H_K + h) * qSeqPerHK * kDk;
            const int kv_offset = (b * H_K + h) * kNumBlocks * kPageBlockSize * kDk;
            const int o_offset = (b * H_K + h) * qSeqPerHK * kDv;
            const int lse_offset = (b * H_K + h) * qSeqPerHK;

            flash_mla_dense_decode_tileop<
                dtype,
                qSeqPerHK,
                kNumBlocks,
                kMaxBlocksPerSeq,
                kDk,
                kDv,
                kDChunk,
                kVChunk,
                kTm,
                kTk,
                kPageBlockSize>(
                out + o_offset,
                lse + lse_offset,
                q + q_offset,
                kv + kv_offset,
                seqlens_k + b,
                block_table + b * kMaxBlocksPerSeq,
                softmax_scale
#ifdef FLASHMLA_DEBUG_BIN
                ,
                dbg_score_p,
                dbg_exp_p,
                dbg_sum_p,
                dbg_prob_p,
                dbg_v_p,
                dbg_pv_p,
                dbg_o_p,
                dbg_raw_score_sub_p,
                dbg_score_sub_p,
                dbg_prob_sub_p,
                dbg_k_sub_p,
                dbg_v_sub_p,
                dbg_pv_sub_p,
                dbg_o_sub_p
#endif
            );
        }
    }
    BENCHEND;

#ifdef RES_CHECK
#define RES_PATH CHK_DIR "/res.bin"
#define LSE_PATH CHK_DIR "/lse.bin"
    writeBinaryFile(RES_PATH, (uint8_t*)out, B * H_K * qSeqPerHK * kDv * sizeof(dtype));
    writeBinaryFile(LSE_PATH, (uint8_t*)lse, B * H_K * qSeqPerHK * sizeof(float));
#ifdef FLASHMLA_DEBUG_BIN
#define DBG_SCORE_PATH CHK_DIR "/dbg_score.bin"
#define DBG_EXP_PATH CHK_DIR "/dbg_exp.bin"
#define DBG_SUM_PATH CHK_DIR "/dbg_sum.bin"
#define DBG_PROB_PATH CHK_DIR "/dbg_prob.bin"
#define DBG_V_PATH CHK_DIR "/dbg_v.bin"
#define DBG_PV_PATH CHK_DIR "/dbg_pv.bin"
#define DBG_O_PATH CHK_DIR "/dbg_o.bin"
    writeBinaryFile(DBG_SCORE_PATH, (uint8_t*)dbg_score_p, kTm * kTk * sizeof(float));
    writeBinaryFile(DBG_EXP_PATH, (uint8_t*)dbg_exp_p, kTm * kTk * sizeof(float));
    writeBinaryFile(DBG_SUM_PATH, (uint8_t*)dbg_sum_p, kTm * 8 * sizeof(float));
    writeBinaryFile(DBG_PROB_PATH, (uint8_t*)dbg_prob_p, kTm * kTk * sizeof(float));
    writeBinaryFile(DBG_V_PATH, (uint8_t*)dbg_v_p, kTk * kVChunk * sizeof(dtype));
    writeBinaryFile(DBG_PV_PATH, (uint8_t*)dbg_pv_p, kTm * kVChunk * sizeof(float));
    writeBinaryFile(DBG_O_PATH, (uint8_t*)dbg_o_p, kTm * kVChunk * sizeof(float));
#define DBG_PV_SUB_PATH CHK_DIR "/dbg_pv_sub.bin"
#define DBG_O_SUB_PATH CHK_DIR "/dbg_o_sub.bin"
#define DBG_V_SUB_PATH CHK_DIR "/dbg_v_sub.bin"
#define DBG_RAW_SCORE_SUB_PATH CHK_DIR "/dbg_raw_score_sub.bin"
#define DBG_SCORE_SUB_PATH CHK_DIR "/dbg_score_sub.bin"
#define DBG_PROB_SUB_PATH CHK_DIR "/dbg_prob_sub.bin"
#define DBG_K_SUB_PATH CHK_DIR "/dbg_k_sub.bin"
    writeBinaryFile(DBG_RAW_SCORE_SUB_PATH, (uint8_t*)dbg_raw_score_sub_p, (kPageBlockSize / kTk) * kTm * kTk * sizeof(float));
    writeBinaryFile(DBG_SCORE_SUB_PATH, (uint8_t*)dbg_score_sub_p, (kPageBlockSize / kTk) * kTm * kTk * sizeof(float));
    writeBinaryFile(DBG_PROB_SUB_PATH, (uint8_t*)dbg_prob_sub_p, (kPageBlockSize / kTk) * kTm * kTk * sizeof(float));
    writeBinaryFile(DBG_K_SUB_PATH, (uint8_t*)dbg_k_sub_p, (kPageBlockSize / kTk) * (kDk / kDChunk) * kDChunk * kTk * sizeof(dtype));
    writeBinaryFile(DBG_V_SUB_PATH, (uint8_t*)dbg_v_sub_p, (kPageBlockSize / kTk) * kTk * kVChunk * sizeof(dtype));
    writeBinaryFile(DBG_PV_SUB_PATH, (uint8_t*)dbg_pv_sub_p, (kPageBlockSize / kTk) * kTm * kVChunk * sizeof(float));
    writeBinaryFile(DBG_O_SUB_PATH, (uint8_t*)dbg_o_sub_p, (kPageBlockSize / kTk) * kTm * kVChunk * sizeof(float));
#endif
#endif

    return 0;
}
