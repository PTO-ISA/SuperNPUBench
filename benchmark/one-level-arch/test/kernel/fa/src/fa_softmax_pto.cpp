#include <common/pto_tileop.hpp>
#include "benchmark.h"
#include "fileop.h"

#include <cstdint>

using namespace pto;

#define B 1
#define H 1

#ifndef Tsq
#define Sq 512
#else
#define Sq Tsq
#endif

#ifndef Tskv
#define Skv 512
#else
#define Skv Tskv
#endif

#ifndef Tm
#define kTm 32
#else
#define kTm Tm
#endif

#ifndef Tk
#define kTk 64
#else
#define kTk Tk
#endif

#define ALIGN_MASK 0xfffffffffffff000ull
#define ALIGN 4 * 1024

// Compute row-wise softmax for a precomputed QK score matrix.
//
// Input:
//   score: [Sq, Skv], row-major. It is assumed to be the final attention logit
//          matrix already, e.g. Q*K^T or scaled Q*K^T.
// Output:
//   out  : [Sq, Skv], row-major, softmax(score) along the Skv dimension.
//
// Tile shapes:
//   tScore   : [kTm, kTk], float, RowMajor, directly loaded/stored by TLSU.
//   tMax/tSum: [kTm, 1] valid shape, physically [kTm, 8] for tile alignment.
//
// The implementation is two-pass over K blocks:
//   pass 1: reduce row max and row sum over all K blocks.
//   pass 2: reload each score tile, normalize by the final row sum, and store.
template <int Sq_, int Skv_, int kTm_, int kTk_>
void softmax_qk_2d_unroll_pto(float* out_ptr, float* score_ptr) {
    static_assert(Sq_ % kTm_ == 0, "Sq must be divisible by kTm");
    static_assert(Skv_ % kTk_ == 0, "Skv must be divisible by kTk");
    static_assert(kTm_ * kTk_ * sizeof(float) <= 8 * 1024,
                  "float score tile must not exceed 8KB per PE");

    using gmScore = global_tensor<float, RowMajor<Sq_, Skv_>>;
    using gmOut = global_tensor<float, RowMajor<Sq_, Skv_>>;

    using tileScore = Tile<Location::Vec, float, kTm_, kTk_, BLayout::RowMajor>;

    using tileMax = Tile<Location::Vec, float, kTm_, 8, BLayout::RowMajor, kTm_, 1>;
    using tileSum = Tile<Location::Vec, float, kTm_, 8, BLayout::RowMajor, kTm_, 1>;
    using tileScale = Tile<Location::Vec, float, kTm_, 8, BLayout::RowMajor, kTm_, 1>;

    using itScore = global_iterator<gmScore, tileScore>;
    using itOut = global_iterator<gmOut, tileScore>;

    itScore gScoreIter(score_ptr);
    itOut gOutIter(out_ptr);

    constexpr int Qb = Sq_ / kTm_;
    constexpr int Kb = Skv_ / kTk_;

    for (int i = 0; i < Qb; ++i) {
        tileMax tMax;
        tileSum tSum;

        // Initial online-softmax state for rows in this Q block.
        // tMax: [kTm, 1] = -inf, tSum: [kTm, 1] = 0.
        TEXPANDS(tMax, -1e30f);
        TEXPANDS(tSum, 0.0f);

        for (int j = 0; j < Kb; ++j) {
            tileScore tScore;
            tileMax tLocalMax;
            tileMax tNewMax;
            tileSum tLocalSum;
            tileSum tScaledOldSum;
            tileSum tNewSum;
            tileScale tScale;

            // TLOAD: global score block [kTm, kTk] -> float Vec tile.
            auto gScore = gScoreIter(i, j);
            TLOAD(tScore, gScore);

            // m_new = max(m_old, rowmax(score_block)).
            TROWMAX(tLocalMax, tScore);
            TMAX(tNewMax, tMax, tLocalMax);

            // Rescale old denominator from m_old base to m_new base:
            //   scaled_old_sum = old_sum * exp(m_old - m_new)
            TSUB(tScale, tMax, tNewMax);
            TEXP(tScale, tScale);
            TMUL(tScaledOldSum, tSum, tScale);

            // Local denominator under m_new:
            //   local_sum = rowsum(exp(score_block - m_new))
            TROWEXPANDSUB(tScore, tScore, tNewMax);
            TEXP(tScore, tScore);
            TROWSUM(tLocalSum, tScore);
            TADD(tNewSum, tScaledOldSum, tLocalSum);

            tMax = tNewMax;
            tSum = tNewSum;
        }

        tileScale tInvSum;
        TRECIP(tInvSum, tSum);

        for (int j = 0; j < Kb; ++j) {
            tileScore tScore;

            // Recompute exp(score - final_row_max), normalize by final row sum,
            // then store softmax(score) for this [kTm, kTk] block.
            auto gScore = gScoreIter(i, j);
            TLOAD(tScore, gScore);

            TROWEXPANDSUB(tScore, tScore, tMax);
            TEXP(tScore, tScore);
            TROWEXPANDMUL(tScore, tScore, tInvSum);

            auto gOut = gOutIter(i, j);
            TSTORE(gOut, tScore);
        }
    }
}

int main() {
    float scorep[B * H * Sq * Skv + 2 * ALIGN];
    float outp[B * H * Sq * Skv + 2 * ALIGN];

    float* score = (float*)(((uint64_t)scorep & ALIGN_MASK) + ALIGN);
    float* out = (float*)(((uint64_t)outp & ALIGN_MASK) + ALIGN);

#ifdef RES_CHECK
#define SRCSCORE_PATH CHK_DIR "/srcqk.bin"
    readBinaryFile(SRCSCORE_PATH, (uint8_t*)score, B * H * Sq * Skv * sizeof(float));
#endif

    BENCHSTART;
    for (int i = 0; i < B; ++i) {
        for (int j = 0; j < H; ++j) {
            softmax_qk_2d_unroll_pto<Sq, Skv, kTm, kTk>(
                out + i * H * Sq * Skv + j * Sq * Skv,
                score + i * H * Sq * Skv + j * Sq * Skv);
        }
    }
    BENCHEND;

#ifdef RES_CHECK
#define RES_PATH CHK_DIR "/res.bin"
    writeBinaryFile(RES_PATH, (uint8_t*)out, B * H * Sq * Skv * sizeof(float));
#endif

    return 0;
}
