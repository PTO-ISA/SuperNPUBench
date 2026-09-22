#pragma once

#include <common/pto_tileop.hpp>
#include <cmath>
#include <cstdint>
#include <type_traits>

using namespace pto;

// Sq/Skv and both outer-loop trip counts are determined at runtime. Head
// dimensions and cooperative tile sizes remain compile-time PTO contracts.
struct FaGmmaTilingData {
    int64_t sq;
    int64_t skv;
};

template <typename matrix_dtype, typename vector_dtype, int PackedFactor,
          int qD, int vD, int kTm, int kTk>
__attribute__((noinline)) bool fa_gmma_dynamic(
    vector_dtype *out_ptr, const matrix_dtype *q_ptr, const matrix_dtype *k_ptr,
    const matrix_dtype *v_ptr, const FaGmmaTilingData *tiling) {
    constexpr int kPeNum = 4;
    constexpr int kStoredQD = qD / PackedFactor;
    constexpr int kStoredTk = kTk / PackedFactor;
    constexpr int kGroupM = kTm <= 128 ? kTm : 128;
    constexpr int kPeTm = kGroupM <= 64 ? 16 : 32;
    constexpr int kTileRows = kGroupM <= 64 ? 64 : 128;
    static_assert(kTm > 0 && kTk > 0 && kTm % kGroupM == 0);
    static_assert(PackedFactor == 1,
                  "dynamic FA currently supports unpacked FP32/BF16/FP8 inputs");
    static_assert(std::is_same_v<vector_dtype, float>,
                  "dynamic FA currently keeps softmax and output in FP32");
    static_assert(kPeTm * kPeNum == kGroupM,
                  "dynamic FA requires 64- or 128-row cooperative groups");

    const int64_t sq = tiling->sq;
    const int64_t skv = tiling->skv;
    const uint32_t tid = get_thread_idx();
    if (sq <= 0 || skv <= 0 || sq % kGroupM != 0 ||
        skv % kTk != 0 || tid >= kPeNum) return false;

    using gmQ = global_tensor<matrix_dtype, RowMajor<-1, kStoredQD>>;
    using gmK = global_tensor<matrix_dtype, RowMajor<-1, kStoredQD>>;
    using gmV = global_tensor<matrix_dtype, RowMajor<-1, vD>>;
    using gmO = global_tensor<vector_dtype, RowMajor<-1, vD>>;
    using qMatrix = SharedMatrixLeft<matrix_dtype, kTileRows, kStoredQD,
                                     kGroupM, kStoredQD>;
    using kMatrix = SharedMatrixRight<matrix_dtype, kTk, kStoredQD>;
    using vMatrix = SharedMatrixRight<matrix_dtype, kStoredTk, vD>;
    using qTile = SharedTile<qMatrix>;
    using kTile = SharedTile<kMatrix>;
    using vTile = SharedTile<vMatrix>;
    using scoreTile = std::conditional_t<(kPeTm <= 16),
        CubeTileM16<vector_dtype, kPeTm, kTk>,
        CubeTileM32<vector_dtype, kPeTm, kTk>>;
    using outputTile = std::conditional_t<(kPeTm <= 16),
        CubeAccumulatorM16<float, kPeTm, vD>,
        CubeAccumulatorM32<float, kPeTm, vD>>;
    using probabilityTile = std::conditional_t<(kPeTm <= 16),
        CubeTileM16<matrix_dtype, kPeTm, kStoredTk>,
        CubeTileM32<matrix_dtype, kPeTm, kStoredTk>>;
    using reduceTile = std::conditional_t<(kPeTm <= 16),
        VecTileM16<vector_dtype, kPeTm, kTk, kPeTm, 1>,
        VecTileM32<vector_dtype, kPeTm, kTk, kPeTm, 1>>;
    using stateTile = std::conditional_t<(kPeTm <= 16),
        VecTileM16<vector_dtype, kPeTm, 1, kPeTm, 1>,
        VecTileM32<vector_dtype, kPeTm, 1, kPeTm, 1>>;
    static_assert(qMatrix::LogicalTileBytes + kMatrix::LogicalTileBytes
                  <= 256 * 1024);
    static_assert(vMatrix::LogicalTileBytes <= 256 * 1024);

    using qIterator = global_iterator<gmQ, qMatrix>;
    using kIterator = global_iterator<gmK, kMatrix>;
    using vIterator = global_iterator<gmV, vMatrix>;
    using oIterator = global_iterator<gmO, outputTile>;
    qIterator gIterQ(const_cast<matrix_dtype *>(q_ptr));
    kIterator gIterK(const_cast<matrix_dtype *>(k_ptr));
    vIterator gIterV(const_cast<matrix_dtype *>(v_ptr));
    oIterator gIterO(out_ptr);

    const int64_t qb = sq / kGroupM;
    const int64_t kb = skv / kTk;
    const float scale = 1.0f / sqrt(static_cast<float>(qD));
    constexpr auto qkOptions = fixp::keep_acc();
    constexpr auto pvOptions = fixp::keep_acc().transpose_b();

    for (int64_t i = 0; i < qb; ++i) {
        outputTile out;
#ifndef FA_TRACE_ONLY
        stateTile max;
        stateTile sum;
        TEXPANDS(max, -1e30f);
        TEXPANDS(sum, 0.0f);
#endif

        qTile q;
        auto gQ = gIterQ(static_cast<int>(i), 0);
        TLOAD<qMatrix, 1>(q, gQ);

        // Single-pass online softmax. score/max/sum/out remain in tile
        // registers across the runtime KV loop; no score or PV scratch is used.
#ifndef FA_TRACE_ONLY
        for (int64_t j = 0; j < kb; ++j) {
            kTile k;
            auto gK = gIterK(static_cast<int>(j), 0);
            TLOAD<kMatrix, 1>(k, gK);

            scoreTile score;
            TMATMUL(score, q, k, qkOptions);
            TMULS(score, score, scale);

            reduceTile localMaxR;
            TROWMAX(localMaxR, score);
            auto localMax = TREDUCEPREFIXVIEW<stateTile>(localMaxR);
            stateTile newMax;
            stateTile oldScale;
            TMAX(newMax, max, localMax);
            if (j != 0) {
                TROWEXPANDEXPDIF(oldScale, max, newMax);
                TROWEXPANDMUL(out, out, oldScale);
            }

            TROWEXPANDEXPDIF(score, score, newMax);
            reduceTile localSumR;
            TROWSUM(localSumR, score);
            auto localSum = TREDUCEPREFIXVIEW<stateTile>(localSumR);
            stateTile newSum;
            if (j == 0) {
                TADD(newSum, sum, localSum);
            } else {
                TFMA(newSum, sum, oldScale, localSum);
            }

            vTile v;
            auto gV = gIterV(static_cast<int>(j), 0);
            TLOAD<vMatrix, 1>(v, gV);
            if constexpr (std::is_same_v<matrix_dtype, float>) {
                if (j == 0) {
                    TMATMUL(out, score, v, pvOptions, kGroupM);
                } else {
                    TMATMUL_ACC(out, out, score, v, pvOptions, kGroupM);
                }
            } else {
                probabilityTile probability;
                TCVT(probability, score);
                if (j == 0) {
                    TMATMUL(out, probability, v, pvOptions, kGroupM);
                } else {
                    TMATMUL_ACC(out, out, probability, v, pvOptions, kGroupM);
                }
            }

            max = newMax;
            sum = newSum;
        }

        TROWEXPANDDIV(out, out, sum);
#else
        // Ping-pong max/sum tiles avoid the loop-carried tile TMOV emitted for
        // `max = newMax` and `sum = newSum`. Reduction-prefix views are
        // consumed directly by TFMA through B.SUBVIEW.
        stateTile maxA;
        stateTile sumA;
        stateTile maxB;
        stateTile sumB;
        TEXPANDS(maxA, -1e30f);
        TEXPANDS(sumA, 0.0f);

        // Block 0 initializes the PV accumulator and writes state A -> B.
        {
            kTile k;
            auto gK = gIterK(0, 0);
            TLOAD<kMatrix, 1>(k, gK);
            scoreTile score;
            TMATMUL(score, q, k, qkOptions);
            TMULS(score, score, scale);
            reduceTile localMaxR;
            TROWMAX(localMaxR, score);
            auto localMax = TREDUCEPREFIXVIEW<stateTile>(localMaxR);
            TMAX(maxB, maxA, localMax);
            TROWEXPANDEXPDIF(score, score, maxB);
            reduceTile localSumR;
            TROWSUM(localSumR, score);
            auto localSum = TREDUCEPREFIXVIEW<stateTile>(localSumR);
            TADD(sumB, sumA, localSum);
            vTile v;
            auto gV = gIterV(0, 0);
            TLOAD<vMatrix, 1>(v, gV);
            if constexpr (std::is_same_v<matrix_dtype, float>) {
                TMATMUL(out, score, v, pvOptions, kGroupM);
            } else {
                probabilityTile probability;
                TCVT(probability, score);
                TMATMUL(out, probability, v, pvOptions, kGroupM);
            }
        }

        int64_t j = 1;
        // Consume two KV blocks per iteration: B -> A, then A -> B. State
        // ownership is therefore known without a tile copy at the back edge.
        for (; j + 1 < kb; j += 2) {
            {
                kTile k;
                auto gK = gIterK(static_cast<int>(j), 0);
                TLOAD<kMatrix, 1>(k, gK);
                scoreTile score;
                TMATMUL(score, q, k, qkOptions);
                TMULS(score, score, scale);
                reduceTile localMaxR;
                TROWMAX(localMaxR, score);
                auto localMax = TREDUCEPREFIXVIEW<stateTile>(localMaxR);
                stateTile oldScale;
                TMAX(maxA, maxB, localMax);
                TROWEXPANDEXPDIF(oldScale, maxB, maxA);
                TROWEXPANDMUL(out, out, oldScale);
                TROWEXPANDEXPDIF(score, score, maxA);
                reduceTile localSumR;
                TROWSUM(localSumR, score);
                auto localSum = TREDUCEPREFIXVIEW<stateTile>(localSumR);
                TFMA(sumA, sumB, oldScale, localSum);
                vTile v;
                auto gV = gIterV(static_cast<int>(j), 0);
                TLOAD<vMatrix, 1>(v, gV);
                if constexpr (std::is_same_v<matrix_dtype, float>) {
                    TMATMUL_ACC(out, out, score, v, pvOptions, kGroupM);
                } else {
                    probabilityTile probability;
                    TCVT(probability, score);
                    TMATMUL_ACC(out, out, probability, v, pvOptions, kGroupM);
                }
            }
            {
                kTile k;
                auto gK = gIterK(static_cast<int>(j + 1), 0);
                TLOAD<kMatrix, 1>(k, gK);
                scoreTile score;
                TMATMUL(score, q, k, qkOptions);
                TMULS(score, score, scale);
                reduceTile localMaxR;
                TROWMAX(localMaxR, score);
                auto localMax = TREDUCEPREFIXVIEW<stateTile>(localMaxR);
                stateTile oldScale;
                TMAX(maxB, maxA, localMax);
                TROWEXPANDEXPDIF(oldScale, maxA, maxB);
                TROWEXPANDMUL(out, out, oldScale);
                TROWEXPANDEXPDIF(score, score, maxB);
                reduceTile localSumR;
                TROWSUM(localSumR, score);
                auto localSum = TREDUCEPREFIXVIEW<stateTile>(localSumR);
                TFMA(sumB, sumA, oldScale, localSum);
                vTile v;
                auto gV = gIterV(static_cast<int>(j + 1), 0);
                TLOAD<vMatrix, 1>(v, gV);
                if constexpr (std::is_same_v<matrix_dtype, float>) {
                    TMATMUL_ACC(out, out, score, v, pvOptions, kGroupM);
                } else {
                    probabilityTile probability;
                    TCVT(probability, score);
                    TMATMUL_ACC(out, out, probability, v, pvOptions, kGroupM);
                }
            }
        }

        if (j < kb) {
            // Odd tail: the current state is in B and the result is written A.
            kTile k;
            auto gK = gIterK(static_cast<int>(j), 0);
            TLOAD<kMatrix, 1>(k, gK);
            scoreTile score;
            TMATMUL(score, q, k, qkOptions);
            TMULS(score, score, scale);
            reduceTile localMaxR;
            TROWMAX(localMaxR, score);
            auto localMax = TREDUCEPREFIXVIEW<stateTile>(localMaxR);
            stateTile oldScale;
            TMAX(maxA, maxB, localMax);
            TROWEXPANDEXPDIF(oldScale, maxB, maxA);
            TROWEXPANDMUL(out, out, oldScale);
            TROWEXPANDEXPDIF(score, score, maxA);
            reduceTile localSumR;
            TROWSUM(localSumR, score);
            auto localSum = TREDUCEPREFIXVIEW<stateTile>(localSumR);
            TFMA(sumA, sumB, oldScale, localSum);
            vTile v;
            auto gV = gIterV(static_cast<int>(j), 0);
            TLOAD<vMatrix, 1>(v, gV);
            if constexpr (std::is_same_v<matrix_dtype, float>) {
                TMATMUL_ACC(out, out, score, v, pvOptions, kGroupM);
            } else {
                probabilityTile probability;
                TCVT(probability, score);
                TMATMUL_ACC(out, out, probability, v, pvOptions, kGroupM);
            }
            TROWEXPANDDIV(out, out, sumA);
        } else {
            TROWEXPANDDIV(out, out, sumB);
        }
#endif
        auto gO = gIterO(static_cast<int>(i * kPeNum + tid), 0);
        TSTORE_CUBE(gO, out);
    }
    return true;
}
