#pragma once

#include <common/pto_tileop.hpp>
#include <cmath>
#include <cstdint>
#include <type_traits>

using namespace pto;

// Runtime sequence lengths. Head dimensions and tile capacities remain
// compile-time PTO contracts; sq/skv may end in partial Q/KV tiles.
struct FaGmmaTilingData {
    int64_t sq;
    int64_t skv;
};

template <typename Gm, typename Src>
__attribute__((always_inline)) inline void fa_dynamic_store_rows(
    Gm &dst, const Src &src, size_t validRow) {
    static_assert(Src::BFractal == BLayout::CubeM16 ||
                  Src::BFractal == BLayout::CubeM32);
    if (validRow == 0) return;
    if constexpr (Src::BFractal == BLayout::CubeM32) {
        asm volatile(
            "BSTART.TLSU TSTORE, %D[DataType]\n"
            "B.DATR M322ND.normal, Null\n"
            "B.DIM zero, %c[ValidCol], ->lb0\n"
            "B.DIM %[ValidRow], 0, ->lb1\n"
            "B.IOT %[Src], mask=1111, last\n"
            "B.IOR [%[Base],%[RowStrideBytes]], []\n"
            :
            : [Base] "r"(dst.data()), [Src] "Tr"(src.data()),
              [RowStrideBytes] "r"(dst.GetStrideBytes(3)),
              [DataType] "i"(type_traits<typename Src::DType>::TypeCode),
              [ValidCol] "i"(Src::ValidCol), [ValidRow] "r"(validRow)
            : "memory");
    } else {
        asm volatile(
            "BSTART.TLSU TSTORE, %D[DataType]\n"
            "B.DATR M162ND.normal, Null\n"
            "B.DIM zero, %c[ValidCol], ->lb0\n"
            "B.DIM %[ValidRow], 0, ->lb1\n"
            "B.IOT %[Src], mask=1111, last\n"
            "B.IOR [%[Base],%[RowStrideBytes]], []\n"
            :
            : [Base] "r"(dst.data()), [Src] "Tr"(src.data()),
              [RowStrideBytes] "r"(dst.GetStrideBytes(3)),
              [DataType] "i"(type_traits<typename Src::DType>::TypeCode),
              [ValidCol] "i"(Src::ValidCol), [ValidRow] "r"(validRow)
            : "memory");
    }
}

template <typename matrix_dtype, typename vector_dtype, int PackedFactor,
          int qD, int vD, int kTm, int kTk, int MaxSq, int MaxSkv>
__attribute__((noinline)) bool fa_gmma_dynamic(
    vector_dtype *out_ptr, const matrix_dtype *q_ptr, const matrix_dtype *k_ptr,
    const matrix_dtype *v_ptr, const vector_dtype *mask_ptr,
    const FaGmmaTilingData *tiling) {
    constexpr int kPeNum = 4;
    constexpr int kStoredQD = qD / PackedFactor;
    constexpr int kStoredTk = kTk / PackedFactor;
    constexpr int kGroupM = kTm <= 128 ? kTm : 128;
    constexpr int kPeTm = kGroupM <= 64 ? 16 : 32;
    constexpr int kTileRows = kGroupM <= 64 ? 64 : 128;

    static_assert(kTm > 0 && kTk > 0 && kTm % kGroupM == 0);
    static_assert(MaxSq > 0 && MaxSkv > 0);
    static_assert(PackedFactor == 1,
                  "dynamic FA currently supports unpacked FP32/BF16/FP8 inputs");
    static_assert(kPeTm * kPeNum == (kGroupM <= 64 ? 64 : 128),
                  "dynamic FA requires a 64- or 128-row physical PE group");

    const int64_t sq = tiling->sq;
    const int64_t skv = tiling->skv;
    const uint32_t tid = get_thread_idx();
    if (sq <= 0 || skv <= 0 || sq > MaxSq || skv > MaxSkv ||
        tid >= kPeNum) return false;

    using gmQ = global_tensor<matrix_dtype, RowMajor<-1, kStoredQD>>;
    using gmK = global_tensor<matrix_dtype, RowMajor<-1, kStoredQD>>;
    using gmV = global_tensor<matrix_dtype, RowMajor<-1, vD>>;
    using gmO = global_tensor<vector_dtype, RowMajor<-1, vD>>;
    using gmMask = global_tensor<vector_dtype, RowMajor<-1, kTk>>;

    // Matrix tiles keep the exact fa_gmma contract. The caller pads Q to a
    // cooperative group and K/V to kTk; runtime valid_row limits the final Q
    // store and runtime valid_col builds the last-KV-block softmax mask.
    using qMatrix = SharedMatrixLeft<matrix_dtype, kTileRows, kStoredQD,
                                     kGroupM, kStoredQD>;
    using kMatrix = SharedMatrixRight<matrix_dtype, kTk, kStoredQD>;
    using vMatrix = SharedMatrixRight<matrix_dtype, kStoredTk, vD>;
    using qTile = SharedTile<qMatrix>;
    using kTile = SharedTile<kMatrix>;
    using vTile = SharedTile<vMatrix>;

    using scoreM16 = CubeTileM16<vector_dtype, kPeTm, kTk>;
    using scoreM32 = CubeTileM32<vector_dtype, kPeTm, kTk>;
    using scoreTile = std::conditional_t<(kPeTm <= 16), scoreM16, scoreM32>;

    using outputM16 = CubeAccumulatorM16<float, kPeTm, vD>;
    using outputM32 = CubeAccumulatorM32<float, kPeTm, vD>;
    using outputTile =
        std::conditional_t<(kPeTm <= 16), outputM16, outputM32>;

    using probabilityM16 =
        CubeTileM16<matrix_dtype, kPeTm, kStoredTk>;
    using probabilityM32 =
        CubeTileM32<matrix_dtype, kPeTm, kStoredTk>;
    using probabilityTile =
        std::conditional_t<(kPeTm <= 16), probabilityM16, probabilityM32>;

    using reduceM16 = VecTileM16<vector_dtype, kPeTm, kTk, kPeTm, 1>;
    using reduceM32 = VecTileM32<vector_dtype, kPeTm, kTk, kPeTm, 1>;
    using reduceTile =
        std::conditional_t<(kPeTm <= 16), reduceM16, reduceM32>;
    using maskTile = scoreTile;

    using stateM16 = VecTileM16<vector_dtype, kPeTm, 1>;
    using stateM32 = VecTileM32<vector_dtype, kPeTm, 1>;
    using stateTile =
        std::conditional_t<(kPeTm <= 16), stateM16, stateM32>;

    using rowFp32M16 = VecTileM16<float, kPeTm, 1>;
    using rowFp32M32 = VecTileM32<float, kPeTm, 1>;
    using rowFp32Tile =
        std::conditional_t<(kPeTm <= 16), rowFp32M16, rowFp32M32>;

    using outputCastM16 =
        CubeAccumulatorM16<vector_dtype, kPeTm, vD>;
    using outputCastM32 =
        CubeAccumulatorM32<vector_dtype, kPeTm, vD>;
    using outputCastTile =
        std::conditional_t<(kPeTm <= 16), outputCastM16, outputCastM32>;

    static_assert(qMatrix::LogicalTileBytes + kMatrix::LogicalTileBytes
                  <= 256 * 1024);
    static_assert(vMatrix::LogicalTileBytes <= 256 * 1024);

    const int64_t qBlockCount = (sq + kGroupM - 1) / kGroupM;
    const int64_t kvBlockCount = (skv + kTk - 1) / kTk;
    const float scale = 1.0f / sqrt(static_cast<float>(qD));
    constexpr auto qkFp32Options = fixp::keep_acc();
    constexpr auto qkBf16Options = fixp::bf16();
    constexpr auto pvOptions = fixp::keep_acc().transpose_b();

    for (int64_t i = 0; i < qBlockCount; ++i) {
        const int64_t qBegin = i * kGroupM;
        const int64_t qRemaining = sq - qBegin;
        const int qValid = static_cast<int>(
            qRemaining < kGroupM ? qRemaining : kGroupM);
        const int peBegin = static_cast<int>(tid) * kPeTm;
        const int peRemaining = qValid - peBegin;
        const int peValid = peRemaining <= 0
            ? 0 : (peRemaining < kPeTm ? peRemaining : kPeTm);

        stateTile tMax;
        rowFp32Tile tSum;
        outputTile tO;
        TEXPANDS(tMax, -1e30f);
        TEXPANDS(tSum, 0.0f);

        qTile tQ;
        gmQ gQ(const_cast<matrix_dtype *>(q_ptr) + qBegin * kStoredQD,
               kStoredQD);
        TLOAD<qMatrix, 1>(tQ, gQ);

#define FA_DYNAMIC_PROCESS_KV_BLOCK(BLOCK_INDEX, IS_FIRST)                   \
        do {                                                                  \
            const int64_t blockIndex = (BLOCK_INDEX);                         \
            const int64_t kvBegin = blockIndex * kTk;                         \
            kTile tK;                                                         \
            gmK gK(const_cast<matrix_dtype *>(k_ptr) +                        \
                       kvBegin * kStoredQD,                                    \
                   kStoredQD);                                                \
            TLOAD<kMatrix, 1>(tK, gK);                                        \
            scoreTile tW;                                                     \
            if constexpr (std::is_same_v<vector_dtype, float>) {              \
                TMATMUL(tW, tQ, tK, qkFp32Options);                           \
            } else {                                                          \
                static_assert(std::is_same_v<vector_dtype, __bf16>,           \
                              "QK fixpipe output supports FP32 or BF16");     \
                TMATMUL(tW, tQ, tK, qkBf16Options);                           \
            }                                                                 \
            TMULS(tW, tW, scale);                                             \
            maskTile tMask;                                                   \
            gmMask gMask(const_cast<vector_dtype *>(mask_ptr) +               \
                             blockIndex * kPeTm * kTk,                         \
                         kTk);                                                 \
            TLOAD(tMask, gMask);                                              \
            TADD(tW, tW, tMask);                                              \
            reduceTile tLocalMaxR;                                            \
            TROWMAX(tLocalMaxR, tW);                                          \
            auto tLocalMax = TREDUCEPREFIXVIEW<stateTile>(tLocalMaxR);        \
            stateTile tNewMax;                                                \
            stateTile tScale;                                                 \
            rowFp32Tile tScaleFp32;                                           \
            TMAX(tNewMax, tMax, tLocalMax);                                   \
            if constexpr (!(IS_FIRST)) {                                      \
                TROWEXPANDEXPDIF(tScale, tMax, tNewMax);                      \
                if constexpr (std::is_same_v<vector_dtype, float>) {          \
                    TROWEXPANDMUL(tO, tO, tScale);                            \
                } else {                                                      \
                    TCVT(tScaleFp32, tScale);                                 \
                    TROWEXPANDMUL(tO, tO, tScaleFp32);                        \
                }                                                             \
            }                                                                 \
            TROWEXPANDEXPDIF(tW, tW, tNewMax);                               \
            reduceTile tLocalSumR;                                            \
            TROWSUM(tLocalSumR, tW);                                          \
            rowFp32Tile tNewSum;                                              \
            if constexpr (std::is_same_v<vector_dtype, float>) {              \
                auto tLocalSum =                                              \
                    TREDUCEPREFIXVIEW<rowFp32Tile>(tLocalSumR);               \
                if constexpr (IS_FIRST) {                                     \
                    TADD(tNewSum, tSum, tLocalSum);                           \
                } else {                                                      \
                    TFMA(tNewSum, tSum, tScale, tLocalSum);                   \
                }                                                             \
            } else {                                                          \
                auto tLocalSumView =                                          \
                    TREDUCEPREFIXVIEW<stateTile>(tLocalSumR);                 \
                stateTile tLocalSumZero;                                      \
                stateTile tLocalSumCompact;                                   \
                TEXPANDS(tLocalSumZero, 0.0f);                                \
                TADD(tLocalSumCompact, tLocalSumZero, tLocalSumView);         \
                rowFp32Tile tLocalSumFp32;                                    \
                TCVT(tLocalSumFp32, tLocalSumCompact);                        \
                if constexpr (IS_FIRST) {                                     \
                    TADD(tNewSum, tSum, tLocalSumFp32);                       \
                } else {                                                      \
                    TFMA(tNewSum, tSum, tScaleFp32, tLocalSumFp32);           \
                }                                                             \
            }                                                                 \
            vTile tV;                                                         \
            gmV gV(const_cast<matrix_dtype *>(v_ptr) + kvBegin * vD, vD);    \
            TLOAD<vMatrix, 1>(tV, gV);                                        \
            if constexpr (std::is_same_v<matrix_dtype, vector_dtype>) {       \
                if constexpr (IS_FIRST) {                                     \
                    TMATMUL(tO, tW, tV, pvOptions, kGroupM);                  \
                } else {                                                      \
                    TMATMUL_ACC(tO, tO, tW, tV, pvOptions, kGroupM);          \
                }                                                             \
            } else {                                                          \
                probabilityTile tP;                                           \
                TCVT(tP, tW);                                                 \
                if constexpr (IS_FIRST) {                                     \
                    TMATMUL(tO, tP, tV, pvOptions, kGroupM);                  \
                } else {                                                      \
                    TMATMUL_ACC(tO, tO, tP, tV, pvOptions, kGroupM);          \
                }                                                             \
            }                                                                 \
            tMax = tNewMax;                                                   \
            tSum = tNewSum;                                                   \
        } while (false)

        FA_DYNAMIC_PROCESS_KV_BLOCK(0, true);
        for (int64_t j = 1; j < kvBlockCount; ++j) {
            FA_DYNAMIC_PROCESS_KV_BLOCK(j, false);
        }
#undef FA_DYNAMIC_PROCESS_KV_BLOCK

        TROWEXPANDDIV(tO, tO, tSum);
        gmO gO(out_ptr + (qBegin + peBegin) * vD, vD);
        if constexpr (std::is_same_v<vector_dtype, float>) {
            fa_dynamic_store_rows(gO, tO, static_cast<size_t>(peValid));
        } else {
            outputCastTile tOCast;
            TCVT(tOCast, tO);
            fa_dynamic_store_rows(gO, tOCast,
                                  static_cast<size_t>(peValid));
        }
    }
    return true;
}
