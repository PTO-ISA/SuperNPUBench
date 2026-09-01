#include <common/pto_tileop.hpp>
#include "benchmark.h"
#include "fileop.h"

#include <cstdint>

using namespace pto;

#ifndef gM
#define gM 16
#endif

#ifndef gN
#define gN 16
#endif

#ifndef gK
#define gK 128
#endif

#ifndef gDChunk
#define gDChunk 64
#endif

#ifndef gRows
#define gRows 4
#endif

#define ALIGN_MASK 0xfffffffffffff000ull
#define ALIGN 4 * 1024

template <typename dtype, int M, int N, int K, int DChunk, int Rows>
void tmatmul_acc_unit_kernel(float* score_ptr, dtype* q_ptr, dtype* k_ptr) {
    using gmQ = global_tensor<dtype, RowMajor<M, K>>;
    using gmKView = global_tensor<dtype, MatrixLayout<K, Rows * N, 1, K>>;
    using gmScore = global_tensor<float, RowMajor<Rows * M, N>>;

    using tileQ = std::conditional_t<
        (M <= 16), CubeTileM16<dtype, M, DChunk>,
        CubeTileM32<dtype, M, DChunk>>;
    using tileK = CubeTileN8<dtype, DChunk, N>;
    using tileAcc = std::conditional_t<
        (M <= 16), CubeAccumulatorM16<float, M, N>,
        CubeAccumulatorM32<float, M, N>>;
    using tileScore = Tile<Location::Vec, float, M, N, BLayout::ColMajor>;

    using itQ = global_iterator<gmQ, tileQ>;
    using itK = global_iterator<gmKView, tileK>;
    using itScore = global_iterator<gmScore, tileScore>;

    itQ q_iter(q_ptr);
    itK k_iter(k_ptr);
    itScore score_iter(score_ptr);

    #pragma clang loop unroll(full)
    for (int row = 0; row < Rows; ++row) {
        tileAcc tAcc;
        #pragma clang loop unroll(full)
        for (int d = 0; d < K / DChunk; ++d) {
            tileQ tQ;
            tileK tK;
            auto gQ = q_iter(0, d);
            auto gKTile = k_iter(d, row);
            TLOAD_CUBE(tQ, gQ);
            TLOAD_CUBE(tK, gKTile);
            if (d == 0) {
                TMATMUL(tAcc, tQ, tK);
            } else {
                TMATMUL_ACC(tAcc, tAcc, tQ, tK);
            }
        }
        auto gScore = score_iter(row, 0);
        TSTORE_CUBE(gScore, tAcc);
    }
}

int main() {
    using dtype = __half;
    dtype q_buf[gM * gK + 2 * ALIGN];
    dtype k_buf[gRows * gN * gK + 2 * ALIGN];
    float score_buf[gRows * gM * gN + 2 * ALIGN];

    dtype* q = (dtype*)(((uint64_t)q_buf & ALIGN_MASK) + ALIGN);
    dtype* k = (dtype*)(((uint64_t)k_buf & ALIGN_MASK) + ALIGN);
    float* score = (float*)(((uint64_t)score_buf & ALIGN_MASK) + ALIGN);

#ifdef RES_CHECK
#define SRCQ_PATH CHK_DIR "/srcq.bin"
#define SRCK_PATH CHK_DIR "/srck.bin"
    readBinaryFile(SRCQ_PATH, (uint8_t*)q, gM * gK * sizeof(dtype));
    readBinaryFile(SRCK_PATH, (uint8_t*)k, gRows * gN * gK * sizeof(dtype));
#endif

    BENCHSTART;
    tmatmul_acc_unit_kernel<dtype, gM, gN, gK, gDChunk, gRows>(score, q, k);
    BENCHEND;

#ifdef RES_CHECK
#define SCORE_PATH CHK_DIR "/score.bin"
    writeBinaryFile(SCORE_PATH, (uint8_t*)score, gRows * gM * gN * sizeof(float));
#endif

    return 0;
}
