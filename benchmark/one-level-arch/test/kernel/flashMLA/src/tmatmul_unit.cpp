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

#define ALIGN_MASK 0xfffffffffffff000ull
#define ALIGN 4 * 1024

template <typename dtype, int M, int N, int K>
void tmatmul_unit_kernel(float* score_ptr, dtype* q_loaded_ptr, dtype* k_loaded_ptr, dtype* q_ptr, dtype* k_ptr) {
    using gmQ = global_tensor<dtype, RowMajor<M, K>>;
    using gmK = global_tensor<dtype, RowMajor<K, N>>;
    using gmScore = global_tensor<float, RowMajor<M, N>>;

    using tileQ = std::conditional_t<(M <= 16), CubeTileM16<dtype, M, K>,
                                     CubeTileM32<dtype, M, K>>;
    using tileK = CubeTileN8<dtype, K, N>;
    using tileAcc = std::conditional_t<
        (M <= 16), CubeAccumulatorM16<float, M, N>,
        CubeAccumulatorM32<float, M, N>>;
    using tileScore = Tile<Location::Vec, float, M, N, BLayout::ColMajor>;

    using itQ = global_iterator<gmQ, tileQ>;
    using itK = global_iterator<gmK, tileK>;
    using itScore = global_iterator<gmScore, tileScore>;

    itQ q_iter(q_ptr);
    itK k_iter(k_ptr);
    itScore score_iter(score_ptr);

    tileQ tQ;
    tileK tK;
    tileAcc tAcc;
    tileScore tScore;

    auto q_tile_gm = q_iter(0, 0);
    auto k_tile_gm = k_iter(0, 0);
    auto gScore = score_iter(0, 0);
    TLOAD_CUBE(tQ, q_tile_gm);
    TLOAD_CUBE(tK, k_tile_gm);
    TMATMUL(tAcc, tQ, tK);
    TSTORE_CUBE(gScore, tAcc);

#ifdef RES_CHECK
    using itQDump = global_iterator<gmQ, tileQ>;
    using itKDump = global_iterator<gmK, tileK>;
    itQDump q_dump_iter(q_loaded_ptr);
    itKDump k_dump_iter(k_loaded_ptr);
    auto qDump = q_dump_iter(0, 0);
    auto kDump = k_dump_iter(0, 0);
    TSTORE_CUBE(qDump, tQ);
    TSTORE_CUBE(kDump, tK);
#endif
}

int main() {
    using dtype = __half;

    dtype q_buf[gM * gK + 2 * ALIGN];
    dtype k_buf[gK * gN + 2 * ALIGN];
    dtype q_loaded_buf[gM * gK + 2 * ALIGN];
    dtype k_loaded_buf[gK * gN + 2 * ALIGN];
    float score_buf[gM * gN + 2 * ALIGN];

    dtype* q = (dtype*)(((uint64_t)q_buf & ALIGN_MASK) + ALIGN);
    dtype* k = (dtype*)(((uint64_t)k_buf & ALIGN_MASK) + ALIGN);
    dtype* q_loaded = (dtype*)(((uint64_t)q_loaded_buf & ALIGN_MASK) + ALIGN);
    dtype* k_loaded = (dtype*)(((uint64_t)k_loaded_buf & ALIGN_MASK) + ALIGN);
    float* score = (float*)(((uint64_t)score_buf & ALIGN_MASK) + ALIGN);

#ifdef RES_CHECK
#define SRCQ_PATH CHK_DIR "/srcq.bin"
#define SRCK_PATH CHK_DIR "/srck.bin"
    readBinaryFile(SRCQ_PATH, (uint8_t*)q, gM * gK * sizeof(dtype));
    readBinaryFile(SRCK_PATH, (uint8_t*)k, gK * gN * sizeof(dtype));
#endif

    BENCHSTART;
    tmatmul_unit_kernel<dtype, gM, gN, gK>(score, q_loaded, k_loaded, q, k);
    BENCHEND;

#ifdef RES_CHECK
#define SCORE_PATH CHK_DIR "/score.bin"
#define Q_LOADED_PATH CHK_DIR "/q_loaded.bin"
#define K_LOADED_PATH CHK_DIR "/k_loaded.bin"
    writeBinaryFile(SCORE_PATH, (uint8_t*)score, gM * gN * sizeof(float));
    writeBinaryFile(Q_LOADED_PATH, (uint8_t*)q_loaded, gM * gK * sizeof(dtype));
    writeBinaryFile(K_LOADED_PATH, (uint8_t*)k_loaded, gK * gN * sizeof(dtype));
#endif

    return 0;
}
