#include <common/pto_tileop.hpp>
#include <cstring>
#include "fileop.h"
#include "common.h"
#include "benchmark.h"

#ifndef globM
#define globM 8192
#endif

#ifndef globN
#define globN 4096
#endif

#ifndef globK
#define globK 1600
#endif

#ifndef tilM
#define tilM  32
#endif

#ifndef tilN
#define tilN  32
#endif

#ifndef tilK
#define tilK  32
#endif

#ifndef NBLOCKS
#define NBLOCKS 32
#endif

#ifndef Batch
#define Batch  1
#endif

#define ALIGN_MASK 0xfffffffffffff000ull
#define ALIGN 4*1024

using namespace pto;

template <typename dtype, int gM, int gN, int gK, int tM, int tN, int tK>
void matmul_mask_multi_block_tileop(float *dst_cat, dtype *src0_cat, dtype *src1) {
    constexpr int kTileByteLimit = 4 * 1024;

    static_assert(gM % tM == 0);
    static_assert(gN % tN == 0);
    static_assert(gK % tK == 0);
    static_assert(tM * tK * sizeof(dtype) <= kTileByteLimit);
    static_assert(tM * tN * sizeof(float) <= kTileByteLimit);
    static_assert(tK * tN * sizeof(dtype) <= kTileByteLimit);

    const uint32_t tid = get_thread_idx();
    src0_cat += tid * gM * gK;
    dst_cat  += tid * gM * gN;

    using gmA = global_tensor<dtype, RowMajor<gM, gK>>;
    using gmB = global_tensor<dtype, RowMajor<gK, gN>>;
    using gmC = global_tensor<float, RowMajor<gM, gN>>;

    using tileA = TileLeft<dtype, tM, tK>;
    using tileB = TileRight<dtype, tK, tN>;
    using tileAcc = Tile<Location::Mat, float, tM, tN>;

    using itA = global_iterator<gmA, tileA>;
    using itB = global_iterator<gmB, tileB>;
    using itC = global_iterator<gmC, tileAcc>;

    itA gAIter(src0_cat);
    itB gBIter(src1);
    itC gCIter(dst_cat);

    const int Mb = gM / tM;
    const int Nb = gN / tN;
    const int Kb = gK / tK;

    for (int i = 0; i < Mb; ++i) {
        for (int j = 0; j < Nb; ++j) {
            tileAcc tAcc;
#pragma clang loop unroll(full)
            for (int k = 0; k < Kb; ++k) {
                tileA tA;
                tileB tB;

                auto gA = gAIter(i, k);
                auto gB = gBIter(k, j);
                TLOAD(tA, gA);
                TLOAD(tB, gB);

                if (k == 0) {
                    TMATMUL(tAcc, tA, tB);
                } else {
                    TMATMUL_ACC(tAcc, tAcc, tA, tB);
                }
            }
            auto gC = gCIter(i, j);
            TSTORE(gC, tAcc);
        }
    }
}

int main() {
    static_assert(globM % NBLOCKS == 0);
    constexpr int blockM = globM / NBLOCKS;
    static_assert(blockM % tilM == 0);
    static_assert(globN % tilN == 0);
    static_assert(globK % tilK == 0);

    using dtype = float;

    static dtype  src0p[NBLOCKS * blockM * globK + 2 * ALIGN];
    static dtype  src1p[globK * globN + 2 * ALIGN];
    static float  dstp[NBLOCKS * blockM * globN + 2 * ALIGN];

    dtype  *src0 = (dtype *)(((uint64_t)src0p & ALIGN_MASK) + ALIGN);
    dtype  *src1 = (dtype *)(((uint64_t)src1p & ALIGN_MASK) + ALIGN);
    float  *dst  = (float *)(((uint64_t)dstp  & ALIGN_MASK) + ALIGN);

#ifdef RES_CHECK
    readBinaryFile(CHK_DIR "/src0.bin", (uint8_t*)src0, NBLOCKS * blockM * globK * sizeof(dtype));
    readBinaryFile(CHK_DIR "/src1.bin", (uint8_t*)src1, globK * globN * sizeof(dtype));
#else
    const int totalA = NBLOCKS * blockM * globK;
    const int totalB = globK * globN;
    for (int idx = 0; idx < totalA; ++idx) src0[idx] = (float)(idx + 1);
    for (int idx = 0; idx < totalB; ++idx) src1[idx] = (float)(idx + 1);
#endif

    BENCHSTART;
    matmul_mask_multi_block_tileop<float, blockM, globN, globK,
                                    tilM, tilN, tilK>(
        dst, src0, src1);
    BENCHEND;

#ifdef RES_CHECK
    writeBinaryFile(CHK_DIR "/res.bin", (uint8_t*)dst, NBLOCKS * blockM * globN * sizeof(float));
#endif

    return 0;
}
