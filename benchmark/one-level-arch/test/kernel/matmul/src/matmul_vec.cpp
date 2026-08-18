#include <common/pto_tileop.hpp>
#include <cstdint>
#include "fileop.h"
#include "common.h"
#include "benchmark.h"

#ifndef globM
#define globM 16
#endif

#ifndef globN
#define globN 16
#endif

#ifndef globK
#define globK 16
#endif

#ifndef tilM
#define tilM 16
#endif

#ifndef tilN
#define tilN 16
#endif

#ifndef tilK
#define tilK 16
#endif

// =====================================================================
// Vector-tile matmul: C[gM,gN] = A[gM,gK] x B[gK,gN]
//
// Uses only Tile<Location::Vec, ...> operands (no TileLeft/TileRight).
// Per k in [0, tK):
//   aCol  = TEXTRACT(A[:, k])               (tM x 1)
//   aFull = TROWEXPAND(aCol)                (tM x tN, row i = A[i][k])
//   bRow  = TEXTRACT(B[k, :])               (1 x tN)
//   tPart = TCOLEXPANDMUL(aFull, bRow)      (tM x tN, [i][j] = A[i][k]*B[k][j])
//   C    += tPart
// =====================================================================

namespace pto {

template <int gM, int gN, int gK, int tM, int tN, int tK>
void matmul_vec(float *dst, const float *src0, const float *src1) {
    using gm_A = global_tensor<float, RowMajor<gM, gK>>;
    using gm_B = global_tensor<float, RowMajor<gK, gN>>;
    using gm_C = global_tensor<float, RowMajor<gM, gN>>;

    using tileA = Tile<Location::Vec, float, tM, tK, BLayout::RowMajor>;
    using tileB = Tile<Location::Vec, float, tK, tN, BLayout::RowMajor>;
    using tileC = Tile<Location::Vec, float, tM, tN, BLayout::RowMajor>;

    using tileACol = Tile<Location::Vec, float, tM, 8, BLayout::RowMajor, tM, 1>;
    using tileBRow = Tile<Location::Vec, float, 8, tN, BLayout::RowMajor, 1, tN>;
    using tileAFull = Tile<Location::Vec, float, tM, tN, BLayout::RowMajor>;

    static_assert(gM % tM == 0 && gN % tN == 0 && gK % tK == 0,
                  "global dims must be divisible by tile dims");
    static_assert(tM % 8 == 0 && tN % 8 == 0 && tK % 8 == 0,
                  "tile dims must satisfy Tile alignment (x8)");
    static_assert(tM * tK * sizeof(float) <= 4 * 1024, "tileA too big");
    static_assert(tK * tN * sizeof(float) <= 4 * 1024, "tileB too big");
    static_assert(tM * tN * sizeof(float) <= 4 * 1024, "tileC too big");

    using itA = global_iterator<gm_A, tileA>;
    using itB = global_iterator<gm_B, tileB>;
    using itC = global_iterator<gm_C, tileC>;

    itA gAIter(const_cast<float *>(src0));
    itB gBIter(const_cast<float *>(src1));
    itC gCIter(dst);

    constexpr int Mb = gM / tM;
    constexpr int Nb = gN / tN;
    constexpr int Kb = gK / tK;

    for (int i = 0; i < Mb; ++i) {
        for (int j = 0; j < Nb; ++j) {
            tileC tC(0.0f);
            for (int kk = 0; kk < Kb; ++kk) {
                auto gA = gAIter(i, kk);
                auto gB = gBIter(kk, j);
                tileA tA;
                tileB tB;
                TLOAD(tA, gA);
                TLOAD(tB, gB);
                for (int k = 0; k < tK; ++k) {
                    tileACol aCol;
                    tileBRow bRow;
                    TEXTRACT(aCol, tA, 0, k);
                    TEXTRACT(bRow, tB, k, 0);
                    tileAFull aFull;
                    TROWEXPAND(aFull, aCol);
                    tileC tPart;
                    TCOLEXPANDMUL(tPart, aFull, bRow);
                    TADD(tC, tC, tPart);
                }
            }
            auto gC = gCIter(i, j);
            TSTORE(gC, tC);
        }
    }
}

} // namespace pto

int main() {
    float src0[globM * globK];
    float src1[globK * globN];
    float dst[globM * globN];

#ifdef RES_CHECK
#define SRC0_PATH CHK_DIR "/src0.bin"
#define SRC1_PATH CHK_DIR "/src1.bin"
    readBinaryFile(SRC0_PATH, (uint8_t *)src0, globM * globK * sizeof(float));
    readBinaryFile(SRC1_PATH, (uint8_t *)src1, globK * globN * sizeof(float));
#endif

    BENCHSTART;
    pto::matmul_vec<globM, globN, globK, tilM, tilN, tilK>(dst, src0, src1);
    BENCHEND;

#ifdef RES_CHECK
#define RES_PATH CHK_DIR "/res.bin"
    writeBinaryFile(RES_PATH, (uint8_t *)dst, globM * globN * sizeof(float));
#endif

    return 0;
}
