// mega_moe_full_sim: full GMM1->SwiGLU->GMM2->Combine pipeline using Cube TMATMUL
// Self-contained, no mega_moe_sim.hpp (avoid int64 structs triggering v2i64 crash)
// All uint32_t/float, no int64_t

#include <common/pto_tileop.hpp>
#include <cstdint>

using namespace pto;

static constexpr uint32_t kMoeBS = 16;
static constexpr uint32_t kMoeH = 128;
static constexpr uint32_t kMoeHiddenDim = 256;
static constexpr uint32_t kMoeExperts = 2;
static constexpr uint32_t kMoeTileW = 128;

// Cube tile sizes (must be multiples of 16 for float)
static constexpr uint32_t tM = 16;
static constexpr uint32_t tK = 128;
static constexpr uint32_t tN = 16;

// We do GMM1 for one N-tile (16 elements) at a time
// GMM1: [tM=16, tK=128] @ [tK=128, tN=16] = [tM=16, tN=16]
// But our token vector is [1, 128] — we need M=1 which is not a multiple of 16.
// Solution: use tM=16 and treat token as first row of a 16-row tile (rest zero).

static float x[kMoeBS * kMoeH] __attribute__((aligned(4096))) = {};
static float w1[kMoeExperts * kMoeH * kMoeHiddenDim] __attribute__((aligned(4096))) = {};
static float w2[kMoeExperts * (kMoeHiddenDim / 2) * kMoeH] __attribute__((aligned(4096))) = {};
static int32_t topkIds[kMoeBS] __attribute__((aligned(4096))) = {};
static float topkWeights[kMoeBS] __attribute__((aligned(4096))) = {};
static float y[kMoeBS * kMoeH] __attribute__((aligned(4096))) = {};
static uint32_t tokOut[kMoeExperts] __attribute__((aligned(4096))) = {};

// Intermediate buffers (padded to tile multiples)
static float y1Buf[tM * tN] __attribute__((aligned(4096))) = {};  // [16,16] GMM1 output tile
static float y2Buf[tM * tN] __attribute__((aligned(4096))) = {};  // [16,16] SwiGLU output

// Padded input for Cube (M=16 rows, only row 0 has real data)
static float xPadded[tM * tK] __attribute__((aligned(4096))) = {};
static float y1Padded[tM * tN] __attribute__((aligned(4096))) = {};  // GMM1 output [16,16]
static float y2Padded[tM * (tN)] __attribute__((aligned(4096))) = {}; // SwiGLU output [16,16]
static float y3Padded[tM * tN] __attribute__((aligned(4096))) = {};  // GMM2 output [16,16]

int main() {
    // Minimal scalar init
    for (uint32_t i = 0; i < kMoeBS * kMoeH; i++) x[i] = 0.01f * (float)i;
    for (uint32_t i = 0; i < kMoeBS; i++) { topkIds[i] = i % kMoeExperts; topkWeights[i] = 1.0f; }
    w1[0] = 0.01f; w2[0] = 0.01f;

    // Cube tile types
    using gmL = global_tensor<float, RowMajor<tM, tK>>;
    using gmR = global_tensor<float, RowMajor<tK, tN>>;
    using gmC = global_tensor<float, RowMajor<tM, tN>>;
    using TileL = TileLeft<float, tM, tK>;
    using TileR = TileRight<float, tK, tN>;
    using TileC = Tile<Location::Vec, float, tM, tN, BLayout::RowMajor>;

    // Vector tile types (for SwiGLU and Combine)
    using gmV = global_tensor<float, RowMajor<1, kMoeTileW>>;
    using tileV = Tile<Location::Vec, float, 1, kMoeTileW, BLayout::RowMajor>;

    for (uint32_t token = 0; token < kMoeBS; token++) {
        const uint32_t expert = (uint32_t)topkIds[token];
        const float weight = topkWeights[token];

        // --- Prepare padded x: copy token row into xPadded[0, :] ---
        for (uint32_t j = 0; j < kMoeH; j++) xPadded[j] = x[token * kMoeH + j];

        // ====== GMM1: xPadded[16,128] @ w1_tile[128,16] = y1Padded[16,16] ======
        {
            gmL gA(xPadded);
            gmR gB(w1 + expert * kMoeH * kMoeHiddenDim);  // w1[expert][0:128][0:16]
            gmC gC(y1Padded);

            TileL tA;
            TileR tB;
            TileC tC;
            TLOAD(tA, gA);
            TLOAD(tB, gB);
            TMATMUL(tC, tA, tB);
            TSTORE(gC, tC);
        }

        // ====== SwiGLU: y2 = silu(y1[:8]) * y1[8:8+8] ======
        // y1Padded is [16,16] → take row 0 as [1,16], split into y1a[1,8]+y1b[1,8]
        // For Vector processing: use tileV (1,128) but only 8 elements valid
        // Simplified: use Vector TMULS/TEXP chain on first 128 elements of y1Padded
        {
            // Load y1a (first 128 floats = 8 rows × 16 cols, but we treat as 1D)
            tileV y1a, y1b;
            gmV gY1a(y1Padded);
            TLOAD(y1a, gY1a);
            // y1b starts at offset 128 (but y1Padded is only 16×16=256, offset 128 = second half)
            gmV gY1b(y1Padded + tM * tN / 2);  // 256/2=128
            TLOAD(y1b, gY1b);

            // sigmoid(y1a) = 1/(1+exp(-y1a))
            tileV neg, expv, one, recip, sig, y2;
            TMULS(neg, y1a, -1.0f);
            TEXP(expv, neg);
            TADDS(one, expv, 1.0f);
            TRECIP(recip, one);
            TMUL(sig, y1a, recip);
            TMUL(y2, sig, y1b);
            gmV gY2(y2Padded);
            TSTORE(gY2, y2);
        }

        // ====== GMM2: y2Padded[16,128] @ w2_tile[128,16] = y3Padded[16,16] ======
        // w2 is [expert][128][128], take first 128×16 slice
        // But w2 has shape [expert][hd/2=128][h=128], so w2[expert][0:128][0:16]
        // Wait, w2 layout: [expert][(hd/2)][h] = [2][128][128]
        // For TMATMUL: need [K=128, N=16] from w2
        // w2[expert] is [128][128], w2[expert][0:128][0:16] → stride = 128
        {
            gmL gA(y2Padded);
            gmR gB(w2 + expert * (kMoeHiddenDim / 2) * kMoeH);
            gmC gC(y3Padded);

            TileL tA;
            TileR tB;
            TileC tC;
            TLOAD(tA, gA);
            TLOAD(tB, gB);
            TMATMUL(tC, tA, tB);
            TSTORE(gC, tC);
        }

        // ====== Combine: y[token] = weight * y3[token_row] ======
        // Extract row 0 from y3Padded [16,16] → y[token][0:16]
        // For now, just copy y3Padded[0:128] as y[token][0:128]
        {
            tileV y3;
            gmV gY3(y3Padded);
            TLOAD(y3, gY3);
            TMULS(y3, y3, weight);
            gmV gY(y + token * kMoeH);
            TSTORE(gY, y3);
        }
    }

    tokOut[0] = kMoeBS / 2;
    tokOut[1] = kMoeBS - kMoeBS / 2;

    return 0;
}
