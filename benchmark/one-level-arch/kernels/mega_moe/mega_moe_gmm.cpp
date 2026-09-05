#include <common/pto_tileop.hpp>
#include <cstdint>
#include "mega_moe/mega_moe_gmm.hpp"

using namespace pto;

// TileW=128 (512B minimum for TLOAD/TSTORE)
constexpr uint32_t kVecW = 128;

// This translation unit contains ONLY tile operations — no int64 structs,
// no Mc2MoeContext, no MegaMoeTilingData. This avoids the linxv5 backend
// v2i64 BUILD_VECTOR crash that occurs when 128-bit zero stores are generated
// alongside tile operations in the same translation unit.

namespace mega_moe_gmm {

// GMM1 for one output tile: acc = sum_k x[k] * w1[e][k][n_tile]
// First iter: TMUL(acc, a, b); rest: TFMA(acc, a, b, acc)
// 3 tiles only (acc, a, b) — TFMA eliminates the 4th (prod) tile
static void gmm1_output_tile(float* y1Out, const float* xIn, const float* w1fp32,
                              uint32_t token, uint32_t e, uint32_t nt, uint32_t h)
{
    using gmFlat = global_tensor<float, RowMajor<1, kVecW>>;
    using tileV  = Tile<Location::Vec, float, 1, kVecW, BLayout::RowMajor>;

    constexpr uint32_t kTiles = 128 / kVecW; // h=128, kVecW=128 → 1 tile
    tileV acc, a, b;

    // Iter 0: acc = a * b
    {
        gmFlat gX(const_cast<float*>(xIn + token * h));
        TLOAD(a, gX);
        const float* w1Row = w1fp32 + e * h * 256 + nt * kVecW;
        gmFlat gW(const_cast<float*>(w1Row));
        TLOAD(b, gW);
        TMUL(acc, a, b);
    }
    // Iter 1+: acc = a * b + acc
    for (uint32_t kt = 1; kt < kTiles; ++kt) {
        gmFlat gX(const_cast<float*>(xIn + token * h + kt * kVecW));
        TLOAD(a, gX);
        const float* w1Row = w1fp32 + e * h * 256 + kt * kVecW * 256 + nt * kVecW;
        gmFlat gW(const_cast<float*>(w1Row));
        TLOAD(b, gW);
        TFMA(acc, a, b, acc);
    }

    gmFlat gY(y1Out + nt * kVecW);
    TSTORE(gY, acc);
}

// SwiGLU: y2 = silu(y1[:hd/2]) * y1[hd/2:]
// silu(z) = z * sigmoid(z) = z / (1 + exp(-z))
static void swiglu_tile(float* y2Out, const float* y1Buf, uint32_t kt, uint32_t hiddenDim)
{
    using gmFlat = global_tensor<float, RowMajor<1, kVecW>>;
    using tileV  = Tile<Location::Vec, float, 1, kVecW, BLayout::RowMajor>;

    tileV y1a, y1b;
    gmFlat gY1a(const_cast<float*>(y1Buf + kt * kVecW));
    TLOAD(y1a, gY1a);
    gmFlat gY1b(const_cast<float*>(y1Buf + hiddenDim / 2 + kt * kVecW));
    TLOAD(y1b, gY1b);

    // sigmoid(y1a) = 1/(1+exp(-y1a))
    tileV neg, expv, one, recip;
    TMULS(neg, y1a, -1.0f);
    TEXP(expv, neg);
    TADDS(one, expv, 1.0f);
    TRECIP(recip, one);

    // y2 = y1a * sigmoid(y1a) * y1b
    tileV sig, y2;
    TMUL(sig, y1a, recip);
    TMUL(y2, sig, y1b);

    gmFlat gY2(y2Out + kt * kVecW);
    TSTORE(gY2, y2);
}

// GMM2 + Combine: acc = sum_k y2[k] * w2[e][k][n_tile], y = weight * acc
static void gmm2_combine_tile(float* yOut, const float* y2Buf, const float* w2fp32,
                               uint32_t token, uint32_t e, uint32_t nt, uint32_t h, float weight)
{
    using gmFlat = global_tensor<float, RowMajor<1, kVecW>>;
    using tileV  = Tile<Location::Vec, float, 1, kVecW, BLayout::RowMajor>;

    constexpr uint32_t kTiles = 128 / kVecW; // hiddenDim/2 = 128, kVecW=128 → 1 tile
    tileV acc, a, b;

    // Iter 0: acc = a * b
    {
        gmFlat gY2(const_cast<float*>(y2Buf));
        TLOAD(a, gY2);
        const float* w2Row = w2fp32 + e * (256 / 2) * h + nt * kVecW;
        gmFlat gW(const_cast<float*>(w2Row));
        TLOAD(b, gW);
        TMUL(acc, a, b);
    }
    for (uint32_t kt = 1; kt < kTiles; ++kt) {
        gmFlat gY2(const_cast<float*>(y2Buf + kt * kVecW));
        TLOAD(a, gY2);
        const float* w2Row = w2fp32 + e * (256 / 2) * h + kt * kVecW * h + nt * kVecW;
        gmFlat gW(const_cast<float*>(w2Row));
        TLOAD(b, gW);
        TFMA(acc, a, b, acc);
    }

    // Combine: y = weight * y3
    TMULS(acc, acc, weight);

    gmFlat gY(yOut + token * h + nt * kVecW);
    TSTORE(gY, acc);
}

// Full pipeline for one token: GMM1 → SwiGLU → GMM2 → Combine
void gmm_pipeline_for_token(
    float* yOut,
    const float* xIn,
    const float* w1fp32,
    const float* w2fp32,
    const int32_t* topkIds,
    const float* topkWeights,
    float* workspace,
    uint32_t token,
    uint32_t bs,
    uint32_t h,
    uint32_t hiddenDim,
    uint32_t expertPerRank)
{
    const int32_t expert = topkIds[token];
    const float weight = topkWeights[token];
    const uint32_t e = static_cast<uint32_t>(expert);

    // workspace layout: [bs * hiddenDim] for y1, [hiddenDim/2] for y2
    float* y1Buf = workspace + token * hiddenDim;
    float* y2Buf = workspace + bs * hiddenDim;

    // GMM1: y1[n] = sum_k x[k] * w1[e][k][n], N=hiddenDim in tiles of kVecW
    for (uint32_t nt = 0; nt < hiddenDim / kVecW; ++nt)
        gmm1_output_tile(y1Buf, xIn, w1fp32, token, e, nt, h);

    // SwiGLU: y2[k] = silu(y1[k]) * y1[k+hd/2], k in [0, hd/2)
    for (uint32_t kt = 0; kt < (hiddenDim / 2) / kVecW; ++kt)
        swiglu_tile(y2Buf, y1Buf, kt, hiddenDim);

    // GMM2 + Combine: y[token][n] = weight * sum_k y2[k] * w2[e][k][n]
    for (uint32_t nt = 0; nt < h / kVecW; ++nt)
        gmm2_combine_tile(yOut, y2Buf, w2fp32, token, e, nt, h, weight);
}

} // namespace mega_moe_gmm
