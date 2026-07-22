// =============================================================================
// pre_big_fuse_pto.hpp — fused MHC pre path (norm, split, sinkhorn, apply)
// =============================================================================
//
// Source: TileKernels/tile_kernels/mhc/pre_big_fuse_kernel.py @ 36d9e45.
// This source-level PTO port keeps the fused dataflow in one function while
// using official tile operations only. All local tile fragments are >=128B.
// =============================================================================
#ifndef SUPERNPU_MHC_PRE_BIG_FUSE_PTO_HPP
#define SUPERNPU_MHC_PRE_BIG_FUSE_PTO_HPP

#include <common/pto_tileop.hpp>
#include <cstddef>
#include <cstdint>

namespace supernpu::tile_isa {
namespace mhc_pre_big_detail {

template <typename TileT>
inline void sigmoid(TileT &out, const TileT &x) {
    using namespace pto;
    TileT neg, ex, denom;
    TMULS(neg, x, -1.0f);
    TEXP(ex, neg);
    TADDS(denom, ex, 1.0f);
    TRECIP(out, denom);
}

template <typename TileM, typename TileR, typename TileC>
inline void sinkhorn(TileM &cm, float eps, int repeat) {
    using namespace pto;
    TileM tmp;
    TileR rmax, rsum, rden, rrecip;
    TileC csum, cden, crecip;

    TROWMAX(rmax, cm);
    TROWEXPAND(tmp, rmax);
    TSUB(tmp, cm, tmp);
    TEXP(cm, tmp);
    TROWSUM(rsum, cm);
    TADDS(rden, rsum, eps);
    TRECIP(rrecip, rden);
    TROWEXPAND(tmp, rrecip);
    TMUL(cm, cm, tmp);
    TADDS(cm, cm, eps);

    TCOLSUM(csum, cm);
    TADDS(cden, csum, eps);
    TRECIP(crecip, cden);
    TCOLEXPAND(tmp, crecip);
    TMUL(cm, cm, tmp);

    for (int r = 1; r < repeat; ++r) {
        TROWSUM(rsum, cm);
        TADDS(rden, rsum, eps);
        TRECIP(rrecip, rden);
        TROWEXPAND(tmp, rrecip);
        TMUL(cm, cm, tmp);
        TCOLSUM(csum, cm);
        TADDS(cden, csum, eps);
        TRECIP(crecip, cden);
        TCOLEXPAND(tmp, crecip);
        TMUL(cm, cm, tmp);
    }
}

} // namespace mhc_pre_big_detail

template <int NumTokens, int Hidden, int NSplits = 16, int MhcMult = 4, int TileH = 64>
void mhc_pre_big_fuse(float *gemm_out_mul,
                      float *gemm_out_sqrsum,
                      const float *mhc_scale,
                      const float *mhc_base,
                      __bf16 *residual,
                      float *post_mix,
                      float *comb_mix,
                      __bf16 *layer_input,
                      float rms_eps,
                      float mhc_pre_eps,
                      float mhc_sinkhorn_eps,
                      float mhc_post_mult_value,
                      int sinkhorn_repeat) {
    static_assert(MhcMult == 4, "source-level port maps MHC=4 into physical 8-wide tiles");
    static_assert(Hidden % TileH == 0 && TileH >= 64 && TileH % 8 == 0, "valid TileH required");
    constexpr int kMhc2 = MhcMult * MhcMult;
    constexpr int kMhc3 = MhcMult * 2 + kMhc2;
    constexpr int kHiddenTiles = Hidden / TileH;

    using namespace pto;
    using gm_mul = global_tensor<float, RowMajor<NSplits * NumTokens, kMhc3>>;
    using gm_sum = global_tensor<float, RowMajor<NSplits * NumTokens, 1>>;
    using gm_mix4 = global_tensor<float, RowMajor<NumTokens, MhcMult>>;
    using gm_comb_mat = global_tensor<float, RowMajor<NumTokens * MhcMult, MhcMult>>;
    using gm_res = global_tensor<__bf16, RowMajor<NumTokens * MhcMult, Hidden>>;
    using gm_li = global_tensor<__bf16, RowMajor<NumTokens, Hidden>>;
    using tile_s = Tile<Location::Vec, float, 4, 8, BLayout::RowMajor, 1, 1>;
    using tile4 = Tile<Location::Vec, float, 4, 8, BLayout::RowMajor, 1, MhcMult>;
    using tile8 = Tile<Location::Vec, float, 4, 8, BLayout::RowMajor, 1, 8>;
    using tile_cm = Tile<Location::Vec, float, 8, 8, BLayout::RowMajor, MhcMult, MhcMult>;
    using tile_res_bf = Tile<Location::Vec, __bf16, MhcMult, TileH, BLayout::RowMajor>;
    using tile_res_f = Tile<Location::Vec, float, MhcMult, TileH, BLayout::RowMajor>;
    using tile_h_bf = Tile<Location::Vec, __bf16, 1, TileH, BLayout::RowMajor>;
    using tile_h_f = Tile<Location::Vec, float, 1, TileH, BLayout::RowMajor>;
    using tile_mcol = Tile<Location::Vec, float, MhcMult, 8, BLayout::RowMajor, MhcMult, 1>;
    static_assert(tile_s::Rows * tile_s::Cols * static_cast<int>(sizeof(float)) >= 128);
    static_assert(tile4::Rows * tile4::Cols * static_cast<int>(sizeof(float)) >= 128);
    static_assert(tile8::Rows * tile8::Cols * static_cast<int>(sizeof(float)) >= 128);
    static_assert(tile_cm::Rows * tile_cm::Cols * static_cast<int>(sizeof(float)) >= 128);
    static_assert(tile_res_bf::Rows * tile_res_bf::Cols * static_cast<int>(sizeof(__bf16)) >= 128);
    static_assert(tile_res_f::Rows * tile_res_f::Cols * static_cast<int>(sizeof(float)) >= 128);
    static_assert(tile_h_bf::Rows * tile_h_bf::Cols * static_cast<int>(sizeof(__bf16)) >= 128);

    using it_mul4 = global_iterator<gm_mul, tile4>;
    using it_mul8 = global_iterator<gm_mul, tile8>;
    using it_sum = global_iterator<gm_sum, tile_s>;
    using it_mix4 = global_iterator<gm_mix4, tile4>;
    using it_cm = global_iterator<gm_comb_mat, tile_cm>;
    using it_res = global_iterator<gm_res, tile_res_bf>;
    using it_li = global_iterator<gm_li, tile_h_bf>;
    it_mul4 mul4(gemm_out_mul);
    it_mul8 mul8(gemm_out_mul);
    it_sum sum_iter(gemm_out_sqrsum);
    it_mix4 post_iter(post_mix);
    it_cm comb_iter(comb_mix);
    it_res res_iter(residual);
    it_li li_iter(layer_input);

    for (int n = 0; n < NumTokens; ++n) {
        tile_s sqsum, rms;
        TEXPANDS(sqsum, 0.0f);
        for (int s = 0; s < NSplits; ++s) {
            auto gs = sum_iter(s * NumTokens + n, 0);
            tile_s part;
            TLOAD(part, gs);
            TADD(sqsum, sqsum, part);
        }
        TMULS(rms, sqsum, 1.0f / static_cast<float>(MhcMult * Hidden));
        TADDS(rms, rms, rms_eps);
        TRSQRT(rms, rms);

        tile_mcol pre_col;
        TEXPANDS(pre_col, 0.0f);
        {
            tile4 pre_acc, post_acc, pre_scale_c, post_scale_c, pre_base_c, post_base_c;
            tile4 pre_scale, post_scale, pre_base, post_base, pre_aff, post_aff, pre_sig, post_sig, post_out;
            TEXPANDS(pre_acc, 0.0f);
            TEXPANDS(post_acc, 0.0f);
            for (int s = 0; s < NSplits; ++s) {
                tile4 p0, p1;
                auto gp0 = mul4(s * NumTokens + n, 0);
                auto gp1 = mul4(s * NumTokens + n, 1);
                TLOAD(p0, gp0);
                TLOAD(p1, gp1);
                TADD(pre_acc, pre_acc, p0);
                TADD(post_acc, post_acc, p1);
            }
            tile4 rms_full;
            TROWEXPAND(rms_full, rms);
            TMUL(pre_acc, pre_acc, rms_full);
            TMUL(post_acc, post_acc, rms_full);
            TLOAD(pre_scale_c, mhc_scale);
            TLOAD(post_scale_c, mhc_scale + 1);
            TLOAD(pre_base_c, mhc_base);
            TLOAD(post_base_c, mhc_base + MhcMult);
            TCOLEXPAND(pre_scale, pre_scale_c);
            TROWEXPAND(pre_scale, pre_scale);
            TCOLEXPAND(post_scale, post_scale_c);
            TROWEXPAND(post_scale, post_scale);
            TCOLEXPAND(pre_base, pre_base_c);
            TCOLEXPAND(post_base, post_base_c);
            TMUL(pre_aff, pre_acc, pre_scale);
            TADD(pre_aff, pre_aff, pre_base);
            TMUL(post_aff, post_acc, post_scale);
            TADD(post_aff, post_aff, post_base);
            mhc_pre_big_detail::sigmoid(pre_sig, pre_aff);
            mhc_pre_big_detail::sigmoid(post_sig, post_aff);
            TADDS(pre_sig, pre_sig, mhc_pre_eps);
            TMULS(post_out, post_sig, mhc_post_mult_value);
            auto gpost = post_iter(n, 0);
            TSTORE(gpost, post_out);
            TCOLEXPAND(pre_col, pre_sig);
        }

        tile_cm cm;
        TEXPANDS(cm, 0.0f);
        for (int tc = 0; tc < kMhc2 / 8; ++tc) {
            tile8 acc, raw, rms_full, scale_c, scale_full, base_c, base_full, affine;
            TEXPANDS(acc, 0.0f);
            for (int s = 0; s < NSplits; ++s) {
                auto gm = mul8(s * NumTokens + n, 1 + tc);
                TLOAD(raw, gm);
                TADD(acc, acc, raw);
            }
            TROWEXPAND(rms_full, rms);
            TMUL(acc, acc, rms_full);
            TLOAD(scale_c, mhc_scale + 2);
            TLOAD(base_c, mhc_base + MhcMult * 2 + tc * 8);
            TCOLEXPAND(scale_full, scale_c);
            TROWEXPAND(scale_full, scale_full);
            TCOLEXPAND(base_full, base_c);
            TMUL(affine, acc, scale_full);
            TADD(affine, affine, base_full);
            if (tc == 0) {
                TCOLEXPAND(cm, affine);
            }
        }
        mhc_pre_big_detail::sinkhorn<tile_cm, tile_mcol, tile4>(cm, mhc_sinkhorn_eps, sinkhorn_repeat);
        auto gcm = comb_iter(n * MhcMult, 0);
        TSTORE(gcm, cm);

        for (int th = 0; th < kHiddenTiles; ++th) {
            auto gr = res_iter(n, th);
            auto go = li_iter(n, th);
            tile_res_bf rb;
            tile_res_f rf, pre_full, prod;
            tile_h_f sum;
            tile_h_bf ob;
            TLOAD(rb, gr);
            TCVT(rf, rb);
            TROWEXPAND(pre_full, pre_col);
            TMUL(prod, rf, pre_full);
            TCOLSUM(sum, prod);
            TCVT(ob, sum);
            TSTORE(go, ob);
        }
    }
}

} // namespace supernpu::tile_isa
#endif
