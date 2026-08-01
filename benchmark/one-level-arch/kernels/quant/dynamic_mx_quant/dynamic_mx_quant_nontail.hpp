#ifndef SUPERNPU_DYNAMIC_MX_QUANT_NONTAIL_HPP
#define SUPERNPU_DYNAMIC_MX_QUANT_NONTAIL_HPP

#include <common/pto_tileop.hpp>
#include <cstdint>
#include "quant/dynamic_mx_quant/dynamic_mx_quant_common.hpp"

namespace supernpu::tile_isa::mxquant {

template <int Pre, int Axis, int Post, ScaleAlg Alg = ScaleAlg::OCP, int TileM = 8, int BlockSize = 32>
void dynamic_mx_quant_nontail(__bf16 *x, __fp8_e4m3 *y, uint16_t *scale) {
    static_assert(Pre > 0 && Axis > 0 && Post > 0, "dims must be positive");
    static_assert(Axis % BlockSize == 0, "Axis must be multiple of BlockSize");
    static_assert(Pre * Post % TileM == 0, "Pre * Post must be multiple of TileM");
    static_assert(BlockSize == 32, "only BlockSize=32 supported");

    constexpr int M = Pre * Post;
    constexpr int K = Axis;
    constexpr int kTM = M / TileM;
    constexpr int numKb = K / BlockSize;

    using namespace pto;

    using tile_x    = Tile<Location::Vec, __bf16,     TileM, BlockSize, BLayout::RowMajor>;
    using tile_f    = Tile<Location::Vec, float,       TileM, BlockSize, BLayout::RowMajor>;
    using tile_o    = Tile<Location::Vec, __fp8_e4m3,  TileM, BlockSize, BLayout::RowMajor>;
    using tile_amax_f = Tile<Location::Vec, float,     TileM, BlockSize, BLayout::RowMajor>;
    using tile_scale = Tile<Location::Vec, uint16_t,   TileM, BlockSize, BLayout::RowMajor>;

    using gm_x     = global_tensor<__bf16,    RowMajor<M, K>>;
    using gm_y     = global_tensor<uint8_t,   RowMajor<M, K>>;
    using gm_s     = global_tensor<uint16_t,  RowMajor<M, numKb * BlockSize>>;

    using it_x     = global_iterator<gm_x,     tile_x>;
    using it_y     = global_iterator<gm_y,     tile_o>;
    using it_s     = global_iterator<gm_s,     tile_scale>;

    it_x x_iter(x);
    it_y y_iter(reinterpret_cast<uint8_t *>(y));
    it_s s_iter(scale);

    for (int m = 0; m < kTM; ++m) {
        for (int kb = 0; kb < numKb; ++kb) {
            auto gx     = x_iter(m, kb);
            auto gy     = y_iter(m, kb);
            auto gs     = s_iter(m, kb);

            tile_x xq;
            TLOAD(xq, gx);

            tile_scale scale_byte;
            compute_scale_byte<Alg, TileM, BlockSize>(xq, scale_byte);

            tile_f xf;
            TCVT(xf, xq);

            tile_f absx;
            TABS(absx, xf);

            tile_amax_f amax_f;
            TROWMAX(amax_f, absx);
            TMAXS(amax_f, amax_f, CLAMP_MIN);

            tile_amax_f inv;
            TRECIP(inv, amax_f);

            tile_amax_f sfinv;
            TMULS(sfinv, inv, FP8_E4M3_DST_MAX);

            tile_f outf;
            TROWEXPANDMUL(outf, xf, sfinv);

            tile_o oq;
            TCAST(oq, outf);

            TSTORE(gs, scale_byte);
            TSTORE(gy, oq);
        }
    }
}

} // namespace supernpu::tile_isa::mxquant

#endif
