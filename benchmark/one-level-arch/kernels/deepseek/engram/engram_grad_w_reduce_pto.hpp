// =============================================================================
// engram_grad_w_reduce_pto.hpp — reduce Engram partial weight gradients
// =============================================================================
//
// Source: TileKernels/tile_kernels/engram/engram_grad_w_reduce_kernel.py
// @ 36d9e45. All tensor data is moved through PTO tile iterators; scalar C++
// only controls head/block loops. All local tile fragments are >=128B.
// =============================================================================
#ifndef SUPERNPU_ENGRAM_GRAD_W_REDUCE_PTO_HPP
#define SUPERNPU_ENGRAM_GRAD_W_REDUCE_PTO_HPP

#include <common/pto_tileop.hpp>
#include <cstddef>
#include <cstdint>

namespace supernpu::tile_isa {

template <int NumPartials, int HcMult, int Hidden, int TileH = 64>
void engram_grad_w_reduce(float *grad_w_partial,
                          __bf16 *weight_hidden,
                          __bf16 *weight_embed,
                          float *grad_weight_hidden,
                          float *grad_weight_embed) {
    static_assert(NumPartials > 0 && HcMult > 0 && Hidden > 0, "dim must be positive");
    static_assert(TileH >= 64 && TileH % 8 == 0 && Hidden % TileH == 0, "TileH must be >=128B aligned");
    constexpr int kTiles = Hidden / TileH;

    using namespace pto;
    using gm_pw = global_tensor<float, RowMajor<NumPartials * HcMult, Hidden>>;
    using gm_bf = global_tensor<__bf16, RowMajor<HcMult, Hidden>>;
    using gm_f  = global_tensor<float, RowMajor<HcMult, Hidden>>;
    using tile_bf = Tile<Location::Vec, __bf16, 1, TileH, BLayout::RowMajor>;
    using tile_f  = Tile<Location::Vec, float, 1, TileH, BLayout::RowMajor>;
    static_assert(tile_bf::Rows * tile_bf::Cols * static_cast<int>(sizeof(__bf16)) >= 128);
    static_assert(tile_f::Rows * tile_f::Cols * static_cast<int>(sizeof(float)) >= 128);

    using it_pw = global_iterator<gm_pw, tile_f>;
    using it_bf = global_iterator<gm_bf, tile_bf>;
    using it_f  = global_iterator<gm_f, tile_f>;
    it_pw partial_iter(grad_w_partial);
    it_bf wh_iter(weight_hidden), we_iter(weight_embed);
    it_f gwh_iter(grad_weight_hidden), gwe_iter(grad_weight_embed);

    for (int h = 0; h < HcMult; ++h) {
        for (int th = 0; th < kTiles; ++th) {
            tile_f grad_w_sum;
            TEXPANDS(grad_w_sum, 0.0f);
            for (int p = 0; p < NumPartials; ++p) {
                auto gp = partial_iter(p * HcMult + h, th);
                tile_f part;
                TLOAD(part, gp);
                TADD(grad_w_sum, grad_w_sum, part);
            }

            auto gwh = wh_iter(h, th);
            auto gwe = we_iter(h, th);
            auto ggwh = gwh_iter(h, th);
            auto ggwe = gwe_iter(h, th);
            tile_bf whb, web;
            tile_f wh, we, old_gwh, old_gwe, add_gwh, add_gwe, out_gwh, out_gwe;
            TLOAD(whb, gwh);
            TLOAD(web, gwe);
            TLOAD(old_gwh, ggwh);
            TLOAD(old_gwe, ggwe);
            TCVT(wh, whb);
            TCVT(we, web);
            TMUL(add_gwh, grad_w_sum, we);
            TMUL(add_gwe, grad_w_sum, wh);
            TADD(out_gwh, old_gwh, add_gwh);
            TADD(out_gwe, old_gwe, add_gwe);
            TSTORE(ggwh, out_gwh);
            TSTORE(ggwe, out_gwe);
        }
    }
}

} // namespace supernpu::tile_isa
#endif
