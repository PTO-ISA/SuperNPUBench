#ifndef SUPERNPU_DYNAMIC_MX_QUANT_NONTAIL_HPP
#define SUPERNPU_DYNAMIC_MX_QUANT_NONTAIL_HPP

#include <common/pto_tileop.hpp>
#include <cstdint>
#include "quant/dynamic_mx_quant/dynamic_mx_quant_common.hpp"

namespace supernpu::tile_isa::mxquant {

template <int Axis, int Post, ScaleAlg Alg = ScaleAlg::OCP, int BlockSize = 32, int TileN = 32>
void dynamic_mx_quant_nontail(__bf16 *x, __fp8_e4m3 *y, uint16_t *scale) {
    static_assert(Axis > 0 && Post > 0, "dims must be positive");
    static_assert(Axis % BlockSize == 0, "Axis must be multiple of BlockSize");
    static_assert(Post % TileN == 0, "Post must be multiple of TileN");
    static_assert(BlockSize == 32, "only BlockSize=32 supported");

    constexpr int numKb = Axis / BlockSize;
    constexpr int numN  = Post / TileN;

    using namespace pto;

    using tile_x      = Tile<Location::Vec, __bf16,    BlockSize, TileN, BLayout::RowMajor>;
    using tile_f      = Tile<Location::Vec, float,     BlockSize, TileN, BLayout::RowMajor>;
    using tile_o      = Tile<Location::Vec, __fp8_e4m3, BlockSize, TileN, BLayout::RowMajor>;
    using tile_scale  = Tile<Location::Vec, uint16_t,  BlockSize, TileN, BLayout::RowMajor>;

    using gm_x = global_tensor<__bf16,  RowMajor<Axis, Post>>;
    using gm_y = global_tensor<uint8_t, RowMajor<Axis, Post>>;
    using gm_s = global_tensor<uint16_t, RowMajor<numKb, Post>>;

    using it_x = global_iterator<gm_x, tile_x>;
    using it_y = global_iterator<gm_y, tile_o>;
    using it_s = global_iterator<gm_s, tile_scale>;

    it_x x_iter(x);
    it_y y_iter(reinterpret_cast<uint8_t *>(y));
    it_s s_iter(scale);

    for (int kb = 0; kb < numKb; ++kb) {
        for (int n = 0; n < numN; ++n) {
            auto gx = x_iter(kb, n);
            auto gy = y_iter(kb, n);
            auto gs = s_iter(kb, n);

            tile_x xq;
            TLOAD(xq, gx);

            tile_scale scale_byte;
            tile_scale shared_exp;
            compute_scale_not_tail<Alg, BlockSize, TileN>(xq, scale_byte, shared_exp);

            tile_f xf;
            TCVT(xf, xq);

            tile_scale neg_exp;
            TXORS(neg_exp, shared_exp, static_cast<uint16_t>(0xFFFF));

            tile_scale inv_scale;
            TADDS(inv_scale, neg_exp, static_cast<uint16_t>(BF16_SCALE_BIAS + 1));

            tile_x inv_bf16;
            TCAST(inv_bf16, inv_scale);

            tile_f inv_scale_f;
            TCVT(inv_scale_f, inv_bf16);

            TCOLEXPANDMUL(xf, xf, inv_scale_f);

            tile_o oq;
            TCAST(oq, xf);

            TSTORE(gs, scale_byte);
            TSTORE(gy, oq);
        }
    }
}

} // namespace supernpu::tile_isa::mxquant

#endif
