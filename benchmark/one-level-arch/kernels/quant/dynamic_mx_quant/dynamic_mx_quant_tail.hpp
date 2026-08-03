#ifndef SUPERNPU_DYNAMIC_MX_QUANT_TAIL_HPP
#define SUPERNPU_DYNAMIC_MX_QUANT_TAIL_HPP

#include "quant/dynamic_mx_quant/dynamic_mx_quant_common.hpp"

namespace supernpu::tile_isa::mxquant {

template <int M, int K, ScaleAlg Alg = ScaleAlg::OCP, int TileM = 8, int BlockSize = 32>
void dynamic_mx_quant_tail(__bf16 *x, __fp8_e4m3 *y, uint16_t *scale) {
    static_assert(M > 0 && K > 0, "dim must be positive");
    static_assert(K % BlockSize == 0, "K must be multiple of BlockSize");

    constexpr int full_m = M / TileM;
    constexpr int M_tail = M % TileM;
    constexpr int numKb = K / BlockSize;

    using namespace pto;

    using tile_x     = Tile<Location::Vec, __bf16,     TileM, BlockSize, BLayout::RowMajor>;
    using tile_f     = Tile<Location::Vec, float,       TileM, BlockSize, BLayout::RowMajor>;
    using tile_o     = Tile<Location::Vec, __fp8_e4m3,  TileM, BlockSize, BLayout::RowMajor>;
    using tile_scale  = Tile<Location::Vec, uint16_t,   TileM, BlockSize, BLayout::RowMajor>;

    using gm_x     = global_tensor<__bf16,    RowMajor<M, K>>;
    using gm_y     = global_tensor<uint8_t,   RowMajor<M, K>>;
    using gm_s     = global_tensor<uint16_t,  RowMajor<M, numKb * BlockSize>>;

    using it_x     = global_iterator<gm_x,     tile_x>;
    using it_y     = global_iterator<gm_y,     tile_o>;
    using it_s     = global_iterator<gm_s,     tile_scale>;

    it_x x_iter(x);
    it_y y_iter(reinterpret_cast<uint8_t *>(y));
    it_s s_iter(scale);

    for (int m = 0; m < full_m; ++m) {
        for (int kb = 0; kb < numKb; ++kb) {
            auto gx     = x_iter(m, kb);
            auto gy     = y_iter(m, kb);
            auto gs     = s_iter(m, kb);

            tile_x xq;
            TLOAD(xq, gx);

            tile_scale scale_byte;
            tile_scale shared_exp;
            compute_scale_tail<Alg, TileM, BlockSize>(xq, scale_byte, shared_exp);

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

            TMUL(xf, xf, inv_scale_f);

            tile_o oq;
            TCAST(oq, xf);

            TSTORE(gs, scale_byte);
            TSTORE(gy, oq);
        }
    }

    if constexpr (M_tail > 0) {
        dynamic_mx_quant_tail<M_tail, K, Alg, TileM, BlockSize>(
            x + full_m * K,
            reinterpret_cast<__fp8_e4m3*>(reinterpret_cast<uint8_t*>(y) + full_m * K),
            scale + full_m * numKb * BlockSize);
    }
}

} // namespace supernpu::tile_isa::mxquant

#endif
