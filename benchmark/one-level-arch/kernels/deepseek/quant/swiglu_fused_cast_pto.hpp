// =============================================================================
// swiglu_fused_cast_pto.hpp — 融合 SwiGLU 前向 + per-token FP8 量化（tile 版）
// =============================================================================
//
// 【功能】
//   输入 (M, 2*Hidden)：前半 gate、后半 up。计算 silu(gate)*up，再按 per-token 块量化缩放：
//     silu(g) = g / (1 + exp(-g)) = g * sigmoid(g)
//     swiglu  = silu(gate) * up
//     out     = swiglu * (max_val / amax(swiglu))      （per-token 块量化）
//   同一 tile 内先 SwiGLU 后量化，省一次 GM 读写。
//
// 【源端】TileKernels/tile_kernels/quant/swiglu_forward_and_per_token_cast_kernel.py
//
// 【迁移映射】
//   silu(g)=g*sigmoid(g)     → TMULS(g,-1)[TNEG 未提供]→TEXP→TADDS(+1)→TRECIP→TMUL(g*sig)
//   gate*up                  → TMUL
//   amax=max|x|              → TMULS(x,-1)+TROWMAX+TMAX（TABS 未提供，同 per_token_cast）
//   量化缩放                 → TRECIP+TMULS+TROWEXPANDMUL（同 per_token_cast）
//   bf16↔fp32                → TCVT
//
// 【约束】Hidden 须被 Npc 整除；Npc 须 16 的倍数（bf16）。
//
// 【算法步骤（每 (tm, group)）】
//   1) TLOAD gate/up bf16→TCVT fp32
//   2) silu: TMULS(-1)→TEXP→TADDS(+1)→TRECIP→TMUL(g*sigmoid)
//   3) TMUL(swiglu, silu, up)
//   4) 量化链（同 per_token_cast step2-7）
// =============================================================================
#ifndef SUPERNPU_SWIGLU_FUSED_CAST_PTO_HPP
#define SUPERNPU_SWIGLU_FUSED_CAST_PTO_HPP
#include <common/pto_tileop.hpp>
#include <cstddef>
#include <cstdint>

namespace supernpu::tile_isa {

template <int M, int Hidden, int Npc, int TileM = 16>
void swiglu_forward_and_per_token_cast(__bf16 *x, __bf16 *out, float *out_sf,
                                       float max_val, float clamp_min) {
    static_assert(M > 0 && Hidden > 0 && Npc > 0, "dim must be positive");
    static_assert(Hidden % Npc == 0 && Npc % 16 == 0, "Hidden%Npc==0 and Npc%16==0 (bf16 32B align)");
    constexpr int kTM = M / TileM;
    const int num_groups = Hidden / Npc;
    using namespace pto;
    using gm_x  = global_tensor<__bf16, RowMajor<M, 1>>;
    using gm_sf = global_tensor<float,   RowMajor<M, 1>>;
    using gm_o  = global_tensor<__bf16, RowMajor<M, 1>>;
    using tile_x = Tile<Location::Vec, __bf16, TileM, Npc, BLayout::RowMajor>;
    using tile_f = Tile<Location::Vec, float,   TileM, Npc, BLayout::RowMajor>;
    using tile_amax = Tile<Location::Vec, float, TileM, 8, BLayout::RowMajor, TileM, 1>;
    using it_x = global_iterator<gm_x, tile_x>;
    using it_sf = global_iterator<gm_sf, tile_amax>;
    using it_o = global_iterator<gm_o, tile_x>;
    it_x x_iter(x); it_sf sf_iter(out_sf); it_o o_iter(out);

    for (int tm = 0; tm < kTM; ++tm) {
        for (int g = 0; g < num_groups; ++g) {
            auto gg = x_iter(tm, g); auto gu = x_iter(tm, num_groups + g); // gate | up
            tile_x gq, uq, oq; tile_f gf, uf, silu, sw, neg_g, expn, op, sig, neg_s, pos, negm, amax, sf, inv, sfinv, outf;
            TLOAD(gq, gg); TLOAD(uq, gu);                 // 1) gate/up bf16
            TCVT(gf, gq); TCVT(uf, uq);                  //    bf16→fp32
            // 2) silu(g) = g * sigmoid(g) = g / (1+exp(-g))
            TMULS(neg_g, gf, -1.0f);                     //    -g（TNEG 未提供）
            TEXP(expn, neg_g);                            //    exp(-g)
            TADDS(op, expn, 1.0f);                        //    1 + exp(-g)
            TRECIP(sig, op);                              //    sigmoid = 1/(1+exp(-g))
            TMUL(silu, gf, sig);                         //    silu = g * sigmoid
            TMUL(sw, silu, uf);                           // 3) swiglu = silu * up
            // 4) per-token 量化链（amax→sf→缩放，同 per_token_cast）
            TMULS(neg_s, sw, -1.0f);
            TROWMAX(pos, sw);
            TROWMAX(negm, neg_s);
            TMAX(amax, pos, negm);                        //    amax = max|swiglu|
            TMAXS(amax, amax, clamp_min);
            TMULS(sf, amax, 1.0f / max_val);
            TRECIP(inv, amax);
            TMULS(sfinv, inv, max_val);
            TROWEXPANDMUL(outf, sw, sfinv);              //    缩放
            TCVT(oq, outf);                              //    fp32→bf16
            auto gsf = sf_iter(tm, g); auto gout = o_iter(tm, g);
            TSTORE(gsf, sf);
            TSTORE(gout, oq);
        }
    }
}

} // namespace supernpu::tile_isa
#endif
