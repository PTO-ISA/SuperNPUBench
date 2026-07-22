// =============================================================================
// engram_gate_pto.hpp — Engram gate forward/backward (source-level PTO tile port)
// =============================================================================
//
// Source: TileKernels/tile_kernels/engram/engram_gate_kernel.py @ 36d9e45.
// Data movement and tensor math are expressed with official PTO tile operations.
// Scalar C++ only sequences token/head/hidden loops and scalar parameters.
// All thread-local tile fragments below are >=128B.
// =============================================================================
#ifndef SUPERNPU_ENGRAM_GATE_PTO_HPP
#define SUPERNPU_ENGRAM_GATE_PTO_HPP

#include <common/pto_tileop.hpp>
#include <cstddef>
#include <cstdint>

namespace supernpu::tile_isa {
namespace engram_gate_detail {

template <typename TileT>
constexpr bool tile_fragment_at_least_128b() {
    return TileT::Rows * TileT::Cols * static_cast<int>(sizeof(typename TileT::DType)) >= 128;
}

template <typename TileT>
inline void sigmoid(TileT &out, const TileT &x) {
    using namespace pto;
    TileT neg, ex, denom;
    TMULS(neg, x, -1.0f);
    TEXP(ex, neg);
    TADDS(denom, ex, 1.0f);
    TRECIP(out, denom);
}

template <typename TileT>
inline void signed_sqrt_gate(TileT &out, const TileT &x, float clamp_value) {
    using namespace pto;
    TileT ax, clamped, root;
    TABS(ax, x);
    TMAXS(clamped, ax, clamp_value);
    TSQRT(root, clamped);
    TDIV(out, x, root);
}

template <typename TileOut, typename TileIn, typename TileRow>
inline void row_broadcast_mul(TileOut &out, const TileIn &x, const TileRow &row) {
    using namespace pto;
    TileOut row_full;
    TROWEXPAND(row_full, row);
    TMUL(out, x, row_full);
}

} // namespace engram_gate_detail

template <int NumTokens, int HcMult, int Hidden, int TileH = 64>
void engram_gate_fwd(__bf16 *hidden_states,
                     __bf16 *k,
                     __bf16 *v,
                     float *__restrict weight_fused,
                     __bf16 *output,
                     float *dot_out,
                     float *gate_score,
                     float *rstd_x,
                     float *rstd_k,
                     float eps = 1.0e-6f,
                     float clamp_value = 1.0e-6f) {
    static_assert(NumTokens > 0 && HcMult > 0 && Hidden > 0, "dim must be positive");
    static_assert(TileH >= 64 && TileH % 8 == 0, "TileH must provide >=128B bf16 fragments and 32B float alignment");
    static_assert(Hidden % TileH == 0, "Hidden must be divisible by TileH");
    const float scalar = 1.0f / __builtin_sqrtf(static_cast<float>(Hidden));
    constexpr int kTiles = Hidden / TileH;

    using namespace pto;
    using gm_hc_bf = global_tensor<__bf16, RowMajor<NumTokens * HcMult, Hidden>>;
    using gm_v_bf  = global_tensor<__bf16, RowMajor<NumTokens, Hidden>>;
    using gm_w     = global_tensor<float, RowMajor<HcMult, Hidden>>;
    using gm_s     = global_tensor<float, RowMajor<NumTokens * HcMult, 1>>;
    using tile_bf  = Tile<Location::Vec, __bf16, 1, TileH, BLayout::RowMajor>;
    using tile_f   = Tile<Location::Vec, float, 1, TileH, BLayout::RowMajor>;
    using tile_s   = Tile<Location::Vec, float, 4, 8, BLayout::RowMajor, 1, 1>;
    static_assert(engram_gate_detail::tile_fragment_at_least_128b<tile_bf>());
    static_assert(engram_gate_detail::tile_fragment_at_least_128b<tile_f>());
    static_assert(engram_gate_detail::tile_fragment_at_least_128b<tile_s>());

    using it_hc_bf = global_iterator<gm_hc_bf, tile_bf>;
    using it_v_bf  = global_iterator<gm_v_bf, tile_bf>;
    using it_w     = global_iterator<gm_w, tile_f>;
    using it_s     = global_iterator<gm_s, tile_s>;
    it_hc_bf x_iter(hidden_states), k_iter(k), out_iter(output);
    it_v_bf v_iter(v);
    it_w w_iter(weight_fused);
    it_s dot_iter(dot_out), gate_iter(gate_score), rx_iter(rstd_x), rk_iter(rstd_k);

    for (int n = 0; n < NumTokens; ++n) {
        for (int h = 0; h < HcMult; ++h) {
            const int row = n * HcMult + h;
            tile_s sx2, sk2, sdot;
            TEXPANDS(sx2, 0.0f);
            TEXPANDS(sk2, 0.0f);
            TEXPANDS(sdot, 0.0f);
            for (int th = 0; th < kTiles; ++th) {
                auto gx = x_iter(row, th);
                auto gk = k_iter(row, th);
                auto gw = w_iter(h, th);
                tile_bf xb, kb;
                tile_f xf, kf, wf, x2, k2, xw, xwk;
                tile_s part;
                TLOAD(xb, gx);
                TLOAD(kb, gk);
                TLOAD(wf, gw);
                TCVT(xf, xb);
                TCVT(kf, kb);
                TMUL(x2, xf, xf);
                TMUL(k2, kf, kf);
                TROWSUM(part, x2);
                TADD(sx2, sx2, part);
                TROWSUM(part, k2);
                TADD(sk2, sk2, part);
                TMUL(xw, xf, wf);
                TMUL(xwk, xw, kf);
                TROWSUM(part, xwk);
                TADD(sdot, sdot, part);
            }

            tile_s mean_x, mean_k, denom_x, denom_k, rx, rk, gate_norm, tmp, gate;
            TMULS(mean_x, sx2, 1.0f / static_cast<float>(Hidden));
            TMULS(mean_k, sk2, 1.0f / static_cast<float>(Hidden));
            TADDS(denom_x, mean_x, eps);
            TADDS(denom_k, mean_k, eps);
            TRSQRT(rx, denom_x);
            TRSQRT(rk, denom_k);
            TMUL(tmp, sdot, rx);
            TMUL(gate_norm, tmp, rk);
            TMULS(gate_norm, gate_norm, scalar);
            engram_gate_detail::signed_sqrt_gate(tmp, gate_norm, clamp_value);
            engram_gate_detail::sigmoid(gate, tmp);

            auto gd = dot_iter(row, 0);
            auto gg = gate_iter(row, 0);
            auto grx = rx_iter(row, 0);
            auto grk = rk_iter(row, 0);
            TSTORE(gd, sdot);
            TSTORE(gg, gate);
            TSTORE(grx, rx);
            TSTORE(grk, rk);

            for (int th = 0; th < kTiles; ++th) {
                auto gx = x_iter(row, th);
                auto gv = v_iter(n, th);
                auto go = out_iter(row, th);
                tile_bf xb, vb, ob;
                tile_f xf, vf, gvprod, of;
                TLOAD(xb, gx);
                TLOAD(vb, gv);
                TCVT(xf, xb);
                TCVT(vf, vb);
                engram_gate_detail::row_broadcast_mul(gvprod, vf, gate);
                TADD(of, xf, gvprod);
                TCVT(ob, of);
                TSTORE(go, ob);
            }
        }
    }
}

template <int NumTokens, int HcMult, int Hidden, int NumPartials = 1, int TileH = 64>
void engram_gate_bwd(__bf16 *grad_out,
                     __bf16 *hidden_states,
                     __bf16 *k,
                     __bf16 *v,
                     float *weight_fused,
                     float *dot_in,
                     float *gate_in,
                     float *rstd_x_in,
                     float *rstd_k_in,
                     __bf16 *grad_x,
                     __bf16 *grad_k,
                     __bf16 *grad_v,
                     float *grad_w_partial,
                     float clamp_value = 1.0e-6f) {
    static_assert(NumPartials == 1, "source-level PTO port accumulates one partial block");
    static_assert(TileH >= 64 && TileH % 8 == 0 && Hidden % TileH == 0, "valid TileH required");
    const float scalar = 1.0f / __builtin_sqrtf(static_cast<float>(Hidden));
    constexpr int kTiles = Hidden / TileH;

    using namespace pto;
    using gm_hc_bf = global_tensor<__bf16, RowMajor<NumTokens * HcMult, Hidden>>;
    using gm_v_bf  = global_tensor<__bf16, RowMajor<NumTokens, Hidden>>;
    using gm_w     = global_tensor<float, RowMajor<HcMult, Hidden>>;
    using gm_s     = global_tensor<float, RowMajor<NumTokens * HcMult, 1>>;
    using gm_pw    = global_tensor<float, RowMajor<NumPartials * HcMult, Hidden>>;
    using tile_bf  = Tile<Location::Vec, __bf16, 1, TileH, BLayout::RowMajor>;
    using tile_f   = Tile<Location::Vec, float, 1, TileH, BLayout::RowMajor>;
    using tile_s   = Tile<Location::Vec, float, 4, 8, BLayout::RowMajor, 1, 1>;
    static_assert(engram_gate_detail::tile_fragment_at_least_128b<tile_bf>());
    static_assert(engram_gate_detail::tile_fragment_at_least_128b<tile_f>());
    static_assert(engram_gate_detail::tile_fragment_at_least_128b<tile_s>());

    using it_hc_bf = global_iterator<gm_hc_bf, tile_bf>;
    using it_v_bf  = global_iterator<gm_v_bf, tile_bf>;
    using it_w     = global_iterator<gm_w, tile_f>;
    using it_s     = global_iterator<gm_s, tile_s>;
    using it_pw    = global_iterator<gm_pw, tile_f>;
    it_hc_bf go_iter(grad_out), x_iter(hidden_states), k_iter(k), gx_iter(grad_x), gk_iter(grad_k);
    it_v_bf v_iter(v), gv_iter(grad_v);
    it_w w_iter(weight_fused);
    it_s dot_iter(dot_in), gate_iter(gate_in), rx_iter(rstd_x_in), rk_iter(rstd_k_in);
    it_pw gwp_iter(grad_w_partial);

    for (int n = 0; n < NumTokens; ++n) {
        for (int th = 0; th < kTiles; ++th) {
            tile_f grad_v_acc;
            TEXPANDS(grad_v_acc, 0.0f);
            for (int h = 0; h < HcMult; ++h) {
                const int row = n * HcMult + h;
                auto ggo = go_iter(row, th);
                auto ggate = gate_iter(row, 0);
                tile_bf gob;
                tile_f gof, addend;
                tile_s gate;
                TLOAD(gob, ggo);
                TLOAD(gate, ggate);
                TCVT(gof, gob);
                engram_gate_detail::row_broadcast_mul(addend, gof, gate);
                TADD(grad_v_acc, grad_v_acc, addend);
            }
            auto ggv = gv_iter(n, th);
            tile_bf gvb;
            TCVT(gvb, grad_v_acc);
            TSTORE(ggv, gvb);
        }
    }

    for (int h = 0; h < HcMult; ++h) {
        for (int th = 0; th < kTiles; ++th) {
            tile_f gw_acc;
            TEXPANDS(gw_acc, 0.0f);
            for (int n = 0; n < NumTokens; ++n) {
                const int row = n * HcMult + h;
                tile_s dldg, dotv, gate, rx, rk, one_minus_gate, gate_deriv;
                TEXPANDS(dldg, 0.0f);
                for (int tv = 0; tv < kTiles; ++tv) {
                    auto ggo = go_iter(row, tv);
                    auto gv = v_iter(n, tv);
                    tile_bf gob, vb;
                    tile_f gof, vf, prod;
                    tile_s part;
                    TLOAD(gob, ggo);
                    TLOAD(vb, gv);
                    TCVT(gof, gob);
                    TCVT(vf, vb);
                    TMUL(prod, gof, vf);
                    TROWSUM(part, prod);
                    TADD(dldg, dldg, part);
                }

                auto gd = dot_iter(row, 0);
                auto gg = gate_iter(row, 0);
                auto grx = rx_iter(row, 0);
                auto grk = rk_iter(row, 0);
                TLOAD(dotv, gd);
                TLOAD(gate, gg);
                TLOAD(rx, grx);
                TLOAD(rk, grk);
                TSUBS(one_minus_gate, gate, 1.0f);
                TMULS(one_minus_gate, one_minus_gate, -1.0f);
                TMUL(gate_deriv, gate, one_minus_gate);
                TMUL(dldg, dldg, gate_deriv);
                TMULS(dldg, dldg, 0.5f);
                tile_s absdot, safe_absdot, rr, ratio, root, recip_root;
                TABS(absdot, dotv);
                TMAXS(safe_absdot, absdot, clamp_value);
                TMUL(rr, rx, rk);
                TMULS(rr, rr, scalar);
                TDIV(ratio, rr, safe_absdot);
                TSQRT(root, ratio);
                TMUL(dldg, dldg, root);

                tile_s dot_x, dot_k;
                TMUL(dot_x, dotv, rx);
                TMUL(dot_x, dot_x, rx);
                TMULS(dot_x, dot_x, 1.0f / static_cast<float>(Hidden));
                TMUL(dot_k, dotv, rk);
                TMUL(dot_k, dot_k, rk);
                TMULS(dot_k, dot_k, 1.0f / static_cast<float>(Hidden));

                auto gx = x_iter(row, th);
                auto gk = k_iter(row, th);
                auto gw = w_iter(h, th);
                auto ggo = go_iter(row, th);
                tile_bf xb, kb, gob, gxb, gkb;
                tile_f xf, kf, wf, gof, dldg_full, dot_full, lhs, rhs, outx, outk, term;
                TLOAD(xb, gx);
                TLOAD(kb, gk);
                TLOAD(wf, gw);
                TLOAD(gob, ggo);
                TCVT(xf, xb);
                TCVT(kf, kb);
                TCVT(gof, gob);
                TROWEXPAND(dldg_full, dldg);

                TMUL(lhs, kf, wf);
                TROWEXPAND(dot_full, dot_x);
                TMUL(rhs, xf, dot_full);
                TSUB(term, lhs, rhs);
                TMUL(term, term, dldg_full);
                TADD(outx, gof, term);

                TMUL(lhs, xf, wf);
                TROWEXPAND(dot_full, dot_k);
                TMUL(rhs, kf, dot_full);
                TSUB(term, lhs, rhs);
                TMUL(outk, term, dldg_full);

                TMUL(term, xf, kf);
                TMUL(term, term, dldg_full);
                TADD(gw_acc, gw_acc, term);

                auto ggx = gx_iter(row, th);
                auto ggk = gk_iter(row, th);
                TCVT(gxb, outx);
                TCVT(gkb, outk);
                TSTORE(ggx, gxb);
                TSTORE(ggk, gkb);
            }
            auto ggw = gwp_iter(h, th);
            TSTORE(ggw, gw_acc);
        }
    }
}

} // namespace supernpu::tile_isa
#endif
