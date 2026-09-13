#ifndef SUPERNPU_DYNAMIC_MX_QUANT_TAIL_OCP_FP8_HPP
#define SUPERNPU_DYNAMIC_MX_QUANT_TAIL_OCP_FP8_HPP

#include "multi_thread/quant/dynamic_mx_quant/dynamic_mx_quant_common.hpp"

namespace supernpu::tile_isa::mxquant {

// =====================================================================
// 未调试 (UNDEBUGGED) — OCP tail scale 核心未逐 op 对齐 AscendC，正确性存疑；
//   且 M%TileM!=0 时递归尾块会无限模板递归（默认 M=8 -> M_tail=0 侥幸绕过）。
//   编译+链接通过不构成验证。测试入口已从 test/.../src 移除。
//   状态定义/调试标准见本目录 README.md「状态总览」。
// =====================================================================
// Tail-axis, OCP scale (scaleAlg=0), FP8 output (E4M3 default, E5M2 valid).
// Reduces along BlockSize columns (TROWMAX); two-pass ComputeScale->ComputeData
// keeps peak live tiles low (see LinxV5RegisterInfo.cpp:403 spill assertion).
template <int M, int K, int TileM = 8, int BlockSize = 32, typename OutT = __fp8_e4m3>
void dynamic_mx_quant_tail_ocp_fp8(__bf16 *x, OutT *y, uint16_t *scale) {
    static_assert(M > 0 && K > 0, "dim must be positive");
    static_assert(K % BlockSize == 0, "K must be multiple of BlockSize");

    constexpr int full_m = M / TileM;
    constexpr int M_tail = M % TileM;
    constexpr int numKb  = K / BlockSize;

    using namespace pto;

    using tile_x     = Tile<Location::Vec, __bf16,   TileM, BlockSize, BLayout::RowMajor>;
    using tile_f     = Tile<Location::Vec, float,    TileM, BlockSize, BLayout::RowMajor>;
    using tile_o     = Tile<Location::Vec, OutT,     TileM, BlockSize, BLayout::RowMajor>;
    using tile_scale = Tile<Location::Vec, uint16_t, TileM, BlockSize, BLayout::RowMajor>;

    using gm_xu = global_tensor<uint16_t, RowMajor<M, K>>;
    using gm_x  = global_tensor<__bf16,   RowMajor<M, K>>;
    using gm_y  = global_tensor<uint8_t,  RowMajor<M, K>>;
    using gm_s  = global_tensor<uint16_t, RowMajor<M, numKb * BlockSize>>;

    global_iterator<gm_x,  tile_x>     x_iter(x);
    global_iterator<gm_xu, tile_scale> xu_iter(reinterpret_cast<uint16_t *>(x));
    global_iterator<gm_y,  tile_o>     y_iter(reinterpret_cast<uint8_t *>(y));
    global_iterator<gm_s,  tile_scale> s_iter(scale);

    for (int m = 0; m < full_m; ++m) {
        for (int kb = 0; kb < numKb; ++kb) {
            auto gx  = x_iter(m, kb);
            auto gxu = xu_iter(m, kb);
            auto gy  = y_iter(m, kb);
            auto gs  = s_iter(m, kb);

            // ComputeScale pass: bit-alias view only.
            tile_scale scale_byte;
            tile_scale recip;
            tile_scale x_u16;
            TLOAD(x_u16, gxu);
            compute_ocp_scale_tail<OutT, TileM, BlockSize>(x_u16, scale_byte, recip);
            TSTORE(gs, scale_byte); // store scale early; scale_byte now dead

            tile_x inv_bf16;
            // WORKAROUND: 寄存器级 reinterpret 未支持，经 HBM 字节别名规避，详见 RECORD.md 问题5
            reinterpret_u16_to_bf16<2, TileM, BlockSize>(recip, inv_bf16);
            tile_f inv_scale_f;
            TCVT(inv_scale_f, inv_bf16);

            // ComputeData pass: load the bf16 value view now.
            tile_x xq;
            TLOAD(xq, gx);
            tile_f xf;
            TCVT(xf, xq);
            TMUL(xf, xf, inv_scale_f);
            tile_o oq;
            TCVT(oq, xf);
            TSTORE(gy, oq);
        }
    }

    if constexpr (M_tail > 0) {
        dynamic_mx_quant_tail_ocp_fp8<M_tail, K, TileM, BlockSize, OutT>(
            x + full_m * K,
            reinterpret_cast<OutT *>(reinterpret_cast<uint8_t *>(y) + full_m * K),
            scale + full_m * numKb * BlockSize);
    }
}

} // namespace supernpu::tile_isa::mxquant

#endif
