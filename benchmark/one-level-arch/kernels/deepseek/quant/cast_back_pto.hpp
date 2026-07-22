// =============================================================================
// cast_back_pto.hpp — FP8/FP4 反量化为 fp32（PTO 一层 tile 版）
// =============================================================================
//
// 【功能】
//   用 scaling factor 把量化张量还原为高精度：
//     per-token   : out[i,j] = (float)data[i,j] * sf[i]            （每行一个 scale）
//     per-channel : out[i,j] = (float)data[i,j] * sf[j]            （每列一个 scale）
//   sf_block = (1, K)（per-token）或 (M, 1)（per-channel）。
//
// 【源端】TileKernels/tile_kernels/quant/cast_back_kernel.py
//   源端 TILE_M×TILE_K 分块，T.Parallel 逐元素 out = x_shared * sf_shared。
//
// 【迁移映射】
//   FP8/FP4→fp32 升精度     → TCVT（TDEQUANT 工具链未提供，走显式 TCVT+乘）
//   per-token 行广播乘      → TROWEXPANDMUL(dst, x, sf)（sf 每行一个标量）
//   per-channel 列广播乘    → TCOLEXPANDMUL(dst, x, sf)（sf 每列一个标量）
//   T.copy                 → TLOAD / TSTORE
//
// 【约束】
//   - TileK 使 DTypeQ 与 float 都 32B 对齐；1×TileK≥128B（TLOAD 最小）。
//   - per-token: sf tile 用物理 8 宽 valid1（每行一个标量）。
//
// 【算法步骤】
//   per-token:  TLOAD(xq)→TCVT(xf)→TLOAD(sf)→TROWEXPANDMUL(of,xf,sf)→TSTORE(of)
//   per-channel: TLOAD(xq)→TCVT(xf)→TLOAD(sf)→TCOLEXPANDMUL(of,xf,sf)→TSTORE(of)
// =============================================================================
#ifndef SUPERNPU_CAST_BACK_PTO_HPP
#define SUPERNPU_CAST_BACK_PTO_HPP
#include <common/pto_tileop.hpp>
#include <cstddef>
#include <cstdint>

namespace supernpu::tile_isa {

// per-token: sf 形状 (M,1)，每行一个 scale，用 TROWEXPANDMUL 行广播乘
template <typename DTypeQ, int M, int K, int TileM = 16, int TileK = 16>
void cast_back_per_token(DTypeQ *x, float *sf, float *out) {
    static_assert(M > 0 && K > 0, "dim must be positive");
    static_assert(TileK * static_cast<int>(sizeof(DTypeQ)) % 32 == 0, "TileK*DTypeQ 须 32B 对齐");
    static_assert(TileK * static_cast<int>(sizeof(float)) % 32 == 0, "TileK*float 须 32B 对齐");
    constexpr int kTM = M / TileM, kTK = K / TileK;
    using namespace pto;
    using gm_x  = global_tensor<DTypeQ, RowMajor<M, K>>;
    using gm_sf = global_tensor<float,   RowMajor<M, 1>>;
    using gm_o  = global_tensor<float,   RowMajor<M, K>>;
    using tile_x  = Tile<Location::Vec, DTypeQ, TileM, TileK, BLayout::RowMajor>;
    using tile_f  = Tile<Location::Vec, float,   TileM, TileK, BLayout::RowMajor>;
    using tile_sf = Tile<Location::Vec, float,   TileM, 8, BLayout::RowMajor, TileM, 1>; // 每行一个 scale
    using it_x  = global_iterator<gm_x,  tile_x>;
    using it_sf = global_iterator<gm_sf, tile_sf>;
    using it_o  = global_iterator<gm_o,  tile_f>;
    it_x x_iter(x); it_sf sf_iter(sf); it_o o_iter(out);
    for (int tm = 0; tm < kTM; ++tm) {
        auto gsf = sf_iter(tm, 0);
        tile_sf sf; TLOAD(sf, gsf);                        // 每行 scale 只 load 一次，复用到所有 K-tile
        for (int tk = 0; tk < kTK; ++tk) {
            auto gx = x_iter(tm, tk); auto go = o_iter(tm, tk);
            tile_x xq; tile_f xf, of;
            TLOAD(xq, gx);                                // 量化数据 tile
            TCVT(xf, xq);                                 // 升精度到 fp32
            TROWEXPANDMUL(of, xf, sf);                   // 行广播乘：of[i,j]=xf[i,j]*sf[i]
            TSTORE(go, of);
        }
    }
}

// per-channel: sf 形状 (1,K)，每列一个 scale，用 TCOLEXPANDMUL 列广播乘
template <typename DTypeQ, int M, int K, int TileM = 16, int TileK = 16>
void cast_back_per_channel(DTypeQ *x, float *sf, float *out) {
    static_assert(M > 0 && K > 0, "dim must be positive");
    static_assert(TileK * static_cast<int>(sizeof(DTypeQ)) % 32 == 0, "TileK*DTypeQ 须 32B 对齐");
    constexpr int kTM = M / TileM, kTK = K / TileK;
    using namespace pto;
    using gm_x  = global_tensor<DTypeQ, RowMajor<M, K>>;
    using gm_sf = global_tensor<float,   RowMajor<1, K>>;
    using gm_o  = global_tensor<float,   RowMajor<M, K>>;
    using tile_x  = Tile<Location::Vec, DTypeQ, TileM, TileK, BLayout::RowMajor>;
    using tile_f  = Tile<Location::Vec, float,   TileM, TileK, BLayout::RowMajor>;
    using tile_sf = Tile<Location::Vec, float, 1, TileK, BLayout::RowMajor, 1, TileK>; // 每列一个 scale
    using it_x  = global_iterator<gm_x,  tile_x>;
    using it_sf = global_iterator<gm_sf, tile_sf>;
    using it_o  = global_iterator<gm_o,  tile_f>;
    it_x x_iter(x); it_sf sf_iter(sf); it_o o_iter(out);
    for (int tk = 0; tk < kTK; ++tk) {
        auto gsf = sf_iter(0, tk);
        tile_sf sf; TLOAD(sf, gsf);                        // 每列 scale 只 load 一次
        for (int tm = 0; tm < kTM; ++tm) {
            auto gx = x_iter(tm, tk); auto go = o_iter(tm, tk);
            tile_x xq; tile_f xf, of;
            TLOAD(xq, gx);
            TCVT(xf, xq);
            TCOLEXPANDMUL(of, xf, sf);                   // 列广播乘：of[i,j]=xf[i,j]*sf[j]
            TSTORE(go, of);
        }
    }
}

} // namespace supernpu::tile_isa
#endif
