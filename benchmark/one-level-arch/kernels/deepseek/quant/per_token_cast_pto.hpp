// =============================================================================
// per_token_cast_pto.hpp — 行级/列级 FP8 量化（per-token / per-channel scaling）
// =============================================================================
//
// 【功能】
//   按 (1 token × Npc channels) 块求绝对值最大 amax，得 scale，把数据缩放到目标动态范围：
//     amax[row]  = max_j |data[row, j]|
//     sf[row]    = amax / max_val
//     out        = data * (max_val / amax)        （等价于除以 sf）
//   per_channel 同理但 amax 沿列方向归约，每列一个 scale。
//
// 【源端】TileKernels/tile_kernels/quant/per_token_cast_kernel.py、per_channel_cast_kernel.py
//   源端 reshape + reduce_absmax(dim) 二维归约。
//
// 【迁移映射】
//   reduce_absmax              → TABS+TROWMAX 模拟（TABS 工具链未提供）：
//                                absmax = max(TROWMAX(x), TROWMAX(-x))，其中 -x = TMULS(x,-1)
//   行 amax                    → TROWMAX
//   列 amax                    → TCOLMAX
//   sf = amax/max_val          → TMULS(amax, 1/max_val)
//   sf_inv = max_val/amax      → TRECIP(amax) 后 TMULS(*, max_val)
//   行广播乘 out = x*sf_inv    → TROWEXPANDMUL
//   列广播乘                   → TCOLEXPANDMUL
//   clamp 下限                 → TMAXS
//   bf16↔fp32                  → TCVT
//
// 【约束】
//   - Npc 须为 16 的倍数（bf16 列宽 32B 对齐）；TileM 使 tile_mat≥128B（默认 16→512B）。
//   - amax 输出 tile 用物理 8 宽 valid1（每行/列一个标量）。
//
// 【算法步骤（per-token 每 (tm, group)）】
//   1) TLOAD x(bf16)→TCVT xf(fp32)
//   2) TMULS(neg, xf, -1) 得 -x
//   3) TROWMAX(pos, xf) + TROWMAX(negm, neg) + TMAX(amax,pos,negm) = max|x|
//   4) TMAXS(amax, amax, clamp_min) 防 0
//   5) TMULS(sf, amax, 1/max_val) ; TRECIP(inv, amax) ; TMULS(sfinv, inv, max_val)
//   6) TROWEXPANDMUL(out, xf, sfinv) → 缩放
//   7) TCVT(oq, out) bf16; TSTORE sf 与 oq
// =============================================================================
#ifndef SUPERNPU_PER_TOKEN_CAST_PTO_HPP
#define SUPERNPU_PER_TOKEN_CAST_PTO_HPP
#include <common/pto_tileop.hpp>
#include <cstddef>
#include <cstdint>

namespace supernpu::tile_isa {

// per-token: 每行一个 scale（sf_block = (1, Npc)）
template <int M, int Npc, int TileM = 16>
void per_token_cast(__bf16 *x, int hidden, float *out_sf, __bf16 *out,
                    float max_val, float clamp_min) {
    static_assert(M > 0 && Npc > 0, "dim must be positive");
    static_assert(Npc % 16 == 0, "Npc must be multiple of 16 (bf16 32B align)");
    constexpr int kTM = M / TileM;
    const int num_groups = hidden / Npc;                   // 每行 scale 个数
    using namespace pto;
    using gm_x  = global_tensor<__bf16, RowMajor<M, 1>>;
    using gm_sf = global_tensor<float,   RowMajor<M, 1>>;
    using gm_o  = global_tensor<__bf16, RowMajor<M, 1>>;
    using tile_x = Tile<Location::Vec, __bf16, TileM, Npc, BLayout::RowMajor>;
    using tile_f = Tile<Location::Vec, float,   TileM, Npc, BLayout::RowMajor>;
    using tile_amax = Tile<Location::Vec, float, TileM, 8, BLayout::RowMajor, TileM, 1>; // 每行一个 amax
    using it_x  = global_iterator<gm_x,  tile_x>;
    using it_sf = global_iterator<gm_sf, tile_amax>;
    using it_o  = global_iterator<gm_o,  tile_x>;
    it_x x_iter(x); it_sf sf_iter(out_sf); it_o o_iter(out);
    for (int tm = 0; tm < kTM; ++tm) {
        for (int g = 0; g < num_groups; ++g) {
            auto gx = x_iter(tm, g); auto gsf = sf_iter(tm, g); auto go = o_iter(tm, g);
            tile_x xq, oq; tile_f xf, neg, outf;
            tile_amax pos, negm, amax, sf, inv, sfinv;
            TLOAD(xq, gx);                               // 1) 输入 bf16
            TCVT(xf, xq);                                //    bf16→fp32
            TMULS(neg, xf, -1.0f);                       // 2) -x（TNEG 未提供）
            TROWMAX(pos, xf);                            // 3) 行最大正值
            TROWMAX(negm, neg);                          //    行最大负值（绝对值）
            TMAX(amax, pos, negm);                        //    amax = max|x|
            TMAXS(amax, amax, clamp_min);                // 4) clamp 下限防 0
            TMULS(sf, amax, 1.0f / max_val);             // 5) sf = amax/max_val
            TRECIP(inv, amax);                            //    1/amax
            TMULS(sfinv, inv, max_val);                  //    sf_inv = max_val/amax
            TROWEXPANDMUL(outf, xf, sfinv);              // 6) 行广播乘缩放
            TCVT(oq, outf);                              // 7) fp32→bf16
            TSTORE(gsf, sf);                             //    写 scale
            TSTORE(go, oq);                                //    写量化结果
        }
    }
}

// per-channel: 每列一个 scale（sf_block = (Npt, 1)），amax 沿行方向归约
template <int Npt, int Hidden, int TileK = 16>
void per_channel_cast(__bf16 *x, __bf16 *out, float *out_sf,
                      float max_val, float clamp_min) {
    static_assert(Npt > 0 && Hidden > 0, "dim must be positive");
    static_assert(TileK % 16 == 0, "TileK must be multiple of 16 (bf16 32B align)");
    constexpr int kTK = Hidden / TileK;
    using namespace pto;
    using gm_x  = global_tensor<__bf16, RowMajor<Npt, Hidden>>;
    using gm_sf = global_tensor<float,   RowMajor<1, Hidden>>;
    using gm_o  = global_tensor<__bf16, RowMajor<Npt, Hidden>>;
    using tile_x = Tile<Location::Vec, __bf16, Npt, TileK, BLayout::RowMajor>;
    using tile_f = Tile<Location::Vec, float,   Npt, TileK, BLayout::RowMajor>;
    using tile_amax = Tile<Location::Vec, float, 1, TileK, BLayout::RowMajor, 1, TileK>; // 每列一个 amax
    using it_x  = global_iterator<gm_x,  tile_x>;
    using it_sf = global_iterator<gm_sf, tile_amax>;
    using it_o  = global_iterator<gm_o,  tile_x>;
    it_x x_iter(x); it_sf sf_iter(out_sf); it_o o_iter(out);
    for (int tk = 0; tk < kTK; ++tk) {
        auto gx = x_iter(0, tk); auto gsf = sf_iter(0, tk); auto go = o_iter(0, tk);
        tile_x xq, oq; tile_f xf, neg, outf;
        tile_amax pos, negm, amax, sf, inv, sfinv;
        TLOAD(xq, gx);
        TCVT(xf, xq);
        TMULS(neg, xf, -1.0f);
        TCOLMAX(pos, xf);                                // 列最大（沿行归约）
        TCOLMAX(negm, neg);
        TMAX(amax, pos, negm);
        TMAXS(amax, amax, clamp_min);
        TMULS(sf, amax, 1.0f / max_val);
        TRECIP(inv, amax);
        TMULS(sfinv, inv, max_val);
        TCOLEXPANDMUL(outf, xf, sfinv);                 // 列广播乘缩放
        TCVT(oq, outf);
        TSTORE(gsf, sf);
        TSTORE(go, oq);
    }
}

} // namespace supernpu::tile_isa
#endif
