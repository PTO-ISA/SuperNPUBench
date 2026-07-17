// =============================================================================
// mask_indices_by_tp_pto.hpp — 按 TP rank 屏蔽并转局部专家索引（tile 版）
// =============================================================================
//
// 【功能】
//   全局专家索引 → 不属本 TP rank 或 <0 置 -1；属则转本 rank 局部索引。
//   q = idx // per_gpu; r = q % num_tp_ranks; 命中本 rank 时 local = idx - tp_rank*per_gpu
//   再按 dp 维扣减 local -= dp_rank*(per_dp-per_gpu); local<0 也置 -1。
//
// 【源端】TileKernels/tile_kernels/moe/mask_indices_by_tp_kernel.py
//
// 【迁移映射】
//   整除/取模 → TDIVS / TREMS
//   减法     → TSUB / TSUBS
//   比较 ==  → TCMP（仅 3 参 EQ；TCMPS 标量比较未提供 → TEXPANDS 广播 + TCMP）
//   比较 >=0 → 无 GE 模式 → 符号位技巧: TAND(local, 0x80000000) → TCMP(==0) 得 local>=0
//   结果选择 → TSEL(dst, mask, src)（dst 预置 -1，命中置 local 否则保持 -1）
//
// 【约束】NumTopk 须 8 的倍数（int32 32B 对齐）。
//
// 【算法步骤（每 tile tm）】
//   1) TLOAD idx; TDIVS(q, idx, per_gpu); TREMS(r, q, num_tp_ranks)
//   2) TEXPANDS(tp_rank)→TCMP(is_rank, r==tp_rank)
//   3) local = idx - tp_rank*per_gpu; TDIVS(dp); TMULS(adj, dp, per_dp-per_gpu); TSUB(local, local, adj)
//   4) is_ge0: TAND(local, 0x80000000)==0; is_ok = is_rank AND is_ge0
//   5) TEXPANDS(result, -1); TSEL(result, is_ok, local); TSTORE
// =============================================================================
#ifndef SUPERNPU_MASK_INDICES_BY_TP_PTO_HPP
#define SUPERNPU_MASK_INDICES_BY_TP_PTO_HPP
#include <common/pto_tileop.hpp>
#include <cstddef>
#include <cstdint>

namespace supernpu::tile_isa {

template <int NumTokens, int NumTopk, int TileM = 16>
void mask_indices_by_tp(std::int32_t *indices, std::int32_t per_gpu,
                        std::int32_t per_dp, std::int32_t num_tp_ranks,
                        std::int32_t tp_rank) {
    static_assert(NumTokens > 0 && NumTopk > 0, "dim must be positive");
    static_assert(NumTopk % 8 == 0, "NumTopk must be multiple of 8");
    constexpr int kTM = NumTokens / TileM;
    using namespace pto;
    using gm = global_tensor<std::int32_t, RowMajor<NumTokens, NumTopk>>;
    using tile = Tile<Location::Vec, std::int32_t, TileM, NumTopk, BLayout::RowMajor>;
    using it = global_iterator<gm, tile>;
    it idx_iter(indices);
    for (int tm = 0; tm < kTM; ++tm) {
        auto gidx = idx_iter(tm, 0);
        tile idx, q, r, local, dp, adj, result, mask_rank, mask_ge0, is_ok;
        TLOAD(idx, gidx);
        // 1) q = idx // per_gpu; r = q % num_tp_ranks
        TDIVS(q, idx, per_gpu);
        TREMS(r, q, num_tp_ranks);
        // 2) is_rank = (r == tp_rank)
        tile tp_bc; TEXPANDS(tp_bc, tp_rank);
        TCMP(mask_rank, r, tp_bc);
        // 3) local = idx - tp_rank*per_gpu; dp = local // per_dp; local -= dp*(per_dp-per_gpu)
        TSUBS(local, idx, tp_rank * per_gpu);
        TDIVS(dp, local, per_dp);
        TMULS(adj, dp, per_dp - per_gpu);
        TSUB(local, local, adj);
        // 4) is_ge0: 无 GE 模式 → 符号位 local&0x80000000 == 0
        tile zero, signbit, sign; TEXPANDS(zero, 0); TEXPANDS(signbit, static_cast<std::int32_t>(0x80000000));
        TAND(sign, local, signbit);                     // 取符号位
        TCMP(mask_ge0, sign, zero);                      // sign==0 即 local>=0
        TAND(is_ok, mask_rank, mask_ge0);                // is_ok = is_rank AND is_ge0
        // 5) result = is_ok ? local : -1
        TEXPANDS(result, -1);
        TSEL(result, is_ok, local);
        auto gout = idx_iter(tm, 0);
        TSTORE(gout, result);
    }
}

} // namespace supernpu::tile_isa
#endif
