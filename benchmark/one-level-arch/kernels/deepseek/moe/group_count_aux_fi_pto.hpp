// =============================================================================
// group_count_aux_fi_pto.hpp — MoE 负载均衡统计（tile 版，直方图模拟）
// =============================================================================
//
// 【功能】
//   group_count: 统计每 group 被选次数（直方图）
//   aux_fi:      group_count 基础上乘 N/(T*K) 归一化为 float（auxiliary-loss-free 负载）
//   两者本质都是"索引张量 → 计数直方图"。
//
// 【源端】TileKernels/tile_kernels/moe/group_count_kernel.py, aux_fi_kernel.py
//
// 【迁移映射】
//   THISTOGRAM 工具链未提供 → 直方图 tile: 每专家 e 用
//     TEXPANDS(e) 广播 + TCMP(==) 掩码 + TSEL(命中→1, 否则 0) + TROWSUM(每行命中)
//     + TINSERT 进 (TileM×NumGroups) 直方图列 e；跨 tile TADD 累加；末尾 TSTORE。
//   避免单标量 tile（1×8<128B 违最小限制）：直方图 tile 横向铺所有专家列。
//   aux_fi 归一化: TCVT(int→fp32)+TMULS(*scale)
//
// 【约束】NumTopk/NumGroups 须 8 的倍数；NumGroups 须≥8 使 1×NumGroups≥128B（小需 padding）。
//
// 【算法步骤（每 expert e，每 tile tm）】
//   TLOAD idx → TEXPANDS(e) → TCMP(mask, idx==e) → TSEL(cnt, mask, 1)
//   → TROWSUM(rowcnt, cnt) 每行命中 → TINSERT(tile_total, rowcnt, 0, e) 直方图列 e
//   跨 tile TADD(grand, grand, tile_total) → TSTORE grand
// =============================================================================
#ifndef SUPERNPU_GROUP_COUNT_AUX_FI_PTO_HPP
#define SUPERNPU_GROUP_COUNT_AUX_FI_PTO_HPP
#include <common/pto_tileop.hpp>
#include <cstddef>
#include <cstdint>

namespace supernpu::tile_isa {

template <int NumTokens, int NumTopk, int NumGroups, int TileM = 16>
void group_count(std::int32_t *group_idx, std::int32_t *out) {
    static_assert(NumTokens > 0 && NumTopk > 0 && NumGroups > 0, "dim must be positive");
    static_assert(NumTopk % 8 == 0 && NumGroups % 8 == 0, "NumTopk/NumGroups must be multiple of 8");
    constexpr int kTM = NumTokens / TileM;
    using namespace pto;
    using gm_idx = global_tensor<std::int32_t, RowMajor<NumTokens, NumTopk>>;
    using gm_out = global_tensor<std::int32_t, RowMajor<1, NumGroups>>;          // 直方图横向铺
    using tile_idx = Tile<Location::Vec, std::int32_t, TileM, NumTopk, BLayout::RowMajor>;
    using tile_hist = Tile<Location::Vec, std::int32_t, 1, NumGroups, BLayout::RowMajor, 1, NumGroups>;
    using tile_rowcnt = Tile<Location::Vec, std::int32_t, TileM, 8, BLayout::RowMajor, TileM, 1>;
    using it_idx = global_iterator<gm_idx, tile_idx>;
    using it_out = global_iterator<gm_out, tile_hist>;
    it_idx idx_iter(group_idx); it_out out_iter(out);

    tile_hist grand; TEXPANDS(grand, 0);                 // 跨 tile 累加器（横向所有专家列）
    for (int tm = 0; tm < kTM; ++tm) {
        auto gidx = idx_iter(tm, 0);
        tile_idx idx; TLOAD(idx, gidx);
        tile_hist tile_total; TEXPANDS(tile_total, 0);   // 本 tile 直方图
        for (int e = 0; e < NumGroups; ++e) {            // 逐专家统计
            tile_idx e_bc, one, cnt, mask; tile_rowcnt rowcnt;
            TEXPANDS(e_bc, e); TEXPANDS(one, 1); TEXPANDS(cnt, 0);
            TCMP(mask, idx, e_bc);                       // 掩码: idx==e
            TSEL(cnt, mask, one);                         // cnt[i]=1 命中, 0 未命中
            TROWSUM(rowcnt, cnt);                        // 每行命中数
            TINSERT(tile_total, rowcnt, 0, e);           // 写入直方图列 e
        }
        TADD(grand, grand, tile_total);                  // 累加跨 tile
    }
    auto gout = out_iter(0, 0);
    TSTORE(gout, grand);
}

// aux_fi: group_count 后 TCVT(int→fp32)+TMULS(*N/(T*K))
template <int NumTokens, int NumTopk, int NumExperts, int TileM = 16>
void aux_fi(std::int32_t *topk_idx, float *out, int num_aux_topk) {
    static_assert(NumTokens > 0 && NumTopk > 0 && NumExperts > 0, "dim must be positive");
    static_assert(NumTopk % 8 == 0 && NumExperts % 8 == 0, "NumTopk/NumExperts must be multiple of 8");
    constexpr int kTM = NumTokens / TileM;
    using namespace pto;
    using gm_idx = global_tensor<std::int32_t, RowMajor<NumTokens, NumTopk>>;
    using gm_out = global_tensor<float, RowMajor<1, NumExperts>>;
    using tile_idx = Tile<Location::Vec, std::int32_t, TileM, NumTopk, BLayout::RowMajor>;
    using tile_hist_i = Tile<Location::Vec, std::int32_t, 1, NumExperts, BLayout::RowMajor, 1, NumExperts>;
    using tile_hist_f = Tile<Location::Vec, float, 1, NumExperts, BLayout::RowMajor, 1, NumExperts>;
    using tile_rowcnt = Tile<Location::Vec, std::int32_t, TileM, 8, BLayout::RowMajor, TileM, 1>;
    using it_idx = global_iterator<gm_idx, tile_idx>;
    using it_out = global_iterator<gm_out, tile_hist_f>;
    it_idx idx_iter(topk_idx); it_out out_iter(out);
    float scale = static_cast<float>(NumExperts) /
                  static_cast<float>(NumTokens * (num_aux_topk > 0 ? num_aux_topk : 1));

    tile_hist_i grand; TEXPANDS(grand, 0);
    for (int tm = 0; tm < kTM; ++tm) {
        auto gidx = idx_iter(tm, 0);
        tile_idx idx; TLOAD(idx, gidx);
        tile_hist_i tile_total; TEXPANDS(tile_total, 0);
        for (int e = 0; e < NumExperts; ++e) {
            tile_idx e_bc, one, cnt, mask; tile_rowcnt rowcnt;
            TEXPANDS(e_bc, e); TEXPANDS(one, 1); TEXPANDS(cnt, 0);
            TCMP(mask, idx, e_bc);
            TSEL(cnt, mask, one);
            TROWSUM(rowcnt, cnt);
            TINSERT(tile_total, rowcnt, 0, e);
        }
        TADD(grand, grand, tile_total);
    }
    tile_hist_f gf, of;
    TCVT(gf, grand);                                    // int→fp32
    TMULS(of, gf, scale);                               // 归一化 count * N/(T*K)
    auto gout = out_iter(0, 0);
    TSTORE(gout, of);
}

} // namespace supernpu::tile_isa
#endif
