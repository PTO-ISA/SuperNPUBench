// =============================================================================
// topk_gate_pto.hpp — MoE top-k 专家选择（tile 版，argmax 模拟）
// =============================================================================
//
// 【功能】
//   scores(tokens, experts) → topk_idx(tokens, topk)，每 token 取 top-k 专家索引，并列取较小索引。
//
// 【源端】TileKernels/tile_kernels/moe/topk_gate_kernel.py
//   源端 warp 32 threads 并行找最大 + reducer('min') 取并列最小索引。
//
// 【迁移映射】
//   TROWARGMAX 工具链未提供 → argmax 模拟：
//     TROWMAX 取最大值 → TROWEXPANDSUB(差) → TCMP(==0) 得最大值位置掩码 →
//     逆索引 rev=N-1-I（实现并列取小：max(rev)=min(idx)）→ TSEL 取掩码处 rev →
//     TROWMAX 得最大 rev = 最小 idx → best_idx = N-1-best_rev →
//     TROWEXPAND 广播 + TCMP(I==) + TSEL(-inf) 掩码置 -inf；重复 NumTopk 轮
//   TCMP/TSEL 三参须同 dtype → 索引运算全用 float，存储时 TCVT 回 int32
//   索引生成: TCI(0)+TREMS(N) 每行 [0..N-1]，再 TCVT 成 float
//
// 【约束】TileN 须 4 的倍数（TileN×8 valid1≥128B）；NumExperts 须 8 的倍数。
//
// 【算法步骤（每 token 块 tn，k 轮）】
//   1) 索引: TCI(0)→TREMS(N)→TCVT 成 float I（每行 [0..N-1]）
//   2) TROWMAX(mx, S) 行最大值
//   3) TROWEXPANDSUB(diff, S, mx) 差；TCMP(is_max, diff, 0) 掩码(diff==0)
//   4) 逆索引 rev=N-1-I；TSEL(picked, is_max, rev)（picked 初值 -1，掩码处置 rev）
//   5) TROWMAX(best_rev, picked) 最大 rev=最小 idx；best_idx=N-1-best_rev
//   6) TCVT(best_idx→int)；TSTORE 写 topk_idx
//   7) 掩码置 -inf: TROWEXPAND(best_idx)→TCMP(I==)→TSEL(S, -inf)
// =============================================================================
#ifndef SUPERNPU_TOPK_GATE_PTO_HPP
#define SUPERNPU_TOPK_GATE_PTO_HPP
#include <common/pto_tileop.hpp>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace supernpu::tile_isa {

template <int NumTokens, int NumExperts, int NumTopk, int TileN = 8>
void topk_gate(float *scores, std::int32_t *topk_idx) {
    static_assert(NumTopk <= NumExperts, "num_topk <= num_experts");
    static_assert(NumExperts % 8 == 0, "NumExperts must be multiple of 8");
    static_assert(TileN % 4 == 0, "TileN must be multiple of 4");
    constexpr int kTN = NumTokens / TileN;
    using namespace pto;
    using gm_s = global_tensor<float, RowMajor<NumTokens, NumExperts>>;
    using gm_i = global_tensor<std::int32_t, RowMajor<NumTokens, NumTopk>>;
    using tile_s = Tile<Location::Vec, float, TileN, NumExperts, BLayout::RowMajor>;
    using tile_i = Tile<Location::Vec, std::int32_t, TileN, NumExperts, BLayout::RowMajor>;
    using tile_v = Tile<Location::Vec, float, TileN, 8, BLayout::RowMajor, TileN, 1>;       // 每行一个值
    using tile_iv = Tile<Location::Vec, std::int32_t, TileN, 8, BLayout::RowMajor, TileN, 1>;
    using it_s = global_iterator<gm_s, tile_s>;
    using it_i = global_iterator<gm_i, tile_iv>;
    it_s s_iter(scores); it_i o_iter(topk_idx);

    for (int tn = 0; tn < kTN; ++tn) {
        auto gin = s_iter(tn, 0);
        tile_s S; TLOAD(S, gin);
        // 索引 I: int 上 TCI(0)+TREMS(N) 每行 [0..N-1]，再 TCVT 成 float 参与 TSEL/TCMP
        tile_i Iraw, Irem; TCI(Iraw, 0); TREMS(Irem, Iraw, NumExperts);
        tile_s I; TCVT(I, Irem);
        for (int k = 0; k < NumTopk; ++k) {
            // 2-3) 行最大值 + 掩码最大值位置
            tile_v mx; TROWMAX(mx, S);
            tile_s diff; TROWEXPANDSUB(diff, S, mx);     // diff = S - mx（最大值处为 0）
            tile_s zero, is_max; TEXPANDS(zero, 0.0f);
            TCMP(is_max, diff, zero);                   // 掩码: diff==0 即最大值位置
            // 4) 逆索引 rev=N-1-I（并列时取 rev 最大=idx 最小）；TSEL 取掩码处 rev
            tile_s n1, rev, picked; TEXPANDS(n1, static_cast<float>(NumExperts - 1));
            TSUB(rev, n1, I);                            // rev = N-1 - I
            TEXPANDS(picked, -1.0f);
            TSEL(picked, is_max, rev);                  // 掩码处取 rev，否则保持 -1
            // 5) TROWMAX 取最大 rev = 最小 idx；best_idx = N-1-best_rev
            tile_v best_rev; TROWMAX(best_rev, picked);
            tile_v n1v, best_idx_f; TEXPANDS(n1v, static_cast<float>(NumExperts - 1));
            TSUB(best_idx_f, n1v, best_rev);
            // 6) TCVT→int；TSTORE
            tile_iv best_idx_i; TCVT(best_idx_i, best_idx_f);
            auto gout = o_iter(tn, k); TSTORE(gout, best_idx_i);
            // 7) 掩码置 -inf: 广播 best_idx + 比较 I== + TSEL(-inf)
            tile_s best_bc; TROWEXPAND(best_bc, best_idx_f);
            tile_s is_pick; TCMP(is_pick, I, best_bc);   // 掩码: I==best_idx
            tile_s neg_inf; TEXPANDS(neg_inf, -std::numeric_limits<float>::infinity());
            TSEL(S, is_pick, neg_inf);                  // 选中位置 S=-inf
        }
    }
}

} // namespace supernpu::tile_isa
#endif
