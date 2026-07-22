// =============================================================================
// topk_sum_and_topk_group_idx_pto.hpp — MoE group top-k selection (tile 版)
// =============================================================================
//
// 【功能】
//   scores(tokens, groups, experts_per_group) -> group_topk_idx(tokens, topk_groups).
//   每个 group 先求组内 top-1 或 top-2 分数和，再按分数降序、group index 升序稳定选 top-k groups。
//
// 【源端】TileKernels/tile_kernels/moe/topk_sum_and_topk_group_idx_kernel.py
//   源端每 warp 处理一个 token，用 get_topk_group_idx macro 计算组内 top-k sum 和稳定 rank。
//
// 【迁移映射】
//   组内 top1/top2       → 对每个 group 的 (TileM×ExpertsPerGroup) tile 用 TROWMAX；
//                          top2 通过 TROWEXPANDSUB + TCMP + TSEL(-inf) 去掉首次最大后再 TROWMAX
//   group top-k 稳定选择 → 横向铺 NumGroups 的 group-score tile，重复 TROWMAX + 逆索引 TSEL，
//                          分数并列时取较小 group index，再把已选 group 置 -inf
//   int64 输出           → 索引 tile 先用 float 参与 TCMP/TSEL，最后 TCVT 到 int64 存储
//
// 【约束】
//   - ExpertsPerGroup 须为 8 的倍数；NumGroups/NumTopkGroups 可用物理列 padding 对齐。
//   - 所有线程本地/向量 tile 物理片段均 >=128B；默认 TileM=16。
//   - scalar C++ 仅控制 group/k 循环；tensor 数据只经 TLOAD/TSTORE 和 tile 指令流动。
// =============================================================================
#ifndef SUPERNPU_TOPK_SUM_AND_TOPK_GROUP_IDX_PTO_HPP
#define SUPERNPU_TOPK_SUM_AND_TOPK_GROUP_IDX_PTO_HPP
#include <common/pto_tileop.hpp>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace supernpu::tile_isa {

template <int NumTokens, int NumGroups, int ExpertsPerGroup, int NumTopkGroups,
          int NumTopkSum = 2, int TileM = 16>
void topk_sum_and_topk_group_idx(float *scores, std::int64_t *group_topk_idx) {
    static_assert(NumTokens > 0 && NumGroups > 0 && ExpertsPerGroup > 0, "dim must be positive");
    static_assert(NumTopkSum == 1 || NumTopkSum == 2, "NumTopkSum must be 1 or 2");
    static_assert(NumTopkGroups <= NumGroups, "NumTopkGroups <= NumGroups");
    static_assert(ExpertsPerGroup % 8 == 0, "ExpertsPerGroup must be multiple of 8");
    constexpr int kGroupCols = ((NumGroups + 7) / 8) * 8;
    constexpr int kOutCols = ((NumTopkGroups + 7) / 8) * 8;
    static_assert(TileM * ExpertsPerGroup * static_cast<int>(sizeof(float)) >= 128,
                  "score tile fragment must be >=128 bytes");
    static_assert(TileM * kGroupCols * static_cast<int>(sizeof(float)) >= 128,
                  "group-score tile fragment must be >=128 bytes");
    static_assert(TileM * kOutCols * static_cast<int>(sizeof(std::int64_t)) >= 128,
                  "output index tile fragment must be >=128 bytes");

    constexpr int kTiles = NumTokens / TileM;
    using namespace pto;
    using gm_scores = global_tensor<float, RowMajor<NumTokens, NumGroups * ExpertsPerGroup>>;
    using gm_out = global_tensor<std::int64_t, RowMajor<NumTokens, NumTopkGroups>>;
    using tile_scores = Tile<Location::Vec, float, TileM, ExpertsPerGroup, BLayout::RowMajor>;
    using tile_group = Tile<Location::Vec, float, TileM, kGroupCols, BLayout::RowMajor, TileM, NumGroups>;
    using tile_vec = Tile<Location::Vec, float, TileM, 8, BLayout::RowMajor, TileM, 1>;
    using tile_out_f = Tile<Location::Vec, float, TileM, kOutCols, BLayout::RowMajor, TileM, NumTopkGroups>;
    using tile_out_i64 = Tile<Location::Vec, std::int64_t, TileM, kOutCols, BLayout::RowMajor, TileM, NumTopkGroups>;
    using it_scores = global_iterator<gm_scores, tile_scores>;
    using it_out = global_iterator<gm_out, tile_out_i64>;

    it_scores score_iter(scores);
    it_out out_iter(group_topk_idx);

    for (int tm = 0; tm < kTiles; ++tm) {
        tile_group group_scores;
        TEXPANDS(group_scores, -std::numeric_limits<float>::infinity());

        for (int g = 0; g < NumGroups; ++g) {
            auto gs = score_iter(tm, g);
            tile_scores s;
            TLOAD(s, gs);

            tile_vec top1;
            TROWMAX(top1, s);
            tile_vec group_sum;
            if constexpr (NumTopkSum == 1) {
                group_sum = top1;
            } else {
                tile_scores expert_index_i, expert_index_mod, expert_index, diff, zero, is_top1;
                TCI(expert_index_i, 0);
                TREMS(expert_index_mod, expert_index_i, ExpertsPerGroup);
                TCVT(expert_index, expert_index_mod);
                TROWEXPANDSUB(diff, s, top1);
                TEXPANDS(zero, 0.0f);
                TCMP(is_top1, diff, zero);

                tile_scores n1, rev, picked_rev;
                TEXPANDS(n1, static_cast<float>(ExpertsPerGroup - 1));
                TSUB(rev, n1, expert_index);
                TEXPANDS(picked_rev, -1.0f);
                TSEL(picked_rev, is_top1, rev);
                tile_vec best_rev;
                TROWMAX(best_rev, picked_rev);
                tile_vec n1v, best_expert;
                TEXPANDS(n1v, static_cast<float>(ExpertsPerGroup - 1));
                TSUB(best_expert, n1v, best_rev);

                tile_scores best_bc, is_pick, neg_inf;
                TROWEXPAND(best_bc, best_expert);
                TCMP(is_pick, expert_index, best_bc);
                TEXPANDS(neg_inf, -std::numeric_limits<float>::infinity());
                TSEL(s, is_pick, neg_inf);
                tile_vec top2;
                TROWMAX(top2, s);
                TADD(group_sum, top1, top2);
            }
            TINSERT(group_scores, group_sum, 0, g);
        }

        tile_group group_index_i, group_index_mod;
        TCI(group_index_i, 0);
        TREMS(group_index_mod, group_index_i, NumGroups);
        tile_group group_index;
        TCVT(group_index, group_index_mod);

        tile_out_f picked_f;
        TEXPANDS(picked_f, -1.0f);
        for (int k = 0; k < NumTopkGroups; ++k) {
            tile_vec best_score;
            TROWMAX(best_score, group_scores);
            tile_group diff, zero, is_best;
            TROWEXPANDSUB(diff, group_scores, best_score);
            TEXPANDS(zero, 0.0f);
            TCMP(is_best, diff, zero);

            tile_group n1, rev, picked_rev;
            TEXPANDS(n1, static_cast<float>(NumGroups - 1));
            TSUB(rev, n1, group_index);
            TEXPANDS(picked_rev, -1.0f);
            TSEL(picked_rev, is_best, rev);
            tile_vec best_rev;
            TROWMAX(best_rev, picked_rev);
            tile_vec n1v, best_group;
            TEXPANDS(n1v, static_cast<float>(NumGroups - 1));
            TSUB(best_group, n1v, best_rev);

            TINSERT(picked_f, best_group, 0, k);

            tile_group best_bc, is_pick, neg_inf;
            TROWEXPAND(best_bc, best_group);
            TCMP(is_pick, group_index, best_bc);
            TEXPANDS(neg_inf, -std::numeric_limits<float>::infinity());
            TSEL(group_scores, is_pick, neg_inf);
        }

        tile_out_i64 picked_i;
        TCVT(picked_i, picked_f);
        auto go = out_iter(tm, 0);
        TSTORE(go, picked_i);
    }
}

} // namespace supernpu::tile_isa
#endif
