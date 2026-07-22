// =============================================================================
// top2_sum_gate_pto.hpp — MoE top-k gate with top-2-sum group filtering (tile 版)
// =============================================================================
//
// 【功能】
//   logits+bias -> topk_idx/topk_weights. 先按 group 内 top-2 sum 选 top groups，
//   再只在选中 groups 内选 top-k experts；权重来自未加 bias 的 scoring 输出并归一化。
//
// 【源端】TileKernels/tile_kernels/moe/top2_sum_gate_kernel.py
//
// 【迁移映射】
//   scoring              → sigmoid: TEXP/TADDS/TRECIP；sqrtsoftplus: TEXP/TADDS/TLOG/TSQRT；
//                          softmax: TROWMAX/TROWEXPANDSUB/TEXP/TROWSUM/TRECIP/TROWEXPANDMUL
//   group top-2 sum      → 每 group TROWMAX 两轮，第二轮用逆索引 tie-break 只屏蔽一个最大值
//   group 过滤 + top-k   → TCI/TDIVS 生成 expert->group，TCMP/TSEL 保留已选 groups；
//                          TROWMAX + 逆索引 TSEL 选 expert，重复 NumTopk 轮
//   权重归一化           → 对选中 experts 的 scoring 输出 TROWSUM，TRECIP 后 TROWEXPANDMUL，再 TMULS(scale)
//
// 【边界】
//   源端 mask/fix_routing_mask/to_physical_map 分支需要按 tensor 值进行 scalar lookup。
//   本 port 保留 tile-native核心路径；scalar C++ 只控制静态 group/k 循环和运行时 scale/rank 常量，
//   不承担 pointer-indexed tensor data path。
//
// 【约束】
//   - NumRoutedExperts = NumGroups * ExpertsPerGroup。
//   - ExpertsPerGroup 须为 8 的倍数；NumGroups/NumPhysicalTopk/NumRoutedExperts 用物理列 padding 对齐。
//   - 所有线程本地/向量 tile 物理片段均 >=128B；默认 TileM=16。
// =============================================================================
#ifndef SUPERNPU_TOP2_SUM_GATE_PTO_HPP
#define SUPERNPU_TOP2_SUM_GATE_PTO_HPP
#include <common/pto_tileop.hpp>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace supernpu::tile_isa {

enum class Top2GateScoring : int {
    Sigmoid = 0,
    SqrtSoftplus = 1,
    Softmax = 2,
    Identity = 3,
};

template <int NumTokens, int NumGroups, int ExpertsPerGroup, int NumTopk,
          int NumTopkGroups, int NumSharedExperts = 0,
          Top2GateScoring Scoring = Top2GateScoring::Sigmoid, int TileM = 16>
void top2_sum_gate(float *logits, float *bias, std::int64_t *topk_idx,
                   float *topk_weights, float routed_scaling_factor = 1.0f) {
    constexpr int NumRoutedExperts = NumGroups * ExpertsPerGroup;
    constexpr int NumPhysicalTopk = NumTopk + NumSharedExperts;
    static_assert(NumTokens > 0 && NumGroups > 0 && ExpertsPerGroup > 0, "dim must be positive");
    static_assert(NumTopk <= NumRoutedExperts, "NumTopk <= NumRoutedExperts");
    static_assert(NumTopkGroups <= NumGroups, "NumTopkGroups <= NumGroups");
    static_assert(NumSharedExperts == 0 || NumSharedExperts == 1 || NumSharedExperts == 2,
                  "source supports 0, 1, or 2 shared experts");
    static_assert(ExpertsPerGroup % 8 == 0, "ExpertsPerGroup must be multiple of 8");
    constexpr int kRoutedCols = ((NumRoutedExperts + 7) / 8) * 8;
    constexpr int kGroupCols = ((NumGroups + 7) / 8) * 8;
    constexpr int kPhysicalTopkCols = ((NumPhysicalTopk + 7) / 8) * 8;
    static_assert(TileM * kRoutedCols * static_cast<int>(sizeof(float)) >= 128,
                  "routed score tile fragment must be >=128 bytes");
    static_assert(TileM * ExpertsPerGroup * static_cast<int>(sizeof(float)) >= 128,
                  "group score tile fragment must be >=128 bytes");
    static_assert(TileM * kPhysicalTopkCols * static_cast<int>(sizeof(float)) >= 128,
                  "weight tile fragment must be >=128 bytes");
    static_assert(TileM * kPhysicalTopkCols * static_cast<int>(sizeof(std::int64_t)) >= 128,
                  "index tile fragment must be >=128 bytes");

    constexpr int kTiles = NumTokens / TileM;
    using namespace pto;
    using gm_logits = global_tensor<float, RowMajor<NumTokens, NumRoutedExperts>>;
    using gm_bias = global_tensor<float, RowMajor<1, NumRoutedExperts>>;
    using gm_idx = global_tensor<std::int64_t, RowMajor<NumTokens, NumPhysicalTopk>>;
    using gm_weight = global_tensor<float, RowMajor<NumTokens, NumPhysicalTopk>>;
    using tile_all = Tile<Location::Vec, float, TileM, kRoutedCols, BLayout::RowMajor, TileM, NumRoutedExperts>;
    using tile_group = Tile<Location::Vec, float, TileM, ExpertsPerGroup, BLayout::RowMajor>;
    using tile_bias = Tile<Location::Vec, float, 4, ExpertsPerGroup, BLayout::RowMajor, 1, ExpertsPerGroup>;
    using tile_groups = Tile<Location::Vec, float, TileM, kGroupCols, BLayout::RowMajor, TileM, NumGroups>;
    using tile_vec = Tile<Location::Vec, float, TileM, 8, BLayout::RowMajor, TileM, 1>;
    using tile_out_f = Tile<Location::Vec, float, TileM, kPhysicalTopkCols, BLayout::RowMajor, TileM, NumPhysicalTopk>;
    using tile_out_i64 = Tile<Location::Vec, std::int64_t, TileM, kPhysicalTopkCols, BLayout::RowMajor, TileM, NumPhysicalTopk>;
    using it_logits_group = global_iterator<gm_logits, tile_group>;
    using it_bias_group = global_iterator<gm_bias, tile_bias>;
    using it_idx = global_iterator<gm_idx, tile_out_i64>;
    using it_weight = global_iterator<gm_weight, tile_out_f>;

    it_logits_group logit_iter(logits);
    it_bias_group bias_iter(bias);
    it_idx idx_iter(topk_idx);
    it_weight weight_iter(topk_weights);

    for (int tm = 0; tm < kTiles; ++tm) {
        tile_all score_wo_bias, rank_score;
        tile_groups group_score;
        TEXPANDS(score_wo_bias, 0.0f);
        TEXPANDS(rank_score, -std::numeric_limits<float>::infinity());
        TEXPANDS(group_score, -std::numeric_limits<float>::infinity());

        for (int g = 0; g < NumGroups; ++g) {
            auto gl = logit_iter(tm, g);
            auto gb = bias_iter(0, g);
            tile_group raw;
            tile_bias bias_row;
            tile_group bias_full, scored, ranked;
            TLOAD(raw, gl);
            TLOAD(bias_row, gb);
            TCOLEXPAND(bias_full, bias_row);

            if constexpr (Scoring == Top2GateScoring::Sigmoid) {
                tile_group neg, expv, den;
                TMULS(neg, raw, -1.0f);
                TEXP(expv, neg);
                TADDS(den, expv, 1.0f);
                TRECIP(scored, den);
                TADD(ranked, scored, bias_full);
            } else if constexpr (Scoring == Top2GateScoring::SqrtSoftplus) {
                tile_group expv, den, sp;
                TEXP(expv, raw);
                TADDS(den, expv, 1.0f);
                TLOG(sp, den);
                TSQRT(scored, sp);
                TADD(ranked, scored, bias_full);
            } else if constexpr (Scoring == Top2GateScoring::Softmax) {
                // Softmax normalization is completed after all groups are loaded.
                scored = raw;
                TADD(ranked, raw, bias_full);
            } else {
                scored = raw;
                TADD(ranked, raw, bias_full);
            }

            TINSERT(score_wo_bias, scored, 0, g * ExpertsPerGroup);
            TINSERT(rank_score, ranked, 0, g * ExpertsPerGroup);

            tile_vec top1;
            TROWMAX(top1, ranked);
            tile_group expert_index_i, expert_index_mod, expert_index, diff, zero, is_top1;
            TCI(expert_index_i, 0);
            TREMS(expert_index_mod, expert_index_i, ExpertsPerGroup);
            TCVT(expert_index, expert_index_mod);
            TROWEXPANDSUB(diff, ranked, top1);
            TEXPANDS(zero, 0.0f);
            TCMP(is_top1, diff, zero);
            tile_group n1, rev, picked_rev;
            TEXPANDS(n1, static_cast<float>(ExpertsPerGroup - 1));
            TSUB(rev, n1, expert_index);
            TEXPANDS(picked_rev, -1.0f);
            TSEL(picked_rev, is_top1, rev);
            tile_vec best_rev;
            TROWMAX(best_rev, picked_rev);
            tile_vec n1v, best_expert;
            TEXPANDS(n1v, static_cast<float>(ExpertsPerGroup - 1));
            TSUB(best_expert, n1v, best_rev);
            tile_group best_bc, is_pick, neg_inf;
            TROWEXPAND(best_bc, best_expert);
            TCMP(is_pick, expert_index, best_bc);
            TEXPANDS(neg_inf, -std::numeric_limits<float>::infinity());
            TSEL(ranked, is_pick, neg_inf);
            tile_vec top2, top2_sum;
            TROWMAX(top2, ranked);
            TADD(top2_sum, top1, top2);
            TINSERT(group_score, top2_sum, 0, g);
        }

        if constexpr (Scoring == Top2GateScoring::Softmax) {
            tile_vec row_max, row_sum, recip;
            tile_all shifted, expv;
            TROWMAX(row_max, score_wo_bias);
            TROWEXPANDSUB(shifted, score_wo_bias, row_max);
            TEXP(expv, shifted);
            TROWSUM(row_sum, expv);
            TRECIP(recip, row_sum);
            TROWEXPANDMUL(score_wo_bias, expv, recip);
        }

        tile_groups group_index_i, group_index_mod, group_index;
        TCI(group_index_i, 0);
        TREMS(group_index_mod, group_index_i, NumGroups);
        TCVT(group_index, group_index_mod);
        tile_groups selected_groups;
        TEXPANDS(selected_groups, -1.0f);
        for (int k = 0; k < NumTopkGroups; ++k) {
            tile_vec best_group_score;
            TROWMAX(best_group_score, group_score);
            tile_groups diff, zero, is_best;
            TROWEXPANDSUB(diff, group_score, best_group_score);
            TEXPANDS(zero, 0.0f);
            TCMP(is_best, diff, zero);
            tile_groups n1, rev, picked_rev;
            TEXPANDS(n1, static_cast<float>(NumGroups - 1));
            TSUB(rev, n1, group_index);
            TEXPANDS(picked_rev, -1.0f);
            TSEL(picked_rev, is_best, rev);
            tile_vec best_rev, n1v, best_group;
            TROWMAX(best_rev, picked_rev);
            TEXPANDS(n1v, static_cast<float>(NumGroups - 1));
            TSUB(best_group, n1v, best_rev);
            TINSERT(selected_groups, best_group, 0, k);
            tile_groups best_bc, is_pick, neg_inf;
            TROWEXPAND(best_bc, best_group);
            TCMP(is_pick, group_index, best_bc);
            TEXPANDS(neg_inf, -std::numeric_limits<float>::infinity());
            TSEL(group_score, is_pick, neg_inf);
        }

        tile_all expert_index_i, expert_index_mod, expert_index, group_of_expert_i, group_of_expert;
        TCI(expert_index_i, 0);
        TREMS(expert_index_mod, expert_index_i, NumRoutedExperts);
        TCVT(expert_index, expert_index_mod);
        TDIVS(group_of_expert_i, expert_index_mod, ExpertsPerGroup);
        TCVT(group_of_expert, group_of_expert_i);

        tile_all selected_mask;
        TEXPANDS(selected_mask, 0.0f);
        for (int gk = 0; gk < NumTopkGroups; ++gk) {
            tile_vec sg;
            TEXTRACT(sg, selected_groups, 0, gk);
            tile_all sg_bc, eq;
            TROWEXPAND(sg_bc, sg);
            TCMP(eq, group_of_expert, sg_bc);
            TOR(selected_mask, selected_mask, eq);
        }
        tile_all candidates;
        TEXPANDS(candidates, -std::numeric_limits<float>::infinity());
        TSEL(candidates, selected_mask, rank_score);

        tile_out_f picked_idx_f, picked_weight_raw, picked_weight_norm;
        TEXPANDS(picked_idx_f, -1.0f);
        TEXPANDS(picked_weight_raw, 0.0f);
        tile_vec weight_sum;
        TEXPANDS(weight_sum, 1e-20f);
        for (int k = 0; k < NumTopk; ++k) {
            tile_vec best_score;
            TROWMAX(best_score, candidates);
            tile_all diff, zero, is_best;
            TROWEXPANDSUB(diff, candidates, best_score);
            TEXPANDS(zero, 0.0f);
            TCMP(is_best, diff, zero);
            tile_all n1, rev, picked_rev;
            TEXPANDS(n1, static_cast<float>(NumRoutedExperts - 1));
            TSUB(rev, n1, expert_index);
            TEXPANDS(picked_rev, -1.0f);
            TSEL(picked_rev, is_best, rev);
            tile_vec best_rev, n1v, best_expert;
            TROWMAX(best_rev, picked_rev);
            TEXPANDS(n1v, static_cast<float>(NumRoutedExperts - 1));
            TSUB(best_expert, n1v, best_rev);
            TINSERT(picked_idx_f, best_expert, 0, k);

            tile_all best_bc, is_pick, zero_all, neg_inf;
            tile_vec selected_weight;
            TROWEXPAND(best_bc, best_expert);
            TCMP(is_pick, expert_index, best_bc);
            TEXPANDS(zero_all, 0.0f);
            TSEL(zero_all, is_pick, score_wo_bias);
            TROWSUM(selected_weight, zero_all);
            TADD(weight_sum, weight_sum, selected_weight);
            TINSERT(picked_weight_raw, selected_weight, 0, k);

            TEXPANDS(neg_inf, -std::numeric_limits<float>::infinity());
            TSEL(candidates, is_pick, neg_inf);
        }

        tile_vec recip_sum;
        TRECIP(recip_sum, weight_sum);
        TROWEXPANDMUL(picked_weight_norm, picked_weight_raw, recip_sum);
        TMULS(picked_weight_norm, picked_weight_norm, routed_scaling_factor);

        if constexpr (NumSharedExperts != 0) {
            for (int s = 0; s < NumSharedExperts; ++s) {
                tile_vec one, shared_idx;
                TEXPANDS(one, 1.0f);
                TEXPANDS(shared_idx, static_cast<float>(NumRoutedExperts + s));
                TINSERT(picked_weight_norm, one, 0, NumTopk + s);
                TINSERT(picked_idx_f, shared_idx, 0, NumTopk + s);
            }
        }

        tile_out_i64 picked_idx_i;
        TCVT(picked_idx_i, picked_idx_f);
        auto gi = idx_iter(tm, 0);
        auto gw = weight_iter(tm, 0);
        TSTORE(gi, picked_idx_i);
        TSTORE(gw, picked_weight_norm);
    }
}

} // namespace supernpu::tile_isa
#endif
