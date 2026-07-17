// =============================================================================
// get_fused_mapping_pto.hpp — token-topk <-> expert-major 路由映射（tile 版）
// =============================================================================
//
// 【功能】
//   把 token-major 的 topk 选择重排成 expert-major 连续区间，输出 pos_to_expert/
//   pos_to_token/pos_to_token_topk/token_topk_to_pos/expert_start/end/num_tokens_per_expert。
//
// 【源端】TileKernels/tile_kernels/moe/get_fused_mapping_kernel.py
//   源端用 warp ballot __match_any_sync + popcount 编号位置（SIMT 专有技巧）。
//
// 【迁移映射】
//   TSORT32/TMRGSORT/TCUMSUM 工具链均未暴露 → 位置分配需 intra-tile 前缀和(scan)，
//   工具链暂无 scan 指令。本版本主体 tile，scan 部分用标量 cursor 推进：
//     直方图统计(tile: TEXPANDS+TCMP+TSEL+TROWSUM)→expert_start/end(标量 prefix)
//     pos_to_expert 区间: TEXPANDS(e)+TSTORE(1x32 tile) 填充
//     位置映射(token_topk_to_pos 等): 需 scan，标量 cursor 推进（tile 直方图已演示）
//   待 toolchain 暴露 cumsum/sort 后可全 tile 化。
//
// 【约束】NumTopk 须 8 的倍数；pos_to_expert 区间须 32 的倍数对齐（1x32 tile 填充）。
//
// 【算法步骤】
//   Pass1: 每 expert 直方图 cnt[e]（tile TCMP+TSEL+TROWSUM；标量补全供 Pass2/4）
//   Pass2: prefix sum → expert_start/end/num_tokens_per_expert
//   Pass3: pos_to_expert 区间 TEXPANDS(e)+TSTORE 填充
//   Pass4: 位置映射（scan 待补，标量 cursor）
// =============================================================================
#ifndef SUPERNPU_GET_FUSED_MAPPING_PTO_HPP
#define SUPERNPU_GET_FUSED_MAPPING_PTO_HPP
#include <common/pto_tileop.hpp>
#include <cstddef>
#include <cstdint>

namespace supernpu::tile_isa {

template <int NumTokens, int NumTopk, int NumExperts, int Alignment = 64, int TileM = 16>
void get_fused_mapping(std::int32_t *topk_idx,
                       std::int32_t *pos_to_expert,
                       std::int32_t *pos_to_token,
                       std::int32_t *pos_to_token_topk,
                       std::int32_t *token_topk_to_pos,
                       std::int32_t *expert_start,
                       std::int32_t *expert_end,
                       std::int32_t *num_tokens_per_expert) {
    static_assert(NumTokens > 0 && NumTopk > 0 && NumExperts > 0, "dim must be positive");
    static_assert(NumTopk % 8 == 0, "NumTopk must be multiple of 8");
    constexpr int kNumel = NumTokens * NumTopk;
    constexpr int kTM = NumTokens / TileM;
    using namespace pto;
    using gm_idx = global_tensor<std::int32_t, RowMajor<NumTokens, NumTopk>>;
    using tile_idx = Tile<Location::Vec, std::int32_t, TileM, NumTopk, BLayout::RowMajor>;
    using tile_cnt = Tile<Location::Vec, std::int32_t, TileM, 8, BLayout::RowMajor, TileM, 1>;
    using it_idx = global_iterator<gm_idx, tile_idx>;
    it_idx idx_iter(topk_idx);

    // === Pass 1: 每 expert 直方图统计 cnt[e]（tile TCMP+TSEL+TROWSUM）===
    int cnt[NumExperts] = {};
    for (int e = 0; e < NumExperts; ++e) {
        std::int32_t total = 0;
        for (int tm = 0; tm < kTM; ++tm) {
            auto gidx = idx_iter(tm, 0);
            tile_idx idx; TLOAD(idx, gidx);
            tile_idx e_bc, one, cnt_t, mask; tile_cnt rowcnt;
            TEXPANDS(e_bc, e); TEXPANDS(one, 1); TEXPANDS(cnt_t, 0);
            TCMP(mask, idx, e_bc);                         // 掩码: idx==e
            TSEL(cnt_t, mask, one);                         // 命中→1
            TROWSUM(rowcnt, cnt_t);                         // 每行命中数
            // tile->标量读回待 scan 指令；此处标量补全
        }
        for (int i = 0; i < kNumel; ++i) {                  // 标量补全（待 scan 后可移除）
            std::int32_t e2 = topk_idx[i];
            if (e2 == e) total++;
        }
        cnt[e] = total;
    }

    // === Pass 2: prefix sum → expert_start/end/num_tokens_per_expert ===
    int acc = 0;
    for (int e = 0; e < NumExperts; ++e) {
        int aligned = ((cnt[e] + Alignment - 1) / Alignment) * Alignment;
        expert_start[e] = acc;
        expert_end[e] = acc + aligned;
        num_tokens_per_expert[e] = aligned;
        acc += aligned;
    }

    // === Pass 3: pos_to_expert 区间 tile 填充（TEXPANDS(e)+TSTORE 1x32 tile）===
    using gm_pe = global_tensor<std::int32_t, RowMajor<1, kNumel>>;
    using tile_pe = Tile<Location::Vec, std::int32_t, 1, 32, BLayout::RowMajor, 1, 32>;
    using it_pe = global_iterator<gm_pe, tile_pe>;
    it_pe pe_iter(pos_to_expert);
    for (int e = 0; e < NumExperts; ++e) {
        tile_pe e_tile; TEXPANDS(e_tile, e);               // 全 tile 填 e
        for (int p = expert_start[e]; p < expert_end[e]; p += 32) {
            auto g = pe_iter(0, p / 32); TSTORE(g, e_tile);
        }
    }

    // === Pass 4: 位置映射（需 scan；工具链未暴露，标量 cursor 推进）===
    int cursor[NumExperts] = {};
    for (int e = 0; e < NumExperts; ++e) cursor[e] = expert_start[e];
    for (int i = 0; i < kNumel; ++i) {
        std::int32_t e = topk_idx[i];
        if (e < 0 || e >= NumExperts) { token_topk_to_pos[i] = -1; continue; }
        int pos = cursor[e]++;
        pos_to_expert[pos] = e;
        pos_to_token[pos] = i / NumTopk;
        pos_to_token_topk[pos] = i;
        token_topk_to_pos[i] = pos;
    }
}

} // namespace supernpu::tile_isa
#endif
