#include <common/pto_tileop.hpp>
#include <cstdint>
#include "solution/moe_combine/moe_combine_v2.hpp"
using namespace pto;
using namespace supernpu::tile_isa;

using dtype = __bf16;

constexpr int BS = 8;
constexpr int H = 128;
constexpr int K = 4;
constexpr int NUM_EXPANDED = BS * K;
constexpr int TILE_W = 128;

static __bf16 expand_x[NUM_EXPANDED * H] __attribute__((aligned(4096))) = {};
static float expert_scales[NUM_EXPANDED * TILE_W] __attribute__((aligned(4096))) = {};
static std::int32_t expand_idx[NUM_EXPANDED * 3] __attribute__((aligned(4096))) = {};
static __bf16 window_data[NUM_EXPANDED * H] __attribute__((aligned(4096))) = {};
static float window_flag[NUM_EXPANDED * TILE_W] __attribute__((aligned(4096))) = {};
static std::int32_t window_triple[NUM_EXPANDED * 3] __attribute__((aligned(4096))) = {};
static uint32_t window_state[16] __attribute__((aligned(4096))) = {};
static float pred_buf[NUM_EXPANDED * TILE_W] __attribute__((aligned(4096))) = {};
static __bf16 out_buf[BS * H] __attribute__((aligned(4096))) = {};

int main() {
    for (int tk = 0; tk < NUM_EXPANDED; tk++) {
        expand_idx[tk * 3 + 0] = 0;
        expand_idx[tk * 3 + 1] = tk / K;
        expand_idx[tk * 3 + 2] = tk % K;
        // [A5 对齐修复 2026-10-13] kernel 密集读 expertScales[n*K + k] (即下标
        // tk, v2.hpp:167); 原按 [tk*TILE_W + j] 填充使 token 1..7 的 scale
        // 实际为 0。改为密集前缀填充 (mt.cpp:78 同款), 每 slot 携带真实权重。
        expert_scales[tk] = 0.25f;
    }

    moe_combine_v2<__bf16, __bf16, BS, H, K, NUM_EXPANDED, TILE_W>(
        expand_x, expert_scales, expand_idx, window_data, window_flag,
        window_state, pred_buf, out_buf);
    return 0;
}
