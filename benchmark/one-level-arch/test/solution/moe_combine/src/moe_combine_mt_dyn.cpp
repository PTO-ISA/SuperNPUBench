/**
 * MoE Combine — 4-PE SPMD 动态 shape test driver (moe_combine_mt_dyn)
 *
 * 基于 moe_combine_mt.cpp 的动态 shape 版驱动:
 *   - 单次运行内顺序执行 2 组运行时 shape:
 *       cfgA: {BS=8, H=128, K=4, NumExpanded=32}   — 与静态版同值, 等价回归
 *       cfgB: {BS=7, H=128, K=3, NumExpanded=21}   — NumExpanded=21%4≠0、
 *             BS=7%4≠0, 覆盖运行时 ceil 分片路径
 *   - GM 缓冲按两组 shape 的最大值定长分配, 运行时只使用有效段;
 *   - 两次 kernel 调用之间把 header 内 sCombineMtPhaseDone barrier 数组清零
 *     (静态相位编号在第二次调用会因陈旧 flag 立即通过而失去同步);
 *   - 验证 PE0 独占, 逐组比对 (逻辑与静态版一致, 循环边界运行时化):
 *     cfgA 失败返回静态版同款诊断码 1..4, cfgB 失败返回 +10 偏移码。
 *
 * 运行 (gfrun 功能仿真, 必须四线程):
 *   gfrun -t 1 -s softcore.multiThreadNum=4 -f <elf>
 */

#include <common/pto_tileop.hpp>
#include <cstdint>
#include <cstring>
#include "benchmark.h"
#include "solution/moe_combine/moe_combine_mt_dyn.hpp"

using namespace supernpu::tile_isa;

using dtype = __bf16;

constexpr int kTileW = 128;
// ---- max-shape 缓冲尺寸 (cfgA/cfgB 的逐维最大值) ----
constexpr int kBSMax = 8;
constexpr int kHMax = 128;
constexpr int kKMax = 4;
constexpr int kNumExpandedMax = 32;

static dtype expand_x[kNumExpandedMax * kHMax] __attribute__((aligned(4096))) = {};
static float expert_scales[kNumExpandedMax] __attribute__((aligned(4096))) = {};
static int32_t expand_idx[kNumExpandedMax * 3] __attribute__((aligned(4096))) = {};
static dtype window_data[kNumExpandedMax * kHMax] __attribute__((aligned(4096))) = {};
static float window_flag[kNumExpandedMax * kTileW] __attribute__((aligned(4096))) = {};
static uint32_t window_state[16] __attribute__((aligned(4096))) = {};
static float pred_buf[kNumExpandedMax * kTileW] __attribute__((aligned(4096))) = {};
static dtype out_buf[kBSMax * kHMax] __attribute__((aligned(4096))) = {};


static inline float bf16ToF32(uint16_t raw)
{
    uint32_t full = static_cast<uint32_t>(raw) << 16;
    float v;
    std::memcpy(&v, &full, 4);
    return v;
}

// Round-to-nearest-even fp32 -> bf16 bits (TCVT rounding contract).
static inline uint16_t f32ToBf16Bits(float v)
{
    uint32_t bits;
    std::memcpy(&bits, &v, 4);
    uint32_t lsb = (bits >> 16) & 1u;
    bits += 0x7fffu + lsb;
    return static_cast<uint16_t>(bits >> 16);
}

// Input init runs redundantly on every PE (deterministic, identical writes —
// same convention as moe_combine_mt.cpp), so no extra barrier is needed.
static void genInputs(int64_t bs, int64_t h, int64_t k, int64_t numExpanded)
{
    for (int64_t i = 0; i < numExpanded * h; i++) {
        float fval = static_cast<float>(i) * 0.1f;
        uint32_t bits; std::memcpy(&bits, &fval, 4);
        uint16_t raw = (uint16_t)(bits >> 16);
        std::memcpy(&expand_x[i], &raw, 2);
    }
    for (int64_t tk = 0; tk < numExpanded; tk++) {
        expand_idx[tk * 3 + 0] = 0;
        expand_idx[tk * 3 + 1] = static_cast<int32_t>(tk / k);
        expand_idx[tk * 3 + 2] = static_cast<int32_t>(tk % k);
    }
    // The kernel reads expertScales densely at [n*K + k]; fill the dense
    // prefix so every slot carries a real weight.
    for (int64_t i = 0; i < numExpanded; i++) {
        expert_scales[i] = 0.25f;
    }
}

// 验证 (静态版 moe_combine_mt.cpp 同逻辑, 循环边界运行时化)。
// 返回 0 = PASS; 1..4 = 静态版同款诊断码。
static int verify(int64_t bs, int64_t h, int64_t k, int64_t numExpanded)
{
    // 1. out rows == fp32 scale-weighted sum of the K window slots,
    //    converted to bf16 (kernel accumulates in fp32 via TCVT/TMULS/TADD,
    //    then TCVT back; allow the last mantissa bit to round either way).
    for (int64_t n = 0; n < bs; n++) {
        for (int64_t j = 0; j < h; j++) {
            float acc = 0.0f;
            for (int64_t kk = 0; kk < k; kk++) {
                uint16_t raw;
                std::memcpy(&raw, &expand_x[(n * k + kk) * h + j], 2);
                acc += expert_scales[n * k + kk] * bf16ToF32(raw);
            }
            uint16_t expBits = f32ToBf16Bits(acc);
            uint16_t actBits;
            std::memcpy(&actBits, &out_buf[n * h + j], 2);
            uint16_t diff = expBits ^ actBits;
            if (diff != 0 && diff != 1) return 1;
        }
    }

    // 2. window flags all cleared by the reduce stage
    for (int64_t i = 0; i < numExpanded * kTileW; i++) {
        if (window_flag[i] != 0.0f) return 2;
    }

    // 3. window state: toggle 0 -> 1, run flags, token-count writeback
    if (window_state[0] != 1u || window_state[1] != 1u ||
        window_state[2] != 0u ||
        window_state[4] != static_cast<uint32_t>(bs)) {
        return 3;
    }

    // 4. predBuf carries the 1.0f flag copies from the last check round
    for (int64_t slot = 0; slot < numExpanded; slot++) {
        if (pred_buf[slot * kTileW] != 1.0f) return 4;
    }

    return 0;
}

int main() {
    const uint32_t tid = get_thread_idx();

    const int64_t cfgA[4] = {8, 128, 4, 32};   // 与静态版同值 (等价回归)
    const int64_t cfgB[4] = {7, 128, 3, 21};   // 非整除 (ceil 分片覆盖)
    const int64_t* cfgs[2] = {cfgA, cfgB};

    int failA = 0;
    for (int c = 0; c < 2; ++c) {
        // Barrier reset: 静态相位编号跨次调用会因陈旧 flag 立即通过而
        // 失去同步, 每次调用前清零 (各 PE 冗余同值写, 无需栅栏)。
        for (int t = 0; t < kCombineMtThreads; ++t) sCombineMtPhaseDone[t] = 0;
        // Window state 头清零: 两次运行各完成一次 0→1 toggle, 保证逐组
        // 验证时 state[0]==1 成立 (各 PE 冗余同值写)。
        window_state[0] = 0;
        window_state[1] = 0;
        window_state[2] = 0;

        genInputs(cfgs[c][0], cfgs[c][1], cfgs[c][2], cfgs[c][3]);

        BENCHSTART;

        moe_combine_mt_dyn<dtype, dtype, kTileW>(
            expand_x, expert_scales, expand_idx, window_data, window_flag,
            window_state, pred_buf, out_buf, cfgs[c]);

        BENCHEND;

        // cfgA 验证须在其输入被 cfgB 数据生成覆盖之前完成: PE0 立即验证,
        // 其余 PE 在 combineMtBarrier(4) 汇合等待 (kernel 内部相位为 1/2)。
        if (c == 0) {
            if (tid == 0) {
                failA = verify(cfgs[0][0], cfgs[0][1], cfgs[0][2], cfgs[0][3]);
            }
            combineMtBarrier(4);
        }
    }

    // Verification runs on PE0 only (all outputs are visible after the
    // kernel's final barrier). Non-leader PEs return right away: since the
    // 09-01 functional-model change (direct-boot PE exits are independent)
    // a worker's exit no longer truncates PE0's verification, and each PE
    // terminating itself lets the simulation finish cleanly.
    if (tid != 0) {
        return 0;
    }

    if (failA != 0) return failA;                    // cfgA: 静态版同款诊断码
    int rcB = verify(cfgs[1][0], cfgs[1][1], cfgs[1][2], cfgs[1][3]);
    return rcB == 0 ? 0 : 10 + rcB;                  // cfgB: +10 偏移诊断码
}
