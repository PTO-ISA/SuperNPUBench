/**
 * MoE Dispatch — 4-PE SPMD 动态 shape test driver (moe_dispatch_mt_dyn)
 *
 * 基于 moe_dispatch_mt.cpp 的动态 shape 版驱动:
 *   - 单次运行内顺序执行 2 组运行时 shape:
 *       cfgA: {BS=8, H=128, K=4, MoeExpertNum=4}   — 与静态版同值, 等价回归
 *       cfgB: {BS=7, H=128, K=3, MoeExpertNum=5}   — slotCount=21%4≠0、
 *             MoeExpertNum=5%4≠0, 覆盖运行时 ceil 分片路径
 *   - GM 缓冲按两组 shape 的最大值定长分配, 运行时只使用有效段;
 *   - 两次 kernel 调用之间把 header 内 sMtPhaseDone barrier 数组清零
 *     (静态相位编号在第二次调用会因陈旧 flag 立即通过而失去同步);
 *   - 验证 PE0 独占, 逐组比对 (逻辑与静态版一致, 循环边界运行时化):
 *     cfgA 失败返回静态版同款诊断码 1..7, cfgB 失败返回 +10 偏移码。
 *
 * 运行 (gfrun 功能仿真, 必须四线程):
 *   gfrun -t 1 -s softcore.multiThreadNum=4 -f <elf>
 */

#include <common/pto_tileop.hpp>
#include <cstdint>
#include <cstring>
#include "benchmark.h"
#include "solution/moe_dispatch/moe_dispatch_mt_dyn.hpp"

using namespace supernpu::tile_isa;

using dtype = __bf16;

constexpr int kTileW = 128;
constexpr int kWindowStride = 256;  // 512B / 2B = 256 bf16 per slot
// ---- max-shape 缓冲尺寸 (cfgA/cfgB 的逐维最大值) ----
constexpr int kBSMax = 8;
constexpr int kHMax = 128;
constexpr int kKMax = 4;
constexpr int kExpertMax = 5;
constexpr int kSlotMax = kBSMax * kKMax;

static dtype x[kBSMax * kHMax] __attribute__((aligned(4096))) = {};
static int32_t expertIds[kSlotMax] __attribute__((aligned(4096))) = {};
static float expertScales[kSlotMax] __attribute__((aligned(4096))) = {};
static dtype expandXOut[kSlotMax * kHMax] __attribute__((aligned(4096))) = {};
static int32_t expandIdxOut[kSlotMax * 3] __attribute__((aligned(4096))) = {};
static float expandScalesOut[kSlotMax] __attribute__((aligned(4096))) = {};
static int32_t sendCountsOut[kExpertMax] __attribute__((aligned(4096))) = {};
static int64_t expertTokenNumsOut[kExpertMax] __attribute__((aligned(4096))) = {};
static dtype windowData[kSlotMax * kWindowStride] __attribute__((aligned(4096))) = {};
static float windowFlag[kSlotMax * kTileW] __attribute__((aligned(4096))) = {};
static float predBuf[kSlotMax * kTileW] __attribute__((aligned(4096))) = {};
static int32_t windowTriple[kSlotMax * 3] __attribute__((aligned(4096))) = {};
// 4 header words + TileW flag floats + slack: keeps the 512B cumsum-flag tile
// (stored at windowState+4) inside the buffer.
static uint32_t windowState[4 + kTileW + 8] __attribute__((aligned(4096))) = {};
static dtype outBuf[kSlotMax * kHMax] __attribute__((aligned(4096))) = {};
// Multi-thread scratch: per-PE histograms + expert-major starts
static int32_t cntLocal[kMtThreadsPerBlock * kExpertMax] __attribute__((aligned(4096))) = {};
static int32_t expertStarts[kExpertMax] __attribute__((aligned(4096))) = {};

// Input init runs redundantly on every PE (deterministic, identical writes —
// same convention as moe_dispatch_mt.cpp), so no extra barrier is needed.
static void genInputs(int64_t bs, int64_t h, int64_t k, int64_t expertNum)
{
    for (int64_t i = 0; i < bs * h; i++) {
        float fval = static_cast<float>(i) * 0.1f;
        uint32_t bits; std::memcpy(&bits, &fval, 4);
        uint16_t raw = (uint16_t)(bits >> 16);
        std::memcpy(&x[i], &raw, 2);
    }
    for (int64_t i = 0; i < bs * k; i++) {
        expertIds[i] = static_cast<int32_t>(i % expertNum);
        expertScales[i] = 0.25f;
    }
}

// 验证 (静态版 moe_dispatch_mt.cpp 同逻辑, 循环边界运行时化)。
// 返回 0 = PASS; 1..7 = 静态版同款诊断码。
static int verify(int64_t bs, int64_t h, int64_t k, int64_t expertNum)
{
    const int64_t slotCnt = bs * k;
    static int32_t counts[kExpertMax];
    for (int64_t e = 0; e < expertNum; e++) counts[e] = 0;
    for (int64_t i = 0; i < slotCnt; i++) counts[expertIds[i]]++;
    static int32_t cum[kExpertMax];
    int32_t ac = 0;
    for (int64_t e = 0; e < expertNum; e++) { ac += counts[e]; cum[e] = ac; }
    for (int64_t e = 0; e < expertNum; e++) {
        if (sendCountsOut[e] != cum[e]) return 1;
        if (expertTokenNumsOut[e] != counts[e]) return 2;
    }
    int64_t q = 0;
    for (int64_t e = 0; e < expertNum; e++) {
        for (int64_t i = 0; i < slotCnt; i++) {
            if (expertIds[i] != e) continue;
            int64_t tokenId = i / k;
            int64_t topkId = i % k;
            if (expandIdxOut[q * 3 + 0] != 0) return 3;
            if (expandIdxOut[q * 3 + 1] != tokenId) return 4;
            if (expandIdxOut[q * 3 + 2] != topkId) return 5;
            for (int64_t j = 0; j < h; j++) {
                uint16_t exp_raw, act_raw;
                std::memcpy(&exp_raw, &x[tokenId * h + j], 2);
                std::memcpy(&act_raw, &expandXOut[q * h + j], 2);
                uint16_t diff = exp_raw ^ act_raw;
                if (diff != 0 && diff != 1) return 6;
            }
            q++;
        }
    }
    return (q == slotCnt) ? 0 : 7;
}

int main() {
    const uint32_t tid = get_thread_idx();

    const int64_t cfgA[4] = {8, 128, 4, 4};   // 与静态版同值 (等价回归)
    const int64_t cfgB[4] = {7, 128, 3, 5};   // 非整除 (ceil 分片覆盖)
    const int64_t* cfgs[2] = {cfgA, cfgB};

    int failA = 0;
    for (int c = 0; c < 2; ++c) {
        // Barrier reset: 静态相位编号跨次调用会因陈旧 flag 立即通过而
        // 失去同步, 每次调用前清零 (各 PE 冗余同值写, 无需栅栏)。
        for (int t = 0; t < kMtThreadsPerBlock; ++t) sMtPhaseDone[t] = 0;

        genInputs(cfgs[c][0], cfgs[c][1], cfgs[c][2], cfgs[c][3]);

        BENCHSTART;

        moe_dispatch_mt_dyn<dtype, kTileW, kWindowStride>(
            x, expertIds, expertScales,
            expandXOut, expandIdxOut, expandScalesOut,
            sendCountsOut, expertTokenNumsOut,
            windowData, windowFlag, predBuf, windowTriple, windowState, outBuf,
            cntLocal, expertStarts, cfgs[c]);

        BENCHEND;

        // cfgA 验证须在其输入被 cfgB 数据生成覆盖之前完成: PE0 立即验证,
        // 其余 PE 在 mtBarrier(4) 汇合等待 (kernel 内部相位为 1/2/3)。
        if (c == 0) {
            if (tid == 0) {
                failA = verify(cfgs[0][0], cfgs[0][1], cfgs[0][2], cfgs[0][3]);
            }
            mtBarrier(4);
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
