#include <common/pto_tileop.hpp>
#include <cstdint>
#include <cmath>
#include <cstring>
#include "benchmark.h"
#include "solution/moe_dispatch/moe_dispatch_mt.hpp"

using namespace supernpu::tile_isa;

using dtype = __bf16;

constexpr int kBS = 8;
constexpr int kH = 128;
constexpr int kK = 4;
constexpr int kMoeExpertNum = 4;
constexpr int kTileW = 128;
constexpr int kWindowStride = 256;  // 512B / 2B = 256 bf16 per slot
constexpr int kSlotCount = kBS * kK;

static dtype x[kBS * kH] __attribute__((aligned(4096))) = {};
static int32_t expertIds[kBS * kK] __attribute__((aligned(4096))) = {};
static float expertScales[kBS * kK] __attribute__((aligned(4096))) = {};
static dtype expandXOut[kSlotCount * kH] __attribute__((aligned(4096))) = {};
static int32_t expandIdxOut[kSlotCount * 3] __attribute__((aligned(4096))) = {};
static float expandScalesOut[kSlotCount] __attribute__((aligned(4096))) = {};
static int32_t sendCountsOut[kMoeExpertNum] __attribute__((aligned(4096))) = {};
static int64_t expertTokenNumsOut[kMoeExpertNum] __attribute__((aligned(4096))) = {};
static dtype windowData[kSlotCount * kWindowStride] __attribute__((aligned(4096))) = {};
static float windowFlag[kSlotCount * kTileW] __attribute__((aligned(4096))) = {};
static float predBuf[kSlotCount * kTileW] __attribute__((aligned(4096))) = {};
static int32_t windowTriple[kSlotCount * 3] __attribute__((aligned(4096))) = {};
// 4 header words + TileW flag floats + slack: keeps the 512B cumsum-flag tile
// (stored at windowState+4) inside the buffer.
static uint32_t windowState[4 + kTileW + 8] __attribute__((aligned(4096))) = {};
static dtype outBuf[kSlotCount * kH] __attribute__((aligned(4096))) = {};
// Multi-thread scratch: per-PE histograms + expert-major starts
static int32_t cntLocal[4 * kMoeExpertNum] __attribute__((aligned(4096))) = {};
static int32_t expertStarts[kMoeExpertNum] __attribute__((aligned(4096))) = {};

// PE0 独占验证 (kernel 末端 barrier(4) 后全量输出可见; 逻辑与原内联版
// 逐条一致, 提取为函数以便汇合屏障结构化)
static int verifyDispatchMt();

int main() {
    const uint32_t tid = get_thread_idx();

    // [2026-10-09 优化] Input init 按 PE 写域分片 (原实现 4 PE 冗余全量写):
    //   - x:           pack 只读本 PE slot 段的 token 行 [tid*BS/4, +BS/4)
    //                  (BS%4==0, slot 段 [tid*slotCount/4) 与 token 段对齐);
    //   - expertIds / expertScales: Phase 1/2/3 只读本 PE 的 slot 段
    //                  [tid*slotCount/4, +slotCount/4)。
    // 全部读点在 kernel 内部屏障前均为自写自读 (写域不相交); PE0 验证读
    // 全量, 位于 kernel 末端屏障之后 —— 分片写已发布。
    constexpr int kTokensPerPE = kBS / 4;
    constexpr int kSlotsPerPE = (kBS * kK) / 4;
    const int tokBase = static_cast<int>(tid) * kTokensPerPE;
    const int slotBase = static_cast<int>(tid) * kSlotsPerPE;
    for (int i = 0; i < kTokensPerPE * kH; i++) {
        const int gi = tokBase * kH + i;
        float fval = static_cast<float>(gi) * 0.1f;
        uint32_t bits; std::memcpy(&bits, &fval, 4);
        uint16_t raw = (uint16_t)(bits >> 16);
        std::memcpy(&x[gi], &raw, 2);
    }
    for (int i = 0; i < kSlotsPerPE; i++) {
        const int s = slotBase + i;
        expertIds[s] = s % kMoeExpertNum;
        expertScales[s] = 0.25f;
    }

    BENCHSTART;

    moe_dispatch_mt<dtype, kBS, kH, kK, kMoeExpertNum, kTileW, kWindowStride>(
        x, expertIds, expertScales,
        expandXOut, expandIdxOut, expandScalesOut,
        sendCountsOut, expertTokenNumsOut,
        windowData, windowFlag, predBuf, windowTriple, windowState, outBuf,
        cntLocal, expertStarts);

    BENCHEND;

    // [2026-10-09 修复] 验证汇合屏障 (相位 5, kernel 内部用 1..4) —— 镜像
    // moe_dispatch_mt_dyn 的 mtBarrier(8c+6) 已证模式: PE0 独占验证期间
    // worker 在屏障自旋 (而非先行 park 到退出 lockstep AND 组), PE0 验证
    // 完成后全 PE 一起放行、同时到达 _end。原实现 "Non-leader PEs return
    // right away" 在输入生成分片加速后, PE0 验证与 worker 退出 ecall 的
    // 交错窗口触发 gfsim BFU nuke 恢复断言 (实测 @ cycle 4935,
    // "Can't find occupied local pipe by global fbid"); 汇合屏障消除该
    // 交错, 同时规避 dispatch_mt_dyn 修复注记载的 exit lockstep 搁浅
    // (T_deadlock=9999) 风险。
    int ret = 0;
    if (tid == 0) {
        ret = verifyDispatchMt();
    }
    mtBarrier(5U);

    return ret;
}

// PE0 独占验证 (kernel 末端 barrier(4) 后全量输出可见; 逻辑与原内联版
// 逐条一致, 提取为函数以便汇合屏障结构化)
static int verifyDispatchMt()
{
    int slotCnt = kBS * kK;
    int32_t counts[4] = {0};
    for (int i = 0; i < slotCnt; i++) counts[expertIds[i]]++;
    int32_t cum[4] = {0};
    int32_t ac = 0;
    for (int e = 0; e < 4; e++) { ac += counts[e]; cum[e] = ac; }
    for (int e = 0; e < 4; e++) {
        if (sendCountsOut[e] != cum[e]) return 1;
        if (expertTokenNumsOut[e] != counts[e]) return 2;
    }
    int q = 0;
    for (int e = 0; e < 4; e++) {
        for (int i = 0; i < slotCnt; i++) {
            if (expertIds[i] != e) continue;
            int tokenId = i / kK;
            int topkId = i % kK;
            if (expandIdxOut[q * 3 + 0] != 0) return 3;
            if (expandIdxOut[q * 3 + 1] != tokenId) return 4;
            if (expandIdxOut[q * 3 + 2] != topkId) return 5;
            for (int j = 0; j < kH; j++) {
                uint16_t exp_raw, act_raw;
                std::memcpy(&exp_raw, &x[tokenId * kH + j], 2);
                std::memcpy(&act_raw, &expandXOut[q * kH + j], 2);
                uint16_t diff = exp_raw ^ act_raw;
                if (diff != 0 && diff != 1) return 6;
            }
            q++;
        }
    }
    return (q == slotCnt) ? 0 : 7;
}
