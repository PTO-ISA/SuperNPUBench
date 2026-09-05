#include <common/pto_tileop.hpp>
#include <cstdint>
#include <cmath>
#include <cstring>
#include "benchmark.h"
#include "moe_dispatch_mt/moe_dispatch_mt.hpp"

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

// Keeps the non-leader park loop side-effecting so -O2 cannot drop it.
static volatile uint32_t sParkSink = 0;

int main() {
    const uint32_t tid = get_thread_idx();

    // Input init runs redundantly on every PE (deterministic, identical
    // writes — same convention as group_token_vec_mt's genTopkIndex), so no
    // extra barrier is needed before the kernel reads x / expertIds.
    for (int i = 0; i < kBS * kH; i++) {
        float fval = static_cast<float>(i) * 0.1f;
        uint32_t bits; std::memcpy(&bits, &fval, 4);
        uint16_t raw = (uint16_t)(bits >> 16);
        std::memcpy(&x[i], &raw, 2);
    }
    for (int i = 0; i < kBS * kK; i++) {
        expertIds[i] = i % kMoeExpertNum;
        expertScales[i] = 0.25f;
    }

    BENCHSTART;

    moe_dispatch_mt<dtype, kBS, kH, kK, kMoeExpertNum, kTileW, kWindowStride>(
        x, expertIds, expertScales,
        expandXOut, expandIdxOut, expandScalesOut,
        sendCountsOut, expertTokenNumsOut,
        windowData, windowFlag, predBuf, windowTriple, windowState, outBuf,
        cntLocal, expertStarts);

    BENCHEND;

    // Verification runs on PE0 only (all outputs are visible after the
    // kernel's final barrier). Non-leader PEs park: a worker reaching _end
    // first raises exit_group and would truncate PE0's verification mid-way
    // (the leader's exit ends the parked workers instead).
    if (tid != 0) {
        for (;;) {
            sParkSink = tid;
        }
    }

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
