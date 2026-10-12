/**
 * MoE Dispatch — 4-PE SPMD 动态 shape test driver (moe_dispatch_mt_dyn)
 *
 * 基于 moe_dispatch_mt.cpp 的动态 shape 版驱动:
 *   - 单次运行内顺序执行 2 组运行时 shape:
 *       cfgA: {BS=8, H=128, K=4, MoeExpertNum=4}   — 与静态版同值, 等价回归
 *       cfgB: {BS=7, H=128, K=3, MoeExpertNum=5}   — slotCount=21%4≠0、
 *             MoeExpertNum=5%4≠0, 覆盖运行时 ceil 分片路径
 *   - GM 缓冲按两组 shape 的最大值定长分配, 运行时只使用有效段;
 *   - barrier 相位由 kernel 内 per-PE 调用计数 (sInvCnt) 单调编号,
 *     跨调用陈旧 flag 天然失效 —— 不做任何跨 PE 的 flag 复位写
 *     (无同步跨 PE 写在时序模型下会抹掉他 PE 已置位的 flag → 活锁,
 *     gtv mt_dyn ROOTCAUSE 同款修复);
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

// [2026-10-09 优化] Input init 按 PE 连续 ceil 分片 (原实现 4 PE 冗余全量
// 写)。分片写域不相交 (单写者/cacheline); 调用方在 kernel 前
// mtBarrier(8c+1) 发布 (pack 的 Phase 1 跨 PE 读 x —— cfgB 的 slot 段
// 边界 token 会被相邻两 PE 读取, 须屏障发布后才能进 kernel)。
static void genInputs(int64_t bs, int64_t h, int64_t k, int64_t expertNum)
{
    const uint32_t tidU = get_thread_idx();
    const int64_t tid = static_cast<int64_t>(tidU);
    // 连续 ceil 分片 [begin, begin+len), 值按绝对下标 (与全量版逐位一致)
    auto slice = [](int64_t n, int64_t me, int64_t &begin, int64_t &len) {
        const int64_t seg = n / 4;
        const int64_t rem = n % 4;
        begin = me * seg + (me < rem ? me : rem);
        len = seg + (me < rem ? 1 : 0);
    };
    {
        int64_t begin, len;
        slice(bs * h, tid, begin, len);
        for (int64_t i = begin; i < begin + len; i++) {
            float fval = static_cast<float>(i) * 0.1f;
            uint32_t bits; std::memcpy(&bits, &fval, 4);
            uint16_t raw = (uint16_t)(bits >> 16);
            std::memcpy(&x[i], &raw, 2);
        }
    }
    {
        int64_t begin, len;
        slice(bs * k, tid, begin, len);
        for (int64_t i = begin; i < begin + len; i++) {
            expertIds[i] = static_cast<int32_t>(i % expertNum);
            expertScales[i] = 0.25f;
        }
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
        genInputs(cfgs[c][0], cfgs[c][1], cfgs[c][2], cfgs[c][3]);

        // [2026-10-09 优化] kernel 前输入发布屏障 (相位 8c+1, 镜像
        // mega_moe_sim_mt_dyn 的 mtBarrierDyn(8c+1) 模式): genInputs 已按
        // PE 分片, pack Phase 1 对 x 的跨 PE 读 (cfgB slot 段边界 token)
        // 须待全部分片写发布。kernel 相位相应移至 8c+2..8c+5。
        mtBarrier(static_cast<uint32_t>(c) * 8U + 1U);

        BENCHSTART;

        moe_dispatch_mt_dyn<dtype, kTileW, kWindowStride>(
            x, expertIds, expertScales,
            expandXOut, expandIdxOut, expandScalesOut,
            sendCountsOut, expertTokenNumsOut,
            windowData, windowFlag, predBuf, windowTriple, windowState, outBuf,
            cntLocal, expertStarts, cfgs[c]);

        BENCHEND;

        // cfgA 验证须在其输入被 cfgB 数据生成覆盖之前完成: PE0 立即验证,
        // 其余 PE 在 mtBarrier(8c+6) 汇合等待 (kernel 单调相位 = inv*8 +
        // 2..5, driver 汇合点取 8*c+6 ——  strictly between cfgA 的 6 与
        // cfgB 的 10, 无陈旧 flag 直通; 见 kernel sInvCnt 注)。
        if (c == 0) {
            if (tid == 0) {
                failA = verify(cfgs[0][0], cfgs[0][1], cfgs[0][2], cfgs[0][3]);
            }
            mtBarrier(6);
        }
    }

    // [2026-10-06 修复] cfgB 验证汇合屏障 (相位 13 = 8*1+5, 本轮 kernel 内部
    // 用 9..12) —— 替换原 "worker 直接 return": 原实现 worker (tid!=0) 在
    // c-loop 末端直接 `return 0`, 先于 PE0 到达 _end 退出 ecall 并 park 到
    // 退出 lockstep AND 组; PE0 独占的 cfgB 验证 (verify 的 expert×slot 扫描
    // + 逐 slot×h 的 expandXOut 逐位比对, gfsim fourpe 标量侧 ~4x 减速下
    // >10^4 cycle) 使退出 AND 组停在 ready=3/4 超过 T_deadlock=9999 →
    // SyscallBarrier.cpp:1889 断言截断 (实测确定性复现 @ cycle 45,448,
    // ready=3/4, bpc=0x11390, 无 stats)。"direct-boot 下各 PE 退出相互独立"
    // 仅对 gfrun 功能模型成立, gfsim 时序模型的退出 ecall 仍按 lockstep
    // AND 组汇聚。
    // 修复镜像 group_token_old_mt / group_token_vec_mt / mega_moe 家族的
    // 已证方案 (cfgA 的 mtBarrier(5) 汇合在本驱动已存在且 gfsim 实证可用,
    // cfgB 补齐同款): 验证在 worker 存活期间执行 (worker 在汇合屏障自旋
    // 而非 park 到退出 lockstep), PE0 验证完成后全 PE 一起放行、同时到达
    // _end, 消除 ready=3/4 长时停滞。
    int ret = 0;
    if (tid == 0) {
        if (failA != 0) {
            ret = failA;                             // cfgA: 静态版同款诊断码
        } else {
            int rcB = verify(cfgs[1][0], cfgs[1][1], cfgs[1][2], cfgs[1][3]);
            ret = (rcB == 0) ? 0 : 10 + rcB;         // cfgB: +10 偏移诊断码
        }
    }
    mtBarrier(8U * 1U + 6U);   // cfgB 验证汇合 (kernel 单调相位 10..13 之后)

    return ret;
}
