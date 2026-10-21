#pragma once

/**
 * @file mega_moe_sim_mt.hpp
 * @brief MegaMoe A8W8 — 真 4PE (multi-thread SPMD) 分片版 kernel 入口。
 *
 * 与原版 kernels/solution/mega_moe/mega_moe_sim.hpp 的关系:
 *   - 算子语义完全一致: GM 缓冲 extern / 常量集 / Tiling 结构 / FP8 (E4M3+E8M0)
 *     解码全部直接复用原头文件 (本文件 include 之, 不复制);
 *   - 仅执行模型不同 —— 原版为幂等冗余执行 (每个 PE 跑全量 16 伪核, 靠幂等写
 *     保证正确); 本版为真 4PE 分工:
 *       16 伪核 = 4 PE × 4 伪核, PE tid 拥有伪核 [tid*4, tid*4+4),
 *       token 域按伪核 ceil 分片, 各 PE 写域不相交;
 *   - 分片划分 (A1/A2/A3/B8/B9/A6 对齐 2026-10-13; 与 flat 版同阶段集):
 *       阶段 1   统计槽清零 (PE 分片) / 权重量化解码 (每 PE 私有 fp16 副本,
 *                ×2 幂等双遍) / mask 表全量幂等散射 (A5 结构载体)
 *       阶段 1.5 dispatch 计划 (topkIds 标量, 幂等同值写 — MSCARRIER→标量读
 *                存在 TLSU tile-store→GM 缺口, 计数不走 mask) + 核内统计
 *                (PE 分片) + 量化 token/scale/metaInfo 搬运 (slot 分片, ×2
 *                幂等双遍) — A5 CopyTokensAndMetaForDispatch 通路
 *       栅栏 2   dispatch 通路全量可见 (A5 flagDispatchToGmm1; GMM 波跨 PE
 *                读 xDisp/metaInfo/starts)
 *       阶段 3   expert wave ×2 幂等双遍 (B8: PE 行片 ∩ expert 段, 权重 tile
 *                组内复用; GMM1/GMM2 = TGEMV_MX, SwiGLU = VEC 链) → combine
 *                窗口未加权行 (B9, slot = TOKEN*topK+TOPK)
 *       栅栏 3   combine 窗口全量可见 (A5 CrossRankSyncInWorldSize/WaitCombine)
 *       Unpermute 加权反排列 (B9: token 分片, 跨 PE 读 combine; 权重读
 *                topk_weights) + yOut 冲刷
 *       栅栏 4   末端汇合; PE0 随后从核内统计导出 tokOut (A6)
 *   - mtBarrier 为写方定向驱逐版 (moe_combine_mt 同款, 2026-10-13: A1 通路
 *     引入 kernel 内跨 PE tile 数据交接); 相位 1 为 driver 的输入发布屏障
 *   - tile 计算核心 (类型/解码/GEMV/SwiGLU/拷贝) 由 mega_moe_sim.hpp 的
 *     "四a、tile 计算核心" 共享段提供 (与 sim / sim_mt_dyn 同源)。
 *   - mtBarrier 为 volatile per-PE phase flags + compiler barrier, 与
 *     kernels/solution/moe_dispatch / kernels/solution/moe_combine / group_token_vec_mt
 *     同一约定。
 *
 * 运行契约 (与 _mt 系列一致): 必须
 *     gfrun -f <elf> -s softcore.multiThreadNum=4
 * 单线程运行会在 mtBarrier 处死锁。
 *
 * [2026-10-04 修复注] 本文件 + 驱动 (test/solution/mega_moe/src/
 * mega_moe_sim_mt.cpp) + 共享段 (mega_moe_sim.hpp) 的联合修复集, 详见
 * mega_moe_sim_mt_gfsim_fix_report.md:
 *   1. tokOut 统计导出改标量等值计数 (gfsim TLSU tile-store→GM 缺口使
 *      mm_count_eq_i32 的 TSTORE→标量回读恒返 0);
 *   2. 驱动层验证汇合屏障 (写方定向驱逐版): 修复 worker 提前 return 的
 *      退出 lockstep 搁浅, 以及安静窗口裸 flag 屏障的陈旧 flag 活锁;
 *   3. golden 权重解码备忘录化 (验证长尾 ~1.7M → ~0.4M cycle);
 *   4. 阶段 3 x 预量化前置 + 幂等双遍执行: 规避/自愈模型 nuke 恢复在
 *      tile 域丢状态导致的末段 token combine 行默认数据污染;
 *   5. 阶段 4 stats 导出两遍式 (load/store 分离, 减少 nuke 触发面);
 *   6. 驱动 SPMD finisher 约定 (全线程写, 判定经汇合行传递);
 *   7. gfsim 时序仿真须用 `--conf fourpe`; `--conf mt` (单引擎共享 +
 *      默认 TREG 池) 下 tile 流水确定性停摆 (ResVerify @ cycle 317,561,
 *      与 group_token_vec_mt 家族的 mt 配置停摆同类, 模型/配置能力限制)。
 * 修复后 gfsim `--conf fourpe`: linx_test_finisher val=0x5555 PASS,
 * Total Cycles = 1,259,952 (3 次运行确定性一致), tokOut=(8,8) dump 实证;
 * gfrun 4 线程 R2=0。
 */

#include "solution/mega_moe/mega_moe_sim.hpp"

namespace mega_moe {

// ============================================================================
// multi-thread 基础设施 (与 moe_dispatch_mt / moe_combine_mt 同约定)
// ============================================================================
constexpr uint32_t kMtThreadsPerBlock = 4U;

alignas(64) static volatile uint32_t sMtPhaseDone[kMtThreadsPerBlock];

// [2026-10-13 修复] 屏障升级为写方定向驱逐版 (moe_combine_mt 的
// combineMtBarrier 同款): A1 dispatch 通路引入 kernel 内跨 PE tile 数据
// 交接 (xDisp/combine 窗口), 时序模型跨 PE 无 snoop —— 裸 volatile flag
// 自旋存在安静窗口陈旧 flag 活锁 (脏 flag 行滞留各 PE 私有 L1D, 非缓存
// 轮询读 L2 永远看不到发布值; moe_combine_mt 2026-10-04 修复实证)。
// 写方到达即驱逐自身脏 flag 行 (强制写回 L2), 等待方每 32 轮驱逐一次。
// tile 数据本身经 TLSU 直达 L2, 屏障仅提供时序序 + flag 可见性。
alignas(16384) static volatile uint32_t sMtEvictSpan[6 * 4096];

static inline void mtCompilerBarrier()
{
    __asm__ volatile("" : : : "memory");
}

static inline void mtBarrier(uint32_t phase)
{
    mtCompilerBarrier();
    const uint32_t tid = get_thread_idx();
    sMtPhaseDone[tid] = phase;
    mtCompilerBarrier();
    const uint32_t wordOff =
        (static_cast<uint32_t>(
             reinterpret_cast<uint64_t>(&sMtPhaseDone[0]) >> 2)) & 4095u;
    for (uint32_t k = 1; k <= 5; ++k) {
        (void)sMtEvictSpan[k * 4096u + wordOff];   // 写方到达驱逐
    }
    for (uint32_t t = 0U; t < kMtThreadsPerBlock; ++t) {
        uint32_t spins = 0;
        while (sMtPhaseDone[t] < phase) {
            if ((++spins & 31u) == 0u) {
                for (uint32_t k = 1; k <= 5; ++k) {
                    (void)sMtEvictSpan[k * 4096u + wordOff];
                }
            }
        }
    }
    mtCompilerBarrier();
}

// ============================================================================
// kernel 入口 — 真 4PE 分片版 (阶段划分与原版 mega_moe_sim_kernel 一一对应)
// ============================================================================
template <int kBatchSize, int kHiddenDim>
void mega_moe_sim_mt_kernel(float* yOut, float* xIn, int64_t* tokOut)
{
    // [AIC 冒烟] 源: AIC 核读 aGmAddr 即弃 (只读, 各 PE 重复无害)
    volatile uint32_t smoke = *reinterpret_cast<const volatile uint32_t*>(xIn);
    (void)smoke;

    // TilingData 填充 (源 main 逐字段初始化, 字段全保留; 各 PE 幂等冗余写同值)
    static_assert(kBlockAivNum % kMtThreadsPerBlock == 0U,
                  "AIV core count must be divisible by PE count");
    static MegaMoeTilingData tilingData;
    tilingData.moeExpertPerRank = kMoeExpertPerRank;
    tilingData.bs = static_cast<uint32_t>(kBatchSize);
    tilingData.h = static_cast<uint32_t>(kHiddenDim);
    tilingData.hiddenDim = kMoeHiddenDim;
    tilingData.epWorldSize = kEpWorldSize;
    tilingData.blockNumPerEP = kBlockNumPerEp;
    tilingData.maxOutputSize = kMaxOutputSize;
    tilingData.topK = kTopK;
    tilingData.aicNum = kAicNum;
    tilingData.blockAivNum = kBlockAivNum;
    // 注: 三个 64 位零字段经 volatile 写, 阻止后端把相邻 i64 零 store 合并为
    //     16B tile store (BLK_TSTORE v2i64, linxv5 后端 Cannot select 崩溃)
    *reinterpret_cast<volatile int64_t*>(&tilingData.combineQuantMode) = 0;
    tilingData.clampLimit = 0.0f;
    tilingData.groupedMatmulMode = 0;
    *reinterpret_cast<volatile int64_t*>(&tilingData.topoType) = 0;
    tilingData.sharedExpertNum = kSharedExpertNum;
    *reinterpret_cast<volatile uint64_t*>(&tilingData.combineSyncSlotCountPerExpert) = 0;
    tilingData.dispatchBufferConfig = {256, 1, 6, 288};
    tilingData.sendMaskConfigForCoreWithExtraExpert = {256, 1, 2, 64};
    tilingData.sendMaskConfigForCoreWithoutExtraExpert = {256, 1, 6, 64};
    tilingData.sendMaskCoreCountWithExtraExpert = 2;
    tilingData.unpermuteConfigForFullTokenChunk = {16, 6, 128, 128, 64, 0};
    // 注: 全零聚合初始化会被后端合并为 16B 零 tile store (v2i64 Cannot select), 逐字段写规避
    tilingData.unpermuteConfigForTailTokenChunk.tokensPerBatch = 0;
    tilingData.unpermuteConfigForTailTokenChunk.inputBufferCount = 0;
    tilingData.unpermuteConfigForTailTokenChunk.bf16SlotElementCount = 0;
    tilingData.unpermuteConfigForTailTokenChunk.fp32SlotElementCount = 0;
    tilingData.unpermuteConfigForTailTokenChunk.topKWeightsBufferBytes = 0;
    tilingData.unpermuteConfigForTailTokenChunk.topKWeightsConversionBufferBytes = 0;
    tilingData.unpermuteFullTokenChunkCoreCount = kBlockAivNum;
    tilingData.topkWeightsPrefetch = 0;
    tilingData.maxTilesPerExpert = kMaxTilesPerExpert;
    tilingData.actMode = 0;
    tilingData.actSubMode = 0;
    tilingData.activationAlpha = 1.0f;
    tilingData.activationBeta = 1.0f;
    tilingData.mGroupsPerWave = 1;
    tilingData.isPerExpertWeightTensor = false;

    // tile 契约 (issue #180): h%32==0 (TGEMV K-tile/MX scale 组宽),
    // hiddenDim%64==0 (GMM2 的 K=hd/2 须整除 32)
    static_assert(kHiddenDim % 32 == 0, "h must be a multiple of the 32-wide MX K-tile");
    static_assert(kMoeHiddenDim % 64 == 0, "hiddenDim/2 must cover whole MX scale groups");

    // workspace 布局 (A1 dispatch 通路版, mega_moe_sim.hpp 四d MoeWsLayout)
    const MoeWsLayout WS = moe_ws_layout(tilingData.bs, tilingData.h,
                                         tilingData.hiddenDim,
                                         tilingData.moeExpertPerRank,
                                         tilingData.topK);
    const uint32_t slots = tilingData.bs * tilingData.topK;

    // PE id 与伪核归属: PE tid 拥有伪核 [tid*4, tid*4+4)
    const uint32_t tid = get_thread_idx();
    constexpr uint32_t kCoresPerPE = kBlockAivNum / kMtThreadsPerBlock;
    // token 域按伪核 ceil 分片 (bs>=16: 每核 bs/16 连续 token; bs<16: 前 bs 核
    // 各 1 token, 尾核空转) —— 与原版一致, 只是外层只遍历本 PE 的 4 个伪核
    const uint32_t perCore = (tilingData.bs + kBlockAivNum - 1U) / kBlockAivNum;

#ifdef MEGA_MOE_SIM_FAKE
    // ============ 源 10321-10374 camodel 仿真快路径 — 4PE 分片 ============
    // x/y 元素域按伪核分片, 每 PE 只搬自己 4 个伪核的块 (写不相交, 免栅栏)
    const uint32_t fakeTotalElems = tilingData.bs * tilingData.h;
    const uint32_t fakeRounds =
        (fakeTotalElems + kBlockAivNum * kChunkElems - 1U) / (kBlockAivNum * kChunkElems);
    using gmShape = global_tensor<float, RowMajor<1, kChunkElems>>;
    using tileShape = Tile<Location::Vec, float, 1, kChunkElems, BLayout::RowMajor>;
    using itGM = global_iterator<gmShape, tileShape>;
    itGM xIter(xIn);
    itGM yIter(yOut);
    for (uint32_t lc = 0U; lc < kCoresPerPE; ++lc) {
        const uint32_t coreIdx = tid * kCoresPerPE + lc;
        for (uint32_t round = 0U; round < fakeRounds; ++round) {
            const uint32_t base = (round * kBlockAivNum + coreIdx) * kChunkElems;
            if (base >= fakeTotalElems) break;  // 尾核空转
            tileShape t;
            auto srcGT = xIter(0, base);
            TLOAD(t, srcGT);
            TMULS(t, t, -1.0f);
            TEXP(t, t);
            TADDS(t, t, 1.0f);
            TRECIP(t, t);
            auto dstGT = yIter(0, base);
            TSTORE(dstGT, t);
        }
    }
    // 自回环固定统计 (源 10369-10372): PE0 独占导出
    // (volatile 规避相邻 i64 store 被合并为 v2i64 tile store)
    if (tid == 0U) {
        volatile int64_t* tokV = tokOut;
        tokV[0] = static_cast<int64_t>(tilingData.bs) / 2;
        tokV[1] = static_cast<int64_t>(tilingData.bs) / 2;
    }
    // 末端汇合: yOut 全量 (含其他 PE 分片) 就绪后才允许 PE0 验证
    // [2026-10-09 优化] 相位 1→2: driver 在 kernel 前新增输入发布屏障
    // mtBarrier(1) (genInputs 已按 PE 分片, 权重 tile 解码读全量 g_mmWeight*
    // 须待分片写发布; 镜像 mega_moe_sim_mt_dyn 的 mtBarrierDyn(8c+1) 模式)
    mtBarrier(2U);
#else
    // ============ 完整真机流水 — 真 4PE 分片执行 ============
    // [A1/A2/A3/B8/B9/A6 对齐 2026-10-13] dispatch 通路 + expert wave 批处理
    // (阶段结构与 flat 版 mega_moe_sim.hpp 一致; PE 分片 + 跨 PE 交接屏障):
    //   阶段 1   统计清零/权重量化解码/mask scatter (各 PE 写域不相交)
    //   barrier(2)   mask 表全量可见 (A5 SendMask/WaitDispatch 协议对齐)
    //   阶段 1.5 dispatch 计划 (幂等) + 数据搬运 (slot 分片) + 核内统计
    //   barrier(3)   xDisp/metaInfo/starts 全量可见 (A5 flagDispatchToGmm1)
    //   阶段 3   expert wave ×2 幂等双遍 (行分片; B8 权重组内复用)
    //   barrier(4)   combine 窗口全量可见 (A5 CrossRankSyncInWorldSize/
    //               WaitCombine, asc:10416-10421)
    //   Unpermute 加权反排列 (token 分片, 跨 PE 读 combine) + yOut 冲刷
    //   barrier(5)   末端汇合; PE0 导出 tokOut (A6)
    // ---- 阶段 1: 输入准备 (各 PE 写域不相交) ----
    // SendAndQuantBuffInit (9731): 统计槽清零 — 每 PE 只清自己 4 个伪核
    {
        int32_t* stats = reinterpret_cast<int32_t*>(g_mmWorkspace + WS.stats);
        mm_fill_i32(stats + tid * kCoresPerPE * tilingData.moeExpertPerRank,
                    kCoresPerPE * tilingData.moeExpertPerRank, 0);
    }
    // QuantizeLocalTokens (3754) — MX 路径: 每 PE 私有 tile 化解码
    MxTileScratch mx = mx_tile_scratch_at(
        g_mmWorkspace + WS.scratch, tid,
        tilingData.moeExpertPerRank, tilingData.h, tilingData.hiddenDim);
    // [2026-10-13 修复] 解码幂等双遍: 解码段 TLOAD(w1)→TSTORE(w1F16) 相邻
    // 块对的 nuke (实测 ld/st bid 相邻 @~54.8k cycle) 恢复丢 tile 状态时,
    // 污染的 w1F16 tile 会沿 GEMM 传播到 y —— 第二遍以同值覆写自愈
    // (guardPass 同哲学; 第二遍几乎不占墙钟, [2026-10-11 试验注 #2])。
    for (uint32_t decodePass = 0U; decodePass < 2U; ++decodePass) {
        mx_decode_weights_tile(mx, tilingData.moeExpertPerRank,
                               tilingData.h, tilingData.hiddenDim);
    }
    // GatherAndSendExpertMasks (3934): 自回环本地 mask 表 — 全量幂等散射
    // ([2026-10-13 模型注] MSCARRIER tile 写 → 标量读存在 TLSU tile-store→
    // GM 缺口 (mm_count_eq_i32 同族, D15): mt 版计数改从 topkIds 标量统计
    // (moe_dispatch_mt cntLocal 同款 proven 路径), mask 表保留为 A5 结构
    // 载体 (写后不读 — 消费端被模型封锁, A2 部分对齐注记); sim 版 (同 PE
    // 程序序无屏障插入) 保留 mask 行求和消费)
    {
        uint8_t* mask = g_mmWorkspace + WS.mask;
        mm_mask_scatter(g_mmTopkIds, mask, 0U, slots,
                        tilingData.bs, tilingData.topK);
    }

    // ---- 阶段 1.5: dispatch 计划 (幂等, 各 PE 同值; 无跨 PE 依赖) ----
    // ComputeExpertTokenCountAndNotify (4540): expertCnt + starts (cumsumInfo)
    // + dstPos (稳定序); 全部从 topkIds 标量计算 (驱动数据, 屏障 1 已覆盖)。
    // [mt 版注] 独立 -1 复位段省略: 跨 PE tile-store(复位)/scalar-store
    // (plan) 无全局序, 迟到的复位 tile store 会覆写 plan 值; -1 复位语义
    // 由 plan 的无效槽 -1 写承载 (mm_dispatch_plan 非法 eid → 0xFFFFFFFF)。
    {
        uint32_t expertCnt[kMoeEprMax];
        uint32_t startsArr[kMoeEprMax];
        uint32_t dstPosArr[64];
        if (tilingData.moeExpertPerRank <= kMoeEprMax && slots <= 64U) {
            for (uint32_t e = 0U; e < tilingData.moeExpertPerRank; ++e) {
                uint32_t cnt = 0U;
                for (uint32_t s = 0U; s < slots; ++s) {
                    if (static_cast<uint32_t>(g_mmTopkIds[s]) == e) ++cnt;
                }
                expertCnt[e] = cnt;
            }
            uint32_t acc = 0U;
            for (uint32_t e = 0U; e < tilingData.moeExpertPerRank; ++e) {
                startsArr[e] = acc;
                acc += expertCnt[e];
            }
            for (uint32_t s = 0U; s < slots; ++s) {
                const uint32_t e = static_cast<uint32_t>(g_mmTopkIds[s]);
                if (e >= tilingData.moeExpertPerRank) {
                    dstPosArr[s] = 0xFFFFFFFFU;
                    continue;
                }
                uint32_t rank = 0U;
                for (uint32_t j = 0U; j < s; ++j) {
                    if (static_cast<uint32_t>(g_mmTopkIds[j]) == e) ++rank;
                }
                dstPosArr[s] = startsArr[e] + rank;
            }
            // workspace 镜像 (A5 expertRevTokenNums/cumsumInfo/dstPosition
            // 载体; GMM 波读 starts): 幂等同值写 (tilingData 填充同模式)
            int32_t* cntWs = reinterpret_cast<int32_t*>(g_mmWorkspace + WS.cnt);
            int32_t* stWs = reinterpret_cast<int32_t*>(g_mmWorkspace + WS.starts);
            float* dstTab = reinterpret_cast<float*>(g_mmWorkspace + WS.dstPos);
            for (uint32_t e = 0U; e < tilingData.moeExpertPerRank; ++e) {
                cntWs[e] = static_cast<int32_t>(expertCnt[e]);
                stWs[e] = static_cast<int32_t>(startsArr[e]);
            }
            for (uint32_t s = 0U; s < slots; ++s) {
                dstTab[s] = (dstPosArr[s] == 0xFFFFFFFFU)
                              ? -1.0f
                              : static_cast<float>(dstPosArr[s]);
            }
        }
    }
    // InitExpertTokenCountExportBuffers 核内累计 (A6: dispatch 期逐核计数,
    // 本 PE 4 伪核; A5 dispatch job 分核语义)
    {
        int32_t* stats = reinterpret_cast<int32_t*>(g_mmWorkspace + WS.stats);
        const uint32_t perSlot = (slots + kBlockAivNum - 1U) / kBlockAivNum;
        for (uint32_t lc = 0U; lc < kCoresPerPE; ++lc) {
            const uint32_t core = tid * kCoresPerPE + lc;
            for (uint32_t e = 0U; e < tilingData.moeExpertPerRank; ++e) {
                uint32_t cnt = 0U;
                for (uint32_t i = 0U; i < perSlot; ++i) {
                    const uint32_t s = core * perSlot + i;
                    if (s >= slots) break;
                    if (static_cast<uint32_t>(g_mmTopkIds[s]) == e) ++cnt;
                }
                stats[core * tilingData.moeExpertPerRank + e] =
                    static_cast<int32_t>(cnt);
            }
        }
    }
    // CopyTokensAndMetaForDispatch (4332) + DispatchExpertTokens (4447):
    // 量化 token+scale 搬运 + metaInfo — slot 分片 (本 PE token 段;
    // xDisp/metaInfo 行域经 dstPos 置换后各 PE 写域不相交)
    {
        __half* xDisp = reinterpret_cast<__half*>(g_mmWorkspace + WS.xDisp);
        uint8_t* scaleDisp = g_mmWorkspace + WS.scaleDisp;
        int32_t* meta = reinterpret_cast<int32_t*>(g_mmWorkspace + WS.metaInfo);
        const float* dstTab = reinterpret_cast<const float*>(
            g_mmWorkspace + WS.dstPos);
        const uint32_t tokBegin = tid * kCoresPerPE * perCore;
        const uint32_t nTok = (tokBegin < tilingData.bs)
                            ? ((tilingData.bs - tokBegin < kCoresPerPE * perCore)
                               ? (tilingData.bs - tokBegin)
                               : (kCoresPerPE * perCore))
                            : 0U;
        // [2026-10-13 修复] 搬运幂等双遍: xDisp 量化 TSTORE 单遍下 TLSU
        // tile-store→GM 缺口的污染行无人自愈 (wave 双遍只覆写 combine),
        // 第二遍以同值覆写 xDisp/metaInfo (guardPass 同哲学)。
        for (uint32_t dispPass = 0U; dispPass < 2U; ++dispPass) {
        for (uint32_t s = tokBegin * tilingData.topK;
             s < (tokBegin + nTok) * tilingData.topK; ++s) {
            const int32_t e = g_mmTopkIds[s];
            if (e < 0 || static_cast<uint32_t>(e) >= tilingData.moeExpertPerRank) {
                continue;   // 非法 eid 守卫
            }
            const float dstV = dstTab[s];   // GM 读回 (dstPos 表活跃消费)
            const uint32_t row = static_cast<uint32_t>(dstV);
            const uint32_t token = s / tilingData.topK;
            const uint32_t kk = s % tilingData.topK;
            mx_quantize_row(xIn + token * tilingData.h,
                            xDisp + row * tilingData.h, tilingData.h);
            volatile uint8_t* sc = scaleDisp + row;
            *sc = 0x00U;                    // E8M0 0x00 = 1.0 (A5 dispatchRevScale)
            volatile int32_t* mrow = meta + row * kMetaInfoWords;
            mrow[kMetaRankId] = 0;
            mrow[kMetaTokenId] = static_cast<int32_t>(token);
            mrow[kMetaTopkIndex] = static_cast<int32_t>(kk);
            mrow[kMetaWeightIndex] = 0;
            mrow[4] = 0;
            mrow[5] = 0;
            mrow[6] = 0;
            mrow[7] = 0;
        }
        }
        // pad 元数据行 (幂等, 任一 PE 写即可 — 同值): {TOKEN=bs, TOPK=topK-1}
        // → combine 写 dummy 尾行 (永不读取)
        {
            volatile int32_t* mrow = meta + slots * kMetaInfoWords;
            mrow[kMetaRankId] = 0;
            mrow[kMetaTokenId] = static_cast<int32_t>(tilingData.bs);
            mrow[kMetaTopkIndex] =
                static_cast<int32_t>(tilingData.topK - 1U);
            mrow[kMetaWeightIndex] = 0;
            mrow[4] = 0;
            mrow[5] = 0;
            mrow[6] = 0;
            mrow[7] = 0;
        }
    }

    // ---- dispatch 通路全量可见 (A5 flagDispatchToGmm1; GMM 波读他 PE 行) ----
    mtBarrier(2U);

    // ---- 阶段 3: MoE 专家流水 (ProcessMoeExpertStages asc:10222) ----
    // B8 expert-major wave ×2 幂等双遍 ([2026-10-11 试验注 #2] 单遍下末段
    // combine 行被 gfsim TLSU tile-store→GM 缺口污染, 双遍以正确值覆盖,
    // 第二遍几乎不占墙钟 —— 保持 2U); PE 行片 = slots ceil 4 分片
    {
        const __half* xDisp = reinterpret_cast<const __half*>(
            g_mmWorkspace + WS.xDisp);
        const int32_t* meta = reinterpret_cast<const int32_t*>(
            g_mmWorkspace + WS.metaInfo);
        float* const combineWin = reinterpret_cast<float*>(
            g_mmWorkspace + WS.combine);
        const int32_t* stWs = reinterpret_cast<const int32_t*>(
            g_mmWorkspace + WS.starts);
        const uint32_t rowsPerPE =
            (slots + kMtThreadsPerBlock - 1U) / kMtThreadsPerBlock;
        const uint32_t rowBegin = tid * rowsPerPE;
        const uint32_t rowEnd = (rowBegin + rowsPerPE < slots)
                                  ? (rowBegin + rowsPerPE) : slots;
        for (uint32_t guardPass = 0U; guardPass < 2U; ++guardPass) {
            (void)guardPass;
            for (uint32_t e = 0U; e < tilingData.moeExpertPerRank; ++e) {
                const uint32_t segBegin = static_cast<uint32_t>(stWs[e]);
                const uint32_t segEnd =
                    (e + 1U < tilingData.moeExpertPerRank)
                        ? static_cast<uint32_t>(stWs[e + 1U]) : slots;
                const uint32_t eb = (segBegin > rowBegin) ? segBegin : rowBegin;
                const uint32_t ee = (segEnd < rowEnd) ? segEnd : rowEnd;
                // [2026-10-14 优化] 自适应组大小分派 (满组 G=4, 尾组 G=rem):
                // 消除 PE 行片 ∩ expert 段边界尾组的越段 pad 槽
                mx_expert_wave_run(mx, xDisp, eb, ee, slots,
                                   tilingData.h, tilingData.hiddenDim,
                                   e, meta, combineWin,
                                   tilingData.topK);
            }
        }
    }

    // ---- combine 窗口全量可见 (A5 CrossRankSyncInWorldSize / WaitCombine,
    //      asc:10416-10421: 等待所有 rank Combine 发送完成再 Unpermute) ----
    mtBarrier(3U);

    // ---- UnpermuteTokens (9163): 加权反排列 (B9) — token 分片 ----
    // y[token] = Σ_kk weight·combineWin[token*topK+kk] (跨 PE 读 combine)
    {
        const float* combineWin = reinterpret_cast<const float*>(
            g_mmWorkspace + WS.combine);
        const uint32_t tokBegin = tid * kCoresPerPE * perCore;
        const uint32_t nTok = (tokBegin < tilingData.bs)
                            ? ((tilingData.bs - tokBegin < kCoresPerPE * perCore)
                               ? (tilingData.bs - tokBegin)
                               : (kCoresPerPE * perCore))
                            : 0U;
        for (uint32_t i = 0U; i < nTok; ++i) {
            mx_unpermute_token(combineWin, g_mmTopkWeights, yOut,
                               tokBegin + i, tilingData.topK, tilingData.h);
        }
    }

    // [2026-10-12 修复] 写方 yOut 冲刷 (mt_dyn 版同款): 每 PE 在末端汇合
    // 屏障发布前对自己写入的 yOut 行做 volatile 读回 (64B 行粒度), 借同
    // PE load/store 块序强制本 PE 在飞 tile-store 先于屏障 flag 落盘。
    {
        for (uint32_t lc = 0U; lc < kCoresPerPE; ++lc) {
            const uint32_t coreIdx = tid * kCoresPerPE + lc;
            for (uint32_t i = 0; i < perCore; ++i) {
                const uint32_t t = coreIdx * perCore + i;
                if (t >= tilingData.bs) break;
                const volatile float* yRow =
                    reinterpret_cast<const volatile float*>(
                        yOut + t * tilingData.h);
                for (uint32_t c = 0U; c < tilingData.h; c += 16U) {
                    (void)yRow[c];   // 64B 行粒度触碰 (16 × fp32)
                }
            }
        }
    }

    // ---- 末端汇合: 全部 yOut 写入完成后汇合; PE0 随后从核内统计导出 ----
    mtBarrier(4U);

    // ---- 阶段 4: 统计导出 (A6: dispatch 期核内累计求和; A5
    //      ExportExpertTokenCounts(FromWorkspace) 4599/4640) — PE0 独占 ----
    // (汇合后全 PE stats 可见; tokOut 单写者; volatile 规避相邻 i64 store
    // 合并为 16B tile store 的 v2i64 崩溃)
    if (tid == 0U) {
        const int32_t* stats = reinterpret_cast<const int32_t*>(
            g_mmWorkspace + WS.stats);
        volatile int64_t* tokExport = tokOut;
        for (uint32_t e = 0U; e < tilingData.moeExpertPerRank; ++e) {
            uint32_t cnt = 0U;
            for (uint32_t c = 0U; c < kBlockAivNum; ++c) {
                cnt += static_cast<uint32_t>(
                    stats[c * tilingData.moeExpertPerRank + e]);
            }
            tokExport[e] = static_cast<int64_t>(cnt);
        }
    }
#endif
}

}  // namespace mega_moe

