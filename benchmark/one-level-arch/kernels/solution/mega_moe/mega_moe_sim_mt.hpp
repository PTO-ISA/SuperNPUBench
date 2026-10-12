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
 *   - 分片划分 (对照原版各阶段; issue #180 tile 化迁移后):
 *       阶段 1  统计槽清零        每 PE 只清自己 4 个伪核的槽
 *               量化解码 w1/w2     每 PE 私有 tile 化解码 (TCVT + TMULS scale
 *                                 折叠) 为 fp16 权重副本, 无 fp32 workspace,
 *                                 权重读由 driver kernel 前的输入发布屏障
 *                                 mtBarrier(1) 覆盖 (genInputs 按 PE 分片)
 *               mask 表/dispatch   按伪核归属 token 分片
 *       阶段 3  GMM 流水          PE tid 只处理自己 4 个伪核的 token;
 *                                 GMM1/GMM2 = TGEMV_MX (MX FP16 pair),
 *                                 SwiGLU = VEC 链, combine 写域 = 本 PE token
 *                                 集 (不相交)
 *               Unpermute          token 分片与 combine 写方同 PE, 程序序
 *                                 保证可见, 免栅栏 (VEC tile 拷贝)
 *       阶段 4  统计导出          stats 按伪核归属分片; tokOut PE0 独占导出
 *       栅栏    mtBarrier(2)      末端汇合: 全部 yOut/tokOut 就绪后才返回,
 *                                 PE0 验证依赖全量输出 (相位 1 为 driver 的
 *                                 kernel 前输入发布屏障)
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

static volatile uint32_t sMtPhaseDone[kMtThreadsPerBlock];

static inline void mtCompilerBarrier()
{
    __asm__ volatile("" : : : "memory");
}

static inline void mtBarrier(uint32_t phase)
{
    mtCompilerBarrier();
    sMtPhaseDone[get_thread_idx()] = phase;
    mtCompilerBarrier();
    for (uint32_t t = 0U; t < kMtThreadsPerBlock; ++t) {
        while (sMtPhaseDone[t] < phase) {
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
    tilingData.unpermuteConfigForTailTokenChunk.chunkRows = 0;
    tilingData.unpermuteConfigForTailTokenChunk.chunksPerCore = 0;
    tilingData.unpermuteConfigForTailTokenChunk.rowElems = 0;
    tilingData.unpermuteConfigForTailTokenChunk.rowStride = 0;
    tilingData.unpermuteConfigForTailTokenChunk.coreStride = 0;
    tilingData.unpermuteConfigForTailTokenChunk.reserved = 0;
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

    // workspace 布局 (MX 路径: 无 fp32 权重缓存) — 运行期 dims
    const uint32_t dispatchOffset = 0U;
    const uint32_t maskOffset = dispatchOffset + tilingData.moeExpertPerRank * tilingData.bs * 4U;
    const uint32_t combineOffset = maskOffset + tilingData.moeExpertPerRank * tilingData.bs * 1U;
    const uint32_t statsOffset = combineOffset + tilingData.bs * tilingData.h * 4U;
    const uint32_t yScratchOffset = statsOffset + kBlockAivNum * tilingData.moeExpertPerRank * 4U;

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
    // ---- 阶段 1: 输入准备 (各 PE 写域不相交) ----
    // SendAndQuantBuffInit (9731): 统计槽清零 — 每 PE 只清自己 4 个伪核
    // (8 i32 = 一个 [1×8] tile, mm_fill_i32; 替代原 volatile 标量清零及其
    // continuous-mem-opt 合并规避 —— tile store 无合并问题)
    {
        int32_t* stats = reinterpret_cast<int32_t*>(g_mmWorkspace + statsOffset);
        mm_fill_i32(stats + tid * kCoresPerPE * tilingData.moeExpertPerRank,
                    kCoresPerPE * tilingData.moeExpertPerRank, 0);
    }
    // QuantizeLocalTokens (3754) — MX 路径: 每 PE 私有 tile 化解码
    // (TLOAD(E4M3[32,32]) → TCVT(fp16) → TMULS(k 组 scale 折叠) → TSTORE);
    // issue #180: 标量 bit-twiddle / fp32 workspace / 跨 PE 权重栅栏全删
    MxTileScratch mx = mx_tile_scratch_at(
        g_mmWorkspace + yScratchOffset, tid,
        tilingData.moeExpertPerRank, tilingData.h, tilingData.hiddenDim);
    mx_decode_weights_tile(mx, tilingData.moeExpertPerRank,
                           tilingData.h, tilingData.hiddenDim);
    // GatherAndSendExpertMasks (3934): 自回环本地 mask 表 — 按伪核归属
    // token 分片; PE tid 的 token 段 [tid*4*perCore, +4*perCore) 连续
    // (coreIdx = tid*4+lc, token = coreIdx*perCore+i) → slot 段连续
    // (topK 倍乘), mm_mask_scatter tile 散射 (topK!=1 时 helper 标量兜底)
    {
        uint8_t* mask = g_mmWorkspace + maskOffset;
        const uint32_t tokBegin = tid * kCoresPerPE * perCore;
        // 尾核空转钳位 (原 token >= bs break 守卫): bs < 16 核时尾 PE 段截断
        const uint32_t nTok = (tokBegin < tilingData.bs)
                            ? ((tilingData.bs - tokBegin < kCoresPerPE * perCore)
                               ? (tilingData.bs - tokBegin)
                               : (kCoresPerPE * perCore))
                            : 0U;
        mm_mask_scatter(g_mmTopkIds, mask, tokBegin * tilingData.topK,
                        nTok * tilingData.topK,
                        tilingData.bs, tilingData.topK);
    }
    // ResetDispatchWorkspace (4051) = DispatchBuffInit: dispatch 表填 -1 —
    // token 分片; 每 e 的 PE 段 [e*bs + tokBegin, +nTok) 连续 → mm_fill_f32
    {
        float* dispatch = reinterpret_cast<float*>(g_mmWorkspace + dispatchOffset);
        const uint32_t tokBegin = tid * kCoresPerPE * perCore;
        const uint32_t nTok = (tokBegin < tilingData.bs)
                            ? ((tilingData.bs - tokBegin < kCoresPerPE * perCore)
                               ? (tilingData.bs - tokBegin)
                               : (kCoresPerPE * perCore))
                            : 0U;
        for (uint32_t e = 0; e < tilingData.moeExpertPerRank; ++e) {
            mm_fill_f32(dispatch + e * tilingData.bs + tokBegin, nTok, -1.0f);
        }
    }

    // ---- 阶段 2: 共享专家输入准备 (源 PrepareSharedExpertInput, sharedExpertNum==0 跳过) ----

    // ---- 阶段 3: MoE 专家流水 (ProcessMoeExpertStages → ProcessGmmPipeline) ----
    // tile 计算核心 (issue #180): GMM1/GMM2 = TGEMV_MX (MX FP16 pair),
    // SwiGLU = VEC 链, Combine = VEC tile (详见 mega_moe_sim.hpp 四a 共享段)
    // [2026-10-04 修复] x 预量化前置为独立一遍 (mx_quantize_row), 主循环用
    // mx_token_compute_pre 消费: 消除原 per-token 内联 "量化 TSTORE(xf16) →
    // GEMM1 TLOAD(xf16)" 相邻块 GM 往返对触发的 nuke (模型 nuke 恢复在
    // tile 域丢状态 → 在飞链 TSTORE 以默认数据 2.0f 提交, 实测 y[14..15]
    // 行恒 2.0f; 见 mega_moe_sim.hpp mx_quantize_row 处修复注)。预量化槽
    // 位于 workspace 4 PE scratch 槽之后的余量区 (驱动 kWorkspaceBytes 的
    // slack 已扩至 16KB), 每 PE 连续 myTokens*h*2 字节, 写域不相交。
    // [2026-10-04 修复] 幂等双遍执行: 模型 nuke(解码段, 与本算子代码无关)的
    // flush 恢复会丢失在飞 tile 链状态, 落飞窗口内本 PE 末段 token 的
    // combine TSTORE 以默认数据 2.0f 提交 (实测 combine[14..15] 行恒 2.0)。
    // 依赖预测器在首碰撞后学习 (同 tpc 对不再 nuke), 第二遍执行无 nuke、
    // 全链数据正常 resolve —— 以正确值覆盖第一遍的污染行。幂等性: 全部
    // store 目标 (xf16 槽/scratch/combine) 按同值重写, 输入不变 (原单 PE
    // 版即按同哲学做 16 伪核冗余执行)。成本 ~+10^4 cycle (占比 <1%)。
    for (uint32_t guardPass = 0U; guardPass < 2U; ++guardPass) {
        (void)guardPass;
    {
        const uint32_t slotBytes = mx_tile_scratch_bytes(
            tilingData.moeExpertPerRank, tilingData.h, tilingData.hiddenDim);
        __half* const xf16Slots = reinterpret_cast<__half*>(
            g_mmWorkspace + yScratchOffset +
            kMtThreadsPerBlock * slotBytes +
            tid * kCoresPerPE * perCore * tilingData.h * 2U);
        uint32_t nq = 0U;
        for (uint32_t lc = 0U; lc < kCoresPerPE; ++lc) {
            const uint32_t coreIdx = tid * kCoresPerPE + lc;
            for (uint32_t i = 0; i < perCore; ++i) {
                const uint32_t token = coreIdx * perCore + i;
                if (token >= tilingData.bs) break;  // bs < 16 核时尾核空转
                mx_quantize_row(xIn + token * tilingData.h,
                                xf16Slots + nq * tilingData.h, tilingData.h);
                ++nq;
            }
        }
        float* const combineBase = reinterpret_cast<float*>(
            g_mmWorkspace + combineOffset);
        uint32_t iq = 0U;
        for (uint32_t lc = 0U; lc < kCoresPerPE; ++lc) {
            const uint32_t coreIdx = tid * kCoresPerPE + lc;
            for (uint32_t i = 0; i < perCore; ++i) {
                const uint32_t token = coreIdx * perCore + i;
                if (token >= tilingData.bs) break;  // bs < 16 核时尾核空转

                for (uint32_t kk = 0U; kk < tilingData.topK; ++kk) {
                    const uint32_t slot = token * tilingData.topK + kk;
                    const uint32_t expert =
                        static_cast<uint32_t>(g_mmTopkIds[slot]);
                    const float weight = g_mmTopkWeights[slot];
                    mx_token_compute_pre(mx,
                                         xf16Slots + iq * tilingData.h,
                                         tilingData.h, tilingData.hiddenDim,
                                         tilingData.moeExpertPerRank, expert,
                                         weight, kk,
                                         combineBase + token * tilingData.h);
                }
                ++iq;
            }
        }
    }
    }  // [2026-10-04 修复] 幂等双遍执行结束
    // UnpermuteTokens (9163): combine 缓冲按 token 序写回 y (VEC tile 拷贝;
    // token 分片与 combine 写方同 PE, 程序序保证可见, 免栅栏)
    {
        const float* combine = reinterpret_cast<const float*>(g_mmWorkspace + combineOffset);
        for (uint32_t lc = 0U; lc < kCoresPerPE; ++lc) {
            const uint32_t coreIdx = tid * kCoresPerPE + lc;
            for (uint32_t i = 0U; i < perCore; ++i) {
                const uint32_t t = coreIdx * perCore + i;
                if (t >= tilingData.bs) break;
                mx_copy_row_tile(combine + t * tilingData.h,
                                 yOut + t * tilingData.h, tilingData.h);
            }
        }
    }

    // ---- 阶段 4: 跨 rank 同步 (自回环本地退化) 与统计导出 ----
    // stats 按伪核归属分片 (每 PE 只写自己 4 个伪核的槽, 写不相交)
    // [2026-10-04 优化] 两遍式: 先集中加载本 PE 全部 token id 到栈, 再计算
    // 并写 stats —— 消除原实现逐 (core,e) 交错 "load g_mmTopkIds / store
    // stats" 的相邻 ld/st 块对 (gfsim 依赖预测器对跨数组基址无法证不相交,
    // 会保守 nuke; nuke 恢复存在丢寄存器生产者的模型竞态, 减少 nuke 触发
    // 面即降低死锁暴露, 见修复报告 §5.3)。
    {
        int32_t* stats = reinterpret_cast<int32_t*>(g_mmWorkspace + statsOffset);
        uint32_t myIds[kCoresPerPE * 8U];   // 每 core 至多 8 token (perCore 上界)
        uint32_t myCnt[kCoresPerPE];
        uint32_t loaded = 0U;
        for (uint32_t lc = 0U; lc < kCoresPerPE; ++lc) {
            const uint32_t core = tid * kCoresPerPE + lc;
            uint32_t c = 0U;
            for (uint32_t i = 0; i < perCore; ++i) {
                const uint32_t token = core * perCore + i;
                if (token >= tilingData.bs) break;  // bs < 16 核时尾核空转
                myIds[loaded++] = static_cast<uint32_t>(g_mmTopkIds[token * tilingData.topK]);
                ++c;
            }
            myCnt[lc] = c;
        }
        for (uint32_t lc = 0U; lc < kCoresPerPE; ++lc) {
            const uint32_t core = tid * kCoresPerPE + lc;
            for (uint32_t e = 0; e < tilingData.moeExpertPerRank; ++e) {
                uint32_t cnt = 0U;
                uint32_t idx = 0U;
                for (uint32_t p = 0U; p < lc; ++p) idx += myCnt[p];
                for (uint32_t i = 0U; i < myCnt[lc]; ++i) {
                    if (myIds[idx + i] == e) ++cnt;
                }
                stats[core * tilingData.moeExpertPerRank + e] = static_cast<int32_t>(cnt);
            }
        }
        // tokOut: PE0 独占导出 — 等值计数
        // [2026-10-04 修复] 原实现 topK==1 时走 mm_count_eq_i32 tile 计数链
        // (TLOAD+TCMPS+TSEL+TROWSUM+TSTORE → scratch 标量读回)。gfsim
        // (TimingSim HEAD 9905a05f) 的 TLSU tile-store→GM 数据通路存在缺口:
        // TSTORE 执行后在架构内存不可见 (探针实证, 同根因见模型 known-gap-list
        // memory 类 gmov/mscatter/timg2col 微基准; 详见
        // kernels/solution/mega_moe/mega_moe_sim_gfsim_fix_report.md §5),
        // 由此 mm_count_eq_i32 恒返 0 → tokOut=(0,0) vs golden=(8,8) →
        // test-finisher val=0x0001 FAIL。gfrun 功能模型 tile 操作同步执行,
        // 同一 elf R2=0 全对 (单 PE 版 mega_moe_sim.hpp 同款修复)。
        // 修复: 计数长度仅 bs*topK(=16) 个元素, 直接标量等值计数循环
        // (与 mm_count_eq_i32 的 <4 标量尾/topK!=1 兜底同款语义), 不依赖
        // tile-store 回读通路——gfrun/gfsim/真机三端语义一致。
        // volatile 规避相邻 i64 store 合并为 16B tile store 的 v2i64 崩溃
        if (tid == 0U) {
            volatile int64_t* tokExport = tokOut;
            for (uint32_t e = 0; e < tilingData.moeExpertPerRank; ++e) {
                uint32_t cnt = 0U;
                for (uint32_t t = 0; t < tilingData.bs; ++t) {
                    if (static_cast<uint32_t>(g_mmTopkIds[t * tilingData.topK]) == e) ++cnt;
                }
                tokExport[e] = static_cast<int64_t>(cnt);
            }
        }
    }

    // ---- 跨 PE 交接点: 末端汇合栅栏 (fp32 解码栅栏已随 MX 原地消费删除) ----
    // 所有 PE 的 yOut / tokOut 写入完成后才返回 (PE0 随后的验证依赖全量输出)
    // [2026-10-09 优化] 相位 1→2 (driver kernel 前新增输入发布屏障 1)
    mtBarrier(2U);
#endif
}

}  // namespace mega_moe
