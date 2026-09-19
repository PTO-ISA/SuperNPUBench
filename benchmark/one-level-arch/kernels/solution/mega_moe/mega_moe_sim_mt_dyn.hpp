#pragma once

/**
 * @file mega_moe_sim_mt_dyn.hpp
 * @brief MegaMoe A8W8 — 真 4PE (multi-thread SPMD) 分片版, runtime DYNAMIC
 *        SHAPE kernel 入口。
 *
 * 与 mega_moe_sim_mt.hpp (静态 shape 4PE 版) 的关系:
 *   - 算子语义与阶段划分完全一致 (统计清零 → 量化解码 → mask/dispatch →
 *     GMM1 → SwiGLU → GMM2 → Combine → Unpermute → 统计导出);
 *   - 唯一区别: bs/h/hiddenDim/expertPerRank/topK 编译期不可知, 运行期由
 *     tiling 指针传入, 全部切分与 workspace 布局参数运行期计算;
 *   - 本文件 include mega_moe_sim.hpp (GM extern / 常量集 / Tiling 结构 /
 *     FP8 E4M3+E8M0 解码 / exp_approx), 不 include mega_moe_sim_mt.hpp;
 *     barrier 静态数组独立命名 (sMtPhaseDoneDyn), 避免多重定义。
 *
 *   tiling = {bs, h, hiddenDim, expertPerRank, topK}   (int64)
 *
 * 调用契约 (违例由各 PE 同值判定并提前返回, 不触达栅栏, 无死锁):
 *   · bs > 0, topK > 0, expertPerRank > 0
 *   · h % 2 == 0, hiddenDim % 2 == 0        (SwiGLU 对半切)
 *   · h % 32 == 0, hiddenDim % 32 == 0      (E4M3 权重量化组: 每 32 元素
 *     一组 scale, scaleStride = (h/32)*8 与 scale 缓冲布局绑定)
 *
 * workspace 契约 (driver 按 max-shape 分配): 布局与 MegaMoeWave::Init 一致,
 *   尾部追加每 PE 私有 GMM scratch (y1/y2/y3/yAcc):
 *     w1F32 / w2F32 / dispatch / mask / combine / stats /
 *     yScratch[4 PE × (hiddenDim + hiddenDim/2 + h + h)]
 *
 * 运行契约 (与 _mt 系列一致): 必须
 *     gfrun -f <elf> -s softcore.multiThreadNum=4
 * 单线程运行会在 mtBarrierDyn 处死锁。
 */

#include "solution/mega_moe/mega_moe_sim.hpp"

namespace mega_moe {

// ============================================================================
// multi-thread 基础设施 (独立命名, 与 mega_moe_sim_mt.hpp 约定相同但可
// 与静态版头文件同 TU 共存)
// ============================================================================
constexpr uint32_t kMtThreadsPerBlockDyn = 4U;

static volatile uint32_t sMtPhaseDoneDyn[kMtThreadsPerBlockDyn];

static inline void mtCompilerBarrierDyn()
{
    __asm__ volatile("" : : : "memory");
}

static inline void mtBarrierDyn(uint32_t phase)
{
    mtCompilerBarrierDyn();
    sMtPhaseDoneDyn[get_thread_idx()] = phase;
    mtCompilerBarrierDyn();
    for (uint32_t t = 0U; t < kMtThreadsPerBlockDyn; ++t) {
        while (sMtPhaseDoneDyn[t] < phase) {
        }
    }
    mtCompilerBarrierDyn();
}

// ============================================================================
// kernel 入口 — 真 4PE 分片 + 运行期动态 shape
// ============================================================================
static inline void mega_moe_sim_mt_dyn_kernel(float* yOut, float* xIn,
                                              int64_t* tokOut,
                                              const int64_t* tiling)
{
    // [AIC 冒烟] 源: AIC 核读 aGmAddr 即弃 (只读, 各 PE 重复无害)
    volatile uint32_t smoke = *reinterpret_cast<const volatile uint32_t*>(xIn);
    (void)smoke;

    const uint32_t tid = get_thread_idx();

    // ---- 运行期 shape 与运行时契约 ----
    const uint32_t bs_r = static_cast<uint32_t>(tiling[0]);
    const uint32_t h_r = static_cast<uint32_t>(tiling[1]);
    const uint32_t hiddenDim_r = static_cast<uint32_t>(tiling[2]);
    const uint32_t expertPerRank_r = static_cast<uint32_t>(tiling[3]);
    const uint32_t topK_r = static_cast<uint32_t>(tiling[4]);
    if (tid >= kMtThreadsPerBlockDyn) return;
    if (bs_r == 0U || h_r == 0U || hiddenDim_r == 0U ||
        expertPerRank_r == 0U || topK_r == 0U) return;
    if ((h_r % 2U) != 0U || (hiddenDim_r % 2U) != 0U) return;   // SwiGLU 对半切
    if ((h_r % 32U) != 0U || (hiddenDim_r % 32U) != 0U) return; // scale 分组

    // TilingData 填充 (源 main 逐字段初始化, 字段全保留; 各 PE 幂等冗余写同值)
    static_assert(kBlockAivNum % kMtThreadsPerBlockDyn == 0U,
                  "AIV core count must be divisible by PE count");
    static MegaMoeTilingData tilingData;
    tilingData.moeExpertPerRank = expertPerRank_r;
    tilingData.bs = bs_r;
    tilingData.h = h_r;
    tilingData.hiddenDim = hiddenDim_r;
    tilingData.epWorldSize = kEpWorldSize;
    tilingData.blockNumPerEP = kBlockNumPerEp;
    tilingData.maxOutputSize = kMaxOutputSize;
    tilingData.topK = topK_r;
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

    // workspace 布局 (与 MegaMoeWave::Init 一致) — 运行期 dims
    const uint32_t w1F32Offset = 0U;
    const uint32_t w2F32Offset = w1F32Offset + tilingData.moeExpertPerRank * tilingData.h * tilingData.hiddenDim * 4U;
    const uint32_t dispatchOffset = w2F32Offset + tilingData.moeExpertPerRank * (tilingData.hiddenDim / 2U) * tilingData.h * 4U;
    const uint32_t maskOffset = dispatchOffset + tilingData.moeExpertPerRank * tilingData.bs * 4U;
    const uint32_t combineOffset = maskOffset + tilingData.moeExpertPerRank * tilingData.bs * 1U;
    const uint32_t statsOffset = combineOffset + tilingData.bs * tilingData.h * 4U;
    // 每 PE 私有 GMM scratch: y1[hiddenDim] y2[hiddenDim/2] y3[h] yAcc[h]
    const uint32_t perPeScratchElems = tilingData.hiddenDim + tilingData.hiddenDim / 2U
                                     + tilingData.h + tilingData.h;
    const uint32_t yScratchOffset = statsOffset + kBlockAivNum * tilingData.moeExpertPerRank * 4U;
    float* const y1Scratch = reinterpret_cast<float*>(g_mmWorkspace + yScratchOffset)
                           + tid * perPeScratchElems;
    float* const y2Scratch = y1Scratch + tilingData.hiddenDim;
    float* const y3Scratch = y2Scratch + tilingData.hiddenDim / 2U;
    float* const yAccScratch = y3Scratch + tilingData.h;

    // PE id 与伪核归属: PE tid 拥有伪核 [tid*4, tid*4+4)
    constexpr uint32_t kCoresPerPE = kBlockAivNum / kMtThreadsPerBlockDyn;
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
    mtBarrierDyn(1U);
#else
    // ============ 完整真机流水 — 真 4PE 分片执行 (运行期 dims) ============
    // ---- 阶段 1: 输入准备 (各 PE 写域不相交) ----
    // SendAndQuantBuffInit (9731): 统计槽清零 — 每 PE 只清自己 4 个伪核
    {
        // 注: volatile 清零 —— 连续 i32 零 store 会被 linxv5 continuous-mem-opt
        //     合并为 32B tile store (v4i64 BUILD_VECTOR, Cannot select 崩溃)
        volatile int32_t* stats = reinterpret_cast<volatile int32_t*>(g_mmWorkspace + statsOffset);
        for (uint32_t lc = 0U; lc < kCoresPerPE; ++lc) {
            const uint32_t core = tid * kCoresPerPE + lc;
            for (uint32_t e = 0; e < tilingData.moeExpertPerRank; ++e) {
                stats[core * tilingData.moeExpertPerRank + e] = 0;
            }
        }
    }
    // QuantizeLocalTokens (3754): A8W8 权重 E4M3→fp32 × E8M0 scale
    // w1 flat 线性域 4 等分 (ceil + 钳位); flat = e*h*hd + k*hd + n
    {
        const uint32_t total1 = tilingData.moeExpertPerRank * tilingData.h * tilingData.hiddenDim;
        const uint32_t perPE1 = (total1 + kMtThreadsPerBlockDyn - 1U) / kMtThreadsPerBlockDyn;
        uint32_t begin1 = tid * perPE1;
        if (begin1 > total1) begin1 = total1;
        uint32_t end1 = begin1 + perPE1;
        if (end1 > total1) end1 = total1;
        float* w1 = reinterpret_cast<float*>(g_mmWorkspace + w1F32Offset);
        for (uint32_t flat = begin1; flat < end1; ++flat) {
            const uint32_t n = flat % tilingData.hiddenDim;
            const uint32_t k = (flat / tilingData.hiddenDim) % tilingData.h;
            const uint32_t e = flat / (tilingData.h * tilingData.hiddenDim);
            const uint32_t group = (k * tilingData.hiddenDim + n) / 32U;
            const uint32_t scaleIdx = e * (tilingData.h * tilingData.hiddenDim / 32U) + group;
            const uint32_t scaleStride = (tilingData.h / 32U) * 2U * 2U * 2U;
            w1[flat] = fp8_e4m3_to_f32(g_mmWeight1[flat]) *
                       fp8_e8m0_scale(g_mmWeightScales1[scaleIdx % scaleStride]);
        }
    }
    // w2 flat 线性域 4 等分; flat = e*(hd/2)*h + k*h + n
    {
        const uint32_t k2 = tilingData.hiddenDim / 2U;
        const uint32_t total2 = tilingData.moeExpertPerRank * k2 * tilingData.h;
        const uint32_t perPE2 = (total2 + kMtThreadsPerBlockDyn - 1U) / kMtThreadsPerBlockDyn;
        uint32_t begin2 = tid * perPE2;
        if (begin2 > total2) begin2 = total2;
        uint32_t end2 = begin2 + perPE2;
        if (end2 > total2) end2 = total2;
        float* w2 = reinterpret_cast<float*>(g_mmWorkspace + w2F32Offset);
        for (uint32_t flat = begin2; flat < end2; ++flat) {
            const uint32_t n = flat % tilingData.h;
            const uint32_t k = (flat / tilingData.h) % k2;
            const uint32_t e = flat / (k2 * tilingData.h);
            const uint32_t group = (k * tilingData.h + n) / 32U;
            const uint32_t scaleIdx = e * (k2 * tilingData.h / 32U) + group;
            const uint32_t scaleStride = (tilingData.h / 32U) * 2U * 2U * 2U;
            w2[flat] = fp8_e4m3_to_f32(g_mmWeight2[flat]) *
                       fp8_e8m0_scale(g_mmWeightScales2[scaleIdx % scaleStride]);
        }
    }
    // GatherAndSendExpertMasks (3934): 自回环本地 mask 表 — 按伪核归属 token 分片
    {
        uint8_t* mask = g_mmWorkspace + maskOffset;
        for (uint32_t lc = 0U; lc < kCoresPerPE; ++lc) {
            const uint32_t coreIdx = tid * kCoresPerPE + lc;
            for (uint32_t i = 0U; i < perCore; ++i) {
                const uint32_t token = coreIdx * perCore + i;
                if (token >= tilingData.bs) break;
                for (uint32_t kk = 0U; kk < tilingData.topK; ++kk) {
                    const uint32_t slot = token * tilingData.topK + kk;
                    const int32_t expert = g_mmTopkIds[slot];
                    mask[static_cast<uint32_t>(expert) * tilingData.bs + slot / tilingData.topK] = 1U;
                }
            }
        }
    }
    // ResetDispatchWorkspace (4051) = DispatchBuffInit: dispatch 表清零 — token 分片
    {
        float* dispatch = reinterpret_cast<float*>(g_mmWorkspace + dispatchOffset);
        for (uint32_t lc = 0U; lc < kCoresPerPE; ++lc) {
            const uint32_t coreIdx = tid * kCoresPerPE + lc;
            for (uint32_t i = 0U; i < perCore; ++i) {
                const uint32_t token = coreIdx * perCore + i;
                if (token >= tilingData.bs) break;
                for (uint32_t e = 0; e < tilingData.moeExpertPerRank; ++e) {
                    dispatch[e * tilingData.bs + token] = -1.0f;
                }
            }
        }
    }

    // ---- 跨 PE 交接点 1: GMM 需读全部 4 PE 解码出的 fp32 权重 → 栅栏 ----
    mtBarrierDyn(1U);

    // ---- 阶段 2: 共享专家输入准备 (源 PrepareSharedExpertInput, sharedExpertNum==0 跳过) ----

    // ---- 阶段 3: MoE 专家流水 (ProcessMoeExpertStages → ProcessGmmPipeline) ----
    // InitTokenUnpermuteBuffers: combine 区不清零 —— Combine 按 token 直接赋值
    // PE tid 只处理自己 4 个伪核的 token; combine 写域 = 本 PE token 集 (不相交)
    // topK 通用语义: yAcc = Σ_kk topkWeight[kk] * y3[kk] (topK==1 时与静态版
    // 直接赋值等价)
    {
        const float* w1 = reinterpret_cast<const float*>(g_mmWorkspace + w1F32Offset);
        const float* w2 = reinterpret_cast<const float*>(g_mmWorkspace + w2F32Offset);
        for (uint32_t lc = 0U; lc < kCoresPerPE; ++lc) {
            const uint32_t coreIdx = tid * kCoresPerPE + lc;
            for (uint32_t i = 0; i < perCore; ++i) {
                const uint32_t token = coreIdx * perCore + i;
                if (token >= tilingData.bs) break;  // bs < 16 核时尾核空转

                for (uint32_t n = 0U; n < tilingData.h; ++n) yAccScratch[n] = 0.0f;

                for (uint32_t kk = 0U; kk < tilingData.topK; ++kk) {
                    const uint32_t slot = token * tilingData.topK + kk;
                    // 路由 (源 expert 分配): topkIds[token*topK + kk]
                    const int32_t expert = g_mmTopkIds[slot];
                    const float weight = g_mmTopkWeights[slot];

                    // GMM1 (源 ProcessGmm1Wave): y1[n] = Σ_k x[k]·w1[e][k][n]
                    {
                        const float* xRow = xIn + token * tilingData.h;
                        const float* wRowBase = w1 + static_cast<uint32_t>(expert) * tilingData.h * tilingData.hiddenDim;
                        for (uint32_t n = 0; n < tilingData.hiddenDim; ++n) {
                            float acc = 0.0f;
                            for (uint32_t k = 0; k < tilingData.h; ++k) {
                                acc += xRow[k] * wRowBase[k * tilingData.hiddenDim + n];
                            }
                            y1Scratch[n] = acc;
                        }
                    }
                    // SwiGLU (源 Gmm1SwigluState): y2 = silu(y1[:k]) * y1[k:]
                    for (uint32_t k = 0; k < tilingData.hiddenDim / 2U; ++k) {
                        const float z = y1Scratch[k];
                        const float sig = 1.0f / (1.0f + exp_approx(-z));
                        y2Scratch[k] = z * sig * y1Scratch[k + tilingData.hiddenDim / 2U];
                    }
                    // GMM2 (源 ProcessGmm2Wave): y3[n] = Σ_k y2[k]·w2[e][k][n]
                    {
                        const uint32_t k2 = tilingData.hiddenDim / 2U;
                        const float* wRowBase = w2 + static_cast<uint32_t>(expert) * k2 * tilingData.h;
                        for (uint32_t n = 0; n < tilingData.h; ++n) {
                            float acc = 0.0f;
                            for (uint32_t k = 0; k < k2; ++k) {
                                acc += y2Scratch[k] * wRowBase[k * tilingData.h + n];
                            }
                            y3Scratch[n] = acc;
                        }
                    }
                    // Combine (源 ProcessCombineExperts): yAcc += topkWeight * y3
                    for (uint32_t n = 0; n < tilingData.h; ++n) {
                        yAccScratch[n] += weight * y3Scratch[n];
                    }
                }

                // combine 写回 (topK==1 时与源/静态版 "直接赋值" 等价)
                {
                    float* combine = reinterpret_cast<float*>(g_mmWorkspace + combineOffset);
                    float* yBase = combine + token * tilingData.h;
                    for (uint32_t n = 0; n < tilingData.h; ++n) {
                        yBase[n] = yAccScratch[n];
                    }
                }
            }
        }
    }
    // UnpermuteTokens (9163): combine 缓冲按 token 序写回 y
    // (token 分片与 combine 写方同 PE, 程序序保证可见, 免栅栏)
    {
        const float* combine = reinterpret_cast<const float*>(g_mmWorkspace + combineOffset);
        for (uint32_t lc = 0U; lc < kCoresPerPE; ++lc) {
            const uint32_t coreIdx = tid * kCoresPerPE + lc;
            for (uint32_t i = 0U; i < perCore; ++i) {
                const uint32_t t = coreIdx * perCore + i;
                if (t >= tilingData.bs) break;
                for (uint32_t n = 0; n < tilingData.h; ++n) {
                    yOut[t * tilingData.h + n] = combine[t * tilingData.h + n];
                }
            }
        }
    }

    // ---- 阶段 4: 跨 rank 同步 (自回环本地退化) 与统计导出 ----
    // stats 按伪核归属分片 (每 PE 只写自己 4 个伪核的槽, 写不相交)
    {
        int32_t* stats = reinterpret_cast<int32_t*>(g_mmWorkspace + statsOffset);
        for (uint32_t lc = 0U; lc < kCoresPerPE; ++lc) {
            const uint32_t core = tid * kCoresPerPE + lc;
            for (uint32_t e = 0; e < tilingData.moeExpertPerRank; ++e) {
                uint32_t cnt = 0U;
                for (uint32_t i = 0; i < perCore; ++i) {
                    const uint32_t token = core * perCore + i;
                    if (token >= tilingData.bs) break;  // bs < 16 核时尾核空转
                    for (uint32_t kk = 0U; kk < tilingData.topK; ++kk) {
                        if (static_cast<uint32_t>(g_mmTopkIds[token * tilingData.topK + kk]) == e) {
                            ++cnt;
                        }
                    }
                }
                stats[core * tilingData.moeExpertPerRank + e] = static_cast<int32_t>(cnt);
            }
        }
        // tokOut: PE0 独占导出 (寄存器计数 + 一次性 volatile 写,
        // 规避相邻 i64 store 合并为 16B tile store 的 v2i64 崩溃)
        if (tid == 0U) {
            volatile int64_t* tokExport = tokOut;
            for (uint32_t e = 0; e < tilingData.moeExpertPerRank; ++e) {
                uint32_t cnt = 0U;
                for (uint32_t t = 0; t < tilingData.bs; ++t) {
                    for (uint32_t kk = 0U; kk < tilingData.topK; ++kk) {
                        if (static_cast<uint32_t>(g_mmTopkIds[t * tilingData.topK + kk]) == e) {
                            ++cnt;
                        }
                    }
                }
                tokExport[e] = static_cast<int64_t>(cnt);
            }
        }
    }

    // ---- 跨 PE 交接点 2: 末端汇合栅栏 ----
    // 所有 PE 的 yOut / tokOut 写入完成后才返回 (PE0 随后的验证依赖全量输出)
    mtBarrierDyn(2U);
#endif
}

}  // namespace mega_moe
