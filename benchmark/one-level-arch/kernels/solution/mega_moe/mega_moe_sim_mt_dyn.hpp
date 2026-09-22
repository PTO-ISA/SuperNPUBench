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
 *   · h % 32 == 0    (TGEMV K-tile = 32, MX scale 组宽)
 *   · hiddenDim % 64 == 0  (GMM1 N-tile=32; GMM2 的 K=hiddenDim/2 须整除
 *     32 以覆盖完整 MX scale 组)
 *
 * 计算路径 (issue #180): GMM1/GMM2 = 逐 token `TGEMV_MX`/`TGEMV_MX_ACC`
 * (MX FP16 pair —— pto-spec: FP16/BF16 侧免 scale; CUBE fp32 累加), SwiGLU/
 * Combine/Unpermute/x 量化 = VEC tile 链 (TMULS/TEXP/TADDS/TRECIP/TMUL/TCVT)。
 * FP8 权重量化消费改为 tile 化解码: TLOAD(E4M3 [32,32]) → TCVT(fp16) →
 * TMULS(按 k 组 scale 折叠) → TSTORE, 产物为每 PE 私有 fp16 权重副本 ——
 * 不再有标量 bit-twiddle / fp32 workspace / 跨 PE 权重栅栏。
 *
 * 注: B 侧 E4M3+E8M0 的 MX 原地消费 (TGEMV_MX 带 ScaleB) 在当前
 * 工具链/模型组合下不可用 —— 编译期契约要求 Local ScaleB 为 RowMajor
 * [ceil(K/32), N], 而 TLOAD 发射的 scale 操作数无 CUBE_M32 布局描述符,
 * 运行时模型断言要求 CUBE_M32 [N, ceil(K/32)] (isa/Block.cpp 的
 * TMATMUL_MX right-scale 校验; 复现探针见 issue 回复)。待模型侧修复后
 * 可平滑切换为真 MX 原地消费。
 *
 * 权重 scale 布局 (每 PE 解码时 TMULS 折叠, bench 自控):
 *   w1Scale[e][k/32]  (epr × h/32 个 E8M0 标量)
 *   w2Scale[e][k/32]  (epr × (hd/2)/32 个 E8M0 标量)
 *
 * workspace 契约 (driver 按 max-shape 分配):
 *   dispatch / mask / combine / stats /
 *   yScratch[4 PE × (h*2 + hiddenDim*4 + hiddenDim*2 + h*4) B]
 *     (xf16[h] + y1[hiddenDim]f32 + y2f16[hiddenDim/2] + y3[h]f32)
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
    if ((h_r % 32U) != 0U) return;          // TGEMV K-tile / MX scale 组宽
    if ((hiddenDim_r % 64U) != 0U) return;  // GMM2 K=hd/2 须整除 32

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

    // workspace 布局 (运行期 dims) — MX 路径无 fp32 权重缓存
    const uint32_t dispatchOffset = 0U;
    const uint32_t maskOffset = dispatchOffset + tilingData.moeExpertPerRank * tilingData.bs * 4U;
    const uint32_t combineOffset = maskOffset + tilingData.moeExpertPerRank * tilingData.bs * 1U;
    const uint32_t statsOffset = combineOffset + tilingData.bs * tilingData.h * 4U;
    // 每 PE 私有 fp16 权重副本 + GMM scratch:
    //   w1F16[epr*h*hd] + w2F16[epr*(hd/2)*h] + xf16[h] + y1[hd]f32 +
    //   y2f16[hd/2] + y3[h]f32
    const uint32_t perPeBytes =
        (tilingData.moeExpertPerRank * tilingData.h * tilingData.hiddenDim
       + tilingData.moeExpertPerRank * (tilingData.hiddenDim / 2U) * tilingData.h) * 2U
      + tilingData.h * 2U
      + tilingData.hiddenDim * 4U
      + tilingData.hiddenDim * 2U
      + tilingData.h * 4U;
    const uint32_t yScratchOffset = statsOffset + kBlockAivNum * tilingData.moeExpertPerRank * 4U;
    uint8_t* const yScratchBase = g_mmWorkspace + yScratchOffset
                                + tid * perPeBytes;
    __half* const w1F16 = reinterpret_cast<__half*>(yScratchBase);
    __half* const w2F16 = w1F16
        + tilingData.moeExpertPerRank * tilingData.h * tilingData.hiddenDim;
    __half* const xf16Scratch = w2F16
        + tilingData.moeExpertPerRank * (tilingData.hiddenDim / 2U) * tilingData.h;
    float* const y1Scratch = reinterpret_cast<float*>(
        reinterpret_cast<uint8_t*>(xf16Scratch) + tilingData.h * 2U);
    __half* const y2f16Scratch = reinterpret_cast<__half*>(
        reinterpret_cast<uint8_t*>(y1Scratch) + tilingData.hiddenDim * 4U);
    float* const y3Scratch = reinterpret_cast<float*>(
        reinterpret_cast<uint8_t*>(y2f16Scratch) + tilingData.hiddenDim * 2U);

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
    // QuantizeLocalTokens (3754): MX 路径原地消费 E4M3+E8M0 权重 —— 不再有
    // fp32 解码 workspace 与对应的跨 PE 栅栏 (issue #180: 解码阶段整体删除)
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

    // ---- 阶段 1.5 (issue #180): FP8 权重 tile 化解码 (每 PE 私有, 免栅栏) ----
    // w1F16[e][k][n] = fp16(E4M3) * 2^(w1Scale[e][k/32]-127), 逐 [32,32] tile:
    // TLOAD(E4M3) → TCVT(fp16) → TMULS(组 scale) → TSTORE
    {
        using DecSrc = Tile<Location::Vec, __fp8_e4m3, 32, 32>;
        using DecDst = Tile<Location::Vec, __half, 32, 32>;
        for (uint32_t e = 0U; e < tilingData.moeExpertPerRank; ++e) {
            for (uint32_t k0 = 0U; k0 < tilingData.h; k0 += 32U) {
                const float s1 = fp8_e8m0_scale(
                    g_mmWeightScales1[e * (tilingData.h / 32U) + k0 / 32U]);
                for (uint32_t n0 = 0U; n0 < tilingData.hiddenDim; n0 += 32U) {
                    DecSrc ws;
                    global_tensor<__fp8_e4m3, RowMajor<-1, -1>> gWS(
                        reinterpret_cast<__fp8_e4m3*>(
                            const_cast<uint8_t*>(
                                g_mmWeight1
                                    + e * tilingData.h * tilingData.hiddenDim
                                    + k0 * tilingData.hiddenDim + n0)),
                        32, tilingData.hiddenDim);
                    TLOAD(ws, gWS);
                    DecDst wd;
                    TCVT(wd, ws);
                    TMULS(wd, wd, s1);
                    global_tensor<__half, RowMajor<-1, -1>> gWD(
                        w1F16 + e * tilingData.h * tilingData.hiddenDim
                              + k0 * tilingData.hiddenDim + n0,
                        32, tilingData.hiddenDim);
                    TSTORE(gWD, wd);
                }
            }
            for (uint32_t k0 = 0U; k0 < tilingData.hiddenDim / 2U; k0 += 32U) {
                const float s2 = fp8_e8m0_scale(
                    g_mmWeightScales2[e * ((tilingData.hiddenDim / 2U) / 32U)
                                     + k0 / 32U]);
                for (uint32_t n0 = 0U; n0 < tilingData.h; n0 += 32U) {
                    DecSrc ws;
                    global_tensor<__fp8_e4m3, RowMajor<-1, -1>> gWS(
                        reinterpret_cast<__fp8_e4m3*>(
                            const_cast<uint8_t*>(
                                g_mmWeight2
                                    + e * (tilingData.hiddenDim / 2U) * tilingData.h
                                    + k0 * tilingData.h + n0)),
                        32, tilingData.h);
                    TLOAD(ws, gWS);
                    DecDst wd;
                    TCVT(wd, ws);
                    TMULS(wd, wd, s2);
                    global_tensor<__half, RowMajor<-1, -1>> gWD(
                        w2F16 + e * (tilingData.hiddenDim / 2U) * tilingData.h
                              + k0 * tilingData.h + n0,
                        32, tilingData.h);
                    TSTORE(gWD, wd);
                }
            }
        }
    }

    // ---- 阶段 2: 共享专家输入准备 (源 PrepareSharedExpertInput, sharedExpertNum==0 跳过) ----

    // ---- 阶段 3: MoE 专家流水 (ProcessMoeExpertStages → ProcessGmmPipeline) ----
    // tile 计算核心 (issue #180): GMM1/GMM2 = 逐 token TGEMV_MX/TGEMV_MX_ACC
    // (A=fp16 免 scale, B=E4M3 + E8M0 scale tile, CUBE fp32 累加), SwiGLU 与
    // Combine/Unpermute/x 量化 = VEC tile 链。tile 几何编译期固定 (K/N-tile=32,
    // MX scale 组宽), 运行期仅循环计数与 GM 寻址; FP8 权重 MX 原地消费。
    {
        using GemvVec = CubeTileM16<__half, 1, 32>;
        using GemvMtx = CubeTileN8<__half, 32, 32>;
        using GemvDst = CubeAccumulatorM16<float, 1, 32>;
        // VEC 链 tile: 物理列 64 使 fp32/fp16 两侧 TCVT 派生行数一致 (128B 下限)
        using ChainF = Tile<Location::Vec, float, 1, 64,
                            BLayout::RowMajor, 1, 32>;
        using ChainH = Tile<Location::Vec, __half, 1, 64,
                            BLayout::RowMajor, 1, 32>;

        uint8_t* const w1Base = g_mmWeight1;
        uint8_t* const w2Base = g_mmWeight2;
        for (uint32_t lc = 0U; lc < kCoresPerPE; ++lc) {
            const uint32_t coreIdx = tid * kCoresPerPE + lc;
            for (uint32_t i = 0; i < perCore; ++i) {
                const uint32_t token = coreIdx * perCore + i;
                if (token >= tilingData.bs) break;  // bs < 16 核时尾核空转

                // x 行 fp32 → fp16 (VEC TCVT; MX A 侧 fp16 免 scale)
                for (uint32_t c = 0U; c < tilingData.h; c += 32U) {
                    ChainF xf;
                    global_tensor<float, RowMajor<1, 32>> gX(
                        xIn + token * tilingData.h + c);
                    TLOAD(xf, gX);
                    ChainH xh;
                    TCVT(xh, xf);
                    global_tensor<__half, RowMajor<1, 32>> gXH(xf16Scratch + c);
                    TSTORE(gXH, xh);
                }

                for (uint32_t kk = 0U; kk < tilingData.topK; ++kk) {
                    const uint32_t slot = token * tilingData.topK + kk;
                    const uint32_t expert =
                        static_cast<uint32_t>(g_mmTopkIds[slot]);
                    const float weight = g_mmTopkWeights[slot];

                    // GMM1 (源 ProcessGmm1Wave): y1[n] = Σ_k x[k]·w1[e][k][n]
                    // (MX FP16 pair —— pto-spec: FP16 侧免 scale; k=0 块剥离
                    //  循环外, 消除运行期 TGEMV_MX/ACC 分支的 tile PHI)
                    for (uint32_t n0 = 0U; n0 < tilingData.hiddenDim;
                         n0 += 32U) {
                        GemvDst d;
                        {
                            GemvVec v;
                            global_tensor<__half, RowMajor<1, 32>> gV(
                                xf16Scratch);
                            TLOAD_CUBE(v, gV);
                            GemvMtx m;
                            global_tensor<__half, RowMajor<-1, -1>> gM(
                                w1F16 + expert * tilingData.h
                                            * tilingData.hiddenDim
                                      + n0,
                                32, tilingData.hiddenDim);
                            TLOAD_CUBE(m, gM);
                            TGEMV_MX(d, m, v, fixp::keep_acc());
                        }
                        for (uint32_t k0 = 32U; k0 < tilingData.h;
                             k0 += 32U) {
                            GemvVec v;
                            global_tensor<__half, RowMajor<1, 32>> gV(
                                xf16Scratch + k0);
                            TLOAD_CUBE(v, gV);
                            GemvMtx m;
                            global_tensor<__half, RowMajor<-1, -1>> gM(
                                w1F16 + expert * tilingData.h
                                            * tilingData.hiddenDim
                                      + k0 * tilingData.hiddenDim + n0,
                                32, tilingData.hiddenDim);
                            TLOAD_CUBE(m, gM);
                            TGEMV_MX_ACC(d, d, m, v, fixp::keep_acc());
                        }
                        global_tensor<float, RowMajor<1, 32>> gY1(
                            y1Scratch + n0);
                        TSTORE_CUBE(gY1, d);
                    }

                    // SwiGLU (源 Gmm1SwigluState):
                    // y2 = silu(y1[:hd/2]) * y1[hd/2:]  (VEC 链)
                    for (uint32_t c = 0U; c < tilingData.hiddenDim / 2U;
                         c += 32U) {
                        ChainF a, b;
                        global_tensor<float, RowMajor<1, 32>> gA(
                            y1Scratch + c);
                        TLOAD(a, gA);
                        global_tensor<float, RowMajor<1, 32>> gB(
                            y1Scratch + tilingData.hiddenDim / 2U + c);
                        TLOAD(b, gB);
                        ChainF neg, e, one, r, sig;
                        TMULS(neg, a, -1.0f);
                        TEXP(e, neg);
                        TADDS(one, e, 1.0f);
                        TRECIP(r, one);
                        TMUL(sig, a, r);
                        TMUL(sig, sig, b);
                        ChainH h;
                        TCVT(h, sig);
                        global_tensor<__half, RowMajor<1, 32>> gY2(
                            y2f16Scratch + c);
                        TSTORE(gY2, h);
                    }

                    // GMM2 (源 ProcessGmm2Wave): y3[n] = Σ_k y2[k]·w2[e][k][n]
                    // (MX FP16 pair, 同 GMM1; k=0 块剥离循环外)
                    for (uint32_t n0 = 0U; n0 < tilingData.h; n0 += 32U) {
                        GemvDst d;
                        {
                            GemvVec v;
                            global_tensor<__half, RowMajor<1, 32>> gV(
                                y2f16Scratch);
                            TLOAD_CUBE(v, gV);
                            GemvMtx m;
                            global_tensor<__half, RowMajor<-1, -1>> gM(
                                w2F16 + expert * (tilingData.hiddenDim / 2U)
                                            * tilingData.h
                                      + n0,
                                32, tilingData.h);
                            TLOAD_CUBE(m, gM);
                            TGEMV_MX(d, m, v, fixp::keep_acc());
                        }
                        for (uint32_t k0 = 32U; k0 < tilingData.hiddenDim / 2U;
                             k0 += 32U) {
                            GemvVec v;
                            global_tensor<__half, RowMajor<1, 32>> gV(
                                y2f16Scratch + k0);
                            TLOAD_CUBE(v, gV);
                            GemvMtx m;
                            global_tensor<__half, RowMajor<-1, -1>> gM(
                                w2F16 + expert * (tilingData.hiddenDim / 2U)
                                            * tilingData.h
                                      + k0 * tilingData.h + n0,
                                32, tilingData.h);
                            TLOAD_CUBE(m, gM);
                            TGEMV_MX_ACC(d, d, m, v, fixp::keep_acc());
                        }
                        global_tensor<float, RowMajor<1, 32>> gY3(
                            y3Scratch + n0);
                        TSTORE_CUBE(gY3, d);
                    }

                    // Combine (源 ProcessCombineExperts):
                    // yAcc = Σ_kk weight·y3 (VEC tile; topK==1 时与源/静态版
                    // 直接赋值等价)
                    float* const yBase = reinterpret_cast<float*>(
                        g_mmWorkspace + combineOffset)
                        + token * tilingData.h;
                    for (uint32_t c = 0U; c < tilingData.h; c += 32U) {
                        ChainF t;
                        global_tensor<float, RowMajor<1, 32>> gT(
                            y3Scratch + c);
                        TLOAD(t, gT);
                        TMULS(t, t, weight);
                        global_tensor<float, RowMajor<1, 32>> gAcc(
                            yBase + c);
                        if (kk == 0U) {
                            TSTORE(gAcc, t);
                        } else {
                            ChainF acc;
                            TLOAD(acc, gAcc);
                            TADD(acc, acc, t);
                            TSTORE(gAcc, acc);
                        }
                    }
                }
            }
        }
    }
    // UnpermuteTokens (9163): combine 缓冲按 token 序写回 y (VEC tile 拷贝;
    // token 分片与 combine 写方同 PE, 程序序保证可见, 免栅栏)
    {
        using ChainF = Tile<Location::Vec, float, 1, 64,
                            BLayout::RowMajor, 1, 32>;
        const float* combine =
            reinterpret_cast<const float*>(g_mmWorkspace + combineOffset);
        for (uint32_t lc = 0U; lc < kCoresPerPE; ++lc) {
            const uint32_t coreIdx = tid * kCoresPerPE + lc;
            for (uint32_t i = 0; i < perCore; ++i) {
                const uint32_t t = coreIdx * perCore + i;
                if (t >= tilingData.bs) break;
                for (uint32_t c = 0U; c < tilingData.h; c += 32U) {
                    ChainF v;
                    global_tensor<float, RowMajor<1, 32>> gSrc(
                        const_cast<float*>(combine + t * tilingData.h + c));
                    TLOAD(v, gSrc);
                    global_tensor<float, RowMajor<1, 32>> gDst(
                        yOut + t * tilingData.h + c);
                    TSTORE(gDst, v);
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

    // ---- 跨 PE 交接点: 末端汇合栅栏 (fp32 解码栅栏已随 MX 原地消费删除) ----
    // 所有 PE 的 yOut / tokOut 写入完成后才返回 (PE0 随后的验证依赖全量输出)
    mtBarrierDyn(1U);
#endif
}

}  // namespace mega_moe
