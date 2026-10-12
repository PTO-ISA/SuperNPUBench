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
 *
 * [2026-10-04 修复注] 本文件 + 驱动 (test/solution/mega_moe/src/
 * mega_moe_sim_mt_dyn.cpp) 的联合修复集, 详见同目录
 * mega_moe_sim_mt_dyn_gfsim_fix_report.md:
 *   1. 阶段 3 改用共享辅助 mx_quantize_row + mx_token_compute_pre
 *      (runtime-h/hd 签名) + mx_copy_row_tile: x 预量化前置 (消除相邻块
 *      量化→GEMM GM 往返 nuke 触发对) + 幂等双遍执行 (自愈模型 nuke 恢复
 *      丢 tile 状态的默认数据污染) + 全宽 [1×32] 静态块末端 store (规避
 *      [C6] valid 泄漏) —— 修复后实测 nuke=0;
 *   2. 阶段 4 tokOut 导出改标量等值计数 (gfsim TLSU tile-store→GM 缺口
 *      使 mm_count_eq_i32 恒返 0)、stats 改两遍式;
 *   3. 驱动层: cfgB 验证后新增写方定向驱逐验证汇合屏障 (修复 worker
 *      提前 return 的末端 exit lockstep 搁浅 @ cycle 3,638,308 与安静
 *      窗口陈旧 flag 活锁) + golden 解码备忘录化 (两轮验证 ~5× 缩短) +
 *      SPMD finisher 全线程写。
 * 修复后 gfsim `--conf fourpe`: linx_test_finisher val=0x5555 PASS
 * (Total Cycles ≈ 3.84M, 确定性), gfrun 4 线程 R2=0; `--conf mt` 同静态
 * 版为模型/配置能力限制 (单引擎停摆), 时序仿真须用 fourpe。
 */

#include "solution/mega_moe/mega_moe_sim.hpp"

namespace mega_moe {

// ============================================================================
// multi-thread 基础设施 (独立命名, 与 mega_moe_sim_mt.hpp 约定相同但可
// 与静态版头文件同 TU 共存)
// ============================================================================
constexpr uint32_t kMtThreadsPerBlockDyn = 4U;

// 屏障 = 单行 flag 自旋, 跳过自己的槽 (自己刚写过, 程序序保证 >= phase):
// 消除自旋窗口内 "同地址 store→load" 对 (nuke/存储序重放的主要触发源 ——
// mega 含 CUBE ACC 链, 重放会二次 SetACC 撞 BROB.cpp:1154 断言, gfsim
// 实测; gtv 无 CUBE 故其定向集驱逐版安全, mega 不用驱逐读)。
// 跨 PE 可见性: mega 各相位偏斜中等 (mt 静态版同型 plain 自旋 gfsim
// PASS 2,010,999 佐证), 若后续实测活锁再引入驱逐读。
alignas(64) static volatile uint32_t sMtPhaseDoneDyn[kMtThreadsPerBlockDyn];

static inline void mtCompilerBarrierDyn()
{
    __asm__ volatile("" : : : "memory");
}

static inline void mtBarrierDyn(uint32_t phase)
{
    mtCompilerBarrierDyn();
    const uint32_t tid = get_thread_idx();
    sMtPhaseDoneDyn[tid] = phase;
    mtCompilerBarrierDyn();
    for (uint32_t t = 0U; t < kMtThreadsPerBlockDyn; ++t) {
        if (t == tid) {
            continue;   // 自己的槽: 本函数刚写入 phase, 无需轮询
        }
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

    // 单调相位编号 (gtv/moe_dispatch mt_dyn 同款修复): 原 driver 在两次 cfg
    // 之间清零 sMtPhaseDoneDyn —— 无同步跨 PE 写, 时序模型下清零晚于他 PE
    // 置位则抹 flag → 互等活锁; 且固定相位跨调用陈旧直通。改为 per-PE 私用
    // 调用计数 sInvCntDyn (零跨 PE 写), 相位 = inv*8 + k 单调递增, driver
    // 复位删除。相位分配 (每轮 inv=c 占 8 个): driver gen 汇合 = 8c+1
    // (genInputs 分片写后), kernel 末端汇合 = 8c+4, cfgA 验证汇合 = 5。
    // 契约同值判定保证各 PE 的 inv 序列一致。
    alignas(64) static uint32_t sInvCntDyn[kMtThreadsPerBlockDyn];  // bss 零初始化
    const uint32_t inv = sInvCntDyn[tid];
    sInvCntDyn[tid] = inv + 1U;
    const uint32_t ph = inv * 8U;

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
    // M2: 解码按 tile 扁平索引分片 (每 tile 仅 1 PE 读写), 目标缓冲改共享
    // (PE0 区); per-PE scratch 偏移不变, GMM 读共享副本 (decode 栅栏后可见)。
    __half* const w1F16 = reinterpret_cast<__half*>(
        g_mmWorkspace + yScratchOffset);
    __half* const w2F16 = w1F16
        + tilingData.moeExpertPerRank * tilingData.h * tilingData.hiddenDim;
    __half* const xf16Scratch = reinterpret_cast<__half*>(
        yScratchBase
        + (tilingData.moeExpertPerRank * tilingData.h * tilingData.hiddenDim
         + tilingData.moeExpertPerRank * (tilingData.hiddenDim / 2U)
               * tilingData.h) * 2U);
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
    mtBarrierDyn(ph + 4U);
#else
    // ============ 完整真机流水 — 真 4PE 分片执行 (运行期 dims) ============
    // ---- 阶段 1: 输入准备 (各 PE 写域不相交) ----
    // SendAndQuantBuffInit: 统计槽清零。volatile 标量逐元素 ([C9]: 常量
    // TEXPANDS+TSTORE 被折叠为零寄存器别名绑定, cfgB 重复调用永不完成)。
    {
        int32_t* stats = reinterpret_cast<int32_t*>(g_mmWorkspace + statsOffset);
        volatile int32_t* vs = stats + tid * kCoresPerPE * tilingData.moeExpertPerRank;
        const uint32_t nStats = kCoresPerPE * tilingData.moeExpertPerRank;
        for (uint32_t i = 0; i < nStats; ++i) {
            vs[i] = 0;
        }
    }
    // QuantizeLocalTokens (3754): MX 路径原地消费 E4M3+E8M0 权重 —— 不再有
    // fp32 解码 workspace 与对应的跨 PE 栅栏 (issue #180: 解码阶段整体删除)
    // GatherAndSendExpertMasks: 自回环本地 mask 表, PE0 独占全量构建
    // (单写者; bs ≤ 36, tile 链无收益, 且该 TLOAD 为模型首 tile 派发缺陷
    // 的受害者, 见 TILE_GFSIM_COMPLETION_REPORT.md rev4)。可见性由 ph+2
    // 屏障保证 (topK!=1 时 helper 标量兜底不变)。
    {
        uint8_t* mask = g_mmWorkspace + maskOffset;
        if (tid == 0U) {
            for (uint32_t s = 0; s < tilingData.bs; ++s) {
                const int32_t expert = g_mmTopkIds[s];
                mask[static_cast<uint32_t>(expert) * tilingData.bs + s] = 1U;
            }
        }
        mtBarrierDyn(ph + 2U);
    }
    // ResetDispatchWorkspace = DispatchBuffInit: dispatch 表填 -1。
    // volatile 标量逐元素填充 ([C9] 同族: 常量 TSTORE cfgB 冻结)。nTok ≤ 8。
    {
        float* dispatch = reinterpret_cast<float*>(g_mmWorkspace + dispatchOffset);
        const uint32_t tokBegin = tid * kCoresPerPE * perCore;
        const uint32_t nTok = (tokBegin < tilingData.bs)
                            ? ((tilingData.bs - tokBegin < kCoresPerPE * perCore)
                               ? (tilingData.bs - tokBegin)
                               : (kCoresPerPE * perCore))
                            : 0U;
        for (uint32_t e = 0; e < tilingData.moeExpertPerRank; ++e) {
            volatile float* v = dispatch + e * tilingData.bs + tokBegin;
            for (uint32_t i = 0; i < nTok; ++i) {
                v[i] = -1.0f;
            }
        }
    }

    // ---- 阶段 1.5 (issue #180): FP8 权重 tile 化解码 (M2: 分片 + 共享) ----
    // w1F16[e][k][n] = fp16(E4M3) * 2^(w1Scale[e][k/32]-127), 逐 [32,32] tile:
    // TLOAD(E4M3) → TCVT(fp16) → TMULS(组 scale) → TSTORE
    {
        using DecSrc = Tile<Location::Vec, __fp8_e4m3, 32, 32>;
        using DecDst = Tile<Location::Vec, __half, 32, 32>;
        uint32_t flatTile = 0U;
        for (uint32_t e = 0U; e < tilingData.moeExpertPerRank; ++e) {
            for (uint32_t k0 = 0U; k0 < tilingData.h; k0 += 32U) {
                const float s1 = fp8_e8m0_scale(
                    g_mmWeightScales1[e * (tilingData.h / 32U) + k0 / 32U]);
                for (uint32_t n0 = 0U; n0 < tilingData.hiddenDim;
                     n0 += 32U, ++flatTile) {
                    if ((flatTile % kMtThreadsPerBlockDyn) != tid) continue;
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
                for (uint32_t n0 = 0U; n0 < tilingData.h;
                     n0 += 32U, ++flatTile) {
                    if ((flatTile % kMtThreadsPerBlockDyn) != tid) continue;
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

    // M2: 分片解码的共享权重缓冲须在 GMM 读取前对全 PE 可见
    // (相位改 3: routing 屏障占用 ph+2, 保持 per-PE 单调)
    mtBarrierDyn(ph + 3U);

    // ---- 阶段 2: 共享专家输入准备 (源 PrepareSharedExpertInput, sharedExpertNum==0 跳过) ----

    // ---- 阶段 3: MoE 专家流水 (ProcessMoeExpertStages → ProcessGmmPipeline) ----
    // [2026-10-04 修复] 与静态 mt 版同款修复 (mega_moe_sim_mt_gfsim_fix_report.md
    // §4-5/6/7), 且直接复用其运行期 shape 的共享辅助 (mx_quantize_row /
    // mx_token_compute_pre / mx_copy_row_tile 均为 runtime-h/hd 签名):
    //   1. x 预量化前置为独立一遍 —— 消除原 per-token 内联 "量化 TSTORE
    //      (xf16) → GEMM1 TLOAD(xf16)" 相邻块 GM 往返对 (模型依赖预测器首
    //      碰撞 nuke 的触发源);
    //   2. 幂等双遍执行 (guardPass) —— 自愈模型 nuke 恢复丢 tile 状态导致
    //      的末段 combine 行默认数据 2.0f 污染 (第二遍预测器已学习、无
    //      nuke, 以正确值覆盖);
    //   3. Combine/Unpermute 由本处内联的 [1×64,v1×32] 掩码链改为共享辅助
    //      的全宽 [1×32] 静态块 (无 ValidRow 掩码不泄漏, [C6] 同类规避);
    //      MxTileScratch 以本 kernel 的共享权重 + 每 PE scratch 指针构造
    //      (M2 分片解码的共享 w1F16/w2F16 与辅助函数的指针语义兼容)。
    // 预量化槽位于 workspace 4 PE scratch 槽之后的余量区 (driver 的
    // kWorkspaceBytes slack 已扩至 16KB), 每 PE 连续 myTok*h*2 字节。
    {
        const uint32_t myTokMax = kCoresPerPE * perCore;
        __half* const xf16Slots = reinterpret_cast<__half*>(
            g_mmWorkspace + yScratchOffset
            + kMtThreadsPerBlockDyn * perPeBytes
            + tid * myTokMax * tilingData.h * 2U);
        MxTileScratch mx;
        mx.w1F16 = w1F16;            // M2 共享权重副本 (ph+3 栅栏后全 PE 可见)
        mx.w2F16 = w2F16;
        mx.xf16 = xf16Scratch;       // _pre 变体不使用 (A 侧读预量化槽)
        mx.y1 = y1Scratch;
        mx.y2f16 = y2f16Scratch;
        mx.y3 = y3Scratch;

        for (uint32_t guardPass = 0U; guardPass < 2U; ++guardPass) {
            (void)guardPass;
            // 预量化: 本 PE 全部 token 的 x 行 (量化 store 与 GEMM load 隔整批)
            {
                uint32_t nq = 0U;
                for (uint32_t lc = 0U; lc < kCoresPerPE; ++lc) {
                    const uint32_t coreIdx = tid * kCoresPerPE + lc;
                    for (uint32_t i = 0; i < perCore; ++i) {
                        const uint32_t token = coreIdx * perCore + i;
                        if (token >= tilingData.bs) break;
                        mx_quantize_row(xIn + token * tilingData.h,
                                        xf16Slots + nq * tilingData.h,
                                        tilingData.h);
                        ++nq;
                    }
                }
            }
            // 主流水: GMM1(A=预量化槽) → SwiGLU → GMM2 → Combine(全宽块)
            float* const combineBase = reinterpret_cast<float*>(
                g_mmWorkspace + combineOffset);
            uint32_t iq = 0U;
            for (uint32_t lc = 0U; lc < kCoresPerPE; ++lc) {
                const uint32_t coreIdx = tid * kCoresPerPE + lc;
                for (uint32_t i = 0; i < perCore; ++i) {
                    const uint32_t token = coreIdx * perCore + i;
                    if (token >= tilingData.bs) break;
                    for (uint32_t kk = 0U; kk < tilingData.topK; ++kk) {
                        const uint32_t slot = token * tilingData.topK + kk;
                        const uint32_t expert =
                            static_cast<uint32_t>(g_mmTopkIds[slot]);
                        const float weight = g_mmTopkWeights[slot];
                        mx_token_compute_pre(mx,
                                             xf16Slots + iq * tilingData.h,
                                             tilingData.h,
                                             tilingData.hiddenDim,
                                             tilingData.moeExpertPerRank,
                                             expert, weight, kk,
                                             combineBase
                                                 + token * tilingData.h);
                    }
                    ++iq;
                }
            }
        }
    }
    // UnpermuteTokens (9163): combine 缓冲按 token 序写回 y
    // [2026-10-04 修复] 改用共享 mx_copy_row_tile (全宽 [1×32] 静态块,
    // 无 valid 泄漏; 原 [1×64,v1×32] 掩码链泄漏会污染 y 相邻 token 行)
    {
        const float* combine =
            reinterpret_cast<const float*>(g_mmWorkspace + combineOffset);
        for (uint32_t lc = 0U; lc < kCoresPerPE; ++lc) {
            const uint32_t coreIdx = tid * kCoresPerPE + lc;
            for (uint32_t i = 0; i < perCore; ++i) {
                const uint32_t t = coreIdx * perCore + i;
                if (t >= tilingData.bs) break;
                mx_copy_row_tile(combine + t * tilingData.h,
                                 yOut + t * tilingData.h, tilingData.h);
            }
        }
    }

    // ---- 阶段 4: 跨 rank 同步 (自回环本地退化) 与统计导出 ----
    // stats 按伪核归属分片 (每 PE 只写自己 4 个伪核的槽, 写不相交)
    // [2026-10-04 优化] 两遍式 (先集中 load 本 PE 全部 token id, 再计算并
    // 写 stats): 消除原逐 (core,e) 交错 "load g_mmTopkIds / store stats"
    // 相邻 ld/st 块对 (模型依赖预测器保守 nuke 的触发面, 同静态 mt 版)。
    {
        int32_t* stats = reinterpret_cast<int32_t*>(g_mmWorkspace + statsOffset);
        // 上界: kCoresPerPE × perCore × topK (本驱动 shape 集内 ≤ 8; 64 留裕)
        uint32_t myIds[64];
        uint32_t myCnt[kCoresPerPE];
        uint32_t loaded = 0U;
        for (uint32_t lc = 0U; lc < kCoresPerPE; ++lc) {
            const uint32_t core = tid * kCoresPerPE + lc;
            uint32_t cnt = 0U;
            for (uint32_t i = 0; i < perCore; ++i) {
                const uint32_t token = core * perCore + i;
                if (token >= tilingData.bs) break;
                for (uint32_t kk = 0U; kk < tilingData.topK; ++kk) {
                    myIds[loaded++] =
                        static_cast<uint32_t>(
                            g_mmTopkIds[token * tilingData.topK + kk]);
                }
                ++cnt;
            }
            myCnt[lc] = cnt;
        }
        for (uint32_t lc = 0U; lc < kCoresPerPE; ++lc) {
            const uint32_t core = tid * kCoresPerPE + lc;
            for (uint32_t e = 0; e < tilingData.moeExpertPerRank; ++e) {
                uint32_t cnt = 0U;
                uint32_t idx = 0U;
                for (uint32_t p = 0U; p < lc; ++p) idx += myCnt[p] * tilingData.topK;
                for (uint32_t i = 0U; i < myCnt[lc]; ++i) {
                    for (uint32_t kk = 0U; kk < tilingData.topK; ++kk) {
                        if (myIds[idx + i * tilingData.topK + kk] == e) {
                            ++cnt;
                        }
                    }
                }
                stats[core * tilingData.moeExpertPerRank + e] =
                    static_cast<int32_t>(cnt);
            }
        }
        // tokOut: PE0 独占导出 — 标量等值计数
        // [2026-10-04 修复] 原 mm_count_eq_i32 tile 计数链 (TSTORE→scratch
        // 标量读回) 在 gfsim TLSU tile-store→GM 缺口下恒返 0 → tokOut=(0,0)
        // → test-finisher val=0x0001 (静态 mt 版同款修复, 探针证据见
        // mega_moe_sim_gfsim_fix_report.md §5)。计数长度 bs*topK(≤18) 个
        // 元素, 标量循环三端 (gfrun/gfsim/真机) 语义一致。
        // volatile 规避相邻 i64 store 合并为 16B tile store
        if (tid == 0U) {
            volatile int64_t* tokExport = tokOut;
            for (uint32_t e = 0; e < tilingData.moeExpertPerRank; ++e) {
                uint32_t cnt = 0U;
                for (uint32_t s = 0U;
                     s < tilingData.bs * tilingData.topK; ++s) {
                    if (static_cast<uint32_t>(g_mmTopkIds[s]) == e) ++cnt;
                }
                tokExport[e] = static_cast<int64_t>(cnt);
            }
        }
    }

    // ---- 跨 PE 交接点: 末端汇合栅栏 (fp32 解码栅栏已随 MX 原地消费删除) ----
    // 所有 PE 的 yOut / tokOut 写入完成后才返回 (PE0 随后的验证依赖全量输出)
    mtBarrierDyn(ph + 4U);
#endif
}

}  // namespace mega_moe
