#pragma once

/**
 * @file mega_moe_sim_mt_dyn.hpp
 * @brief MegaMoe A8W8 — 真 4PE (multi-thread SPMD) 分片版, runtime DYNAMIC
 *        SHAPE kernel 入口。
 *
 * 与 mega_moe_sim_mt.hpp (静态 shape 4PE 版) 的关系:
 *   - 算子语义与阶段划分完全一致 ([A1/B8/B9/A6 对齐 2026-10-13] 统计清零 →
 *     量化解码(共享) → mask/dispatch 计划 → 量化搬运(xDisp/scale/metaInfo,
 *     ×2) → expert wave GMM1→SwiGLU→GMM2→combine 窗口(未加权, ×2) →
 *     Unpermute 加权反排列 → 统计导出; 相位: ph+2=dispatch/解码可见,
 *     ph+3=combine 可见, ph+4=末端汇合);
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
// [2026-10-11 试验注] 曾在本屏障轮询间加 16 步寄存器乘法延迟链 (缓解
// PE0 独占段自旋抢占, gtv/old_mt 同款): 与幂等双遍 (guardPass=2) 叠加时
// gfsim fourpe 在 cfgA kernel 段 (C:1.13M) 触发 RAS "spec table valid
// should be true at commit" 断言 —— 时序偏移落入模型恢复路径缺陷窗口
// (单遍版反而完整跑完, 佐证为时序竞态而非语义错误), 故 kernel 内屏障
// 回退原版紧凑自旋。驱动末端 mtRendezvousDynFin 的同款降频链亦实测
// 暴露 "kernel 末端 TSTORE→GM 落盘 vs PE0 验证读" 竞态 (周期虽 −13%
// 但 finisher val=0x0001 fail, 基线靠 worker 紧凑自旋干扰延迟侥幸
// 通过), 一并回退。

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
    // [A1 2026-10-13] dispatch 计划标量栈数组上界 (kMoeEprMax / 64)
    if (expertPerRank_r > kMoeEprMax) return;
    if (bs_r * topK_r > 64U) return;

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
    tilingData.maxOutputSize = bs_r * topK_r;   // [C11] A5 语义: bs*topK (运行期)
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

    // workspace 布局 (A1 dispatch 通路版, mega_moe_sim.hpp 四d MoeWsLayout)
    const MoeWsLayout WS = moe_ws_layout(tilingData.bs, tilingData.h,
                                         tilingData.hiddenDim,
                                         tilingData.moeExpertPerRank,
                                         tilingData.topK);
    const uint32_t slots = tilingData.bs * tilingData.topK;
    // M2: 解码按 tile 扁平索引分片 (每 tile 仅 1 PE 读写), 目标缓冲共享
    // (PE0 区); 每 PE 私有 wave 组 scratch (y1[G][hd] + y2f16[G][hd/2] +
    // y3[G][h], G = kMoeWaveGroupMax —— A5 wave 组中间量)
    __half* const w1F16 = reinterpret_cast<__half*>(
        g_mmWorkspace + WS.scratch);
    __half* const w2F16 = w1F16
        + tilingData.moeExpertPerRank * tilingData.h * tilingData.hiddenDim;
    const uint32_t weightsBytes = mx_weight_copy_bytes(
        tilingData.moeExpertPerRank, tilingData.h, tilingData.hiddenDim);
    const uint32_t groupBytes = mx_wave_group_bytes(
        tilingData.h, tilingData.hiddenDim);
    uint8_t* const groupBase = g_mmWorkspace + WS.scratch + weightsBytes
                              + tid * groupBytes;
    float* const y1Scratch = reinterpret_cast<float*>(groupBase);
    __half* const y2f16Scratch = reinterpret_cast<__half*>(
        reinterpret_cast<uint8_t*>(y1Scratch)
        + kMoeWaveGroupMax * tilingData.hiddenDim * 4U);
    float* const y3Scratch = reinterpret_cast<float*>(
        reinterpret_cast<uint8_t*>(y2f16Scratch)
        + kMoeWaveGroupMax * (tilingData.hiddenDim / 2U) * 2U);

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
    // [A1/A2/A3/B8/B9/A6 对齐 2026-10-13] 与静态 mt 版同阶段集 (dispatch
    // 通路 + expert wave 批处理), 运行期 shape; 相位: ph+2 = dispatch/解码
    // 共享缓冲可见, ph+3 = combine 窗口可见, ph+4 = 末端汇合。
    // ---- 阶段 1: 输入准备 (各 PE 写域不相交) ----
    // SendAndQuantBuffInit: 统计槽清零。volatile 标量逐元素 ([C9]: 常量
    // TEXPANDS+TSTORE 被折叠为零寄存器别名绑定, cfgB 重复调用永不完成)。
    {
        int32_t* stats = reinterpret_cast<int32_t*>(g_mmWorkspace + WS.stats);
        volatile int32_t* vs = stats + tid * kCoresPerPE * tilingData.moeExpertPerRank;
        const uint32_t nStats = kCoresPerPE * tilingData.moeExpertPerRank;
        for (uint32_t i = 0; i < nStats; ++i) {
            vs[i] = 0;
        }
    }
    // GatherAndSendExpertMasks: 自回环本地 mask 表, PE0 独占全量构建
    // (单写者; bs ≤ 36, tile 链无收益, 且该 TLOAD 为模型首 tile 派发缺陷
    // 的受害者, 见 TILE_GFSIM_COMPLETION_REPORT.md rev4)。
    // [A2 模型注 2026-10-13] mask 为 A5 结构载体 (写后不读): MSCARRIER/
    // tile 写 → 标量读存在 TLSU tile-store→GM 缺口 (D15 同族), dispatch
    // 计数从 topkIds 标量统计 (moe_dispatch_mt cntLocal 同款路径)。
    {
        uint8_t* mask = g_mmWorkspace + WS.mask;
        if (tid == 0U) {
            for (uint32_t s = 0; s < slots; ++s) {
                const int32_t expert = g_mmTopkIds[s];
                if (expert >= 0 &&
                    static_cast<uint32_t>(expert) < tilingData.moeExpertPerRank) {
                    mask[static_cast<uint32_t>(expert) * tilingData.bs
                         + s / tilingData.topK] = 1U;
                }
            }
        }
    }

    // ---- 阶段 1.5a (issue #180): FP8 权重 tile 化解码 (M2: 分片 + 共享) ----
    // w1F16[e][k][n] = fp16(E4M3) * 2^(w1Scale[e][k/32]-127), 逐 [32,32] tile:
    // TLOAD(E4M3) → TCVT(fp16) → TMULS(组 scale) → TSTORE
    // [2026-10-13 注] 解码保持单遍 (静态 mt 版为 ×2 幂等双遍; 本版解码目标
    // 为 M2 共享缓冲, 双遍使共享行 tile-store WAW 违例 3× 放大, 实测将
    // cfgB 段推入 M7 类 tile-store 排序门停摆窗口 @cycle≈1.97M; 单遍为
    // v3/v4 基线已证形态, 污染自愈由 kernel 内 dispatch/wave 的 ×2 覆写
    // 与轮间 L2 刷洗承载)。
    for (uint32_t decodePass = 0U; decodePass < 1U; ++decodePass) {
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

    // ---- 阶段 1.5b: dispatch 计划 (幂等, 各 PE 同值; 无跨 PE 依赖) ----
    // ComputeExpertTokenCountAndNotify (4540): expertCnt + starts (cumsumInfo)
    // + dstPos (稳定序) — 全部从 topkIds 标量计算 (驱动数据, 屏障 8c+1 覆盖)。
    // [mt 版同款注] 独立 -1 复位段省略 (跨 PE tile/scalar store 无全局序);
    // -1 复位语义由 plan 的无效槽 -1 写承载。
    {
        uint32_t expertCnt[kMoeEprMax];
        uint32_t startsArr[kMoeEprMax];
        uint32_t dstPosArr[64];
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
        // workspace 镜像 (A5 expertRevTokenNums/cumsumInfo/dstPosition 载体;
        // GMM 波读 starts): 幂等同值写
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
    // InitExpertTokenCountExportBuffers 核内累计 (A6: dispatch 期逐核计数,
    // 本 PE 4 伪核)
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
    // 量化 token+scale 搬运 + metaInfo — slot 分片 ×2 幂等双遍 (静态 mt 版
    // 同款: xDisp 量化 TSTORE 的 TLSU 缺口污染行以同值覆写自愈)
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
        // pad 元数据行 (幂等): {TOKEN=bs, TOPK=topK-1} → combine 写 dummy
        // 尾行 (永不读取)
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

    // ---- dispatch 通路 + 共享解码权重全量可见 (A5 flagDispatchToGmm1;
    //      GMM 波跨 PE 读 xDisp/metaInfo/starts/w1F16/w2F16) ----
    mtBarrierDyn(ph + 2U);

    // ---- 阶段 3: MoE 专家流水 (ProcessMoeExpertStages asc:10222) ----
    // B8 expert-major wave ×2 幂等双遍 ([2026-10-11 试验注 #2] 单遍下末段
    // combine 行被 gfsim TLSU tile-store→GM 缺口污染, 双遍以正确值覆盖);
    // PE 行片 = slots ceil 4 分片; MxTileScratch 以共享权重 + 本 PE 组
    // scratch 构造 (M2 契约)
    {
        const __half* xDisp = reinterpret_cast<const __half*>(
            g_mmWorkspace + WS.xDisp);
        const int32_t* meta = reinterpret_cast<const int32_t*>(
            g_mmWorkspace + WS.metaInfo);
        float* const combineWin = reinterpret_cast<float*>(
            g_mmWorkspace + WS.combine);
        const int32_t* stWs = reinterpret_cast<const int32_t*>(
            g_mmWorkspace + WS.starts);
        MxTileScratch mx;
        mx.w1F16 = w1F16;            // M2 共享权重副本 (ph+2 栅栏后全 PE 可见)
        mx.w2F16 = w2F16;
        mx.y1 = y1Scratch;           // 每 PE wave 组 scratch
        mx.y2f16 = y2f16Scratch;
        mx.y3 = y3Scratch;
        const uint32_t rowsPerPE =
            (slots + kMtThreadsPerBlockDyn - 1U) / kMtThreadsPerBlockDyn;
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
                // [2026-10-14 注] 本入口维持固定 G=4 分组, 不用 mx_expert_wave_run
                // 自适应分派 (sim/mt 入口已用): mt_dyn 实测自适应分派的代码
                // 布局变化重掷时序骰子, cfgA kernel 末端 (cycle≈79.3K) 落入
                // M7 类 tile-store 排序门停摆窗口 (TPC 参考队列满 256,
                // thread0 末端冲刷 STD 块 allocated 9999 cyc 不完成, 分辨
                // 实验 A 实证回退后通过) —— 与解码单遍 (mt 版为双遍) 同类
                // 的入口形态差异, cfgB pad 槽削减 (~3K cycle, 占 Total
                // 0.15%) 不值得冒险。数值上两形态等价 (cfgA 无 pad 槽,
                // tile op 序列相同)。
                for (uint32_t r0 = eb; r0 < ee; r0 += kMoeWaveGroupMax) {
                    mx_expert_wave_group<kMoeWaveGroupMax>(
                        mx, xDisp, r0, ee, slots,
                        tilingData.h, tilingData.hiddenDim,
                        e, meta, combineWin,
                        tilingData.topK);
                }
            }
        }
    }

    // ---- combine 窗口全量可见 (A5 CrossRankSyncInWorldSize / WaitCombine,
    //      asc:10416-10421) ----
    mtBarrierDyn(ph + 3U);

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

    // [2026-10-12 修复] 写方 yOut 冲刷 (golden 提速后新增的竞态窗口):
    // 每 PE 在末端汇合屏障 (ph+4) 发布前, 对自己写入的 yOut 行做 volatile
    // 读回 (64B 行粒度), 借同 PE load/store 块序强制本 PE 在飞 tile-store
    // 先于屏障 flag 落盘 (写方定向驱逐协议同族); yOut 数值不变 (纯读回)。
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

    // ---- 阶段 4: 统计导出 (A6: dispatch 期核内累计求和; A5
    //      ExportExpertTokenCounts(FromWorkspace) 4599/4640) — PE0 独占 ----
    {
        const int32_t* stats = reinterpret_cast<const int32_t*>(
            g_mmWorkspace + WS.stats);
        volatile int64_t* tokExport = tokOut;
        if (tid == 0U) {
            for (uint32_t e = 0U; e < tilingData.moeExpertPerRank; ++e) {
                uint32_t cnt = 0U;
                for (uint32_t c = 0U; c < kBlockAivNum; ++c) {
                    cnt += static_cast<uint32_t>(
                        stats[c * tilingData.moeExpertPerRank + e]);
                }
                tokExport[e] = static_cast<int64_t>(cnt);
            }
        }
    }

    // ---- 跨 PE 交接点: 末端汇合栅栏 (全部 yOut/tokOut 写入完成后返回) ----
    mtBarrierDyn(ph + 4U);
#endif
}

}  // namespace mega_moe

