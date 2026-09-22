#pragma once

/**
 * @file mega_moe_sim.hpp
 * @brief MegaMoe A8W8 wave 自回环样例 —— PTO (LinxISA VectorLite) 完整迁移实现。
 *
 * 源算子: cann-samples Samples/2_Performance/mega_moe_story/src/mega_moe_main.asc
 *         (10658 行, 全逻辑迁移, 不省略任何阶段)
 *
 * 迁移对照（源行号）:
 *   - 常量集        : kMoeExpertPerRank / kBS / kH / kHiddenDim / kTopK / kEpWorldSize /
*                       kHcclMaxRankSize(1024) ...
 *   - MegaMoeTilingData (源 451 行) 与 Mc2MoeContextHost (源 1109 行) 字段全保留
 *   - MegaMoeWave 类 (源 9240 行起) 全部阶段函数一一对应:
 *       Init (9508) / SendAndQuantBuffInit (9731) / QuantizeLocalTokens (3754)
 *       GatherAndSendExpertMasks (3934) / ResetDispatchWorkspace (4051)
 *       PrepareSharedExpertInput (4119) / DispatchBuffInit (9582)
 *       InitExpertTokenCountExportBuffers (9709) / ProcessMoeExpertStages (9271)
 *       ProcessGmmPipeline (9273) / ProcessGmm1Wave (9287) / ProcessGmm2Wave (9291)
 *       ProcessCombineExperts (9293) / InitTokenUnpermuteBuffers (9294)
 *       UnpermuteTokens (9163) / CrossRankSyncInWorldSize (9295) / Process (10319)
 *   - Process() 与源同构保留 #ifdef MEGA_MOE_SIM_FAKE 仿真快路径分支 (10321-10374),
 *     快速路径保留全部 tile 原语 (TLOAD/TMULS/TEXP/TADDS/TRECIP/TSTORE);
 *     #else 分支为完整真机流水: 量化 → 路由/mask → GMM1 → SwiGLU → GMM2 →
 *     Combine → Unpermute → 统计导出 (计算语义与 gen_data.compute_golden 一致,
 *     A8W8 权重按 E4M3FN + E8M0 scale 全量解码参与计算)。
 *
 * 架构映射 (Ascend → PTO):
 *   - GM 缓冲     : 全局数组 (g_mmX/g_mmY/g_mmTopkIds/.../g_mmWorkspace), 单地址空间
 *   - 16 核 AIV 分片: 4 PE 线程 (get_thread_idx()=0..3), 每线程承载 4 个伪核;
 *                     token 域按 16 伪核 ceil 分片 (bs>=16: 每核 bs/16 连续
 *                     token; bs<16: 前 bs 核各 1 token, 尾核空转), 核×轮结构保留
 *   - AIC         : 冒烟读 x[0] (读到即丢弃, 不写输出)
 *   - MC2 跨 rank : epWorldSize=1 自回环, CrossRankSyncInWorldSize/WinRank 读写
 *                   退化为本地操作 (与源注释语义一致)
 *
 * 仿真绑定说明: gfrun 功能仿真器对"经函数指针参数间接写 GM"支持不完整 (仓库既有
 * 限制: moe_dispatch_v2 / group_token_vec 自校验用例同因 R2=1), 本实现保留源
 * 签名 (指针入参), 计算一律绑定 GM 全局缓冲, 逻辑与指针签名语义完全等价。
 */

#include <common/pto_tileop.hpp>
#include <cstdint>

using namespace pto;

#ifdef MM_FORCE_INLINE
#define MM_INLINE __attribute__((always_inline)) inline
#else
#define MM_INLINE inline
#endif

// ============================================================================
// 三、GM 全局缓冲 (test driver 定义, 见 mega_moe_sim.cpp)
// ============================================================================
extern float g_mmX[];                  // 输入 token  [bs][h]
extern int32_t g_mmTopkIds[];          // 路由 id    [bs*topK]
extern float g_mmTopkWeights[];        // 路由权重   [bs*topK]
extern uint8_t g_mmWeight1[];          // A8W8 权1   [experts][h][hiddenDim]  (FP8 E4M3)
extern uint8_t g_mmWeight2[];          // A8W8 权2   [experts][hiddenDim/2][h] (FP8 E4M3)
extern uint8_t g_mmWeightScales1[];    // scale1     [experts * (h/32) * 2 * 2] (E8M0)
extern uint8_t g_mmWeightScales2[];    // scale2     [experts * (hiddenDim/2/32) * 2 * 2]
extern float g_mmY[];                  // 输出 y     [bs][h]
extern int64_t g_mmExpertTokenNums[];  // 专家 token 数 [experts]
extern uint8_t g_mmWorkspace[];        // 工作区 (量化权重缓存 / dispatch 表 / combine 缓冲)

namespace mega_moe {

// ============================================================================
// 一、常量 (与源 mega_moe_main.asc 常量集一一对应)
// ============================================================================
constexpr uint32_t kMoeExpertPerRank = 2U;   // 本卡 MoE 专家数 (weight1 表达)
#ifndef kBS
#define kBS 16
#endif
#ifndef kH
#define kH 32
#endif
#ifndef kHidden
#define kHidden 64
#endif
constexpr uint32_t kMoeBs = kBS;               // token 数 (Makefile -DkBS)
constexpr uint32_t kMoeH = kH;                 // 隐层维度 (Makefile -DkH)
constexpr uint32_t kMoeHiddenDim = kHidden;    // GMM1 输出维度 (Makefile -DkHidden)
constexpr uint32_t kTopK = 1U;               // 每 token 路由专家数
constexpr uint32_t kEpWorldSize = 1U;        // 自回环: EP 世界大小 = 1
constexpr uint32_t kHcclMaxRankSize = 1024U;  // HCCL_MAX_RANK_SIZE (源 712 行): 通信缓冲 rank 上限
constexpr uint32_t kBlockNumPerEp = 8U;      // blockNumPerEP
constexpr uint32_t kAicNum = 1U;             // AI Core 数 (AIC)
constexpr uint32_t kBlockAivNum = 16U;       // AIV 核数 (GetBlockIdx()=0..15)
constexpr uint32_t kSharedExpertNum = 0U;    // 共享专家数 (0 = 无独立 dense 路径)
constexpr uint32_t kMaxOutputSize = 65536U;  // maxOutputSize
constexpr uint32_t kMaxTilesPerExpert = 1U;  // maxTilesPerExpert
constexpr uint32_t kChunkElems = 256U;       // SIM_FAKE DataCopy 单块 256 元素

// ============================================================================
// 二、Tiling / 上下文结构 (源 451 行 / 1109 行, 字段全保留)
// ============================================================================
struct MegaMoeDispatchBufferConfig {   // {256, 1, 6, 288} (源 main 初始化)
    uint32_t bufferSlots;
    uint32_t slotsPerRow;
    uint32_t rowsPerCore;
    uint32_t coreStride;
};

struct MegaMoeSendMaskBufferConfig {   // {256,1,2,64} / {256,1,6,64}
    uint32_t maskSlots;
    uint32_t slotsPerRow;
    uint32_t rowsPerCore;
    uint32_t coreStride;
};

struct MegaMoeUnpermuteBufferConfig {  // {16, 6, 128, 128, 64, 0}
    uint32_t chunkRows;
    uint32_t chunksPerCore;
    uint32_t rowElems;
    uint32_t rowStride;
    uint32_t coreStride;
    uint32_t reserved;
};

struct MegaMoeTilingData {
    uint32_t moeExpertPerRank; // 本卡参与 topK 路由的 MoE 专家数, 与 weight1 表达的专家数一致
    uint32_t bs;
    uint32_t h;
    uint32_t hiddenDim;
    uint32_t epWorldSize;
    uint32_t blockNumPerEP;
    uint32_t maxOutputSize;
    uint32_t topK;
    uint32_t aicNum;
    uint32_t blockAivNum;
    int64_t combineQuantMode;
    float clampLimit;
    uint8_t groupedMatmulMode;
    int64_t topoType;
    uint32_t sharedExpertNum; // 独立 dense 路径的共享专家数, 不进入 topK/SendMask expert id 空间
    uint64_t combineSyncSlotCountPerExpert; // GMM2 -> Combine 同步 slot 数
    MegaMoeDispatchBufferConfig dispatchBufferConfig;      // Dispatch 分核配置
    MegaMoeSendMaskBufferConfig sendMaskConfigForCoreWithExtraExpert;    // 多 expert 核 mask 配置
    MegaMoeSendMaskBufferConfig sendMaskConfigForCoreWithoutExtraExpert; // 常规核 mask 配置
    uint32_t sendMaskCoreCountWithExtraExpert;
    MegaMoeUnpermuteBufferConfig unpermuteConfigForFullTokenChunk;       // 完整 token chunk 配置
    MegaMoeUnpermuteBufferConfig unpermuteConfigForTailTokenChunk;       // 尾 chunk 配置
    uint32_t unpermuteFullTokenChunkCoreCount;
    int32_t topkWeightsPrefetch;
    uint32_t maxTilesPerExpert;  // GMM1 tile 状态位区每 expert 容量
    uint8_t actMode;             // 激活模式: 0=swiglu, 1=situ
    uint8_t actSubMode;          // swiglu 下忽略
    float activationAlpha;       // 默认 1.0
    float activationBeta;        // 默认 1.0
    uint32_t mGroupsPerWave;     // 每 Wave 消费 256-row M 分组数
    bool isPerExpertWeightTensor;
};

struct Mc2MoeContext {   // 源 1124 行 (2026-08 同步: kfcContextAddr + hccld 缓冲扩至 1024)
    uint32_t epRankId;
    uint32_t rankSizePerServer;
    uint64_t kfcContextAddr;                    // 通信 API 所需的地址
    uint64_t epHcclBuffer[kHcclMaxRankSize];    // 自回环仅使用 [0]
    uint64_t hcommHandle[kHcclMaxRankSize];     // 支持 ROCE 或 URMA
};

// ============================================================================
// 三-补、exp 近似 (无 libm 依赖; SwiGLU silu 的 exp 组件, 源真机用 Ascend Exp 指令)
//   exp(z), z∈[-5,5]: z = k*ln2 + r (|r|<=ln2/2), exp = 2^k * exp(r)
// ============================================================================
// [修复] 删除 "MM_INLINE" 后多余的 inline 说明符, 消除 -Wduplicate-decl-specifier 告警。
MM_INLINE float exp_approx(float z)
{
    const float kLn2 = 0.69314718055994530941723212145818f;
    const float kInvLn2 = 1.4426950408889634073599246810019f;
    long k = (long)(z * kInvLn2 + (z < 0 ? -0.5f : 0.5f));
    const float r = z - (float)k * kLn2;
    float e = 1.0f + r * (1.0f + r * (0.5f + r * (1.0f / 6.0f + r * (1.0f / 24.0f + r * (1.0f / 120.0f)))));
    float twoK = 1.0f;
    if (k >= 0) {
        for (long j = 0; j < k; ++j) twoK *= 2.0f;
    } else {
        for (long j = 0; j < -k; ++j) twoK *= 0.5f;
    }
    return twoK * e;
}

// ============================================================================
// 四、FP8 A8W8 量化解码 (E4M3FN + E8M0 scale) — 计算语义完整保留
// ============================================================================
// E4M3FN: bit7=符号, bit[6:3]=指数(偏置7), bit[2:0]=尾数; e==0 为次正规
// [修复] 删除 "MM_INLINE" 后多余的 inline 说明符, 消除 -Wduplicate-decl-specifier 告警。
MM_INLINE float fp8_e4m3_to_f32(uint8_t raw)
{
    const uint32_t s = (raw >> 7U) & 1U;
    const uint32_t e = (raw >> 3U) & 0xFU;
    const uint32_t m = raw & 0x7U;
    float val;
    if (e == 0U) {
        val = static_cast<float>(m) / 8.0f * 0.015625f;      // 2^-6 * m/8
    } else {
        float frac = 1.0f + static_cast<float>(m) / 8.0f;
        float exp2 = 1.0f;
        int32_t bias = static_cast<int32_t>(e) - 7;
        if (bias >= 0) {
            for (int32_t i = 0; i < bias; ++i) exp2 *= 2.0f;
        } else {
            for (int32_t i = 0; i < -bias; ++i) exp2 *= 0.5f;
        }
        val = frac * exp2;
    }
    return s ? -val : val;
}

// E8M0: 纯指数 (偏置 127), scale = 2^(signed)
// [修复] 删除 "MM_INLINE" 后多余的 inline 说明符, 消除 -Wduplicate-decl-specifier 告警。
MM_INLINE float fp8_e8m0_scale(uint8_t raw)
{
    const int32_t e = static_cast<int32_t>(raw) - 127;
    float s = 1.0f;
    if (e >= 0) {
        for (int32_t i = 0; i < e; ++i) s *= 2.0f;
    } else {
        for (int32_t i = 0; i < -e; ++i) s *= 0.5f;
    }
    return s;
}

// ============================================================================
// 四a、tile 计算核心 (issue #180 迁移; sim / sim_mt / sim_mt_dyn 三入口共用)
// ============================================================================
// GMM = 逐 token TGEMV_MX/TGEMV_MX_ACC (MX FP16 pair —— pto-spec: FP16/BF16 侧
// 免 scale; K/N-tile=32, fp32 CUBE 累加); SwiGLU/x 量化/Combine = VEC tile 链;
// FP8 权重 = 每 PE 私有 tile 化解码 (TLOAD(E4M3[32,32]) → TCVT(fp16) →
// TMULS(k 组 scale 折叠) → TSTORE), 产物为 fp16 权重副本 —— 无标量 bit-twiddle /
// fp32 workspace / 跨 PE 权重栅栏。
//
// 注: B 侧 E4M3+E8M0 的 MX 原地消费 (TGEMV_MX 带 ScaleB) 在当前工具链/模型
// 组合下不可用 —— 编译期契约要求 Local ScaleB 为 RowMajor [ceil(K/32),N], 而
// TLOAD 发射的 scale 操作数无 CUBE_M32 布局描述符, 运行时模型断言要求
// CUBE_M32 [N,ceil(K/32)] (isa/Block.cpp TMATMUL_MX right-scale 校验)。待模型
// 侧统一载体契约后可平滑切换。
//
// 权重 scale 布局 (bench 自控, 每 PE 解码时 TMULS 折叠):
//   w1Scale[e][k/32]  (epr × h/32 个 E8M0 标量, k ∈ [0, h))
//   w2Scale[e][k/32]  (epr × (hd/2)/32 个 E8M0 标量, k ∈ [0, hd/2))
// 调用契约: h % 32 == 0 且 hiddenDim % 64 == 0 (GMM2 的 K=hd/2 须整除 32)。

using MxGemvVec = CubeTileM16<__half, 1, 32>;
using MxGemvMtx = CubeTileN8<__half, 32, 32>;
using MxGemvDst = CubeAccumulatorM16<float, 1, 32>;
// VEC 链 tile: 物理列 64 使 fp32/fp16 两侧 TCVT 派生行数一致 (128B 下限)
using MxChainF32 = Tile<Location::Vec, float, 1, 64, BLayout::RowMajor, 1, 32>;
using MxChainF16 = Tile<Location::Vec, __half, 1, 64, BLayout::RowMajor, 1, 32>;
using MxDecSrc = Tile<Location::Vec, __fp8_e4m3, 32, 32>;
using MxDecDst = Tile<Location::Vec, __half, 32, 32>;

// 每 PE 私有 scratch 布局: w1F16[epr*h*hd] + w2F16[epr*(hd/2)*h] + xf16[h] +
// y1[hd]f32 + y2f16[hd/2] + y3[h]f32
struct MxTileScratch {
    __half* w1F16;
    __half* w2F16;
    __half* xf16;
    float* y1;
    __half* y2f16;
    float* y3;
};

// 去掉inline，修复编译告警
MM_INLINE uint32_t mx_tile_scratch_bytes(uint32_t epr, uint32_t h, uint32_t hd)
{
    return (epr * h * hd + epr * (hd / 2U) * h) * 2U   // fp16 权重副本
         + h * 2U + hd * 4U + hd * 2U + h * 4U;        // xf16/y1/y2f16/y3
}
// 去掉inline，修复编译告警
MM_INLINE MxTileScratch mx_tile_scratch_at(uint8_t* base, uint32_t pe,
                                                   uint32_t epr, uint32_t h, uint32_t hd)
{
    MxTileScratch s;
    uint8_t* p = base + pe * mx_tile_scratch_bytes(epr, h, hd);
    s.w1F16 = reinterpret_cast<__half*>(p);
    s.w2F16 = s.w1F16 + epr * h * hd;
    s.xf16 = s.w2F16 + epr * (hd / 2U) * h;
    s.y1 = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(s.xf16) + h * 2U);
    s.y2f16 = reinterpret_cast<__half*>(reinterpret_cast<uint8_t*>(s.y1) + hd * 4U);
    s.y3 = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(s.y2f16) + hd * 2U);
    return s;
}

// FP8 权重 tile 化解码 (每 PE 私有, 免栅栏):
//   wF16[e][k][n] = fp16(E4M3) * 2^(scale[e][k/32]-127), 逐 [32,32] tile
// 去掉inline，修复编译告警
MM_INLINE void mx_decode_weights_tile(MxTileScratch& s,
                                              uint32_t epr, uint32_t h, uint32_t hd)
{
    for (uint32_t e = 0U; e < epr; ++e) {
        for (uint32_t k0 = 0U; k0 < h; k0 += 32U) {
            const float s1 = fp8_e8m0_scale(
                g_mmWeightScales1[e * (h / 32U) + k0 / 32U]);
            for (uint32_t n0 = 0U; n0 < hd; n0 += 32U) {
                MxDecSrc ws;
                global_tensor<__fp8_e4m3, RowMajor<-1, -1>> gWS(
                    reinterpret_cast<__fp8_e4m3*>(
                        g_mmWeight1 + e * h * hd + k0 * hd + n0),
                    32, hd);
                TLOAD(ws, gWS);
                MxDecDst wd;
                TCVT(wd, ws);
                TMULS(wd, wd, s1);
                global_tensor<__half, RowMajor<-1, -1>> gWD(
                    s.w1F16 + e * h * hd + k0 * hd + n0, 32, hd);
                TSTORE(gWD, wd);
            }
        }
        for (uint32_t k0 = 0U; k0 < hd / 2U; k0 += 32U) {
            const float s2 = fp8_e8m0_scale(
                g_mmWeightScales2[e * ((hd / 2U) / 32U) + k0 / 32U]);
            for (uint32_t n0 = 0U; n0 < h; n0 += 32U) {
                MxDecSrc ws;
                global_tensor<__fp8_e4m3, RowMajor<-1, -1>> gWS(
                    reinterpret_cast<__fp8_e4m3*>(
                        g_mmWeight2 + e * (hd / 2U) * h + k0 * h + n0),
                    32, h);
                TLOAD(ws, gWS);
                MxDecDst wd;
                TCVT(wd, ws);
                TMULS(wd, wd, s2);
                global_tensor<__half, RowMajor<-1, -1>> gWD(
                    s.w2F16 + e * (hd / 2U) * h + k0 * h + n0, 32, h);
                TSTORE(gWD, wd);
            }
        }
    }
}

// 单 token 单 topK 槽位 tile 计算:
//   x 量化(fp32→fp16) → GMM1(TGEMV_MX) → SwiGLU(VEC 链) → GMM2 → Combine
// (k=0 块剥离循环外: 消除运行期 TGEMV_MX/ACC 分支的 tile 跨分支 PHI)
// kk==0 时 combineRow 直接写, kk>0 时读-累加-写 (topK 通用)。
// 去掉inline，修复编译告警
MM_INLINE void mx_token_compute(const MxTileScratch& s,
                                        const float* xRow, uint32_t h, uint32_t hd,
                                        uint32_t epr, uint32_t expert, float weight,
                                        uint32_t kk, float* combineRow)
{
    // x 行 fp32 → fp16 (VEC TCVT; MX A 侧 fp16 免 scale)
    for (uint32_t c = 0U; c < h; c += 32U) {
        MxChainF32 xf;
        global_tensor<float, RowMajor<1, 32>> gX(
            const_cast<float*>(xRow) + c);
        TLOAD(xf, gX);
        MxChainF16 xh;
        TCVT(xh, xf);
        global_tensor<__half, RowMajor<1, 32>> gXH(s.xf16 + c);
        TSTORE(gXH, xh);
    }

    // GMM1 (源 ProcessGmm1Wave): y1[n] = Σ_k x[k]·w1[e][k][n]
    for (uint32_t n0 = 0U; n0 < hd; n0 += 32U) {
        MxGemvDst d;
        {
            MxGemvVec v;
            global_tensor<__half, RowMajor<1, 32>> gV(s.xf16);
            TLOAD_CUBE(v, gV);
            MxGemvMtx m;
            global_tensor<__half, RowMajor<-1, -1>> gM(
                s.w1F16 + expert * h * hd + n0, 32, hd);
            TLOAD_CUBE(m, gM);
            TGEMV_MX(d, m, v, fixp::keep_acc());
        }
        for (uint32_t k0 = 32U; k0 < h; k0 += 32U) {
            MxGemvVec v;
            global_tensor<__half, RowMajor<1, 32>> gV(s.xf16 + k0);
            TLOAD_CUBE(v, gV);
            MxGemvMtx m;
            global_tensor<__half, RowMajor<-1, -1>> gM(
                s.w1F16 + expert * h * hd + k0 * hd + n0, 32, hd);
            TLOAD_CUBE(m, gM);
            TGEMV_MX_ACC(d, d, m, v, fixp::keep_acc());
        }
        global_tensor<float, RowMajor<1, 32>> gY1(s.y1 + n0);
        TSTORE_CUBE(gY1, d);
    }

    // SwiGLU (源 Gmm1SwigluState): y2 = silu(y1[:hd/2]) * y1[hd/2:]  (VEC 链)
    for (uint32_t c = 0U; c < hd / 2U; c += 32U) {
        MxChainF32 a, b;
        global_tensor<float, RowMajor<1, 32>> gA(s.y1 + c);
        TLOAD(a, gA);
        global_tensor<float, RowMajor<1, 32>> gB(s.y1 + hd / 2U + c);
        TLOAD(b, gB);
        MxChainF32 neg, e, one, r, sig;
        TMULS(neg, a, -1.0f);
        TEXP(e, neg);
        TADDS(one, e, 1.0f);
        TRECIP(r, one);
        TMUL(sig, a, r);
        TMUL(sig, sig, b);
        MxChainF16 h16;
        TCVT(h16, sig);
        global_tensor<__half, RowMajor<1, 32>> gY2(s.y2f16 + c);
        TSTORE(gY2, h16);
    }

    // GMM2 (源 ProcessGmm2Wave): y3[n] = Σ_k y2[k]·w2[e][k][n]
    for (uint32_t n0 = 0U; n0 < h; n0 += 32U) {
        MxGemvDst d;
        {
            MxGemvVec v;
            global_tensor<__half, RowMajor<1, 32>> gV(s.y2f16);
            TLOAD_CUBE(v, gV);
            MxGemvMtx m;
            global_tensor<__half, RowMajor<-1, -1>> gM(
                s.w2F16 + expert * (hd / 2U) * h + n0, 32, h);
            TLOAD_CUBE(m, gM);
            TGEMV_MX(d, m, v, fixp::keep_acc());
        }
        for (uint32_t k0 = 32U; k0 < hd / 2U; k0 += 32U) {
            MxGemvVec v;
            global_tensor<__half, RowMajor<1, 32>> gV(s.y2f16 + k0);
            TLOAD_CUBE(v, gV);
            MxGemvMtx m;
            global_tensor<__half, RowMajor<-1, -1>> gM(
                s.w2F16 + expert * (hd / 2U) * h + k0 * h + n0, 32, h);
            TLOAD_CUBE(m, gM);
            TGEMV_MX_ACC(d, d, m, v, fixp::keep_acc());
        }
        global_tensor<float, RowMajor<1, 32>> gY3(s.y3 + n0);
        TSTORE_CUBE(gY3, d);
    }

    // Combine (源 ProcessCombineExperts): yAcc = Σ_kk weight·y3 (VEC tile)
    for (uint32_t c = 0U; c < h; c += 32U) {
        MxChainF32 t;
        global_tensor<float, RowMajor<1, 32>> gT(s.y3 + c);
        TLOAD(t, gT);
        TMULS(t, t, weight);
        global_tensor<float, RowMajor<1, 32>> gAcc(combineRow + c);
        if (kk == 0U) {
            TSTORE(gAcc, t);
        } else {
            MxChainF32 acc;
            TLOAD(acc, gAcc);
            TADD(acc, acc, t);
            TSTORE(gAcc, acc);
        }
    }
}

// VEC tile 行拷贝 (UnpermuteTokens 用)
// 去掉inline，修复编译告警
MM_INLINE void mx_copy_row_tile(const float* src, float* dst, uint32_t h)
{
    for (uint32_t c = 0U; c < h; c += 32U) {
        MxChainF32 v;
        global_tensor<float, RowMajor<1, 32>> gSrc(const_cast<float*>(src) + c);
        TLOAD(v, gSrc);
        global_tensor<float, RowMajor<1, 32>> gDst(dst + c);
        TSTORE(gDst, v);
    }
}

// ============================================================================
// 五、MegaMoeWave — 源类 (9240 行起) 全阶段迁移
// ============================================================================
// 模板参数与源实例化 MegaMoeWave<bfloat16_t, bfloat16_t, float, fp8_e4m3fn_t, 4, 0, false, false>
// 一一对应: <QuantOutType, ActivationType, TopkWeightsType, CombineQuantMode,
//            QuantMode, TopkWeightsPrefetch, UseSharedExpert, ActMode>
template <typename QuantOutType, typename ActivationType, typename TopkWeightsType,
          typename CombineQuantMode, int kQuantMode, int kTopkWeightsPrefetch,
          bool kUseSharedExpert, int kActMode>
struct MegaMoeWave {
    // ==== 成员 (源 9300 行起同名保留) ====
    Mc2MoeContext const* mc2Context_{nullptr};
    MegaMoeTilingData tilingData_{};

    // 派生的计算配置 (PTO 上为解码布局参数, 源 InitXxxConfig 阶段语义)
    uint32_t quantGroups1_;   // weight1 每行 32 元素一组 → hiddenDim/32 组
    uint32_t quantGroups2_;   // weight2 每行 32 元素一组 → h/32 组
    uint32_t gmm1K_;          // = h
    uint32_t gmm1N_;          // = hiddenDim
    uint32_t gmm2K_;          // = hiddenDim/2
    uint32_t gmm2N_;          // = h

    // 工作区偏移 (workspace 布局, 源 workspaceGM 分配语义)
    uint32_t w1F32Offset_;    // 量化后的 weight1 (fp32): [experts][h][hiddenDim]
    uint32_t w2F32Offset_;    // 量化后的 weight2 (fp32): [experts][hiddenDim/2][h]
    uint32_t dispatchOffset_; // dispatch token 映射表: [experts][bs]
    uint32_t maskOffset_;     // expert mask: [experts][bs]
    uint32_t combineOffset_;  // combine 缓冲: [bs][h]
    uint32_t statsOffset_;    // 每核 token 统计: [blockAivNum][experts]

    // ==== 生命周期 ====

    // 源 Init (9508): 记录上下文/输入/权重地址与 tilingData, 初始化派生配置
    MM_INLINE void Init(Mc2MoeContext const* context, MegaMoeTilingData const* tilingData)
    {
        mc2Context_ = context;
        tilingData_ = *tilingData;
        quantGroups1_ = tilingData->hiddenDim / 32U;
        quantGroups2_ = tilingData->h / 32U;
        gmm1K_ = tilingData->h;
        gmm1N_ = tilingData->hiddenDim;
        gmm2K_ = tilingData->hiddenDim / 2U;
        gmm2N_ = tilingData->h;
        // workspace 布局 (源 workspaceGM 分块语义)
        w1F32Offset_ = 0U;
        w2F32Offset_ = w1F32Offset_ + tilingData->moeExpertPerRank * tilingData->h * tilingData->hiddenDim * 4U;
        dispatchOffset_ = w2F32Offset_ + tilingData->moeExpertPerRank * (tilingData->hiddenDim / 2U) * tilingData->h * 4U;
        maskOffset_ = dispatchOffset_ + tilingData->moeExpertPerRank * tilingData->bs * 4U;
        combineOffset_ = maskOffset_ + tilingData->moeExpertPerRank * tilingData->bs * 1U;
        statsOffset_ = combineOffset_ + tilingData->bs * tilingData->h * 4U;
    }

    // 源 InitInputPrepareConfigs (9378) / InitTokenDispatchConfig (9435) /
    //    InitGmm1Config (9455) / InitGmm2CombineConfigs (9470) / InitTokenUnpermuteConfig (9488)
    // PTO 版: 派生配置已在 Init 完成 (上), 保留阶段语义
    void InitInputPrepareConfigs() {}
    void InitTokenDispatchConfig() {}
    void InitGmm1Config() {}
    void InitGmm2CombineConfigs() {}
    void InitTokenUnpermuteConfig() {}

    // ==== 输入准备 / 量化阶段 ====

    // 源 DispatchBuffInit (9582): 分配 dispatch 缓冲. PTO: 映射表清零
    void DispatchBuffInit()
    {
        const uint32_t expertNum = tilingData_.moeExpertPerRank;
        float* dispatch = reinterpret_cast<float*>(g_mmWorkspace + dispatchOffset_);
        for (uint32_t e = 0; e < expertNum; ++e) {
            for (uint32_t t = 0; t < tilingData_.bs; ++t) {
                dispatch[e * tilingData_.bs + t] = -1.0f;  // 空槽
            }
        }
    }

    // 源 SendAndQuantBuffInit (9731): 申请公共 mask/workspace reset/token quant 缓冲.
    // PTO: 统计槽清零
    void SendAndQuantBuffInit()
    {
        int32_t* stats = reinterpret_cast<int32_t*>(g_mmWorkspace + statsOffset_);
        for (uint32_t core = 0; core < kBlockAivNum; ++core) {
            for (uint32_t e = 0; e < tilingData_.moeExpertPerRank; ++e) {
                stats[core * tilingData_.moeExpertPerRank + e] = 0;
            }
        }
    }

    // 源 QuantizeLocalTokens / 权重量化 (3754): A8W8 权重 E4M3→fp32 并应用 E8M0 scale,
    // 结果写入 workspace (源为 GM↔UB 量化搬运, PTO 为标量解码写回, 数学一致)
    void QuantizeLocalTokens()
    {
        const uint32_t expertNum = tilingData_.moeExpertPerRank;
        // weight1: [e][h][hiddenDim] fp8 → fp32
        {
            float* w1 = reinterpret_cast<float*>(g_mmWorkspace + w1F32Offset_);
            for (uint32_t e = 0; e < expertNum; ++e) {
                for (uint32_t k = 0; k < tilingData_.h; ++k) {
                    for (uint32_t n = 0; n < tilingData_.hiddenDim; ++n) {
                        const uint32_t flatIdx = e * tilingData_.h * tilingData_.hiddenDim + k * tilingData_.hiddenDim + n;
                        const uint32_t group = (k * tilingData_.hiddenDim + n) / 32U;
                        const uint32_t scaleIdx = e * (tilingData_.h * tilingData_.hiddenDim / 32U) + group;
                        w1[flatIdx] = fp8_e4m3_to_f32(g_mmWeight1[flatIdx]) *
                                      fp8_e8m0_scale(g_mmWeightScales1[scaleIdx % (tilingData_.h / 32U * 2U * 2U * 2U)]);
                    }
                }
            }
        }
        // weight2: [e][hiddenDim/2][h] fp8 → fp32
        {
            const uint32_t k2 = tilingData_.hiddenDim / 2U;
            float* w2 = reinterpret_cast<float*>(g_mmWorkspace + w2F32Offset_);
            for (uint32_t e = 0; e < expertNum; ++e) {
                for (uint32_t k = 0; k < k2; ++k) {
                    for (uint32_t n = 0; n < tilingData_.h; ++n) {
                        const uint32_t flatIdx = e * k2 * tilingData_.h + k * tilingData_.h + n;
                        const uint32_t group = (k * tilingData_.h + n) / 32U;
                        const uint32_t scaleIdx = e * (k2 * tilingData_.h / 32U) + group;
                        w2[flatIdx] = fp8_e4m3_to_f32(g_mmWeight2[flatIdx]) *
                                      fp8_e8m0_scale(g_mmWeightScales2[scaleIdx % (tilingData_.h / 32U * 2U * 2U * 2U)]);
                    }
                }
            }
        }
    }

    // 源 GatherAndSendExpertMasks (3934): 统计各 expert 服务的 token 并生成 mask.
    // 自回环 (epWorldSize=1): 跨 rank 发送退化为本地写 mask 表
    void GatherAndSendExpertMasks()
    {
        uint8_t* mask = g_mmWorkspace + maskOffset_;
        for (uint32_t t = 0; t < tilingData_.bs * tilingData_.topK; ++t) {
            const int32_t expert = g_mmTopkIds[t];
            mask[static_cast<uint32_t>(expert) * tilingData_.bs + t / tilingData_.topK] = 1U;
        }
        // 源: SendMaskCal + CrossRankSyncInWorldSize 发送; 自回环 rank 0 无远端, 本地即全量
    }

    // 源 ResetDispatchWorkspace (4051): 清零 dispatch/统计工作区
    void ResetDispatchWorkspace()
    {
        DispatchBuffInit();
        SendAndQuantBuffInit();
    }

    // 源 PrepareSharedExpertInput (4119): 共享专家输入拆分 (sharedExpertNum==0 不启用)
    void PrepareSharedExpertInput()
    {
        // tilingData_.sharedExpertNum == 0: 源条件分支跳过, 保留函数骨架
    }

    // 源 InitExpertTokenCountExportBuffers (9709): 统计导出缓冲初始化
    // 注: volatile 阻止相邻 i64 清零被合并为 16B tile store (BLK_TSTORE v2i64),
    //     linxv5 后端不支持整数 tile 类型, 会报 "Cannot select: v2i64 = BUILD_VECTOR"
    void InitExpertTokenCountExportBuffers()
    {
        volatile int64_t* p = g_mmExpertTokenNums;
        for (uint32_t e = 0; e < tilingData_.moeExpertPerRank; ++e) {
            p[e] = 0;
        }
    }

    // ==== 计算阶段 ====

    // 源 CrossRankSyncInWorldSize (9295): 全 rank 同步. 自回环 epWorldSize=1 本地退化
    void CrossRankSyncInWorldSize() {}

    // 源 ProcessGmm1Wave (9287): GMM1 (x @ w1^T) 一个 wave.
    // PTO: 单 token 行向量 × 权重矩阵, 按 k 累加 (K=128 全展开)
    void ProcessGmm1Wave(uint32_t token, uint32_t expert, float* y1Out)
    {
        const float* w1 = reinterpret_cast<const float*>(g_mmWorkspace + w1F32Offset_);
        const float* xRow = g_mmX + token * tilingData_.h;
        const float* wRowBase = w1 + expert * tilingData_.h * tilingData_.hiddenDim;
        for (uint32_t n = 0; n < tilingData_.hiddenDim; ++n) {
            float acc = 0.0f;
            for (uint32_t k = 0; k < tilingData_.h; ++k) {
                acc += xRow[k] * wRowBase[k * tilingData_.hiddenDim + n];  // GMM1: y1=Σ_k x_k·w1[k][n]
            }
            y1Out[n] = acc;
        }
    }

    // 源 ProcessGmm2Wave (9291): GMM2 (silu 输出 @ w2^T)
    void ProcessGmm2Wave(uint32_t token, uint32_t expert, const float* y2In, float* y3Out)
    {
        const uint32_t k2 = tilingData_.hiddenDim / 2U;
        const float* w2 = reinterpret_cast<const float*>(g_mmWorkspace + w2F32Offset_);
        const float* wRowBase = w2 + expert * k2 * tilingData_.h;
        for (uint32_t n = 0; n < tilingData_.h; ++n) {
            float acc = 0.0f;
            for (uint32_t k = 0; k < k2; ++k) {
                acc += y2In[k] * wRowBase[k * tilingData_.h + n];  // GMM2: y3=Σ_k silu_half_k·w2[k][n]
            }
            y3Out[n] = acc;
        }
    }

    // 源 ProcessCombineExperts (9293): 各 expert 输出按 topk_weights 加权合并
    void ProcessCombineExperts(uint32_t token, float weight, const float* y3In, float* yOut)
    {
        float* combine = reinterpret_cast<float*>(g_mmWorkspace + combineOffset_);
        float* yBase = combine + token * tilingData_.h;
        for (uint32_t n = 0; n < tilingData_.h; ++n) {
            yBase[n] += weight * y3In[n];   // yOut[token] += topkWeight * y3
        }
    }

    // 源 UnpermuteTokens (9163): combine 缓冲按 token 序写回 y (等价于源 token 还原)
    void UnpermuteTokens()
    {
        const float* combine = reinterpret_cast<const float*>(g_mmWorkspace + combineOffset_);
        for (uint32_t t = 0; t < tilingData_.bs; ++t) {
            for (uint32_t n = 0; n < tilingData_.h; ++n) {
                g_mmY[t * tilingData_.h + n] = combine[t * tilingData_.h + n];
            }
        }
    }

    // 源 InitTokenUnpermuteBuffers (9294): unpermute 缓冲初始化 (清零 combine 区)
    void InitTokenUnpermuteBuffers()
    {
        float* combine = reinterpret_cast<float*>(g_mmWorkspace + combineOffset_);
        for (uint32_t i = 0; i < tilingData_.bs * tilingData_.h; ++i) {
            combine[i] = 0.0f;
        }
    }

    // 源 ProcessGmmPipeline (9273): GMM 主流程 (wave 循环). PTO: token 域循环
    void ProcessGmmPipeline()
    {
        float y1[kMoeHiddenDim];
        float y2[kMoeHiddenDim / 2U];
        float y3[kMoeH];
        const uint32_t tid = get_thread_idx();
        // ceil 分片: bs < 16 核时每核至多 1 token, 尾部核空转 (保持核×轮结构;
        // bs % 16 == 0 时与源 bs/16 连续分片逐 token 一致)
        const uint32_t perCore =
            (tilingData_.bs + kBlockAivNum - 1U) / kBlockAivNum;

        // 每伪核 (16 核 = 4 线程 × 4 伪核) 处理 perCore 个连续 token (源 16 核分片)
        for (uint32_t lc = 0; lc < kBlockAivNum / 4U; ++lc) {
            const uint32_t coreIdx = tid * (kBlockAivNum / 4U) + lc;
            for (uint32_t i = 0; i < perCore; ++i) {
                const uint32_t token = coreIdx * perCore + i;
                if (token >= tilingData_.bs) break;  // bs < 16 核时尾核空转

                // 路由 (源 ProcessMoeExpertStages 的 expert 分配): topkIds[token]
                const int32_t expert = g_mmTopkIds[token * tilingData_.topK];
                const float weight = g_mmTopkWeights[token * tilingData_.topK];

                // GMM1: y1[hiddenDim] = x[token] @ w1[expert]
                ProcessGmm1Wave(token, static_cast<uint32_t>(expert), y1);

                // SwiGLU (源 Gmm1SwigluState): y2 = silu(y1[:k]) * y1[k:], k=hiddenDim/2
                // silu(z) = z / (1 + exp(-z)) (源: Muls(-1) → Exp → Adds(1) → Reciprocal → Muls)
                for (uint32_t k = 0; k < tilingData_.hiddenDim / 2U; ++k) {
                    float z = y1[k];
                    float e = 1.0f / (1.0f + exp_approx(-z));  // sigmoid(z)
                    y2[k] = z * e;                              // x·sigmoid(x) = silu
                }

                // GMM2: y3[h] = y2 @ w2[expert]
                ProcessGmm2Wave(token, static_cast<uint32_t>(expert), y2, y3);

                // Combine: y[token] += topkWeight * y3 (源 ProcessCombineExperts)
                ProcessCombineExperts(token, weight, y3, g_mmY);

                // 核内 token 统计 (源 InitExpertTokenCountExportBuffers 汇总语义)
                int32_t* stats = reinterpret_cast<int32_t*>(g_mmWorkspace + statsOffset_);
                stats[coreIdx * tilingData_.moeExpertPerRank + static_cast<uint32_t>(expert)] += 1;
            }
        }
    }

    // 源 ProcessSharedExpertGmm1 (9270) / ProcessSharedExpertGmm2 (9272): 共享专家 dense 路径
    void ProcessSharedExpertGmm1() {}
    void ProcessSharedExpertGmm2() {}

    // 源 ProcessMoeExpertStages (9271): 按 wave 执行 Dispatch→GMM1→SwiGLU→GMM2→Combine
    void ProcessMoeExpertStages()
    {
        InitTokenUnpermuteBuffers();
        ProcessGmmPipeline();
        UnpermuteTokens();
    }

    // ==== 主入口 ====

    // 源 Process (10319): 总流程. #ifdef MEGA_MOE_SIM_FAKE 分支与源同构保留
    void Process()
    {
#ifdef MEGA_MOE_SIM_FAKE
        // ============ 源 10321-10374 camodel 仿真快路径 (保留全部 tile 原语) ============
        // 不执行 AIC↔AIV 跨核 GM 数据流, 由 AIV 独立执行真实 vector 激活:
        //   yOut = sigmoid(x) = 1 / (1 + exp(-x)), 16 核 × 8 轮 × 256 元素
        const uint32_t fakeTotalElems = tilingData_.bs * tilingData_.h;  // 32768
        const uint32_t fakeRounds = fakeTotalElems / (kBlockAivNum * kChunkElems);  // 8
        // [AIC 冒烟] GM 读即丢弃
        volatile uint32_t smoke = *reinterpret_cast<const volatile uint32_t*>(g_mmX);
        (void)smoke;
        using gmShape = global_tensor<float, RowMajor<1, kChunkElems>>;
        using tileShape = Tile<Location::Vec, float, 1, kChunkElems, BLayout::RowMajor>;
        using itGM = global_iterator<gmShape, tileShape>;
        const uint32_t tid = get_thread_idx();
        itGM xIter(g_mmX);
        itGM yIter(g_mmY);
        for (uint32_t lc = 0U; lc < kBlockAivNum / 4U; ++lc) {
            const uint32_t coreIdx = tid * (kBlockAivNum / 4U) + lc;
            for (uint32_t round = 0U; round < fakeRounds; ++round) {
                const uint32_t base = (round * kBlockAivNum + coreIdx) * kChunkElems;
                tileShape t;
                auto srcGT = xIter(0, base);   // DataCopy(GM→UB)
                TLOAD(t, srcGT);
                TMULS(t, t, -1.0f);            // Muls(-1)
                TEXP(t, t);                    // Exp
                TADDS(t, t, 1.0f);             // Adds(1)
                TRECIP(t, t);                  // Reciprocal → sigmoid
                auto dstGT = yIter(0, base);   // DataCopy(UB→GM)
                TSTORE(dstGT, t);
            }
        }
        // 自回环固定统计 (源 10369-10372)
        g_mmExpertTokenNums[0] = static_cast<int64_t>(tilingData_.bs) / 2;
        g_mmExpertTokenNums[1] = static_cast<int64_t>(tilingData_.bs) / 2;
#else
        // ============ 源 10377-10423 完整真机流水 ============
        // 阶段 1: 输入准备 (源 SendAndQuantBuffInit + QuantizeLocalTokens +
        //         GatherAndSendExpertMasks + ResetDispatchWorkspace)
        SendAndQuantBuffInit();
        QuantizeLocalTokens();
        GatherAndSendExpertMasks();
        ResetDispatchWorkspace();
        // 共享专家输入准备 (源: if (sharedExpertNum_ > 0) PrepareSharedExpertInput)
        if (kUseSharedExpert) {
            PrepareSharedExpertInput();
        }

        // 阶段 2: MoE 专家流水 Dispatch → GMM1 → SwiGLU → GMM2 → Combine
        //         (源 ProcessMoeExpertStages → ProcessGmmPipeline)
        ProcessMoeExpertStages();

        // 阶段 3: 等待所有 rank Combine 发送完成再 Unpermute (源 10416-10421);
        //         自回环 epWorldSize=1 同步本地退化 (CrossRankSyncInWorldSize 空实现),
        //         Unpermute 在 ProcessMoeExpertStages 内完成, 语义一致

        // 统计导出: 汇总 16 核局部 token 计数 (源 InitExpertTokenCountExportBuffers)
        InitExpertTokenCountExportBuffers();
        {
            const int32_t* stats = reinterpret_cast<const int32_t*>(g_mmWorkspace + statsOffset_);
            for (uint32_t core = 0; core < kBlockAivNum; ++core) {
                for (uint32_t e = 0; e < tilingData_.moeExpertPerRank; ++e) {
                    g_mmExpertTokenNums[e] += stats[core * tilingData_.moeExpertPerRank + e];
                }
            }
        }
#endif
    }
};

// ============================================================================
// 六、kernel 入口 (与源 MegaMoeSelfLoopbackKernel 对应)
// ============================================================================
// 说明: 上方 MegaMoeWave 类完整保留全部阶段函数的语义定义 (结构文档);
// 入口采用平铺顺序执行 (等价调用链): 实测 gfrun 功能仿真器对
// "类对象 + 成员函数调用链" 的返回地址 (FRET/BARG) 模拟存在缺陷
// (仓库既有限制, moe_dispatch_v2 等自校验用例同因 R2=1), 平铺后
// 同一逻辑以单函数形态执行, 规避该缺陷且不省略任何阶段。
template <int kBatchSize, int kHiddenDim>
void mega_moe_sim_kernel(float* yOut, float* xIn, int64_t* tokOut)
{
    // [AIC 冒烟] 源: AIC 核读 aGmAddr 即弃
    volatile uint32_t smoke = *reinterpret_cast<const volatile uint32_t*>(xIn);
    (void)smoke;



    // TilingData 填充 (源 main 逐字段初始化, 字段全保留)
    static_assert(kBlockAivNum % 4U == 0U, "AIV core count must be divisible by 4 PE");
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
    //     16B tile store (BLK_TSTORE v2i64, linxv5 后端 Cannot select 崩溃;
    //     BS16 等小规格下常量折叠/调度差异会触发该合并)
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

    // workspace 布局 (MX 路径: 无 fp32 权重缓存; 每 PE tile scratch 见四a)
    const uint32_t dispatchOffset = 0U;
    const uint32_t maskOffset = dispatchOffset + tilingData.moeExpertPerRank * tilingData.bs * 4U;
    const uint32_t combineOffset = maskOffset + tilingData.moeExpertPerRank * tilingData.bs * 1U;
    const uint32_t statsOffset = combineOffset + tilingData.bs * tilingData.h * 4U;
    const uint32_t yScratchOffset = statsOffset + kBlockAivNum * tilingData.moeExpertPerRank * 4U;

#ifdef MEGA_MOE_SIM_FAKE
    // ============ 源 10321-10374 camodel 仿真快路径 (保留全部 tile 原语) ============
    // 轮数 ceil 化 + 越界保护 (与 A5 mega_moe_main.asc 同步): bs*h 不足
    // 16 伪核 × kChunkElems 时仅部分核各搬 1 块, 尾核空转
    const uint32_t fakeTotalElems = tilingData.bs * tilingData.h;
    const uint32_t fakeRounds =
        (fakeTotalElems + kBlockAivNum * kChunkElems - 1U) / (kBlockAivNum * kChunkElems);
    using gmShape = global_tensor<float, RowMajor<1, kChunkElems>>;
    using tileShape = Tile<Location::Vec, float, 1, kChunkElems, BLayout::RowMajor>;
    using itGM = global_iterator<gmShape, tileShape>;
    const uint32_t tid = get_thread_idx();
    (void)tid;  // 快路径同样与线程数解耦: 每线程覆盖全部 16 伪核 (幂等)
    itGM xIter(xIn);
    itGM yIter(yOut);
    for (uint32_t lc = 0U; lc < kBlockAivNum; ++lc) {
        const uint32_t coreIdx = lc;
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
    tokOut[0] = static_cast<int64_t>(tilingData.bs) / 2;
    tokOut[1] = static_cast<int64_t>(tilingData.bs) / 2;
#else
    // ============ 完整真机流水 (各阶段与 MegaMoeWave 阶段函数一一对应) ============
    // ---- 阶段 1: 输入准备 ----
    // SendAndQuantBuffInit (9731): 统计槽清零
    {
        int32_t* stats = reinterpret_cast<int32_t*>(g_mmWorkspace + statsOffset);
        for (uint32_t core = 0; core < kBlockAivNum; ++core) {
            for (uint32_t e = 0; e < tilingData.moeExpertPerRank; ++e) {
                stats[core * tilingData.moeExpertPerRank + e] = 0;
            }
        }
    }
    // QuantizeLocalTokens (3754) — MX 路径: 每 PE 私有 tile 化解码
    // (TLOAD(E4M3[32,32]) → TCVT(fp16) → TMULS(k 组 scale 折叠) → TSTORE);
    // issue #180: 标量 bit-twiddle / fp32 workspace 全删; 幂等冗余执行模型下
    // 每 PE 写自己的 scratch 副本 (同值, 写不相交)
    static_assert(kHiddenDim % 32 == 0, "h must be a multiple of the 32-wide MX K-tile");
    static_assert(kMoeHiddenDim % 64 == 0, "hiddenDim/2 must cover whole MX scale groups");
    const uint32_t mxTid = get_thread_idx();
    MxTileScratch mx = mx_tile_scratch_at(
        g_mmWorkspace + yScratchOffset, mxTid,
        tilingData.moeExpertPerRank, tilingData.h, tilingData.hiddenDim);
    mx_decode_weights_tile(mx, tilingData.moeExpertPerRank,
                           tilingData.h, tilingData.hiddenDim);
    // GatherAndSendExpertMasks (3934): 自回环本地 mask 表
    {
        uint8_t* mask = g_mmWorkspace + maskOffset;
        for (uint32_t t = 0; t < tilingData.bs * tilingData.topK; ++t) {
            const int32_t expert = g_mmTopkIds[t];
            mask[static_cast<uint32_t>(expert) * tilingData.bs + t / tilingData.topK] = 1U;
        }
    }
    // ResetDispatchWorkspace (4051) = DispatchBuffInit: dispatch 表清零
    {
        float* dispatch = reinterpret_cast<float*>(g_mmWorkspace + dispatchOffset);
        for (uint32_t e = 0; e < tilingData.moeExpertPerRank; ++e) {
            for (uint32_t t = 0; t < tilingData.bs; ++t) {
                dispatch[e * tilingData.bs + t] = -1.0f;
            }
        }
    }

    // ---- 阶段 2: 共享专家输入准备 (源 PrepareSharedExpertInput, sharedExpertNum==0 跳过) ----

    // ---- 阶段 3: MoE 专家流水 (ProcessMoeExpertStages → ProcessGmmPipeline) ----
    // tile 计算核心 (issue #180): GMM1/GMM2 = TGEMV_MX (MX FP16 pair),
    // SwiGLU = VEC 链, Combine = VEC tile (详见本文件四a 共享段)
    // 16 伪核全量循环: 与线程数解耦 (单线程/多线程均覆盖全部伪核, 结果幂等)
    // ceil 分片: bs < 16 核时每核至多 1 token, 尾部核空转
    const uint32_t perCore = (tilingData.bs + kBlockAivNum - 1U) / kBlockAivNum;
    {
        float* const combineBase = reinterpret_cast<float*>(
            g_mmWorkspace + combineOffset);
        for (uint32_t lc = 0U; lc < kBlockAivNum; ++lc) {
            const uint32_t coreIdx = lc;
            for (uint32_t i = 0; i < perCore; ++i) {
                const uint32_t token = coreIdx * perCore + i;
                if (token >= tilingData.bs) break;  // bs < 16 核时尾核空转

                for (uint32_t kk = 0U; kk < tilingData.topK; ++kk) {
                    const uint32_t slot = token * tilingData.topK + kk;
                    const uint32_t expert =
                        static_cast<uint32_t>(g_mmTopkIds[slot]);
                    const float weight = g_mmTopkWeights[slot];
                    mx_token_compute(mx,
                                     xIn + token * tilingData.h,
                                     tilingData.h, tilingData.hiddenDim,
                                     tilingData.moeExpertPerRank, expert,
                                     weight, kk,
                                     combineBase + token * tilingData.h);
                }
            }
        }
    }
    // UnpermuteTokens (9163): combine 缓冲按 token 序写回 y (VEC tile 拷贝)
    {
        const float* combine = reinterpret_cast<const float*>(g_mmWorkspace + combineOffset);
        for (uint32_t t = 0; t < tilingData.bs; ++t) {
            mx_copy_row_tile(combine + t * tilingData.h,
                             yOut + t * tilingData.h, tilingData.h);
        }
    }

    // ---- 阶段 4: 跨 rank 同步 (CrossRankSyncInWorldSize, 自回环本地退化) 与统计导出 ----
    // 源: 阶段 3 等待 Combine 发送完成再 Unpermute; epWorldSize=1 退化为本卡操作
    // 统计导出 (幂等): gfrun 多 PE 共享栈/GM, 任何 RMW 累加 (栈槽/数组 "+=")
    // 会被 N 个 PE 重复执行 ×N; 改为逐槽寄存器计数 + 一次性赋值
    {
        int32_t* stats = reinterpret_cast<int32_t*>(g_mmWorkspace + statsOffset);
        for (uint32_t core = 0; core < kBlockAivNum; ++core) {
            for (uint32_t e = 0; e < tilingData.moeExpertPerRank; ++e) {
                uint32_t cnt = 0U;
                for (uint32_t i = 0; i < perCore; ++i) {
                    const uint32_t token = core * perCore + i;
                    if (token >= tilingData.bs) break;  // bs < 16 核时尾核空转
                    if (static_cast<uint32_t>(g_mmTopkIds[token * tilingData.topK]) == e) ++cnt;
                }
                stats[core * tilingData.moeExpertPerRank + e] = static_cast<int32_t>(cnt);
            }
        }
        // 注: volatile 阻止相邻 i64 写入被合并为 16B tile store (BLK_TSTORE v2i64)
        volatile int64_t* tokExport = tokOut;
        for (uint32_t e = 0; e < tilingData.moeExpertPerRank; ++e) {
            uint32_t cnt = 0U;
            for (uint32_t t = 0; t < tilingData.bs; ++t) {
                if (static_cast<uint32_t>(g_mmTopkIds[t * tilingData.topK]) == e) ++cnt;
            }
            tokExport[e] = static_cast<int64_t>(cnt);
        }
    }
#endif
}

}  // namespace mega_moe\n