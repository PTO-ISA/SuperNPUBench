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
 *                       kHcclMaxRankSize(1024) ... (kAicNum/kMaxOutputSize 取 A5 同值,
 *                       asc:574-575; BufferConfig 字段名同 A5 asc:429-450)
 *   - MegaMoeTilingData (源 451 行) 与 Mc2MoeContextHost (源 1109 行) 字段全保留
 *   - MegaMoeWave (源 9240 行起) 的阶段集以平铺 kernel 入口
 *     (mega_moe_sim_kernel) 内逐阶段注释承载 (源行号标注):
 *       Init (9508) / SendAndQuantBuffInit (9731) / QuantizeLocalTokens (3754)
 *       GatherAndSendExpertMasks (3934) / ResetDispatchWorkspace (4051)
 *       PrepareSharedExpertInput (4119) / DispatchBuffInit (9582)
 *       InitExpertTokenCountExportBuffers (9709) / ProcessMoeExpertStages (9271)
 *       ProcessGmmPipeline (9273) / ProcessGmm1Wave (9287) / ProcessGmm2Wave (9291)
 *       ProcessCombineExperts (9293) / InitTokenUnpermuteBuffers (9294)
 *       UnpermuteTokens (9163) / CrossRankSyncInWorldSize (9295) / Process (10319)
 *   - Process() 与源同构保留 #ifdef MEGA_MOE_SIM_FAKE 仿真快路径分支 (10321-10374),
 *     快速路径保留全部 tile 原语 (TLOAD/TMULS/TEXP/TADDS/TRECIP/TSTORE);
 *     #else 分支为完整真机流水: 量化 → 路由/mask → dispatch 重排 → GMM1 →
 *     SwiGLU → GMM2 → Combine 窗口 → Unpermute 加权 → 统计导出 (计算语义与
 *     gen_data.compute_golden 一致, A8W8 权重按 E4M3FN + E8M0 scale 全量解码
 *     参与计算)。
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
 *
 * [2026-10-04 修复注] gfsim (TimingSim) 默认 single-PE 下 test-finisher
 * val=0x0001 FAIL 的根因是 tokOut 统计导出走 mm_count_eq_i32 的
 * tile-store→标量回读通路, 而该模型构建的 TLSU tile-store→GM 数据通路
 * 存在缺口 (TSTORE 架构不可见, 同根因见模型 known-gap-list memory 类
 * gmov/mscatter/timg2col 微基准)。已改为标量等值计数 (语义一致, 三端
 * gfrun/gfsim/真机通用), 详见同目录 mega_moe_sim_gfsim_fix_report.md。
 * 另: 本算子 y 输出在 gfsim 下同样受模型 tile-store 缺口影响 (y=0),
 * 仅因测试数据 E8M0 scale=0x00 使 golden ~1e-113 而弱校验通过——
 * gfsim PASS ≠ tile 流水数值全验证, 功能正确性以 gfrun R2=0 为准
 * (报告 §6 详述与数据侧建议)。
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
// [C11 对齐 2026-10-13] kAicNum/kMaxOutputSize 改 A5 同值 (asc:574-575:
// kAicNum=8, kMaxOutputSize=kBS*kTopK); 两者仅进 tilingData 占位字段,
// 无逻辑消费方, 数值对齐消歧
constexpr uint32_t kAicNum = 8U;             // AI Core 数 (AIC)
constexpr uint32_t kBlockAivNum = 16U;       // AIV 核数 (GetBlockIdx()=0..15)
constexpr uint32_t kSharedExpertNum = 0U;    // 共享专家数 (0 = 无独立 dense 路径)
// [C11 对齐 2026-10-13] kAicNum/kMaxOutputSize 改 A5 同值 (asc:574-575:
// kAicNum=8, kMaxOutputSize=kBS*kTopK)。注: A5 值曾在旧逐 token mt kernel
// 上落入时序竞态窗口 (cycle≈267k Deadlock, 时序骰子类); Phase 2 dispatch
// 通路重构后实测三入口均通过, 恢复 A5 值。
constexpr uint32_t kMaxOutputSize = kMoeBs * kTopK;  // maxOutputSize (= bs*topK, A5 asc:575)
constexpr uint32_t kMaxTilesPerExpert = 1U;  // maxTilesPerExpert
constexpr uint32_t kChunkElems = 256U;       // SIM_FAKE DataCopy 单块 256 元素

// ============================================================================
// 二、Tiling / 上下文结构 (源 451 行 / 1109 行, 字段全保留)
// ============================================================================
// [C12 对齐 2026-10-13] 三结构字段名恢复 A5原名 (asc:429-450), 消除
// "数值 1:1 但语义漂移" 的歧义映射; 初始化值 {256,1,6,288} 等与 A5 main
// (asc:10653-10658) 逐字段同值同位
struct MegaMoeDispatchBufferConfig {   // {256, 1, 6, 288} (源 main 初始化, asc:10653)
    int32_t routeItemsPerBatch;   // 每 batch 的路由项数
    int32_t routeBatchCount;      // batch 数
    int32_t bufferCount;          // 拷贝缓冲环槽数
    uint32_t copyBufferBytes;     // 每槽字节数
};

struct MegaMoeSendMaskBufferConfig {   // {256,1,2,64} / {256,1,6,64} (asc:10654-10655)
    int32_t routeItemsPerBatch;
    int32_t routeBatchCount;
    int32_t bufferCount;
    uint32_t bufferBytes;
};

struct MegaMoeUnpermuteBufferConfig {  // {16, 6, 128, 128, 64, 0} (asc:10657)
    int32_t tokensPerBatch;                    // 每 batch token 数
    int32_t inputBufferCount;                  // 输入缓冲槽数
    uint32_t bf16SlotElementCount;             // bf16 槽元素数
    uint32_t fp32SlotElementCount;             // fp32 槽元素数
    uint32_t topKWeightsBufferBytes;           // topK 权重缓冲字节
    uint32_t topKWeightsConversionBufferBytes; // 权重转换缓冲字节
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
// 三-补、2^k 位级 O(1) 构造 (替换原 O(|k|) 标量连乘循环 —— 那是全算子仅存的
//   标量热点: E8M0 scale 为 0x00 时单次调用迭代 127 次, 占 kernel 标量块的大头。
//   位级构造与连乘结果在全部定义域逐位一致 (含上溢 +inf / 下溢次正规 / 全下溢 0))
// ============================================================================
MM_INLINE float mx_exp2i_f32(int32_t k)
{
    union { uint32_t u; float f; } cvt;
    if (k >= 128) {
        cvt.u = 0x7F800000u;                          // +inf (连乘同上溢)
    } else if (k >= -126) {
        cvt.u = static_cast<uint32_t>(k + 127) << 23; // 规格数域
    } else if (k >= -149) {
        cvt.u = 1u << (k + 149);                      // 次正规域 (至 2^-149)
    } else {
        cvt.u = 0u;                                   // 完全下溢 (连乘同得 0)
    }
    return cvt.f;
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
        // e ∈ [1,15] → k = e-7 ∈ [-6,8], 全在规格数域, 位级结果与连乘逐位一致
        val = (1.0f + static_cast<float>(m) / 8.0f) *
              mx_exp2i_f32(static_cast<int32_t>(e) - 7);
    }
    return s ? -val : val;
}

// E8M0: 纯指数 (偏置 127), scale = 2^(raw-127)
// [perf] 位级 O(1) 构造替换 O(|raw-127|) 连乘循环 (raw=0x00 时 127 次迭代/调用,
//       是 mx_decode_weights_tile 标量块的主要来源; 结果逐位一致)
// [修复] 删除 "MM_INLINE" 后多余的 inline 说明符, 消除 -Wduplicate-decl-specifier 告警。
MM_INLINE float fp8_e8m0_scale(uint8_t raw)
{
    return mx_exp2i_f32(static_cast<int32_t>(raw) - 127);
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

// [A1/B8 对齐 2026-10-13] wave 组常量与 metaInfo 块布局:
//   - kMoeWaveGroupMax: expert wave 批处理的组行数上界 (B8 权重复用粒度,
//     对齐 mt 版每 PE 4 伪核的行片宽度; 组内不足时由行存在性守卫跳过)
//   - metaInfo 256B 块 (A5 asc:706/716-719: INT32_PER_256B=8,
//     RANK_ID=0/TOKEN_ID=1/TOPK_INDEX=2/WEIGHT_INDEX=3)
constexpr uint32_t kMoeWaveGroupMax = 4U;
constexpr uint32_t kMetaInfoWords = 8U;
constexpr uint32_t kMetaRankId = 0U;
constexpr uint32_t kMetaTokenId = 1U;
constexpr uint32_t kMetaTopkIndex = 2U;
constexpr uint32_t kMetaWeightIndex = 3U;  // TopkWeightsPrefetch=0 (A5 激活实例) 不写

// 每 PE 私有 scratch 布局 (wave 组版): w1F16[epr*h*hd] + w2F16[epr*(hd/2)*h] +
// y1[kMoeWaveGroupMax][hd]f32 + y2f16[kMoeWaveGroupMax][hd/2] +
// y3[kMoeWaveGroupMax][h]f32 (GMM1/SwiGLU/GMM2 中间量按组分行;
// 原 xf16 单行槽由 dispatch 的 xDisp expert-major 行取代)
struct MxTileScratch {
    __half* w1F16;
    __half* w2F16;
    float* y1;      // [G][hd]
    __half* y2f16;  // [G][hd/2]
    float* y3;      // [G][h]
};

// fp16 权重副本字节数 (sim/mt: 每 PE 私有; mt_dyn: 解码分片共享)
MM_INLINE constexpr uint32_t mx_weight_copy_bytes(uint32_t epr, uint32_t h, uint32_t hd)
{
    return (epr * h * hd + epr * (hd / 2U) * h) * 2U;
}

// wave 组中间量字节数 (每 PE)
MM_INLINE constexpr uint32_t mx_wave_group_bytes(uint32_t h, uint32_t hd)
{
    return kMoeWaveGroupMax * (hd * 4U + (hd / 2U) * 2U + h * 4U);
}

// 每 PE 私有 scratch 总字节数 (sim/mt: 权重副本 + 组中间量)
MM_INLINE constexpr uint32_t mx_tile_scratch_bytes(uint32_t epr, uint32_t h, uint32_t hd)
{
    return mx_weight_copy_bytes(epr, h, hd) + mx_wave_group_bytes(h, hd);
}

MM_INLINE MxTileScratch mx_tile_scratch_at(uint8_t* base, uint32_t pe,
                                           uint32_t epr, uint32_t h, uint32_t hd)
{
    MxTileScratch s;
    uint8_t* p = base + pe * mx_tile_scratch_bytes(epr, h, hd);
    s.w1F16 = reinterpret_cast<__half*>(p);
    s.w2F16 = s.w1F16 + epr * h * hd;
    s.y1 = reinterpret_cast<float*>(
        reinterpret_cast<uint8_t*>(s.w2F16) + epr * (hd / 2U) * h * 2U);
    s.y2f16 = reinterpret_cast<__half*>(
        reinterpret_cast<uint8_t*>(s.y1) + kMoeWaveGroupMax * hd * 4U);
    s.y3 = reinterpret_cast<float*>(
        reinterpret_cast<uint8_t*>(s.y2f16) + kMoeWaveGroupMax * (hd / 2U) * 2U);
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

// ============================================================================
// 预量化辅助 (x 行 fp32 → fp16 tile 链; dispatch 搬运复用 —— 量化 store 与
// GEMM load 之间隔整个 dispatch 阶段, 2026-10-04 修复的 "预量化前置" 语义
// 由 A1 dispatch 通路天然承载: xDisp 行在 dispatch 段写、GMM 波段读)
// ============================================================================
MM_INLINE void mx_quantize_row(const float* xRow, __half* dst, uint32_t h)
{
    for (uint32_t c = 0U; c < h; c += 32U) {
        MxChainF32 xf;
        global_tensor<float, RowMajor<1, 32>> gX(
            const_cast<float*>(xRow) + c);
        TLOAD(xf, gX);
        MxChainF16 xh;
        TCVT(xh, xf);
        global_tensor<__half, RowMajor<1, 32>> gXH(dst + c);
        TSTORE(gXH, xh);
    }
}

// ============================================================================
// [A1/B8/B9 对齐 2026-10-13] expert wave 批处理核心 —— A5 ProcessGmm1Wave/
// ProcessGmm2Wave/ProcessCombineExperts (asc:10071/8902/10195) 的 wave 语义:
// 按 expert 分组连续处理, 同 expert 行批共享权重 tile 载入 (A5 L1/L0 级复用)。
// V300 原逐 token 实现每 token 重复 TLOAD 同一 expert 的全部权重 tile
// (pipe 实测 kernelB TLOAD|WS 630 次, 大量为同 expert 重复载入 —— 比对报告
// §C.1-B8: V300↔A5 kernel 段性能模型的最大失真项)。
//
// mx_expert_wave_group: expert-major 连续 rows (≤kMoeWaveGroupMax) 行的
//   GMM1 → SwiGLU → GMM2 → Combine 窗口一体计算:
//   - GMM1/GMM2 的每个 (n0,k0) 权重 tile 仅 TLOAD 一次, 组内各行依次
//     TLOAD v → TGEMV_MX(_ACC) 到各自命名累加 tile d0..d3 (行存在性由
//     运行期守卫跳过 —— 与既有 "token >= bs break" 条件 tile op 同款模式;
//     守卫条件 rows 为组内不变量, 累加链要么全在要么全无);
//   - SwiGLU 逐行 VEC 链 (六 op 与 A5 asc:10476-10499 逐一对应);
//   - Combine 窗口行 **未加权** 直写 (B9: 权重施加点移至 Unpermute 段,
//     对齐 A5 AccumulateMoeExpertsForToken asc:9060-9076 的 Muls/Add 位置);
//     slot = metaInfo[row].TOKEN*topK + TOPK (A5 GMM2 行发往 combine 窗口
//     的行号公式, asc:7742)。
// 数值与原逐 token 路径逐位一致 (A 侧同为 fp16 预量化行, GEMV/SwiGLU 链
// 逐 op 同序; 权重乘从 Combine 移至 Unpermute, 单乘位置交换 bit 等价)。
// ============================================================================
#define MM_WAVE_VROW(G)                                                       \
    (((rowBegin + (G)) < rowEnd) ? (xDisp + (rowBegin + (G)) * h) : padX)
#define MM_WAVE_MROW(G)                                                       \
    (((rowBegin + (G)) < rowEnd)                                              \
         ? (metaInfo + (rowBegin + (G)) * kMetaInfoWords)                     \
         : (metaInfo + slots * kMetaInfoWords))
#define MM_WAVE_FIRST(DST, VPTR)                                              \
    {                                                                         \
        MxGemvVec v;                                                          \
        global_tensor<__half, RowMajor<1, 32>> gV(const_cast<__half*>(VPTR)); \
        TLOAD_CUBE(v, gV);                                                    \
        TGEMV_MX(DST, m0, v, fixp::keep_acc());                               \
    }
#define MM_WAVE_ACC(DST, VPTR)                                                \
    {                                                                         \
        MxGemvVec v;                                                          \
        global_tensor<__half, RowMajor<1, 32>> gV(const_cast<__half*>(VPTR)); \
        TLOAD_CUBE(v, gV);                                                    \
        TGEMV_MX_ACC(DST, DST, m, v, fixp::keep_acc());                       \
    }
#define MM_WAVE_STORE(DST, YBASE)                                             \
    {                                                                         \
        global_tensor<float, RowMajor<1, 32>> gY(YBASE);                      \
        TSTORE_CUBE(gY, DST);                                                 \
    }

// rowBegin/rowEnd: expert 段 (或 PE 行片 ∩ expert 段) 的有效行区间;
// 组槽位 [rowBegin, rowBegin+4) 中越过 rowEnd 的槽位为 **pad 行**:
// 读 xDisp[slots] 零行 (bss 零初始化) 与 metaInfo[slots] pad 元数据
// (dispatch 期写 {TOKEN=bs, TOPK=topK-1} → combine 写 dummy 尾行,
// Unpermute 永不读取)。全部 tile op 无条件执行 —— 条件 tile op 的
// 跨分支 tile PHI 会让后端生成 TCOPY (tile 寄存器搬运), 其编码路径
// 即 moe_combine 已归档的 TMOV/0.58.4 B.DATR 保留字缺陷 (实测 gfrun
// "reserved/deleted tile selector" 中止), pad 行方案从源头消除。
//
// [2026-10-14 优化] 模板化组大小 G (1..kMoeWaveGroupMax): 槽位数成为
// 编译期常量 (if constexpr 逐槽展开), 供 mx_expert_wave_run 按剩余行数
// 自适应选择组大小 —— 消除尾组越段 pad 槽 (cfgB bs=18 的 PE 行片 ∩
// expert 段边界, pipe v5 实测 10 个 pad 槽 ×2 幂等双遍的 SwiGLU/GEMV/
// Combine 全链浪费, 参见 mega_moe_pipeline_analysis.md §5-#6)。G 为
// 模板参数 → 每个实例内全部 tile op 仍无条件执行 (无跨分支 tile PHI →
// 无 TCOPY); G=4 实例与原固定 4 槽版逐 op 等价。
template <uint32_t G>
MM_INLINE void mx_expert_wave_group(const MxTileScratch& s,
                                    const __half* xDisp, uint32_t rowBegin,
                                    uint32_t rowEnd, uint32_t slots,
                                    uint32_t h, uint32_t hd, uint32_t expert,
                                    const int32_t* metaInfo, float* combineWin,
                                    uint32_t topK)
{
    static_assert(G >= 1U && G <= kMoeWaveGroupMax,
                  "wave group size must be in [1, kMoeWaveGroupMax]");
    const __half* const padX = xDisp + slots * h;   // pad 零行 (安全网, 自适应分派下不触达)

    // ---- GMM1 (源 ProcessGmm1Wave): y1[g][n] = Σ_k xDisp[row][k]·w1[e][k][n]
    // (k=0 块剥离循环外: 消除运行期 TGEMV_MX/ACC 分支的 tile 跨分支 PHI;
    //  槽位展开为编译期 if constexpr —— 实例内无运行期分支)
    for (uint32_t n0 = 0U; n0 < hd; n0 += 32U) {
        MxGemvMtx m0;
        global_tensor<__half, RowMajor<-1, -1>> gM0(
            s.w1F16 + expert * h * hd + n0, 32, hd);
        TLOAD_CUBE(m0, gM0);
        [[maybe_unused]] MxGemvDst d0, d1, d2, d3;
        if constexpr (G > 0U) { MM_WAVE_FIRST(d0, MM_WAVE_VROW(0)) }
        if constexpr (G > 1U) { MM_WAVE_FIRST(d1, MM_WAVE_VROW(1)) }
        if constexpr (G > 2U) { MM_WAVE_FIRST(d2, MM_WAVE_VROW(2)) }
        if constexpr (G > 3U) { MM_WAVE_FIRST(d3, MM_WAVE_VROW(3)) }
        for (uint32_t k0 = 32U; k0 < h; k0 += 32U) {
            MxGemvMtx m;
            global_tensor<__half, RowMajor<-1, -1>> gM(
                s.w1F16 + expert * h * hd + k0 * hd + n0, 32, hd);
            TLOAD_CUBE(m, gM);
            if constexpr (G > 0U) { MM_WAVE_ACC(d0, MM_WAVE_VROW(0) + k0) }
            if constexpr (G > 1U) { MM_WAVE_ACC(d1, MM_WAVE_VROW(1) + k0) }
            if constexpr (G > 2U) { MM_WAVE_ACC(d2, MM_WAVE_VROW(2) + k0) }
            if constexpr (G > 3U) { MM_WAVE_ACC(d3, MM_WAVE_VROW(3) + k0) }
        }
        if constexpr (G > 0U) { MM_WAVE_STORE(d0, s.y1 + 0U * hd + n0) }
        if constexpr (G > 1U) { MM_WAVE_STORE(d1, s.y1 + 1U * hd + n0) }
        if constexpr (G > 2U) { MM_WAVE_STORE(d2, s.y1 + 2U * hd + n0) }
        if constexpr (G > 3U) { MM_WAVE_STORE(d3, s.y1 + 3U * hd + n0) }
    }

    // ---- SwiGLU (源 Gmm1SwigluState): y2 = silu(y1[:hd/2]) * y1[hd/2:] (VEC 链)
    // (组内 G 槽全量处理; pad 行输入为零 → 输出零, 写 dummy 行无害)
    for (uint32_t g = 0U; g < G; ++g) {
        const float* y1g = s.y1 + g * hd;
        __half* y2g = s.y2f16 + g * (hd / 2U);
        for (uint32_t c = 0U; c < hd / 2U; c += 32U) {
            MxChainF32 a, b;
            global_tensor<float, RowMajor<1, 32>> gA(
                const_cast<float*>(y1g) + c);
            TLOAD(a, gA);
            global_tensor<float, RowMajor<1, 32>> gB(
                const_cast<float*>(y1g) + hd / 2U + c);
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
            global_tensor<__half, RowMajor<1, 32>> gY2(y2g + c);
            TSTORE(gY2, h16);
        }
    }

    // ---- GMM2 (源 ProcessGmm2Wave): y3[g][n] = Σ_k y2[g][k]·w2[e][k][n]
    for (uint32_t n0 = 0U; n0 < h; n0 += 32U) {
        MxGemvMtx m0;
        global_tensor<__half, RowMajor<-1, -1>> gM0(
            s.w2F16 + expert * (hd / 2U) * h + n0, 32, h);
        TLOAD_CUBE(m0, gM0);
        [[maybe_unused]] MxGemvDst d0, d1, d2, d3;
        if constexpr (G > 0U) { MM_WAVE_FIRST(d0, s.y2f16 + 0U * (hd / 2U)) }
        if constexpr (G > 1U) { MM_WAVE_FIRST(d1, s.y2f16 + 1U * (hd / 2U)) }
        if constexpr (G > 2U) { MM_WAVE_FIRST(d2, s.y2f16 + 2U * (hd / 2U)) }
        if constexpr (G > 3U) { MM_WAVE_FIRST(d3, s.y2f16 + 3U * (hd / 2U)) }
        for (uint32_t k0 = 32U; k0 < hd / 2U; k0 += 32U) {
            MxGemvMtx m;
            global_tensor<__half, RowMajor<-1, -1>> gM(
                s.w2F16 + expert * (hd / 2U) * h + k0 * h + n0, 32, h);
            TLOAD_CUBE(m, gM);
            if constexpr (G > 0U) { MM_WAVE_ACC(d0, s.y2f16 + 0U * (hd / 2U) + k0) }
            if constexpr (G > 1U) { MM_WAVE_ACC(d1, s.y2f16 + 1U * (hd / 2U) + k0) }
            if constexpr (G > 2U) { MM_WAVE_ACC(d2, s.y2f16 + 2U * (hd / 2U) + k0) }
            if constexpr (G > 3U) { MM_WAVE_ACC(d3, s.y2f16 + 3U * (hd / 2U) + k0) }
        }
        if constexpr (G > 0U) { MM_WAVE_STORE(d0, s.y3 + 0U * h + n0) }
        if constexpr (G > 1U) { MM_WAVE_STORE(d1, s.y3 + 1U * h + n0) }
        if constexpr (G > 2U) { MM_WAVE_STORE(d2, s.y3 + 2U * h + n0) }
        if constexpr (G > 3U) { MM_WAVE_STORE(d3, s.y3 + 3U * h + n0) }
    }

    // ---- Combine 窗口 (源 ProcessCombineExperts/RunWaveExpertCombineStage):
    // GMM2 未加权行发往 combine 窗口 slot = TOKEN*topK + TOPK (B9);
    // pad 行经 metaInfo[slots] (TOKEN=bs, TOPK=topK-1) 落 dummy 尾行
    {
        using MxFullF32 = Tile<Location::Vec, float, 1, 32, BLayout::RowMajor>;
        for (uint32_t g = 0U; g < G; ++g) {
            const int32_t* meta = MM_WAVE_MROW(g);
            const uint32_t slot =
                static_cast<uint32_t>(meta[kMetaTokenId]) * topK +
                static_cast<uint32_t>(meta[kMetaTopkIndex]);
            float* dst = combineWin + slot * h;
            const float* src = s.y3 + g * h;
            for (uint32_t c = 0U; c < h; c += 32U) {
                MxFullF32 t;
                global_tensor<float, RowMajor<1, 32>> gS(
                    const_cast<float*>(src) + c);
                TLOAD(t, gS);
                global_tensor<float, RowMajor<1, 32>> gD(dst + c);
                TSTORE(gD, t);
            }
        }
    }
}
#undef MM_WAVE_VROW
#undef MM_WAVE_MROW
#undef MM_WAVE_FIRST
#undef MM_WAVE_ACC
#undef MM_WAVE_STORE

// ============================================================================
// [2026-10-14 优化] wave 组自适应分派 (替代固定 kMoeWaveGroupMax 步进的组
// 循环): 满组 (rem ≥ 4) 走 G=4 (与原固定组逐 op 等价, B8 权重复用粒度
// 不变); 尾组 rem < 4 时按剩余行数选 G=rem —— 组槽位与有效行恰好重合,
// 越段 pad 槽 (cfgB bs=18: PE 行片 ∩ expert 段边界, 10 个 pad 槽 ×2 双遍)
// 全部消除。数值不变: 有效行的 GEMV/SwiGLU/Combine op 序列与固定组版
// 完全一致, 被消除的仅是 pad 槽的无效计算 (其 combine 写 dummy 尾行
// 本就永不读取); 函数选择为运行期标量分支, 各实例内 tile op 无条件
// (无跨分支 tile PHI → 无 TCOPY)。
// ============================================================================
MM_INLINE void mx_expert_wave_run(const MxTileScratch& s,
                                  const __half* xDisp, uint32_t rowBegin,
                                  uint32_t rowEnd, uint32_t slots,
                                  uint32_t h, uint32_t hd, uint32_t expert,
                                  const int32_t* metaInfo, float* combineWin,
                                  uint32_t topK)
{
    uint32_t r0 = rowBegin;
    while (r0 < rowEnd) {
        const uint32_t rem = rowEnd - r0;
        if (rem >= kMoeWaveGroupMax) {
            mx_expert_wave_group<kMoeWaveGroupMax>(
                s, xDisp, r0, rowEnd, slots, h, hd, expert, metaInfo,
                combineWin, topK);
            r0 += kMoeWaveGroupMax;
        } else if (rem >= 3U) {
            mx_expert_wave_group<3U>(
                s, xDisp, r0, rowEnd, slots, h, hd, expert, metaInfo,
                combineWin, topK);
            r0 += 3U;
        } else if (rem >= 2U) {
            mx_expert_wave_group<2U>(
                s, xDisp, r0, rowEnd, slots, h, hd, expert, metaInfo,
                combineWin, topK);
            r0 += 2U;
        } else {
            mx_expert_wave_group<1U>(
                s, xDisp, r0, rowEnd, slots, h, hd, expert, metaInfo,
                combineWin, topK);
            r0 += 1U;
        }
    }
}

// VEC tile 加权反排列 (A5 UnpermuteTokens asc:9178 + AccumulateMoeExpertsFor-
// Token asc:9040-9076): y[token] = Σ_kk weight·combineWin[token*topK+kk]
// (kk==0: Muls 直写; kk>0: Muls+Add 累加 —— 权重施加点与 A5 同位; 权重读
// topk_weights GM, A5 激活实例 TopkWeightsPrefetch=0 同款; B9: combine 窗口
// 行未加权)。全宽 [1×32] 静态块 tile (无 ValidRow 掩码不泄漏, [C6] 规避)。
MM_INLINE void mx_unpermute_token(const float* combineWin,
                                  const float* topkWeights, float* yOut,
                                  uint32_t token, uint32_t topK, uint32_t h)
{
    using MxFullF32 = Tile<Location::Vec, float, 1, 32, BLayout::RowMajor>;
    for (uint32_t kk = 0U; kk < topK; ++kk) {
        const float weight = topkWeights[token * topK + kk];
        const float* src = combineWin + (token * topK + kk) * h;
        for (uint32_t c = 0U; c < h; c += 32U) {
            MxFullF32 t;
            global_tensor<float, RowMajor<1, 32>> gS(
                const_cast<float*>(src) + c);
            TLOAD(t, gS);
            TMULS(t, t, weight);
            global_tensor<float, RowMajor<1, 32>> gD(yOut + token * h + c);
            if (kk == 0U) {
                TSTORE(gD, t);
            } else {
                MxFullF32 acc;
                TLOAD(acc, gD);
                TADD(acc, acc, t);
                TSTORE(gD, acc);
            }
        }
    }
}

// ============================================================================
// [A1/A2 对齐 2026-10-13] dispatch 计划 (标量; slots ≤ 18 低于 tile 粒度):
//   - expertCnt: mask 行求和 (A2: mask 表写后有读 —— A5 GatherAndSendExpert-
//     MaskBatch 的匹配计数语义; (token,expert) 唯一契约下与 slot 计数一致,
//     测试数据 topK=1 恒满足);
//   - starts: 独占前缀 (A5 cumsumInfo 行偏移);
//   - dstPos: 稳定序 (A5 接收侧 dstPosition = Σ_{e'<e}count(e') +
//     #{j<i: eid(j)=e}, asc:2294-2295; 与 moe_dispatch_v2 的成对 rank 同一
//     公式, 比对报告 §4.1 数学证明级等价)。
// ============================================================================
constexpr uint32_t kMoeEprMax = 16U;   // dispatch 标量栈数组上界 (运行期契约)

MM_INLINE void mm_dispatch_plan(const uint8_t* mask, const int32_t* topkIds,
                                uint32_t bs, uint32_t slots, uint32_t epr,
                                uint32_t* expertCnt, uint32_t* starts,
                                uint32_t* dstPos)
{
    for (uint32_t e = 0U; e < epr; ++e) {
        uint32_t cnt = 0U;
        for (uint32_t t = 0U; t < bs; ++t) {
            cnt += mask[e * bs + t];              // A2: mask 行求和 (写后有读)
        }
        expertCnt[e] = cnt;
    }
    uint32_t acc = 0U;
    for (uint32_t e = 0U; e < epr; ++e) {
        starts[e] = acc;
        acc += expertCnt[e];
    }
    for (uint32_t s = 0U; s < slots; ++s) {
        const uint32_t e = static_cast<uint32_t>(topkIds[s]);
        if (e >= epr) {
            dstPos[s] = 0xFFFFFFFFU;              // 非法 eid: 哨兵 (copy 期跳过)
            continue;
        }
        uint32_t rank = 0U;
        for (uint32_t j = 0U; j < s; ++j) {
            if (static_cast<uint32_t>(topkIds[j]) == e) ++rank;
        }
        dstPos[s] = starts[e] + rank;
    }
}

// ============================================================================
// [A1 对齐 2026-10-13] workspace 布局 (dispatch 通路版; sim/mt/mt_dyn 三入口
// 与 driver 尺寸计算共用):
//   dstPos 表 [slots] f32     — A5 接收侧 dstPosition 载体 (-1 复位→写→读)
//   mask 表 [epr*bs] u8       — A5 bit-mask 本地退化 (scatter 写→计数读)
//   expertCnt [epr] i32       — A5 expertRevTokenNums
//   starts [epr] i32          — A5 cumsumInfo 独占前缀 (GMM 波段读)
//   xDisp [slots][h] f16      — A5 dispatchRevData (量化 token expert-major 行)
//   scaleDisp [slots] u8      — A5 dispatchRevScale (E8M0; 退化 0x00=1.0)
//   metaInfo [slots][8] i32   — A5 metaInfo 256B 块 (RANK/TOKEN/TOPK/WEIGHT)
//   combine [slots][h] f32    — A5 combine 窗口 (GMM2 未加权行, slot 序)
//   stats [16][epr] i32       — A5 核内统计 (dispatch 期累计, 导出求和)
//   scratch 起 — 每 PE MX scratch (调用方布局: sim/mt 4 PE 全私有;
//   mt_dyn 共享权重副本 + 4 PE 组中间量)
// ============================================================================
struct MoeWsLayout {
    uint32_t dstPos;
    uint32_t mask;
    uint32_t cnt;
    uint32_t starts;
    uint32_t xDisp;
    uint32_t scaleDisp;
    uint32_t metaInfo;
    uint32_t combine;
    uint32_t stats;
    uint32_t scratch;
    uint32_t total;
};

MM_INLINE constexpr MoeWsLayout moe_ws_layout(uint32_t bs, uint32_t h, uint32_t hd,
                                    uint32_t epr, uint32_t topK)
{
    const uint32_t slots = bs * topK;
    MoeWsLayout L;
    L.dstPos = 0U;
    L.mask = L.dstPos + slots * 4U;
    L.cnt = L.mask + epr * bs;
    L.starts = L.cnt + epr * 4U;
    L.xDisp = L.starts + epr * 4U;                  // +1 pad 零行 (wave 组)
    L.scaleDisp = L.xDisp + (slots + 1U) * h * 2U;
    L.metaInfo = L.scaleDisp + slots;               // +1 pad 元数据行
    L.combine = L.metaInfo + (slots + 1U) * kMetaInfoWords * 4U;
    L.stats = L.combine + (slots + topK) * h * 4U;  // +topK dummy 尾行
    L.scratch = L.stats + kBlockAivNum * epr * 4U;
    L.total = L.scratch;   // scratch 区由调用方/driver 追加 (4 PE scratch + 余量)
    return L;
}

// ============================================================================
// 四b、控制表 tile 化工具 (sim / sim_mt / sim_mt_dyn 三入口共用)
//
// 契约 (tile_probe 探针 gfrun 实证, 详注见 group_token_vec/gt_tile_common.hpp):
//   [C2] MSCATTER_ADD 须数字编码 BSTART.TLSU 21; index = 字节位移;
//        tile 内重复下标 row-major 顺序 RMW (确定序)。
//   [C4] 原子族/标量 TEPL 一律 [1×N] 行 tile (物理 Cols>=2 → lb0/lb2 存在);
//        归约输出 [32×1,vN×1] 只可 TSTORE。
//   [C6] 动态 ValidRow 链泄漏 lane → 运行时段长一律 32/16/8/4 静态块二元
//        分解 + <4 元素标量尾 (低于 128B tile 粒度)。
//   [C9] 小数组标量初始化 volatile 逐元素 (16B 合并 store 写丢失规避)。
// ============================================================================

// int32 段填充 (stats 清零等): 32/8/4 静态块 + <4 volatile 标量尾
MM_INLINE void mm_fill_i32(int32_t* dst, uint32_t len, int32_t value)
{
    using T32 = Tile<Location::Vec, int32_t, 1, 32, BLayout::RowMajor>;
    using T8  = Tile<Location::Vec, int32_t, 1, 32, BLayout::RowMajor, 1, 8>;
    using T4  = Tile<Location::Vec, int32_t, 1, 32, BLayout::RowMajor, 1, 4>;
    uint32_t done = 0U;
    for (; done + 32U <= len; done += 32U) {
        T32 z; TEXPANDS(z, value);
        global_tensor<int32_t, RowMajor<1, 32>> g(dst + done);
        TSTORE(g, z);
    }
    for (; done + 8U <= len; done += 8U) {
        T8 z; TEXPANDS(z, value);
        global_tensor<int32_t, RowMajor<1, 8>> g(dst + done);
        TSTORE(g, z);
    }
    for (; done + 4U <= len; done += 4U) {
        T4 z; TEXPANDS(z, value);
        global_tensor<int32_t, RowMajor<1, 4>> g(dst + done);
        TSTORE(g, z);
    }
    volatile int32_t* v = dst;
    for (; done < len; ++done) v[done] = value;   // <4 元素, 低于 tile 粒度
}

// fp32 段填充 (dispatch 表 -1.0f 等): 同款分解
MM_INLINE void mm_fill_f32(float* dst, uint32_t len, float value)
{
    using T32 = Tile<Location::Vec, float, 1, 32, BLayout::RowMajor>;
    using T8  = Tile<Location::Vec, float, 1, 32, BLayout::RowMajor, 1, 8>;
    using T4  = Tile<Location::Vec, float, 1, 32, BLayout::RowMajor, 1, 4>;
    uint32_t done = 0U;
    for (; done + 32U <= len; done += 32U) {
        T32 z; TEXPANDS(z, value);
        global_tensor<float, RowMajor<1, 32>> g(dst + done);
        TSTORE(g, z);
    }
    for (; done + 8U <= len; done += 8U) {
        T8 z; TEXPANDS(z, value);
        global_tensor<float, RowMajor<1, 8>> g(dst + done);
        TSTORE(g, z);
    }
    for (; done + 4U <= len; done += 4U) {
        T4 z; TEXPANDS(z, value);
        global_tensor<float, RowMajor<1, 4>> g(dst + done);
        TSTORE(g, z);
    }
    volatile float* v = dst;
    for (; done < len; ++done) v[done] = value;   // <4 元素, 低于 tile 粒度
}

// mask 表散射: mask[topkIds[slot]*bs + slot/topK] = 1, slot ∈ [begin, begin+n)
// 链: TLOAD(ids) + TMULS(×bs) + TCI(slot ramp) + TADD → idx (u8 元素下标 =
// 字节位移, ×1) → MSCATTER(val=1 u8)。32/16/8/4 静态块 + <4 标量尾。
// 契约: topK == 1 (kTopK 常量; token = slot), 违例整体标量兜底。
MM_INLINE void mm_mask_scatter(const int32_t* topkIds, uint8_t* mask,
                                       uint32_t slotBegin, uint32_t nSlots,
                                       uint32_t bs, uint32_t topK)
{
    if (topK != 1U) {   // 通用 topK: slot/topK 需 TDIVS(int32 未探针) → 标量
        for (uint32_t s = slotBegin; s < slotBegin + nSlots; ++s) {
            const int32_t expert = topkIds[s];
            mask[static_cast<uint32_t>(expert) * bs + s / topK] = 1U;
        }
        return;
    }
    using TI32 = Tile<Location::Vec, int32_t, 1, 32, BLayout::RowMajor>;
    using TI16 = Tile<Location::Vec, int32_t, 1, 32, BLayout::RowMajor, 1, 16>;
    using TI8  = Tile<Location::Vec, int32_t, 1, 32, BLayout::RowMajor, 1, 8>;
    using TI4  = Tile<Location::Vec, int32_t, 1, 32, BLayout::RowMajor, 1, 4>;
    using TU32 = Tile<Location::Vec, uint8_t, 1, 128, BLayout::RowMajor, 1, 32>;
    using TU16 = Tile<Location::Vec, uint8_t, 1, 128, BLayout::RowMajor, 1, 16>;
    using TU8  = Tile<Location::Vec, uint8_t, 1, 128, BLayout::RowMajor, 1, 8>;
    using TU4  = Tile<Location::Vec, uint8_t, 1, 128, BLayout::RowMajor, 1, 4>;
    global_tensor<uint8_t, RowMajor<-1, -1>> gMask(
        mask, static_cast<int>(nSlots), 1);

    uint32_t done = 0U;
    // 二元分解宏 (C = 块宽): idx = ids*bs + slotRamp → MSCATTER(1u8)
#define MM_MASK_CHUNK(C, TIdx, TVal)                                          \
    for (; done + (C) <= nSlots; done += (C)) {                               \
        TIdx ids;                                                             \
        global_tensor<int32_t, RowMajor<-1, -1>> gIds(                        \
            const_cast<int32_t*>(topkIds) + slotBegin + done,                 \
            1, static_cast<int>(C));                                          \
        TLOAD(ids, gIds);                                                     \
        TIdx eb;                                                              \
        TMULS(eb, ids, static_cast<int32_t>(bs));                             \
        TIdx ramp;                                                            \
        TCI(ramp, static_cast<int32_t>(slotBegin + done));                    \
        TIdx idx;                                                             \
        TADD(idx, eb, ramp);                                                  \
        TVal val;                                                             \
        TEXPANDS(val, static_cast<uint8_t>(1));                               \
        MSCATTER(gMask, val, idx);                                            \
    }
    MM_MASK_CHUNK(32U, TI32, TU32)
    MM_MASK_CHUNK(16U, TI16, TU16)
    MM_MASK_CHUNK(8U,  TI8,  TU8)
    MM_MASK_CHUNK(4U,  TI4,  TU4)
#undef MM_MASK_CHUNK
    for (; done < nSlots; ++done) {   // <4 slot, 低于 tile 粒度
        const uint32_t s = slotBegin + done;
        const int32_t expert = topkIds[s];
        mask[static_cast<uint32_t>(expert) * bs + s] = 1U;
    }
}


// ============================================================================
// 五、MegaMoeWave — 源类 (9240 行起) 阶段迁移说明
// ============================================================================
// [C10 清理 2026-10-13] 原此处保留的 MegaMoeWave 模板类 (与源 asc:9240 起
// 类同名同阶段集) 为**从未实例化的死代码**, 且其中 12 个函数与 A5 同名异义
// (V300 类内版本为标量三重循环/fp32 解码等旧形态, 与实际执行的 tile 路径
// 不一致), 类 Init 的 fp32 权重缓存布局 (w1F32Offset_ 等) 同样不再对应
// 任何执行路径 —— 按比对报告 §C.1-C10 整体删除, 防止按函数名逐行等价
// 阅读的误读。A5 阶段 ↔ V300 执行路径的对应关系以下方平铺 kernel 入口
// (mega_moe_sim_kernel) 内的逐阶段注释为准 (源行号标注)。

// ============================================================================
// 六、kernel 入口 (与源 MegaMoeSelfLoopbackKernel 对应)
// ============================================================================
// 说明: 入口采用平铺顺序执行: 实测 gfrun 功能仿真器对
// "类对象 + 成员函数调用链" 的返回地址 (FRET/BARG) 模拟存在缺陷
// (仓库既有限制, moe_dispatch_v2 等自校验用例同因 R2=1), 平铺后
// 同一逻辑以单函数形态执行, 规避该缺陷且不省略任何阶段。
// [C10 清理 2026-10-13] 原 "上方 MegaMoeWave 类" 死代码已删除 (见五节注),
// 阶段对应关系以本函数内逐阶段源行号注释为准。
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

    // workspace 布局 (A1 dispatch 通路版, 四d 节 MoeWsLayout; driver 的
    // kWorkspaceBytes 以 moe_ws_layout 同式计算)
    const MoeWsLayout WS = moe_ws_layout(tilingData.bs, tilingData.h,
                                         tilingData.hiddenDim,
                                         tilingData.moeExpertPerRank,
                                         tilingData.topK);
    const uint32_t slots = tilingData.bs * tilingData.topK;

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
    // ============ 完整真机流水 (各阶段与 A5 wave 流水一一对应) ============
    // [A1/A2/A3/B8/B9/A6 对齐 2026-10-13] 补全 dispatch token 排列-重组通路
    // (A5 CopyTokensAndMetaForDispatch(4332) + DispatchExpertTokens(4447) +
    // ComputeExpertTokenCountAndNotify(4540) + RunMoeExpertDispatchStage(4678)):
    // mask 计数 → cumsum 行偏移 → 量化 token+scale 搬运到 expert-major 排列 →
    // metaInfo(RANK/TOKEN/TOPK) → GMM 按 expert 波处理 (B8 权重复用) →
    // combine 窗口未加权行 (B9) → Unpermute 加权反排列 (B9)。
    // topK=1 自回环下最终 y/tokOut 与原逐 token 实现逐位一致 (golden 双精度
    // 验证); 单程序幂等模型: 全部阶段与线程数解耦 (每 PE 覆盖全量, 同值写)。
    // ---- 阶段 1: 输入准备 ----
    // SendAndQuantBuffInit (9731): 统计槽清零 (复位语义; dispatch 期被
    // 逐核计数覆写) — tile 填充
    {
        int32_t* stats = reinterpret_cast<int32_t*>(g_mmWorkspace + WS.stats);
        mm_fill_i32(stats, kBlockAivNum * tilingData.moeExpertPerRank, 0);
    }
    // QuantizeLocalTokens (3754) — MX 路径: 每 PE 私有 tile 化解码
    // (TLOAD(E4M3[32,32]) → TCVT(fp16) → TMULS(k 组 scale 折叠) → TSTORE);
    // 幂等冗余执行模型下每 PE 写自己的 scratch 副本 (同值, 写不相交)
    static_assert(kHiddenDim % 32 == 0, "h must be a multiple of the 32-wide MX K-tile");
    static_assert(kMoeHiddenDim % 64 == 0, "hiddenDim/2 must cover whole MX scale groups");
    const uint32_t mxTid = get_thread_idx();
    MxTileScratch mx = mx_tile_scratch_at(
        g_mmWorkspace + WS.scratch, mxTid,
        tilingData.moeExpertPerRank, tilingData.h, tilingData.hiddenDim);
    mx_decode_weights_tile(mx, tilingData.moeExpertPerRank,
                           tilingData.h, tilingData.hiddenDim);
    // GatherAndSendExpertMasks (3934): 自回环本地 mask 表 — tile 散射
    // (mm_mask_scatter: TLOAD ids + TMULS(×bs) + TCI ramp + TADD → MSCATTER
    // u8; topK==1 契约 (kTopK 常量), 违例 helper 内标量兜底)
    {
        uint8_t* mask = g_mmWorkspace + WS.mask;
        mm_mask_scatter(g_mmTopkIds, mask, 0U, slots,
                        tilingData.bs, tilingData.topK);
    }
    // ResetDispatchWorkspace (4051) = DispatchBuffInit (9582): dstPos 表 -1
    // 复位 (A5 dispatch 工作区 flag 清零语义的载体; dispatch 期被真实
    // dstPosition 覆写 — 表写后有读, 原占位语义消除, A3)
    {
        float* dstPosTab = reinterpret_cast<float*>(g_mmWorkspace + WS.dstPos);
        mm_fill_f32(dstPosTab, slots, -1.0f);
    }

    // ---- 阶段 2: 共享专家输入准备 (源 PrepareSharedExpertInput, sharedExpertNum==0 跳过) ----

    // ---- 阶段 2.5: dispatch 计划与数据搬运 (A1/A2/A6) ----
    // ComputeExpertTokenCountAndNotify (4540): expertRevNums 计数 (A2: mask
    // 行求和, 写后有读) + cumsumInfo 独占前缀 (starts) + 稳定序 dstPos
    {
        uint32_t expertCnt[kMoeEprMax];
        uint32_t startsArr[kMoeEprMax];
        uint32_t dstPosArr[kBatchSize * kTopK];
        static_assert(kMoeExpertPerRank <= kMoeEprMax, "epr exceeds scalar plan bound");
        mm_dispatch_plan(g_mmWorkspace + WS.mask, g_mmTopkIds,
                         tilingData.bs, slots, tilingData.moeExpertPerRank,
                         expertCnt, startsArr, dstPosArr);
        // workspace 镜像 (A5 expertRevTokenNums/cumsumInfo 载体; GMM 波段
        // 读 starts): 幂等同值写 (tilingData 填充同模式)
        {
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
    // CopyTokensAndMetaForDispatch (4332) + DispatchExpertTokens (4447):
    // 量化 token+scale 搬运到 expert-major 行 (dispatchRevData/dispatchRevScale)
    // + metaInfo(RANK/TOKEN/TOPK) 256B 块 (FillTriple/Store 源 4286-4324;
    // WEIGHT 仅 TopkWeightsPrefetch 写, A5 激活实例=0 不写, 零余量)
    {
        __half* xDisp = reinterpret_cast<__half*>(g_mmWorkspace + WS.xDisp);
        uint8_t* scaleDisp = g_mmWorkspace + WS.scaleDisp;
        int32_t* meta = reinterpret_cast<int32_t*>(g_mmWorkspace + WS.metaInfo);
        const float* dstTab = reinterpret_cast<const float*>(
            g_mmWorkspace + WS.dstPos);
        for (uint32_t s = 0U; s < slots; ++s) {
            const int32_t e = g_mmTopkIds[s];
            if (e < 0 || static_cast<uint32_t>(e) >= tilingData.moeExpertPerRank) {
                continue;   // 非法 eid 守卫 (A5 dispatch eid∈[0,E) 同款)
            }
            const float dstV = dstTab[s];   // GM 读回 (dstPos 表活跃消费)
            const uint32_t row = static_cast<uint32_t>(dstV);
            const uint32_t token = s / tilingData.topK;
            const uint32_t kk = s % tilingData.topK;
            // 量化 token 行 fp32→fp16 (MX A 侧 fp16 免 scale 的文档化退化,
            // D14 注; A5 为 BF16→E4M3+E8M0 写 peermem 窗口)
            mx_quantize_row(xIn + token * tilingData.h,
                            xDisp + row * tilingData.h, tilingData.h);
            // scale 行 (A5 dispatchRevScale): E8M0 0x00 = 1.0
            volatile uint8_t* sc = scaleDisp + row;
            *sc = 0x00U;
            // metaInfo 256B 块 ([C9] 小数组 volatile 逐元素写)
            volatile int32_t* mrow = meta + row * kMetaInfoWords;
            mrow[kMetaRankId] = 0;                          // 自回环 rank 0
            mrow[kMetaTokenId] = static_cast<int32_t>(token);
            mrow[kMetaTopkIndex] = static_cast<int32_t>(kk);
            mrow[kMetaWeightIndex] = 0;
            mrow[4] = 0;
            mrow[5] = 0;
            mrow[6] = 0;
            mrow[7] = 0;
        }
        // pad 元数据行 (wave 组越段槽位): {TOKEN=bs, TOPK=topK-1} →
        // combine 写 dummy 尾行 slot = bs*topK+topK-1 (永不读取)
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
    // InitExpertTokenCountExportBuffers (9709) 的核内累计 (A6: dispatch 期
    // 逐核计数): 伪核 c 拥有 slot 段 [c*perSlot, +perSlot) (ceil 分片,
    // 与 mask/dispatch 的 token 域切分同构; A5 dispatch job 分核语义)
    {
        int32_t* stats = reinterpret_cast<int32_t*>(g_mmWorkspace + WS.stats);
        const uint32_t perSlot = (slots + kBlockAivNum - 1U) / kBlockAivNum;
        for (uint32_t c = 0U; c < kBlockAivNum; ++c) {
            for (uint32_t e = 0U; e < tilingData.moeExpertPerRank; ++e) {
                uint32_t cnt = 0U;
                for (uint32_t i = 0U; i < perSlot; ++i) {
                    const uint32_t s = c * perSlot + i;
                    if (s >= slots) break;
                    if (static_cast<uint32_t>(g_mmTopkIds[s]) == e) ++cnt;
                }
                stats[c * tilingData.moeExpertPerRank + e] =
                    static_cast<int32_t>(cnt);
            }
        }
    }

    // ---- 阶段 3: MoE 专家流水 (ProcessMoeExpertStages asc:10222) ----
    // B8 expert-major wave: 按 expert 分组连续处理 (A5 wave 多核流水 ↔
    // 单程序 expert 波, 调度粒度退化注记); starts 读 workspace 镜像
    // (A5 GMM1 wave 读 cumsumInfo 同款); 行域总长 = slots
    {
        const __half* xDisp = reinterpret_cast<const __half*>(
            g_mmWorkspace + WS.xDisp);
        const int32_t* meta = reinterpret_cast<const int32_t*>(
            g_mmWorkspace + WS.metaInfo);
        float* combineWin = reinterpret_cast<float*>(
            g_mmWorkspace + WS.combine);
        const int32_t* stWs = reinterpret_cast<const int32_t*>(
            g_mmWorkspace + WS.starts);
        for (uint32_t e = 0U; e < tilingData.moeExpertPerRank; ++e) {
            const uint32_t begin = static_cast<uint32_t>(stWs[e]);
            const uint32_t end = (e + 1U < tilingData.moeExpertPerRank)
                ? static_cast<uint32_t>(stWs[e + 1U])
                : slots;
            // [2026-10-14 优化] 自适应组大小分派 (满组 G=4, 尾组 G=rem):
            // 消除尾组越段 pad 槽的无效 VEC/CUBE 计算
            mx_expert_wave_run(mx, xDisp, begin, end, slots,
                               tilingData.h, tilingData.hiddenDim,
                               e, meta, combineWin, tilingData.topK);
        }
    }

    // ---- 阶段 3.5: CrossRankSyncInWorldSize (9295): 自回环本地退化 ----
    // (epWorldSize=1 无远端; mt 版为 combine 窗口跨 PE 可见性屏障)

    // ---- UnpermuteTokens (9163): 加权反排列 (B9) ----
    // y[token] = Σ_kk weight·combineWin[token*topK+kk]
    // (A5 AccumulateMoeExpertsForToken 9040: kk==0 Muls / kk>0 Muls+Add;
    //  权重读 topk_weights, 激活实例 TopkWeightsPrefetch=0)
    {
        const float* combineWin = reinterpret_cast<const float*>(
            g_mmWorkspace + WS.combine);
        for (uint32_t t = 0U; t < tilingData.bs; ++t) {
            mx_unpermute_token(combineWin, g_mmTopkWeights, yOut, t,
                               tilingData.topK, tilingData.h);
        }
    }

    // [2026-10-12 修复] 写方 yOut 冲刷 (mt/mt_dyn 版同款): 驱动 golden
    // GEMM 重构 (备忘录化 + k-outer 分块, ~2× 提速) 后, 验证读与 kernel
    // 末端在飞 TSTORE 的同 PE 竞态显形 (mt_dyn 实测 nuke_pair_diag 相邻
    // ld/st 块 → nuke 恢复丢 tile 状态 → finisher fail)。kernel 返回前
    // 对全部 yOut 行做 volatile 读回 (64B 行粒度, 单 PE 拥有全量行),
    // 借同 PE load/store 块序强制在飞 tile-store 先落盘。yOut 数值
    // 不变 (纯读回), 冲刷开销 <10^3 cycle。
    {
        const volatile float* yV =
            reinterpret_cast<const volatile float*>(yOut);
        for (uint32_t t = 0U; t < tilingData.bs; ++t) {
            for (uint32_t c = 0U; c < tilingData.h; c += 16U) {
                (void)yV[t * tilingData.h + c];   // 64B 行粒度触碰
            }
        }
    }

    // ---- 阶段 4: 统计导出 (A6: dispatch 期核内累计的求和; A5
    //      ExportExpertTokenCounts(FromWorkspace) 4599/4640) ----
    // [2026-10-04 修复沿承] tokOut 导出走标量循环 (原 mm_count_eq_i32
    // tile 计数链在 gfsim TLSU tile-store→GM 缺口下恒返 0, 详见
    // mega_moe_sim_gfsim_fix_report.md); volatile 规避相邻 i64 写入被
    // 合并为 16B tile store (BLK_TSTORE v2i64)。
    {
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

}  // namespace mega_moe\n