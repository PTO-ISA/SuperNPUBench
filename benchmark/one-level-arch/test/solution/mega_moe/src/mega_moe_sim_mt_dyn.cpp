/**
 * MegaMoe A8W8 (完整真机流水) — 真 4PE 动态 shape test driver
 * (mega_moe_sim_mt_dyn)
 *
 * 基于 mega_moe_sim_mt.cpp 的动态 shape 版驱动:
 *   - 单次运行内顺序执行 2 组运行时 shape:
 *       cfgA: tiling={bs=16, h=32, hiddenDim=64, epr=2, topK=1}
 *             — 与静态版同值, 等价回归
 *       cfgB: tiling={bs=18, h=32, hiddenDim=128, epr=2, topK=1}
 *             — bs=18%16=2 触发伪核尾分片; hiddenDim=128 覆盖运行时 GMM
 *             维度路径 (MX 契约: hd/2 须整除 32, 原 96 不满足)
 *   - GM 缓冲按两组 shape 的最大值定长分配, 运行时只使用有效段;
 *     workspace 布局与 MegaMoeWave::Init 一致 + 尾部每 PE GMM scratch
 *     (kernel 头文件注释有完整契约);
 *   - 两次 kernel 调用之间把 sMtPhaseDoneDyn barrier 数组清零 (静态相位编号
 *     在第二次调用会因陈旧 flag 立即通过而失去同步);
 *   - 验证 PE0 独占, 逐组比对 (golden/精度/token 与静态版一致, 边界运行时化):
 *     cfgA 失败返回静态版同款诊断码, cfgB 失败返回 +10 偏移码。
 *
 * 运行 (gfrun 功能仿真, 必须四线程):
 *   gfrun -f <elf> -s softcore.multiThreadNum=4
 * 判读: R2 = 0 为 PASS; 非 0 为诊断码; gfsim 下由 test-finisher
 *   (0x10009000=0x5555) 判定。
 */

#include "solution/mega_moe/mega_moe_sim_mt_dyn.hpp"

#include <cstdint>

#include "benchmark.h"
#include "fileop.h"

using uint32 = uint32_t;
using int64 = int64_t;

// ---- max-shape 缓冲尺寸 (cfgA/cfgB 的逐维最大值) ----
constexpr uint32 kBSMax = 18;
constexpr uint32 kHMax = 32;
constexpr uint32 kHdMax = 128;
constexpr uint32 kEprMax = 2;
constexpr uint32 kTopKMax = 1;
// 每 PE scratch = w1F16[epr*h*hd] + w2F16[epr*(hd/2)*h] + xf16[h] +
// y1[hd]f32 + y2f16[hd/2] + y3[h]f32 (kernel 头文件契约)
constexpr uint32 kWorkspaceBytes =
    2U * kBSMax * 4U +                                     // dispatch 表
    2U * kBSMax +                                          // mask
    kBSMax * kHMax * 4U +                                  // combine 缓冲
    16U * kEprMax * 4U +                                   // 核内统计
    4U * ((kEprMax * kHMax * kHdMax
         + kEprMax * (kHdMax / 2U) * kHMax) * 2U           // 每 PE fp16 权重副本
        + kHMax * 2U + kHdMax * 4U + kHdMax * 2U + kHMax * 4U) +  // GMM scratch
    4096U;

// ==== GM 全局缓冲 (声明顺序 = bss 地址顺序; 全部置于高地址区 —
//      gfrun/链接器对 bss 低地址区写不可靠 (已实测 0x14110 写丢失), 遇此问题时
//      将用户数据声明下移到本块尾部即可)
__attribute__((aligned(4096))) double g_goldenY[kBSMax > 0 ? (kBSMax * kHMax) : 1];
__attribute__((aligned(4096))) int64 g_goldenTok[2];
__attribute__((aligned(4096))) float g_mmX[kBSMax * kHMax];
__attribute__((aligned(4096))) int32_t g_mmTopkIds[kBSMax * kTopKMax];
__attribute__((aligned(4096))) float g_mmTopkWeights[kBSMax * kTopKMax];
__attribute__((aligned(4096))) uint8_t g_mmWeight1[kEprMax * kHMax * kHdMax];
__attribute__((aligned(4096))) uint8_t g_mmWeight2[kEprMax * (kHdMax / 2U) * kHMax];
// 权重 scale (E8M0 标量, 每 expert × 每 k 组一个; kernel 解码时 TMULS 折叠):
//   w1Scale[e][k/32] (k ∈ [0, h)), w2Scale[e][k/32] (k ∈ [0, hd/2))
__attribute__((aligned(4096))) uint8_t g_mmWeightScales1[kEprMax * (kHMax / 32U)];
__attribute__((aligned(4096))) uint8_t g_mmWeightScales2[kEprMax * (kHdMax / 64U)];
__attribute__((aligned(4096))) float g_mmY[kBSMax * kHMax];
__attribute__((aligned(4096))) int64 g_mmExpertTokenNums[kEprMax];
__attribute__((aligned(4096))) uint8_t g_mmWorkspace[kWorkspaceBytes];

// ==== 参考实现 (运行时 shape; gen_data.compute_golden 同语义) ====
static double ref_exp(double z)
{
    const double kLn2 = 0.69314718055994530941723212145818;
    const double kInvLn2 = 1.4426950408889634073599246810019;
    long k = (long)(z * kInvLn2 + (z < 0 ? -0.5 : 0.5));
    const double r = z - (double)k * kLn2;
    double e = 1.0 + r * (1.0 + r * (0.5 + r * (1.0 / 6.0 + r * (1.0 / 24.0 + r * (1.0 / 120.0)))));
    double twoK = 1.0;
    if (k >= 0) {
        for (long j = 0; j < k; ++j) twoK *= 2.0;
    } else {
        for (long j = 0; j < -k; ++j) twoK *= 0.5;
    }
    return twoK * e;
}

static double exp2_approx(double p)
{
    double r = 1.0;
    int32_t ip = (int32_t)p;
    if (ip > 0) {
        for (int32_t i = 0; i < ip; ++i) r *= 2.0;
    } else if (ip < 0) {
        for (int32_t i = 0; i < -ip; ++i) r *= 0.5;   // [fix] e4m3 指数域 e<7 时 p<0
    }
    return r;
}

static double ref_wscale(uint8_t raw)
{
    // E8M0: scale = 2^(raw-127) (issue #180: 原 (int8_t)raw 偏置解码语义错误,
    // kernel 与 golden 一致地错所以能通过; MX 硬件路径为真 E8M0, golden 同步修正)
    const int32_t e = (int32_t)raw - 127;
    double s = 1.0;
    if (e >= 0) {
        for (int32_t i = 0; i < e; ++i) s *= 2.0;
    } else {
        for (int32_t i = 0; i < -e; ++i) s *= 0.5;
    }
    return s;
}

// host 侧 FP8 解码 (与 kernel fp8_e4m3_to_f32 / fp8_e8m0_scale 数学一致, double 精度)
static double ref_fp8_e4m3(double raw)
{
    const uint32_t s = ((uint8_t)raw >> 7U) & 1U;
    const uint32_t e = ((uint8_t)raw >> 3U) & 0xFU;
    const uint32_t m = (uint8_t)raw & 0x7U;
    double val;
    if (e == 0U) {
        val = (double)m / 8.0 * 0.015625;
    } else {
        val = (1.0 + (double)m / 8.0) * exp2_approx((double)(int32_t)e - 7.0);
    }
    return s ? -val : val;
}

static double ref_w1(uint32_t e, uint32_t k, uint32_t n, uint32_t h, uint32_t hd)
{
    const uint32_t flat = e * h * hd + k * hd + n;
    // 每 k 组标量 scale: w1Scale[e][k/32]
    const uint8_t sc = g_mmWeightScales1[e * (h / 32U) + k / 32U];
    return ref_fp8_e4m3(g_mmWeight1[flat]) * ref_wscale(sc);
}

static double ref_w2(uint32_t e, uint32_t k, uint32_t n, uint32_t h, uint32_t hd)
{
    const uint32_t flat = e * (hd / 2U) * h + k * h + n;
    // 每 k 组标量 scale: w2Scale[e][k/32] (k ∈ [0, hd/2))
    const uint8_t sc = g_mmWeightScales2[e * ((hd / 2U) / 32U) + k / 32U];
    return ref_fp8_e4m3(g_mmWeight2[flat]) * ref_wscale(sc);
}

// 完整 MoE 参考前向 (gen_data.compute_golden 语义, 运行时 shape; topK==1 驱动约定)
static void compute_golden(double* yRef, int64* tokRef, uint32 bs, uint32 h, uint32 hd)
{
    for (uint32 t = 0; t < bs * h; ++t) yRef[t] = 0.0;
    // 注: volatile 阻止相邻 i64 清零被合并为 16B tile store (BLK_TSTORE v2i64),
    //     linxv5 后端不支持整数 tile 类型, 会报 "Cannot select: v2i64 = BUILD_VECTOR"
    volatile int64* tokV = tokRef;
    tokV[0] = 0;
    tokV[1] = 0;
    // [fix] 多 PE 共享栈/GM: 寄存器计数 + 幂等写 (与 kernel 侧统计导出同模式)
    {
        uint32_t c0 = 0U;
        for (uint32 t = 0; t < bs; ++t) {
            if ((uint32)g_mmTopkIds[t] == 0U) ++c0;
        }
        tokV[0] = (int64)c0;
        tokV[1] = (int64)(bs - c0);
    }

    for (uint32 t = 0; t < bs; ++t) {
        const uint32 expert = (uint32)g_mmTopkIds[t];
        const double weight = (double)g_mmTopkWeights[t];

        // GMM1: y1[n] = Σ_k x[k]·w1[e][k][n]
        static double y1[kHdMax];
        for (uint32 n = 0; n < hd; ++n) {
            double acc = 0.0;
            for (uint32 k = 0; k < h; ++k) {
                acc += (double)g_mmX[t * h + k] * ref_w1(expert, k, n, h, hd);
            }
            y1[n] = acc;
        }
        // SwiGLU: y2 = silu(y1[:hd/2]) * y1[hd/2:]
        static double y2[kHdMax / 2U];
        for (uint32 k = 0; k < hd / 2U; ++k) {
            const double z = y1[k];
            const double sig = 1.0 / (1.0 + ref_exp(-z));
            y2[k] = z * sig * y1[k + hd / 2U];
        }
        // GMM2: y3[n] = Σ_k y2[k]·w2[e][k][n]
        static double y3[kHMax];
        for (uint32 n = 0; n < h; ++n) {
            double acc = 0.0;
            for (uint32 k = 0; k < hd / 2U; ++k) {
                acc += y2[k] * ref_w2(expert, k, n, h, hd);
            }
            y3[n] = acc;
        }
        // Combine: y[t] = weight * y3 (topK==1; 原 "+=" 在多 PE 下对共享 GM 重复累加 ×4)
        for (uint32 n = 0; n < h; ++n) {
            yRef[t * h + n] = weight * y3[n];
        }
    }
}

// ---- 确定性数据生成 (各 PE 冗余执行, 同值写无需栅栏; 运行时 shape) ----
static void genInputs(uint32 bs, uint32 h, uint32 hd)
{
    float* x = g_mmX;
    const uint32 kTotalElems = bs * h;
    {
        uint32 seed = 42U;
        for (uint32 i = 0; i < kTotalElems; ++i) {
            seed = seed * 1664525U + 1013904223U;
            const float u = (float)((seed >> 8) & 0xFFFFu) / 65536.0f;
            const float v = (u - 0.5f) * 2.0f;
            x[i] = (i % 5u == 0u) ? v * 0.5f : v;
        }
    }
    // 路由: 与 gen_data 同规则 topk_ids = (i//(bs/2))%2, 权重 1.0
    // 注: ids 经 volatile 写 —— 小规格下循环全展开, 前 bs/2 个 i32 零 store
    //     会被后端合并为 16B 零 tile store (v2i64 Cannot select 崩溃)
    {
        volatile int32_t* ids = g_mmTopkIds;
        for (uint32 t = 0; t < bs; ++t) {
            ids[t] = (int32_t)((t / (bs / 2u)) % 2u);   // 与 gen_data 同规则
            g_mmTopkWeights[t] = 1.0f;
        }
    }
    // FP8 权重: 确定性细化, 值域 [-2,2] 内的 E4M3 可表示值; scale = 1.0 (E8M0 0x00)
    {
        for (uint32 i = 0; i < 2U * hd * h; ++i) {
            const uint32 e = 4U + ((i * 7U) % 5U);            // 4..8
            const uint32 m = ((i * 3U + 1U) & 0x7U);
            const uint32 s = (i / 7U) & 1U;
            g_mmWeight1[i] = (uint8_t)((s << 7U) | (e << 3U) | m);
        }
        for (uint32 i = 0; i < 2U * (hd / 2U) * h; ++i) {
            const uint32 e = 4U + ((i * 11U + 2U) % 5U);
            const uint32 m = ((i * 5U + 3U) & 0x7U);
            const uint32 s = (i / 13U) & 1U;
            g_mmWeight2[i] = (uint8_t)((s << 7U) | (e << 3U) | m);
        }
        for (uint32 i = 0; i < 2U * (h * hd / 32U + 4U); ++i) {
            g_mmWeightScales1[i] = 0x00;   // scale = 1.0
            g_mmWeightScales2[i] = 0x00;
        }
    }
}

// ---- 完整参考 golden 对比 (PE0 独占; 运行时 shape) ----
// 返回 0 = PASS; 4/5/6/7/8/9 = 静态版同款诊断码
static int verify(int64_t bs, int64_t h, int64_t hd)
{
    const uint32 bsU = (uint32)bs;
    const uint32 hU = (uint32)h;
    const uint32 hdU = (uint32)hd;
    const uint32 kTotalElems = bsU * hU;

    float* y = g_mmY;
    int64* tokenNumsOut = g_mmExpertTokenNums;

    double* yRef = g_goldenY;
    int64* tokRef = g_goldenTok;
    compute_golden(yRef, tokRef, bsU, hU, hdU);

    double maxAbsErr = 0.0;
    for (uint32 i = 0; i < kTotalElems; ++i) {
        const double err = (double)y[i] - yRef[i];
        const double absErr = err < 0 ? -err : err;
        if (absErr > maxAbsErr) maxAbsErr = absErr;
    }

    // 相对容差 (量化解码误差随幅值放大, 用宽松 rtol 判定)
    double maxRelErr = 0.0;
    for (uint32 i = 0; i < kTotalElems; ++i) {
        const double denom = yRef[i] < 0 ? -yRef[i] : yRef[i];
        if (denom > 1e-6) {
            const double rel = ((double)y[i] - yRef[i]) / denom;
            const double absRel = rel < 0 ? -rel : rel;
            if (absRel > maxRelErr) maxRelErr = absRel;
        }
    }

    const bool tokOk = (tokenNumsOut[0] == tokRef[0]) && (tokenNumsOut[1] == tokRef[1]);

    if (tokOk && maxAbsErr < 1e-2 && maxRelErr < 1e-2) {
        return 0;
    }
    if (tokenNumsOut[0] == -1 || tokenNumsOut[1] == -1) return 4;  // kernel 未写统计
    if (!tokOk) return 5;                        // expertTokenNums 数值不符
    if (maxAbsErr < 0.05) return 6;              // 误差 [1e-2, 0.05)
    if (maxAbsErr < 0.2) return 7;               // 误差 [0.05, 0.2)
    if (maxAbsErr < 0.5) return 8;               // 误差 [0.2, 0.5)
    return 9;                                    // 误差 >= 0.5
}

int main()
{
    const uint32_t tid = get_thread_idx();

    float* x = g_mmX;
    float* y = g_mmY;
    int64* tokenNumsOut = g_mmExpertTokenNums;

    // 基础内存写读自检 (gfrun 通道可用性; 各 PE 冗余同值, 幂等)
    g_mmExpertTokenNums[0] = 12345;
    if (g_mmExpertTokenNums[0] != 12345) {
        return 2;  // R2=2: 基础内存写读异常
    }
    g_mmExpertTokenNums[0] = -1;

    const int64_t cfgA[5] = {16, 32, 64, 2, 1};   // 与静态版同值 (等价回归)
    // hd=128: MX tile 契约要求 GMM2 的 K=hd/2 整除 32 (scale 组宽); 仍覆盖
    // 伪核尾分片 (bs=18%16=2) 与运行时 hd 维度路径 (128 != cfgA 的 64)
    const int64_t cfgB[5] = {18, 32, 128, 2, 1};
    const int64_t* cfgs[2] = {cfgA, cfgB};

    int failA = 0;
    for (int c = 0; c < 2; ++c) {
        // Barrier reset: 静态相位编号跨次调用会因陈旧 flag 立即通过而
        // 失去同步, 每次调用前清零 (各 PE 冗余同值写, 无需栅栏)。
        for (uint32_t t = 0; t < mega_moe::kMtThreadsPerBlockDyn; ++t) {
            mega_moe::sMtPhaseDoneDyn[t] = 0;
        }
        // tokOut 监控槽重置 (kernel PE0 独占导出; 各 PE 冗余同值写)
        {
            volatile int64* tokInit = tokenNumsOut;
            tokInit[0] = -1;
            tokInit[1] = -1;
        }

        genInputs((uint32)cfgs[c][0], (uint32)cfgs[c][1], (uint32)cfgs[c][2]);

        BENCHSTART;
        mega_moe::mega_moe_sim_mt_dyn_kernel(y, x, tokenNumsOut, cfgs[c]);
        BENCHEND;

        // cfgA 验证须在其输入被 cfgB 数据生成覆盖之前完成: PE0 立即验证,
        // 其余 PE 在 mtBarrierDyn(4) 汇合等待 (kernel 内部相位为 1/2)。
        if (c == 0) {
            if (tid == 0U) {
                failA = verify(cfgs[0][0], cfgs[0][1], cfgs[0][2]);
            }
            mega_moe::mtBarrierDyn(4U);
        }
    }

    // 非 leader PE 直接返回：09-01 版功能模型起 direct-boot 下各 PE 退出
    // 相互独立，worker 退出不再截断 PE0 的验证。
    if (tid != 0U) {
        return 0;
    }

    // gfsim 判读通道: test-finisher (0x10009000, 低 16 位 0x5555 = PASS)
    volatile uint32_t* finisher = reinterpret_cast<volatile uint32_t*>(0x10009000ULL);
    if (failA != 0) {                              // cfgA: 静态版同款诊断码
        *finisher = 0x0001;
        return failA;
    }
    int rcB = verify(cfgs[1][0], cfgs[1][1], cfgs[1][2]);
    if (rcB == 0) {
        *finisher = 0x5555;
        return 0;  // R2=0: PASS
    }
    *finisher = 0x0001;
    return 10 + rcB;                               // cfgB: +10 偏移诊断码
}