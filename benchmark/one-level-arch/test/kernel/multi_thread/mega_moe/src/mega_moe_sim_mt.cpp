/**
 * MegaMoe A8W8 (完整真机流水) — 真 4PE 版 test driver (mega_moe_sim_mt)
 *
 * 与 mega_moe_sim.cpp (原版 driver) 的差异仅执行模型:
 *   - 数据生成: 各 PE 冗余执行 (确定性同值写, 与 group_token_vec_mt /
 *     moe_dispatch_mt 同约定, 无需额外栅栏);
 *   - kernel: mega_moe_sim_mt_kernel 真 4PE 分片 (16 伪核 = 4 PE × 4 伪核,
 *     内部 mtBarrier 同步);
 *   - 验证 (golden 计算 / 精度对比 / finisher / R2): PE0 独占; 非 leader PE
 *     park 自旋 —— worker 先到 _end 会触发 exit_group 截断 PE0 验证
 *     (leader 的 exit 会结束 parked workers)。
 *
 * 运行 (gfrun 功能仿真, 必须四线程):
 *   gfrun -f <elf> -s softcore.multiThreadNum=4
 * 判读: R2 = 0 为 PASS; 非 0 为分层诊断码 (2=基础内存, 3=x 填充, 4=tokenNums,
 *       5/6/7/8=精度误差分级); gfsim 下由 test-finisher (0x10009000=0x5555) 判定。
 */

#include "mega_moe_mt/mega_moe_sim_mt.hpp"

#include <cstdint>

#include "benchmark.h"
#include "fileop.h"

#ifndef kBS
#define kBS 16
#endif

#ifndef kH
#define kH 32
#endif

using uint32 = uint32_t;
using int64 = int64_t;
constexpr uint32 kHiddenDim = mega_moe::kMoeHiddenDim;
constexpr uint32 kTotalWorkspace = 2U * 2U * 2U * 2U * 8U;
constexpr uint32 kWorkspaceBytes =
    2U * kH * kHiddenDim * 4U +                    // w1 fp32
    2U * (kHiddenDim / 2U) * kH * 4U +             // w2 fp32
    2U * kBS * 4U +                                // dispatch 表
    2U * kBS +                                     // mask
    kBS * kH * 4U +                                // combine 缓冲
    16U * 2U * 4U +                                // 核内统计
    4096U;

// ==== GM 全局缓冲 (声明顺序 = bss 地址顺序; 全部置于高地址区 —
//      gfrun/链接器对 bss 低地址区写不可靠 (已实测 0x14110 写丢失), 遇此问题时
//      将用户数据声明下移到本块尾部即可)
__attribute__((aligned(4096))) double g_goldenY[kBS > 0 ? (kBS * kH) : 1];
__attribute__((aligned(4096))) int64 g_goldenTok[2];
__attribute__((aligned(4096))) float g_mmX[kBS * kH];
__attribute__((aligned(4096))) int32_t g_mmTopkIds[kBS];
__attribute__((aligned(4096))) float g_mmTopkWeights[kBS];
__attribute__((aligned(4096))) uint8_t g_mmWeight1[2U * kH * kHiddenDim];
__attribute__((aligned(4096))) uint8_t g_mmWeight2[2U * (kHiddenDim / 2U) * kH];
__attribute__((aligned(4096))) uint8_t g_mmWeightScales1[2U * (kH * kHiddenDim / 32U + 4U)];
__attribute__((aligned(4096))) uint8_t g_mmWeightScales2[2U * (kH * kHiddenDim / 32U + 4U)];
__attribute__((aligned(4096))) float g_mmY[kBS * kH];
__attribute__((aligned(4096))) int64 g_mmExpertTokenNums[2];
__attribute__((aligned(4096))) uint8_t g_mmWorkspace[kWorkspaceBytes];

// Keeps the non-leader park loop side-effecting so -O2 cannot drop it.
static volatile uint32_t sParkSink = 0;

// ==== 参考实现 (gen_data.compute_golden 同语义) ====
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
        for (int32_t i = 0; i < -ip; ++i) r *= 0.5;   // e4m3 指数域 e<7 时 p<0
    }
    return r;
}

static double ref_wscale(uint8_t raw)
{
    const int32_t e = (int32_t)(int8_t)raw;
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

static double ref_w1(uint32_t e, uint32_t k, uint32_t n)
{
    const uint32_t flat = e * kH * kHiddenDim + k * kHiddenDim + n;
    const uint32_t group = (k * kHiddenDim + n) / 32U;
    const uint32_t scaleIdx = e * (kH * kHiddenDim / 32U) + group;
    const uint8_t raw = g_mmWeight1[flat];
    const uint8_t sc = g_mmWeightScales1[scaleIdx % (kH / 32U * 2U * 2U * 2U)];
    return ref_fp8_e4m3(raw) * ref_wscale(sc);
}

static double ref_w2(uint32_t e, uint32_t k, uint32_t n)
{
    const uint32_t flat = e * (kHiddenDim / 2U) * kH + k * kH + n;
    const uint32_t group = (k * kH + n) / 32U;
    const uint32_t scaleIdx = e * ((kHiddenDim / 2U) * kH / 32U) + group;
    const uint8_t raw = g_mmWeight2[flat];
    const uint8_t sc = g_mmWeightScales2[scaleIdx % (kH / 32U * 2U * 2U * 2U)];
    return ref_fp8_e4m3(raw) * ref_wscale(sc);
}

// 完整 MoE 参考前向 (gen_data.compute_golden 语义)
static void compute_golden(double* yRef, int64* tokRef)
{
    for (uint32 t = 0; t < kBS * kH; ++t) yRef[t] = 0.0;
    // 注: volatile 阻止相邻 i64 清零被合并为 16B tile store (BLK_TSTORE v2i64),
    //     linxv5 后端不支持整数 tile 类型, 会报 "Cannot select: v2i64 = BUILD_VECTOR"
    volatile int64* tokV = tokRef;
    tokV[0] = 0;
    tokV[1] = 0;
    {
        uint32_t c0 = 0U;
        for (uint32 t = 0; t < kBS; ++t) {
            if ((uint32)g_mmTopkIds[t] == 0U) ++c0;
        }
        tokV[0] = (int64)c0;
        tokV[1] = (int64)(kBS - c0);
    }

    for (uint32 t = 0; t < kBS; ++t) {
        const uint32 expert = (uint32)g_mmTopkIds[t];
        const double weight = (double)g_mmTopkWeights[t];

        // GMM1: y1[n] = Σ_k x[k]·w1[e][k][n]
        double y1[kHiddenDim];
        for (uint32 n = 0; n < kHiddenDim; ++n) {
            double acc = 0.0;
            for (uint32 k = 0; k < kH; ++k) {
                acc += (double)g_mmX[t * kH + k] * ref_w1(expert, k, n);
            }
            y1[n] = acc;
        }
        // SwiGLU: y2 = silu(y1[:64]) * y1[64:], silu(z)=z·sigmoid(z)
        double y2[kHiddenDim / 2U];
        for (uint32 k = 0; k < kHiddenDim / 2U; ++k) {
            const double z = y1[k];
            const double sig = 1.0 / (1.0 + ref_exp(-z));
            y2[k] = z * sig * y1[k + kHiddenDim / 2U];
        }
        // GMM2: y3[n] = Σ_k y2[k]·w2[e][k][n]
        double y3[kH];
        for (uint32 n = 0; n < kH; ++n) {
            double acc = 0.0;
            for (uint32 k = 0; k < kHiddenDim / 2U; ++k) {
                acc += y2[k] * ref_w2(expert, k, n);
            }
            y3[n] = acc;
        }
        // Combine: y[t] = weight * y3 (topK==1)
        for (uint32 n = 0; n < kH; ++n) {
            yRef[t * kH + n] = weight * y3[n];
        }
    }
}

int main()
{
#ifdef MEGA_MOE_SIM_FAKE
    static_assert(kH % 2 == 0, "kH must be even");
#else
    static_assert(kBS > 0, "kBS must be positive");  // 分片为 ceil 制, 允许 bs < 16 伪核
    static_assert(kH % 2 == 0, "kH must be even");
#endif
    constexpr uint32 kTotalElems = kBS * kH;
    const uint32_t tid = get_thread_idx();

    float* x = g_mmX;
    float* y = g_mmY;
    int64* tokenNumsOut = g_mmExpertTokenNums;
    // 注: volatile 阻止相邻 i64 写入被合并为 16B tile store (BLK_TSTORE v2i64)
    volatile int64* tokInit = tokenNumsOut;
    tokInit[0] = -1;
    tokInit[1] = -1;

    // 基础内存写读自检 (gfrun 通道可用性; 各 PE 冗余同值, 幂等)
    g_mmExpertTokenNums[0] = 12345;
    if (g_mmExpertTokenNums[0] != 12345) {
        return 2;  // R2=2: 基础内存写读异常
    }
    g_mmExpertTokenNums[0] = -1;

    // ---- 确定性数据生成 (各 PE 冗余执行, 同值写无需栅栏) ----
    // x: LCG 均匀 [-1,1] 混合边界
    {
        uint32 seed = 42U;
        for (uint32 i = 0; i < kTotalElems; ++i) {
            seed = seed * 1664525U + 1013904223U;
            const float u = (float)((seed >> 8) & 0xFFFFu) / 65536.0f;
            const float v = (u - 0.5f) * 2.0f;
            x[i] = (i % 5u == 0u) ? v * 0.5f : v;
        }
    }
    // 路由: 与 gen_data 同规则 topk_ids = (i//128)%2, 权重 1.0
    // 注: ids 经 volatile 写 —— BS16 等小规格下循环全展开, 前 kBS/2 个 i32 零 store
    //     会被后端合并为 16B 零 tile store (v2i64 Cannot select 崩溃)
    {
        volatile int32_t* ids = g_mmTopkIds;
        for (uint32 t = 0; t < kBS; ++t) {
            ids[t] = (int32_t)((t / (kBS / 2u)) % 2u);   // 与 gen_data 同规则
            g_mmTopkWeights[t] = 1.0f;
        }
    }
    // FP8 权重: 小随机值 (E4M3 可表示), scale 取 1.0 (E8M0 0x00 → 2^0)
    {
        uint32 seed = 7U;
        for (uint32 i = 0; i < 2U * kHiddenDim * kH; ++i) {
            seed = seed * 1664525U + 1013904223U;
            const uint32 m = (seed >> 1U) & 0x7U;            // 尾数
            const uint32 e = ((seed >> 4U) & 0xFU) == 0U ? 1U : ((seed >> 4U) & 0xFU);  // 指数 1..15
            const uint32 s = (seed >> 8U) & 1U;
            g_mmWeight1[i] = (uint8_t)((s << 7U) | (e << 3U) | m);   // E4M3 ✓ 但范围 [-15,15]
            // 控制幅值: 用指数 4..8 → 值域 ~2^-3..2^1
        }
        // 用确定性细化: 值域 [-2, 2] 内的可表示值
        for (uint32 i = 0; i < 2U * kHiddenDim * kH; ++i) {
            const uint32 e = 4U + ((i * 7U) % 5U);            // 4..8
            const uint32 m = ((i * 3U + 1U) & 0x7U);
            const uint32 s = (i / 7U) & 1U;
            g_mmWeight1[i] = (uint8_t)((s << 7U) | (e << 3U) | m);
        }
        for (uint32 i = 0; i < 2U * (kHiddenDim / 2U) * kH; ++i) {
            const uint32 e = 4U + ((i * 11U + 2U) % 5U);
            const uint32 m = ((i * 5U + 3U) & 0x7U);
            const uint32 s = (i / 13U) & 1U;
            g_mmWeight2[i] = (uint8_t)((s << 7U) | (e << 3U) | m);
        }
        for (uint32 i = 0; i < 2U * (kH * kHiddenDim / 32U + 4U); ++i) {
            g_mmWeightScales1[i] = 0x00;   // scale = 1.0
            g_mmWeightScales2[i] = 0x00;
        }
    }

    BENCHSTART;
    mega_moe::mega_moe_sim_mt_kernel<kBS, kH>(y, x, tokenNumsOut);
    BENCHEND;

    // 非 leader PE park: worker 先到 _end 会触发 exit_group 截断 PE0 的验证
    // (leader 的 SYS_exit 结束 parked workers —— 与 moe_dispatch_mt 同约定)
    if (tid != 0U) {
        for (;;) {
            sParkSink = tid;
        }
    }

    // ---- 完整参考 golden 对比 (PE0 独占; kernel 末端 mtBarrier 后全量输出可见) ----
    double* yRef = g_goldenY;
    int64* tokRef = g_goldenTok;
    compute_golden(yRef, tokRef);

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

    // gfsim 判读通道: test-finisher (0x10009000, 低 16 位 0x5555 = PASS)
    volatile uint32_t* finisher = reinterpret_cast<volatile uint32_t*>(0x10009000ULL);
    if (tokOk && maxAbsErr < 1e-2 && maxRelErr < 1e-2) {
        *finisher = 0x5555;
        return 0;  // R2=0: PASS
    }
    *finisher = 0x0001;
    if (tokenNumsOut[0] == -1 || tokenNumsOut[1] == -1) return 4;  // R2=4: kernel 未写统计
    if (!tokOk) return 5;                        // R2=5: expertTokenNums 数值不符
    if (maxAbsErr < 0.05) return 6;            // 误差 [1e-2, 0.05)
    if (maxAbsErr < 0.2) return 7;             // 误差 [0.05, 0.2)
    if (maxAbsErr < 0.5) return 8;             // 误差 [0.2, 0.5)
    return 9;                                  // 误差 >= 0.5
}
