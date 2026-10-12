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
 *   - barrier 相位由 kernel 内 per-PE 调用计数 (sInvCntDyn) 单调编号
 *     (inv*8+1), 跨调用陈旧 flag 天然失效 —— 不做任何跨 PE 的 flag 复位写;
 *     driver 的 cfg 间汇合点取相位 5 (严格落在 cfgA=1 与 cfgB=9 之间);
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
    // [2026-10-04 修复] slack 4K→16K: 承载 x 预量化槽 (4 PE × myTok×h×2B,
    // bs=18 时 2KB) + 裕量 (kernel 阶段 3 修复注)
    16384U;

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
// [2026-10-04 优化] golden 权重预解码备忘录 (double; 按 max-shape 定长):
// 见 compute_golden 注
__attribute__((aligned(4096))) double g_w1Dbl[kEprMax * kHMax * kHdMax];
__attribute__((aligned(4096))) double g_w2Dbl[kEprMax * (kHdMax / 2U) * kHMax];

// ==== 参考实现 (运行时 shape; gen_data.compute_golden 同语义) ====
// [perf] PR #194 同款 golden 侧优化: 2^k 的 O(|k|) 连乘循环 → 位级 O(1) 构造。
// 2.0/0.5 连乘在 IEEE double 下精确, 位级构造在全部定义域逐位一致 (含上溢
// 2^1024→+inf、次正规域至 2^-1074、完全下溢→0 的阈值行为), golden 数值不变
// (gfrun R2=0 佐证)。连乘循环为 gfsim 周期绝对主体 (每 FP8 权重元素调用
// ref_wscale/exp2_approx, E8M0 偏置大时单次上百次迭代)。
static double ref_exp2i(int64_t k)
{
    union { uint64_t u; double d; } cvt;
    if (k >= 1024) {
        cvt.u = 0x7FF0000000000000ULL;                  // +inf (连乘同上溢)
    } else if (k >= -1022) {
        cvt.u = static_cast<uint64_t>(k + 1023) << 52;   // 规格数域
    } else if (k >= -1074) {
        cvt.u = 1ULL << (k + 1074);                      // 次正规域 (至 2^-1074)
    } else {
        cvt.u = 0ULL;                                    // 完全下溢 (连乘同得 0)
    }
    return cvt.d;
}

static double ref_exp(double z)
{
    const double kLn2 = 0.69314718055994530941723212145818;
    const double kInvLn2 = 1.4426950408889634073599246810019;
    long k = (long)(z * kInvLn2 + (z < 0 ? -0.5 : 0.5));
    const double r = z - (double)k * kLn2;
    double e = 1.0 + r * (1.0 + r * (0.5 + r * (1.0 / 6.0 + r * (1.0 / 24.0 + r * (1.0 / 120.0)))));
    return ref_exp2i(static_cast<int64_t>(k)) * e;
}

static double exp2_approx(double p)
{
    // 语义保留: ip = (int32_t)p, 返回 2^ip (含 e<7 负分支修复)
    return ref_exp2i(static_cast<int64_t>(static_cast<int32_t>(p)));
}

static double ref_wscale(uint8_t raw)
{
    // E8M0: scale = 2^(raw-127) (issue #180: 原 (int8_t)raw 偏置解码语义错误,
    // kernel 与 golden 一致地错所以能通过; MX 硬件路径为真 E8M0, golden 同步修正)
    return ref_exp2i(static_cast<int64_t>(raw) - 127);
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
// [2026-10-04 优化] golden 权重解码备忘录化 (静态 mt 版同款): 原实现
// ref_w1/ref_w2 在 GMM 内循环按访问逐次解码 FP8 (每元素 ~20 op), cfgA/cfgB
// 各 16/18 token 共享 2 expert → 每权重重复解码 8-9 次 (~2×49k 次冗余解码)。
// gfsim fourpe 实测两轮验证为总周期主体, 且 PE0 独占验证长尾是末端 exit
// lockstep 搁浅 (cycle 3,638,308 断言) 的直接成因。改为进入 golden 前按
// 当前 shape 一次性预解码 (≤6144/12288 元素), GMM 内循环退化为纯乘加 ——
// 数值逐位不变, gfrun R2 语义不变, 验证段缩短 ~5×。
static void compute_golden(double* yRef, int64* tokRef, uint32 bs, uint32 h, uint32 hd)
{
    // 一次性预解码当前 shape 的全部权重 (e×k×n 展平, 与 ref_w1/ref_w2 同下标式)
    for (uint32 e = 0; e < 2U; ++e) {
        for (uint32 k = 0; k < h; ++k) {
            for (uint32 n = 0; n < hd; ++n) {
                g_w1Dbl[e * h * hd + k * hd + n] = ref_w1(e, k, n, h, hd);
            }
        }
    }
    for (uint32 e = 0; e < 2U; ++e) {
        for (uint32 k = 0; k < hd / 2U; ++k) {
            for (uint32 n = 0; n < h; ++n) {
                g_w2Dbl[e * (hd / 2U) * h + k * h + n] =
                    ref_w2(e, k, n, h, hd);
            }
        }
    }

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
        const double* w1e = g_w1Dbl + expert * h * hd;
        for (uint32 n = 0; n < hd; ++n) {
            double acc = 0.0;
            for (uint32 k = 0; k < h; ++k) {
                acc += (double)g_mmX[t * h + k] * w1e[k * hd + n];
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
        const double* w2e = g_w2Dbl + expert * (hd / 2U) * h;
        for (uint32 n = 0; n < h; ++n) {
            double acc = 0.0;
            for (uint32 k = 0; k < hd / 2U; ++k) {
                acc += y2[k] * w2e[k * h + n];
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
// genInputs —— 按 PE 连续分片写 (单写者/缓存行; 4 PE 冗余同值全量写会在
// cfgB 与 tile 读叠加时触发 SL2 硬停摆)。调用方在 kernel 前
// mtBarrierDyn(8c+1) 汇合。
static void genInputs(uint32 bs, uint32 h, uint32 hd)
{
    const uint32 tid = get_thread_idx();
    // 连续 ceil 分片 [begin, end)
    auto slice = [](uint32 n, uint32 me, uint32 &begin, uint32 &end) {
        const uint32 seg = n / 4U;
        const uint32 rem = n % 4U;
        begin = me * seg + (me < rem ? me : rem);
        end = begin + seg + (me < rem ? 1U : 0U);
    };
    float* x = g_mmX;
    const uint32 kTotalElems = bs * h;
    {
        uint32 b, e;
        slice(kTotalElems, tid, b, e);
        uint32 seed = 42U;
        for (uint32 i = 0; i < e; ++i) {   // LCG 链推进到切片尾 (值与全量版逐位一致)
            seed = seed * 1664525U + 1013904223U;
            if (i < b) {
                continue;                  // 他人切片: 只推进链, 不写
            }
            const float u = (float)((seed >> 8) & 0xFFFFu) / 65536.0f;
            const float v = (u - 0.5f) * 2.0f;
            x[i] = (i % 5u == 0u) ? v * 0.5f : v;
        }
    }
    // 路由: 与 gen_data 同规则 topk_ids = (i//(bs/2))%2, 权重 1.0
    // 注: ids 经 volatile 写 —— 小规格下循环全展开, 前 bs/2 个 i32 零 store
    //     会被后端合并为 16B 零 tile store (v2i64 Cannot select 崩溃)
    {
        uint32 b, e;
        slice(bs, tid, b, e);
        volatile int32_t* ids = g_mmTopkIds;
        for (uint32 t = b; t < e; ++t) {
            ids[t] = (int32_t)((t / (bs / 2u)) % 2u);   // 与 gen_data 同规则
            g_mmTopkWeights[t] = 1.0f;
        }
    }
    // FP8 权重: 确定性细化, 值域 [-2,2] 内的 E4M3 可表示值; scale = 1.0 (E8M0 0x00)
    {
        uint32 b, e;
        slice(2U * hd * h, tid, b, e);
        for (uint32 i = b; i < e; ++i) {
            const uint32 e4 = 4U + ((i * 7U) % 5U);            // 4..8
            const uint32 m = ((i * 3U + 1U) & 0x7U);
            const uint32 s = (i / 7U) & 1U;
            g_mmWeight1[i] = (uint8_t)((s << 7U) | (e4 << 3U) | m);
        }
        slice(2U * (hd / 2U) * h, tid, b, e);
        for (uint32 i = b; i < e; ++i) {
            const uint32 e4 = 4U + ((i * 11U + 2U) % 5U);
            const uint32 m = ((i * 5U + 3U) & 0x7U);
            const uint32 s = (i / 13U) & 1U;
            g_mmWeight2[i] = (uint8_t)((s << 7U) | (e4 << 3U) | m);
        }
        slice(2U * (h * hd / 32U + 4U), tid, b, e);
        for (uint32 i = b; i < e; ++i) {
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

// ============================================================================
// [2026-10-04 修复] cfgB 验证汇合屏障 — 写方定向驱逐版 (静态 mt 版同款)
//
// 根因 1 (退出 lockstep 搁浅, 用户所报 "全量 workload 执行完毕后被末端
// exit lockstep 断言截断 @ cycle 3,638,308, 无 stats"): 原实现 c-loop 后
// worker 直接 `return 0` 先行 park 到退出 lockstep AND 组; PE0 独占的
// cfgB 验证长尾 (compute_golden bs=18×hd=128 双精度参考, 原逐次解码版
// ~10^6 cycle 级) 使 AND 组停在 ready=3/4 超过 T_deadlock=9999 →
// SyscallBarrier.cpp:1889 断言。"direct-boot 下各 PE 退出相互独立" 仅对
// gfrun 成立。
// 根因 2 (安静窗口陈旧 flag 活锁): 裸 volatile flag 屏障的发布为普通
// 缓存 store (脏行滞留各 PE 私有 L1D, 跨 PE 无 snoop), 非缓存轮询读 L2
// 永远看不到发布值 —— 静态 mt 版实测四线程全部到达屏障却互相等待
// (mega_moe_sim_mt_gfsim_fix_report.md §3.3-2)。
// 修复: 验证汇合屏障 (写方到达定向驱逐 + 非缓存轮询; 判定值经汇合行
// sRzvDyn[4] 同 cacheline 传递, 随驱逐写回 L2), 全 PE 一起放行到 _end。
// ============================================================================
alignas(16384) static volatile uint32_t sRzvEvictSpanDyn[6 * 4096];  // 96KB 驱逐区
// 汇合行: [0..3]=flags, [4]=PE0 判定值 (同一 cacheline, 随屏障驱逐写回 L2)
alignas(64) static volatile uint32_t sRzvDyn[8];

static void mtRendezvousDynFin(uint32_t verdict)
{
    const uint32_t tid = get_thread_idx();
    if (tid == 0U) {
        sRzvDyn[4] = verdict;           // PE0 publishes the verdict (same line)
    }
    mega_moe::mtCompilerBarrierDyn();
    sRzvDyn[tid] = 1U;                  // publish my arrival
    mega_moe::mtCompilerBarrierDyn();
    // 写方到达驱逐: flag 行所在 set 的同 set 偏移行 (发布值立即写回 L2)
    const uint32_t wordOff =
        (static_cast<uint32_t>(
            reinterpret_cast<uint64_t>(&sRzvDyn[0]) >> 2)) & 4095u;
    for (uint32_t k = 1; k <= 5; ++k) {
        (void)sRzvEvictSpanDyn[k * 4096u + wordOff];
    }
    mega_moe::mtCompilerBarrierDyn();
    for (uint32_t t = 0U; t < mega_moe::kMtThreadsPerBlockDyn; ++t) {
        if (t == tid) continue;         // 自己的槽: 程序序保证已置位
        while (sRzvDyn[t] < 1U) {
        }
    }
    mega_moe::mtCompilerBarrierDyn();
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

    // 注: 不采用 gtv 的视图惰性初始化预热 —— mega 含 CUBE ACC 链, 预热段
    // 的守卫 store→默认值 load 同址对是 nuke/重放触发源, 重放二次 SetACC
    // 撞 BROB.cpp:1154 distBid<distLast 断言 (两构建实测); 且 mega 屏障为
    // plain 自旋跳自槽 + 延迟验证 (无长自旋), 守卫运行期初始化的暴露窗口
    // 与静态 mt 版 (gfsim PASS 2,010,999) 相同。
    int failA = 0;
    for (int c = 0; c < 2; ++c) {
        // Barrier 相位由 kernel 内 per-PE 调用计数 (sInvCntDyn) 单调编号
        // (inv*8+4), 跨调用陈旧 flag 天然失效 —— 不做任何跨 PE 的 flag
        // 复位写。相位分配 (每轮 inv=c 占 8 个): gen 汇合 = 8c+1 (driver),
        // routing 汇合 = 8c+2 (kernel, PE0 独占 mask 构建后), 解码汇合 =
        // 8c+3 (kernel, M2 分片解码后), kernel 末端 = 8c+4, 验证汇合 =
        // 8c+5 (driver, PE0 验证前)。
        // tokOut 监控槽重置 (kernel PE0 独占导出; 单写者 = PE0)
        if (tid == 0U) {
            volatile int64* tokInit = tokenNumsOut;
            tokInit[0] = -1;
            tokInit[1] = -1;
        }

        genInputs((uint32)cfgs[c][0], (uint32)cfgs[c][1], (uint32)cfgs[c][2]);
        // genInputs 按 PE 分片写 (单写者/行) → kernel 读他 PE 切片前须汇合
        mega_moe::mtBarrierDyn((uint32)c * 8U + 1U);

        BENCHSTART;
        mega_moe::mega_moe_sim_mt_dyn_kernel(y, x, tokenNumsOut, cfgs[c]);
        BENCHEND;

        // 即时验证: kernel 末端屏障 (8c+4) 保证全 PE 输出就绪; PE0 独占
        // 验证本轮 (仅 c==0), 其余 PE 在 8c+5 汇合点自旋; 验证完成后才
        // 进入下一轮 genInputs (避免覆写竞争)。
        if (c == 0 && tid == 0U) {
            failA = verify(cfgs[0][0], cfgs[0][1], cfgs[0][2]);
        }
        mega_moe::mtBarrierDyn((uint32)c * 8U + 5U);
    }

    // [2026-10-04 修复] cfgB 验证 + 验证汇合 + SPMD finisher (静态 mt 版同款):
    // 1. 原 worker 提前 return 改为验证汇合屏障 —— PE0 的 cfgB 验证在 worker
    //    存活期间执行 (worker 在汇合屏障自旋而非 park 到退出 lockstep), 完成
    //    后全 PE 一起放行、同时到达 _end (消除 ready=3/4 长时停滞);
    // 2. gfsim 判定行按 "全部 threadCount 线程都写过 finisher" 翻转 (SPMD
    //    assumption) —— 全线程在汇合后以汇合行判定值各写一次 (四方同值,
    //    次序无关; 早写会与 kernel 末端 nuke 竞态窗口叠加, 静态 mt 版实测
    //    弃用), 判定值经驱逐屏障保证全 PE 可见。
    int ret = 0;
    uint32_t verdict = 0x5555u;
    if (tid == 0U) {
        // cfgB 验证 (PE0 独占; live 缓冲即 cfgB 数据)。
        // 诊断码优先级保持原语义 (failA 优先)。
        int rcB = verify(cfgs[1][0], cfgs[1][1], cfgs[1][2]);
        if (failA != 0) {                        // cfgA: 静态版同款诊断码
            verdict = 0x0001u;
            ret = failA;
        } else if (rcB == 0) {
            verdict = 0x5555u;
            ret = 0;                             // R2=0: PASS
        } else {
            verdict = 0x0001u;
            ret = 10 + rcB;                      // cfgB: +10 偏移诊断码
        }
    }

    // [2026-10-04 修复] 验证汇合屏障 (写方定向驱逐版, 见 mtRendezvousDynFin 注)
    mtRendezvousDynFin(verdict);

    // gfsim 判读通道: test-finisher (0x10009000, 低 16 位 0x5555 = PASS)
    // [2026-10-04 修复] SPMD 全线程写: 值取自汇合行 (PE0 发布、屏障驱逐
    // 保证可见), 四方同值次序无关。
    volatile uint32_t* finisher = reinterpret_cast<volatile uint32_t*>(0x10009000ULL);
    *finisher = sRzvDyn[4];

    return ret;
}