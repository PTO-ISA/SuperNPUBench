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

#include "solution/mega_moe/mega_moe_sim_mt.hpp"

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
// [A1 对齐 2026-10-13] workspace = dispatch 通路布局 (moe_ws_layout: dstPos/
// mask/cnt/starts/xDisp/scaleDisp/metaInfo/combine/stats) + 4 PE MX scratch
// (fp16 权重副本 + wave 组 y1/y2/y3) + 余量 —— 与 kernel 四d 节同式
constexpr uint32 kWorkspaceBytes =
    mega_moe::moe_ws_layout(kBS, kH, kHiddenDim,
                            mega_moe::kMoeExpertPerRank,
                            mega_moe::kTopK).total +
    4U * mega_moe::mx_tile_scratch_bytes(mega_moe::kMoeExpertPerRank,
                                         kH, kHiddenDim) +
    16384U;

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
// 权重 scale (E8M0 标量, 每 expert × 每 k 组一个; kernel tile 解码时 TMULS 折叠):
//   w1Scale[e][k/32] (k ∈ [0, h)), w2Scale[e][k/32] (k ∈ [0, hd/2))
__attribute__((aligned(4096))) uint8_t g_mmWeightScales1[2U * (kH / 32U)];
__attribute__((aligned(4096))) uint8_t g_mmWeightScales2[2U * (kHiddenDim / 64U)];
// [2026-10-04 注] gfsim nuke-恢复竞态的时序相位衬垫 (详见修复报告 §5.3):
// 模型的 nuke 恢复在 kernel 末端 tile 块在飞窗口存在丢块竞态 (F 构建
// nuke@313,527 恢复成功, H 构建 nuke@313,517 丢块死锁, 10-cycle 窗口),
// 唯一基准侧可控变量是 .bss 布局相位。此衬垫使 g_mmWorkspace 及其后
// 全部符号平移, 重掷时序骰子; 语义零影响 (BENCHSTART 前无人读写)。
__attribute__((aligned(4096))) uint8_t g_timingPad[4096];
__attribute__((aligned(4096))) float g_mmY[kBS * kH];
__attribute__((aligned(4096))) int64 g_mmExpertTokenNums[2];
__attribute__((aligned(4096))) uint8_t g_mmWorkspace[kWorkspaceBytes];
// [2026-10-04 优化] golden 权重预解码备忘录 (double): 见 compute_golden 注
__attribute__((aligned(4096))) double g_w1Dbl[2U * kH * kHiddenDim];
__attribute__((aligned(4096))) double g_w2Dbl[2U * (kHiddenDim / 2U) * kH];
// [2026-10-14 优化] E4M3→double 解码 LUT (256 值一次性查表; 见
// compute_golden 内构建注)。置于 bss 尾部, 不影响既有符号布局
// (g_timingPad 衬垫域之后的符号相对关系不变)。
__attribute__((aligned(4096))) double g_e4m3Lut[256];


// ==== 参考实现 (gen_data.compute_golden 同语义) ====
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
    // E8M0: scale = 2^(raw-127) — issue #180 修正 (原 (int8_t)raw 偏置解码
    // 语义错误; tile 路径 kernel/golden 统一为正确 E8M0)
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

static double ref_w1(uint32_t e, uint32_t k, uint32_t n)
{
    const uint32_t flat = e * kH * kHiddenDim + k * kHiddenDim + n;
    // 每 k 组标量 scale: w1Scale[e][k/32]
    const uint8_t sc = g_mmWeightScales1[e * (kH / 32U) + k / 32U];
    return ref_fp8_e4m3(g_mmWeight1[flat]) * ref_wscale(sc);
}

static double ref_w2(uint32_t e, uint32_t k, uint32_t n)
{
    const uint32_t flat = e * (kHiddenDim / 2U) * kH + k * kH + n;
    // 每 k 组标量 scale: w2Scale[e][k/32] (k ∈ [0, hd/2))
    const uint8_t sc = g_mmWeightScales2[e * (kHiddenDim / 64U) + k / 32U];
    return ref_fp8_e4m3(g_mmWeight2[flat]) * ref_wscale(sc);
}

// ============================================================================
// 参考实现 (gen_data.compute_golden 同语义)
// [2026-10-04 优化] golden 权重解码备忘录化: 原实现 ref_w1/ref_w2 在 GMM
// 内循环按访问逐次解码 FP8 (ref_fp8_e4m3 + ref_wscale, 每元素 ~20 op);
// 本数据分布 16 token 共享 2 个 expert → 每权重被重复解码 8 次
// (2×32×64 + 2×32×32 = 6144 权重 × 8 ≈ 49k 次冗余解码)。gfsim fourpe
// 实测该验证长尾 ~1.7×10^6 cycle (workers parked 无竞争基线,
// -s core.deadlock_cycles 提高看门狗后实测 Total Cycles = 2,011,065,
// 其中 kernel+数据生成仅 ~313k), 是 `--conf fourpe` 退出 lockstep 崩溃
// 与仿真超时的主因。改为进入 golden 前一次性预解码到 double 备忘录
// (6144 元素), GMM 内循环退化为纯乘加——数值逐位不变 (解码为元素字节
// 的纯函数), gfrun R2 语义不变, 验证段缩短 ~5×。
// ============================================================================
static void compute_golden(double* yRef, int64* tokRef)
{
    // [2026-10-14 优化] E4M3 解码 LUT 一次性构建 (256 值; 构建用原
    // ref_fp8_e4m3 逐值计算 → 查表值与逐次解码逐位一致)。compute_golden
    // 仅由 PE0 独占调用 → 单写者无竞争; 解码内循环由 "位提取 + exp2 位级
    // 构造" (~20 op/元素) 退化为 1 load + 1 fmul, 乘法操作数不变 →
    // golden 数值逐位一致。
    static bool sE4m3LutReady = false;
    if (!sE4m3LutReady) {
        for (uint32 b = 0U; b < 256U; ++b) {
            g_e4m3Lut[b] = ref_fp8_e4m3(static_cast<double>(b));
        }
        sE4m3LutReady = true;
    }
    // 一次性预解码全部权重 (e×k×n 展平, 与 ref_w1/ref_w2 同下标式)
    // [2026-10-12 优化] ref_wscale 按 (e,k) 组提升 (原 ref_w1 逐元素解码,
    // 同 (e,k) 的 kHiddenDim/kH 个元素共享组 scale, 重复调用 ~10 op/元素;
    // 乘法操作数不变, 解码值逐位一致)
    // [2026-10-14 优化] E4M3 值经 g_e4m3Lut 查表 (数值逐位一致)
    // [2026-10-14 v7 优化] 解码内循环瘦身 (LUT 基址局部指针 + n 循环 4 路
    // 展开; sim 版同款, 见该注)
    const double* lut = g_e4m3Lut;
    for (uint32 e = 0; e < 2U; ++e) {
        for (uint32 k = 0; k < kH; ++k) {
            const double scD =
                ref_wscale(g_mmWeightScales1[e * (kH / 32U) + k / 32U]);
            double* dst = g_w1Dbl + e * kH * kHiddenDim + k * kHiddenDim;
            const uint8_t* src =
                g_mmWeight1 + e * kH * kHiddenDim + k * kHiddenDim;
            uint32 n = 0U;
            for (; n + 4U <= kHiddenDim; n += 4U) {
                dst[n] = lut[src[n]] * scD;
                dst[n + 1U] = lut[src[n + 1U]] * scD;
                dst[n + 2U] = lut[src[n + 2U]] * scD;
                dst[n + 3U] = lut[src[n + 3U]] * scD;
            }
            for (; n < kHiddenDim; ++n) {
                dst[n] = lut[src[n]] * scD;
            }
        }
    }
    for (uint32 e = 0; e < 2U; ++e) {
        for (uint32 k = 0; k < kHiddenDim / 2U; ++k) {
            const double scD =
                ref_wscale(g_mmWeightScales2[e * (kHiddenDim / 64U) + k / 32U]);
            double* dst =
                g_w2Dbl + e * (kHiddenDim / 2U) * kH + k * kH;
            const uint8_t* src =
                g_mmWeight2 + e * (kHiddenDim / 2U) * kH + k * kH;
            uint32 n = 0U;
            for (; n + 4U <= kH; n += 4U) {
                dst[n] = lut[src[n]] * scD;
                dst[n + 1U] = lut[src[n + 1U]] * scD;
                dst[n + 2U] = lut[src[n + 2U]] * scD;
                dst[n + 3U] = lut[src[n + 3U]] * scD;
            }
            for (; n < kH; ++n) {
                dst[n] = lut[src[n]] * scD;
            }
        }
    }

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
        const double* w1e = g_w1Dbl + expert * kH * kHiddenDim;
        const double* w2e = g_w2Dbl + expert * (kHiddenDim / 2U) * kH;

        // GMM1: y1[n] = Σ_k x[k]·w1[e][k][n]   (读预解码备忘录)
        // [2026-10-12 优化] k-outer × n 16 宽寄存器分块 (mt_dyn 版同款):
        // 原内循环 w1e[k*kHiddenDim+n] 随 k 步进跨 kHiddenDim×8 B —— 每次
        // 权重访存跨 cacheline 且 x[k] 逐 n 重复加载/转换 (gfsim 实测
        // ~20 cycle/fmadd)。分块后权重访问连续流式, 每元素累加序列不变
        // (k 升序同款 fmadd 链), golden 数值逐位一致。
        // [2026-10-14 v7 优化] x 行每 token 预转换 (sim 版同款, 见该注)
        double xrow[kH];
        for (uint32 k = 0; k < kH; ++k) {
            xrow[k] = (double)g_mmX[t * kH + k];
        }
        double y1[kHiddenDim];
        constexpr uint32 kNb = 16U;   // n 分块宽 (kHiddenDim=64 整除)
        uint32 n0 = 0U;
        for (; n0 + kNb <= kHiddenDim; n0 += kNb) {
            double acc[kNb];
            for (uint32 j = 0; j < kNb; ++j) acc[j] = 0.0;
            for (uint32 k = 0; k < kH; ++k) {
                const double xv = xrow[k];
                const double* wk = w1e + k * kHiddenDim + n0;
                for (uint32 j = 0; j < kNb; ++j) {
                    acc[j] += xv * wk[j];
                }
            }
            for (uint32 j = 0; j < kNb; ++j) y1[n0 + j] = acc[j];
        }
        for (; n0 < kHiddenDim; ++n0) {   // 尾块 (防御性; 本 spec 不触达)
            double accT = 0.0;
            for (uint32 k = 0; k < kH; ++k) {
                accT += (double)g_mmX[t * kH + k] * w1e[k * kHiddenDim + n0];
            }
            y1[n0] = accT;
        }
        // SwiGLU: y2 = silu(y1[:64]) * y1[64:], silu(z)=z·sigmoid(z)
        double y2[kHiddenDim / 2U];
        for (uint32 k = 0; k < kHiddenDim / 2U; ++k) {
            const double z = y1[k];
            const double sig = 1.0 / (1.0 + ref_exp(-z));
            y2[k] = z * sig * y1[k + kHiddenDim / 2U];
        }
        // GMM2: y3[n] = Σ_k y2[k]·w2[e][k][n]  (读预解码备忘录)
        // [2026-10-12 优化] 同 GMM1 的 k-outer × n 分块重构
        double y3[kH];
        uint32 m0 = 0U;
        for (; m0 + kNb <= kH; m0 += kNb) {
            double acc[kNb];
            for (uint32 j = 0; j < kNb; ++j) acc[j] = 0.0;
            for (uint32 k = 0; k < kHiddenDim / 2U; ++k) {
                const double yv = y2[k];
                const double* wk = w2e + k * kH + m0;
                for (uint32 j = 0; j < kNb; ++j) {
                    acc[j] += yv * wk[j];
                }
            }
            for (uint32 j = 0; j < kNb; ++j) y3[m0 + j] = acc[j];
        }
        for (; m0 < kH; ++m0) {   // 尾块 (防御性; 本 spec 不触达)
            double accT = 0.0;
            for (uint32 k = 0; k < kHiddenDim / 2U; ++k) {
                accT += y2[k] * w2e[k * kH + m0];
            }
            y3[m0] = accT;
        }
        // Combine: y[t] = weight * y3 (topK==1)
        for (uint32 n = 0; n < kH; ++n) {
            yRef[t * kH + n] = weight * y3[n];
        }
    }
}

// ============================================================================
// [2026-10-04 修复] 验证汇合屏障 — 写方定向驱逐版 (mtBarrier(2) 专用)
//
// 根因 (gfsim `--conf fourpe` 实测, cycle 1,262,407 ResVerify 死锁取证):
// mtBarrier 的 flag 发布为普通缓存 store (sw, 脏行滞留各 PE 私有 L1D,
// 跨 PE 无 snoop), 而等待方轮询为非缓存 lwi.u (直读 L2)——发布值不到达
// L2 则永远不可见。kernel 内部 mtBarrier(1) 处于 busy 窗口 (重度 tile/
// 内存流量把脏 flag 行容量驱逐到 L2, "侥幸"可见); 本驱动新增的验证汇合
// 屏障处于安静窗口 (worker 静止自旋、PE0 独占验证 ~4×10^5 cycle), 脏 flag
// 行无人驱逐 → 四线程全部到达屏障却互相看不到对方的 phase=2 → 确定性
// 活锁 (实测 BPC 0x11e68, poll 读值 data=0x1)。与 group_token_vec_mt
// 修复报告 §3.3 记载的 "裸热自旋在长偏斜下读陈旧 flag 活锁" 同一机理。
//
// 修复 (group_token_vec_mt.hpp 驱逐屏障同款, 该机制在 gfsim fourpe 实证
// 可靠): 写方到达后立即对 flag 行所在 L1D set 做定向驱逐读 (同 set 5 行),
// 迫使脏 flag 行写回 L2; 等待方保持非缓存轮询 (lwi.u) 即可看到新值。
// 96KB 驱逐区 16KB 对齐: sEvictSpan[k*4096+wordOff] 与 flag 行同 set。
// ============================================================================
alignas(16384) static volatile uint32_t sMtEvictSpan[6 * 4096];  // 96KB 驱逐区
// 汇合行: [0..3]=phase flags, [4]=PE0 判定值 (同一 cacheline, 随屏障驱逐
// 一并写回 L2; 屏障后全线程非缓存读回, 四方 finisher 写同值 → 次序无关)
alignas(64) static volatile uint32_t sMtRendezvous[8];

static void mtRendezvousBarrier2(uint32_t verdict)
{
    volatile uint32_t* flags = sMtRendezvous;
    const uint32_t tid = get_thread_idx();
    if (tid == 0U) {
        flags[4] = verdict;             // PE0 publishes the verdict (same line)
    }
    mega_moe::mtCompilerBarrier();
    flags[tid] = 2U;                    // publish my arrival (phase 2)
    mega_moe::mtCompilerBarrier();
    // 写方到达驱逐: 立即定向驱逐一次, 使 L2 尽早看到本次到达 store
    // (flag 行所在 set 的同 set 偏移行)
    const uint32_t wordOff =
        (static_cast<uint32_t>(
            reinterpret_cast<uint64_t>(&flags[0]) >> 2)) & 4095u;
    for (uint32_t k = 1; k <= 5; ++k) {
        (void)sMtEvictSpan[k * 4096u + wordOff];
    }
    mega_moe::mtCompilerBarrier();
    // 等待全部到达 (非缓存轮询, 直读 L2)
    for (uint32_t t = 0U; t < mega_moe::kMtThreadsPerBlock; ++t) {
        while (flags[t] < 2U) {
        }
    }
    mega_moe::mtCompilerBarrier();
}

// ============================================================================
// [2026-10-15 v8 优化] 验证 golden 4-PE 并行分割 (mt_dyn 同款, 见该文件
// v8 块头注; 依据 = v7 pipe/swim: PE0 独占验证 ~280K cycle 期间 TID1-3 在
// 0x123d2 STD COND 自旋, 与 golden 块竞争共享 SPE 发射口)。
// mt 为单 cfg (kernel 后无后续 kernel/combine 行需要保护), compute_golden
// 本就为分块形态 —— 无 legacy 保护约束, 可安全分片。
// 结构: per-PE LUT 冗余构建 (mt 无 cfgA 轮刷洗可发布共享 LUT, 每 PE 自建
// 本地副本消除跨 PE 可见性问题) → decode 行分片 → 发布扫描 (mega_moe::
// sMtEvictSpan 96KB cached-load 扫描) → mtBarrier(5) → GEMM token 分片
// (栈上 scratch per-PE 天然私有; 分块形态逐字节复制自 compute_golden,
// 每元素累加序列不变 → yRef 逐位一致) → 发布扫描 → mtBarrier(6) →
// PE0 独占比较尾部 (tokRef + abs/rel 统计 + 诊断码, 逐位同语义)。
// 相位 5/6 空闲 (已用: 1=驱动发布, 2/3/4=kernel 内部); mtBarrier 为写方
// 定向驱逐版, 安静窗口安全 (见其实现注)。
// ============================================================================
static double sV8LutPE[4][256];

static inline void v8SliceMt(uint32 n, uint32 me, uint32& begin, uint32& end)
{
    const uint32 seg = n / 4U;
    const uint32 rem = n % 4U;
    begin = me * seg + (me < rem ? me : rem);
    end = begin + seg + (me < rem ? 1U : 0U);
}

static void v8PublishSweepMt()
{
    const uint32_t* p = reinterpret_cast<const uint32_t*>(
        reinterpret_cast<uintptr_t>(&mega_moe::sMtEvictSpan[0]));
    volatile uint32_t sink = 0U;
    const uint32_t lines =
        static_cast<uint32_t>(sizeof(mega_moe::sMtEvictSpan) / 64U);
    for (uint32_t i = 0U; i < lines; ++i) {
        sink = p[i * 16U];
    }
    (void)sink;
}

static void compute_golden_decode_slice_mt(uint32 tid)
{
    double* const lut = sV8LutPE[tid];
    for (uint32 b = 0U; b < 256U; ++b) {
        lut[b] = ref_fp8_e4m3(static_cast<double>(b));
    }
    {
        const uint32 rows = 2U * kH;
        uint32 rb, re;
        v8SliceMt(rows, tid, rb, re);
        for (uint32 r = rb; r < re; ++r) {
            const uint32 e = r / kH;
            const uint32 k = r % kH;
            const double scD =
                ref_wscale(g_mmWeightScales1[e * (kH / 32U) + k / 32U]);
            double* dst = g_w1Dbl + e * kH * kHiddenDim + k * kHiddenDim;
            const uint8_t* src =
                g_mmWeight1 + e * kH * kHiddenDim + k * kHiddenDim;
            uint32 n = 0U;
            for (; n + 4U <= kHiddenDim; n += 4U) {
                dst[n] = lut[src[n]] * scD;
                dst[n + 1U] = lut[src[n + 1U]] * scD;
                dst[n + 2U] = lut[src[n + 2U]] * scD;
                dst[n + 3U] = lut[src[n + 3U]] * scD;
            }
            for (; n < kHiddenDim; ++n) {
                dst[n] = lut[src[n]] * scD;
            }
        }
    }
    {
        const uint32 rows = 2U * (kHiddenDim / 2U);
        uint32 rb, re;
        v8SliceMt(rows, tid, rb, re);
        for (uint32 r = rb; r < re; ++r) {
            const uint32 e = r / (kHiddenDim / 2U);
            const uint32 k = r % (kHiddenDim / 2U);
            const double scD =
                ref_wscale(g_mmWeightScales2[e * (kHiddenDim / 64U) + k / 32U]);
            double* dst =
                g_w2Dbl + e * (kHiddenDim / 2U) * kH + k * kH;
            const uint8_t* src =
                g_mmWeight2 + e * (kHiddenDim / 2U) * kH + k * kH;
            uint32 n = 0U;
            for (; n + 4U <= kH; n += 4U) {
                dst[n] = lut[src[n]] * scD;
                dst[n + 1U] = lut[src[n + 1U]] * scD;
                dst[n + 2U] = lut[src[n + 2U]] * scD;
                dst[n + 3U] = lut[src[n + 3U]] * scD;
            }
            for (; n < kH; ++n) {
                dst[n] = lut[src[n]] * scD;
            }
        }
    }
}

static void compute_golden_gemm_slice_mt(double* yRef, uint32 tid)
{
    uint32 tb, te;
    v8SliceMt(kBS, tid, tb, te);
    for (uint32 t = tb; t < te; ++t) {
        const uint32 expert = (uint32)g_mmTopkIds[t];
        const double weight = (double)g_mmTopkWeights[t];
        const double* w1e = g_w1Dbl + expert * kH * kHiddenDim;
        const double* w2e = g_w2Dbl + expert * (kHiddenDim / 2U) * kH;

        double xrow[kH];
        for (uint32 k = 0; k < kH; ++k) {
            xrow[k] = (double)g_mmX[t * kH + k];
        }
        double y1[kHiddenDim];
        constexpr uint32 kNb = 16U;
        uint32 n0 = 0U;
        for (; n0 + kNb <= kHiddenDim; n0 += kNb) {
            double acc[kNb];
            for (uint32 j = 0; j < kNb; ++j) acc[j] = 0.0;
            for (uint32 k = 0; k < kH; ++k) {
                const double xv = xrow[k];
                const double* wk = w1e + k * kHiddenDim + n0;
                for (uint32 j = 0; j < kNb; ++j) {
                    acc[j] += xv * wk[j];
                }
            }
            for (uint32 j = 0; j < kNb; ++j) y1[n0 + j] = acc[j];
        }
        for (; n0 < kHiddenDim; ++n0) {   // 尾块 (防御性; 本 spec 不触达)
            double accT = 0.0;
            for (uint32 k = 0; k < kH; ++k) {
                accT += (double)g_mmX[t * kH + k] * w1e[k * kHiddenDim + n0];
            }
            y1[n0] = accT;
        }
        double y2[kHiddenDim / 2U];
        for (uint32 k = 0; k < kHiddenDim / 2U; ++k) {
            const double z = y1[k];
            const double sig = 1.0 / (1.0 + ref_exp(-z));
            y2[k] = z * sig * y1[k + kHiddenDim / 2U];
        }
        double y3[kH];
        uint32 m0 = 0U;
        for (; m0 + kNb <= kH; m0 += kNb) {
            double acc[kNb];
            for (uint32 j = 0; j < kNb; ++j) acc[j] = 0.0;
            for (uint32 k = 0; k < kHiddenDim / 2U; ++k) {
                const double yv = y2[k];
                const double* wk = w2e + k * kH + m0;
                for (uint32 j = 0; j < kNb; ++j) {
                    acc[j] += yv * wk[j];
                }
            }
            for (uint32 j = 0; j < kNb; ++j) y3[m0 + j] = acc[j];
        }
        for (; m0 < kH; ++m0) {   // 尾块 (防御性; 本 spec 不触达)
            double accT = 0.0;
            for (uint32 k = 0; k < kHiddenDim / 2U; ++k) {
                accT += y2[k] * w2e[k * kH + m0];
            }
            y3[m0] = accT;
        }
        for (uint32 n = 0; n < kH; ++n) {
            yRef[t * kH + n] = weight * y3[n];
        }
    }
}

// PE0 独占比较尾部 (与 main 原 PE0 分支尾部逐位同语义; yRef 已由 GEMM
// 分片全行覆盖写, 无需 compute_golden 的前置清零)
static int verify_tail_mt()
{
    double* yRef = g_goldenY;
    int64* tokRef = g_goldenTok;
    float* y = g_mmY;
    int64* tokenNumsOut = g_mmExpertTokenNums;

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

    double maxAbsErr = 0.0;
    double maxRelErr = 0.0;
    for (uint32 i = 0; i < kBS * kH; ++i) {
        const double err = (double)y[i] - yRef[i];
        const double absErr = err < 0 ? -err : err;
        if (absErr > maxAbsErr) maxAbsErr = absErr;
        const double denom = yRef[i] < 0 ? -yRef[i] : yRef[i];
        if (denom > 1e-6) {
            const double rel = err / denom;
            const double absRel = rel < 0 ? -rel : rel;
            if (absRel > maxRelErr) maxRelErr = absRel;
        }
    }

    const bool tokOk = (tokenNumsOut[0] == tokRef[0]) && (tokenNumsOut[1] == tokRef[1]);

    if (tokOk && maxAbsErr < 1e-2 && maxRelErr < 1e-2) {
        return 0;             // R2=0: PASS
    }
    if (tokenNumsOut[0] == -1 || tokenNumsOut[1] == -1) return 4;
    if (!tokOk) return 5;
    if (maxAbsErr < 0.05) return 6;
    if (maxAbsErr < 0.2) return 7;
    if (maxAbsErr < 0.5) return 8;
    return 9;
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

    // ---- 确定性数据生成 (按 PE 连续 ceil 分片; 镜像 mega_moe_sim_mt_dyn 的
    //      genInputs 模式) ----
    // [2026-10-09 优化] 原实现各 PE 冗余全量写 (~10.9K 迭代 × 4 PE), 其中
    // w1 的随机小值循环 (原第一遍) 被后续确定性细化循环逐元素完全覆盖 ——
    // 纯死代码, 一并删除。分片后每 PE 只写 1/4 (单写者/cacheline), kernel
    // 前 mtBarrier(1) 发布 (tile 量化解码读全量 g_mmWeight* 需要)。
    {
        const uint32 me = tid;
        // 连续 ceil 分片 [begin, end)
        auto slice = [me](uint32 n, uint32 &begin, uint32 &end) {
            const uint32 seg = n / 4U;
            const uint32 rem = n % 4U;
            begin = me * seg + (me < rem ? me : rem);
            end = begin + seg + (me < rem ? 1U : 0U);
        };
        // x: LCG 均匀 [-1,1] 混合边界
        {
            uint32 b, e;
            slice(kTotalElems, b, e);
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
        // 路由: 与 gen_data 同规则 topk_ids = (i//128)%2, 权重 1.0
        // 注: ids 经 volatile 写 —— BS16 等小规格下循环全展开, 前 kBS/2 个
        //     i32 零 store 会被后端合并为 16B 零 tile store (v2i64 崩溃)
        {
            uint32 b, e;
            slice(kBS, b, e);
            volatile int32_t* ids = g_mmTopkIds;
            for (uint32 t = b; t < e; ++t) {
                ids[t] = (int32_t)((t / (kBS / 2u)) % 2u);   // 与 gen_data 同规则
                g_mmTopkWeights[t] = 1.0f;
            }
        }
        // FP8 权重: 确定性细化, 值域 [-2,2] 内的 E4M3 可表示值; scale = 1.0
        // (E8M0 0x00 → 2^0)。原第一遍随机循环为死代码 (被本循环逐元素
        // 覆盖) 已删除 —— 值与原版逐位一致。
        // [2026-10-14 v7 优化] 填充模式周期化 + 模式表 (sim 版同款, 见
        // 该注): w1/w2 字节模式周期 280/520, 首次经增量递推构建 (递推与
        // 闭式公式逐位一致), 填充退化为表游走; 表构建按 PE 独立触发
        // (sPatReadyPE[me], 各 PE 冗余同值写幂等)。权重数据逐字节不变。
        {
            static uint8_t sW1Pat[280];
            static uint8_t sW2Pat[520];
            static bool sPatReadyPE[4];
            if (!sPatReadyPE[me]) {
                {
                    uint32 a = 0U, m = 1U, d = 7U, sg = 0U;
                    for (uint32 j = 0U; j < 280U; ++j) {
                        sW1Pat[j] =
                            (uint8_t)((sg << 7U) | ((4U + a) << 3U) | m);
                        a += 2U; if (a >= 5U) a -= 5U;       // (i*7)%5: 7≡2
                        m = (m + 3U) & 7U;                    // (i*3+1)&7
                        if (--d == 0U) { d = 7U; sg ^= 1U; }  // (i/7)&1
                    }
                }
                {
                    uint32 a = 2U, m = 3U, d = 13U, sg = 0U;
                    for (uint32 j = 0U; j < 520U; ++j) {
                        sW2Pat[j] =
                            (uint8_t)((sg << 7U) | ((4U + a) << 3U) | m);
                        a += 1U; if (a >= 5U) a -= 5U;       // (i*11+2)%5
                        m = (m + 5U) & 7U;                    // (i*5+3)&7
                        if (--d == 0U) { d = 13U; sg ^= 1U; } // (i/13)&1
                    }
                }
                sPatReadyPE[me] = true;
            }
            uint32 b, e;
            slice(2U * kHiddenDim * kH, b, e);
            {
                const uint8_t* p = sW1Pat + (b % 280U);
                for (uint32 i = b; i < e; ++i) {
                    g_mmWeight1[i] = *p++;
                    if (p == sW1Pat + 280U) p = sW1Pat;
                }
            }
            slice(2U * (kHiddenDim / 2U) * kH, b, e);
            {
                const uint8_t* p = sW2Pat + (b % 520U);
                for (uint32 i = b; i < e; ++i) {
                    g_mmWeight2[i] = *p++;
                    if (p == sW2Pat + 520U) p = sW2Pat;
                }
            }
            slice(2U * (kH * kHiddenDim / 32U + 4U), b, e);
            for (uint32 i = b; i < e; ++i) {
                g_mmWeightScales1[i] = 0x00;   // scale = 1.0
                g_mmWeightScales2[i] = 0x00;
            }
        }
    }

    // [2026-10-09 优化] kernel 前输入发布屏障 (相位 1; 镜像 mega_moe_sim_mt_dyn
    // 的 mtBarrierDyn(8c+1) 模式): genInputs 已按 PE 分片, kernel 内 tile 量化
    // 解码读全量 g_mmWeight* 须待全部分片写发布。
    // [A1 对齐 2026-10-13] kernel 内部相位: 2=dispatch 可见, 3=combine 可见,
    // 4=末端汇合 (PE0 随后导出 tokOut)。
    mega_moe::mtBarrier(1U);

    BENCHSTART;
    mega_moe::mega_moe_sim_mt_kernel<kBS, kH>(y, x, tokenNumsOut);
    BENCHEND;

    // [2026-10-04 修复] 退出 lockstep 搁浅 (gfsim `--conf fourpe` 确定性崩溃
    // 的根因, 实测 cycle=323,593 断言复现): 原实现 worker (tid!=0) 在 kernel
    // 末端 mtBarrier(1) 放行后直接 `return 0`, 先于 PE0 到达 _end 退出 ecall
    // 并 park 到退出 lockstep AND 组 (实测 ready=3/4, bpc=0x11458); PE0 独占
    // 的参考计算 + 比对 (compute_golden 16 token × GMM1/SwiGLU/GMM2 双精度
    // 参考) 使退出 AND 组停在 ready=3/4 超过 T_deadlock=9999 cycle →
    // SyscallBarrier.cpp:1889 LockstepWatchdog 断言中止仿真。"direct-boot
    // 下各 PE 退出相互独立" 的机制仅对功能模型 (gfrun) 成立, gfsim 时序
    // 模型的退出 ecall 仍按 lockstep AND 组汇聚。
    // 修复镜像 group_token_old_mt / group_token_vec_mt 的已证方案: 验证
    // 汇合屏障 (相位 2, kernel 内部用 1) —— 验证在 worker 存活期间执行
    // (worker 在汇合屏障自旋而非 park 到退出 lockstep), PE0 验证完成后
    // 全 PE 一起放行、同时到达 _end (消除 ready=3/4 长时停滞)。
    int ret = 0;
    uint32_t verdict = 0x5555u;
    // gfsim 判读通道: test-finisher (0x10009000, 低 16 位 0x5555 = PASS)
    volatile uint32_t* finisher = reinterpret_cast<volatile uint32_t*>(0x10009000ULL);
    // [2026-10-04 修复] gfsim 的 SPMD finisher 约定: 汇总判定行
    // ("linx_test_finisher write ... pass/fail") 按 "全部 threadCount
    // 线程都写过 finisher" 翻转 (SimSys commitTestFinisherBlock 的
    // SPMD assumption) —— 只 PE0 写时仿真停止但判定行不输出 (实测
    // exit_parks=3 无 finisher 行)。故 worker 也须写 finisher, 但写
    // 入时机必须在汇合屏障之后 (kernel 末端 tile 块在飞窗口): 实测
    // worker 在屏障前提前写会触发 per-thread terminate 路径与 kernel
    // 末端 nuke 恢复竞态叠加, 确定性丢块死锁 @ cycle 323,621。判定
    // 值经汇合行 sMtRendezvous[4] 传递 (与 flags 同 cacheline, 随
    // 屏障写方定向驱逐写回 L2), 屏障后全线程同值写 → 次序无关。
    //
    // [2026-10-15 v8 优化] 验证 golden 4-PE 并行分割 (mt_dyn 同款, 见
    // v8 块头注): 原 PE0 独占 compute_golden + 比对 (~280K cycle, v7
    // gfsim 日志: TID1-3 在 0x123d2 STD COND 自旋竞争共享 SPE 发射口)
    // 改为全 PE 分片 (per-PE LUT → decode 行分片 → 发布扫描 →
    // mtBarrier(5) → GEMM token 分片 → 发布扫描 → mtBarrier(6)),
    // 比较尾部 PE0 独占。golden 与误差统计逐位不变 (gfrun R2 佐证)。
    {
        compute_golden_decode_slice_mt(tid);
        v8PublishSweepMt();
        mega_moe::mtBarrier(5U);
        compute_golden_gemm_slice_mt(g_goldenY, tid);
        v8PublishSweepMt();
        mega_moe::mtBarrier(6U);
    }
    if (tid == 0U) {
        ret = verify_tail_mt();
        verdict = (ret == 0) ? 0x5555u : 0x0001u;
    }  // --- PE0 验证结束 ---

    // [2026-10-04 修复] 验证汇合屏障 (相位 2, kernel 内部用 1): PE0 的长验证
    // 完成前 worker 在此自旋 (存活、不触退出 lockstep), 完成后全 PE 一起
    // 放行到 _end —— 消除退出 AND 组 ready=3/4 长时停滞。
    // [2026-10-04 修复] 汇合采用写方定向驱逐屏障 (mtRendezvousBarrier2,
    // 见其实现注): 安静窗口下裸 volatile flag 屏障存在确定性陈旧 flag
    // 活锁 (脏 flag 行滞留各 PE 私有 L1D, 非缓存轮询读 L2 永远看不到
    // 发布值, 实测四线程全部到达屏障却互相等待 → ResVerify 死锁);
    // 写方到达定向驱逐使发布值立即写回 L2, 与 group_token_vec_mt.hpp
    // 驱逐屏障同机制 (gfsim fourpe 实证可靠)。
    // [2026-10-04 注] 曾试"寄存器乘法延迟链礼貌轮询"变体: 在 gfsim fourpe
    // (标量侧 1 SPE + 4 SMT) 下于 cycle≈1.13M 触发 SMT 流水停摆, 已弃用;
    // 验证长尾由 golden 解码备忘录化 (见 compute_golden 注) 缩短 ~5×,
    // 自旋窗口与轮询流量随之大幅缩小。
    mtRendezvousBarrier2(verdict);

    // [2026-10-04 修复] SPMD finisher 约定: 全线程在汇合屏障后各写一次
    // finisher, 值取自汇合行 (PE0 发布、屏障驱逐保证可见) —— 四方同值,
    // 写入次序无关; 且全部发生在 kernel tile 块排空之后 (屏障后), 不与
    // 模型 nuke 恢复竞态窗口叠加。
    *finisher = sMtRendezvous[4];

    return ret;
}
