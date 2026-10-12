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
// MX tile 路径 (issue #180): 无 fp32 权重缓存; 每 PE scratch =
// w1F16[epr*h*hd] + w2F16[epr*(hd/2)*h] + xf16[h] + y1[hd]f32 +
// y2f16[hd/2] + y3[h]f32 (kernel 头文件契约; 4 PE 槽位)
constexpr uint32 kScratchBytes = 4U * (
    (2U * kH * kHiddenDim + 2U * (kHiddenDim / 2U) * kH) * 2U +
    kH * 2U + kHiddenDim * 4U + kHiddenDim * 2U + kH * 4U);
constexpr uint32 kWorkspaceBytes =
    2U * kBS * 4U +                                // dispatch 表
    2U * kBS +                                     // mask
    kBS * kH * 4U +                                // combine 缓冲
    16U * 2U * 4U +                                // 核内统计
    kScratchBytes +                                // 每 PE tile scratch
    // [2026-10-04 修复] slack 4KB→16KB: 承载 x 预量化槽 (4 PE × myTokens×h×2B,
    // bs=16 时 1KB) + 裕量 (见 kernel 阶段 3 修复注)
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
    // 一次性预解码全部权重 (e×k×n 展平, 与 ref_w1/ref_w2 同下标式)
    for (uint32 e = 0; e < 2U; ++e) {
        for (uint32 k = 0; k < kH; ++k) {
            for (uint32 n = 0; n < kHiddenDim; ++n) {
                g_w1Dbl[e * kH * kHiddenDim + k * kHiddenDim + n] =
                    ref_w1(e, k, n);
            }
        }
    }
    for (uint32 e = 0; e < 2U; ++e) {
        for (uint32 k = 0; k < kHiddenDim / 2U; ++k) {
            for (uint32 n = 0; n < kH; ++n) {
                g_w2Dbl[e * (kHiddenDim / 2U) * kH + k * kH + n] =
                    ref_w2(e, k, n);
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
        double y1[kHiddenDim];
        for (uint32 n = 0; n < kHiddenDim; ++n) {
            double acc = 0.0;
            for (uint32 k = 0; k < kH; ++k) {
                acc += (double)g_mmX[t * kH + k] * w1e[k * kHiddenDim + n];
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
        // GMM2: y3[n] = Σ_k y2[k]·w2[e][k][n]  (读预解码备忘录)
        double y3[kH];
        for (uint32 n = 0; n < kH; ++n) {
            double acc = 0.0;
            for (uint32 k = 0; k < kHiddenDim / 2U; ++k) {
                acc += y2[k] * w2e[k * kH + n];
            }
            y3[n] = acc;
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
        {
            uint32 b, e;
            slice(2U * kHiddenDim * kH, b, e);
            for (uint32 i = b; i < e; ++i) {
                const uint32 e4 = 4U + ((i * 7U) % 5U);            // 4..8
                const uint32 m = ((i * 3U + 1U) & 0x7U);
                const uint32 s = (i / 7U) & 1U;
                g_mmWeight1[i] = (uint8_t)((s << 7U) | (e4 << 3U) | m);
            }
            slice(2U * (kHiddenDim / 2U) * kH, b, e);
            for (uint32 i = b; i < e; ++i) {
                const uint32 e4 = 4U + ((i * 11U + 2U) % 5U);
                const uint32 m = ((i * 5U + 3U) & 0x7U);
                const uint32 s = (i / 13U) & 1U;
                g_mmWeight2[i] = (uint8_t)((s << 7U) | (e4 << 3U) | m);
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
    // 解码读全量 g_mmWeight* 须待全部分片写发布。kernel 末端汇合相位为 2。
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
    if (tid != 0U) {
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
    } else {
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

    if (tokOk && maxAbsErr < 1e-2 && maxRelErr < 1e-2) {
        verdict = 0x5555u;   // PASS
        ret = 0;             // R2=0: PASS
    } else {
        verdict = 0x0001u;
        if (tokenNumsOut[0] == -1 || tokenNumsOut[1] == -1) ret = 4;  // R2=4: kernel 未写统计
        else if (!tokOk) ret = 5;                        // R2=5: expertTokenNums 数值不符
        else if (maxAbsErr < 0.05) ret = 6;            // 误差 [1e-2, 0.05)
        else if (maxAbsErr < 0.2) ret = 7;             // 误差 [0.05, 0.2)
        else if (maxAbsErr < 0.5) ret = 8;             // 误差 [0.2, 0.5)
        else ret = 9;                                  // 误差 >= 0.5
    }
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
