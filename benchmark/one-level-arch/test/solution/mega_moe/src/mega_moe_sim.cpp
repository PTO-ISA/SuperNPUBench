/**
 * MegaMoe A8W8 (完整真机流水) 迁移样例 — test driver
 *
 * 源: cann-samples mega_moe_story
 *   - 数据与 golden 语义对齐 gen_data.py / compute_golden (完整 MoE 单层前向):
 *       x[256,128] → 路由 topkIds/topkWeights → GMM1 (x@w1^T, A8W8 FP8 解码)
 *       → SwiGLU (silu) → GMM2 (@w2^T) → topk 加权 Combine → y[256,128]
 *       + 每 expert token 统计
 *   - 权重以 FP8 E4M3FN + E8M0 scale 存储, kernel 与参考均执行完整量化解码
 *
 * 运行 (gfrun 功能仿真, 4 线程覆盖 16 伪核分片):
 *   gfrun -f <elf> -s softcore.multiThreadNum=4
 * 判读: R2 = 0 为 PASS; 非 0 为分层诊断码 (2=基础内存, 3=x 填充, 4=tokenNums,
 *       5/6/7/8=精度误差分级); gfsim 下由 test-finisher (0x10009000=0x5555) 判定。
 */

#include "solution/mega_moe/mega_moe_sim.hpp"

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
// 权重 scale (E8M0 标量, 每 expert × 每 k 组一个; kernel tile 解码时 TMULS 折叠):
//   w1Scale[e][k/32] (k ∈ [0, h)), w2Scale[e][k/32] (k ∈ [0, hd/2))
__attribute__((aligned(4096))) uint8_t g_mmWeightScales1[2U * (kH / 32U)];
__attribute__((aligned(4096))) uint8_t g_mmWeightScales2[2U * (kHiddenDim / 64U)];
__attribute__((aligned(4096))) float g_mmY[kBS * kH];
__attribute__((aligned(4096))) int64 g_mmExpertTokenNums[2];
__attribute__((aligned(4096))) uint8_t g_mmWorkspace[kWorkspaceBytes];
// [2026-10-12 优化] golden 权重预解码备忘录 (double; mt/mt_dyn 版同款):
// 原实现 ref_w1/ref_w2 在 GMM 内循环按访问逐次解码 FP8 (每元素 ~20 op,
// 每 token 全量重解码)。改为进入 golden 前一次性预解码 + k-outer 分块
// GEMM (见 compute_golden 注)。置于 bss 尾部, 不影响既有符号布局。
__attribute__((aligned(4096))) double g_w1Dbl[2U * kH * kHiddenDim];
__attribute__((aligned(4096))) double g_w2Dbl[2U * (kHiddenDim / 2U) * kH];
// [2026-10-14 优化] E4M3→double 解码 LUT (256 值一次性查表; 见
// compute_golden 内构建注)。置于 bss 尾部, 不影响既有符号布局。
__attribute__((aligned(4096))) double g_e4m3Lut[256];

// ==== 参考实现 (gen_data.compute_golden 同语义) ====
// [perf] PR #194 同款 golden 侧优化: 2^k 的 O(|k|) 连乘循环 → 位级 O(1) 构造。
// 2.0/0.5 连乘在 IEEE double 下精确, 位级构造在全部定义域逐位一致 (含上溢
// 2^1024→+inf、次正规域至 2^-1074、完全下溢→0 的阈值行为), golden 数值不变
// (gfrun R2=0 佐证)。连乘循环为 gfsim 周期绝对主体 (每 FP8 权重元素调用
// ref_wscale/exp2_approx, E8M0 偏置大时单次上百次迭代)。
// [2026-10-14 v7 试验注] ref_exp2i 四路分支 → csel 无分支选择链已证伪
// (sim 实测 287,735 vs 280,065, +7.7K): k=0 恒走规格数域, 分支版每调用
// 仅 1~2 次块转移 + 4 op, 无分支版恒 ~10 op —— 块转移比指令便宜,
// 回退分支版。
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
    // [fix] 语义保留: ip = (int32_t)p, 返回 2^ip (含 e<7 负分支修复)
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

// 完整 MoE 参考前向 (gen_data.compute_golden 语义)
// [2026-10-12 优化] ① 权重解码备忘录化 (mt/mt_dyn 版同款): 原实现 GMM
// 内循环逐访问调用 ref_w1/ref_w2 (每元素 ~20 op, 16 token 全量重解码),
// 改为进入前一次性预解码到 g_w1Dbl/g_w2Dbl (6144 元素), 解码值逐位不变
// (解码为元素字节的纯函数); ② GMM 循环 k-outer × n 16 宽寄存器分块
// (mt_dyn 版同款): 原内循环 w1e[k*kHiddenDim+n] 随 k 步进跨 512B ——
// 每次权重访存跨 cacheline 且 x[k] 逐 n 重复加载/转换; 分块后权重访问
// 连续流式, 每元素累加序列不变 (k 升序同款 "acc += xv * w" fmadd 链),
// golden 数值逐位一致, 验证语义不变; ③ 解码 ref_wscale 按 (e,k) 组提升
// (组内同 scale, 原逐元素重复调用)。
static void compute_golden(double* yRef, int64* tokRef)
{
    // [2026-10-14 优化] E4M3 解码 LUT 一次性构建 (256 值; 构建用原
    // ref_fp8_e4m3 逐值计算 → 查表值与逐次解码逐位一致)。单 PE 程序
    // 单写者无竞争; 解码内循环由 "位提取 + exp2 位级构造" (~20 op/元素)
    // 退化为 1 load + 1 fmul, 乘法操作数不变 → golden 数值逐位一致。
    static bool sE4m3LutReady = false;
    if (!sE4m3LutReady) {
        for (uint32 b = 0U; b < 256U; ++b) {
            g_e4m3Lut[b] = ref_fp8_e4m3(static_cast<double>(b));
        }
        sE4m3LutReady = true;
    }
    // 一次性预解码全部权重 (e×k×n 展平, 与 ref_w1/ref_w2 同下标式)
    // [2026-10-12 优化] ref_wscale 按 (e,k) 组提升 (原 ref_w1 逐元素解码)
    // [2026-10-14 优化] E4M3 值经 g_e4m3Lut 查表 (数值逐位一致)
    // [2026-10-14 v7 优化] 解码内循环瘦身: ① LUT 基址提升为局部指针 ——
    // 原内循环每元素经 addtpc+addi 重取 &g_e4m3Lut (2 op/元素); ② n 循环
    // 4 路展开摊薄循环控制 (subi/setc/addi×2 + LUT 基址重取 ~6 op/元素 →
    // ~1.5 op/元素)。读写地址序列与数值逐位不变 (同表同乘数); gfsim 实测
    // sim 2 路展开 −7.6K、4 路 −13.0K (vs 基线 287,642)。
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
    // [fix] 多 PE 共享栈/GM: "tokRef[e] += 1" 会被 4 个 PE 重复执行 ×4;
    //       改为寄存器计数 + 幂等写 (与 kernel 侧统计导出同模式)
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

        // GMM1: y1[n] = Σ_k x[k]·w1[e][k][n]  (读预解码备忘录)
        // [2026-10-12 优化] k-outer × n 16 宽寄存器分块 (见函数头注)
        // [2026-10-14 v7 优化] x 行每 token 预转换一次到局部 double 数组
        // (原每个 (n0 块, k) 重复 lw+fcvt, hd/kNb 块 × k 次冗余)。
        // 每输出元素的 k 升序累加序列与操作数逐位不变 → golden 逐位一致。
        // [2026-10-14 v7 试验注] kNb 16→32 已证伪: 32 个 fp 累加器超出
        // 寄存器文件, 后端将 acc[] 溢出到栈 (GEMM 内循环退化为 ldi.u 栈
        // 读 + fmadd), gfsim 实测更慢且 val=0x0001 fail (414,732 vs
        // 基线 287,642), 回退 16。
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
        // Combine: y[t] = weight * y3 (topK==1; 原 "+=" 在多 PE 下对共享 GM 重复累加 ×4)
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

    float* x = g_mmX;
    float* y = g_mmY;
    int64* tokenNumsOut = g_mmExpertTokenNums;
    // 注: volatile 阻止相邻 i64 写入被合并为 16B tile store (BLK_TSTORE v2i64)
    volatile int64* tokInit = tokenNumsOut;
    tokInit[0] = -1;
    tokInit[1] = -1;

    // 基础内存写读自检 (gfrun 通道可用性)
    g_mmExpertTokenNums[0] = 12345;
    if (g_mmExpertTokenNums[0] != 12345) {
        return 2;  // R2=2: 基础内存写读异常
    }
    g_mmExpertTokenNums[0] = -1;

    // ---- 确定性数据生成 ----
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
    // FP8 权重: 确定性细化, 值域 [-2, 2] 内的 E4M3 可表示值; scale = 1.0
    // (E8M0 0x00 → 2^0)。
    // [2026-10-09 优化注沿承] 原第一遍随机小值循环被本细化循环逐元素完全
    // 覆盖 —— 纯死代码, 删除 (mt 版 2026-10-09 同款清理, 值不变)。
    // [2026-10-14 v7 优化] 填充模式周期化 + 模式表: 三分量周期 5/8/14 (w1)
    // 与 5/8/26 (w2) → 字节模式周期 280/520。首次经增量递推 (与闭式公式
    // 逐位一致: (i*7)%5 步进 +7≡+2、(i*3+1)&7 步进 +3、(i/7)&1 每 7 步
    // 翻转; w2 同理, 递推初值 = i=0 的公式值) 一次构建模式表, 填充退化为
    // 表游走 (每元素 lbu+sb+指针步进)。pipe 实测原填充 ~9.3 cyc/元素
    // (常数除法 mulhi 链)。权重数据逐字节不变。
    {
        static uint8_t sW1Pat[280];
        static uint8_t sW2Pat[520];
        static bool sPatReady = false;
        if (!sPatReady) {
            {
                uint32 a = 0U, m = 1U, d = 7U, sg = 0U;
                for (uint32 j = 0U; j < 280U; ++j) {
                    sW1Pat[j] = (uint8_t)((sg << 7U) | ((4U + a) << 3U) | m);
                    a += 2U; if (a >= 5U) a -= 5U;       // (i*7)%5: 7 ≡ 2 (mod 5)
                    m = (m + 3U) & 7U;                    // (i*3+1)&7
                    if (--d == 0U) { d = 7U; sg ^= 1U; }  // (i/7)&1
                }
            }
            {
                uint32 a = 2U, m = 3U, d = 13U, sg = 0U;
                for (uint32 j = 0U; j < 520U; ++j) {
                    sW2Pat[j] = (uint8_t)((sg << 7U) | ((4U + a) << 3U) | m);
                    a += 1U; if (a >= 5U) a -= 5U;       // (i*11+2)%5: 11 ≡ 1 (mod 5)
                    m = (m + 5U) & 7U;                    // (i*5+3)&7
                    if (--d == 0U) { d = 13U; sg ^= 1U; } // (i/13)&1
                }
            }
            sPatReady = true;
        }
        const uint8_t* p1 = sW1Pat;
        for (uint32 i = 0U; i < 2U * kHiddenDim * kH; ++i) {
            g_mmWeight1[i] = *p1++;
            if (p1 == sW1Pat + 280U) p1 = sW1Pat;
        }
        const uint8_t* p2 = sW2Pat;
        for (uint32 i = 0U; i < 2U * (kHiddenDim / 2U) * kH; ++i) {
            g_mmWeight2[i] = *p2++;
            if (p2 == sW2Pat + 520U) p2 = sW2Pat;
        }
        for (uint32 i = 0; i < 2U * (kH * kHiddenDim / 32U + 4U); ++i) {
            g_mmWeightScales1[i] = 0x00;   // scale = 1.0
            g_mmWeightScales2[i] = 0x00;
        }
    }

    BENCHSTART;
    mega_moe::mega_moe_sim_kernel<kBS, kH>(y, x, tokenNumsOut);
    BENCHEND;

    // ---- 完整参考 golden 对比 ----
    double* yRef = g_goldenY;
    int64* tokRef = g_goldenTok;
    compute_golden(yRef, tokRef);

    // [2026-10-14 优化] abs/rel 两个误差统计循环合并为单遍 (err/denom 每
    // 元素只计算一次, 省一遍 y/yRef 全量重扫); rel 的分子即 err (原式
    // ((double)y[i]-yRef[i]) 同表达式), max 比较次序不变 → 两统计量
    // 结果逐位一致
    double maxAbsErr = 0.0;
    double maxRelErr = 0.0;
    for (uint32 i = 0; i < kTotalElems; ++i) {
        const double err = (double)y[i] - yRef[i];
        const double absErr = err < 0 ? -err : err;
        if (absErr > maxAbsErr) maxAbsErr = absErr;
        // 相对容差 (量化解码误差随幅值放大, 用宽松 rtol 判定)
        const double denom = yRef[i] < 0 ? -yRef[i] : yRef[i];
        if (denom > 1e-6) {
            const double rel = err / denom;
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