#ifndef QUANT_BATCH_MATMUL_HIF4_REUSEA_KERNEL_HPP
#define QUANT_BATCH_MATMUL_HIF4_REUSEA_KERNEL_HPP

#include <common/pto_tileop.hpp>
#include <cstdint>
#include <type_traits>

#include "basic_op/matmul/matmul_lowp_reuseA.hpp"

#ifndef Batch
#define Batch 1
#endif

using namespace pto;

// ============================================================================
// quant_batch_matmul_hif4_reuseA — reuseA 调度（对标 matmul_lowp_reuseA）。
//
// 直接复用 basic_op/matmul/matmul_lowp_reuseA（通用 lowp reuseA 模板），具备：
//   1. full-K A cache：A/scaleA 整条 K 轴驻留 CellReg，跨 N group 复用；
//   2. 驻留输出链限制（MAX_RESIDENT_OUTPUTS=2）：只驻留 2 个 C 累加器链，
//      N 分块执行，避免 N 大时累加器溢出 CellReg；
//   3. scaleA 用 CUBE_M32 布局（CubeTileM32），与 primary A 对齐（非 legacy
//      RowMajor/NORM 兼容路径）。
//
// 数据布局与旧实现一致：A [M, K/2] row-major、B [N, K/2] row-major（B-major）、
// A/scaleA 每 64 逻辑 K 一个 U32 scale word。前置条件（编译期）：
// gM%tM==0、gN%tN==0、gK%tK==0、tK%64==0、gM%64==0（M 无需 tail padding）。
// ============================================================================
template <int tM, int tN, int tK, int gM, int gN, int gK>
__attribute__((noinline)) void quant_batch_matmul_hif4_reuseA(
    float *dst, __fp4_hif4x2 *src0, __fp4_hif4x2 *src1,
    uint32_t *src0_mx, uint32_t *src1_mx)
{
    static_assert(tM == 64 || tM == 128, "4-PE cooperative requires tM == 64 or 128");
    static_assert(gM % tM == 0, "gM must be divisible by tM");
    static_assert(gN % tN == 0, "gN must be divisible by tN");
    static_assert(gK % tK == 0, "gK must be divisible by tK");
    static_assert(gK % 64 == 0 && tK % 64 == 0,
                  "HiF4 requires K and tK divisible by scale group (64)");
    static_assert(gM % 64 == 0,
                  "reuseA requires M padded to a multiple of 64 (no M tail)");

    constexpr int kPackedFactor = 2;
    constexpr int kScaleGroup = 64;
    constexpr int kStoredGK = gK / kPackedFactor;
    constexpr int kGScaleK = gK / kScaleGroup;
    constexpr int kInputM = gM;  // gM % 64 == 0, no tail padding

    for (int b = 0; b < Batch; ++b) {
        matmul_lowp_reuseA<__fp4_hif4x2, uint32_t, kPackedFactor, /*UseMx*/true,
                           kScaleGroup, gM, gN, gK, tM, tN, tK,
                           /*TailShared*/false>(
            dst + (size_t)b * gM * gN,
            src0 + (size_t)b * kInputM * kStoredGK,
            src1 + (size_t)b * kStoredGK * gN,
            src0_mx + (size_t)b * kInputM * kGScaleK,
            src1_mx + (size_t)b * kGScaleK * gN);
    }
}

#endif  // QUANT_BATCH_MATMUL_HIF4_REUSEA_KERNEL_HPP
