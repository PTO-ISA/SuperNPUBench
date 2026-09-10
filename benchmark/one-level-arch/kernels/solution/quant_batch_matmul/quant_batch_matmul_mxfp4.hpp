#ifndef QUANT_BATCH_MATMUL_MXFP4_KERNEL_HPP
#define QUANT_BATCH_MATMUL_MXFP4_KERNEL_HPP

#include <common/pto_tileop.hpp>

#ifndef Batch
#define Batch 1
#endif

using namespace pto;

// ============================================================================
// quant_batch_matmul_mxfp4 — e2m1x2 fp4 多 block MX matmul（PTO 一层编程）
//
// 每个 PE（NBLOCKS 个）独立计算 M 维的一段：
//   block tid 处理 rows [tid*blockM, (tid+1)*blockM)
//   A 和 C 是 PE-local（按 block_id 偏移），B 在所有 block 间共享。
//
// A/B 为 e2m1x2 fp4 打包（__fp4_e2m1x2，每字节 2 个元素），
// MX scaling (e8m0, smatrix_wfactor=32) 用 TLOAD 路径。
//
// 编译时尺寸：M/N/K 均需被 tile 整除。
// ============================================================================
template <typename dtypeA, const int blockM, const int gN, const int gKv,
          const int tM, const int tN, const int tKv,
          typename dtypeB = dtypeA, const int smatrix_wfactor = 32>
__attribute__((noinline)) void quant_batch_matmul_mxfp4(
    float *dst_cat, dtypeA *src0_cat, dtypeB *src1,
    __fp8_e8m0 *src0_mx_cat, __fp8_e8m0 *src1_mx)
{
    constexpr int kTileByteLimit = 4 * 1024;
    static_assert(blockM % tM == 0);
    static_assert(gN % tN == 0);
    static_assert(gKv % tKv == 0);
    static_assert(tM * tKv * sizeof(dtypeA) <= kTileByteLimit);
    static_assert(tM * tN  * sizeof(float)   <= kTileByteLimit);
    static_assert(tKv * tN * sizeof(dtypeB) <= kTileByteLimit);

    const uint32_t tid = get_thread_idx();
    const int Mb = blockM / tM;
    const int Nb = gN / tN;
    const int Kb = gKv / tKv;

    dtypeA     *src0    = src0_cat    + tid * blockM * gKv;
    float      *dst     = dst_cat     + tid * blockM * gN;
    __fp8_e8m0 *src0_mx = src0_mx_cat + tid * blockM * (gKv / smatrix_wfactor);

    using gmA  = global_tensor<dtypeA, RowMajor<blockM, gKv>>;
    using gmB  = global_tensor<dtypeB, RowMajor<gKv, gN>>;
    using gmC  = global_tensor<float,  RowMajor<blockM, gN>>;

    using tileA = Tile<Location::Left,  dtypeA, tM, tKv, BLayout::CubeM32, tM, tKv, SLayout::NoneBox>;
    using tileB = Tile<Location::Right, dtypeB, tKv, tN, BLayout::CubeN8,  tKv, tN, SLayout::NoneBox>;
    using tileC = Tile<Location::Vec,   float,  tM, tN, BLayout::CubeM32, tM, tN, SLayout::NoneBox>;

    using itA = global_iterator<gmA, tileA>;
    using itB = global_iterator<gmB, tileB>;
    using itC = global_iterator<gmC, tileC>;

    itA gIterA(src0);
    itB gIterB(src1);
    itC gIterC(dst);

    using gmAMX  = global_tensor<__fp8_e8m0, RowMajor<blockM, gKv / smatrix_wfactor>>;
    using gmBMX  = global_tensor<__fp8_e8m0, RowMajor<gKv / smatrix_wfactor, gN>>;
    using tileAMX = Tile<Location::Scaling, __fp8_e8m0, tM, tKv,
                         BLayout::RowMajor, tM, tKv / smatrix_wfactor, SLayout::NoneBox>;
    using tileBMX = Tile<Location::Scaling, __fp8_e8m0, tKv, tN,
                         BLayout::RowMajor, tKv / smatrix_wfactor, tN, SLayout::NoneBox>;
    using itAMX = global_iterator<gmAMX, tileAMX>;
    using itBMX = global_iterator<gmBMX, tileBMX>;

    itAMX gIterAMX(src0_mx);
    itBMX gIterBMX(src1_mx);

    for (int i = 0; i < Mb; ++i) {
        for (int j = 0; j < Nb; ++j) {
            tileC tC;
#pragma clang loop unroll(full)
            for (int k = 0; k < Kb; ++k) {
                auto gA = gIterA(i, k);
                auto gB = gIterB(k, j);
                auto gAMX = gIterAMX(i, k);
                auto gBMX = gIterBMX(k, j);
                tileA tA;
                tileB tB;
                tileAMX tAMX;
                tileBMX tBMX;

                TLOAD(tA, gA);
                TLOAD(tB, gB);
                TLOAD(tAMX, gAMX);
                TLOAD(tBMX, gBMX);

                TMATMUL_MX(tC, tA, tAMX, tB, tBMX);
            }
            auto gC = gIterC(i, j);
            TSTORE_CUBE(gC, tC);
        }
    }
}

#endif  // QUANT_BATCH_MATMUL_MXFP4_KERNEL_HPP
