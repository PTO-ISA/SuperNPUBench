#ifndef QUANT_BATCH_MATMUL_E1M2_KERNEL_HPP
#define QUANT_BATCH_MATMUL_E1M2_KERNEL_HPP

#include <common/pto_tileop.hpp>
#include "single_thread/matmul/matmul_mx.hpp"

// ============================================================================
// quant_batch_matmul_e1m2 — e1m2x2 fp4 多 block MX matmul (e8m0/group-32)
//
// e1m2x2 是唯一支持且可用 MX 操作的另一种 fp4 格式。
// 注意：TileOP API f94bc12 下 MX scale 必须使用 __fp8_e8m0，
// uint32_t 不被 validate_matrix_scale_contract 接受。
// ============================================================================
template <const int blockM, const int gN, const int gKv,
          const int tM, const int tN, const int tKv, const int smatrix_wfactor = 32>
__attribute__((noinline)) void quant_batch_matmul_e1m2(
    float *dst_cat, __fp4_e1m2x2 *src0_cat, __fp4_e1m2x2 *src1,
    __fp8_e8m0 *src0_mx_cat, __fp8_e8m0 *src1_mx)
{
    static_assert(blockM % tM == 0); static_assert(gN % tN == 0); static_assert(gKv % tKv == 0);

    const uint32_t tid = get_thread_idx();
    const int Mb = blockM / tM, Nb = gN / tN, Kb = gKv / tKv;

    __fp4_e1m2x2 *src0 = src0_cat + tid * blockM * gKv;
    float  *dst     = dst_cat  + tid * blockM * gN;
    __fp8_e8m0 *src0_mx = src0_mx_cat + tid * blockM * (gKv / smatrix_wfactor);

    using gmA = global_tensor<__fp4_e1m2x2, RowMajor<blockM, gKv>>;
    using gmB = global_tensor<__fp4_e1m2x2, RowMajor<gKv, gN>>;
    using gmC = global_tensor<float, RowMajor<blockM, gN>>;
    using tileA = CubeTileA<__fp4_e1m2x2, tM, tKv>;
    using tileB = CubeTileN8<__fp4_e1m2x2, tKv, tN>;
    using tileC = TileAcc<float, tM, tN>;
    using tAS = Tile<Location::Scaling, __fp8_e8m0, tM, tKv,
                     BLayout::RowMajor, tM, tKv / smatrix_wfactor>;
    using tBS = Tile<Location::Scaling, __fp8_e8m0, tKv, tN,
                     BLayout::RowMajor, tKv / smatrix_wfactor, tN>;

    using itA = global_iterator<gmA, tileA>;
    using itB = global_iterator<gmB, tileB>;
    using itC = global_iterator<gmC, tileC>;
    using gmS = global_tensor<__fp8_e8m0, RowMajor<blockM, gKv / smatrix_wfactor>>;
    using gmT = global_tensor<__fp8_e8m0, RowMajor<gKv / smatrix_wfactor, gN>>;
    using itAS = global_iterator<gmS, tAS>;
    using itBS = global_iterator<gmT, tBS>;

    itA  gA(src0); itB  gB(src1); itC  gC(dst);
    itAS gAS(src0_mx); itBS gBS(src1_mx);

    for (int i = 0; i < Mb; ++i) {
        for (int j = 0; j < Nb; ++j) {
            tileC c;
#pragma clang loop unroll(full)
            for (int k = 0; k < Kb; ++k) {
                tileA a; tileB b; tAS sa; tBS sb;
                auto ga = gA(i,k); auto gb = gB(k,j);
                auto gsa = gAS(i,k); auto gsb = gBS(k,j);
                TLOAD_CUBE(a, ga); TLOAD_CUBE(b, gb);
                TLOAD(sa, gsa);    TLOAD(sb, gsb);
                if (k == 0) TMATMUL_MX(c, a, sa, b, sb, fixp::keep_acc());
                else        TMATMUL_MX_ACC(c, c, a, sa, b, sb, fixp::keep_acc());
            }
            auto gc = gC(i, j); TSTORE_CUBE(gc, c);
        }
    }
}
#endif
