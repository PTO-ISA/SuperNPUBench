#ifndef MATMUL_TEST_KERNEL_HPP
#define MATMUL_TEST_KERNEL_HPP

#include <common/pto_tileop.hpp>

#ifndef Batch
#define Batch 1
#endif

using namespace pto;

// ============================================================================
// matmul_test — 动态 shape 的 fp16 GEMM 样例（PTO 一层编程模型，PTO 0.58.3+）
//
// 契约:
//   1. 模板参数只有 tile size (tM, tN, tK)；M/N/K 全部为运行时参数 (gM/gN/gK)。
//   2. 输入 A[M,K] / B[K,N] 为 fp16 (__half)；累加在 fp32（CUBE 累加器）；
//      输出 C[M,N] 为 fp16 (__half)。
//   3. 接口带 leading dimension（lda/ldb/ldc）。
//   4. PTO 0.58.3 起 TMATMUL 强制 CUBE CELL layout（A/D=CUBE_M16/M32、
//      B=CUBE_N8）且拒绝动态 valid 形状（"Matrix dynamic valid shapes are not
//      supported"），因此尾块改为零填充（padding）方案：
//        - Mpad = ceil(gM/tM)*tM, Kpad = ceil(gK/tK)*tK, Npad = ceil(gN/tN)*tN；
//        - A 缓冲 [Mpad, Kpad]（lda >= Kpad）：gM..Mpad 行与 gK..Kpad 列清零；
//        - B 缓冲 [Kpad, Npad]（ldb >= Npad）：gK..Kpad 行与 gN..Npad 列清零；
//        - C 缓冲 [Mpad, Npad]（ldc >= Npad）：填充区写入 0，外部只读 [gM,gN]。
//      零填充不改变累加结果，故逻辑结果仍为 C[0:gM,0:gN] = A*B。
//   5. fp32 CUBE 累加器 -> fp16 输出：模型要求 TCVT 源/目的 layout 一致
//      （CUBE 源会被 NORM 目的的断言拒绝），因此沿用 fa_2d_unroll_gmma 的
//      GM 往返模式：TSTORE_CUBE(acc -> GM fp32 scratch) -> TLOAD(Vec fp32)
//      -> TCVT(Vec fp16) -> TSTORE(GM fp16)。scratch 由调用方提供，
//      大小 >= tM*tN 个 float。
//   6. 精度/性能验证:
//        - 精度: res_check=on 构建后由 verify_matmul_test.py 与 fp32 中间
//          golden（torch/numpy）比对；
//        - 性能: BENCHSTART/BENCHEND 标记 + gfsim 周期统计，或 gfrun 的
//          block/inst 工作量统计。
// ============================================================================
template <int tM = 16, int tN = 16, int tK = 16>
using MatmulTestCubeA = std::conditional_t<
    (tM <= 16), CubeTileM16<__half, tM, tK>, CubeTileM32<__half, tM, tK>>;

template <int tM = 16, int tN = 16, int tK = 16>
using MatmulTestCubeAcc = std::conditional_t<
    (tM <= 16), CubeAccumulatorM16<float, tM, tN>,
    CubeAccumulatorM32<float, tM, tN>>;

template <int tM = 16, int tN = 16, int tK = 16>
__attribute__((noinline)) void matmul_test(__half *dst, __half *src0,
                                            __half *src1, float *scratch,
                                            int gM, int gN, int gK,
                                            int lda, int ldb, int ldc) {
  using gm_shapeA = global_tensor<__half, RowMajor<-1, -1>>;
  using gm_shapeB = global_tensor<__half, RowMajor<-1, -1>>;
  using gm_shapeC = global_tensor<__half, RowMajor<-1, -1>>;
  using gm_shapeScratch = global_tensor<float, RowMajor<tM, tN>>;

  // CUBE CELL layout 的静态满宽瓦片（TMATMUL 契约）。
  using tile_shapeA = MatmulTestCubeA<tM, tN, tK>;
  using tile_shapeB = CubeTileN8<__half, tK, tN>;
  using tile_shapeACC = MatmulTestCubeAcc<tM, tN, tK>;
  // GM 往返 + TCVT 的 Vec 中转 tile。
  using tile_shapeAccVec = Tile<Location::Vec, float, tM, tN, BLayout::RowMajor>;
  using tile_shapeC = Tile<Location::Vec, __half, tM, tN, BLayout::RowMajor>;

  if (gM <= 0 || gN <= 0 || gK <= 0) return;

  const int Mpad = ((gM + tM - 1) / tM) * tM;
  const int Kpad = ((gK + tK - 1) / tK) * tK;
  const int Npad = ((gN + tN - 1) / tN) * tN;

  for (int b = 0; b < Batch; b++) {
    __half *a_base = src0 + (size_t)b * Mpad * lda;
    __half *b_base = src1 + (size_t)b * Kpad * ldb;
    __half *c_base = dst + (size_t)b * Mpad * ldc;
    for (int i = 0; i < Mpad; i += tM) {
      for (int j = 0; j < Npad; j += tN) {
        tile_shapeACC tACC;

        for (int k = 0; k < Kpad; k += tK) {
          gm_shapeA gA(a_base + (size_t)i * lda + k, Mpad, lda);
          gm_shapeB gB(b_base + (size_t)k * ldb + j, Kpad, ldb);
          tile_shapeA tA;
          tile_shapeB tB;
          TLOAD_CUBE(tA, gA);
          TLOAD_CUBE(tB, gB);
          if (k == 0) {
            TMATMUL(tACC, tA, tB);
          } else {
            TMATMUL_ACC(tACC, tACC, tA, tB);
          }
        }

        // fp32 CUBE 累加器 -> fp16 GM：经 fp32 GM scratch 往返（FA 同款）。
        gm_shapeScratch gScratch(scratch);
        TSTORE_CUBE(gScratch, tACC);
        tile_shapeAccVec tAccVec;
        TLOAD(tAccVec, gScratch);
        tile_shapeC tC;
        TCVT(tC, tAccVec);
        gm_shapeC gC(c_base + (size_t)i * ldc + j, Mpad, ldc);
        TSTORE(gC, tC);
      }
    }
  }
}

#endif  // MATMUL_TEST_KERNEL_HPP
