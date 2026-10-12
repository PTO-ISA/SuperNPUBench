#ifndef MATMUL_TEST_KERNEL_HPP
#define MATMUL_TEST_KERNEL_HPP

#include <common/pto_tileop.hpp>

#if !defined(Batch)
#define Batch 1
#endif

using namespace pto;

// ============================================================================
// matmul_test — 4-PE cooperative fp16 GEMM（A Local CellReg / B Shared）
//
// C[gM,gN] = A[gM,gK] × B[gK,gN]（fp16 in / fp32 acc / fp16 out）
//
// A（左矩阵）：每 PE 通过 TLOAD_CUBE 装载自己的 [kPeM, tK] 行切片到本地
//   CellReg（不同 PE 走不同带宽池，避免与 B 争 SharedTReg）。
// B（右矩阵）：PE0 单发 TLOAD 到 SharedTReg，4 PE 共读。
//
// 约束（模型侧，编译期拦截）：
//   tM ∈ {64,128}、tN/tK 为 2 的幂（ADR-0100 pow2 + shared 行粒度）
// ============================================================================

template <int tM = 128, int tN = 64, int tK = 64>
void matmul_test(__half *c_ptr, __half *a_ptr,
                 __half *b_ptr, float *scratch, int gM, int gN, int gK) {
  constexpr int kPeNum = 4;
  static_assert(tM == 64 || tM == 128,
                "tM must be 64 or 128 (cooperative group_M contract)");
  static_assert((tN & (tN - 1)) == 0, "tN must be pow2 (model constraint)");
  static_assert((tK & (tK - 1)) == 0, "tK must be pow2 (model constraint)");
  constexpr int kPeM = tM / kPeNum;

  if (gM <= 0 || gN <= 0 || gK <= 0) return;
  if (gM % tM != 0 || gN % tN != 0 || gK % tK != 0) return;

  const uint32_t tid = get_thread_idx();

  using gmA = global_tensor<__half, RowMajor<-1, -1>>;
  using gmB = global_tensor<__half, RowMajor<-1, -1>>;
  using gmC = global_tensor<__half, RowMajor<-1, -1>>;

  // A：每 PE 本地 CUBE 瓦片 [kPeM, tK]。
  using tileALocalM16 = CubeTileM16<__half, kPeM, tK>;
  using tileALocalM32 = CubeTileM32<__half, kPeM, tK>;
  using tileALocal =
      std::conditional_t<(kPeM <= 16), tileALocalM16, tileALocalM32>;

  // B：Shared 瓦片（PTO #257 [K,N] 物理 + TransB）。
  using tileBMatrix = SharedMatrixRight<__half, tK, tN>;
  using tileBSharedTile = SharedTile<tileBMatrix>;

  // 累加器 [kPeM, tN] + GM 往返转换 tile。
  using tileAccM16 = CubeAccumulatorM16<float, kPeM, tN>;
  using tileAccM32 = CubeAccumulatorM32<float, kPeM, tN>;
  using tileAcc =
      std::conditional_t<(kPeM <= 16), tileAccM16, tileAccM32>;
  using tileAccVec = Tile<Location::Vec, float, kPeM, tN, BLayout::RowMajor>;
  using tileC = Tile<Location::Vec, __half, kPeM, tN, BLayout::RowMajor>;
  using gmScratch = global_tensor<float, RowMajor<kPeM, tN>>;

  const int Mb = gM / tM;
  const int Nb = gN / tN;
  const int Kb = gK / tK;

  // K 链 CCTRL（双变体近似；PTO-ISA #236）。
  constexpr FixpAttr kOptBegin = FixpAttr{}.transpose_b()
      .with_cube_ctrl(CubeCtrlRawAccumulator);
  constexpr FixpAttr kOptChain = FixpAttr{}.transpose_b()
      .with_cube_ctrl(CubeCtrlRawAccAndHint);

  for (int b = 0; b < Batch; ++b) {
    __half *a_base = a_ptr + (size_t)b * gM * gK;
    __half *b_base = b_ptr + (size_t)b * gK * gN;
    __half *c_base = c_ptr + (size_t)b * gM * gN;
    for (int i = 0; i < Mb; ++i) {
      for (int j = 0; j < Nb; ++j) {
        tileAcc tACC;

        for (int k = 0; k < Kb; ++k) {
          // B → Shared（PE0 单发，全 PE 共读）。
          gmB gB(b_base + ((size_t)k * tK) * gN + (size_t)j * tN, gK, gN);
          tileBSharedTile tBShared;
          TLOAD<tileBMatrix, 1>(tBShared, gB);

          // A → 本地 CellReg（每 PE 独立装载自己的行切片）。
          gmA gA(a_base +
                     ((size_t)i * tM + (size_t)tid * kPeM) * gK +
                     (size_t)k * tK,
                 kPeM, gK);
          tileALocal tA;
          TLOAD_CUBE(tA, gA);

          // Local-A/Shared-B cooperative TMATMUL（groupM 运行时参数）。
          if (k == 0) {
            TMATMUL<kOptBegin>(tACC, tA, tBShared, (size_t)tM);
          } else {
            TMATMUL_ACC<kOptChain>(tACC, tACC, tA, tBShared, (size_t)tM);
          }
        }

        // fp32 acc → GM scratch → Vec → TCVT → fp16 C（每 PE 写回自己的行切片）。
        float *pe_scratch = scratch + (size_t)tid * kPeM * tN;
        gmScratch gScratch(pe_scratch);
        TSTORE_CUBE(gScratch, tACC);
        tileAccVec tAccVec;
        TLOAD(tAccVec, gScratch);
        tileC tC;
        TCVT(tC, tAccVec);
        gmC gC(c_base +
                   ((size_t)i * tM + (size_t)tid * kPeM) * gN +
                   (size_t)j * tN,
               gM, gN);
        TSTORE(gC, tC);
      }
    }
  }
}

#endif  // MATMUL_TEST_KERNEL_HPP
