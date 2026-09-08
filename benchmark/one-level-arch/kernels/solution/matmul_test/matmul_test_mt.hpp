#ifndef MATMUL_TEST_MT_KERNEL_HPP
#define MATMUL_TEST_MT_KERNEL_HPP

#include <common/pto_tileop.hpp>

#if !defined(Batch)
#define Batch 1
#endif

using namespace pto;

// ============================================================================
// matmul_test_mt — 动态 shape 的 4-PE 多线程 fp16 GEMM 样例（PTO 一层+共享瓦片，
// PTO 0.58.3+）
//
// 语义: C[M,N] = A[M,K] x B[K,N]
//   - 输入 A / B 为 fp16 (__half)
//   - 累加 fp32（每 PE 私有 CUBE 累加器，CubeAccumulatorM16/M32）
//   - 输出 C 为 fp16 (__half)：模型要求 TCVT 源/目的 layout 一致（CUBE 源
//     被 NORM 目的断言拒绝），沿用 fa_2d_unroll_gmma 的 GM 往返：
//     TSTORE_CUBE(acc -> 每 PE 私有 fp32 GM scratch) -> TLOAD(Vec fp32)
//     -> TCVT(Vec fp16) -> TSTORE(GM fp16)。scratch 由调用方提供，
//     大小 >= 4 * (tM/4) * tN 个 float。
//
// 多 PE（4-PE cooperative / Group TMATMUL，PTO 0.58.3 契约）:
//   - A、B 通过 GM->Shared TLOAD（B.IOS）装入共享瓦片：A 为
//     SharedMatrixLeft[tM,tK]（普通 RowMajor 矩形），B 为
//     SharedMatrixRight[tK,tN]；
//   - 4 个 PE 协作执行 TMATMUL/TMATMUL_ACC：组内 group_M=tM 行按
//     cooperative_group_m_rows_per_pe 契约每个 PE 持有 kPeM =
//     (tM<=64 ? 16 : 32) 行切片，本 PE (tid) 写回
//     [i*tM + tid*kPeM, i*tM + (tid+1)*kPeM) 行；因此 tM 仅支持
//     64（kPeM=16）与 128（kPeM=32），恰好 tM == 4*kPeM；
//   - 宿主可见布局与仓库 multi_thread/matmul 一致：C 是完整的 [M,N] 矩阵。
//
// 动态 shape:
//   - 模板参数只有 tile size <tM, tN, tK>；gM/gN/gK 为运行时参数；
//   - 模型侧 Group TMATMUL 要求共享瓦片为静态有效形状（TMATMUL 拒绝动态
//     valid 形状），因此运行时要求 gM % tM == 0 && gN % tN == 0 &&
//     gK % tK == 0（不满足直接返回，无尾瓦片回退）。动态性体现在
//     M/N/K 不参与模板，可自由设定。
//
// 精度/性能验证:
//   - 构建: make TESTCASE=matmul_test_mt ...（见 multi_thread/matmul/Makefile）；
//   - 运行: gfrun -t 1 -s softcore.multiThreadNum=4 -f <elf>；
//   - 精度: res_check=on + verify_matmul_test.py（--multi-thread 追加
//     multithreadNum）。
// ============================================================================
template <int tM = 128, int tN = 64, int tK = 64>
void matmul_test_mt(__half *c_ptr, __half *a_ptr,
                    __half *b_ptr, float *scratch, int gM, int gN, int gK) {
  constexpr int kPeNum = 4;
  static_assert(tM == 64 || tM == 128,
                "4-PE row split requires tM == 64 (kPeM=16) or tM == 128 "
                "(kPeM=32) per the cooperative group_M contract");
  constexpr int kPeM = tM / kPeNum;  // 本 PE 负责的行数（= RowsPerPE 契约值）

  if (gM <= 0 || gN <= 0 || gK <= 0) return;
  // 模型侧限制：共享瓦片不支持动态有效形状 -> 要求整除。
  if (gM % tM != 0 || gN % tN != 0 || gK % tK != 0) return;

  const uint32_t tid = get_thread_idx();

  using gmA = global_tensor<__half, RowMajor<-1, -1>>;
  using gmB = global_tensor<__half, RowMajor<-1, -1>>;
  using gmC = global_tensor<__half, RowMajor<-1, -1>>;

  // 共享矩阵主操作数：普通 RowMajor 矩形（PTO 0.58.3 Shared 契约）。
  using tileAMatrix = SharedMatrixLeft<__half, tM, tK>;
  using tileBMatrix = SharedMatrixRight<__half, tK, tN>;
  using tileAShared = SharedTile<tileAMatrix>;
  using tileBShared = SharedTile<tileBMatrix>;
  // Group TMATMUL 每 PE 私有 CUBE 累加器 [kPeM, tN]。
  using tileAccM16 = CubeAccumulatorM16<float, kPeM, tN>;
  using tileAccM32 = CubeAccumulatorM32<float, kPeM, tN>;
  using tileAcc =
      std::conditional_t<(kPeM <= 16), tileAccM16, tileAccM32>;
  // GM 往返 + TCVT 的 Vec 中转 tile。
  using tileAccVec = Tile<Location::Vec, float, kPeM, tN, BLayout::RowMajor>;
  using tileC = Tile<Location::Vec, __half, kPeM, tN, BLayout::RowMajor>;
  using gmScratch = global_tensor<float, RowMajor<kPeM, tN>>;

  const int Mb = gM / tM;
  const int Nb = gN / tN;
  const int Kb = gK / tK;

  for (int b = 0; b < Batch; ++b) {
    __half *a_base = a_ptr + (size_t)b * gM * gK;
    __half *b_base = b_ptr + (size_t)b * gK * gN;
    __half *c_base = c_ptr + (size_t)b * gM * gN;
    for (int i = 0; i < Mb; ++i) {
      for (int j = 0; j < Nb; ++j) {
        tileAcc tACC;

        for (int k = 0; k < Kb; ++k) {
          gmA gA(a_base + ((size_t)i * tM) * gK + (size_t)k * tK, gM, gK);
          gmB gB(b_base + ((size_t)k * tK) * gN + (size_t)j * tN, gK, gN);
          tileAShared tAShared;
          tileBShared tBShared;
          TLOAD<tileAMatrix, 1>(tAShared, gA);
          TLOAD<tileBMatrix, 1>(tBShared, gB);
          if (k == 0) {
            TMATMUL(tACC, tAShared, tBShared);
          } else {
            TMATMUL_ACC(tACC, tACC, tAShared, tBShared);
          }
        }

        // 每 PE 只转换并写回自己的 [kPeM, tN] 行切片（fp32 GM scratch 往返）。
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

#endif  // MATMUL_TEST_MT_KERNEL_HPP
