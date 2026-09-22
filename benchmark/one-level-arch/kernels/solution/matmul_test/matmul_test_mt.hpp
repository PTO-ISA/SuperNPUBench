#ifndef MATMUL_TEST_MT_KERNEL_HPP
#define MATMUL_TEST_MT_KERNEL_HPP

#include <common/pto_tileop.hpp>

#if !defined(Batch)
#define Batch 1
#endif

using namespace pto;

// ============================================================================
// matmul_test_mt — 动态 shape 的 4-PE 多线程 fp16 GEMM 样例（PTO 一层+共享瓦片）
//
// 语义: C[gM,gN] = A[gM,gK] x B[gK,gN]（fp16 输入 / fp32 累加 / fp16 输出）
//
// 接口约定（09-20 起）：gM/gN 为**填充后**维度（Mpad/Npad），gK 为精确 K：
//   - M/N 尾块由调用方零填充（Mpad = ceil(M/tM)*tM、Npad = ceil(N/tN)*tN，
//     A [Mpad, gK] 行距 gK、B [gK, Npad] 行距 Npad、C [Mpad, Npad] 行距 Npad，
//     填充区恒 0 -> 计算得精确 0，外部只读逻辑 [M, N]）；
//   - 内核退化为精确整除版（gM%tM==0 && gN%tN==0 && gK%tK==0，不满足返回），
//     调用方以 constexpr 计算填充维度传入（调用点常量折叠）。
//
// tile 约束（模型侧硬约束，编译期拦截）:
//   - tM ∈ {64, 128}（cooperative group_M 契约；组内均匀 4xkPeM 切分）；
//   - tM/tN/tK 必须为 2 的幂：cooperative TMATMUL 要求 group_M/N/K/dstCol
//     全部 2 的幂（gfrun "ADR-0100" 断言），且 Shared TLOAD 要求行位宽
//     （physicalCols x elementBits）整除 2 的幂容量（ExecuteSharedTMA 断言）。
//
// 输出转换: fp32 CUBE 累加器 -> fp16 经 GM 往返（TSTORE_CUBE -> scratch ->
// TLOAD Vec fp32 -> TCVT -> TSTORE），scratch >= 4 * kPeM * tN 个 float。
//
// PTO #257（TileOP 697f5d8 起）: B 声明 SharedMatrixRight<tK,tN> 后必须带
// TransB（GM 中 B 为自然 [K,N] 行优先）；经 3 参重载模板参数传递（Options
// 对象重载 + 动态循环会触发 llvm 寄存器合并崩溃，见验证记录 09-17）。
//
// K 链 CCTRL（PTO-ISA #236，matmul_shared 同款）：begin=raw_acc /
// middle=raw_acc|acc_hint / end=acc_hint。功能语义不变（NOSAT），但 gfsim
// 需据此（配 -s cube.enable_internal_acc=true）把累加器链放进 cube 内部
// 累加寄存器——否则每个 TMATMUL_ACC 的改名 acc 瓦片驻留 tile RF，大瓦片
// 长 K 链会耗尽 256KiB/PE 容量。
// ============================================================================

template <int tM = 128, int tN = 64, int tK = 64>
void matmul_test_mt(__half *c_ptr, __half *a_ptr,
                    __half *b_ptr, float *scratch, int gM, int gN, int gK) {
  constexpr int kPeNum = 4;
  static_assert(tM == 64 || tM == 128,
                "4-PE row split requires tM == 64 (kPeM=16) or tM == 128 "
                "(kPeM=32) per the cooperative group_M contract");
  static_assert((tN & (tN - 1)) == 0,
                "tN must be a power of two: the model's shared TLOAD requires "
                "the row bit-width to divide a pow2 capacity, and the "
                "cooperative TMATMUL requires pow2 N (ADR-0100)");
  static_assert((tK & (tK - 1)) == 0,
                "tK must be a power of two (model ADR-0100 / shared row "
                "granularity constraints)");
  constexpr int kPeM = tM / kPeNum;  // 本 PE 负责的行数（均匀切分）

  if (gM <= 0 || gN <= 0 || gK <= 0) return;
  // 调用方保证填充后整除；此处保留契约检查（不满足直接返回）。
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
          // 双变体 K 链：begin=raw_acc，其余=hint（matmul_shared 完整链的
          // 双变体近似——middle=raw|hint / end=hint / Kb==1 特例会引入更多
          // 变体，触发 llvm 寄存器合并崩溃）。
          constexpr FixpAttr kOptBegin = FixpAttr{}.transpose_b()
              .with_cube_ctrl(CubeCtrlRawAccumulator);
          constexpr FixpAttr kOptChain = FixpAttr{}.transpose_b()
              .with_cube_ctrl(CubeCtrlRawAccAndHint);
          if (k == 0) {
            TMATMUL<kOptBegin>(tACC, tAShared, tBShared);
          } else {
            TMATMUL_ACC<kOptChain>(tACC, tACC, tAShared, tBShared);
          }
        }

        // 每 PE 只转换并写回自己的 [kPeM, tN] 行切片（fp32 GM scratch 往返）。
        {
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
}

#endif  // MATMUL_TEST_MT_KERNEL_HPP
