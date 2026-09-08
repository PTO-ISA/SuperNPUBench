#ifndef SFA_PTO_HPP
#define SFA_PTO_HPP

// =============================================================================
// sfa.hpp — Sparse Flash Attention (block-sparse, PTO tile-op variant)
// =============================================================================
//
// 【计算语义】
//   O = softmax( (Q * K^T) / sqrt(qD) ) * V          （仅在稀疏块上求和）
//   Q: [Sq, qD], K: [Skv, qD], V: [Skv, vD], O: [Sq, vD]
//
// 【稀疏模式】
//   CSR 风格的块级稀疏：K/V 按 kTk 行切成 Kb 块，Q 按 kTm 行切成 Qb 块。对第 i 个
//   Q 块仅遍历其活跃 K/V 块（由 kv_idx_ptr / kv_off_ptr 指定）：
//     for p in [kv_off_ptr[i], kv_off_ptr[i+1]):  j = kv_idx_ptr[p]   (0<=j<Kb)
//   这是 BigBird/Longformer 等块稀疏注意力的标准表示。
//
// 【实现：两遍式 (two-pass)】
//   稠密 flash attention 的 online softmax 把“缩放旧输出”与“累加新 PV”放在同一遍，
//   会使 score/rescale/PV/输出 同时存活，tile 寄存器压力极大。linx 后端的 tile
//   寄存器无法从栈槽 reload-fold（LinxV5InstrInfo.cpp 的 loadRegFromStackSlot 对
//   tile 类寄存器命中 UNREACHABLE），一旦发生 tile 溢出即编译器崩溃。为此这里采用
//   两遍式（参考 test 侧 fa_softmax_pto 的结构）：
//     遍1 (reduce): 对每个 Q 块逐个活跃 K 块做 online row-max / row-sum 归约，
//                   得到该 Q 块的最终 row max(m) 与 row sum(l)。不触碰 V / O。
//     遍2 (attend): 用最终 m,l 把每个活跃块的 p=exp((QK)/sqrt(d)-m)/l 直接归一化，
//                   再 P·V 并累加进 O。遍2 不需要“缩放旧输出”，tile 压力显著降低。
//   代价是对 QK 乘积算两遍（遍1 与遍2 各一次 Q·K^T），换取可编译性/低寄存器压力。
//
// 【Tile 布局约束】
//   Q/P 使用 Local CUBE_M16/M32，K^T/V 使用 CUBE_N8，矩阵结果使用对应的
//   CUBE accumulator。softmax 状态为 [kTm,1]，因此归约和广播都沿 query 行使用
//   TROWMAX/TROWSUM/TROWEXPAND*。CUBE 与普通 Vec tile 之间通过显式 GM scratch
//   边界转换，避免把 CUBE cell layout 当作 RowMajor/ColMajor 直接解释。
//
// 【tile 尺寸约束】
//   每个 tile 活跃尺寸须落在 128B..8KB（DavinciOO active PE-local profile，imm4=3..9）。
//   典型配置：dtype=__half, qD=vD=128, kTm=16, kTk=32, Skv 为 kTk 的倍数。
// =============================================================================

#include <common/pto_tileop.hpp>

using namespace pto;

template <typename E_, int R_, int C_, int VR_=R_, int VC_=C_>
using CubeTileA = std::conditional_t<
    (R_ <= 16), CubeTileM16<E_, R_, C_, VR_, VC_>,
    CubeTileM32<E_, R_, C_, VR_, VC_>>;

template <typename E_, int R_, int C_, int VR_=R_, int VC_=C_>
using CubeTileAcc = std::conditional_t<
    (R_ <= 16), CubeAccumulatorM16<E_, R_, C_, VR_, VC_>,
    CubeAccumulatorM32<E_, R_, C_, VR_, VC_>>;

template <typename dtype, int Sq, int Skv, int qD, int vD, int kTm, int kTk,
          int scaleD = qD>
void sparse_flash_attention_pto(dtype* out_ptr, dtype* q_ptr, dtype* k_ptr,
                                dtype* v_ptr, const int* kv_idx_ptr,
                                const int* kv_off_ptr) {
    // 全局张量形状与内存布局（与测试侧线性 buffer 一致）。
    using gmQ = global_tensor<dtype, RowMajor<Sq, qD>>;
    // K is physically [Skv,qD] row-major. Expose the transposed [qD,Skv]
    // view so the right CUBE operand has the canonical [K,N] descriptor.
    using gmK = global_tensor<dtype, MatrixLayout<qD, Skv, 1, qD>>;
    using gmV = global_tensor<dtype, RowMajor<Skv, vD>>;
    using gmO = global_tensor<dtype, RowMajor<Sq, vD>>;

    // tile 寄存器形状（与 fa_2d_unroll_pto / fa_hif4_pto 一致）。
    static_assert(kTm <= 32, "Local CUBE_M16/M32 SFA supports kTm <= 32");
    using tileQ      = CubeTileA<dtype, kTm, (qD == 192 ? 256 : qD), kTm, qD>;
    using tileK      = CubeTileN8<dtype, (qD == 192 ? 256 : qD), kTk, qD, kTk>;
    using tileW_out  = CubeTileAcc<float, kTm, kTk>;
    using tileW      = Tile<Location::Vec, float, kTm, kTk, BLayout::ColMajor>;
    using tileW_cast = Tile<Location::Vec, dtype, kTm, kTk, BLayout::ColMajor>;
    using tileW_left = CubeTileA<dtype, kTm, kTk>;

    using tileO_out  = CubeTileAcc<float, kTm, vD>;
    using tileO      = Tile<Location::Vec, float, kTm, vD, BLayout::ColMajor>;
    using tileO_cast = Tile<Location::Vec, dtype, kTm, vD, BLayout::ColMajor>;

    using tileV      = CubeTileN8<dtype, kTk, vD>;
    using tileMax    = Tile<Location::Vec, float, kTm, 1, BLayout::ColMajor, kTm, 1>;
    using tileSum    = Tile<Location::Vec, float, kTm, 1, BLayout::ColMajor, kTm, 1>;

    using itQ = global_iterator<gmQ, tileQ>;
    using itK = global_iterator<gmK, tileK>;
    using itV = global_iterator<gmV, tileV>;
    using itO = global_iterator<gmO, tileO>;

    itQ gIterQ(q_ptr);
    itK gIterK(k_ptr);
    itV gIterV(v_ptr);
    itO gIterO(out_ptr);

    // Explicit GM conversion boundaries bridge persistent CUBE CELL layouts
    // and the vector tiles used by softmax.
    float score_cube_scratch[kTm * kTk];
    dtype prob_cube_scratch[kTm * kTk];
    float pv_cube_scratch[kTm * vD];
    global_tensor<float, RowMajor<kTm, kTk>> gScoreCubeScratch(score_cube_scratch);
    global_tensor<dtype, RowMajor<kTm, kTk>> gProbCubeScratch(prob_cube_scratch);
    global_tensor<float, RowMajor<kTm, vD>> gPVCubeScratch(pv_cube_scratch);

    const float scale = 1.0f / sqrt((float)scaleD);
    const int Qb = (Sq + kTm - 1) / kTm;
    const int Kb = (Skv + kTk - 1) / kTk;
    (void)Kb;

    for (int i = 0; i < Qb; ++i) {
        const int kv_begin = kv_off_ptr[i];
        const int kv_end   = kv_off_ptr[i + 1];

        // ============================================================
        //  遍1 (reduce): online 归约 row max (m) 与 row sum (l)
        // ============================================================
        tileQ tQ;
        auto gQ = gIterQ(i, 0);
        TLOAD_CUBE(tQ, gQ);

        tileMax tMax;  TEXPANDS(tMax, -1e30f);   // m = -inf 近似
        tileSum tSum;  TEXPANDS(tSum, 0.0f);    // l = 0

        for (int p = kv_begin; p < kv_end; ++p) {
            const int j = kv_idx_ptr[p];

            tileK tK;
            auto gK = gIterK(0, j);
            TLOAD_CUBE(tK, gK);

            tileW tW;
            tileW_out tWCube;
            TMATMUL(tWCube, tQ, tK);
            TSTORE_CUBE(gScoreCubeScratch, tWCube);
            TLOAD(tW, gScoreCubeScratch);
            TMULS(tW, tW, scale);

            // m_new = max(m_old, rowmax(score))
            tileMax tLocalMax;
            TROWMAX(tLocalMax, tW);
            tileMax tNewMax;
            TMAX(tNewMax, tMax, tLocalMax);

            // rescale = exp(m_old - m_new); l_old' = l_old * rescale
            tileMax tScale;
            TSUB(tScale, tMax, tNewMax);
            TEXP(tScale, tScale);
            tileSum tScaledOldSum;
            TMUL(tScaledOldSum, tSum, tScale);

            // local_sum = rowsum(exp(score - m_new))
            TROWEXPANDSUB(tW, tW, tNewMax);
            TEXP(tW, tW);
            tileSum tLocalSum;
            TROWSUM(tLocalSum, tW);

            // l_new = l_old' + local_sum
            tileSum tNewSum;
            TADD(tNewSum, tScaledOldSum, tLocalSum);

            tMax = tNewMax;
            tSum = tNewSum;
        }

        // ============================================================
        //  遍2 (attend): p = exp((QK)/sqrt(d) - m) / l 已归一化,  O = Σ p·V
        //    因 p 已用最终 m,l 归一化, 每块只需 fresh TMATMUL(p,V) 再 TADD 进 O,
        //    无需 TMATMUL_ACC（避开工具链后端在累加分支上的 tile 溢出崩溃）。
        // ============================================================
        tileQ tQ2;
        auto gQ2 = gIterQ(i, 0);
        TLOAD_CUBE(tQ2, gQ2);            // 重新载入 Q（缩短 tQ 活跃区间）

        tileSum tInvSum;
        TRECIP(tInvSum, tSum);                    // 1 / l_final

        tileO tO;
        TEXPANDS(tO, 0.0f);

        for (int p = kv_begin; p < kv_end; ++p) {
            const int j = kv_idx_ptr[p];

            tileK tK;
            auto gK = gIterK(0, j);
            TLOAD_CUBE(tK, gK);

            tileW tW;
            tileW_out tWCube;
            TMATMUL(tWCube, tQ2, tK);
            TSTORE_CUBE(gScoreCubeScratch, tWCube);
            TLOAD(tW, gScoreCubeScratch);
            TMULS(tW, tW, scale);

            // p = exp(score - m) / l   （column-broadcast 减 + exp + 归一化乘）
            TROWEXPANDSUB(tW, tW, tMax);
            TEXP(tW, tW);
            TROWEXPANDMUL(tW, tW, tInvSum);

            // cast p -> dtype Left（TMATMUL 左操作数）
            tileW_cast tExpW;
            TCVT(tExpW, tW);
            tileW_left tW_left;
            TSTORE(gProbCubeScratch, tExpW);
            TLOAD_CUBE(tW_left, gProbCubeScratch);

            tileV tV;
            auto gV = gIterV(j, 0);
            TLOAD_CUBE(tV, gV);

            // PV = p * V （fresh TMATMUL；p 已归一化，O = Σ PV）
            tileO_out tPVCube;
            TMATMUL(tPVCube, tW_left, tV);
            TSTORE_CUBE(gPVCubeScratch, tPVCube);
            tileO tPV;
            TLOAD(tPV, gPVCubeScratch);

            TADD(tO, tO, tPV);
        }

        // ---- 写回 O 块 ----
        tileO_cast tO_cast;
        TCVT(tO_cast, tO);
        auto gO = gIterO(i, 0);
        TSTORE(gO, tO_cast);
    }
}

#endif
