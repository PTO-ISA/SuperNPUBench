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
// 【工具链约束】
//   linx_blockisa_llvm_musl 在 -D__linx 下 Tile::TileDType 实例化为打包向量
//   `DType __attribute__((ext_vector_type(Rows*Cols)))`，而 DType 仍为标量。工具链自带
//   广播算子（TROWEXPANDSUB/MUL、TCOLEXPANDSUB/MUL）的 static_assert 错误地比较
//   `src0::DType`(标量) 与 `dst::TileDType`(打包向量)，任何多元素 tile 上均触发
//   "src0/dst dtype must match" 前端编译失败。规避：广播算子改用
//   test/common/template_asm.h 的 *_TEPL 实现（纯内联汇编，无该断言）；其余算子
//   （TLOAD/TSTORE/TMATMUL/TMATMUL_ACC/ACCCVT/TCVT/TMULS/TEXPANDS/TCOLMAX/TCOLSUM/
//    TMAX/TSUB/TEXP/TMUL/TADD/TRECIP）直接用 pto 命名空间实现，不含该错误断言。
//   softmax 采用 column-broadcast 形式（score tile ColMajor，状态 tile [kTm,1]），
//   与 TCOLEXPAND*_TEPL 语义一致（同 fa_hif4_pto 的 softmax 形式）。
//
// 【tile 尺寸约束】
//   每个 tile 活跃尺寸须落在 128B..8KB（DavinciOO active PE-local profile，imm4=3..9）。
//   典型配置：dtype=__half, qD=vD=128, kTm=16, kTk=32, Skv 为 kTk 的倍数。
// =============================================================================

#include <common/pto_tileop.hpp>

using namespace pto;

template <typename E_, int R_, int C_, int VR_=R_, int VC_=C_>
using TileAcc = Tile<Location::Vec, E_, R_, C_, BLayout::RowMajor, VR_, VC_>;

template <typename dtype, int Sq, int Skv, int qD, int vD, int kTm, int kTk,
          int scaleD = qD>
void sparse_flash_attention_pto(dtype* out_ptr, dtype* q_ptr, dtype* k_ptr,
                                dtype* v_ptr, const int* kv_idx_ptr,
                                const int* kv_off_ptr) {
    // 全局张量形状与内存布局（与测试侧线性 buffer 一致）。
    using gmQ = global_tensor<dtype, RowMajor<Sq, qD>>;
    using gmK = global_tensor<dtype, RowMajor<Skv, qD>>;
    using gmV = global_tensor<dtype, RowMajor<Skv, vD>>;
    using gmO = global_tensor<dtype, RowMajor<Sq, vD>>;

    // tile 寄存器形状（与 fa_2d_unroll_pto / fa_hif4_pto 一致）。
    using tileQ      = TileLeft<dtype, kTm, (qD == 192 ? 256 : qD), kTm, qD>;
    using tileK      = TileRight<dtype, kTk, (qD == 192 ? 256 : qD), kTk, qD>;
    using tW_type  = TileAcc<float, kTm, kTk>;
    using tileW      = Tile<Location::Vec, float, kTm, kTk, BLayout::ColMajor>;
    using tileW_cast = Tile<Location::Vec, dtype, kTm, kTk, BLayout::ColMajor>;
    using tileW_left = TileLeft<dtype, kTm, kTk>;

    using tileO_out  = TileAcc<float, kTm, vD>;
    using tileO      = Tile<Location::Vec, float, kTm, vD, BLayout::ColMajor>;
    using tileO_cast = Tile<Location::Vec, dtype, kTm, vD, BLayout::ColMajor>;

    using tileV      = TileRight<dtype, kTk, vD>;
    using tileMax    = Tile<Location::Vec, float, kTm, 8, BLayout::ColMajor, kTm, 1>;
    using tileSum    = Tile<Location::Vec, float, kTm, 8, BLayout::ColMajor, kTm, 1>;

    using itQ = global_iterator<gmQ, tileQ>;
    using itK = global_iterator<gmK, tileK>;
    using itV = global_iterator<gmV, tileV>;
    using itO = global_iterator<gmO, tileO>;

    itQ gIterQ(q_ptr);
    itK gIterK(k_ptr);
    itV gIterV(v_ptr);
    itO gIterO(out_ptr);

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
        TLOAD(tQ, gQ);

        tileMax tMax;  TEXPANDS(tMax, -1e30f);   // m = -inf 近似
        tileSum tSum;  TEXPANDS(tSum, 0.0f);    // l = 0

        for (int p = kv_begin; p < kv_end; ++p) {
            const int j = kv_idx_ptr[p];

            tileK tK;
            auto gK = gIterK(j, 0);
            TLOAD(tK, gK);

            tileW tW;
            TMATMUL(tW, tQ, tK);
            TMULS(tW, tW, scale);

            // m_new = max(m_old, colmax(score))
            tileMax tLocalMax;
            TCOLMAX(tLocalMax, tW);
            tileMax tNewMax;
            TMAX(tNewMax, tMax, tLocalMax);

            // rescale = exp(m_old - m_new); l_old' = l_old * rescale
            tileMax tScale;
            TSUB(tScale, tMax, tNewMax);
            TEXP(tScale, tScale);
            tileSum tScaledOldSum;
            TMUL(tScaledOldSum, tSum, tScale);

            // local_sum = colsum(exp(score - m_new))
            TCOLEXPANDSUB(tW, tW, tNewMax);
            TEXP(tW, tW);
            tileSum tLocalSum;
            TCOLSUM(tLocalSum, tW);

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
        TLOAD(tQ2, gQ2);                 // 重新载入 Q（缩短 tQ 活跃区间）

        tileSum tInvSum;
        TRECIP(tInvSum, tSum);                    // 1 / l_final

        tileO tO;
        TEXPANDS(tO, 0.0f);

        for (int p = kv_begin; p < kv_end; ++p) {
            const int j = kv_idx_ptr[p];

            tileK tK;
            auto gK = gIterK(j, 0);
            TLOAD(tK, gK);

            tileW tW;
            TMATMUL(tW, tQ2, tK);
            TMULS(tW, tW, scale);

            // p = exp(score - m) / l   （column-broadcast 减 + exp + 归一化乘）
            TCOLEXPANDSUB(tW, tW, tMax);
            TEXP(tW, tW);
            TCOLEXPANDMUL(tW, tW, tInvSum);

            // cast p -> dtype Left（TMATMUL 左操作数）
            tileW_cast tExpW;
            TCVT(tExpW, tW);
            tileW_left tW_left;
            TCVT(tW_left, tExpW);

            tileV tV;
            auto gV = gIterV(j, 0);
            TLOAD(tV, gV);

            // PV = p * V （fresh TMATMUL；p 已归一化，O = Σ PV）
            tileO tPV;
            TMATMUL(tPV, tW_left, tV);

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
