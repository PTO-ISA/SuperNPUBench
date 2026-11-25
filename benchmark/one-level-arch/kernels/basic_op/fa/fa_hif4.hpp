#pragma once
#include <common/pto_tileop.hpp>
#include <cmath>
#include <cstdint>

// Native HiF4 Matrix-MX attention baseline. Q/K: packed [rows,QD/2];
// V: B-major packed [VD,Skv/2]. U32 scales are [rows,logical_K/64].
// Softmax/P are BF16; PV mixes BF16 P with HiF4 V (ScaleMask=2), so it
// never performs an illegal standalone HiF4 conversion of probabilities.
// Variant 0: full-QD QK; 1: 64-wide QK accumulator chain; 2: fused RowMax.
template<int Sq, int Skv, int QD, int VD, int BlockK, int Variant>
void flash_attention_hif4(__bf16 *out, __fp4_hif4x2 *q,
                         __fp4_hif4x2 *k, __fp4_hif4x2 *v,
                         uint32_t *qs, uint32_t *ks, uint32_t *vs) {
    using namespace pto;
    static_assert(Sq % 128 == 0 && Skv % BlockK == 0 && QD % 64 == 0 && BlockK % 64 == 0);
    constexpr int QChunk = Variant == 1 ? 64 : QD;
    using Q = SharedMatrixLeft<__fp4_hif4x2,128,QChunk>;
    using K = SharedMatrixRight<__fp4_hif4x2,BlockK,QChunk>;
    using V = SharedMatrixRight<__fp4_hif4x2,VD,BlockK>;
    using QS = SharedMatrixLeft<uint32_t,128,32,128,QChunk/64>;
    using KS = SharedMatrixRight<uint32_t,BlockK,32,BlockK,QChunk/64>;
    using VS = SharedMatrixRight<uint32_t,VD,32,VD,BlockK/64>;
    static_assert(Q::LogicalTileBytes + K::LogicalTileBytes + V::LogicalTileBytes +
                  QS::LogicalTileBytes + KS::LogicalTileBytes + VS::LogicalTileBytes <= 262144,
                  "HiF4 FA operands and all U32 scale carriers exceed 256 KiB");
    using W = CubeTileM32<__bf16,32,BlockK>;
    using WAcc = CubeAccumulatorM32<float,32,BlockK>;
    using Row = VecTileM32<__bf16,32,1>;
    using RowF = VecTileM32<float,32,1>;
    using Reduce = VecTileM32<__bf16,32,BlockK,32,1>;
    using ReduceF = VecTileM32<float,32,BlockK,32,1>;
    using O = CubeAccumulatorM32<float,32,VD>;
    using Out = CubeAccumulatorM32<__bf16,32,VD>;
    const int pe = get_thread_idx();
    const __bf16 factor = static_cast<__bf16>(1.0f / sqrt(float(QD)));
    #pragma clang loop unroll(full)
    for (int qb = 0; qb < Sq / 128; ++qb) {
        Row max;
        RowF sum;
        O acc;
        TEXPANDS(max, -1e30f);
        TEXPANDS(sum, 0.0f);
        #pragma clang loop unroll(full)
        for (int kb = 0; kb < Skv / BlockK; ++kb) {
            W scores;
            Row local_max;
            WAcc chain;
            #pragma clang loop unroll(full)
            for (int kc = 0; kc < QD / QChunk; ++kc) {
                // GM shapes/strides count logical 4-bit lanes; pointers advance in bytes.
                global_tensor<__fp4_hif4x2,RowMajor<128,QD>> gq(q + qb*128*(QD/2) + kc*QChunk/2);
                global_tensor<__fp4_hif4x2,RowMajor<BlockK,QD>> gk(k + kb*BlockK*(QD/2) + kc*QChunk/2);
                global_tensor<uint32_t,RowMajor<128,QD/64>> gqs(qs + qb*128*(QD/64) + kc*QChunk/64);
                global_tensor<uint32_t,RowMajor<BlockK,QD/64>> gks(ks + kb*BlockK*(QD/64) + kc*QChunk/64);
                auto tq = TLOAD<Q,1>(gq); auto tk = TLOAD<K,1>(gk);
                auto tqs = TLOAD<QS,1>(gqs); auto tks = TLOAD<KS,1>(gks);
                if constexpr (Variant == 1 && QD > 64) {
                    if (kc == 0) TMATMUL_MX<3>(chain,tq,tqs,tk,tks,fixp::keep_acc());
                    else TMATMUL_MX_ACC<3>(chain,chain,tq,tqs,tk,tks,fixp::keep_acc());
                    if (kc == QD / QChunk - 1) TCVT(scores,chain);
                } else if constexpr (Variant == 2) {
                    TMATMUL_MX<3>(scores,tq,tqs,tk,tks,fixp::bf16().row_max(local_max));
                } else TMATMUL_MX<3>(scores,tq,tqs,tk,tks,fixp::bf16());
            }
            TMULS(scores,scores,factor);
            if constexpr (Variant == 2) TMULS(local_max,local_max,factor);
            else {
                Reduce wide_max;
                TROWMAX(wide_max,scores);
                auto view = TREDUCEPREFIXVIEW<Row>(wide_max);
                Row zero; TEXPANDS(zero,0.0f); TADD(local_max,zero,view);
            }
            Row next_max, correction;
            RowF correction_f;
            TMAX(next_max,max,local_max);
            if (kb != 0) {
                TROWEXPANDEXPDIF(correction,max,next_max);
                TCVT(correction_f,correction);
                TROWEXPANDMUL(acc,acc,correction_f);
            }
            TROWEXPANDEXPDIF(scores,scores,next_max);
            // Sum BF16 probabilities in FP32. BF16 accumulation loses
            // small probabilities for BlockK=256 and biases normalization.
            WAcc probabilities_f; TCVT(probabilities_f,scores);
            ReduceF wide_sum; TROWSUM(wide_sum,probabilities_f);
            auto sum_view = TREDUCEPREFIXVIEW<RowF>(wide_sum);
            RowF zero, local_sum_f, next_sum;
            TEXPANDS(zero,0.0f); TADD(local_sum_f,zero,sum_view);
            if (kb == 0) TADD(next_sum,sum,local_sum_f);
            else TFMA(next_sum,sum,correction_f,local_sum_f);
            global_tensor<__fp4_hif4x2,RowMajor<VD,Skv>> gv(v + kb*BlockK/2);
            global_tensor<uint32_t,RowMajor<VD,Skv/64>> gvs(vs + kb*BlockK/64);
            auto tv = TLOAD<V,1>(gv); auto tvs = TLOAD<VS,1>(gvs);
            pto_matmul_detail::NoScaleOperand no_scale;
            if (kb == 0) TMATMUL_MX<2>(acc,scores,no_scale,tv,tvs,fixp::keep_acc(),128);
            else TMATMUL_MX_ACC<2>(acc,acc,scores,no_scale,tv,tvs,fixp::keep_acc(),128);
            max = next_max; sum = next_sum;
        }
        TROWEXPANDDIV(acc,acc,sum);
        Out result; TCVT(result,acc);
        global_tensor<__bf16,RowMajor<32,VD>> gout(out + (qb*128+pe*32)*VD);
        TSTORE_CUBE(gout,result);
    }
}
