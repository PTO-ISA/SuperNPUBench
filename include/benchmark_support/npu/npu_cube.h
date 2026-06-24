#include <common/pto_tileop.hpp>

template <const int kM, const int kN, const int kK, 
          const int kTM, const int kTN, const int kTK>
void matmul_kernel_a16w8(__half *c_ptr, __half *a_ptr, __fp8_e4m3 *b_ptr, float *dequant) {
    using gm_shapeA = global_tensor<__half, RowMajor<kM, kK>>;
    using gm_shapeB = global_tensor<__fp8_e4m3, ColMajor<kK, kN>>;
    using gm_shapeC = global_tensor<__half, RowMajor<kM, kN>>;

    using tile_shapeA = TileLeft<__half, kTM, kTK>;

    using tile_shapeB = TileRight<__fp8_e4m3, kTK, kTN>;
    using tile_shapeB_cvt = TileRight<__half, kTK, kTN>;

    using tile_shapeACC = TileAcc<float, kTM, kTN>;
    using tile_shapeACC_cvt = TileLeft<__half, kTM, kTN>;

    using gm_iteratorA = global_iterator<gm_shapeA, tile_shapeA>;
    using gm_iteratorB = global_iterator<gm_shapeB, tile_shapeB>;
    using gm_iteratorC = global_iterator<gm_shapeC, tile_shapeACC>;

    gm_iteratorA gAIter(a_ptr);
    gm_iteratorB gBIter(b_ptr);
    gm_iteratorC gCIter(c_ptr);

    const int Mb = kM / kTM;
    const int Nb = kN / kTN;
    const int Kb = kK / kTK;

    const int rmd_M = kM % kTM;
    const int rmd_N = kN % kTN;
    const int rmd_K = kK % kTK;

    using tile_shapeA_trows = TileLeft<__half, kTM, kTK,  kTM, rmd_K>;
    using tile_shapeA_tcols = TileLeft<__half, kTM, kTK, rmd_M, kTK>;
    using tile_shapeA_tcorner = TileLeft<__half, kTM, kTK, rmd_M, rmd_K>;

    using tile_shapeB_trows = TileRight<__fp8_e4m3, kTK, kTN, kTK, rmd_N>;
    using tile_shapeB_tcols = TileRight<__fp8_e4m3, kTK, kTN, rmd_K, kTN>;
    using tile_shapeB_tcorner = TileRight<__fp8_e4m3, kTK, kTN, rmd_K, rmd_N>;

    using tile_shapeB_trows_cvt = TileRight<__half, kTK, kTN, kTK, rmd_N>;
    using tile_shapeB_tcols_cvt = TileRight<__half, kTK, kTN, rmd_K, kTN>;
    using tile_shapeB_tcorner_cvt = TileRight<__half, kTK, kTN, rmd_K, rmd_N>;

    using tile_shapeC_trows = TileAcc<float, kTM, kTN, kTM, rmd_N>;
    using tile_shapeC_tcols = TileAcc<float, kTM, kTN, rmd_M, kTN>;
    using tile_shapeC_tcorner = TileAcc<float, kTM, kTN, rmd_M, rmd_N>;

    using tile_shapeC_trows_cvt = TileLeft<__half, kTM, kTN, kTM, rmd_N>;
    using tile_shapeC_tcols_cvt = TileLeft<__half, kTM, kTN, rmd_M, kTN>;
    using tile_shapeC_tcorner_cvt = TileLeft<__half, kTM, kTN, rmd_M, rmd_N>;

    using tile_shapeC_trows_scale = TileAcc<float, kTM, kTN, kTM, 1>;
    using tile_shapeC_tcols_scale = TileAcc<float, kTM, kTN, rmd_M, 1>;
    using tile_shapeC_tcorner_scale = TileAcc<float, kTM, kTN, rmd_M, 1>;

    for (int i = 0; i < Mb; ++i) {
        for (int j = 0; j < Nb; ++j) {
        auto gC = gCIter(i, j);

        tile_shapeACC tACC;
        tile_shapeA tA(0);
        tile_shapeB_cvt tB(0);
        MATMUL(tACC, tA, tB);
        #pragma clang loop unroll(full)
        for (int k = 0; k < Kb; ++k) {
            auto gA = gAIter(i, k);
            auto gB = gBIter(k, j);
            tile_shapeA tA;
            tile_shapeB tB_ori;
            tile_shapeB_cvt tB;
            TCOPYIN(tA, gA);
            TCOPYIN(tB_ori, gB);
            TCVT(tB, tB_ori);
            MATMACC(tACC, tA, tB);
        }

        if constexpr (rmd_K) {
            auto gA = gAIter(i, Kb);
            auto gB = gBIter(Kb, j);

            tile_shapeA_trows tA;
            tile_shapeB_tcols tB_ori;
            tile_shapeB_tcols_cvt tB;
            TCOPYIN(tA, gA);
            TCOPYIN(tB_ori, gB);
            TCVT(tB, tB_ori);
            MATMACC(tACC, tA, tB);
        }

        //quant pre VQF322BF16_PRE
        tile_shapeACC_cvt tACC_cvt;
        TCVT(tACC_cvt, tACC);
        TCOPYOUT(gC, tACC_cvt);
        }
        if constexpr (rmd_N) {
        auto gC = gCIter(i, Nb);

        tile_shapeC_trows tACC;
        tile_shapeA tA(0);
        tile_shapeB_trows_cvt tB(0);
        MATMUL(tACC, tA, tB);
        #pragma clang loop unroll(full)
        for (int k = 0; k < Kb; ++k) {
            auto gA = gAIter(i, k);
            auto gB = gBIter(k, Nb);

            tile_shapeA tA;
            tile_shapeB_trows tB_ori;
            tile_shapeB_trows_cvt tB;
            TCOPYIN(tA, gA);
            TCOPYIN(tB_ori, gB);
            TCVT(tB, tB_ori);
            MATMACC(tACC, tA, tB);
        }
        if constexpr (rmd_K) {
            auto gA = gAIter(i, Kb);
            auto gB = gBIter(Kb, Nb);

            tile_shapeA_trows tA;
            tile_shapeB_tcorner tB_ori;
            tile_shapeB_tcorner_cvt tB;
            TCOPYIN(tA, gA);
            TCOPYIN(tB_ori, gB);
            TCVT(tB, tB_ori);
            MATMACC(tACC, tA, tB);
        }

        //quant pre
        tile_shapeC_trows_cvt tACC_cvt;
        TCVT(tACC_cvt, tACC);
        TCOPYOUT(gC, tACC_cvt);
        }
    }
    if constexpr (rmd_M) {
        for (int j = 0; j < Nb; ++j) {
        auto gC = gCIter(Mb, j);

        tile_shapeC_tcols tACC;
        tile_shapeA_tcols tA(0);
        tile_shapeB_cvt tB(0);
        MATMUL(tACC, tA, tB);
        #pragma clang loop unroll(full)
        for (int k = 0; k < Kb; ++k) {
            auto gA = gAIter(Mb, k);
            auto gB = gBIter(k, j);

            tile_shapeA_tcols tA;
            tile_shapeB tB_ori;
            tile_shapeB_cvt tB;
            TCOPYIN(tA, gA);
            TCOPYIN(tB_ori, gB);
            TCVT(tB, tB_ori);
            MATMACC(tACC, tA, tB);
        }
        if constexpr (rmd_K) {
            auto gA = gAIter(Mb, Kb);
            auto gB = gBIter(Kb, j);

            tile_shapeA_tcorner tA;
            tile_shapeB_tcols tB_ori;
            tile_shapeB_tcols_cvt tB;
            TCOPYIN(tA, gA);
            TCOPYIN(tB_ori, gB);
            TCVT(tB, tB_ori);
            MATMACC(tACC, tA, tB);
        }

        tile_shapeC_tcols_cvt tACC_cvt;
        TCVT(tACC_cvt, tACC);
        TCOPYOUT(gC, tACC_cvt);
        }
        if constexpr (rmd_N) {
        auto gC = gCIter(Mb, Nb);

        tile_shapeC_tcorner tACC;
        tile_shapeA_tcols tA(0);
        tile_shapeB_trows_cvt tB(0);
        MATMUL(tACC, tA, tB);  
        #pragma clang loop unroll(full)
        for (int k = 0; k < Kb; ++k) {
            auto gA = gAIter(Mb, k);
            auto gB = gBIter(k, Nb);

            tile_shapeA_tcols tA;
            tile_shapeB_trows tB_ori;
            tile_shapeB_trows_cvt tB;
            TCOPYIN(tA, gA);
            TCOPYIN(tB_ori, gB);
            TCVT(tB, tB_ori);
            MATMACC(tACC, tA, tB);
        }
        if constexpr (rmd_K) {
            auto gA = gAIter(Mb, Kb);
            auto gB = gBIter(Kb, Nb);

            tile_shapeA_tcorner tA;
            tile_shapeB_tcorner tB_ori;
            tile_shapeB_tcorner_cvt tB;
            TCOPYIN(tA, gA);
            TCOPYIN(tB_ori, gB);
            TCVT(tB, tB_ori);
            MATMACC(tACC, tA, tB);
        }

        tile_shapeC_tcorner_cvt tACC_cvt;
        TCVT(tACC_cvt, tACC);
        TCOPYOUT(gC, tACC_cvt);
        }
    }
}

template <const int kM, const int kN, const int kK, 
          const int kTM, const int kTN, const int kTK>
void matmul_kernel_a8w8(__fp8_e4m3 *c_ptr, __fp8_e4m3 *a_ptr, __fp8_e5m2 *b_ptr, float *dequant) {
    using gm_shapeA = global_tensor<__fp8_e4m3, RowMajor<kM, kK>>;
    using gm_shapeB = global_tensor<__fp8_e5m2, ColMajor<kK, kN>>;
    using gm_shapeC = global_tensor<__fp8_e4m3, RowMajor<kM, kN>>;

    using tile_shapeA = TileLeft<__fp8_e4m3, kTM, kTK>;

    using tile_shapeB = TileRight<__fp8_e5m2, kTK, kTN>;
    using tile_shapeB_cvt = TileRight<__fp8_e4m3, kTK, kTN>;

    using tile_shapeACC = TileAcc<float, kTM, kTN>;
    using tile_shapeACC_cvt = TileLeft<__fp8_e4m3, kTM, kTN>;

    using gm_iteratorA = global_iterator<gm_shapeA, tile_shapeA>;
    using gm_iteratorB = global_iterator<gm_shapeB, tile_shapeB>;
    using gm_iteratorC = global_iterator<gm_shapeC, tile_shapeACC>;

    gm_iteratorA gAIter(a_ptr);
    gm_iteratorB gBIter(b_ptr);
    gm_iteratorC gCIter(c_ptr);

    const int Mb = kM / kTM;
    const int Nb = kN / kTN;
    const int Kb = kK / kTK;

    const int rmd_M = kM % kTM;
    const int rmd_N = kN % kTN;
    const int rmd_K = kK % kTK;

    using tile_shapeA_trows = TileLeft<__fp8_e4m3, kTM, kTK,  kTM, rmd_K>;
    using tile_shapeA_tcols = TileLeft<__fp8_e4m3, kTM, kTK, rmd_M, kTK>;
    using tile_shapeA_tcorner = TileLeft<__fp8_e4m3, kTM, kTK, rmd_M, rmd_K>;

    using tile_shapeB_trows = TileRight<__fp8_e5m2, kTK, kTN, kTK, rmd_N>;
    using tile_shapeB_tcols = TileRight<__fp8_e5m2, kTK, kTN, rmd_K, kTN>;
    using tile_shapeB_tcorner = TileRight<__fp8_e5m2, kTK, kTN, rmd_K, rmd_N>;

    using tile_shapeB_trows_cvt = TileRight<__fp8_e4m3, kTK, kTN, kTK, rmd_N>;
    using tile_shapeB_tcols_cvt = TileRight<__fp8_e4m3, kTK, kTN, rmd_K, kTN>;
    using tile_shapeB_tcorner_cvt = TileRight<__fp8_e4m3, kTK, kTN, rmd_K, rmd_N>;

    using tile_shapeC_trows = TileAcc<float, kTM, kTN, kTM, rmd_N>;
    using tile_shapeC_tcols = TileAcc<float, kTM, kTN, rmd_M, kTN>;
    using tile_shapeC_tcorner = TileAcc<float, kTM, kTN, rmd_M, rmd_N>;

    using tile_shapeC_trows_cvt = TileLeft<__fp8_e4m3, kTM, kTN, kTM, rmd_N>;
    using tile_shapeC_tcols_cvt = TileLeft<__fp8_e4m3, kTM, kTN, rmd_M, kTN>;
    using tile_shapeC_tcorner_cvt = TileLeft<__fp8_e4m3, kTM, kTN, rmd_M, rmd_N>;

    for (int i = 0; i < Mb; ++i) {
        for (int j = 0; j < Nb; ++j) {
        auto gC = gCIter(i, j);

        tile_shapeACC tACC;
        TileLeft<__half, kTM, kTK> tA(0);
        TileRight<__half, kTK, kTN> tB(0);
        MATMUL(tACC, tA, tB);
        #pragma clang loop unroll(full)
        for (int k = 0; k < Kb; ++k) {
            auto gA = gAIter(i, k);
            auto gB = gBIter(k, j);
            tile_shapeA tA;
            tile_shapeB tB_ori;
            tile_shapeB_cvt tB;
            TCOPYIN(tA, gA);
            TCOPYIN(tB_ori, gB);
            TCVT(tB, tB_ori);
            MATMACC(tACC, tA, tB);
        }

        if constexpr (rmd_K) {
            auto gA = gAIter(i, Kb);
            auto gB = gBIter(Kb, j);

            tile_shapeA_trows tA;
            tile_shapeB_tcols tB_ori;
            tile_shapeB_tcols_cvt tB;
            TCOPYIN(tA, gA);
            TCOPYIN(tB_ori, gB);
            TCVT(tB, tB_ori);
            MATMACC(tACC, tA, tB);
        }

        //quant pre VQF322BF16_PRE
        tile_shapeACC_cvt tACC_cvt;
        TCVT(tACC_cvt, tACC);
        TCOPYOUT(gC, tACC_cvt);
        }
        if constexpr (rmd_N) {
        auto gC = gCIter(i, Nb);

        tile_shapeC_trows tACC;
        TileLeft<__half, kTM, kTK> tA(0);
        TileRight<__half, kTK, kTN, kTK, rmd_N> tB(0);
        MATMUL(tACC, tA, tB);
        #pragma clang loop unroll(full)
        for (int k = 0; k < Kb; ++k) {
            auto gA = gAIter(i, k);
            auto gB = gBIter(k, Nb);

            tile_shapeA tA;
            tile_shapeB_trows tB_ori;
            tile_shapeB_trows_cvt tB;
            TCOPYIN(tA, gA);
            TCOPYIN(tB_ori, gB);
            TCVT(tB, tB_ori);
            MATMACC(tACC, tA, tB);
        }
        if constexpr (rmd_K) {
            auto gA = gAIter(i, Kb);
            auto gB = gBIter(Kb, Nb);

            tile_shapeA_trows tA;
            tile_shapeB_tcorner tB_ori;
            tile_shapeB_tcorner_cvt tB;
            TCOPYIN(tA, gA);
            TCOPYIN(tB_ori, gB);
            TCVT(tB, tB_ori);
            MATMACC(tACC, tA, tB);
        }

        //quant pre
        tile_shapeC_trows_cvt tACC_cvt;
        TCVT(tACC_cvt, tACC);
        TCOPYOUT(gC, tACC_cvt);
        }
    }
    if constexpr (rmd_M) {
        for (int j = 0; j < Nb; ++j) {
        auto gC = gCIter(Mb, j);

        tile_shapeC_tcols tACC;
        TileLeft<__half, kTM, kTK, rmd_M, kTK> tA(0);
        TileRight<__half, kTK, kTN> tB(0);
        MATMUL(tACC, tA, tB);
        #pragma clang loop unroll(full)
        for (int k = 0; k < Kb; ++k) {
            auto gA = gAIter(Mb, k);
            auto gB = gBIter(k, j);

            tile_shapeA_tcols tA;
            tile_shapeB tB_ori;
            tile_shapeB_cvt tB;
            TCOPYIN(tA, gA);
            TCOPYIN(tB_ori, gB);
            TCVT(tB, tB_ori);
            MATMACC(tACC, tA, tB);
        }
        if constexpr (rmd_K) {
            auto gA = gAIter(Mb, Kb);
            auto gB = gBIter(Kb, j);

            tile_shapeA_tcorner tA;
            tile_shapeB_tcols tB_ori;
            tile_shapeB_tcols_cvt tB;
            TCOPYIN(tA, gA);
            TCOPYIN(tB_ori, gB);
            TCVT(tB, tB_ori);
            MATMACC(tACC, tA, tB);
        }

        tile_shapeC_tcols_cvt tACC_cvt;
        TCVT(tACC_cvt, tACC);
        TCOPYOUT(gC, tACC_cvt);
        }
        if constexpr (rmd_N) {
        auto gC = gCIter(Mb, Nb);

        tile_shapeC_tcorner tACC;
        TileLeft<__half, kTM, kTK, rmd_M, kTK> tA(0);
        TileRight<__half, kTK, kTN, kTK, rmd_N> tB(0);
        MATMUL(tACC, tA, tB);  
        #pragma clang loop unroll(full)
        for (int k = 0; k < Kb; ++k) {
            auto gA = gAIter(Mb, k);
            auto gB = gBIter(k, Nb);

            tile_shapeA_tcols tA;
            tile_shapeB_trows tB_ori;
            tile_shapeB_trows_cvt tB;
            TCOPYIN(tA, gA);
            TCOPYIN(tB_ori, gB);
            TCVT(tB, tB_ori);
            MATMACC(tACC, tA, tB);
        }
        if constexpr (rmd_K) {
            auto gA = gAIter(Mb, Kb);
            auto gB = gBIter(Kb, Nb);

            tile_shapeA_tcorner tA;
            tile_shapeB_tcorner tB_ori;
            tile_shapeB_tcorner_cvt tB;
            TCOPYIN(tA, gA);
            TCOPYIN(tB_ori, gB);
            TCVT(tB, tB_ori);
            MATMACC(tACC, tA, tB);
        }

        tile_shapeC_tcorner_cvt tACC_cvt;
        TCVT(tACC_cvt, tACC);
        TCOPYOUT(gC, tACC_cvt);
        }
    }
}

template <const int kM, const int kN, const int kK, 
          const int kTM, const int kTN, const int kTK>
void matmul_kernel_mx_a8w8(__bf16 *c_ptr, __fp8_e4m3 *a_ptr, __fp8_e5m2 *b_ptr, __fp8_e4m3 *amx, __fp8_e4m3 *bmx, float *dequant) {
    using gm_shapeA = global_tensor<__fp8_e4m3, RowMajor<kM, kK>>;
    using gm_shapeB = global_tensor<__fp8_e5m2, ColMajor<kK, kN>>;
    using gm_shapeC = global_tensor<__bf16, RowMajor<kM, kN>>;
    using gm_shapeDQ = global_tensor<float, RowMajor<1, kN>>;

    using tile_shapeA = TileLeft<__fp8_e4m3, kTM, kTK>;

    using tile_shapeB = TileRight<__fp8_e5m2, kTK, kTN>;
    using tile_shapeB_cvt = TileRight<__fp8_e4m3, kTK, kTN>;

    using tile_shapeACC = TileAcc<float, kTM, kTN>;
    using tile_shapeACC_cvt = TileAcc<__bf16, kTM, kTN>;
    using tile_shapeACC_scale = TileAcc<float, kTM, 4, kTM, 1>;

    using tile_shapeDQ = TileAcc<float, kTM, kTN, 1, kTN>;

    using gm_iteratorA = global_iterator<gm_shapeA, tile_shapeA>;
    using gm_iteratorB = global_iterator<gm_shapeB, tile_shapeB>;
    using gm_iteratorC = global_iterator<gm_shapeC, tile_shapeACC>;

    using gm_iteratorDQ = global_iterator<gm_shapeDQ, tile_shapeDQ>;

    gm_iteratorA gAIter(a_ptr);
    gm_iteratorB gBIter(b_ptr);
    gm_iteratorC gCIter(c_ptr);
    gm_iteratorDQ gDQIter(dequant);

    const int Mb = kM / kTM;
    const int Nb = kN / kTN;
    const int Kb = kK / kTK;

    const int rmd_M = kM % kTM;
    const int rmd_N = kN % kTN;
    const int rmd_K = kK % kTK;

    using tile_shapeA_trows = TileLeft<__fp8_e4m3, kTM, kTK,  kTM, rmd_K>;
    using tile_shapeA_tcols = TileLeft<__fp8_e4m3, kTM, kTK, rmd_M, kTK>;
    using tile_shapeA_tcorner = TileLeft<__fp8_e4m3, kTM, kTK, rmd_M, rmd_K>;

    using tile_shapeB_trows = TileRight<__fp8_e5m2, kTK, kTN, kTK, rmd_N>;
    using tile_shapeB_tcols = TileRight<__fp8_e5m2, kTK, kTN, rmd_K, kTN>;
    using tile_shapeB_tcorner = TileRight<__fp8_e5m2, kTK, kTN, rmd_K, rmd_N>;

    using tile_shapeB_trows_cvt = TileRight<__fp8_e4m3, kTK, kTN, kTK, rmd_N>;
    using tile_shapeB_tcols_cvt = TileRight<__fp8_e4m3, kTK, kTN, rmd_K, kTN>;
    using tile_shapeB_tcorner_cvt = TileRight<__fp8_e4m3, kTK, kTN, rmd_K, rmd_N>;

    using tile_shapeC_trows = TileAcc<float, kTM, kTN, kTM, rmd_N>;
    using tile_shapeC_tcols = TileAcc<float, kTM, kTN, rmd_M, kTN>;
    using tile_shapeC_tcorner = TileAcc<float, kTM, kTN, rmd_M, rmd_N>;

    using tile_shapeDQ_trows = TileAcc<float, kTM, kTN, 1, rmd_N>;
    using tile_shapeDQ_tcols = TileAcc<float, kTM, kTN, 1, kTN>;
    using tile_shapeDQ_tcorner = TileAcc<float, kTM, kTN, 1, rmd_N>;

    using tile_shapeC_trows_cvt = TileAcc<__bf16, kTM, kTN, kTM, rmd_N>;
    using tile_shapeC_tcols_cvt = TileAcc<__bf16, kTM, kTN, rmd_M, kTN>;
    using tile_shapeC_tcorner_cvt = TileAcc<__bf16, kTM, kTN, rmd_M, rmd_N>;

    using tile_shapeC_trows_scale = TileAcc<float, kTM, kTN, kTM, 1>;
    using tile_shapeC_tcols_scale = TileAcc<float, kTM, kTN, rmd_M, 1>;
    using tile_shapeC_tcorner_scale = TileAcc<float, kTM, kTN, rmd_M, 1>;

    for (int i = 0; i < Mb; ++i) {
        for (int j = 0; j < Nb; ++j) {
        auto gC = gCIter(i, j);

        tile_shapeACC tACC;
        tile_shapeA tA(0);
        tile_shapeB_cvt tB(0);
        MATMUL(tACC, tA, tB);
        #pragma clang loop unroll(full)
        for (int k = 0; k < Kb; ++k) {
            auto gA = gAIter(i, k);
            auto gB = gBIter(k, j);
            tile_shapeA tA;
            tile_shapeB tB_ori;
            tile_shapeB_cvt tB;
            TCOPYIN(tA, gA);
            TCOPYIN(tB_ori, gB);
            TCAST(tB, tB_ori);
            MATMACC(tACC, tA, tB);
        }

        if constexpr (rmd_K) {
            auto gA = gAIter(i, Kb);
            auto gB = gBIter(Kb, j);

            tile_shapeA_trows tA;
            tile_shapeB_tcols tB_ori;
            tile_shapeB_tcols_cvt tB;
            TCOPYIN(tA, gA);
            TCOPYIN(tB_ori, gB);
            TCAST(tB, tB_ori);
            MATMACC(tACC, tA, tB);
        }

        //quant pre(acc * dequant_scale) 
        auto gDQ = gDQIter(0,j);
        tile_shapeDQ tDQ;
        TCOPYIN(tDQ, gDQ);
        tile_shapeACC tDQ_expand;
        TEXPANDROW(tDQ_expand,tDQ);
        TMULS(tACC, tACC, tDQ_expand);
        //quant after(scale out)
        // tile_shapeACC_scale tACC_scale;
        // TROWMAX(tACC_scale, tACC);
        // TMULS(tACC_scale, tACC_scale, static_cast<float>(2));
        tile_shapeACC_cvt tACC_cvt;
        TCAST(tACC_cvt, tACC);
        TCOPYOUT(gC, tACC_cvt);
        }
        if constexpr (rmd_N) {
        auto gC = gCIter(i, Nb);

        tile_shapeC_trows tACC;
        tile_shapeA tA(0);
        tile_shapeB_trows_cvt tB(0);
        MATMUL(tACC, tA, tB);
        #pragma clang loop unroll(full)
        for (int k = 0; k < Kb; ++k) {
            auto gA = gAIter(i, k);
            auto gB = gBIter(k, Nb);

            tile_shapeA tA;
            tile_shapeB_trows tB_ori;
            tile_shapeB_trows_cvt tB;
            TCOPYIN(tA, gA);
            TCOPYIN(tB, gB);
            TCAST(tB, tB_ori);
            MATMACC(tACC, tA, tB);
        }
        if constexpr (rmd_K) {
            auto gA = gAIter(i, Kb);
            auto gB = gBIter(Kb, Nb);

            tile_shapeA_trows tA;
            tile_shapeB_tcorner tB_ori;
            tile_shapeB_tcorner_cvt tB;
            TCOPYIN(tA, gA);
            TCOPYIN(tB, gB);
            TCAST(tB, tB_ori);
            MATMACC(tACC, tA, tB);
        }

        //quant pre(acc * dequant_scale) 
        auto gDQ = gDQIter(0,Nb);
        tile_shapeDQ tDQ;
        TCOPYIN(tDQ, gDQ);
        tile_shapeC_trows tDQ_expand;
        TEXPANDROW(tDQ_expand,tDQ);
        TMULS(tACC, tACC, tDQ_expand);
        tile_shapeC_trows_cvt tACC_cvt;
        TCAST(tACC_cvt, tACC);
        TCOPYOUT(gC, tACC_cvt);
        }
    }
    if constexpr (rmd_M) {
        for (int j = 0; j < Nb; ++j) {
        auto gC = gCIter(Mb, j);

        tile_shapeC_tcols tACC;
        tile_shapeA_tcols tA(0);
        tile_shapeB_cvt tB(0);
        MATMUL(tACC, tA, tB);
        #pragma clang loop unroll(full)
        for (int k = 0; k < Kb; ++k) {
            auto gA = gAIter(Mb, k);
            auto gB = gBIter(k, j);

            tile_shapeA_tcols tA;
            tile_shapeB tB_ori;
            tile_shapeB_cvt tB;
            TCOPYIN(tA, gA);
            TCOPYIN(tB, gB);
            TCAST(tB, tB_ori);
            MATMACC(tACC, tA, tB);
        }
        if constexpr (rmd_K) {
            auto gA = gAIter(Mb, Kb);
            auto gB = gBIter(Kb, j);

            tile_shapeA_tcorner tA;
            tile_shapeB_tcols tB_ori;
            tile_shapeB_tcols_cvt tB;
            TCOPYIN(tA, gA);
            TCOPYIN(tB, gB);
            TCAST(tB, tB_ori);
            MATMACC(tACC, tA, tB);
        }

        //need quantization for C
        auto gDQ = gDQIter(0,j);
        tile_shapeDQ tDQ;
        TCOPYIN(tDQ, gDQ);
        tile_shapeC_tcols tDQ_expand;
        TEXPANDROW(tDQ_expand,tDQ);
        TMULS(tACC, tACC, tDQ_expand);
        tile_shapeC_tcols_cvt tACC_cvt;
        TCAST(tACC_cvt, tACC);
        TCOPYOUT(gC, tACC_cvt);
        }
        if constexpr (rmd_N) {
        auto gC = gCIter(Mb, Nb);

        tile_shapeC_tcorner tACC;
        tile_shapeA_tcols tA(0);
        tile_shapeB_trows_cvt tB(0);
        MATMUL(tACC, tA, tB);  
        #pragma clang loop unroll(full)
        for (int k = 0; k < Kb; ++k) {
            auto gA = gAIter(Mb, k);
            auto gB = gBIter(k, Nb);

            tile_shapeA_tcols tA;
            tile_shapeB_trows tB_ori;
            tile_shapeB_trows_cvt tB;
            TCOPYIN(tA, gA);
            TCOPYIN(tB, gB);
            TCAST(tB, tB_ori);
            MATMACC(tACC, tA, tB);
        }
        if constexpr (rmd_K) {
            auto gA = gAIter(Mb, Kb);
            auto gB = gBIter(Kb, Nb);

            tile_shapeA_tcorner tA;
            tile_shapeB_tcorner tB_ori;
            tile_shapeB_tcorner_cvt tB;
            TCOPYIN(tA, gA);
            TCOPYIN(tB, gB);
            TCAST(tB, tB_ori);
            MATMACC(tACC, tA, tB);
        }


        //need quantization for C
        auto gDQ = gDQIter(0,Nb);
        tile_shapeDQ tDQ;
        TCOPYIN(tDQ, gDQ);
        tile_shapeC_tcorner tDQ_expand;
        TEXPANDROW(tDQ_expand,tDQ);
        TMULS(tACC, tACC, tDQ_expand);
        tile_shapeC_tcorner_cvt tACC_cvt;
        TCAST(tACC_cvt, tACC);
        TCOPYOUT(gC, tACC_cvt);
        }
    }
}

template <const int kM, const int kN, const int kK, const int kTM,
          const int kTN, const int kTK>
void matmul_a32w32(float *c_ptr, float *a_ptr, float *b_ptr) {

  using gm_shapeA = global_tensor<float, RowMajor<kM, kK>>;
  using gm_shapeB = global_tensor<float, ColMajor<kK, kN>>;
  using gm_shapeC = global_tensor<float, RowMajor<kM, kN>>;
  using tile_shapeA = TileLeft<float, kTM, kTK>;
  using tile_shapeB = TileRight<float, kTK, kTN>;
  using tile_shapeACC = TileAcc<float, kTM, kTN>;
  using gm_iteratorA = global_iterator<gm_shapeA, tile_shapeA>;
  using gm_iteratorB = global_iterator<gm_shapeB, tile_shapeB>;
  using gm_iteratorC = global_iterator<gm_shapeC, tile_shapeACC>;

  gm_iteratorA gAIter(a_ptr);
  gm_iteratorB gBIter(b_ptr);
  gm_iteratorC gCIter(c_ptr);

  const int Mb = kM / kTM;
  const int Nb = kN / kTN;
  const int Kb = kK / kTK;

  const int rmd_M = kM % kTM;
  const int rmd_N = kN % kTN;
  const int rmd_K = kK % kTK;

  using tile_shapeA_trows = TileLeft<float, kTM, kTK,  kTM, rmd_K>;
  using tile_shapeA_tcols = TileLeft<float, kTM, kTK, rmd_M, kTK>;
  using tile_shapeA_tcorner = TileLeft<float, kTM, kTK, rmd_M, rmd_K>;

  using tile_shapeB_trows = TileRight<float, kTK, kTN, kTK, rmd_N>;
  using tile_shapeB_tcols = TileRight<float, kTK, kTN, rmd_K, kTN>;
  using tile_shapeB_tcorner = TileRight<float, kTK, kTN, rmd_K, rmd_N>;

  using tile_shapeC_trows = TileAcc<float, kTM, kTN, kTM, rmd_N>;
  using tile_shapeC_tcols = TileAcc<float, kTM, kTN, rmd_M, kTN>;
  using tile_shapeC_tcorner = TileAcc<float, kTM, kTN, rmd_M, rmd_N>;

  for (int i = 0; i < Mb; ++i) {
    for (int j = 0; j < Nb; ++j) {
      auto gC = gCIter(i, j);

      tile_shapeACC tACC;
      tile_shapeA tA(0);
      tile_shapeB tB(0);
      MATMUL(tACC, tA, tB);
      #pragma clang loop unroll(full)
      for (int k = 0; k < Kb; ++k) {
        auto gA = gAIter(i, k);
        auto gB = gBIter(k, j);

        tile_shapeA tA;
        tile_shapeB tB;
        TCOPYIN(tA, gA);
        TCOPYIN(tB, gB);
        MATMACC(tACC, tA, tB);
      }

      if constexpr (rmd_K) {
        auto gA = gAIter(i, Kb);
        auto gB = gBIter(Kb, j);

        tile_shapeA_trows tA;
        tile_shapeB_tcols tB;
        TCOPYIN(tA, gA);
        TCOPYIN(tB, gB);
        MATMACC(tACC, tA, tB);
      }

      TCOPYOUT(gC, tACC);
    }
    if constexpr (rmd_N) {
      auto gC = gCIter(i, Nb);

      tile_shapeC_trows tACC;
      tile_shapeA tA(0);
      tile_shapeB_trows tB(0);
      MATMUL(tACC, tA, tB);
      #pragma clang loop unroll(full)
      for (int k = 0; k < Kb; ++k) {
        auto gA = gAIter(i, k);
        auto gB = gBIter(k, Nb);

        tile_shapeA tA;
        tile_shapeB_trows tB;
        TCOPYIN(tA, gA);
        TCOPYIN(tB, gB);
        MATMACC(tACC, tA, tB);
      }
      if constexpr (rmd_K) {
        auto gA = gAIter(i, Kb);
        auto gB = gBIter(Kb, Nb);

        tile_shapeA_trows tA;
        tile_shapeB_tcorner tB;
        TCOPYIN(tA, gA);
        TCOPYIN(tB, gB);
        MATMACC(tACC, tA, tB);
      }
      TCOPYOUT(gC, tACC);
    }
  }
  if constexpr (rmd_M) {
    for (int j = 0; j < Nb; ++j) {
      auto gC = gCIter(Mb, j);

      tile_shapeC_tcols tACC;
      tile_shapeA_tcols tA(0);
      tile_shapeB tB(0);
      MATMUL(tACC, tA, tB);
      #pragma clang loop unroll(full)
      for (int k = 0; k < Kb; ++k) {
        auto gA = gAIter(Mb, k);
        auto gB = gBIter(k, j);

        tile_shapeA_tcols tA;
        tile_shapeB tB;
        TCOPYIN(tA, gA);
        TCOPYIN(tB, gB);
        MATMACC(tACC, tA, tB);
      }
      if constexpr (rmd_K) {
        auto gA = gAIter(Mb, Kb);
        auto gB = gBIter(Kb, j);

        tile_shapeA_tcorner tA;
        tile_shapeB_tcols tB;
        TCOPYIN(tA, gA);
        TCOPYIN(tB, gB);
        MATMACC(tACC, tA, tB);
      }
      TCOPYOUT(gC, tACC);
    }
    if constexpr (rmd_N) {
      auto gC = gCIter(Mb, Nb);

      tile_shapeC_tcorner tACC;
      tile_shapeA_tcols tA(0);
      tile_shapeB_trows tB(0);
      MATMUL(tACC, tA, tB);  
      #pragma clang loop unroll(full)
      for (int k = 0; k < Kb; ++k) {
        auto gA = gAIter(Mb, k);
        auto gB = gBIter(k, Nb);

        tile_shapeA_tcols tA;
        tile_shapeB_trows tB;
        TCOPYIN(tA, gA);
        TCOPYIN(tB, gB);
        MATMACC(tACC, tA, tB);
      }
      if constexpr (rmd_K) {
        auto gA = gAIter(Mb, Kb);
        auto gB = gBIter(Kb, Nb);

        tile_shapeA_tcorner tA;
        tile_shapeB_tcorner tB;
        TCOPYIN(tA, gA);
        TCOPYIN(tB, gB);
        MATMACC(tACC, tA, tB);
      }
      TCOPYOUT(gC, tACC);
    }
  }
}



