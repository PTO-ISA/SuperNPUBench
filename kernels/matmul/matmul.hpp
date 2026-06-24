#ifndef MATMUL_KERNEL_HPP
#define MATMUL_KERNEL_HPP

#include <common/pto_tileop.hpp>

#ifndef Batch
#define Batch 1
#endif

using namespace pto;

template <is_global_data_v GmOut, is_tile_data_v TileAcc>
void TCOPYOUT_ACC(GmOut &Gout, TileAcc &tAcc){
    using TileAccOut = Tile<Location::Vec, typename TileAcc::DType, TileAcc::Rows, TileAcc::Cols, BLayout::RowMajor, TileAcc::ValidRow, TileAcc::ValidCol>;
    TileAccOut tAccOut;
    TCVT(tAccOut, tAcc);
    TCOPYOUT(Gout, tAccOut);
}

template <is_global_data_v GmOut, is_tile_data_v TileAcc>
void TCOPYOUT_ACC_DYNAMIC(GmOut &Gout, TileAcc &tAcc, size_t valid_row, size_t valid_col){
    using TileAccOut = Tile<Location::Vec, typename TileAcc::DType, TileAcc::Rows, TileAcc::Cols, BLayout::RowMajor, -1, -1>;
    TileAccOut tAccOut(valid_row, valid_col);
    TCVT(tAccOut, tAcc);
    TCOPYOUT(Gout, tAccOut);
}

// A * B -> C with any shape
// activation * weight( int8_t->FP16/FP8-> FP32 -> int8_t)
template <typename dtype, const int gM, const int gN, const int gK, const int tM, const int tN, const int tK>
void matmul_mask(float *c_ptr, dtype *a_ptr, dtype *b_ptr) {

  using gm_shapeA = global_tensor<dtype, RowMajor<gM, gK>>;
  using gm_shapeB = global_tensor<dtype, RowMajor<gK, gN>>;
  using gm_shapeC = global_tensor<float, RowMajor<gM, gN>>;
  using tile_shapeA = TileLeft<dtype, tM, tK>;
  using tile_shapeB = TileRight<dtype, tK, tN>;
  using tile_shapeACC = TileAcc<float, tM, tN>;
  // using tile_shapecast = Tile<Location::Vec, __bf16, kTm, kTk, BLayout::ColMajor>;

  using itA = global_iterator<gm_shapeA, tile_shapeA>;
  using itB = global_iterator<gm_shapeB, tile_shapeB>;
  using itC = global_iterator<gm_shapeC, tile_shapeACC>;

  itA gAIter(a_ptr);
  itB gBIter(b_ptr);
  itC gCIter(c_ptr);

  const int Mb = gM / tM;
  const int Nb = gN / tN;
  const int Kb = gK / tK;


  const int rmd_M = gM % tM;
  const int rmd_N = gN % tN;
  const int rmd_K = gK % tK;

  using tile_shapeA_trows = TileLeft<dtype, tM, tK,  tM, rmd_K>;
  using tile_shapeA_tcols = TileLeft<dtype, tM, tK, rmd_M, tK>;
  using tile_shapeA_tcorner = TileLeft<dtype, tM, tK, rmd_M, rmd_K>;

  using tile_shapeB_trows = TileRight<dtype, tK, tN, tK, rmd_N>;
  using tile_shapeB_tcols = TileRight<dtype, tK, tN, rmd_K, tN>;
  using tile_shapeB_tcorner = TileRight<dtype, tK, tN, rmd_K, rmd_N>;

  using tile_shapeC_trows = TileAcc<float, tM, tN, tM, rmd_N>;
  using tile_shapeC_tcols = TileAcc<float, tM, tN, rmd_M, tN>;
  using tile_shapeC_tcorner = TileAcc<float, tM, tN, rmd_M, rmd_N>;

  for (int b=0;b<Batch;b++){
    for (int i = 0; i < Mb; ++i) {
      for (int j = 0; j < Nb; ++j) {
        auto gC = gCIter(i, j);

        tile_shapeACC tACC;
        // tile_shapecast tcast;
        
        if constexpr(Kb>0){
          auto gA = gAIter(i, 0);
          auto gB = gBIter(0, j);

          tile_shapeA tA;
          tile_shapeB tB;
          TCOPYIN(tA, gA);
          TCOPYIN(tB, gB);
          MATMUL(tACC, tA, tB);        
        }
        #pragma clang loop unroll(full)
        for (int k = 1; k < Kb; ++k) {
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
          if constexpr(Kb>0){
            MATMACC(tACC, tA, tB);
          } else {
            MATMUL(tACC, tA, tB);
          }
        }
        // TCVT(tCast, tACC);
        // TCOPYOUT(gC, tCast);
        TCOPYOUT_ACC(gC, tACC);
      }
      if constexpr (rmd_N) {
        auto gC = gCIter(i, Nb);

        tile_shapeC_trows tACC;
        if constexpr(Kb>0){
          auto gA = gAIter(i, 0);
          auto gB = gBIter(0, Nb);

          tile_shapeA tA;
          tile_shapeB_trows tB;
          TCOPYIN(tA, gA);
          TCOPYIN(tB, gB);
          MATMUL(tACC, tA, tB);        
        }
        #pragma clang loop unroll(full)
        for (int k = 1; k < Kb; ++k) {
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
          if constexpr(Kb>0){
            MATMACC(tACC, tA, tB);
          } else {
            MATMUL(tACC, tA, tB);
          }
        }
        TCOPYOUT_ACC(gC, tACC);
      }
    }
    if constexpr (rmd_M) {
      for (int j = 0; j < Nb; ++j) {
        auto gC = gCIter(Mb, j);

        tile_shapeC_tcols tACC;
        if constexpr(Kb>0){
          auto gA = gAIter(Mb, 0);
          auto gB = gBIter(0, j);

          tile_shapeA_tcols tA;
          tile_shapeB tB;
          TCOPYIN(tA, gA);
          TCOPYIN(tB, gB);
          MATMUL(tACC, tA, tB);        
        }
        #pragma clang loop unroll(full)
        for (int k = 1; k < Kb; ++k) {
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
          if constexpr(Kb>0){
            MATMACC(tACC, tA, tB);
          } else {
            MATMUL(tACC, tA, tB);
          }
        }
        TCOPYOUT_ACC(gC, tACC);
      }
      if constexpr (rmd_N) {
        auto gC = gCIter(Mb, Nb);

        tile_shapeC_tcorner tACC;
        if constexpr(Kb>0){
          auto gA = gAIter(Mb, 0);
          auto gB = gBIter(0, Nb);

          tile_shapeA_tcols tA;
          tile_shapeB_trows tB;
          TCOPYIN(tA, gA);
          TCOPYIN(tB, gB);
          MATMUL(tACC, tA, tB);        
        }
        #pragma clang loop unroll(full)
        for (int k = 1; k < Kb; ++k) {
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
          if constexpr(Kb>0){
            MATMACC(tACC, tA, tB);
          } else {
            MATMUL(tACC, tA, tB);
          }
        }
        TCOPYOUT_ACC(gC, tACC);
      }
    }
  }
}

template<typename dtype, const int gM, const int gN, const int gK, const int tM, const int tN, const int tK>
void matmul_frac(float* dst, dtype* src0, dtype* src1){
    using gm_shapeA = global_tensor<dtype, RowMajor<gM, gK>>;
    using gm_shapeB = global_tensor<dtype, RowMajor<gK, gN>>;
    using gm_shapeC = global_tensor<float, RowMajor<gM, gN>>;
    using tile_shapeA = TileLeft<dtype, tM, tK>;
    using tile_shapeB = TileRight<dtype, tK, tN>;
    using tile_shapeACC = TileAcc<float, tM, tN>;
    using itA = global_iterator<gm_shapeA, tile_shapeA>;
    using itB = global_iterator<gm_shapeB, tile_shapeB>;
    using itC = global_iterator<gm_shapeC, tile_shapeACC>;

    itA gAIter(src0);
    itB gBIter(src1);
    itC gCIter(dst);

    const int Mb = gM / tM;
    const int Nb = gN / tN;
    const int Kb = gK / tK;

    for(int i=0;i<Mb;i++){
        for(int j=0;j<Nb;j++){
            auto gC = gCIter(i, j);

            tile_shapeACC tACC;
            if constexpr(Kb>0){
              auto gA = gAIter(i, 0);
              auto gB = gBIter(0, j);

              tile_shapeA tA;
              tile_shapeB tB;
              TCOPYIN(tA, gA);
              TCOPYIN(tB, gB);
              MATMUL(tACC, tA, tB);
            }
            #pragma clang loop unroll(full)
            for(int k=1;k<Kb;k++){
                auto gA = gAIter(i,k);
                auto gB = gBIter(k,j);
                tile_shapeA tA;
                tile_shapeB tB;
                TCOPYIN(tA, gA);
                TCOPYIN(tB, gB);
                MATMACC(tACC, tA, tB);
            }
            TCOPYOUT_ACC(gC, tACC);
        }
    }
}

struct ResA {
    int m;
    int k;
    int val;
};

struct ResB {
    int n;
    int k;
    int val;
};

constexpr ResA find_reuseA(int Mb, int Kb, int MAX_TILE_NUM) {
    int best_m = 0, best_k = 0, best_val = -1;

    #pragma clang loop unroll(full)
    for (int m = 1; m <= Mb; ++m) {
        #pragma clang loop unroll(full)
        for (int k = 1; k <= Kb; ++k) {
            int v = m * k;
            if (v <= MAX_TILE_NUM && v > best_val) {
                best_val = v;
                best_m = m;
                best_k = k;
            }
        }
    }
    return {best_m, best_k, best_val};
}

constexpr ResB find_reuseB(int Nb, int Kb, int MAX_TILE_NUM) {
    int best_n = 0, best_k = 0, best_val = -1;

    #pragma clang loop unroll(full)
    for (int n = 1; n <= Nb; ++n) {
        #pragma clang loop unroll(full)
        for (int k = 1; k <= Kb; ++k) {
            int v = n * k;
            if (v <= MAX_TILE_NUM && v > best_val) {
                best_val = v;
                best_n = n;
                best_k = k;
            }
        }
    }
    return {best_n, best_k, best_val};
}

template<typename dtype, const int gM, const int gN, const int gK, const int tM, const int tN, const int tK>
void matmul_mask_reuseA(float *dst, dtype *src0, dtype *src1){
  // const int MAX_TILE_NUM = gK/tK;
  const int MAX_TILE_NUM = 24;
  using gm_shapeA = global_tensor<dtype, RowMajor<gM, gK>>;
  using gm_shapeB = global_tensor<dtype, RowMajor<gK, gN>>;
  using gm_shapeC = global_tensor<float, RowMajor<gM, gN>>;
  using tile_shapeA = TileLeft<dtype, tM, tK>;
  using tile_shapeB = TileRight<dtype, tK, tN>;
  using tile_shapeACC = TileAcc<float, tM, tN>;

  using itA = global_iterator<gm_shapeA, tile_shapeA>;
  using itB = global_iterator<gm_shapeB, tile_shapeB>;
  using itC = global_iterator<gm_shapeC, tile_shapeACC>;

  itA gIterA(src0);
  itB gIterB(src1);
  itC gIterC(dst);

  const int Mb = gM / tM;
  const int Nb = gN / tN;
  const int Kb = gK / tK;
  // static_assert(Kb == 56);
  const int rmd_M = gM % tM;
  const int rmd_N = gN % tN;
  const int rmd_K = gK % tK;

  using tile_shapeA_trows = TileLeft<dtype, tM, tK,  tM, rmd_K>;
  using tile_shapeA_tcols = TileLeft<dtype, tM, tK, rmd_M, tK>;
  using tile_shapeA_tcorner = TileLeft<dtype, tM, tK, rmd_M, rmd_K>;

  using tile_shapeB_trows = TileRight<dtype, tK, tN, tK, rmd_N>;
  using tile_shapeB_tcols = TileRight<dtype, tK, tN, rmd_K, tN>;
  using tile_shapeB_tcorner = TileRight<dtype, tK, tN, rmd_K, rmd_N>;

  using tile_shapeC_trows = TileAcc<float, tM, tN, tM, rmd_N>;
  using tile_shapeC_tcols = TileAcc<float, tM, tN, rmd_M, tN>;
  using tile_shapeC_tcorner = TileAcc<float, tM, tN, rmd_M, rmd_N>;

  // constexpr ResA R = find_reuseA(Mb, Kb, MAX_TILE_NUM);
  constexpr ResA R = {1, (Kb > MAX_TILE_NUM) ? MAX_TILE_NUM : Kb, R.m * R.k};
  // R.m = 1;
  // R.k = (Kb > MAX_TILE_NUM) ? MAX_TILE_NUM : Kb;
  // R.val = R.m * R.k;

  const int dM = R.m == 0? 0 : Mb / R.m;
  const int rM = R.m == 0? 0 : Mb % R.m;

  static_assert(R.val <= MAX_TILE_NUM, "R.val is biggger than MAX_TILE_NUM");
  //printf("R.m is %d, R.k is %d\n", R.m, R.k);
  static_assert(Batch == 1);
  for (int b=0;b<Batch;b++){

    #pragma clang loop unroll(full)
    for(int i=0;i<dM;i++){
      tile_shapeA tA[R.m][R.k];

      #pragma clang loop unroll(full)
      for(int ii=0;ii<R.m;ii++){

        // // copy in all reuse A sub tiles in K dim
        // #pragma clang loop unroll(full)
        // for(int k=0;k<R.k;k++){
        //   auto gA = gIterA(ii+i*R.m,k);
        //   TCOPYIN(tA[ii][k], gA);
        // }

        #pragma clang loop unroll(full)
        for(int j=0;j<Nb;j++){
          tile_shapeACC tACC;

          #pragma clang loop unroll(full)
          for(int k=0;k<R.k;k++){
            auto gB = gIterB(k,j);
            tile_shapeB tB;
            TCOPYIN(tB, gB);
            if(j==0){
              // eliminate head cost
              auto gA = gIterA(ii+i*R.m,k);
              TCOPYIN(tA[ii][k], gA);
            }
            if(k==0){
              MATMUL(tACC, tA[ii][k], tB);
            }else{
              MATMACC(tACC, tA[ii][k], tB);
            }
          }

          if constexpr(R.k < Kb){
            for(int k=R.k;k<Kb;k++){
              tile_shapeA tA;
              tile_shapeB tB;
              auto gA = gIterA(i*R.m+ii,k);
              auto gB = gIterB(k,j);
              TCOPYIN(tA,gA);
              TCOPYIN(tB,gB);
              MATMACC(tACC, tA, tB);
            }
          }

          // [m, n, rmd_K]
          if constexpr (rmd_K) {
            auto gA = gIterA(i*R.m+ii, Kb);
            auto gB = gIterB(Kb, j);

            tile_shapeA_trows tA;
            tile_shapeB_tcols tB;

            TCOPYIN(tA, gA);
            TCOPYIN(tB, gB);
            if constexpr(Kb>0){
            MATMACC(tACC, tA, tB);
            } else {
              MATMUL(tACC, tA, tB);
            }
          }

          auto gC = gIterC(i*R.m+ii,j);
          TCOPYOUT_ACC(gC, tACC);
        }

        // [m, rmd_N, k]
        if constexpr (rmd_N) {
          tile_shapeC_trows tACC;

          #pragma clang loop unroll(full)
          for(int k=0;k<R.k;k++){
            auto gB = gIterB(k,Nb);
            tile_shapeB_trows tB;
            TCOPYIN(tB, gB);
            if(k==0){
              MATMUL(tACC, tA[ii][k], tB);
            }else{
              MATMACC(tACC, tA[ii][k], tB);
            }
          }
          static_assert(R.k > Kb);
          if constexpr(R.k < Kb){
            
            for(int k=R.k;k<Kb;k++){
              tile_shapeA tA;
              tile_shapeB_trows tB;
              auto gA = gIterA(i*R.m+ii,k);
              auto gB = gIterB(k,Nb);
              TCOPYIN(tA,gA);
              TCOPYIN(tB,gB);
              MATMACC(tACC, tA, tB);
            }
          }

          // [m, rmd_N, rmd_K]
          if constexpr (rmd_K) {
            auto gA = gIterA(i*R.m+ii, Kb);
            auto gB = gIterB(Kb, Nb);

            tile_shapeA_trows tA;
            tile_shapeB_tcorner tB;
            
            TCOPYIN(tA, gA);
            TCOPYIN(tB, gB);
            if constexpr(Kb>0){
              MATMACC(tACC, tA, tB);
            } else {
              MATMUL(tACC, tA, tB);
            }
          }

          auto gC = gIterC(i*R.m+ii,Nb);
          TCOPYOUT_ACC(gC, tACC);       
        }

      }
    }

    if constexpr(rM>0){
      tile_shapeA tA[rM][R.k];
      
      #pragma clang loop unroll(full)
      for(int i=0;i<rM;i++){

        //copy in remaining M dimension A tile in K dim
        #pragma clang loop unroll(full)
        for(int k=0;k<R.k;k++){
          auto gA = gIterA(i+dM*R.m,k);
          TCOPYIN(tA[i][k], gA);
        }

        #pragma clang loop unroll(full)
        for(int j=0;j<Nb;j++){
          tile_shapeACC tACC;

          #pragma clang loop unroll(full)
          for(int k=0;k<R.k;k++){
            auto gB = gIterB(k,j);
            tile_shapeB tB;
            TCOPYIN(tB, gB);
            if(k==0){
              MATMUL(tACC, tA[i][k], tB);
            }else{
              MATMACC(tACC, tA[i][k], tB);
            }
          }

          if constexpr(R.k < Kb){
            for(int k=R.k;k<Kb;k++){
              tile_shapeA tA;
              tile_shapeB tB;
              auto gA = gIterA(i+dM*R.m,k);
              auto gB = gIterB(k,j);
              TCOPYIN(tA,gA);
              TCOPYIN(tB,gB);
              MATMACC(tACC, tA, tB);
            }
          }

          // [rM, n, rmd_K]
          if constexpr (rmd_K) {
            auto gA = gIterA(i+dM*R.m, Kb);
            auto gB = gIterB(Kb, j);

            tile_shapeA_trows tA;
            tile_shapeB_tcols tB;

            TCOPYIN(tA, gA);
            TCOPYIN(tB, gB);
            if constexpr(Kb>0){
            MATMACC(tACC, tA, tB);
            } else {
              MATMUL(tACC, tA, tB);
            }
          }
          auto gC = gIterC(i+dM*R.m,j);
          TCOPYOUT_ACC(gC, tACC);
        }

        // [rM, rmd_N, k]
        if constexpr (rmd_N) {
          tile_shapeC_trows tACC;

          #pragma clang loop unroll(full)
          for(int k=0;k<R.k;k++){
            auto gB = gIterB(k,Nb);
            tile_shapeB_trows tB;
            TCOPYIN(tB, gB);
            if(k==0){
              MATMUL(tACC, tA[i][k], tB);
            }else{
              MATMACC(tACC, tA[i][k], tB);
            }
          }

          if constexpr(R.k < Kb){
            for(int k=R.k;k<Kb;k++){
              tile_shapeA tA;
              tile_shapeB_trows tB;
              auto gA = gIterA(i+dM*R.m,k);
              auto gB = gIterB(k,Nb);
              TCOPYIN(tA,gA);
              TCOPYIN(tB,gB);
              MATMACC(tACC, tA, tB);
            }
          }

          // [rM, rmd_N, rmd_K]
          if constexpr (rmd_K) {
            auto gA = gIterA(i+dM*R.m, Kb);
            auto gB = gIterB(Kb, Nb);

            tile_shapeA_trows tA;
            tile_shapeB_tcorner tB;

            TCOPYIN(tA, gA);
            TCOPYIN(tB, gB);
            if constexpr(Kb>0){
            MATMACC(tACC, tA, tB);
            } else {
              MATMUL(tACC, tA, tB);
            }
          }
          auto gC = gIterC(i+dM*R.m,Nb);
          TCOPYOUT_ACC(gC, tACC);        
        }
      }
    }

    // [rmd_M, n, k]
    if constexpr (rmd_M) {
      tile_shapeA_tcols tA[R.k];
      
      #pragma clang loop unroll(full)
      for(int k=0;k<R.k;k++){
        auto gA = gIterA(Mb,k);
        TCOPYIN(tA[k], gA);
      }

      #pragma clang loop unroll(full)
      for(int j=0;j<Nb;j++){
        tile_shapeC_tcols tACC;

        #pragma clang loop unroll(full)
        for(int k=0;k<R.k;k++){
          auto gB = gIterB(k,j);
          tile_shapeB tB;
          TCOPYIN(tB, gB);
          if(k==0){
            MATMUL(tACC, tA[k], tB);
          }else{
            MATMACC(tACC, tA[k], tB);
          }
        }

        if constexpr(R.k < Kb){
          for(int k=R.k;k<Kb;k++){
            tile_shapeA_tcols tA;
            tile_shapeB tB;
            auto gA = gIterA(Mb,k);
            auto gB = gIterB(k,j);
            TCOPYIN(tA,gA);
            TCOPYIN(tB,gB);
            MATMACC(tACC, tA, tB);
          }
        }

        // [rmd_M, n, rmd_K]
        if constexpr (rmd_K) {
          auto gA = gIterA(Mb, Kb);
          auto gB = gIterB(Kb, j);

          tile_shapeA_tcorner tA;
          tile_shapeB_tcols tB;

          TCOPYIN(tA, gA);
          TCOPYIN(tB, gB);
          if constexpr(Kb>0){
          MATMACC(tACC, tA, tB);
          } else {
            MATMUL(tACC, tA, tB);
          }
        }
        auto gC = gIterC(Mb,j);
        TCOPYOUT_ACC(gC, tACC);
      }

      // [rmd_M, rmd_N, k]
      if constexpr (rmd_N) {
        tile_shapeC_tcorner tACC;

        #pragma clang loop unroll(full)
        for(int k=0;k<R.k;k++){
          auto gB = gIterB(k,Nb);
          tile_shapeB_trows tB;
          TCOPYIN(tB, gB);
          if(k==0){
            MATMUL(tACC, tA[k], tB);
          }else{
            MATMACC(tACC, tA[k], tB);
          }
        }

        if constexpr(R.k < Kb){
          for(int k=R.k;k<Kb;k++){
            tile_shapeA_tcols tA;
            tile_shapeB_trows tB;
            auto gA = gIterA(Mb,k);
            auto gB = gIterB(k,Nb);
            TCOPYIN(tA,gA);
            TCOPYIN(tB,gB);
            MATMACC(tACC, tA, tB);
          }
        }

        // [rmd_M, rmd_N, rmd_K]
        if constexpr (rmd_K) {
          auto gA = gIterA(Mb, Kb);
          auto gB = gIterB(Kb, Nb);

          tile_shapeA_tcorner tA;
          tile_shapeB_tcorner tB;

          TCOPYIN(tA, gA);
          TCOPYIN(tB, gB);
          if constexpr(Kb>0){
          MATMACC(tACC, tA, tB);
          } else {
            MATMUL(tACC, tA, tB);
          }
        }
        auto gC = gIterC(Mb,Nb);
        TCOPYOUT_ACC(gC, tACC);        
      }
    }
  }// Batch
}

template<typename dtype, const int gM, const int gN, const int gK, const int tM, const int tN, const int tK>
void matmul_mask_reuseA_OPT(float *dst, dtype *src0, dtype *src1){
  const int MAX_TILE_NUM = 20;
  using gm_shapeA = global_tensor<dtype, RowMajor<gM, gK>>;
  using gm_shapeB = global_tensor<dtype, RowMajor<gK, gN>>;
  using gm_shapeC = global_tensor<float, RowMajor<gM, gN>>;
  using tile_shapeA = TileLeft<dtype, tM, tK>;
  using tile_shapeB = TileRight<dtype, tK, tN>;
  using tile_shapeACC = TileAcc<float, tM, tN>;

  using itA = global_iterator<gm_shapeA, tile_shapeA>;
  using itB = global_iterator<gm_shapeB, tile_shapeB>;
  using itC = global_iterator<gm_shapeC, tile_shapeACC>;

  itA gIterA(src0);
  itB gIterB(src1);
  itC gIterC(dst);

  constexpr int Mb = gM / tM;
  constexpr int Nb = gN / tN;
  constexpr int Kb = gK / tK;

  constexpr int rmd_M = gM % tM;
  constexpr int rmd_N = gN % tN;
  constexpr int rmd_K = gK % tK;

  using tile_shapeA_trows   = TileLeft<dtype,  tM, tK,  tM,    rmd_K>;
  using tile_shapeA_tcols   = TileLeft<dtype,  tM, tK,  rmd_M, tK>;
  using tile_shapeA_tcorner = TileLeft<dtype,  tM, tK,  rmd_M, rmd_K>;

  using tile_shapeB_trows   = TileRight<dtype, tK, tN,  tK,    rmd_N>;
  using tile_shapeB_tcols   = TileRight<dtype, tK, tN,  rmd_K, tN>;
  using tile_shapeB_tcorner = TileRight<dtype, tK, tN,  rmd_K, rmd_N>;

  using tile_shapeC_trows   = TileAcc<float,   tM, tN,  tM,    rmd_N>;
  using tile_shapeC_tcols   = TileAcc<float,   tM, tN,  rmd_M, tN>;
  using tile_shapeC_tcorner = TileAcc<float,   tM, tN,  rmd_M, rmd_N>;

  constexpr ResA R = find_reuseA(Mb, Kb, MAX_TILE_NUM);
  static_assert(R.val <= MAX_TILE_NUM, "R.val is bigger than MAX_TILE_NUM");

  const int dM = (R.m == 0) ? 0 : Mb / R.m;
  const int rM = (R.m == 0) ? 0 : Mb % R.m;

  // 剩余 K 轴（超出 R.k 的部分）按 MAX_TILE_NUM 再做切分
  constexpr int remain_K  = (Kb > R.k) ? (Kb - R.k) : 0;
  constexpr int K2_chunks = (remain_K > 0) ? (remain_K / MAX_TILE_NUM) : 0;
  constexpr int K2_rem    = (remain_K > 0) ? (remain_K % MAX_TILE_NUM) : 0;

  for (int b = 0; b < Batch; b++) {

    // ========================================================
    // SECTION 1: 主 M 块 (dM * R.m 行)
    // ========================================================
    #pragma clang loop unroll(full)
    for (int i = 0; i < dM; i++) {

      #pragma clang loop unroll(full)
      for (int ii = 0; ii < R.m; ii++) {
        const int row = i * R.m + ii;

        // Phase A: 加载 A[row][0..R.k)，复用于所有 N 列
        tile_shapeA tA_phase0[R.k];
        #pragma clang loop unroll(full)
        for (int k = 0; k < R.k; k++) {
          auto gA = gIterA(row, k);
          TCOPYIN(tA_phase0[k], gA);
        }

        // --- N 主列 ---
        #pragma clang loop unroll(full)
        for (int j = 0; j < Nb; j++) {
          tile_shapeACC tACC;
          #pragma clang loop unroll(full)
          for (int k = 0; k < R.k; k++) {
            tile_shapeB tB;
            auto gB = gIterB(k, j);
            TCOPYIN(tB, gB);
            if (k == 0) MATMUL (tACC, tA_phase0[k], tB);
            else        MATMACC(tACC, tA_phase0[k], tB);
          }
          auto gC = gIterC(row, j);
          TCOPYOUT_ACC(gC, tACC);
        }

        // --- N 余列 (rmd_N) ---
        if constexpr (rmd_N) {
          tile_shapeC_trows tACC;
          #pragma clang loop unroll(full)
          for (int k = 0; k < R.k; k++) {
            tile_shapeB_trows tB;
            auto gB = gIterB(k, Nb);
            TCOPYIN(tB, gB);
            if (k == 0) MATMUL (tACC, tA_phase0[k], tB);
            else        MATMACC(tACC, tA_phase0[k], tB);
          }
          auto gC = gIterC(row, Nb);
          TCOPYOUT_ACC(gC, tACC);
        }

        // Phase B-1: 剩余 K 轴 Full chunks (每块 MAX_TILE_NUM 个 k tile)
        // K 在外加载 A，N 在内复用 A，每块输出一个部分和
        if constexpr (K2_chunks > 0) {
          #pragma clang loop unroll(full)
          for (int kc = 0; kc < K2_chunks; kc++) {
            const int k_base = R.k + kc * MAX_TILE_NUM;

            tile_shapeA tA_chunk[MAX_TILE_NUM];
            #pragma clang loop unroll(full)
            for (int k = 0; k < MAX_TILE_NUM; k++) {
              auto gA = gIterA(row, k_base + k);
              TCOPYIN(tA_chunk[k], gA);
            }

            // --- N 主列 ---
            #pragma clang loop unroll(full)
            for (int j = 0; j < Nb; j++) {
              tile_shapeACC tACC;
              #pragma clang loop unroll(full)
              for (int k = 0; k < MAX_TILE_NUM; k++) {
                tile_shapeB tB;
                auto gB = gIterB(k_base + k, j);
                TCOPYIN(tB, gB);
                if (k == 0) MATMUL (tACC, tA_chunk[k], tB);
                else        MATMACC(tACC, tA_chunk[k], tB);
              }
              auto gC = gIterC(row, j);
              TCOPYOUT_ACC(gC, tACC);
            }

            // --- N 余列 ---
            if constexpr (rmd_N) {
              tile_shapeC_trows tACC;
              #pragma clang loop unroll(full)
              for (int k = 0; k < MAX_TILE_NUM; k++) {
                tile_shapeB_trows tB;
                auto gB = gIterB(k_base + k, Nb);
                TCOPYIN(tB, gB);
                if (k == 0) MATMUL (tACC, tA_chunk[k], tB);
                else        MATMACC(tACC, tA_chunk[k], tB);
              }
              auto gC = gIterC(row, Nb);
              TCOPYOUT_ACC(gC, tACC);
            }
          }
        }

        // Phase B-2: 剩余 K 轴 Tail chunk (K2_rem 个 k tile)
        if constexpr (K2_rem > 0) {
          const int k_base = R.k + K2_chunks * MAX_TILE_NUM;

          tile_shapeA tA_tail[K2_rem];
          #pragma clang loop unroll(full)
          for (int k = 0; k < K2_rem; k++) {
            auto gA = gIterA(row, k_base + k);
            TCOPYIN(tA_tail[k], gA);
          }

          // --- N 主列 ---
          #pragma clang loop unroll(full)
          for (int j = 0; j < Nb; j++) {
            tile_shapeACC tACC;
            #pragma clang loop unroll(full)
            for (int k = 0; k < K2_rem; k++) {
              tile_shapeB tB;
              auto gB = gIterB(k_base + k, j);
              TCOPYIN(tB, gB);
              if (k == 0) MATMUL (tACC, tA_tail[k], tB);
              else        MATMACC(tACC, tA_tail[k], tB);
            }
            auto gC = gIterC(row, j);
            TCOPYOUT_ACC(gC, tACC);
          }

          // --- N 余列 ---
          if constexpr (rmd_N) {
            tile_shapeC_trows tACC;
            #pragma clang loop unroll(full)
            for (int k = 0; k < K2_rem; k++) {
              tile_shapeB_trows tB;
              auto gB = gIterB(k_base + k, Nb);
              TCOPYIN(tB, gB);
              if (k == 0) MATMUL (tACC, tA_tail[k], tB);
              else        MATMACC(tACC, tA_tail[k], tB);
            }
            auto gC = gIterC(row, Nb);
            TCOPYOUT_ACC(gC, tACC);
          }
        }

        // Phase C: rmd_K 尾列（A 只有一个 tile，直接复用于所有 N）
        if constexpr (rmd_K) {
          tile_shapeA_trows tA_rmdK;
          auto gA = gIterA(row, Kb);
          TCOPYIN(tA_rmdK, gA);

          // --- N 主列 ---
          #pragma clang loop unroll(full)
          for (int j = 0; j < Nb; j++) {
            tile_shapeACC tACC;
            tile_shapeB_tcols tB;
            auto gB = gIterB(Kb, j);
            TCOPYIN(tB, gB);
            if constexpr (Kb > 0) MATMACC(tACC, tA_rmdK, tB);
            else                  MATMUL (tACC, tA_rmdK, tB);
            auto gC = gIterC(row, j);
            TCOPYOUT_ACC(gC, tACC);
          }

          // --- N 余列 ---
          if constexpr (rmd_N) {
            tile_shapeC_trows tACC;
            tile_shapeB_tcorner tB;
            auto gB = gIterB(Kb, Nb);
            TCOPYIN(tB, gB);
            if constexpr (Kb > 0) MATMACC(tACC, tA_rmdK, tB);
            else                  MATMUL (tACC, tA_rmdK, tB);
            auto gC = gIterC(row, Nb);
            TCOPYOUT_ACC(gC, tACC);
          }
        }

      } // ii
    } // i (dM)

    // ========================================================
    // SECTION 2: 余 M 块 (rM 行)
    // ========================================================
    if constexpr (rM > 0) {
      #pragma clang loop unroll(full)
      for (int i = 0; i < rM; i++) {
        const int row = dM * R.m + i;

        // Phase A
        tile_shapeA tA_phase0[R.k];
        #pragma clang loop unroll(full)
        for (int k = 0; k < R.k; k++) {
          auto gA = gIterA(row, k);
          TCOPYIN(tA_phase0[k], gA);
        }

        #pragma clang loop unroll(full)
        for (int j = 0; j < Nb; j++) {
          tile_shapeACC tACC;
          #pragma clang loop unroll(full)
          for (int k = 0; k < R.k; k++) {
            tile_shapeB tB;
            auto gB = gIterB(k, j);
            TCOPYIN(tB, gB);
            if (k == 0) MATMUL (tACC, tA_phase0[k], tB);
            else        MATMACC(tACC, tA_phase0[k], tB);
          }
          auto gC = gIterC(row, j);
          TCOPYOUT_ACC(gC, tACC);
        }

        if constexpr (rmd_N) {
          tile_shapeC_trows tACC;
          #pragma clang loop unroll(full)
          for (int k = 0; k < R.k; k++) {
            tile_shapeB_trows tB;
            auto gB = gIterB(k, Nb);
            TCOPYIN(tB, gB);
            if (k == 0) MATMUL (tACC, tA_phase0[k], tB);
            else        MATMACC(tACC, tA_phase0[k], tB);
          }
          auto gC = gIterC(row, Nb);
          TCOPYOUT_ACC(gC, tACC);
        }

        // Phase B-1: Full chunks
        if constexpr (K2_chunks > 0) {
          #pragma clang loop unroll(full)
          for (int kc = 0; kc < K2_chunks; kc++) {
            const int k_base = R.k + kc * MAX_TILE_NUM;

            tile_shapeA tA_chunk[MAX_TILE_NUM];
            #pragma clang loop unroll(full)
            for (int k = 0; k < MAX_TILE_NUM; k++) {
              auto gA = gIterA(row, k_base + k);
              TCOPYIN(tA_chunk[k], gA);
            }

            #pragma clang loop unroll(full)
            for (int j = 0; j < Nb; j++) {
              tile_shapeACC tACC;
              #pragma clang loop unroll(full)
              for (int k = 0; k < MAX_TILE_NUM; k++) {
                tile_shapeB tB;
                auto gB = gIterB(k_base + k, j);
                TCOPYIN(tB, gB);
                if (k == 0) MATMUL (tACC, tA_chunk[k], tB);
                else        MATMACC(tACC, tA_chunk[k], tB);
              }
              auto gC = gIterC(row, j);
              TCOPYOUT_ACC(gC, tACC);
            }

            if constexpr (rmd_N) {
              tile_shapeC_trows tACC;
              #pragma clang loop unroll(full)
              for (int k = 0; k < MAX_TILE_NUM; k++) {
                tile_shapeB_trows tB;
                auto gB = gIterB(k_base + k, Nb);
                TCOPYIN(tB, gB);
                if (k == 0) MATMUL (tACC, tA_chunk[k], tB);
                else        MATMACC(tACC, tA_chunk[k], tB);
              }
              auto gC = gIterC(row, Nb);
              TCOPYOUT_ACC(gC, tACC);
            }
          }
        }

        // Phase B-2: Tail chunk
        if constexpr (K2_rem > 0) {
          const int k_base = R.k + K2_chunks * MAX_TILE_NUM;

          tile_shapeA tA_tail[K2_rem];
          #pragma clang loop unroll(full)
          for (int k = 0; k < K2_rem; k++) {
            auto gA = gIterA(row, k_base + k);
            TCOPYIN(tA_tail[k], gA);
          }

          #pragma clang loop unroll(full)
          for (int j = 0; j < Nb; j++) {
            tile_shapeACC tACC;
            #pragma clang loop unroll(full)
            for (int k = 0; k < K2_rem; k++) {
              tile_shapeB tB;
              auto gB = gIterB(k_base + k, j);
              TCOPYIN(tB, gB);
              if (k == 0) MATMUL (tACC, tA_tail[k], tB);
              else        MATMACC(tACC, tA_tail[k], tB);
            }
            auto gC = gIterC(row, j);
            TCOPYOUT_ACC(gC, tACC);
          }

          if constexpr (rmd_N) {
            tile_shapeC_trows tACC;
            #pragma clang loop unroll(full)
            for (int k = 0; k < K2_rem; k++) {
              tile_shapeB_trows tB;
              auto gB = gIterB(k_base + k, Nb);
              TCOPYIN(tB, gB);
              if (k == 0) MATMUL (tACC, tA_tail[k], tB);
              else        MATMACC(tACC, tA_tail[k], tB);
            }
            auto gC = gIterC(row, Nb);
            TCOPYOUT_ACC(gC, tACC);
          }
        }

        // Phase C: rmd_K
        if constexpr (rmd_K) {
          tile_shapeA_trows tA_rmdK;
          auto gA = gIterA(row, Kb);
          TCOPYIN(tA_rmdK, gA);

          #pragma clang loop unroll(full)
          for (int j = 0; j < Nb; j++) {
            tile_shapeACC tACC;
            tile_shapeB_tcols tB;
            auto gB = gIterB(Kb, j);
            TCOPYIN(tB, gB);
            if constexpr (Kb > 0) MATMACC(tACC, tA_rmdK, tB);
            else                  MATMUL (tACC, tA_rmdK, tB);
            auto gC = gIterC(row, j);
            TCOPYOUT_ACC(gC, tACC);
          }

          if constexpr (rmd_N) {
            tile_shapeC_trows tACC;
            tile_shapeB_tcorner tB;
            auto gB = gIterB(Kb, Nb);
            TCOPYIN(tB, gB);
            if constexpr (Kb > 0) MATMACC(tACC, tA_rmdK, tB);
            else                  MATMUL (tACC, tA_rmdK, tB);
            auto gC = gIterC(row, Nb);
            TCOPYOUT_ACC(gC, tACC);
          }
        }

      } // i (rM)
    }

    // ========================================================
    // SECTION 3: rmd_M 尾行 (行索引 = Mb，A 类型为 tcols)
    // ========================================================
    if constexpr (rmd_M) {

      // Phase A
      tile_shapeA_tcols tA_phase0[R.k];
      #pragma clang loop unroll(full)
      for (int k = 0; k < R.k; k++) {
        auto gA = gIterA(Mb, k);
        TCOPYIN(tA_phase0[k], gA);
      }

      #pragma clang loop unroll(full)
      for (int j = 0; j < Nb; j++) {
        tile_shapeC_tcols tACC;
        #pragma clang loop unroll(full)
        for (int k = 0; k < R.k; k++) {
          tile_shapeB tB;
          auto gB = gIterB(k, j);
          TCOPYIN(tB, gB);
          if (k == 0) MATMUL (tACC, tA_phase0[k], tB);
          else        MATMACC(tACC, tA_phase0[k], tB);
        }
        auto gC = gIterC(Mb, j);
        TCOPYOUT_ACC(gC, tACC);
      }

      if constexpr (rmd_N) {
        tile_shapeC_tcorner tACC;
        #pragma clang loop unroll(full)
        for (int k = 0; k < R.k; k++) {
          tile_shapeB_trows tB;
          auto gB = gIterB(k, Nb);
          TCOPYIN(tB, gB);
          if (k == 0) MATMUL (tACC, tA_phase0[k], tB);
          else        MATMACC(tACC, tA_phase0[k], tB);
        }
        auto gC = gIterC(Mb, Nb);
        TCOPYOUT_ACC(gC, tACC);
      }

      // Phase B-1: Full chunks (rmd_M 行，A 类型为 tcols)
      if constexpr (K2_chunks > 0) {
        #pragma clang loop unroll(full)
        for (int kc = 0; kc < K2_chunks; kc++) {
          const int k_base = R.k + kc * MAX_TILE_NUM;

          tile_shapeA_tcols tA_chunk[MAX_TILE_NUM];
          #pragma clang loop unroll(full)
          for (int k = 0; k < MAX_TILE_NUM; k++) {
            auto gA = gIterA(Mb, k_base + k);
            TCOPYIN(tA_chunk[k], gA);
          }

          #pragma clang loop unroll(full)
          for (int j = 0; j < Nb; j++) {
            tile_shapeC_tcols tACC;
            #pragma clang loop unroll(full)
            for (int k = 0; k < MAX_TILE_NUM; k++) {
              tile_shapeB tB;
              auto gB = gIterB(k_base + k, j);
              TCOPYIN(tB, gB);
              if (k == 0) MATMUL (tACC, tA_chunk[k], tB);
              else        MATMACC(tACC, tA_chunk[k], tB);
            }
            auto gC = gIterC(Mb, j);
            TCOPYOUT_ACC(gC, tACC);
          }

          if constexpr (rmd_N) {
            tile_shapeC_tcorner tACC;
            #pragma clang loop unroll(full)
            for (int k = 0; k < MAX_TILE_NUM; k++) {
              tile_shapeB_trows tB;
              auto gB = gIterB(k_base + k, Nb);
              TCOPYIN(tB, gB);
              if (k == 0) MATMUL (tACC, tA_chunk[k], tB);
              else        MATMACC(tACC, tA_chunk[k], tB);
            }
            auto gC = gIterC(Mb, Nb);
            TCOPYOUT_ACC(gC, tACC);
          }
        }
      }

      // Phase B-2: Tail chunk (rmd_M 行)
      if constexpr (K2_rem > 0) {
        const int k_base = R.k + K2_chunks * MAX_TILE_NUM;

        tile_shapeA_tcols tA_tail[K2_rem];
        #pragma clang loop unroll(full)
        for (int k = 0; k < K2_rem; k++) {
          auto gA = gIterA(Mb, k_base + k);
          TCOPYIN(tA_tail[k], gA);
        }

        #pragma clang loop unroll(full)
        for (int j = 0; j < Nb; j++) {
          tile_shapeC_tcols tACC;
          #pragma clang loop unroll(full)
          for (int k = 0; k < K2_rem; k++) {
            tile_shapeB tB;
            auto gB = gIterB(k_base + k, j);
            TCOPYIN(tB, gB);
            if (k == 0) MATMUL (tACC, tA_tail[k], tB);
            else        MATMACC(tACC, tA_tail[k], tB);
          }
          auto gC = gIterC(Mb, j);
          TCOPYOUT_ACC(gC, tACC);
        }

        if constexpr (rmd_N) {
          tile_shapeC_tcorner tACC;
          #pragma clang loop unroll(full)
          for (int k = 0; k < K2_rem; k++) {
            tile_shapeB_trows tB;
            auto gB = gIterB(k_base + k, Nb);
            TCOPYIN(tB, gB);
            if (k == 0) MATMUL (tACC, tA_tail[k], tB);
            else        MATMACC(tACC, tA_tail[k], tB);
          }
          auto gC = gIterC(Mb, Nb);
          TCOPYOUT_ACC(gC, tACC);
        }
      }

      // Phase C: rmd_K (rmd_M 行，A 类型为 tcorner)
      if constexpr (rmd_K) {
        tile_shapeA_tcorner tA_rmdK;
        auto gA = gIterA(Mb, Kb);
        TCOPYIN(tA_rmdK, gA);

        #pragma clang loop unroll(full)
        for (int j = 0; j < Nb; j++) {
          tile_shapeC_tcols tACC;
          tile_shapeB_tcols tB;
          auto gB = gIterB(Kb, j);
          TCOPYIN(tB, gB);
          if constexpr (Kb > 0) MATMACC(tACC, tA_rmdK, tB);
          else                  MATMUL (tACC, tA_rmdK, tB);
          auto gC = gIterC(Mb, j);
          TCOPYOUT_ACC(gC, tACC);
        }

        if constexpr (rmd_N) {
          tile_shapeC_tcorner tACC;
          tile_shapeB_tcorner tB;
          auto gB = gIterB(Kb, Nb);
          TCOPYIN(tB, gB);
          if constexpr (Kb > 0) MATMACC(tACC, tA_rmdK, tB);
          else                  MATMUL (tACC, tA_rmdK, tB);
          auto gC = gIterC(Mb, Nb);
          TCOPYOUT_ACC(gC, tACC);
        }
      }
    } // rmd_M
  } // Batch
}

template<typename dtype, const int gM, const int gN, const int gK,
         const int tM, const int tN, const int tK>
void matmul_mask_reuseA_OPT2(float *dst, dtype *src0, dtype *src1){
  constexpr int MAX_TILE_NUM = 14;

  using gm_shapeA = global_tensor<dtype, RowMajor<gM, gK>>;
  using gm_shapeB = global_tensor<dtype, RowMajor<gK, gN>>;
  using gm_shapeC = global_tensor<float, ColMajor<gM, gN>>;

  using tile_shapeA   = TileLeft <dtype, tM, tK>;
  using tile_shapeB   = TileRight<dtype, tK, tN>;
  using tile_shapeACC = TileAcc  <float, tM, tN>;

  using itA = global_iterator<gm_shapeA, tile_shapeA>;
  using itB = global_iterator<gm_shapeB, tile_shapeB>;
  using itC = global_iterator<gm_shapeC, tile_shapeACC>;

  itA gIterA(src0);
  itB gIterB(src1);
  itC gIterC(dst);

  constexpr int Mb = gM / tM;
  constexpr int Nb = gN / tN;
  constexpr int Kb = gK / tK;

  constexpr int rmd_M = gM % tM;
  constexpr int rmd_N = gN % tN;
  constexpr int rmd_K = gK % tK;

  // —— A / B 余块 ——
  using tile_shapeA_trows   = TileLeft <dtype, tM, tK, tM,    rmd_K>;
  using tile_shapeA_tcols   = TileLeft <dtype, tM, tK, rmd_M, tK>;
  using tile_shapeA_tcorner = TileLeft <dtype, tM, tK, rmd_M, rmd_K>;

  using tile_shapeB_trows   = TileRight<dtype, tK, tN, tK,    rmd_N>;
  using tile_shapeB_tcols   = TileRight<dtype, tK, tN, rmd_K, tN>;
  using tile_shapeB_tcorner = TileRight<dtype, tK, tN, rmd_K, rmd_N>;

  // —— ACC 余块 ——
  using tile_shapeACC_trows   = TileAcc<float, tM, tN, tM,    rmd_N>;
  using tile_shapeACC_tcols   = TileAcc<float, tM, tN, rmd_M, tN>;
  using tile_shapeACC_tcorner = TileAcc<float, tM, tN, rmd_M, rmd_N>;

  // —— Vec 常驻 C（fp32, ColMajor）——
  using tile_C_out         = Tile<Location::Vec, float, tM, tN,
                                  BLayout::ColMajor, tM,    tN>;
  using tile_C_out_trows   = Tile<Location::Vec, float, tM, tN,
                                  BLayout::ColMajor, tM,    rmd_N>;
  using tile_C_out_tcols   = Tile<Location::Vec, float, tM, tN,
                                  BLayout::ColMajor, rmd_M, tN>;
  using tile_C_out_tcorner = Tile<Location::Vec, float, tM, tN,
                                  BLayout::ColMajor, rmd_M, rmd_N>;

  // —— 写回时的 bf16 临时 tile（RowMajor）——
  using tile_C_bf16         = Tile<Location::Vec, float, tM, tN,
                                   BLayout::RowMajor, tM,    tN>;
  using tile_C_bf16_trows   = Tile<Location::Vec, float, tM, tN,
                                   BLayout::RowMajor, tM,    rmd_N>;
  using tile_C_bf16_tcols   = Tile<Location::Vec, float, tM, tN,
                                   BLayout::RowMajor, rmd_M, tN>;
  using tile_C_bf16_tcorner = Tile<Location::Vec, float, tM, tN,
                                   BLayout::RowMajor, rmd_M, rmd_N>;

  // —— K 切分 ——
  constexpr int K_chunks = Kb / MAX_TILE_NUM;
  constexpr int K_rem    = Kb % MAX_TILE_NUM;

  static_assert(K_chunks > 0 || K_rem > 0 || rmd_K > 0, "empty K");

  for (int b = 0; b < Batch; b++) {

    // Vec 常驻 C
    tile_C_out         tC_main[Mb][Nb];
    tile_C_out_trows   tC_rcol[Mb];
    tile_C_out_tcols   tC_rrow[Nb];
    tile_C_out_tcorner tC_corner;

    // ============================================================
    // 段 1：等长 chunks (LEN = MAX_TILE_NUM)
    //   chunk[0]    : is_first = true
    //   chunk[1..*] : is_first = false
    // ============================================================
    if constexpr (K_chunks > 0) {

      // ---------- chunk[0]: is_first = true ----------
      {
        constexpr int LEN     = MAX_TILE_NUM;
        constexpr int k_base  = 0;

        // ----- 主 M 块 -----
        #pragma clang loop unroll(full)
        for (int row = 0; row < Mb; row++) {
          tile_shapeA tA[LEN];
          #pragma clang loop unroll(full)
          for (int k = 0; k < LEN; k++) {
            auto gA = gIterA(row, k_base + k);
            TCOPYIN(tA[k], gA);
          }

          #pragma clang loop unroll(full)
          for (int j = 0; j < Nb; j++) {
            tile_shapeACC tACC;
            #pragma clang loop unroll(full)
            for (int k = 0; k < LEN; k++) {
              tile_shapeB tB;
              auto gB = gIterB(k_base + k, j);
              TCOPYIN(tB, gB);
              if (k == 0) MATMUL (tACC, tA[k], tB);
              else        MATMACC(tACC, tA[k], tB);
            }
            TCVT(tC_main[row][j], tACC);   // is_first: 直接覆盖
          }

          if constexpr (rmd_N) {
            tile_shapeACC_trows tACC;
            #pragma clang loop unroll(full)
            for (int k = 0; k < LEN; k++) {
              tile_shapeB_trows tB;
              auto gB = gIterB(k_base + k, Nb);
              TCOPYIN(tB, gB);
              if (k == 0) MATMUL (tACC, tA[k], tB);
              else        MATMACC(tACC, tA[k], tB);
            }
            TCVT(tC_rcol[row], tACC);
          }
        } // row

        // ----- rmd_M 尾行 -----
        if constexpr (rmd_M) {
          tile_shapeA_tcols tA[LEN];
          #pragma clang loop unroll(full)
          for (int k = 0; k < LEN; k++) {
            auto gA = gIterA(Mb, k_base + k);
            TCOPYIN(tA[k], gA);
          }

          #pragma clang loop unroll(full)
          for (int j = 0; j < Nb; j++) {
            tile_shapeACC_tcols tACC;
            #pragma clang loop unroll(full)
            for (int k = 0; k < LEN; k++) {
              tile_shapeB tB;
              auto gB = gIterB(k_base + k, j);
              TCOPYIN(tB, gB);
              if (k == 0) MATMUL (tACC, tA[k], tB);
              else        MATMACC(tACC, tA[k], tB);
            }
            TCVT(tC_rrow[j], tACC);
          }

          if constexpr (rmd_N) {
            tile_shapeACC_tcorner tACC;
            #pragma clang loop unroll(full)
            for (int k = 0; k < LEN; k++) {
              tile_shapeB_trows tB;
              auto gB = gIterB(k_base + k, Nb);
              TCOPYIN(tB, gB);
              if (k == 0) MATMUL (tACC, tA[k], tB);
              else        MATMACC(tACC, tA[k], tB);
            }
            TCVT(tC_corner, tACC);
          }
        }
      } // chunk[0]

      // ---------- chunk[1 .. K_chunks-1]: is_first = false ----------
      if constexpr (K_chunks > 1) {
        #pragma clang loop unroll(full)
        for (int kc = 1; kc < K_chunks; kc++) {
          constexpr int LEN = MAX_TILE_NUM;
          const int k_base  = kc * MAX_TILE_NUM;

          // 主 M 块
          #pragma clang loop unroll(full)
          for (int row = 0; row < Mb; row++) {
            tile_shapeA tA[LEN];
            #pragma clang loop unroll(full)
            for (int k = 0; k < LEN; k++) {
              auto gA = gIterA(row, k_base + k);
              TCOPYIN(tA[k], gA);
            }

            #pragma clang loop unroll(full)
            for (int j = 0; j < Nb; j++) {
              tile_shapeACC tACC;
              #pragma clang loop unroll(full)
              for (int k = 0; k < LEN; k++) {
                tile_shapeB tB;
                auto gB = gIterB(k_base + k, j);
                TCOPYIN(tB, gB);
                if (k == 0) MATMUL (tACC, tA[k], tB);
                else        MATMACC(tACC, tA[k], tB);
              }
              tile_C_out tACC_vec;
              TCVT(tACC_vec, tACC);
              TADD(tC_main[row][j], tC_main[row][j], tACC_vec);
            }

            if constexpr (rmd_N) {
              tile_shapeACC_trows tACC;
              #pragma clang loop unroll(full)
              for (int k = 0; k < LEN; k++) {
                tile_shapeB_trows tB;
                auto gB = gIterB(k_base + k, Nb);
                TCOPYIN(tB, gB);
                if (k == 0) MATMUL (tACC, tA[k], tB);
                else        MATMACC(tACC, tA[k], tB);
              }
              tile_C_out_trows tACC_vec;
              TCVT(tACC_vec, tACC);
              TADD(tC_rcol[row], tC_rcol[row], tACC_vec);
            }
          } // row

          if constexpr (rmd_M) {
            tile_shapeA_tcols tA[LEN];
            #pragma clang loop unroll(full)
            for (int k = 0; k < LEN; k++) {
              auto gA = gIterA(Mb, k_base + k);
              TCOPYIN(tA[k], gA);
            }

            #pragma clang loop unroll(full)
            for (int j = 0; j < Nb; j++) {
              tile_shapeACC_tcols tACC;
              #pragma clang loop unroll(full)
              for (int k = 0; k < LEN; k++) {
                tile_shapeB tB;
                auto gB = gIterB(k_base + k, j);
                TCOPYIN(tB, gB);
                if (k == 0) MATMUL (tACC, tA[k], tB);
                else        MATMACC(tACC, tA[k], tB);
              }
              tile_C_out_tcols tACC_vec;
              TCVT(tACC_vec, tACC);
              TADD(tC_rrow[j], tC_rrow[j], tACC_vec);
            }

            if constexpr (rmd_N) {
              tile_shapeACC_tcorner tACC;
              #pragma clang loop unroll(full)
              for (int k = 0; k < LEN; k++) {
                tile_shapeB_trows tB;
                auto gB = gIterB(k_base + k, Nb);
                TCOPYIN(tB, gB);
                if (k == 0) MATMUL (tACC, tA[k], tB);
                else        MATMACC(tACC, tA[k], tB);
              }
              tile_C_out_tcorner tACC_vec;
              TCVT(tACC_vec, tACC);
              TADD(tC_corner, tC_corner, tACC_vec);
            }
          }
        } // kc
      } // K_chunks > 1
    } // K_chunks > 0

    // ============================================================
    // 段 2：K_rem (LEN = K_rem)
    //   is_first = (K_chunks == 0)
    // ============================================================
    if constexpr (K_rem > 0) {
      constexpr int LEN     = K_rem;
      constexpr int k_base  = K_chunks * MAX_TILE_NUM;
      constexpr bool is_first = (K_chunks == 0);

      #pragma clang loop unroll(full)
      for (int row = 0; row < Mb; row++) {
        tile_shapeA tA[LEN];
        #pragma clang loop unroll(full)
        for (int k = 0; k < LEN; k++) {
          auto gA = gIterA(row, k_base + k);
          TCOPYIN(tA[k], gA);
        }

        #pragma clang loop unroll(full)
        for (int j = 0; j < Nb; j++) {
          tile_shapeACC tACC;
          #pragma clang loop unroll(full)
          for (int k = 0; k < LEN; k++) {
            tile_shapeB tB;
            auto gB = gIterB(k_base + k, j);
            TCOPYIN(tB, gB);
            if (k == 0) MATMUL (tACC, tA[k], tB);
            else        MATMACC(tACC, tA[k], tB);
          }
          if constexpr (is_first) {
            TCVT(tC_main[row][j], tACC);
          } else {
            tile_C_out tACC_vec;
            TCVT(tACC_vec, tACC);
            TADD(tC_main[row][j], tC_main[row][j], tACC_vec);
          }
        }

        if constexpr (rmd_N) {
          tile_shapeACC_trows tACC;
          #pragma clang loop unroll(full)
          for (int k = 0; k < LEN; k++) {
            tile_shapeB_trows tB;
            auto gB = gIterB(k_base + k, Nb);
            TCOPYIN(tB, gB);
            if (k == 0) MATMUL (tACC, tA[k], tB);
            else        MATMACC(tACC, tA[k], tB);
          }
          if constexpr (is_first) {
            TCVT(tC_rcol[row], tACC);
          } else {
            tile_C_out_trows tACC_vec;
            TCVT(tACC_vec, tACC);
            TADD(tC_rcol[row], tC_rcol[row], tACC_vec);
          }
        }
      }

      if constexpr (rmd_M) {
        tile_shapeA_tcols tA[LEN];
        #pragma clang loop unroll(full)
        for (int k = 0; k < LEN; k++) {
          auto gA = gIterA(Mb, k_base + k);
          TCOPYIN(tA[k], gA);
        }

        #pragma clang loop unroll(full)
        for (int j = 0; j < Nb; j++) {
          tile_shapeACC_tcols tACC;
          #pragma clang loop unroll(full)
          for (int k = 0; k < LEN; k++) {
            tile_shapeB tB;
            auto gB = gIterB(k_base + k, j);
            TCOPYIN(tB, gB);
            if (k == 0) MATMUL (tACC, tA[k], tB);
            else        MATMACC(tACC, tA[k], tB);
          }
          if constexpr (is_first) {
            TCVT(tC_rrow[j], tACC);
          } else {
            tile_C_out_tcols tACC_vec;
            TCVT(tACC_vec, tACC);
            TADD(tC_rrow[j], tC_rrow[j], tACC_vec);
          }
        }

        if constexpr (rmd_N) {
          tile_shapeACC_tcorner tACC;
          #pragma clang loop unroll(full)
          for (int k = 0; k < LEN; k++) {
            tile_shapeB_trows tB;
            auto gB = gIterB(k_base + k, Nb);
            TCOPYIN(tB, gB);
            if (k == 0) MATMUL (tACC, tA[k], tB);
            else        MATMACC(tACC, tA[k], tB);
          }
          if constexpr (is_first) {
            TCVT(tC_corner, tACC);
          } else {
            tile_C_out_tcorner tACC_vec;
            TCVT(tACC_vec, tACC);
            TADD(tC_corner, tC_corner, tACC_vec);
          }
        }
      }
    } // K_rem

    // ============================================================
    // 段 3：rmd_K
    //   is_first = (K_chunks == 0) && (K_rem == 0)
    // ============================================================
    if constexpr (rmd_K) {
      constexpr bool is_first = (K_chunks == 0) && (K_rem == 0);

      #pragma clang loop unroll(full)
      for (int row = 0; row < Mb; row++) {
        tile_shapeA_trows tA_rmdK;
        auto gA = gIterA(row, Kb);
        TCOPYIN(tA_rmdK, gA);

        #pragma clang loop unroll(full)
        for (int j = 0; j < Nb; j++) {
          tile_shapeACC tACC;
          tile_shapeB_tcols tB;
          auto gB = gIterB(Kb, j);
          TCOPYIN(tB, gB);
          MATMUL(tACC, tA_rmdK, tB);

          if constexpr (is_first) {
            TCVT(tC_main[row][j], tACC);
          } else {
            tile_C_out tACC_vec;
            TCVT(tACC_vec, tACC);
            TADD(tC_main[row][j], tC_main[row][j], tACC_vec);
          }
        }

        if constexpr (rmd_N) {
          tile_shapeACC_trows tACC;
          tile_shapeB_tcorner tB;
          auto gB = gIterB(Kb, Nb);
          TCOPYIN(tB, gB);
          MATMUL(tACC, tA_rmdK, tB);

          if constexpr (is_first) {
            TCVT(tC_rcol[row], tACC);
          } else {
            tile_C_out_trows tACC_vec;
            TCVT(tACC_vec, tACC);
            TADD(tC_rcol[row], tC_rcol[row], tACC_vec);
          }
        }
      }

      if constexpr (rmd_M) {
        tile_shapeA_tcorner tA_rmdK;
        auto gA = gIterA(Mb, Kb);
        TCOPYIN(tA_rmdK, gA);

        #pragma clang loop unroll(full)
        for (int j = 0; j < Nb; j++) {
          tile_shapeACC_tcols tACC;
          tile_shapeB_tcols tB;
          auto gB = gIterB(Kb, j);
          TCOPYIN(tB, gB);
          MATMUL(tACC, tA_rmdK, tB);

          if constexpr (is_first) {
            TCVT(tC_rrow[j], tACC);
          } else {
            tile_C_out_tcols tACC_vec;
            TCVT(tACC_vec, tACC);
            TADD(tC_rrow[j], tC_rrow[j], tACC_vec);
          }
        }

        if constexpr (rmd_N) {
          tile_shapeACC_tcorner tACC;
          tile_shapeB_tcorner tB;
          auto gB = gIterB(Kb, Nb);
          TCOPYIN(tB, gB);
          MATMUL(tACC, tA_rmdK, tB);

          if constexpr (is_first) {
            TCVT(tC_corner, tACC);
          } else {
            tile_C_out_tcorner tACC_vec;
            TCVT(tACC_vec, tACC);
            TADD(tC_corner, tC_corner, tACC_vec);
          }
        }
      }
    } // rmd_K

    // ============================================================
    // 全部 K 累加完成后，统一写回 global
    // ============================================================
    #pragma clang loop unroll(full)
    for (int m = 0; m < Mb; m++) {
      #pragma clang loop unroll(full)
      for (int n = 0; n < Nb; n++) {
        tile_C_bf16 tC_b;
        // TMOV_NZ2DN(tC_b, tC_main[m][n]);
        auto gC = gIterC(m, n);
        // TCOPYOUT(gC, tC_b);
        TCOPYOUT(gC, tC_main[m][n]);
      }
      if constexpr (rmd_N) {
        tile_C_bf16_trows tC_b;
        // TMOV_NZ2DN(tC_b, tC_rcol[m]);
        auto gC = gIterC(m, Nb);
        TCOPYOUT(gC, tC_rcol[m]);
        // TCOPYOUT(gC, tC_b);
      }
    }
    if constexpr (rmd_M) {
      #pragma clang loop unroll(full)
      for (int n = 0; n < Nb; n++) {
        tile_C_bf16_tcols tC_b;
        // TMOV_NZ2DN(tC_b, tC_rrow[n]);
        auto gC = gIterC(Mb, n);
        TCOPYOUT(gC,  tC_rrow[n]);
        // TCOPYOUT(gC, tC_b);
      }
      if constexpr (rmd_N) {
        tile_C_bf16_tcorner tC_b;
        // TMOV_NZ2DN(tC_b, tC_corner);
        auto gC = gIterC(Mb, Nb);
        TCOPYOUT(gC, tC_corner);
        // TCOPYOUT(gC, tC_b);
      }
    }

  } // Batch
}

template<typename dtype, const int gM, const int gN, const int gK,
         const int tM, const int tN, const int tK>
void matmul_mask_reuseB_OPT2(float *dst, dtype *src0, dtype *src1){
  constexpr int MAX_TILE_NUM = 14;

  using gm_shapeA = global_tensor<dtype, RowMajor<gM, gK>>;
  using gm_shapeB = global_tensor<dtype, RowMajor<gK, gN>>;
  using gm_shapeC = global_tensor<float, ColMajor<gM, gN>>;

  using tile_shapeA   = TileLeft <dtype, tM, tK>;
  using tile_shapeB   = TileRight<dtype, tK, tN>;
  using tile_shapeACC = TileAcc  <float, tM, tN>;

  using itA = global_iterator<gm_shapeA, tile_shapeA>;
  using itB = global_iterator<gm_shapeB, tile_shapeB>;
  using itC = global_iterator<gm_shapeC, tile_shapeACC>;

  itA gIterA(src0);
  itB gIterB(src1);
  itC gIterC(dst);

  constexpr int Mb = gM / tM;
  constexpr int Nb = gN / tN;
  constexpr int Kb = gK / tK;

  constexpr int rmd_M = gM % tM;
  constexpr int rmd_N = gN % tN;
  constexpr int rmd_K = gK % tK;

  // —— A / B 余块 ——
  using tile_shapeA_trows   = TileLeft <dtype, tM, tK, tM,    rmd_K>;
  using tile_shapeA_tcols   = TileLeft <dtype, tM, tK, rmd_M, tK>;
  using tile_shapeA_tcorner = TileLeft <dtype, tM, tK, rmd_M, rmd_K>;

  using tile_shapeB_trows   = TileRight<dtype, tK, tN, tK,    rmd_N>;
  using tile_shapeB_tcols   = TileRight<dtype, tK, tN, rmd_K, tN>;
  using tile_shapeB_tcorner = TileRight<dtype, tK, tN, rmd_K, rmd_N>;

  // —— ACC 余块 ——
  using tile_shapeACC_trows   = TileAcc<float, tM, tN, tM,    rmd_N>;
  using tile_shapeACC_tcols   = TileAcc<float, tM, tN, rmd_M, tN>;
  using tile_shapeACC_tcorner = TileAcc<float, tM, tN, rmd_M, rmd_N>;

  // —— Vec 常驻 C（fp32, ColMajor）——
  using tile_C_out         = Tile<Location::Vec, float, tM, tN,
                                  BLayout::ColMajor, tM,    tN>;
  using tile_C_out_trows   = Tile<Location::Vec, float, tM, tN,
                                  BLayout::ColMajor, tM,    rmd_N>;
  using tile_C_out_tcols   = Tile<Location::Vec, float, tM, tN,
                                  BLayout::ColMajor, rmd_M, tN>;
  using tile_C_out_tcorner = Tile<Location::Vec, float, tM, tN,
                                  BLayout::ColMajor, rmd_M, rmd_N>;

  // —— 写回时的 bf16 临时 tile（RowMajor）——
  using tile_C_bf16         = Tile<Location::Vec, float, tM, tN,
                                   BLayout::RowMajor, tM,    tN>;
  using tile_C_bf16_trows   = Tile<Location::Vec, float, tM, tN,
                                   BLayout::RowMajor, tM,    rmd_N>;
  using tile_C_bf16_tcols   = Tile<Location::Vec, float, tM, tN,
                                   BLayout::RowMajor, rmd_M, tN>;
  using tile_C_bf16_tcorner = Tile<Location::Vec, float, tM, tN,
                                   BLayout::RowMajor, rmd_M, rmd_N>;

  // —— K 切分 ——
  constexpr int K_chunks = Kb / MAX_TILE_NUM;
  constexpr int K_rem    = Kb % MAX_TILE_NUM;

  static_assert(K_chunks > 0 || K_rem > 0 || rmd_K > 0, "empty K");

  for (int b = 0; b < Batch; b++) {

    // Vec 常驻 C
    tile_C_out         tC_main[Mb][Nb];
    tile_C_out_trows   tC_rcol[Mb];
    tile_C_out_tcols   tC_rrow[Nb];
    tile_C_out_tcorner tC_corner;

    // ============================================================
    // 段 1：等长 chunks (LEN = MAX_TILE_NUM)
    //   chunk[0]    : is_first = true
    //   chunk[1..*] : is_first = false
    // ============================================================
    if constexpr (K_chunks > 0) {

      // ---------- chunk[0]: is_first = true ----------
      {
        constexpr int LEN     = MAX_TILE_NUM;
        constexpr int k_base  = 0;

        // ----- 主 N 块 -----
        #pragma clang loop unroll(full)
        for (int col = 0; col < Nb; col++) {
          tile_shapeB tB[LEN];
          #pragma clang loop unroll(full)
          for (int k = 0; k < LEN; k++) {
            auto gB = gIterB(k_base + k, col);
            TCOPYIN(tB[k], gB);
          }

          #pragma clang loop unroll(full)
          for (int row = 0; row < Mb; row++) {
            tile_shapeACC tACC;
            #pragma clang loop unroll(full)
            for (int k = 0; k < LEN; k++) {
              tile_shapeA tA;
              auto gA = gIterA(row, k_base + k);
              TCOPYIN(tA, gA);
              if (k == 0) MATMUL (tACC, tA, tB[k]);
              else        MATMACC(tACC, tA, tB[k]);
            }
            TCVT(tC_main[row][col], tACC);   // is_first: 直接覆盖
          }

          if constexpr (rmd_M) {
            tile_shapeACC_tcols tACC;
            #pragma clang loop unroll(full)
            for (int k = 0; k < LEN; k++) {
              tile_shapeA_tcols tA;
              auto gA = gIterA(Mb, k_base + k);
              TCOPYIN(tA, gA);
              if (k == 0) MATMUL (tACC, tA, tB[k]);
              else        MATMACC(tACC, tA, tB[k]);
            }
            TCVT(tC_rrow[col], tACC);
          }
        } // col

        // ----- rmd_N 尾列 -----
        if constexpr (rmd_N) {
          tile_shapeB_trows tB[LEN];
          #pragma clang loop unroll(full)
          for (int k = 0; k < LEN; k++) {
            auto gB = gIterB(k_base + k, Nb);
            TCOPYIN(tB[k], gB);
          }

          #pragma clang loop unroll(full)
          for (int row = 0; row < Mb; row++) {
            tile_shapeACC_trows tACC;
            #pragma clang loop unroll(full)
            for (int k = 0; k < LEN; k++) {
              tile_shapeA tA;
              auto gA = gIterA(row, k_base + k);
              TCOPYIN(tA, gA);
              if (k == 0) MATMUL (tACC, tA, tB[k]);
              else        MATMACC(tACC, tA, tB[k]);
            }
            TCVT(tC_rcol[row], tACC);
          }

          if constexpr (rmd_M) {
            tile_shapeACC_tcorner tACC;
            #pragma clang loop unroll(full)
            for (int k = 0; k < LEN; k++) {
              tile_shapeA_tcols tA;
              auto gA = gIterA(Mb, k_base + k);
              TCOPYIN(tA, gA);
              if (k == 0) MATMUL (tACC, tA, tB[k]);
              else        MATMACC(tACC, tA, tB[k]);
            }
            TCVT(tC_corner, tACC);
          }
        }
      } // chunk[0]

      // ---------- chunk[1 .. K_chunks-1]: is_first = false ----------
      if constexpr (K_chunks > 1) {
        #pragma clang loop unroll(full)
        for (int kc = 1; kc < K_chunks; kc++) {
          constexpr int LEN = MAX_TILE_NUM;
          const int k_base  = kc * MAX_TILE_NUM;

          // 主 N 块
          #pragma clang loop unroll(full)
          for (int col = 0; col < Nb; col++) {
            tile_shapeB tB[LEN];
            #pragma clang loop unroll(full)
            for (int k = 0; k < LEN; k++) {
              auto gB = gIterB(k_base + k, col);
              TCOPYIN(tB[k], gB);
            }

            #pragma clang loop unroll(full)
            for (int row = 0; row < Mb; row++) {
              tile_shapeACC tACC;
              #pragma clang loop unroll(full)
              for (int k = 0; k < LEN; k++) {
                tile_shapeA tA;
                auto gA = gIterA(row, k_base + k);
                TCOPYIN(tA, gA);
                if (k == 0) MATMUL (tACC, tA, tB[k]);
                else        MATMACC(tACC, tA, tB[k]);
              }
              tile_C_out tACC_vec;
              TCVT(tACC_vec, tACC);
              TADD(tC_main[row][col], tC_main[row][col], tACC_vec);
            }

            if constexpr (rmd_M) {
              tile_shapeACC_tcols tACC;
              #pragma clang loop unroll(full)
              for (int k = 0; k < LEN; k++) {
                tile_shapeA_tcols tA;
                auto gA = gIterA(Mb, k_base + k);
                TCOPYIN(tA, gA);
                if (k == 0) MATMUL (tACC, tA, tB[k]);
                else        MATMACC(tACC, tA, tB[k]);
              }
              tile_C_out_tcols tACC_vec;
              TCVT(tACC_vec, tACC);
              TADD(tC_rrow[col], tC_rrow[col], tACC_vec);
            }
          } // col

          if constexpr (rmd_N) {
            tile_shapeB_trows tB[LEN];
            #pragma clang loop unroll(full)
            for (int k = 0; k < LEN; k++) {
              auto gB = gIterB(k_base + k, Nb);
              TCOPYIN(tB[k], gB);
            }

            #pragma clang loop unroll(full)
            for (int row = 0; row < Mb; row++) {
              tile_shapeACC_trows tACC;
              #pragma clang loop unroll(full)
              for (int k = 0; k < LEN; k++) {
                tile_shapeA tA;
                auto gA = gIterA(row, k_base + k);
                TCOPYIN(tA, gA);
                if (k == 0) MATMUL (tACC, tA, tB[k]);
                else        MATMACC(tACC, tA, tB[k]);
              }
              tile_C_out_trows tACC_vec;
              TCVT(tACC_vec, tACC);
              TADD(tC_rcol[row], tC_rcol[row], tACC_vec);
            }

            if constexpr (rmd_M) {
              tile_shapeACC_tcorner tACC;
              #pragma clang loop unroll(full)
              for (int k = 0; k < LEN; k++) {
                tile_shapeA_tcols tA;
                auto gA = gIterA(Mb, k_base + k);
                TCOPYIN(tA, gA);
                if (k == 0) MATMUL (tACC, tA, tB[k]);
                else        MATMACC(tACC, tA, tB[k]);
              }
              tile_C_out_tcorner tACC_vec;
              TCVT(tACC_vec, tACC);
              TADD(tC_corner, tC_corner, tACC_vec);
            }
          }
        } // kc
      } // K_chunks > 1
    } // K_chunks > 0

    // ============================================================
    // 段 2：K_rem (LEN = K_rem)
    //   is_first = (K_chunks == 0)
    // ============================================================
    if constexpr (K_rem > 0) {
      constexpr int LEN     = K_rem;
      constexpr int k_base  = K_chunks * MAX_TILE_NUM;
      constexpr bool is_first = (K_chunks == 0);

      #pragma clang loop unroll(full)
      for (int col = 0; col < Nb; col++) {
        tile_shapeB tB[LEN];
        #pragma clang loop unroll(full)
        for (int k = 0; k < LEN; k++) {
          auto gB = gIterB(k_base + k, col);
          TCOPYIN(tB[k], gB);
        }

        #pragma clang loop unroll(full)
        for (int row = 0; row < Mb; row++) {
          tile_shapeACC tACC;
          #pragma clang loop unroll(full)
          for (int k = 0; k < LEN; k++) {
            tile_shapeA tA;
            auto gA = gIterA(row, k_base + k);
            TCOPYIN(tA, gA);
            if (k == 0) MATMUL (tACC, tA, tB[k]);
            else        MATMACC(tACC, tA, tB[k]);
          }
          if constexpr (is_first) {
            TCVT(tC_main[row][col], tACC);
          } else {
            tile_C_out tACC_vec;
            TCVT(tACC_vec, tACC);
            TADD(tC_main[row][col], tC_main[row][col], tACC_vec);
          }
        }

        if constexpr (rmd_M) {
          tile_shapeACC_tcols tACC;
          #pragma clang loop unroll(full)
          for (int k = 0; k < LEN; k++) {
            tile_shapeA_tcols tA;
            auto gA = gIterA(Mb, k_base + k);
            TCOPYIN(tA, gA);
            if (k == 0) MATMUL (tACC, tA, tB[k]);
            else        MATMACC(tACC, tA, tB[k]);
          }
          if constexpr (is_first) {
            TCVT(tC_rrow[col], tACC);
          } else {
            tile_C_out_tcols tACC_vec;
            TCVT(tACC_vec, tACC);
            TADD(tC_rrow[col], tC_rrow[col], tACC_vec);
          }
        }
      }

      if constexpr (rmd_N) {
        tile_shapeB_trows tB[LEN];
        #pragma clang loop unroll(full)
        for (int k = 0; k < LEN; k++) {
          auto gB = gIterB(k_base + k, Nb);
          TCOPYIN(tB[k], gB);
        }

        #pragma clang loop unroll(full)
        for (int row = 0; row < Mb; row++) {
          tile_shapeACC_trows tACC;
          #pragma clang loop unroll(full)
          for (int k = 0; k < LEN; k++) {
            tile_shapeA tA;
            auto gA = gIterA(row, k_base + k);
            TCOPYIN(tA, gA);
            if (k == 0) MATMUL (tACC, tA, tB[k]);
            else        MATMACC(tACC, tA, tB[k]);
          }
          if constexpr (is_first) {
            TCVT(tC_rcol[row], tACC);
          } else {
            tile_C_out_trows tACC_vec;
            TCVT(tACC_vec, tACC);
            TADD(tC_rcol[row], tC_rcol[row], tACC_vec);
          }
        }

        if constexpr (rmd_M) {
          tile_shapeACC_tcorner tACC;
          #pragma clang loop unroll(full)
          for (int k = 0; k < LEN; k++) {
            tile_shapeA_tcols tA;
            auto gA = gIterA(Mb, k_base + k);
            TCOPYIN(tA, gA);
            if (k == 0) MATMUL (tACC, tA, tB[k]);
            else        MATMACC(tACC, tA, tB[k]);
          }
          if constexpr (is_first) {
            TCVT(tC_corner, tACC);
          } else {
            tile_C_out_tcorner tACC_vec;
            TCVT(tACC_vec, tACC);
            TADD(tC_corner, tC_corner, tACC_vec);
          }
        }
      }
    } // K_rem

    // ============================================================
    // 段 3：rmd_K
    //   is_first = (K_chunks == 0) && (K_rem == 0)
    // ============================================================
    if constexpr (rmd_K) {
      constexpr bool is_first = (K_chunks == 0) && (K_rem == 0);

      #pragma clang loop unroll(full)
      for (int col = 0; col < Nb; col++) {
        tile_shapeB_tcols tB_rmdK;
        auto gB = gIterB(Kb, col);
        TCOPYIN(tB_rmdK, gB);

        #pragma clang loop unroll(full)
        for (int row = 0; row < Mb; row++) {
          tile_shapeACC tACC;
          tile_shapeA_trows tA;
          auto gA = gIterA(row, Kb);
          TCOPYIN(tA, gA);
          MATMUL(tACC, tA, tB_rmdK);

          if constexpr (is_first) {
            TCVT(tC_main[row][col], tACC);
          } else {
            tile_C_out tACC_vec;
            TCVT(tACC_vec, tACC);
            TADD(tC_main[row][col], tC_main[row][col], tACC_vec);
          }
        }

        if constexpr (rmd_M) {
          tile_shapeACC_tcols tACC;
          tile_shapeA_tcorner tA;
          auto gA = gIterA(Mb, Kb);
          TCOPYIN(tA, gA);
          MATMUL(tACC, tA, tB_rmdK);

          if constexpr (is_first) {
            TCVT(tC_rrow[col], tACC);
          } else {
            tile_C_out_tcols tACC_vec;
            TCVT(tACC_vec, tACC);
            TADD(tC_rrow[col], tC_rrow[col], tACC_vec);
          }
        }
      }

      if constexpr (rmd_N) {
        tile_shapeB_tcorner tB_rmdK;
        auto gB = gIterB(Kb, Nb);
        TCOPYIN(tB_rmdK, gB);

        #pragma clang loop unroll(full)
        for (int row = 0; row < Mb; row++) {
          tile_shapeACC_trows tACC;
          tile_shapeA_trows tA;
          auto gA = gIterA(row, Kb);
          TCOPYIN(tA, gA);
          MATMUL(tACC, tA, tB_rmdK);

          if constexpr (is_first) {
            TCVT(tC_rcol[row], tACC);
          } else {
            tile_C_out_trows tACC_vec;
            TCVT(tACC_vec, tACC);
            TADD(tC_rcol[row], tC_rcol[row], tACC_vec);
          }
        }

        if constexpr (rmd_M) {
          tile_shapeACC_tcorner tACC;
          tile_shapeA_tcorner tA;
          auto gA = gIterA(Mb, Kb);
          TCOPYIN(tA, gA);
          MATMUL(tACC, tA, tB_rmdK);

          if constexpr (is_first) {
            TCVT(tC_corner, tACC);
          } else {
            tile_C_out_tcorner tACC_vec;
            TCVT(tACC_vec, tACC);
            TADD(tC_corner, tC_corner, tACC_vec);
          }
        }
      }
    } // rmd_K

    // ============================================================
    // 全部 K 累加完成后，统一写回 global
    // ============================================================
    #pragma clang loop unroll(full)
    for (int m = 0; m < Mb; m++) {
      #pragma clang loop unroll(full)
      for (int n = 0; n < Nb; n++) {
        tile_C_bf16 tC_b;
        // TMOV_NZ2DN(tC_b, tC_main[m][n]);
        auto gC = gIterC(m, n);
        // TCOPYOUT(gC, tC_b);
        TCOPYOUT(gC, tC_main[m][n]);
      }
      if constexpr (rmd_N) {
        tile_C_bf16_trows tC_b;
        // TMOV_NZ2DN(tC_b, tC_rcol[m]);
        auto gC = gIterC(m, Nb);
        TCOPYOUT(gC, tC_rcol[m]);
        // TCOPYOUT(gC, tC_b);
      }
    }
    if constexpr (rmd_M) {
      #pragma clang loop unroll(full)
      for (int n = 0; n < Nb; n++) {
        tile_C_bf16_tcols tC_b;
        // TMOV_NZ2DN(tC_b, tC_rrow[n]);
        auto gC = gIterC(Mb, n);
        TCOPYOUT(gC,  tC_rrow[n]);
        // TCOPYOUT(gC, tC_b);
      }
      if constexpr (rmd_N) {
        tile_C_bf16_tcorner tC_b;
        // TMOV_NZ2DN(tC_b, tC_corner);
        auto gC = gIterC(Mb, Nb);
        TCOPYOUT(gC, tC_corner);
        // TCOPYOUT(gC, tC_b);
      }
    }

  } // Batch
}


template<typename dtype, const int gM, const int gN, const int gK, const int tM, const int tN, const int tK>
void matmul_mask_reuseB(float *dst, dtype *src0, dtype *src1){
  const int MAX_TILE_NUM = 24;
  using gm_shapeA = global_tensor<dtype, RowMajor<gM, gK>>;
  using gm_shapeB = global_tensor<dtype, RowMajor<gK, gN>>;
  using gm_shapeC = global_tensor<float, RowMajor<gM, gN>>;
  using tile_shapeA = TileLeft<dtype, tM, tK>;
  using tile_shapeB = TileRight<dtype, tK, tN>;
  using tile_shapeACC = TileAcc<float, tM, tN>;

  using itA = global_iterator<gm_shapeA, tile_shapeA>;
  using itB = global_iterator<gm_shapeB, tile_shapeB>;
  using itC = global_iterator<gm_shapeC, tile_shapeACC>;

  itA gIterA(src0);
  itB gIterB(src1);
  itC gIterC(dst);

  const int Mb = gM / tM;
  const int Nb = gN / tN;
  const int Kb = gK / tK;

  const int rmd_M = gM % tM;
  const int rmd_N = gN % tN;
  const int rmd_K = gK % tK;

  using tile_shapeA_trows = TileLeft<dtype, tM, tK,  tM, rmd_K>;
  using tile_shapeA_tcols = TileLeft<dtype, tM, tK, rmd_M, tK>;
  using tile_shapeA_tcorner = TileLeft<dtype, tM, tK, rmd_M, rmd_K>;

  using tile_shapeB_trows = TileRight<dtype, tK, tN, tK, rmd_N>;
  using tile_shapeB_tcols = TileRight<dtype, tK, tN, rmd_K, tN>;
  using tile_shapeB_tcorner = TileRight<dtype, tK, tN, rmd_K, rmd_N>;

  using tile_shapeC_trows = TileAcc<float, tM, tN, tM, rmd_N>;
  using tile_shapeC_tcols = TileAcc<float, tM, tN, rmd_M, tN>;
  using tile_shapeC_tcorner = TileAcc<float, tM, tN, rmd_M, rmd_N>;

  // constexpr ResB R = find_reuseB(Nb, Kb, MAX_TILE_NUM);
  constexpr ResB R = {1, (Kb > MAX_TILE_NUM) ? MAX_TILE_NUM : Kb, R.n * R.k};
  // static_assert(R.n == 1);
  const int dN = R.n == 0? 0 : Nb / R.n;
  const int rN = R.n == 0? 0 : Nb % R.n;

  static_assert(R.val <= MAX_TILE_NUM, "R.val is biggger than MAX_TILE_NUM");
  //printf("R.m is %d, R.k is %d\n", R.m, R.k);

  for (int b=0;b<Batch;b++){

    #pragma clang loop unroll(full)
    for(int i=0;i<dN;i++){
      tile_shapeB tB[R.k][R.n];

      #pragma clang loop unroll(full)
      for(int ii=0;ii<R.n;ii++){

        //copy in all reuse B sub tiles in K dim
        // #pragma clang loop unroll(full)
        // for(int k=0;k<R.k;k++){
        //   auto gB = gIterB(k, ii+i*R.n);
        //   TCOPYIN(tB[k][ii], gB);
        // }

        #pragma clang loop unroll(full)
        for(int j=0;j<Mb;j++){
          tile_shapeACC tACC;

          #pragma clang loop unroll(full)
          for(int k=0;k<R.k;k++){
            auto gA = gIterA(j,k);
            tile_shapeA tA;
            TCOPYIN(tA, gA);
            if(j==0){
              // eliminate head cost
              auto gB = gIterB(k, ii+i*R.n);
              TCOPYIN(tB[k][ii], gB);
            }
            if(k==0){
              MATMUL(tACC, tA, tB[k][ii]);
            }else{
              MATMACC(tACC, tA, tB[k][ii]);
            }
          }

          if constexpr(R.k < Kb){
            #pragma clang loop unroll(full)
            for(int k=R.k;k<Kb;k++){
              tile_shapeB tB;
              tile_shapeA tA;
              auto gB = gIterB(k,i*R.n+ii);
              auto gA = gIterA(j,k);

              TCOPYIN(tA,gA);
              TCOPYIN(tB,gB);
              MATMACC(tACC, tA, tB);
            }
          }

          // [n, m, rmd_K]
          if constexpr (rmd_K) {
            auto gB = gIterB(Kb, i*R.n+ii);
            auto gA = gIterA(j, Kb);
            tile_shapeB_tcols tB;
            tile_shapeA_trows tA;


            TCOPYIN(tA, gA);
            TCOPYIN(tB, gB);
            if constexpr(Kb>0){
            MATMACC(tACC, tA, tB);
            } else {
              MATMUL(tACC, tA, tB);
            }
          }

          auto gC = gIterC(j, i*R.n+ii);
          TCOPYOUT_ACC(gC, tACC);
        }

        // [n, rmd_M, k]
        if constexpr (rmd_M) {
          tile_shapeC_tcols tACC;

          #pragma clang loop unroll(full)
          for(int k=0;k<R.k;k++){
            auto gA = gIterA(Mb,k);
            tile_shapeA_tcols tA;
            TCOPYIN(tA, gA);
            if(k==0){
              MATMUL(tACC, tA, tB[k][ii]);
            }else{
              MATMACC(tACC, tA, tB[k][ii]);
            }
          }

          if constexpr(R.k < Kb){
            #pragma clang loop unroll(full)
            for(int k=R.k;k<Kb;k++){
              tile_shapeB tB;
              tile_shapeA_tcols tA;
              auto gB = gIterB(k, i*R.n+ii);
              auto gA = gIterA(Mb,k);

              TCOPYIN(tA,gA);
              TCOPYIN(tB,gB);
              MATMACC(tACC, tA, tB);
            }
          }

          // [n, rmd_M, rmd_K]
          if constexpr (rmd_K) {
            auto gB = gIterB(Kb, i*R.n+ii);
            auto gA = gIterA(Mb, Kb);

            tile_shapeB_tcols tB;
            tile_shapeA_tcorner tA;

            TCOPYIN(tA, gA);
            TCOPYIN(tB, gB);
            if constexpr(Kb>0){
            MATMACC(tACC, tA, tB);
            } else {
              MATMUL(tACC, tA, tB);
            }
          }

          auto gC = gIterC(Mb, i*R.n+ii);
          TCOPYOUT_ACC(gC, tACC);       
        }

      }
    }

    // [rN, m, k]
    if constexpr(rN>0){
      tile_shapeB tB[R.k][rN];
      
      #pragma clang loop unroll(full)
      for(int i=0;i<rN;i++){

        //copy in remaining N dimension B tile in K dim
        #pragma clang loop unroll(full)
        for(int k=0;k<R.k;k++){
          auto gB = gIterB(k, i+dN*R.n);
          TCOPYIN(tB[k][i], gB);
        }

        #pragma clang loop unroll(full)
        for(int j=0;j<Mb;j++){
          tile_shapeACC tACC;

          #pragma clang loop unroll(full)
          for(int k=0;k<R.k;k++){
            auto gA = gIterA(j,k);
            tile_shapeA tA;
            TCOPYIN(tA, gA);
            if(k==0){
              MATMUL(tACC, tA, tB[k][i]);
            }else{
              MATMACC(tACC, tA, tB[k][i]);
            }
          }

          if constexpr(R.k < Kb){
            #pragma clang loop unroll(full)
            for(int k=R.k;k<Kb;k++){
              tile_shapeB tB;
              tile_shapeA tA;
              auto gB = gIterB(k, i+dN*R.n);
              auto gA = gIterA(j, k);

              TCOPYIN(tA,gA);
              TCOPYIN(tB,gB);
              if constexpr (R.k == 0) {
                MATMUL(tACC, tA, tB);
              } else
                MATMACC(tACC, tA, tB);
            }
          }

          // [rN, m, rmd_K]
          if constexpr (rmd_K) {
            auto gB = gIterB(Kb, i+dN*R.n);
            auto gA = gIterA(j, Kb);

            tile_shapeB_tcols tB;
            tile_shapeA_trows tA;

            TCOPYIN(tA, gA);
            TCOPYIN(tB, gB);
            if constexpr(Kb>0){
            MATMACC(tACC, tA, tB);
            } else {
              MATMUL(tACC, tA, tB);
            }
          }
          auto gC = gIterC(j, i+dN*R.n);
          TCOPYOUT_ACC(gC, tACC);
        }

        // [rN, rmd_M, k]
        if constexpr (rmd_M) {
          tile_shapeC_tcols tACC;

          #pragma clang loop unroll(full)
          for(int k=0;k<R.k;k++){
            auto gA = gIterA(Mb,k);
            tile_shapeA_tcols tA;
            TCOPYIN(tA, gA);
            if(k==0){
              MATMUL(tACC, tA, tB[k][i]);
            }else{
              MATMACC(tACC, tA, tB[k][i]);
            }
          }

          if constexpr(R.k < Kb){
            #pragma clang loop unroll(full)
            for(int k=R.k;k<Kb;k++){
              tile_shapeB tB;
              tile_shapeA_tcols tA;
              auto gB = gIterB(k, i+dN*R.n);
              auto gA = gIterA(Mb,k);
              TCOPYIN(tA,gA);
              TCOPYIN(tB,gB);
              if constexpr (R.k == 0)
                MATMUL(tACC, tA, tB);
              else
                MATMACC(tACC, tA, tB);
            }
          }

          // [rN, rmd_M, rmd_K]
          if constexpr (rmd_K) {
            auto gB = gIterB(Kb, i+dN*R.n);
            auto gA = gIterA(Mb, Kb);

            tile_shapeB_tcols tB;
            tile_shapeA_tcorner tA;

            TCOPYIN(tA, gA);
            TCOPYIN(tB, gB);
            if constexpr(Kb>0){
            MATMACC(tACC, tA, tB);
            } else {
              MATMUL(tACC, tA, tB);
            }
          }
          auto gC = gIterC(Mb, i+dN*R.n);
          TCOPYOUT_ACC(gC, tACC);        
        }
      }
    }

    // [rmd_N, m, k]
    if constexpr (rmd_N) {
      tile_shapeB_trows tB[R.k];
      
      #pragma clang loop unroll(full)
      for(int k=0;k<R.k;k++){
        auto gB = gIterB(k, Nb);
        TCOPYIN(tB[k], gB);
      }

      #pragma clang loop unroll(full)
      for(int j=0;j<Mb;j++){
        tile_shapeC_trows tACC;

        #pragma clang loop unroll(full)
        for(int k=0;k<R.k;k++){
          auto gA = gIterA(j,k);
          tile_shapeA tA;
          TCOPYIN(tA, gA);
          if(k==0){
            MATMUL(tACC, tA, tB[k]);
          }else{
            MATMACC(tACC, tA, tB[k]);
          }
        }

        if constexpr(R.k < Kb){
          #pragma clang loop unroll(full)
          for(int k=R.k;k<Kb;k++){
            tile_shapeB_trows tB;
            tile_shapeA tA;
            auto gB = gIterB(k,Nb);
            auto gA = gIterA(j,k);
            TCOPYIN(tA,gA);
            TCOPYIN(tB,gB);
            if constexpr (R.k == 0)
              MATMUL(tACC, tA, tB);
            else
              MATMACC(tACC, tA, tB);
          }
        }

        // [rmd_N, m, rmd_K]
        if constexpr (rmd_K) {
          auto gB = gIterB(Kb, Nb);
          auto gA = gIterA(j, Kb);

          tile_shapeB_tcorner tB;
          tile_shapeA_trows tA;

          TCOPYIN(tA, gA);
          TCOPYIN(tB, gB);
          if constexpr(Kb>0){
          MATMACC(tACC, tA, tB);
          } else {
            MATMUL(tACC, tA, tB);
          }
        }
        auto gC = gIterC(j, Nb);
        TCOPYOUT_ACC(gC, tACC);
      }

      // [rmd_N, rmd_M, k]
      if constexpr (rmd_M) {
        tile_shapeC_tcorner tACC;

        #pragma clang loop unroll(full)
        for(int k=0;k<R.k;k++){
          auto gA = gIterA(Mb,k);
          tile_shapeA_tcols tA;
          TCOPYIN(tA, gA);
          if(k==0){
            MATMUL(tACC, tA, tB[k]);
          }else{
            MATMACC(tACC, tA, tB[k]);
          }
        }

        if constexpr(R.k < Kb){
          #pragma clang loop unroll(full)
          for(int k=R.k;k<Kb;k++){
            tile_shapeB_trows tB;
            tile_shapeA_tcols tA;
            auto gB = gIterB(k,Nb);
            auto gA = gIterA(Mb,k);
            TCOPYIN(tA,gA);
            TCOPYIN(tB,gB);
            if constexpr (R.k == 0)
              MATMUL(tACC, tA, tB);
            else
              MATMACC(tACC, tA, tB);
          }
        }

        // [rmd_N, rmd_M, rmd_K]
        if constexpr (rmd_K) {
          auto gB = gIterB(Kb, Nb);
          auto gA = gIterA(Mb, Kb);

          tile_shapeB_tcorner tB;
          tile_shapeA_tcorner tA;

          TCOPYIN(tA, gA);
          TCOPYIN(tB, gB);
          if constexpr(Kb>0){
          MATMACC(tACC, tA, tB);
          } else {
            MATMUL(tACC, tA, tB);
          }
        }
        auto gC = gIterC(Mb,Nb);
        TCOPYOUT_ACC(gC, tACC);        
      }
    }

  }// Batch
}

struct ResAB {
    int m;
    int n;
    int k;
    int val;
};

constexpr ResAB find_reuseAB(int Mb, int Nb, int Kb, int MAX_TILE_NUM) {
    int best_m = 0, best_k = 0, best_n = 0, best_val = -1;

    #pragma clang loop unroll(full)
    for (int n = 1; n <= Nb; ++n) {
      #pragma clang loop unroll(full)
      for (int m = 1; m <= Mb; ++m) {
        #pragma clang loop unroll(full)
        for (int k = 1; k <= Kb; ++k) {
            int v = m * k + n * k;
            if (v <= MAX_TILE_NUM && v > best_val) {
                best_val = v;
                best_k = k;
                best_m = m;
                best_n = n;
            }
        }
      }
    }
    return {best_m, best_n, best_k, best_val};
}


template<typename dtype, const int gM, const int gN, const int gK, const int tM, const int tN, const int tK>
void matmul_mask_reuseAB(float *dst, dtype *src0, dtype *src1){
  const int MAX_TILE_NUM = 24;
  using gm_shapeA = global_tensor<dtype, RowMajor<gM, gK>>;
  using gm_shapeB = global_tensor<dtype, RowMajor<gK, gN>>;
  using gm_shapeC = global_tensor<float, RowMajor<gM, gN>>;
  using tile_shapeA = TileLeft<dtype, tM, tK>;
  using tile_shapeB = TileRight<dtype, tK, tN>;
  using tile_shapeACC = TileAcc<float, tM, tN>;

  using itA = global_iterator<gm_shapeA, tile_shapeA>;
  using itB = global_iterator<gm_shapeB, tile_shapeB>;
  using itC = global_iterator<gm_shapeC, tile_shapeACC>;

  itA gIterA(src0);
  itB gIterB(src1);
  itC gIterC(dst);

  const int Mb = (gM + tM - 1) / tM;
  const int Nb = (gN + tN - 1) / tN;
  const int Kb = (gK + tK - 1) / tK;

  constexpr ResAB R = find_reuseAB(Mb, Nb, Kb, MAX_TILE_NUM);

  const int dM = Mb / R.m;
  const int rM = Mb % R.m;

  const int dN = Nb / R.n;
  const int rN = Nb % R.n;

  const int dK = Kb / R.k;
  const int rK = Kb % R.k;

  static_assert(R.val <= MAX_TILE_NUM, "R.val is biggger than MAX_TILE_NUM");
  // printf("R.m is %d, R.n is %d, R.k is %d\n", R.m, R.n, R.k);

  for (int b=0;b<Batch;b++){

    #pragma clang loop unroll(full)
    for(int i=0;i<dM;i++){
      tile_shapeA tA[R.m][R.k];
      //copy in all reuse A sub tiles
      #pragma clang loop unroll(full)
      for(int m=0;m<R.m;m++){
        #pragma clang loop unroll(full)
        for(int k=0;k<R.k;k++){
          auto gA = gIterA(m+i*R.m,k);
          TCOPYIN(tA[m][k], gA);
        }
      }
      
      #pragma clang loop unroll(full)
      for(int j=0;j<dN;j++){

        tile_shapeB tB[R.k][R.n];
        #pragma clang loop unroll(full)
        for(int n=0;n<R.n;n++){
          #pragma clang loop unroll(full)
          for(int k=0;k<R.k;k++){
            auto gB = gIterB(k, n+j*R.n);
            TCOPYIN(tB[k][n], gB);
          }
        }

        #pragma clang loop unroll(full)
        for(int ii=0;ii<R.m;ii++){
          #pragma clang loop unroll(full)
          for(int jj=0;jj<R.n;jj++){
            tile_shapeACC tACC;
            #pragma clang loop unroll(full)
            for(int k=0;k<R.k;k++){
              if(k==0){
                MATMUL(tACC, tA[ii][k], tB[k][jj]);
              }else{
                MATMACC(tACC, tA[ii][k], tB[k][jj]);
              }
            }

            if constexpr(R.k < Kb){
              for(int k=R.k;k<Kb;k++){
                tile_shapeA tA;
                tile_shapeB tB;
                auto gA = gIterA(i*R.m+ii,k);
                auto gB = gIterB(k,j*R.n+jj);
                TCOPYIN(tA,gA);
                TCOPYIN(tB,gB);
                MATMACC(tACC, tA, tB);
              }
            }
            auto gC = gIterC(i*R.m+ii,j*R.n+jj);
            TCOPYOUT_ACC(gC, tACC);    
          }
        }
      }

      if constexpr(rN){
        tile_shapeB tB[R.k][rN];
        //copy in remaining N dimension B tile
        #pragma clang loop unroll(full)
        for(int n=0;n<rN;n++){
          #pragma clang loop unroll(full)
          for(int k=0;k<R.k;k++){
            auto gB = gIterB(k, n+dN*R.n);
            TCOPYIN(tB[k][n], gB);
          }
        }

        #pragma clang loop unroll(full)
        for(int ii=0;ii<R.m;ii++){
          #pragma clang loop unroll(full)
          for(int jj=0;jj<rN;jj++){
            tile_shapeACC tACC;

            #pragma clang loop unroll(full)
            for(int k=0;k<R.k;k++){
              if(k==0){
                MATMUL(tACC, tA[ii][k], tB[k][jj]);
              }else{
                MATMACC(tACC, tA[ii][k], tB[k][jj]);
              }
            }

            if constexpr(R.k < Kb){
              for(int k=R.k;k<Kb;k++){
                tile_shapeA tA;
                tile_shapeB tB;
                auto gA = gIterA(i*R.m+ii,k);
                auto gB = gIterB(k,dN*R.n+jj);
                TCOPYIN(tA,gA);
                TCOPYIN(tB,gB);
                MATMACC(tACC, tA, tB);
              }
            }
            auto gC = gIterC(i*R.m+ii,dN*R.n+jj);
            TCOPYOUT_ACC(gC, tACC);    
          }
        }
      }
    }
    
    if constexpr(rM){
      tile_shapeA tA[rM][R.k];
      //copy in remaining M dimension A tile
      #pragma clang loop unroll(full)
      for(int m=0;m<rM;m++){
        #pragma clang loop unroll(full)
        for(int k=0;k<R.k;k++){
          auto gA = gIterA(m+dM*R.m,k);
          TCOPYIN(tA[m][k], gA);
        }
      }
      
      #pragma clang loop unroll(full)
      for(int j=0;j<dN;j++){

        tile_shapeB tB[R.k][R.n];
        #pragma clang loop unroll(full)
        for(int n=0;n<R.n;n++){
          #pragma clang loop unroll(full)
          for(int k=0;k<R.k;k++){
            auto gB = gIterB(k, n+j*R.n);
            TCOPYIN(tB[k][n], gB);
          }
        }

        #pragma clang loop unroll(full)
        for(int ii=0;ii<rM;ii++){
          #pragma clang loop unroll(full)
          for(int jj=0;jj<R.n;jj++){
            tile_shapeACC tACC;
            #pragma clang loop unroll(full)
            for(int k=0;k<R.k;k++){
              if(k==0){
                MATMUL(tACC, tA[ii][k], tB[k][jj]);
              }else{
                MATMACC(tACC, tA[ii][k], tB[k][jj]);
              }
            }

            if constexpr(R.k < Kb){
              for(int k=R.k;k<Kb;k++){
                tile_shapeA tA;
                tile_shapeB tB;
                auto gA = gIterA(dM*R.m+ii,k);
                auto gB = gIterB(k,j*R.n+jj);
                TCOPYIN(tA,gA);
                TCOPYIN(tB,gB);
                MATMACC(tACC, tA, tB);
              }
            }
            auto gC = gIterC(dM*R.m+ii,j*R.n+jj);
            TCOPYOUT_ACC(gC, tACC);    
          }
        }
      }

      if constexpr(rN){
        tile_shapeB tB[R.k][rN];
        //copy in remaining N dimension B tile
        #pragma clang loop unroll(full)
        for(int n=0;n<rN;n++){
          #pragma clang loop unroll(full)
          for(int k=0;k<R.k;k++){
            auto gB = gIterB(k, n+dN*R.n);
            TCOPYIN(tB[k][n], gB);
          }
        }

        #pragma clang loop unroll(full)
        for(int ii=0;ii<rM;ii++){
          #pragma clang loop unroll(full)
          for(int jj=0;jj<rN;jj++){
            tile_shapeACC tACC;

            #pragma clang loop unroll(full)
            for(int k=0;k<R.k;k++){
              if(k==0){
                MATMUL(tACC, tA[ii][k], tB[k][jj]);
              }else{
                MATMACC(tACC, tA[ii][k], tB[k][jj]);
              }
            }

            if constexpr(R.k < Kb){
              for(int k=R.k;k<Kb;k++){
                tile_shapeA tA;
                tile_shapeB tB;
                auto gA = gIterA(dM*R.m+ii,k);
                auto gB = gIterB(k,dN*R.n+jj);
                TCOPYIN(tA,gA);
                TCOPYIN(tB,gB);
                MATMACC(tACC, tA, tB);
              }
            }
            auto gC = gIterC(dM*R.m+ii,dN*R.n+jj);
            TCOPYOUT_ACC(gC, tACC);    
          }
        }
      }
    }

  } // Batch
}


template<typename dtype, const int gM, const int gN, const int gK, const int tM, const int tN, const int tK>
void matmul_mask_multi4_B(float *dst, dtype *src0, dtype *src1){
    using gm_shapeA = global_tensor<dtype, RowMajor<gM, gK>>;
    using gm_shapeB = global_tensor<dtype, RowMajor<gK, gN>>;
    using gm_shapeC = global_tensor<float, RowMajor<gM, gN>>;
    using tile_shapeA = TileLeft<dtype, tM, tK>;
    using tile_shapeB = TileRight<dtype, tK, tN>;
    using tile_shapeACC = TileAcc<float, tM, tN>;

    using itA = global_iterator<gm_shapeA, tile_shapeA>;
    using itB = global_iterator<gm_shapeB, tile_shapeB>;
    using itC = global_iterator<gm_shapeC, tile_shapeACC>;

    itA gIterA(src0);
    itB gIterB(src1);
    itC gIterC(dst);

    const int Mb = gM / tM;
    const int Nb = gN / tN;
    const int Kb = gK / tK;
    for (int b=0;b<Batch;b++){
      #pragma clang loop unroll(full)
      for(int i=0;i<Mb;i+=1){
        #pragma clang loop unroll(full)
        for(int j=0;j<Nb;j+=4){
          tile_shapeACC tACC;
          tile_shapeB tB[Kb][4];

          for(int k=0;k<Kb;k++){
            auto gB = gIterB(k,j);
            TLOAD4_ND2ZN(tB[k][3], tB[k][2], tB[k][1], tB[k][0], gB);
          }

          #pragma clang loop unroll(full)
          for(int jj=0;jj<4;jj+=1){
            #pragma clang loop unroll(full)
            for(int k=0;k<Kb;k++){
              tile_shapeA tA;
              auto gA = gIterA(i,k);
              TCOPYIN(tA, gA);
              if(k==0){
                MATMUL(tACC, tA, tB[k][jj]);
              }else{
                MATMACC(tACC, tA, tB[k][jj]);
              }
            }
            auto gC = gIterC(i,j+jj);
            TCOPYOUT_ACC(gC, tACC);
          }
        }
      }
    }
}

template<typename dtype, const int gM, const int gN, const int gK, const int tM, const int tN, const int tK>
void matmul_mask_multi4_AB(float *dst, dtype *src0, dtype *src1){
    using gm_shapeA = global_tensor<dtype, RowMajor<gM, gK>>;
    using gm_shapeB = global_tensor<dtype, RowMajor<gK, gN>>;
    using gm_shapeC = global_tensor<float, RowMajor<gM, gN>>;
    using tile_shapeA = TileLeft<dtype, tM, tK>;
    using tile_shapeB = TileRight<dtype, tK, tN>;
    using tile_shapeACC = TileAcc<float, tM, tN>;

    using itA = global_iterator<gm_shapeA, tile_shapeA>;
    using itB = global_iterator<gm_shapeB, tile_shapeB>;
    using itC = global_iterator<gm_shapeC, tile_shapeACC>;

    itA gIterA(src0);
    itB gIterB(src1);
    itC gIterC(dst);

    const int Mb = gM / tM;
    const int Nb = gN / tN;
    const int Kb = gK / tK;

    #pragma clang loop unroll(full)
    for(int i=0;i<Mb;i+=1){
      #pragma clang loop unroll(full)
      for(int j=0;j<Nb;j+=4){
        tile_shapeACC tACC;
        tile_shapeA tA[Kb];
        tile_shapeB tB[Kb][4];

        #pragma clang loop unroll(full)
        for(int k=0;k<Kb;k+=4){
          auto gA = gIterA(i,k);
          TLOAD4_ND2NZ(tA[k+3], tA[k+2], tA[k+1], tA[k], gA);

          #pragma clang loop unroll(full)
          for(int kk=0;kk<4;kk++){
          auto gB = gIterB(k+kk,j);
          TLOAD4_ND2ZN(tB[k+kk][3], tB[k+kk][2], tB[k+kk][1], tB[k+kk][0], gB);
          }
        }

        #pragma clang loop unroll(full)
        for(int jj=0;jj<4;jj+=1){
          #pragma clang loop unroll(full)
          for(int k=0;k<Kb;k+=4){
            if(k==0){
              MATMUL(tACC, tA[k], tB[k][jj]);
            }else{
              MATMACC(tACC, tA[k], tB[k][jj]);
            }
            MATMACC(tACC, tA[k+1], tB[k+1][jj]);
            MATMACC(tACC, tA[k+2], tB[k+2][jj]);
            MATMACC(tACC, tA[k+3], tB[k+3][jj]);
          }
          auto gC = gIterC(i,j);
          TCOPYOUT_ACC(gC, tACC);
        }
      }
    }
}

template<typename dtype, const int tM, const int tN, const int tK>
__attribute__((noinline)) void matmul_dynamic_new(float* dst, dtype* src0, dtype* src1, int gM, int gN, int gK){
    using gm_shapeA = global_tensor<dtype, RowMajor<-1, -1>>;
    using gm_shapeB = global_tensor<dtype, RowMajor<-1, -1>>;
    using gm_shapeC = global_tensor<float, RowMajor<-1, -1>>;
    using tile_shapeA = TileLeft<dtype, tM, tK, -1, -1>;
    using tile_shapeB = TileRight<dtype, tK, tN, -1, -1>;
    using tile_shapeACC = TileAcc<float, tM, tN, -1, -1>;

    if (gM <= 0 || gN <= 0 || gK <= 0)
      return;

    for (int b=0;b<Batch;b++){
      for(int i=0;i<gM;i+=tM){
          for(int j=0;j<gN;j+=tN){
              size_t offset_C = i * gN + j;
              gm_shapeC gC(dst + offset_C, gM, gN);
              int dyn_m = std::min(tM, gM - i);
              int dyn_n = std::min(tN, gN - j);

              tile_shapeACC tACC(dyn_m, dyn_n);
              for(int k=0;k<gK;k+=tK){
                  size_t offset_A = i * gK + k;
                  size_t offset_B = k * gN + j;
                  gm_shapeA gA(src0 + offset_A, gM, gK);
                  gm_shapeB gB(src1 + offset_B, gK, gN);
                  int dyn_k = gK - k > tK ? tK : gK - k;
                  tile_shapeA tA(dyn_m, dyn_k);
                  tile_shapeB tB(dyn_k, dyn_n);
                  TCOPYIN(tA, gA);
                  TCOPYIN(tB, gB);
                  if(k==0){
                    MATMUL(tACC, tA, tB);
                  }else{
                    MATMACC(tACC, tA, tB);
                  }
              }
              TCOPYOUT_ACC_DYNAMIC(gC, tACC, tACC.GetValidRow(), tACC.GetValidCol());
          }
      }
    }
}

template<typename dtype, const int tM, const int tN, const int tK>
__attribute__((noinline)) void matmul_dynamic(float* dst, dtype* src0, dtype* src1, int gM, int gN, int gK){
    using gm_shapeA = global_tensor<dtype, RowMajor<-1, -1>>;
    using gm_shapeB = global_tensor<dtype, RowMajor<-1, -1>>;
    using gm_shapeC = global_tensor<float, RowMajor<-1, -1>>;
    using tile_shapeA = TileLeft<dtype, tM, tK, -1, -1>;
    using tile_shapeB = TileRight<dtype, tK, tN, -1, -1>;
    using tile_shapeACC = TileAcc<float, tM, tN, -1, -1>;

    int Mb = (gM + tM - 1) / tM;
    int Nb = (gN + tN - 1) / tN;
    int Kb = (gK + tK - 1) / tK;

    int rem_m = gM % tM;
    int rem_n = gN % tN;
    int rem_k = gK % tK;

    for (int b=0;b<Batch;b++){
      for(int i=0;i<Mb;i++){
          for(int j=0;j<Nb;j++){
              size_t offset_C = i * gN * tile_shapeACC::Rows + j * tile_shapeACC::Cols;
              gm_shapeC gC(dst + offset_C, gM, gN);
              int dyn_m = (i+1) * tM > gM ? rem_m:tM;
              int dyn_n = (j+1) * tN > gN ? rem_n:tN;

              tile_shapeACC tACC(dyn_m, dyn_n);
              for(int k=0;k<Kb;k++){
                  size_t offset_A = i * gK * tile_shapeA::Rows + k * tile_shapeA::Cols;
                  size_t offset_B = k * gN * tile_shapeB::Rows + j * tile_shapeB::Cols;
                  gm_shapeA gA(src0 + offset_A, gM, gK);
                  gm_shapeB gB(src1 + offset_B, gK, gN);
                  int dyn_k = (k+1) * tK > gK ? rem_k:tK;
                  tile_shapeA tA(dyn_m, dyn_k);
                  tile_shapeB tB(dyn_k, dyn_n);
                  TCOPYIN(tA, gA);
                  TCOPYIN(tB, gB);
                  if(k==0){
                    MATMUL(tACC, tA, tB);
                  }else{
                    MATMACC(tACC, tA, tB);
                  }
              }
              TCOPYOUT_ACC_DYNAMIC(gC, tACC, tACC.GetValidRow(), tACC.GetValidCol());
          }
      }
    }
}

/*
ResA find_reuseA_dynamic(int Mb, int Kb, int MAX_TILE_NUM) {
    int best_m = 0, best_k = 0, best_val = -1;

    for (int m = 1; m <= Mb; ++m) {
        for (int k = 1; k <= Kb; ++k) {
            int v = m * k;
            if (v <= MAX_TILE_NUM && v > best_val) {
                best_val = v;
                best_m = m;
                best_k = k;
            }
        }
    }
    return {best_m, best_k, best_val};
}

template<typename dtype, const int tM, const int tN, const int tK>
__attribute__((noinline)) void matmul_dynamic_reuseA(float* dst, dtype* src0, dtype* src1, int gM, int gN, int gK){
    const int MAX_TILE_NUM = 24;

    using gm_shapeA = global_tensor<dtype, RowMajor<-1, -1>>;
    using gm_shapeB = global_tensor<dtype, RowMajor<-1, -1>>;
    using gm_shapeC = global_tensor<float, RowMajor<-1, -1>>;
    using tile_shapeA = TileLeft<dtype, tM, tK, -1, -1>;
    using tile_shapeB = TileRight<dtype, tK, tN, -1, -1>;
    using tile_shapeACC = TileAcc<float, tM, tN, -1, -1>;

    int Mb = (gM + tM - 1) / tM;
    int Nb = (gN + tN - 1) / tN;
    int Kb = (gK + tK - 1) / tK;

    int rem_m = gM % tM;
    int rem_n = gN % tN;
    int rem_k = gK % tK;

    ResA R = find_reuseA_dynamic(Mb, Kb, MAX_TILE_NUM);
    
    int dM = R.m == 0? 0 : Mb / R.m;
    int rM = R.m == 0? 0 : Mb % R.m;

    // printf("Mb:%d, Nb:%d Kb:%d, R.m:%d, R.k:%d\n", Mb, Nb, Kb, R.m, R.k);
    for (int b=0;b<Batch;b++){
      for(int i=0;i<Mb;){
        int m_step = 1;
        if(R.m != 0){
          if((i+R.m) <= Mb){
            m_step = R.m;
          }else if( (rM != 0) && ((i+rM) <= Mb)) {
            m_step = rM;
          }else{
            m_step = 1;
          }
        }
        //printf("m_step is %d\n", m_step);

        tile_shapeA tA[m_step][R.k];

        for (int mm=0;mm<m_step;mm++) {
          for (int kk=0;kk<R.k;kk++) { 
            if( (i+mm+1) * tM > gM ){
              tA[mm][kk]= tile_shapeA(rem_m, tK);
            }else{
              tA[mm][kk]= tile_shapeA(tM, tK);
            }
          }
        }

        for(int ii=0;ii<m_step;ii++){
          for(int k=0;k<R.k;k++){
            size_t offset_A = (i+ii) * gK * tile_shapeA::Rows + k * tile_shapeA::Cols;
            gm_shapeA gA(src0+offset_A, gM, gK);
            TCOPYIN(tA[ii][k], gA);
          }

          int dyn_m = (i+ii+1) * tM > gM? rem_m:tM;

          for(int j=0;j<Nb;j++){
            int dyn_n = (j+1) * tN > gN ? rem_n:tN;
            tile_shapeACC tACC(dyn_m, dyn_n);

            for(int k=0;k<R.k;k++){
              size_t offset_B = k * gN * tile_shapeB::Rows + j * tile_shapeB::Cols;
              gm_shapeB gB(src1 + offset_B, gK, gN);
              tile_shapeB tB(tK, dyn_n);
              TCOPYIN(tB, gB);
              if(k==0){
                MATMUL(tACC, tA[ii][k], tB);
              }else{
                MATMACC(tACC, tA[ii][k], tB);
              }
            }

            if(R.k < Kb){
              for(int k=R.k;k<Kb;k++){
                size_t offset_A = (i+ii) * gK * tile_shapeA::Rows + k * tile_shapeA::Cols;
                size_t offset_B = k * gN * tile_shapeB::Rows + j * tile_shapeB::Cols;
                gm_shapeA gA(src0 + offset_A, gM, gK);
                gm_shapeB gB(src1 + offset_B, gK, gN);

                int dyn_k = (k+1) * tK > gK ? rem_k:tK;
                tile_shapeA tA(dyn_m, dyn_k);
                tile_shapeB tB(dyn_k, dyn_n);

                TCOPYIN(tA, gA);
                TCOPYIN(tB, gB);
                if(k==0){
                  MATMUL(tACC, tA, tB);
                }else{
                  MATMACC(tACC, tA, tB);
                }
              }
            }

            size_t offset_C = (i+ii) * gN * tile_shapeACC::Rows + j * tile_shapeACC::Cols;
            gm_shapeC gC(dst + offset_C, gM, gN);
            TCOPYOUT_ACC_DYNAMIC(gC, tACC, tACC.GetValidRow(), tACC.GetValidCol());
          }
        }

        i+= m_step;
      }
    }
}

template<typename dtype, const int tM, const int tN, const int tK>
__attribute__((noinline)) void matmul_dynamic_reuseB(float* dst, dtype* src0, dtype* src1, int gM, int gN, int gK){
    const int MAX_TILE_NUM = 24;

    using gm_shapeA = global_tensor<dtype, RowMajor<-1, -1>>;
    using gm_shapeB = global_tensor<dtype, RowMajor<-1, -1>>;
    using gm_shapeC = global_tensor<float, RowMajor<-1, -1>>;
    using tile_shapeA = TileLeft<dtype, tM, tK, -1, -1>;
    using tile_shapeB = TileRight<dtype, tK, tN, -1, -1>;
    using tile_shapeACC = TileAcc<float, tM, tN, -1, -1>;

    int Mb = (gM + tM - 1) / tM;
    int Nb = (gN + tN - 1) / tN;
    int Kb = (gK + tK - 1) / tK;

    int rem_m = gM % tM;
    int rem_n = gN % tN;
    int rem_k = gK % tK;

    ResA Ra = find_reuseA_dynamic(Nb, Kb, MAX_TILE_NUM);
    ResB R;
    R.n = Ra.m;
    R.k = Ra.k;
    R.val = Ra.val;
    
    int dN = R.n == 0? 0 : Nb / R.n;
    int rN = R.n == 0? 0 : Nb % R.n;

    for (int b=0;b<Batch;b++){
      for(int i=0;i<Nb;){
        int n_step = 1;
        if(R.n != 0){
          if((i+R.n) <= Nb){
            n_step = R.n;
          }else if( (rN != 0) && ((i+rN) <= Nb)) {
            n_step = rN;
          }else{
            n_step = 1;
          }
        }

        tile_shapeB tB[R.k][n_step];

        for (int nn=0;nn<n_step;nn++) {
          for (int kk=0;kk<R.k;kk++) { 
            if( (i+nn+1) * tN > gN ){
              tB[kk][nn]= tile_shapeB(tK, rem_n);
            }else{
              tB[kk][nn]= tile_shapeB(tK, tN);
            }
          }
        }

        for(int ii=0;ii<n_step;ii++){
          for(int k=0;k<R.k;k++){
            size_t offset_B = k * gN * tile_shapeB::Rows + (i+ii) * tile_shapeB::Cols;
            gm_shapeB gB(src1+offset_B, gK, gN);
            TCOPYIN(tB[k][ii], gB);
          }

          int dyn_n = (i+ii+1) * tN > gN? rem_n:tN;

          for(int j=0;j<Mb;j++){
            int dyn_m = (j+1) * tM > gM ? rem_m:tM;
            tile_shapeACC tACC(dyn_m, dyn_n);

            for(int k=0;k<R.k;k++){
              size_t offset_A = j * gK * tile_shapeA::Rows + k * tile_shapeA::Cols;
              gm_shapeA gA(src0 + offset_A, gM, gK);
              tile_shapeA tA(dyn_m, tK);
              TCOPYIN(tA, gA);
              if(k==0){
                MATMUL(tACC, tA, tB[k][ii]);
              }else{
                MATMACC(tACC, tA, tB[k][ii]);
              }
            }

            if(R.k < Kb){
              for(int k=R.k;k<Kb;k++){
                size_t offset_A = j * gK * tile_shapeA::Rows + k * tile_shapeA::Cols;
                size_t offset_B = k * gN * tile_shapeB::Rows + (i+ii) * tile_shapeB::Cols;
                gm_shapeA gA(src0 + offset_A, gM, gK);
                gm_shapeB gB(src1 + offset_B, gK, gN);

                int dyn_k = (k+1) * tK > gK ? rem_k:tK;
                tile_shapeA tA(dyn_m, dyn_k);
                tile_shapeB tB(dyn_k, dyn_n);

                TCOPYIN(tA, gA);
                TCOPYIN(tB, gB);
                if(k==0){
                  MATMUL(tACC, tA, tB);
                }else{
                  MATMACC(tACC, tA, tB);
                }
              }
            }

            size_t offset_C =  j * gN * tile_shapeACC::Rows + (i+ii) * tile_shapeACC::Cols;
            gm_shapeC gC(dst + offset_C, gM, gN);
            TCOPYOUT_ACC_DYNAMIC(gC, tACC, tACC.GetValidRow(), tACC.GetValidCol());
          }
        }

        i+= n_step;
      }
    }
}
*/

#include "matmul_dynamic_reuse.hpp"

template <typename dtype, const int gM, const int gN, const int gK, const int tM, const int tN, const int tK>
void matmul_mx(float *dst, dtype *src0, dtype *src1, uint8_t *src0_mx, uint8_t *src1_mx) {
  using gm_shapeA = global_tensor<dtype, RowMajor<gM, gK>>;
  using gm_shapeB = global_tensor<dtype, RowMajor<gK, gN>>;
  using gm_shapeAMX = global_tensor<uint8_t, RowMajor<gM, gK/32>>;
  using gm_shapeBMX = global_tensor<uint8_t, RowMajor<gK/32, gN>>;
  using gm_shapeC = global_tensor<float, RowMajor<gM, gN>>;

  using tile_shapeA = TileLeft<dtype, tM, tK>;
  using tile_shapeB = TileRight<dtype, tK, tN>;
  using tile_shapeAMX = Tile<Location::Scaling, uint8_t, tM, tK/32, BLayout::RowMajor, tM, tK/32, SLayout::NoneBox>;
  using tile_shapeBMX = Tile<Location::Scaling, uint8_t, tK/32, tN, BLayout::ColMajor, tK/32, tN, SLayout::NoneBox>;
  using tile_shapeACC = TileAcc<float, tM, tN>;
  using itA = global_iterator<gm_shapeA, tile_shapeA>;
  using itB = global_iterator<gm_shapeB, tile_shapeB>;
  using itAMX = global_iterator<gm_shapeAMX, tile_shapeAMX>;
  using itBMX = global_iterator<gm_shapeBMX, tile_shapeBMX>;
  using itC = global_iterator<gm_shapeC, tile_shapeACC>;

  itA gAIter(src0);
  itB gBIter(src1);
  itAMX gAMXIter(src0_mx);
  itBMX gBMXIter(src1_mx);
  itC gCIter(dst);

  const int Mb = gM / tM;
  const int Nb = gN / tN;
  const int Kb = gK / tK;

  for(int i=0;i<Mb;i++){
    for(int j=0;j<Nb;j++){
        auto gC = gCIter(i, j);

        tile_shapeACC tACC;
        #pragma clang loop unroll(full)
        for(int k=0;k<Kb;k++){
          auto gA = gAIter(i,k);
          auto gB = gBIter(k,j);
          auto gAMX = gAIter(i,k);
          auto gBMX = gBIter(k,j);
          tile_shapeA tA;
          tile_shapeB tB;
          tile_shapeAMX tAMX;
          tile_shapeBMX tBMX;
          TCOPYIN(tA, gA);
          TCOPYIN(tB, gB);

          blk_tload(tAMX.GetValidCol(), tAMX.GetValidRow(), tile_shapeAMX::Cols,
          type_traits<typename tile_shapeAMX::DType>::TypeCode,
          PadValue::Null,
          LayoutCvtEnum::ND2ZZ,
          tAMX.data(),
          gAMX.data(),
          (gAMX.GetStride(3) * type_traits<typename gm_shapeAMX::DType>::bits + 7) / 8);

          blk_tload(tBMX.GetValidCol(), tBMX.GetValidRow(), tile_shapeBMX::Cols,
          type_traits<typename tile_shapeBMX::DType>::TypeCode,
          PadValue::Null,
          LayoutCvtEnum::ND2NN,
          tBMX.data(),
          gBMX.data(),
          (gBMX.GetStride(3) * type_traits<typename gm_shapeBMX::DType>::bits + 7) / 8);

          if(k==0){
            MATMULMX(tACC, tA, tAMX, tB, tBMX);
          }else{
            MATMACCMX(tACC, tA, tAMX, tB, tBMX);
          }
        }
        TCOPYOUT_ACC(gC, tACC);
    }
  }
}

template <typename dtype, const int gM, const int gN, const int gK, const int tM,
          const int tN, const int tK>
void matmul_mask_2lvl(float *c_ptr, dtype *a_ptr, dtype *b_ptr) {

  using gm_shapeA = global_tensor<dtype, RowMajor<gM, gK>>;
  using gm_shapeB = global_tensor<dtype, RowMajor<gK, gN>>;
  using gm_shapeC = global_tensor<float, RowMajor<gM, gN>>;
  using tile_shapeA = TileLeft<dtype, tM, tK>;
  using tile_shapeB = TileRight<dtype, tK, tN>;
  using tile_shapeACC = TileAcc<float, tM, tN>;
  using itA = global_iterator<gm_shapeA, tile_shapeA>;
  using itB = global_iterator<gm_shapeB, tile_shapeB>;
  using itC = global_iterator<gm_shapeC, tile_shapeACC>;

  itA gAIter(a_ptr);
  itB gBIter(b_ptr);
  itC gCIter(c_ptr);

  const int Mb = gM / tM;
  const int Nb = gN / tN;
  const int Kb = gK / tK;

  const int rmd_M = gM % tM;
  const int rmd_N = gN % tN;
  const int rmd_K = gK % tK;

  using tile_shapeA_trows = TileLeft<dtype, tM, tK,  tM, rmd_K>;
  using tile_shapeA_tcols = TileLeft<dtype, tM, tK, rmd_M, tK>;
  using tile_shapeA_tcorner = TileLeft<dtype, tM, tK, rmd_M, rmd_K>;

  using tile_shapeB_trows = TileRight<dtype, tK, tN, tK, rmd_N>;
  using tile_shapeB_tcols = TileRight<dtype, tK, tN, rmd_K, tN>;
  using tile_shapeB_tcorner = TileRight<dtype, tK, tN, rmd_K, rmd_N>;

  using tile_shapeC_trows = TileAcc<float, tM, tN, tM, rmd_N>;
  using tile_shapeC_tcols = TileAcc<float, tM, tN, rmd_M, tN>;
  using tile_shapeC_tcorner = TileAcc<float, tM, tN, rmd_M, rmd_N>;

  for (int b=0;b<Batch;b++){
    for (int i = 0; i < Mb; ++i) {
      for (int j = 0; j < Nb; ++j) {
        auto gC = gCIter(i, j);

        tile_shapeACC tACC;
        
        if constexpr(Kb>0){
          auto gA = gAIter(i, 0);
          auto gB = gBIter(0, j);

          tile_shapeA tA;
          tile_shapeB tB;
          TCOPYIN_2LVL(tA, gA);
          TCOPYIN_2LVL(tB, gB);
          MATMUL(tACC, tA, tB);        
        }
        #pragma clang loop unroll(full)
        for (int k = 1; k < Kb; ++k) {
          auto gA = gAIter(i, k);
          auto gB = gBIter(k, j);

          tile_shapeA tA;
          tile_shapeB tB;
          TCOPYIN_2LVL(tA, gA);
          TCOPYIN_2LVL(tB, gB);
          MATMACC(tACC, tA, tB);
        }

        if constexpr (rmd_K) {
          auto gA = gAIter(i, Kb);
          auto gB = gBIter(Kb, j);

          tile_shapeA_trows tA;
          tile_shapeB_tcols tB;
          TCOPYIN_2LVL(tA, gA);
          TCOPYIN_2LVL(tB, gB);
          if constexpr(Kb>0){
            MATMACC(tACC, tA, tB);
          } else {
            MATMUL(tACC, tA, tB);
          }
        }
        TCOPYOUT_ACC(gC, tACC);
      }
      if constexpr (rmd_N) {
        auto gC = gCIter(i, Nb);

        tile_shapeC_trows tACC;
        if constexpr(Kb>0){
          auto gA = gAIter(i, 0);
          auto gB = gBIter(0, Nb);

          tile_shapeA tA;
          tile_shapeB_trows tB;
          TCOPYIN_2LVL(tA, gA);
          TCOPYIN_2LVL(tB, gB);
          MATMUL(tACC, tA, tB);        
        }
        #pragma clang loop unroll(full)
        for (int k = 1; k < Kb; ++k) {
          auto gA = gAIter(i, k);
          auto gB = gBIter(k, Nb);

          tile_shapeA tA;
          tile_shapeB_trows tB;
          TCOPYIN_2LVL(tA, gA);
          TCOPYIN_2LVL(tB, gB);
          MATMACC(tACC, tA, tB);
        }
        if constexpr (rmd_K) {
          auto gA = gAIter(i, Kb);
          auto gB = gBIter(Kb, Nb);

          tile_shapeA_trows tA;
          tile_shapeB_tcorner tB;
          TCOPYIN_2LVL(tA, gA);
          TCOPYIN_2LVL(tB, gB);
          if constexpr(Kb>0){
            MATMACC(tACC, tA, tB);
          } else {
            MATMUL(tACC, tA, tB);
          }
        }
        TCOPYOUT_ACC(gC, tACC);
      }
    }
    if constexpr (rmd_M) {
      for (int j = 0; j < Nb; ++j) {
        auto gC = gCIter(Mb, j);

        tile_shapeC_tcols tACC;
        if constexpr(Kb>0){
          auto gA = gAIter(Mb, 0);
          auto gB = gBIter(0, j);

          tile_shapeA_tcols tA;
          tile_shapeB tB;
          TCOPYIN_2LVL(tA, gA);
          TCOPYIN_2LVL(tB, gB);
          MATMUL(tACC, tA, tB);        
        }
        #pragma clang loop unroll(full)
        for (int k = 1; k < Kb; ++k) {
          auto gA = gAIter(Mb, k);
          auto gB = gBIter(k, j);

          tile_shapeA_tcols tA;
          tile_shapeB tB;
          TCOPYIN_2LVL(tA, gA);
          TCOPYIN_2LVL(tB, gB);
          MATMACC(tACC, tA, tB);
        }
        if constexpr (rmd_K) {
          auto gA = gAIter(Mb, Kb);
          auto gB = gBIter(Kb, j);

          tile_shapeA_tcorner tA;
          tile_shapeB_tcols tB;
          TCOPYIN_2LVL(tA, gA);
          TCOPYIN_2LVL(tB, gB);
          if constexpr(Kb>0){
            MATMACC(tACC, tA, tB);
          } else {
            MATMUL(tACC, tA, tB);
          }
        }
        TCOPYOUT_ACC(gC, tACC);
      }
      if constexpr (rmd_N) {
        auto gC = gCIter(Mb, Nb);

        tile_shapeC_tcorner tACC;
        if constexpr(Kb>0){
          auto gA = gAIter(Mb, 0);
          auto gB = gBIter(0, Nb);

          tile_shapeA_tcols tA;
          tile_shapeB_trows tB;
          TCOPYIN_2LVL(tA, gA);
          TCOPYIN_2LVL(tB, gB);
          MATMUL(tACC, tA, tB);        
        }
        #pragma clang loop unroll(full)
        for (int k = 1; k < Kb; ++k) {
          auto gA = gAIter(Mb, k);
          auto gB = gBIter(k, Nb);

          tile_shapeA_tcols tA;
          tile_shapeB_trows tB;
          TCOPYIN_2LVL(tA, gA);
          TCOPYIN_2LVL(tB, gB);
          MATMACC(tACC, tA, tB);
        }
        if constexpr (rmd_K) {
          auto gA = gAIter(Mb, Kb);
          auto gB = gBIter(Kb, Nb);

          tile_shapeA_tcorner tA;
          tile_shapeB_tcorner tB;
          TCOPYIN_2LVL(tA, gA);
          TCOPYIN_2LVL(tB, gB);
          if constexpr(Kb>0){
            MATMACC(tACC, tA, tB);
          } else {
            MATMUL(tACC, tA, tB);
          }
        }
        TCOPYOUT_ACC(gC, tACC);
      }
    }
  }
}

template<const int gM, const int gN, const int gK, const int tM, const int tN, const int tK>
void matmul_vec(float* dst, float* src0, float* src1){
    using gm_shapeA = global_tensor<float, RowMajor<gM, gK>>;
    using gm_shapeB = global_tensor<float, RowMajor<gK, gN>>;
    using gm_shapeC = global_tensor<float, RowMajor<gM, gN>>;
    using tile_shapeA = Tile<Location::Vec, float, tM, tK, BLayout::RowMajor>;
    using tile_shapeB = Tile<Location::Vec, float, tK, tN, BLayout::RowMajor>;
    using tile_shapeACC = Tile<Location::Vec, float, tM, tN, BLayout::RowMajor>;
    using gm_iteratorA = global_iterator<gm_shapeA, tile_shapeA>;
    using gm_iteratorB = global_iterator<gm_shapeB, tile_shapeB>;
    using gm_iteratorC = global_iterator<gm_shapeC, tile_shapeACC>;

    gm_iteratorA gAIter(src0);
    gm_iteratorB gBIter(src1);
    gm_iteratorC gCIter(dst);

    const int Mb = gM / tM;
    const int Nb = gN / tN;
    const int Kb = gK / tK;

    for(int i=0;i<Mb;i++){
        for(int j=0;j<Nb;j++){
            auto gC = gCIter(i, j);

            tile_shapeACC tACC(0);
            for(int k=0;k<Kb;k++){
                auto gA = gAIter(i,k);
                auto gB = gBIter(k,j);
                tile_shapeA tA;
                tile_shapeB tB;
                TCOPYIN(tA, gA);
                TCOPYIN(tB, gB);
                MATMACC(tACC, tA, tB);
            }
            TCOPYOUT(gC, tACC);
        }
    }
}

template <uint16_t M, uint16_t N, uint16_t K>
void matmul_tile_vec(float* dst, float* src0, float* src1) {
    using gm_shape_A = global_tensor<float, RowMajor<M, K>>;
    using gm_shape_B = global_tensor<float, RowMajor<K, N>>;
    using gm_shape_C = global_tensor<float, RowMajor<M, N>>;

    using tile_shape_A = Tile<Location::Vec, float, M, K, BLayout::RowMajor>;
    using tile_shape_B = Tile<Location::Vec, float, K, N, BLayout::RowMajor>;
    using tile_shape_C = Tile<Location::Vec, float, M, N, BLayout::RowMajor>;

    gm_shape_A s0(src0);
    gm_shape_B s1(src1);
    gm_shape_C res(dst);

    tile_shape_A d0;
    tile_shape_B d1;
    tile_shape_C d2;

    TCOPYIN(d0, s0);
    TCOPYIN(d1, s1);
    MATMUL(d2, d0, d1);
    TCOPYOUT(res, d2);
}

template <uint16_t M, uint16_t N, uint16_t K>
void matmul_tile_frac(float* dst, float* src0, float* src1) {
    using gm_shape_A = global_tensor<float, RowMajor<M, K>>;
    using gm_shape_B = global_tensor<float, ColMajor<K, N>>;
    using gm_shape_C = global_tensor<float, RowMajor<M, N>>;

    using tile_shape_A = TileLeft<float, M, K>;
    using tile_shape_B = TileRight<float, K, N>;
    using tile_shape_C = TileAcc<float, M, N>;

    gm_shape_A s0(src0);
    gm_shape_B s1(src1);
    gm_shape_C res(dst);

    tile_shape_A d0;
    tile_shape_B d1;
    tile_shape_C d2;

    TCOPYIN(d0, s0);
    TCOPYIN(d1, s1);
    MATMUL(d2, d0, d1);
    TCOPYOUT_ACC(res, d2);
}



#endif
