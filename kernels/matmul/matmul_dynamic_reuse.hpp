#ifndef MATMUL_DYNAMIC_REUSE_HPP
#define MATMUL_DYNAMIC_REUSE_HPP

#include <common/pto_tileop.hpp>
#include "template_asm.h"

#define CODE_REUSEA(m_step) \
        for (int mm=0;mm<m_step;mm++) { \
          for (int kk=0;kk<RK;kk++) {   \
            if( (i+mm+1) * tM > gM ){   \
              tA[mm][kk]= tile_shapeA(rem_m, tK); \
            }else{ \
              tA[mm][kk]= tile_shapeA(tM, tK); \
            } \
          } \
        } \
 \
        for(int ii=0;ii<m_step;ii++){ \
          for(int k=0;k<RK;k++){ \
            size_t offset_A = (i+ii) * gK * tile_shapeA::Rows + k * tile_shapeA::Cols; \
            gm_shapeA gA(src0+offset_A, gM, gK); \
            TCOPYIN(tA[ii][k], gA); \
          } \
 \
          int dyn_m = (i+ii+1) * tM > gM? rem_m:tM; \
 \
          for(int j=0;j<Nb;j++){ \
            int dyn_n = (j+1) * tN > gN ? rem_n:tN; \
            tile_shapeACC tACC(dyn_m, dyn_n);       \
 \
            for(int k=0;k<RK;k++){ \
              size_t offset_B = k * gN * tile_shapeB::Rows + j * tile_shapeB::Cols; \
              gm_shapeB gB(src1 + offset_B, gK, gN); \
              tile_shapeB tB(tK, dyn_n); \
              TCOPYIN(tB, gB); \
              if(k==0){ \
                MATMUL(tACC, tA[ii][k], tB); \
              }else{ \
                MATMACC(tACC, tA[ii][k], tB); \
              } \
            } \
 \
            if(RK < Kb){ \
              for(int k=RK;k<Kb;k++){ \
                size_t offset_A = (i+ii) * gK * tile_shapeA::Rows + k * tile_shapeA::Cols; \
                size_t offset_B = k * gN * tile_shapeB::Rows + j * tile_shapeB::Cols; \
                gm_shapeA gA(src0 + offset_A, gM, gK); \
                gm_shapeB gB(src1 + offset_B, gK, gN); \
 \
                int dyn_k = (k+1) * tK > gK ? rem_k:tK; \
                tile_shapeA tA(dyn_m, dyn_k); \
                tile_shapeB tB(dyn_k, dyn_n); \
 \
                TCOPYIN(tA, gA); \
                TCOPYIN(tB, gB); \
                if(k==0){ \
                  MATMUL(tACC, tA, tB); \
                }else{ \
                  MATMACC(tACC, tA, tB); \
                } \
              } \
            } \
 \
            size_t offset_C = (i+ii) * gN * tile_shapeACC::Rows + j * tile_shapeACC::Cols; \
            gm_shapeC gC(dst + offset_C, gM, gN); \
            TCOPYOUT_ACC_DYNAMIC(gC, tACC, tACC.GetValidRow(), tACC.GetValidCol()); \
          } \
        }

template<typename dtype, const int tM, const int tN, const int tK, const int MAX_TILE_NUM=24, const int RM, const int RK>
__attribute__((noinline)) void matmul_dynamic_reuseA(float* dst, dtype* src0, dtype* src1, int gM, int gN, int gK){
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
      for(int i=0;i<Mb;){
        if((i+RM) <= Mb){
          tile_shapeA tA[RM][RK];
          CODE_REUSEA(RM);
          i += RM;
        } else{
          tile_shapeA tA[1][RK];
          CODE_REUSEA(1);
          i += 1;
        }
      }
    }
}

#define CODE_REUSEB(n_step) \
        for (int nn=0;nn<n_step;nn++) { \
          for (int kk=0;kk<RK;kk++) { \
            if( (i+nn+1) * tN > gN ){ \
              tB[kk][nn]= tile_shapeB(tK, rem_n); \
            }else{ \
              tB[kk][nn]= tile_shapeB(tK, tN); \
            } \
          } \
        } \
 \
        for(int ii=0;ii<n_step;ii++){ \
          for(int k=0;k<RK;k++){ \
            size_t offset_B = k * gN * tile_shapeB::Rows + (i+ii) * tile_shapeB::Cols; \
            gm_shapeB gB(src1+offset_B, gK, gN); \
            TCOPYIN(tB[k][ii], gB); \
          } \
 \
          int dyn_n = (i+ii+1) * tN > gN? rem_n:tN; \
 \
          for(int j=0;j<Mb;j++){ \
            int dyn_m = (j+1) * tM > gM ? rem_m:tM; \
            tile_shapeACC tACC(dyn_m, dyn_n); \
 \
            for(int k=0;k<RK;k++){ \
              size_t offset_A = j * gK * tile_shapeA::Rows + k * tile_shapeA::Cols; \
              gm_shapeA gA(src0 + offset_A, gM, gK); \
              tile_shapeA tA(dyn_m, tK); \
              TCOPYIN(tA, gA); \
              if(k==0){ \
                MATMUL(tACC, tA, tB[k][ii]); \
              }else{ \
                MATMACC(tACC, tA, tB[k][ii]); \
              } \
            } \
 \
            if(RK < Kb){ \
              for(int k=RK;k<Kb;k++){ \
                size_t offset_A = j * gK * tile_shapeA::Rows + k * tile_shapeA::Cols; \
                size_t offset_B = k * gN * tile_shapeB::Rows + (i+ii) * tile_shapeB::Cols; \
                gm_shapeA gA(src0 + offset_A, gM, gK); \
                gm_shapeB gB(src1 + offset_B, gK, gN); \
 \
                int dyn_k = (k+1) * tK > gK ? rem_k:tK; \
                tile_shapeA tA(dyn_m, dyn_k); \
                tile_shapeB tB(dyn_k, dyn_n); \
 \
                TCOPYIN(tA, gA); \
                TCOPYIN(tB, gB); \
                if(k==0){ \
                  MATMUL(tACC, tA, tB); \
                }else{ \
                  MATMACC(tACC, tA, tB); \
                } \
              } \
            } \
 \
            size_t offset_C =  j * gN * tile_shapeACC::Rows + (i+ii) * tile_shapeACC::Cols; \
            gm_shapeC gC(dst + offset_C, gM, gN); \
            TCOPYOUT_ACC_DYNAMIC(gC, tACC, tACC.GetValidRow(), tACC.GetValidCol()); \
          } \
        }

template<typename dtype, const int tM, const int tN, const int tK, const int MAX_TILE_NUM=24, const int RK, const int RN>
__attribute__((noinline)) void matmul_dynamic_reuseB(float* dst, dtype* src0, dtype* src1, int gM, int gN, int gK){
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
      for(int i=0;i<Nb;){
        if((i+RN) <= Nb){
          tile_shapeB tB[RK][RN];
          CODE_REUSEB(RN);
          i += RN;
        } else{
          tile_shapeB tB[RK][1];
          CODE_REUSEB(1);
          i += 1;
        }
      }
    }
}

#endif