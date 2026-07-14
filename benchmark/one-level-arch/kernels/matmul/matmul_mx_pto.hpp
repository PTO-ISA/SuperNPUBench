#ifndef MATMUL_MX_HPP
#define MATMUL_MX_HPP

#include <common/pto_tileop.hpp>
#include <utils/layout_transform.hpp>
#include <cstdint>
#include <cstdio>


// #define DUMP_TILE(label, TileVar, DumpBuf, Rows, Cols) \
//     do { \
//         GlobalTensor<typename decltype(TileVar)::DType, \
//                      Shape<1,1,1,Rows,Cols>, \
//                      Stride<1,1,1,Cols,1>> _g(DumpBuf); \
//         TSTORE(_g, TileVar); \
//         printf("[DUMP] %s (shape=%dx%d):\n", label, Rows, Cols); \
//         for (int ri = 0; ri < Rows; ri++) { \
//             printf("  row%2d: ", ri); \
//             for (int ci = 0; ci < Cols; ci++) \
//                 printf("%.4f ", DumpBuf[ri * Cols + ci]); \
//             printf("\n"); \
//         } \
//         fflush(stdout); \
//     } while (0)

#ifndef Batch
#define Batch 1
#endif

using namespace pto;

// TODO, move to utils.cpp
template <is_global_data_v GmOut, is_tile_data_v TileAcc>
void store_acc_tile(GmOut &Gout, TileAcc &tAcc){
    using TileAccOut = Tile<Location::Vec, typename TileAcc::DType, TileAcc::Rows, TileAcc::Cols, BLayout::RowMajor, TileAcc::ValidRow, TileAcc::ValidCol>;
    TileAccOut tAccOut;
    TCVT(tAccOut, tAcc);
    TSTORE(Gout, tAccOut);
}

// typeb_wfactor 表明typeA和typeB的位宽比例，比如fp8是fp4x2的两倍，
// smatrix_wfactor : scaling matrix 与计算matrix位宽比
template <typename dtypeA, const int gM, const int gN, const int gK, const int tM, const int tN, const int tK,
          typename dtypeB = dtypeA, const int typeb_wfactor = 1, const int smatrix_wfactor=1>
void matmul_mxfp(float *dst, dtypeA *src0, dtypeB *src1, uint8_t *src0_mx, uint8_t *src1_mx) {
  // only support regular shape now for this operator!
  // static_assert(gM % tM == 0);
  // static_assert(gN % tN == 0);
  // static_assert(gK % tK == 0);
  static const uint32_t valid_row = (tM > gM) ? gM : tM;
  static const uint32_t valid_col = (tN > gN) ? gN : tN;
  using gm_shapeA = global_tensor<dtypeA, RowMajor<gM, gK>>;
  using gm_shapeB = global_tensor<dtypeB, RowMajor<gK/typeb_wfactor, gN>>;
  using gm_shapeC = global_tensor<float, RowMajor<gM, gN>>;

  using tile_shapeA = TileLeft<dtypeA, tM, tK, valid_row, tK>;
  using tile_shapeB = TileRight<dtypeB, tK/typeb_wfactor, tN, tK/typeb_wfactor, valid_col>;
  using tile_shapeACC = TileAcc<float, tM, tN, valid_row, valid_col>;
  using itA = global_iterator<gm_shapeA, tile_shapeA>;
  using itB = global_iterator<gm_shapeB, tile_shapeB>;
  using itC = global_iterator<gm_shapeC, tile_shapeACC>;

  itA gAIter(src0);
  itB gBIter(src1);
  itC gCIter(dst);

  using gm_shapeAMX = global_tensor<uint8_t, RowMajor<gM, gK/smatrix_wfactor>>;
  gm_shapeAMX gAMX(src0_mx);
  using tile_shapeAMX = Tile<Location::Scaling, uint8_t, tM, tK, BLayout::RowMajor, valid_row, tK/smatrix_wfactor, SLayout::RowMajor>; // 实际tile尺寸<tM, tK/32>, 需初始化为0
  using itAMX = global_iterator<gm_shapeAMX, tile_shapeAMX>;
  using tile_ND2ZZOffset = Tile<Location::Vec, uint16_t, 1, tM*tK/smatrix_wfactor, BLayout::RowMajor>;
  tile_ND2ZZOffset nd2zz_offset;

  using gm_shapeBMX = global_tensor<uint8_t, RowMajor<gK/smatrix_wfactor, gN>>;
  gm_shapeBMX gBMX(src1_mx);
  using tile_shapeBMX = Tile<Location::Scaling, uint8_t, tK, tN, BLayout::ColMajor, tK/smatrix_wfactor, valid_col, SLayout::ColMajor>;
  using itBMX = global_iterator<gm_shapeBMX, tile_shapeBMX>;
  using tile_ND2NNOffset = Tile<Location::Vec, uint16_t, 1, tK/smatrix_wfactor*tN, BLayout::RowMajor>;
  tile_ND2NNOffset nd2nn_offset;

  const int Mb = (gM) / tM;
  const int Nb = (gN) / tN;
  const int Kb = (gK) / tK;

  const int rmd_M = gM % tM;
  const int rmd_N = gN % tN;
  const int rmd_K = gK % tK;

  using tile_shapeA_trows = TileLeft<dtypeA, tM, tK,  valid_row, rmd_K>;
  using tile_shapeA_tcols = TileLeft<dtypeA, tM, tK, rmd_M, tK>;
  using tile_shapeA_tcorner = TileLeft<dtypeA, tM, tK, rmd_M, rmd_K>;

  using tile_shapeAMX_trows = Tile<Location::Scaling, uint8_t, tM, tK, BLayout::RowMajor, valid_row, rmd_K/smatrix_wfactor, SLayout::RowMajor>;
  using tile_shapeAMX_tcols = Tile<Location::Scaling, uint8_t, tM, tK, BLayout::RowMajor, rmd_M, tK/smatrix_wfactor, SLayout::RowMajor>;
  using tile_shapeAMX_tcorner = Tile<Location::Scaling, uint8_t, tM, tK, BLayout::RowMajor, rmd_M, rmd_K/smatrix_wfactor, SLayout::RowMajor>;

  using tile_shapeB_trows = TileRight<dtypeB, tK, tN, tK, rmd_N>;
  using tile_shapeB_tcols = TileRight<dtypeB, tK, tN, rmd_K, valid_col>;
  using tile_shapeB_tcorner = TileRight<dtypeB, tK, tN, rmd_K, rmd_N>;

  using tile_shapeBMX_trows = Tile<Location::Scaling, uint8_t, tK, tN, BLayout::ColMajor, tK/smatrix_wfactor, rmd_N, SLayout::ColMajor>;
  using tile_shapeBMX_tcols = Tile<Location::Scaling, uint8_t, tK, tN, BLayout::ColMajor, rmd_K/smatrix_wfactor, valid_col, SLayout::ColMajor>;
  using tile_shapeBMX_tcorner = Tile<Location::Scaling, uint8_t, tK, tN, BLayout::ColMajor, rmd_K/smatrix_wfactor, rmd_N, SLayout::ColMajor>;

  using tile_shapeC_trows = TileAcc<float, tM, tN, valid_row, rmd_N>;
  using tile_shapeC_tcols = TileAcc<float, tM, tN, rmd_M, valid_col>;
  using tile_shapeC_tcorner = TileAcc<float, tM, tN, rmd_M, rmd_N>;

  alignas(256) static uint16_t  g_dump_intTile[tM*tK/smatrix_wfactor];
  for(int i=0;i<Mb;i++){
    for(int j=0;j<Nb;j++){
      auto gC = gCIter(i, j);

      tile_shapeACC tACC;
      #pragma clang loop unroll(full)
      for(int k=0;k<Kb;k++){
        auto gA = gAIter(i,k);
        auto gB = gBIter(k,j);
        tile_shapeA tA;
        tile_shapeB tB;
        TLOAD(tA, gA);
        TLOAD(tB, gB);

        // if (src0_mx != nullptr && src1_mx != nullptr) {
        tile_shapeAMX tAMX;
        gen_ND2ZZ_offset_Impl<gm_shapeAMX, tile_shapeAMX, tile_ND2ZZOffset>(gAMX, tAMX, nd2zz_offset, i, k);
        // DUMP_TILE("111", nd2zz_offset, g_dump_intTile, 1, tM*tK/smatrix_wfactor);
        MGATHER(tAMX, gAMX, nd2zz_offset);  // ND2ZZ, tile masked
        tile_shapeBMX tBMX;
        gen_ND2NN_offset_Impl<gm_shapeBMX, tile_shapeBMX, tile_ND2NNOffset>(gBMX, tBMX, nd2nn_offset, k, j);
        MGATHER(tBMX, gBMX, nd2nn_offset);  // ND2NN, tile masked
        if(k==0){
          TMATMUL_MX(tACC, tA, tAMX, tB, tBMX);
        }else{
          TMATMUL_MX(tACC, tA, tAMX, tB, tBMX);
        }
      }
      if constexpr (rmd_K) {
        auto gA = gAIter(i,Kb);
        auto gB = gBIter(Kb,j);
        tile_shapeA_trows tA;
        tile_shapeB_tcols tB;
        TLOAD(tA, gA);
        TLOAD(tB, gB);

        tile_shapeAMX_trows tAMX;
        gen_ND2ZZ_offset_Impl<gm_shapeAMX, tile_shapeAMX_trows, tile_ND2ZZOffset>(gAMX, tAMX, nd2zz_offset, i, Kb);
        // DUMP_TILE("111", nd2zz_offset, g_dump_intTile, 1, tM*tK/smatrix_wfactor);
        MGATHER(tAMX, gAMX, nd2zz_offset);  // ND2ZZ, tile masked
        tile_shapeBMX_tcols tBMX;
        gen_ND2NN_offset_Impl<gm_shapeBMX, tile_shapeBMX_tcols, tile_ND2NNOffset>(gBMX, tBMX, nd2nn_offset, Kb, j);
        MGATHER(tBMX, gBMX, nd2nn_offset);  // ND2NN, tile masked

        if constexpr(Kb>0){
          TMATMUL_MX(tACC, tA, tAMX, tB, tBMX);
        } else {
          TMATMUL_MX(tACC, tA, tAMX, tB, tBMX);
        }
      }
      store_acc_tile(gC, tACC);
    }
    if constexpr (rmd_N) {
      auto gC = gCIter(i, Nb);
      tile_shapeC_trows tACC;
      for(int k=0;k<Kb;k++){
        auto gA = gAIter(i, k);
        auto gB = gBIter(k, Nb);
        tile_shapeA tA;
        tile_shapeB_trows tB;
        TLOAD(tA, gA);
        TLOAD(tB, gB);

        tile_shapeAMX tAMX;
        gen_ND2ZZ_offset_Impl<gm_shapeAMX, tile_shapeAMX, tile_ND2ZZOffset>(gAMX, tAMX, nd2zz_offset, i, k);
        MGATHER(tAMX, gAMX, nd2zz_offset);  // ND2ZZ, tile masked
        tile_shapeBMX_trows tBMX;
        gen_ND2NN_offset_Impl<gm_shapeBMX, tile_shapeBMX_trows, tile_ND2NNOffset>(gBMX, tBMX, nd2nn_offset, k, Nb);
        MGATHER(tBMX, gBMX, nd2nn_offset);  // ND2NN, tile masked

        if(k==0){
          TMATMUL_MX(tACC, tA, tAMX, tB, tBMX);
        }else{
          TMATMUL_MX(tACC, tA, tAMX, tB, tBMX);
        }
      }

      if constexpr (rmd_K) {
        auto gA = gAIter(i, Kb);
        auto gB = gBIter(Kb, Nb);

        tile_shapeA_trows tA;
        tile_shapeB_tcorner tB;
        TLOAD(tA, gA);
        TLOAD(tB, gB);

        tile_shapeAMX_trows tAMX;
        gen_ND2ZZ_offset_Impl<gm_shapeAMX, tile_shapeAMX_trows, tile_ND2ZZOffset>(gAMX, tAMX, nd2zz_offset, i, Kb);
        MGATHER(tAMX, gAMX, nd2zz_offset);  // ND2ZZ, tile masked
        tile_shapeBMX_tcorner tBMX;
        gen_ND2NN_offset_Impl<gm_shapeBMX, tile_shapeBMX_tcorner, tile_ND2NNOffset>(gBMX, tBMX, nd2nn_offset, Kb, Nb);
        MGATHER(tBMX, gBMX, nd2nn_offset);  // ND2NN, tile masked
        if constexpr(Kb>0){
          TMATMUL_MX(tACC, tA, tAMX, tB, tBMX);
        } else {
          TMATMUL_MX(tACC, tA, tAMX, tB, tBMX);
        }
      }
      store_acc_tile(gC, tACC);
    }
  }
  if constexpr (rmd_M) {
    for (int j = 0; j < Nb; ++j) {
      auto gC = gCIter(Mb, j);

      tile_shapeC_tcols tACC;
      #pragma clang loop unroll(full)
      for (int k = 0; k < Kb; ++k) {
        auto gA = gAIter(Mb, k);
        auto gB = gBIter(k, j);

        tile_shapeA_tcols tA;
        tile_shapeB tB;
        TLOAD(tA, gA);
        TLOAD(tB, gB);

        tile_shapeAMX_tcols tAMX;
        gen_ND2ZZ_offset_Impl<gm_shapeAMX, tile_shapeAMX_tcols, tile_ND2ZZOffset>(gAMX, tAMX, nd2zz_offset, Mb, k);
        MGATHER(tAMX, gAMX, nd2zz_offset);  // ND2ZZ, tile masked
        tile_shapeBMX tBMX;
        gen_ND2NN_offset_Impl<gm_shapeBMX, tile_shapeBMX, tile_ND2NNOffset>(gBMX, tBMX, nd2nn_offset, k, j);
        MGATHER(tBMX, gBMX, nd2nn_offset);  // ND2NN, tile masked

        if(k==0){
          TMATMUL_MX(tACC, tA, tAMX, tB, tBMX);
        }else{
          TMATMUL_MX(tACC, tA, tAMX, tB, tBMX);
        }
      }

      if constexpr (rmd_K) {
        auto gA = gAIter(Mb, Kb);
        auto gB = gBIter(Kb, j);

        tile_shapeA_tcorner tA;
        tile_shapeB_tcols tB;
        TLOAD(tA, gA);
        TLOAD(tB, gB);

        tile_shapeAMX_tcorner tAMX;
        gen_ND2ZZ_offset_Impl<gm_shapeAMX, tile_shapeAMX_tcorner, tile_ND2ZZOffset>(gAMX, tAMX, nd2zz_offset, Mb, Kb);
        MGATHER(tAMX, gAMX, nd2zz_offset);  // ND2ZZ, tile masked
        tile_shapeBMX_tcols tBMX;
        gen_ND2NN_offset_Impl<gm_shapeBMX, tile_shapeBMX_tcols, tile_ND2NNOffset>(gBMX, tBMX, nd2nn_offset, Kb, j);
        MGATHER(tBMX, gBMX, nd2nn_offset);  // ND2NN, tile masked

        if constexpr(Kb>0){
          TMATMUL_MX(tACC, tA, tAMX, tB, tBMX);
        } else {
          TMATMUL_MX(tACC, tA, tAMX, tB, tBMX);
        }
      }
      store_acc_tile(gC, tACC);
    }
    if constexpr (rmd_N) {
      auto gC = gCIter(Mb, Nb);

      tile_shapeC_tcorner tACC;
      #pragma clang loop unroll(full)
      for (int k = 0; k < Kb; ++k) {
        auto gA = gAIter(Mb, k);
        auto gB = gBIter(k, Nb);

        tile_shapeA_tcols tA;
        tile_shapeB_trows tB;
        TLOAD(tA, gA);
        TLOAD(tB, gB);

        tile_shapeAMX_tcols tAMX;
        gen_ND2ZZ_offset_Impl<gm_shapeAMX, tile_shapeAMX_tcols, tile_ND2ZZOffset>(gAMX, tAMX, nd2zz_offset, Mb, k);
        MGATHER(tAMX, gAMX, nd2zz_offset);  // ND2ZZ, tile masked
        tile_shapeBMX_trows tBMX;
        gen_ND2NN_offset_Impl<gm_shapeBMX, tile_shapeBMX_trows, tile_ND2NNOffset>(gBMX, tBMX, nd2nn_offset, k, Nb);
        MGATHER(tBMX, gBMX, nd2nn_offset);  // ND2NN, tile masked

        if(k==0){
          TMATMUL_MX(tACC, tA, tAMX, tB, tBMX);
        }else{
          TMATMUL_MX(tACC, tA, tAMX, tB, tBMX);
        }
      }
      if constexpr (rmd_K) {
        auto gA = gAIter(Mb, Kb);
        auto gB = gBIter(Kb, Nb);

        tile_shapeA_tcorner tA;
        tile_shapeB_tcorner tB;
        TLOAD(tA, gA);
        TLOAD(tB, gB);

        tile_shapeAMX_tcorner tAMX;
        gen_ND2ZZ_offset_Impl<gm_shapeAMX, tile_shapeAMX_tcorner, tile_ND2ZZOffset>(gAMX, tAMX, nd2zz_offset, Mb, Kb);
        MGATHER(tAMX, gAMX, nd2zz_offset);  // ND2ZZ, tile masked
        tile_shapeBMX_tcorner tBMX;
        gen_ND2NN_offset_Impl<gm_shapeBMX, tile_shapeBMX_tcorner, tile_ND2NNOffset>(gBMX, tBMX, nd2nn_offset, Kb, Nb);
        MGATHER(tBMX, gBMX, nd2nn_offset);  // ND2NN, tile masked
        if constexpr(Kb>0){
          TMATMUL_MX(tACC, tA, tAMX, tB, tBMX);
        } else {
          TMATMUL_MX(tACC, tA, tAMX, tB, tBMX);
        }
      }
      store_acc_tile(gC, tACC);
    }
  }
}

template <typename dtypeA, const int gM, const int gN, const int gK, const int tM, const int tN, const int tK,
          typename dtypeB = dtypeA, const int typeb_wfactor = 1, const int smatrix_wfactor=1>
void matmul_mxfp_notcvt(float *dst, dtypeA *src0, dtypeB *src1, uint8_t *src0_mx, uint8_t *src1_mx) {
  static_assert(typeb_wfactor == 1 );
  static const uint32_t valid_row = (tM > gM) ? gM : tM;
  static const uint32_t valid_col = (tN > gN) ? gN : tN;
  using gm_shapeA = global_tensor<dtypeA, RowMajor<gM, gK/typeb_wfactor>>;
  using gm_shapeB = global_tensor<dtypeB, RowMajor<gK/typeb_wfactor, gN>>;
  using gm_shapeC = global_tensor<float, RowMajor<gM, gN>>;

  using tile_shapeA = TileLeft<dtypeA, tM, tK, valid_row, tK/typeb_wfactor>;
  using tile_shapeB = TileRight<dtypeB, tK/typeb_wfactor, tN, tK/typeb_wfactor, valid_col>;
  using tile_shapeACC = TileAcc<float, tM, tN, valid_row, valid_col>;
  using itA = global_iterator<gm_shapeA, tile_shapeA>;
  using itB = global_iterator<gm_shapeB, tile_shapeB>;
  using itC = global_iterator<gm_shapeC, tile_shapeACC>;

  itA gAIter(src0);
  itB gBIter(src1);
  itC gCIter(dst);

  using gm_shapeAMX = global_tensor<uint8_t, RowMajor<gM, gK/smatrix_wfactor>>;
  using tile_shapeAMX = Tile<Location::Scaling, uint8_t, tM, tK, BLayout::RowMajor, valid_row, tK/smatrix_wfactor>; // 实际tile尺寸<tM, tK/32>, 需初始化为0
  using itAMX = global_iterator<gm_shapeAMX, tile_shapeAMX>;
  itAMX gAMXIter(src0_mx);

  using gm_shapeBMX = global_tensor<uint8_t, RowMajor<gK/smatrix_wfactor, gN>>;
  using tile_shapeBMX = Tile<Location::Scaling, uint8_t, tK, tN, BLayout::RowMajor, tK/smatrix_wfactor, valid_col>;
  using itBMX = global_iterator<gm_shapeBMX, tile_shapeBMX>;
  itBMX gBMXIter(src1_mx);

  const int Mb = (gM) / tM;
  const int Nb = (gN) / tN;
  const int Kb = (gK) / tK;

  const int rmd_M = gM % tM;
  const int rmd_N = gN % tN;
  const int rmd_K = gK % tK;

  using tile_shapeA_trows = TileLeft<dtypeA, tM, tK,  valid_row, rmd_K>;
  using tile_shapeA_tcols = TileLeft<dtypeA, tM, tK, rmd_M, tK>;
  using tile_shapeA_tcorner = TileLeft<dtypeA, tM, tK, rmd_M, rmd_K>;

  using tile_shapeAMX_trows = Tile<Location::Scaling, uint8_t, tM, tK, BLayout::RowMajor, valid_row, rmd_K/smatrix_wfactor>;
  using tile_shapeAMX_tcols = Tile<Location::Scaling, uint8_t, tM, tK, BLayout::RowMajor, rmd_M, tK/smatrix_wfactor>;
  using tile_shapeAMX_tcorner = Tile<Location::Scaling, uint8_t, tM, tK, BLayout::RowMajor, rmd_M, rmd_K/smatrix_wfactor>;

  using tile_shapeB_trows = TileRight<dtypeB, tK, tN, tK, rmd_N>;
  using tile_shapeB_tcols = TileRight<dtypeB, tK, tN, rmd_K, valid_col>;
  using tile_shapeB_tcorner = TileRight<dtypeB, tK, tN, rmd_K, rmd_N>;

  using tile_shapeBMX_trows = Tile<Location::Scaling, uint8_t, tK, tN, BLayout::RowMajor, tK/smatrix_wfactor, rmd_N>;
  using tile_shapeBMX_tcols = Tile<Location::Scaling, uint8_t, tK, tN, BLayout::RowMajor, rmd_K/smatrix_wfactor, valid_col>;
  using tile_shapeBMX_tcorner = Tile<Location::Scaling, uint8_t, tK, tN, BLayout::RowMajor, rmd_K/smatrix_wfactor, rmd_N>;

  using tile_shapeC_trows = TileAcc<float, tM, tN, valid_row, rmd_N>;
  using tile_shapeC_tcols = TileAcc<float, tM, tN, rmd_M, valid_col>;
  using tile_shapeC_tcorner = TileAcc<float, tM, tN, rmd_M, rmd_N>;

  // alignas(256) static uint16_t  g_dump_intTile[tM*tK/smatrix_wfactor];
  for(int i=0;i<Mb;i++){
    for(int j=0;j<Nb;j++){
      auto gC = gCIter(i, j);

      tile_shapeACC tACC;
      #pragma clang loop unroll(full)
      for(int k=0;k<Kb;k++){
        auto gA = gAIter(i,k);
        auto gB = gBIter(k,j);
        auto gAMX = gAMXIter(i, k);
        auto gBMX = gBMXIter(k, j);
        tile_shapeA tA;
        tile_shapeB tB;
        tile_shapeAMX tAMX;
        tile_shapeBMX tBMX;
        TLOAD(tA, gA);
        TLOAD(tB, gB);
        TLOAD(tAMX, gAMX);
        TLOAD(tBMX, gBMX);

        if(k==0){
          TMATMUL_MX(tACC, tA, tAMX, tB, tBMX);
        }else{
          TMATMUL_MX(tACC, tA, tAMX, tB, tBMX);
        }
      }
      if constexpr (rmd_K) {
        auto gA = gAIter(i,Kb);
        auto gB = gBIter(Kb,j);
        tile_shapeA_trows tA;
        tile_shapeB_tcols tB;
        TLOAD(tA, gA);
        TLOAD(tB, gB);
        auto gAMX = gAMXIter(i, Kb);
        auto gBMX = gBMXIter(Kb, j);
        tile_shapeAMX_trows tAMX;
        tile_shapeBMX_tcols tBMX;
        TLOAD(tAMX, gAMX);
        TLOAD(tBMX, gBMX);

        if constexpr(Kb>0){
          TMATMUL_MX(tACC, tA, tAMX, tB, tBMX);
        } else {
          TMATMUL_MX(tACC, tA, tAMX, tB, tBMX);
        }
      }
      store_acc_tile(gC, tACC);
    }
    if constexpr (rmd_N) {
      auto gC = gCIter(i, Nb);
      tile_shapeC_trows tACC;
      for(int k=0;k<Kb;k++){
        auto gA = gAIter(i, k);
        auto gB = gBIter(k, Nb);
        tile_shapeA tA;
        tile_shapeB_trows tB;
        TLOAD(tA, gA);
        TLOAD(tB, gB);

        auto gAMX = gAMXIter(i, k);
        auto gBMX = gBMXIter(k, Nb);
        tile_shapeAMX tAMX;
        tile_shapeBMX_trows tBMX;
        TLOAD(tAMX, gAMX);
        TLOAD(tBMX, gBMX);

        if(k==0){
          TMATMUL_MX(tACC, tA, tAMX, tB, tBMX);
        }else{
          TMATMUL_MX(tACC, tA, tAMX, tB, tBMX);
        }
      }

      if constexpr (rmd_K) {
        auto gA = gAIter(i, Kb);
        auto gB = gBIter(Kb, Nb);

        tile_shapeA_trows tA;
        tile_shapeB_tcorner tB;
        TLOAD(tA, gA);
        TLOAD(tB, gB);

        auto gAMX = gAMXIter(i, Kb);
        auto gBMX = gBMXIter(Kb, Nb);
        tile_shapeAMX_trows tAMX;
        tile_shapeBMX_tcorner tBMX;
        TLOAD(tAMX, gAMX);
        TLOAD(tBMX, gBMX);

        if constexpr(Kb>0){
          TMATMUL_MX(tACC, tA, tAMX, tB, tBMX);
        } else {
          TMATMUL_MX(tACC, tA, tAMX, tB, tBMX);
        }
      }
      store_acc_tile(gC, tACC);
    }
  }
  if constexpr (rmd_M) {
    for (int j = 0; j < Nb; ++j) {
      auto gC = gCIter(Mb, j);

      tile_shapeC_tcols tACC;
      #pragma clang loop unroll(full)
      for (int k = 0; k < Kb; ++k) {
        auto gA = gAIter(Mb, k);
        auto gB = gBIter(k, j);

        tile_shapeA_tcols tA;
        tile_shapeB tB;
        TLOAD(tA, gA);
        TLOAD(tB, gB);

        auto gAMX = gAMXIter(Mb, k);
        auto gBMX = gBMXIter(k, j);
        tile_shapeAMX_tcols tAMX;
        tile_shapeBMX tBMX;
        TLOAD(tAMX, gAMX);
        TLOAD(tBMX, gBMX);

        if(k==0){
          TMATMUL_MX(tACC, tA, tAMX, tB, tBMX);
        }else{
          TMATMUL_MX(tACC, tA, tAMX, tB, tBMX);
        }
      }

      if constexpr (rmd_K) {
        auto gA = gAIter(Mb, Kb);
        auto gB = gBIter(Kb, j);

        tile_shapeA_tcorner tA;
        tile_shapeB_tcols tB;
        TLOAD(tA, gA);
        TLOAD(tB, gB);

        auto gAMX = gAMXIter(Mb, Kb);
        auto gBMX = gBMXIter(Kb, j);
        tile_shapeAMX_tcorner tAMX;
        tile_shapeBMX_tcols tBMX;
        TLOAD(tAMX, gAMX);
        TLOAD(tBMX, gBMX);

        if constexpr(Kb>0){
          TMATMUL_MX(tACC, tA, tAMX, tB, tBMX);
        } else {
          TMATMUL_MX(tACC, tA, tAMX, tB, tBMX);
        }
      }
      store_acc_tile(gC, tACC);
    }
    if constexpr (rmd_N) {
      auto gC = gCIter(Mb, Nb);

      tile_shapeC_tcorner tACC;
      #pragma clang loop unroll(full)
      for (int k = 0; k < Kb; ++k) {
        auto gA = gAIter(Mb, k);
        auto gB = gBIter(k, Nb);

        tile_shapeA_tcols tA;
        tile_shapeB_trows tB;
        TLOAD(tA, gA);
        TLOAD(tB, gB);

        auto gAMX = gAMXIter(Mb, k);
        auto gBMX = gBMXIter(k, Nb);
        tile_shapeAMX_tcols tAMX;
        tile_shapeBMX_trows tBMX;
        TLOAD(tAMX, gAMX);
        TLOAD(tBMX, gBMX);

        if(k==0){
          TMATMUL_MX(tACC, tA, tAMX, tB, tBMX);
        }else{
          TMATMUL_MX(tACC, tA, tAMX, tB, tBMX);
        }
      }
      if constexpr (rmd_K) {
        auto gA = gAIter(Mb, Kb);
        auto gB = gBIter(Kb, Nb);

        tile_shapeA_tcorner tA;
        tile_shapeB_tcorner tB;
        TLOAD(tA, gA);
        TLOAD(tB, gB);

        auto gAMX = gAMXIter(Mb, Kb);
        auto gBMX = gBMXIter(Kb, Nb);
        tile_shapeAMX_tcorner tAMX;
        tile_shapeBMX_tcorner tBMX;
        TLOAD(tAMX, gAMX);
        TLOAD(tBMX, gBMX);
        if constexpr(Kb>0){
          TMATMUL_MX(tACC, tA, tAMX, tB, tBMX);
        } else {
          TMATMUL_MX(tACC, tA, tAMX, tB, tBMX);
        }
      }
      store_acc_tile(gC, tACC);
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

template <typename dtypeA, const int gM, const int gN, const int gK, const int tM, const int tN, const int tK,
          typename dtypeB = dtypeA, const int typeb_wfactor = 1, const int smatrix_wfactor=1>
void matmul_mxfp_notcvt_reuseA(float *dst, dtypeA *src0, dtypeB *src1, uint8_t *src0_mx, uint8_t *src1_mx) {
  static_assert(typeb_wfactor == 1 );
  static const uint32_t valid_row = (tM > gM) ? gM : tM;
  static const uint32_t valid_col = (tN > gN) ? gN : tN;
  static const uint32_t MAX_TILE_NUM = 24; // TODO, check this value

  using gm_shapeA = global_tensor<dtypeA, RowMajor<gM, gK/typeb_wfactor>>;
  using gm_shapeB = global_tensor<dtypeB, RowMajor<gK/typeb_wfactor, gN>>;
  using gm_shapeC = global_tensor<float, RowMajor<gM, gN>>;

  using tile_shapeA = TileLeft<dtypeA, tM, tK, valid_row, tK/typeb_wfactor>;
  using tile_shapeB = TileRight<dtypeB, tK/typeb_wfactor, tN, tK/typeb_wfactor, valid_col>;
  using tile_shapeACC = TileAcc<float, tM, tN, valid_row, valid_col>;

  using itA = global_iterator<gm_shapeA, tile_shapeA>;
  using itB = global_iterator<gm_shapeB, tile_shapeB>;
  using itC = global_iterator<gm_shapeC, tile_shapeACC>;

  itA gAIter(src0);
  itB gBIter(src1);
  itC gCIter(dst);

  using gm_shapeAMX = global_tensor<uint8_t, RowMajor<gM, gK/smatrix_wfactor>>;
  using tile_shapeAMX = Tile<Location::Scaling, uint8_t, tM, tK, BLayout::RowMajor, valid_row, tK/smatrix_wfactor>;
  using itAMX = global_iterator<gm_shapeAMX, tile_shapeAMX>;
  itAMX gAMXIter(src0_mx);

  using gm_shapeBMX = global_tensor<uint8_t, RowMajor<gK/smatrix_wfactor, gN>>;
  using tile_shapeBMX = Tile<Location::Scaling, uint8_t, tK, tN, BLayout::RowMajor, tK/smatrix_wfactor, valid_col>;
  using itBMX = global_iterator<gm_shapeBMX, tile_shapeBMX>;
  itBMX gBMXIter(src1_mx);

  const int Mb = (gM) / tM;
  const int Nb = (gN) / tN;
  const int Kb = (gK) / tK;

  const int rmd_M = gM % tM;
  const int rmd_N = gN % tN;
  const int rmd_K = gK % tK;

  using tile_shapeA_trows = TileLeft<dtypeA, tM, tK,  valid_row, rmd_K>;
  using tile_shapeA_tcols = TileLeft<dtypeA, tM, tK, rmd_M, tK>;
  using tile_shapeA_tcorner = TileLeft<dtypeA, tM, tK, rmd_M, rmd_K>;

  using tile_shapeAMX_trows = Tile<Location::Scaling, uint8_t, tM, tK, BLayout::RowMajor, valid_row, rmd_K/smatrix_wfactor>;
  using tile_shapeAMX_tcols = Tile<Location::Scaling, uint8_t, tM, tK, BLayout::RowMajor, rmd_M, tK/smatrix_wfactor>;
  using tile_shapeAMX_tcorner = Tile<Location::Scaling, uint8_t, tM, tK, BLayout::RowMajor, rmd_M, rmd_K/smatrix_wfactor>;

  using tile_shapeB_trows = TileRight<dtypeB, tK, tN, tK, rmd_N>;
  using tile_shapeB_tcols = TileRight<dtypeB, tK, tN, rmd_K, valid_col>;
  using tile_shapeB_tcorner = TileRight<dtypeB, tK, tN, rmd_K, rmd_N>;

  using tile_shapeBMX_trows = Tile<Location::Scaling, uint8_t, tK, tN, BLayout::RowMajor, tK/smatrix_wfactor, rmd_N>;
  using tile_shapeBMX_tcols = Tile<Location::Scaling, uint8_t, tK, tN, BLayout::RowMajor, rmd_K/smatrix_wfactor, valid_col>;
  using tile_shapeBMX_tcorner = Tile<Location::Scaling, uint8_t, tK, tN, BLayout::RowMajor, rmd_K/smatrix_wfactor, rmd_N>;

  using tile_shapeC_trows = TileAcc<float, tM, tN, valid_row, rmd_N>;
  using tile_shapeC_tcols = TileAcc<float, tM, tN, rmd_M, valid_col>;
  using tile_shapeC_tcorner = TileAcc<float, tM, tN, rmd_M, rmd_N>;

  // 计算复用因子
  constexpr ResA R = find_reuseA(Mb < 1 ? 1 : Mb, Kb, MAX_TILE_NUM);
  const int dM = R.m == 0 ? 0 : Mb / R.m;
  const int rM = R.m == 0 ? 0 : Mb % R.m;

  static_assert(R.val <= MAX_TILE_NUM, "R.val is bigger than MAX_TILE_NUM");

  #pragma clang loop unroll(full)
  for(int i=0; i<dM; i++){
    tile_shapeA tA[R.m][R.k];
    tile_shapeAMX tAMX[R.m][R.k];

    #pragma clang loop unroll(full)
    for(int ii=0; ii<R.m; ii++){

      // Copy in all reuse A and AMX sub tiles in K dim
      // #pragma clang loop unroll(full)
      // for(int k=0; k<R.k; k++){
      //   auto gA = gAIter(ii+i*R.m, k);
      //   auto gAMX = gAMXIter(ii+i*R.m, k);
      //   TLOAD(tA[ii][k], gA);
      //   TLOAD(tAMX[ii][k], gAMX);
      // }

      #pragma clang loop unroll(full)
      for(int j=0; j<Nb; j++){
        tile_shapeACC tACC;

        #pragma clang loop unroll(full)
        for(int k=0; k<R.k; k++){
          auto gB = gBIter(k, j);
          auto gBMX = gBMXIter(k, j);
          tile_shapeB tB;
          tile_shapeBMX tBMX;
          TLOAD(tB, gB);
          TLOAD(tBMX, gBMX);
          if(j==0){
            // eliminate head cost
            auto gA = gAIter(ii+i*R.m, k);
            auto gAMX = gAMXIter(ii+i*R.m, k);
            TLOAD(tA[ii][k], gA);
            TLOAD(tAMX[ii][k], gAMX);
          }
          if(k==0){
            TMATMUL_MX(tACC, tA[ii][k], tAMX[ii][k], tB, tBMX);
          }else{
            TMATMUL_MX(tACC, tA[ii][k], tAMX[ii][k], tB, tBMX);
          }
        }

        if constexpr(R.k < Kb){
          for(int k=R.k; k<Kb; k++){
            tile_shapeA tA_tmp;
            tile_shapeAMX tAMX_tmp;
            tile_shapeB tB;
            tile_shapeBMX tBMX;
            auto gA = gAIter(i*R.m+ii, k);
            auto gAMX = gAMXIter(i*R.m+ii, k);
            auto gB = gBIter(k, j);
            auto gBMX = gBMXIter(k, j);
            TLOAD(tA_tmp, gA);
            TLOAD(tAMX_tmp, gAMX);
            TLOAD(tB, gB);
            TLOAD(tBMX, gBMX);
            TMATMUL_MX(tACC, tA_tmp, tAMX_tmp, tB, tBMX);
          }
        }

        // [m, n, rmd_K]
        if constexpr (rmd_K) {
          auto gA = gAIter(i*R.m+ii, Kb);
          auto gAMX = gAMXIter(i*R.m+ii, Kb);
          auto gB = gBIter(Kb, j);
          auto gBMX = gBMXIter(Kb, j);

          tile_shapeA_trows tA_tmp;
          tile_shapeAMX_trows tAMX_tmp;
          tile_shapeB_tcols tB;
          tile_shapeBMX_tcols tBMX;

          TLOAD(tA_tmp, gA);
          TLOAD(tAMX_tmp, gAMX);
          TLOAD(tB, gB);
          TLOAD(tBMX, gBMX);
          if constexpr(Kb>0){
            TMATMUL_MX(tACC, tA_tmp, tAMX_tmp, tB, tBMX);
          } else {
            TMATMUL_MX(tACC, tA_tmp, tAMX_tmp, tB, tBMX);
          }
        }

        auto gC = gCIter(i*R.m+ii, j);
        store_acc_tile(gC, tACC);
      }

      // [m, rmd_N, k]
      if constexpr (rmd_N) {
        tile_shapeC_trows tACC;

        #pragma clang loop unroll(full)
        for(int k=0; k<R.k; k++){
          auto gB = gBIter(k, Nb);
          auto gBMX = gBMXIter(k, Nb);
          tile_shapeB_trows tB;
          tile_shapeBMX_trows tBMX;
          TLOAD(tB, gB);
          TLOAD(tBMX, gBMX);
          if(k==0){
            TMATMUL_MX(tACC, tA[ii][k], tAMX[ii][k], tB, tBMX);
          }else{
            TMATMUL_MX(tACC, tA[ii][k], tAMX[ii][k], tB, tBMX);
          }
        }

        if constexpr(R.k < Kb){
          for(int k=R.k; k<Kb; k++){
            tile_shapeA tA_tmp;
            tile_shapeAMX tAMX_tmp;
            tile_shapeB_trows tB;
            tile_shapeBMX_trows tBMX;
            auto gA = gAIter(i*R.m+ii, k);
            auto gAMX = gAMXIter(i*R.m+ii, k);
            auto gB = gBIter(k, Nb);
            auto gBMX = gBMXIter(k, Nb);
            TLOAD(tA_tmp, gA);
            TLOAD(tAMX_tmp, gAMX);
            TLOAD(tB, gB);
            TLOAD(tBMX, gBMX);
            TMATMUL_MX(tACC, tA_tmp, tAMX_tmp, tB, tBMX);
          }
        }

        // [m, rmd_N, rmd_K]
        if constexpr (rmd_K) {
          auto gA = gAIter(i*R.m+ii, Kb);
          auto gAMX = gAMXIter(i*R.m+ii, Kb);
          auto gB = gBIter(Kb, Nb);
          auto gBMX = gBMXIter(Kb, Nb);

          tile_shapeA_trows tA_tmp;
          tile_shapeAMX_trows tAMX_tmp;
          tile_shapeB_tcorner tB;
          tile_shapeBMX_tcorner tBMX;

          TLOAD(tA_tmp, gA);
          TLOAD(tAMX_tmp, gAMX);
          TLOAD(tB, gB);
          TLOAD(tBMX, gBMX);
          if constexpr(Kb>0){
            TMATMUL_MX(tACC, tA_tmp, tAMX_tmp, tB, tBMX);
          } else {
            TMATMUL_MX(tACC, tA_tmp, tAMX_tmp, tB, tBMX);
          }
        }

        auto gC = gCIter(i*R.m+ii, Nb);
        store_acc_tile(gC, tACC);
      }
    }
  }

  if constexpr(rM>0){
    tile_shapeA tA[rM][R.k];
    tile_shapeAMX tAMX[rM][R.k];

    #pragma clang loop unroll(full)
    for(int i=0; i<rM; i++){

      //copy in remaining M dimension A and AMX tile in K dim
      #pragma clang loop unroll(full)
      for(int k=0; k<R.k; k++){
        auto gA = gAIter(i+dM*R.m, k);
        auto gAMX = gAMXIter(i+dM*R.m, k);
        TLOAD(tA[i][k], gA);
        TLOAD(tAMX[i][k], gAMX);
      }

      #pragma clang loop unroll(full)
      for(int j=0; j<Nb; j++){
        tile_shapeACC tACC;

        #pragma clang loop unroll(full)
        for(int k=0; k<R.k; k++){
          auto gB = gBIter(k, j);
          auto gBMX = gBMXIter(k, j);
          tile_shapeB tB;
          tile_shapeBMX tBMX;
          TLOAD(tB, gB);
          TLOAD(tBMX, gBMX);
          if(j==0){
            auto gA = gAIter(i+dM*R.m, k);
            auto gAMX = gAMXIter(i+dM*R.m, k);
            TLOAD(tA[i][k], gA);
            TLOAD(tAMX[i][k], gAMX);
          }
          if(k==0){
            TMATMUL_MX(tACC, tA[i][k], tAMX[i][k], tB, tBMX);
          }else{
            TMATMUL_MX(tACC, tA[i][k], tAMX[i][k], tB, tBMX);
          }
        }

        if constexpr(R.k < Kb){
          for(int k=R.k; k<Kb; k++){
            tile_shapeA tA_tmp;
            tile_shapeAMX tAMX_tmp;
            tile_shapeB tB;
            tile_shapeBMX tBMX;
            auto gA = gAIter(i+dM*R.m, k);
            auto gAMX = gAMXIter(i+dM*R.m, k);
            auto gB = gBIter(k, j);
            auto gBMX = gBMXIter(k, j);
            TLOAD(tA_tmp, gA);
            TLOAD(tAMX_tmp, gAMX);
            TLOAD(tB, gB);
            TLOAD(tBMX, gBMX);
            TMATMUL_MX(tACC, tA_tmp, tAMX_tmp, tB, tBMX);
          }
        }

        // [rM, n, rmd_K]
        if constexpr (rmd_K) {
          auto gA = gAIter(i+dM*R.m, Kb);
          auto gAMX = gAMXIter(i+dM*R.m, Kb);
          auto gB = gBIter(Kb, j);
          auto gBMX = gBMXIter(Kb, j);

          tile_shapeA_trows tA_tmp;
          tile_shapeAMX_trows tAMX_tmp;
          tile_shapeB_tcols tB;
          tile_shapeBMX_tcols tBMX;

          TLOAD(tA_tmp, gA);
          TLOAD(tAMX_tmp, gAMX);
          TLOAD(tB, gB);
          TLOAD(tBMX, gBMX);
          if constexpr(Kb>0){
            TMATMUL_MX(tACC, tA_tmp, tAMX_tmp, tB, tBMX);
          } else {
            TMATMUL_MX(tACC, tA_tmp, tAMX_tmp, tB, tBMX);
          }
        }
        auto gC = gCIter(i+dM*R.m, j);
        store_acc_tile(gC, tACC);
      }

      // [rM, rmd_N, k]
      if constexpr (rmd_N) {
        tile_shapeC_trows tACC;

        #pragma clang loop unroll(full)
        for(int k=0; k<R.k; k++){
          auto gB = gBIter(k, Nb);
          auto gBMX = gBMXIter(k, Nb);
          tile_shapeB_trows tB;
          tile_shapeBMX_trows tBMX;
          TLOAD(tB, gB);
          TLOAD(tBMX, gBMX);
          if(k==0){
            TMATMUL_MX(tACC, tA[i][k], tAMX[i][k], tB, tBMX);
          }else{
            TMATMUL_MX(tACC, tA[i][k], tAMX[i][k], tB, tBMX);
          }
        }

        if constexpr(R.k < Kb){
          for(int k=R.k; k<Kb; k++){
            tile_shapeA tA_tmp;
            tile_shapeAMX tAMX_tmp;
            tile_shapeB_trows tB;
            tile_shapeBMX_trows tBMX;
            auto gA = gAIter(i+dM*R.m, k);
            auto gAMX = gAMXIter(i+dM*R.m, k);
            auto gB = gBIter(k, Nb);
            auto gBMX = gBMXIter(k, Nb);
            TLOAD(tA_tmp, gA);
            TLOAD(tAMX_tmp, gAMX);
            TLOAD(tB, gB);
            TLOAD(tBMX, gBMX);
            TMATMUL_MX(tACC, tA_tmp, tAMX_tmp, tB, tBMX);
          }
        }

        // [rM, rmd_N, rmd_K]
        if constexpr (rmd_K) {
          auto gA = gAIter(i+dM*R.m, Kb);
          auto gAMX = gAMXIter(i+dM*R.m, Kb);
          auto gB = gBIter(Kb, Nb);
          auto gBMX = gBMXIter(Kb, Nb);

          tile_shapeA_trows tA_tmp;
          tile_shapeAMX_trows tAMX_tmp;
          tile_shapeB_tcorner tB;
          tile_shapeBMX_tcorner tBMX;

          TLOAD(tA_tmp, gA);
          TLOAD(tAMX_tmp, gAMX);
          TLOAD(tB, gB);
          TLOAD(tBMX, gBMX);
          if constexpr(Kb>0){
            TMATMUL_MX(tACC, tA_tmp, tAMX_tmp, tB, tBMX);
          } else {
            TMATMUL_MX(tACC, tA_tmp, tAMX_tmp, tB, tBMX);
          }
        }
        auto gC = gCIter(i+dM*R.m, Nb);
        store_acc_tile(gC, tACC);
      }
    }
  }

  // [rmd_M, n, k]
  if constexpr (rmd_M) {
    tile_shapeA_tcols tA[R.k];
    tile_shapeAMX_tcols tAMX[R.k];

    #pragma clang loop unroll(full)
    for(int k=0; k<R.k; k++){
      auto gA = gAIter(Mb, k);
      auto gAMX = gAMXIter(Mb, k);
      TLOAD(tA[k], gA);
      TLOAD(tAMX[k], gAMX);
    }

    #pragma clang loop unroll(full)
    for(int j=0; j<Nb; j++){
      tile_shapeC_tcols tACC;

      #pragma clang loop unroll(full)
      for(int k=0; k<R.k; k++){
        auto gB = gBIter(k, j);
        auto gBMX = gBMXIter(k, j);
        tile_shapeB tB;
        tile_shapeBMX tBMX;
        TLOAD(tB, gB);
        TLOAD(tBMX, gBMX);
        if(k==0){
          TMATMUL_MX(tACC, tA[k], tAMX[k], tB, tBMX);
        }else{
          TMATMUL_MX(tACC, tA[k], tAMX[k], tB, tBMX);
        }
      }

      if constexpr(R.k < Kb){
        for(int k=R.k; k<Kb; k++){
          tile_shapeA_tcols tA_tmp;
          tile_shapeAMX_tcols tAMX_tmp;
          tile_shapeB tB;
          tile_shapeBMX tBMX;
          auto gA = gAIter(Mb, k);
          auto gAMX = gAMXIter(Mb, k);
          auto gB = gBIter(k, j);
          auto gBMX = gBMXIter(k, j);
          TLOAD(tA_tmp, gA);
          TLOAD(tAMX_tmp, gAMX);
          TLOAD(tB, gB);
          TLOAD(tBMX, gBMX);
          TMATMUL_MX(tACC, tA_tmp, tAMX_tmp, tB, tBMX);
        }
      }

      // [rmd_M, n, rmd_K]
      if constexpr (rmd_K) {
        auto gA = gAIter(Mb, Kb);
        auto gAMX = gAMXIter(Mb, Kb);
        auto gB = gBIter(Kb, j);
        auto gBMX = gBMXIter(Kb, j);

        tile_shapeA_tcorner tA_tmp;
        tile_shapeAMX_tcorner tAMX_tmp;
        tile_shapeB_tcols tB;
        tile_shapeBMX_tcols tBMX;

        TLOAD(tA_tmp, gA);
        TLOAD(tAMX_tmp, gAMX);
        TLOAD(tB, gB);
        TLOAD(tBMX, gBMX);
        if constexpr(Kb>0){
          TMATMUL_MX(tACC, tA_tmp, tAMX_tmp, tB, tBMX);
        } else {
          TMATMUL_MX(tACC, tA_tmp, tAMX_tmp, tB, tBMX);
        }
      }
      auto gC = gCIter(Mb, j);
      store_acc_tile(gC, tACC);
    }

    // [rmd_M, rmd_N, k]
    if constexpr (rmd_N) {
      tile_shapeC_tcorner tACC;

      #pragma clang loop unroll(full)
      for(int k=0; k<R.k; k++){
        auto gB = gBIter(k, Nb);
        auto gBMX = gBMXIter(k, Nb);
        tile_shapeB_trows tB;
        tile_shapeBMX_trows tBMX;
        TLOAD(tB, gB);
        TLOAD(tBMX, gBMX);
        if(k==0){
          TMATMUL_MX(tACC, tA[k], tAMX[k], tB, tBMX);
        }else{
          TMATMUL_MX(tACC, tA[k], tAMX[k], tB, tBMX);
        }
      }

      if constexpr(R.k < Kb){
        for(int k=R.k; k<Kb; k++){
          tile_shapeA_tcols tA_tmp;
          tile_shapeAMX_tcols tAMX_tmp;
          tile_shapeB_trows tB;
          tile_shapeBMX_trows tBMX;
          auto gA = gAIter(Mb, k);
          auto gAMX = gAMXIter(Mb, k);
          auto gB = gBIter(k, Nb);
          auto gBMX = gBMXIter(k, Nb);
          TLOAD(tA_tmp, gA);
          TLOAD(tAMX_tmp, gAMX);
          TLOAD(tB, gB);
          TLOAD(tBMX, gBMX);
          TMATMUL_MX(tACC, tA_tmp, tAMX_tmp, tB, tBMX);
        }
      }

      // [rmd_M, rmd_N, rmd_K]
      if constexpr (rmd_K) {
        auto gA = gAIter(Mb, Kb);
        auto gAMX = gAMXIter(Mb, Kb);
        auto gB = gBIter(Kb, Nb);
        auto gBMX = gBMXIter(Kb, Nb);

        tile_shapeA_tcorner tA_tmp;
        tile_shapeAMX_tcorner tAMX_tmp;
        tile_shapeB_tcorner tB;
        tile_shapeBMX_tcorner tBMX;

        TLOAD(tA_tmp, gA);
        TLOAD(tAMX_tmp, gAMX);
        TLOAD(tB, gB);
        TLOAD(tBMX, gBMX);
        if constexpr(Kb>0){
          TMATMUL_MX(tACC, tA_tmp, tAMX_tmp, tB, tBMX);
        } else {
          TMATMUL_MX(tACC, tA_tmp, tAMX_tmp, tB, tBMX);
        }
      }
      auto gC = gCIter(Mb, Nb);
      store_acc_tile(gC, tACC);
    }
  }
}

// typeb_wfactor 表明typeA和typeB的位宽比例，比如fp8是fp4x2的两倍，
// smatrix_wfactor : scaling matrix 与计算matrix位宽比
template <typename dtypeA, const int gM, const int gN, const int gK, const int tM, const int tN, const int tK,
          typename dtypeB = dtypeA, const int typeb_wfactor = 1, const int smatrix_wfactor=1>
void matmul_mxfp_notcvt_old(float *dst, dtypeA *src0, dtypeB *src1, uint8_t *src0_mx, uint8_t *src1_mx) {
  // only support regular shape now for this operator!
  static_assert(gM % tM == 0);
  static_assert(gN % tN == 0);
  static_assert(gK % tK == 0);
  using gm_shapeA = global_tensor<dtypeA, RowMajor<gM, gK/typeb_wfactor>>;
  using gm_shapeB = global_tensor<dtypeB, RowMajor<gK/typeb_wfactor, gN>>;
  using gm_shapeC = global_tensor<float, RowMajor<gM, gN>>;

  using tile_shapeA = TileLeft<dtypeA, tM, tK/typeb_wfactor>;
  using tile_shapeB = TileRight<dtypeB, tK/typeb_wfactor, tN>;
  using tile_shapeACC = TileAcc<float, tM, tN>;
  using itA = global_iterator<gm_shapeA, tile_shapeA>;
  using itB = global_iterator<gm_shapeB, tile_shapeB>;
  using itC = global_iterator<gm_shapeC, tile_shapeACC>;

  itA gAIter(src0);
  itB gBIter(src1);
  itC gCIter(dst);

  using gm_shapeAMX = global_tensor<uint8_t, RowMajor<gM, gK/smatrix_wfactor>>;
  // gm_shapeAMX gAMX(src0_mx);
  // using tile_shapeAMX = Tile<Location::Scaling, uint8_t, tM, tK/smatrix_wfactor, BLayout::RowMajor, tM, tK/smatrix_wfactor>; // 实际tile尺寸<tM, tK/32>, 需初始化为0
  using tile_shapeAMX = Tile<Location::Scaling, uint8_t, tM, tK, BLayout::RowMajor, tM, tK/smatrix_wfactor>; // 实际tile尺寸<tM, tK/32>, 需初始化为0
  using itAMX = global_iterator<gm_shapeAMX, tile_shapeAMX>;
  // using tile_ND2ZZOffset = Tile<Location::Vec, uint16_t, 1, tM*tK/smatrix_wfactor, BLayout::RowMajor>;
  itAMX gAMXIter(src0_mx);

  using gm_shapeBMX = global_tensor<uint8_t, RowMajor<gK/smatrix_wfactor, gN>>;
  // gm_shapeBMX gBMX(src1_mx);
  // using tile_shapeBMX = Tile<Location::Scaling, uint8_t, tK/smatr_wfactor, tN, BLayout::RowMajor, tK/smatrix_wfactor, tN>; // 8*4B, tload 256B,
  using tile_shapeBMX = Tile<Location::Scaling, uint8_t, tK, tN, BLayout::RowMajor, tK/smatrix_wfactor, tN>; // 8*4B, tload 256B,
  using itBMX = global_iterator<gm_shapeBMX, tile_shapeBMX>;
  // using tile_ND2NNOffset = Tile<Location::Vec, uint16_t, 1, tK/smatrix_wfactor*tN, BLayout::RowMajor>;
  // tile_ND2NNOffset nd2nn_offset;
  itBMX gBMXIter(src1_mx);

  const int Mb = (gM) / tM;
  const int Nb = (gN) / tN;
  const int Kb = (gK) / tK;

  alignas(256) static uint16_t  g_dump_intTile[tM*tK/smatrix_wfactor];
  for(int i=0;i<Mb;i++){
    for(int j=0;j<Nb;j++){
        auto gC = gCIter(i, j);

        tile_shapeACC tACC;
        #pragma clang loop unroll(full)
        for(int k=0;k<Kb;k++){
          auto gA = gAIter(i,k);
          auto gB = gBIter(k,j);
          auto gAMX = gAMXIter(i, k);
          auto gBMX = gBMXIter(k, j);
          tile_shapeA tA;
          tile_shapeB tB;
          tile_shapeAMX tAMX;
          tile_shapeBMX tBMX;
          TLOAD(tA, gA);
          TLOAD(tB, gB);
          TLOAD(tAMX, gAMX);
          TLOAD(tBMX, gBMX);

            if(k==0){
              TMATMUL_MX(tACC, tA, tAMX, tB, tBMX);
              // TMATMUL(tACC, tA, tB);
            }else{
              TMATMUL_MX(tACC, tA, tAMX, tB, tBMX);
              // TMATMUL_ACC(tACC, tA, tB);
            }
        }
        store_acc_tile(gC, tACC);
    }
  }
}

// typeb_wfactor 表明typeA和typeB的位宽比例，比如fp8是fp4x2的两倍，
// smatrix_wfactor : scaling matrix 与计算matrix位宽比
template <typename dtypeA, const int gM, const int gN, const int gK, const int tM, const int tN, const int tK,
          typename dtypeB = dtypeA, const int typeb_wfactor = 1, const int smatrix_wfactor=1>
void matmul_fp_notcvt(float *dst, dtypeA *src0, dtypeB *src1, uint8_t *src0_mx, uint8_t *src1_mx) {
  // only support regular shape now for this operator!
  static_assert(gM % tM == 0);
  static_assert(gN % tN == 0);
  static_assert(gK % tK == 0);
  using gm_shapeA = global_tensor<dtypeA, RowMajor<gM, gK/typeb_wfactor>>;
  using gm_shapeB = global_tensor<dtypeB, RowMajor<gK/typeb_wfactor, gN>>;
  using gm_shapeC = global_tensor<float, RowMajor<gM, gN>>;

  using tile_shapeA = TileLeft<dtypeA, tM, tK/typeb_wfactor>;
  using tile_shapeB = TileRight<dtypeB, tK/typeb_wfactor, tN>;
  using tile_shapeACC = TileAcc<float, tM, tN>;
  using itA = global_iterator<gm_shapeA, tile_shapeA>;
  using itB = global_iterator<gm_shapeB, tile_shapeB>;
  using itC = global_iterator<gm_shapeC, tile_shapeACC>;

  itA gAIter(src0);
  itB gBIter(src1);
  itC gCIter(dst);

  using gm_shapeAMX = global_tensor<uint8_t, RowMajor<gM, gK/smatrix_wfactor>>;
  // gm_shapeAMX gAMX(src0_mx);
  // using tile_shapeAMX = Tile<Location::Scaling, uint8_t, tM, tK/smatrix_wfactor, BLayout::RowMajor, tM, tK/smatrix_wfactor>; // 实际tile尺寸<tM, tK/32>, 需初始化为0
  using tile_shapeAMX = Tile<Location::Scaling, uint8_t, tM, tK, BLayout::RowMajor, tM, tK/smatrix_wfactor>; // 实际tile尺寸<tM, tK/32>, 需初始化为0
  using itAMX = global_iterator<gm_shapeAMX, tile_shapeAMX>;
  // using tile_ND2ZZOffset = Tile<Location::Vec, uint16_t, 1, tM*tK/smatrix_wfactor, BLayout::RowMajor>;
  itAMX gAMXIter(src0_mx);

  using gm_shapeBMX = global_tensor<uint8_t, RowMajor<gK/smatrix_wfactor, gN>>;
  // gm_shapeBMX gBMX(src1_mx);
  // using tile_shapeBMX = Tile<Location::Scaling, uint8_t, tK/smatr_wfactor, tN, BLayout::RowMajor, tK/smatrix_wfactor, tN>; // 8*4B, tload 256B,
  using tile_shapeBMX = Tile<Location::Scaling, uint8_t, tK, tN, BLayout::RowMajor, tK/smatrix_wfactor, tN>; // 8*4B, tload 256B,
  using itBMX = global_iterator<gm_shapeBMX, tile_shapeBMX>;
  // using tile_ND2NNOffset = Tile<Location::Vec, uint16_t, 1, tK/smatrix_wfactor*tN, BLayout::RowMajor>;
  // tile_ND2NNOffset nd2nn_offset;
  itBMX gBMXIter(src1_mx);

  const int Mb = (gM) / tM;
  const int Nb = (gN) / tN;
  const int Kb = (gK) / tK;

  alignas(256) static uint16_t  g_dump_intTile[tM*tK/smatrix_wfactor];
  for(int i=0;i<Mb;i++){
    for(int j=0;j<Nb;j++){
        auto gC = gCIter(i, j);

        tile_shapeACC tACC;
        #pragma clang loop unroll(full)
        for(int k=0;k<Kb;k++){
          auto gA = gAIter(i,k);
          auto gB = gBIter(k,j);
          auto gAMX = gAMXIter(i, k);
          auto gBMX = gBMXIter(k, j);
          tile_shapeA tA;
          tile_shapeB tB;
          tile_shapeAMX tAMX;
          tile_shapeBMX tBMX;
          TLOAD(tA, gA);
          TLOAD(tB, gB);
          TLOAD(tAMX, gAMX);
          TLOAD(tBMX, gBMX);

            if(k==0){
              TMATMUL(tACC, tA, tB);
            }else{
              // TMATMUL_MX(tACC, tA, tAMX, tB, tBMX);
              TMATMUL_ACC(tACC, tA, tB);
            }
        }
        store_acc_tile(gC, tACC);
    }
  }
}

template<is_tile_data_v PsumTile, is_tile_data_v ScaleTile, is_tile_data_v OutTile>
void dequant_acc_pto(OutTile &out, PsumTile &data, ScaleTile &scale, OutTile &adder) {
    OutTile scaled;
    TCOLEXPANDMUL(scaled, data, scale);
    TADD(out, scaled, adder);
}

// mixed precision matmul dequant(A*B)， fp4x2 width_factor: 2
template <typename dtypeA, typename dtypeB, const int gM, const int gN, const int gK, const int tM, const int tN, const int tK, const int width_factor>
void matmul_mp(float *acc_ptr, dtypeA *a_ptr, dtypeB *b_ptr, float *c_ptr) {
  static_assert(gK % (128) == 0);
  static_assert(tK == 128);
  static const uint32_t trow = (tM >= 16) ? tM : 16;
  static const uint32_t tcol = (tN >= 16) ? tN : 16;
  using gm_shapeA = global_tensor<dtypeA, RowMajor<gM, gK>>;
  using gm_shapeB = global_tensor<dtypeB, RowMajor<gK/width_factor, gN>>;
  // 伪量化固定float, group 大小128， 128个fp4共享一个scaling factor, 128的partial sum* scale
  using gm_shape_scale = global_tensor<float, RowMajor<gK/128, gN>>;
  using gm_shapeACC = global_tensor<float, RowMajor<gM, gN>>;
  using tile_shapeA = TileLeft<dtypeA, trow, tK, tM, tK>;
  using tile_shapeB = TileRight<dtypeB, tK/width_factor, tcol, tK/width_factor, tcol>;
  using tile_shape_scale = Tile<Location::Vec, float, tK/128, tcol, BLayout::RowMajor, tK/128, tN>;
  using tile_shape_dequant = Tile<Location::Vec, float, trow, tcol, BLayout::RowMajor, tM, tN>;
  using tile_shapeACC = TileAcc<float, trow, tcol, tM, tN>;
  // copy of acc, input as vector
  using tile_ACCin = Tile<Location::Vec, float, trow, tcol, BLayout::ColMajor, tM, tN>;

  using itA = global_iterator<gm_shapeA, tile_shapeA>;
  using itB = global_iterator<gm_shapeB, tile_shapeB>;
  using itScale = global_iterator<gm_shape_scale, tile_shape_scale>;
  using itACC = global_iterator<gm_shapeACC, tile_shapeACC>;

  itA gAIter(a_ptr);
  itB gBIter(b_ptr);
  itScale gScaleIter(c_ptr);
  itACC gACCIter(acc_ptr);

  const int Mb = gM / tM;
  const int Nb = gN / tN;
  const int Kb = gK / tK;

  const int rmd_M = gM % tM;
  const int rmd_N = gN % tN;
  const int rmd_K = gK % tK;

  using tile_shapeA_trows = TileLeft<dtypeA, trow, tK,  tM, rmd_K>;
  using tile_shapeA_tcols = TileLeft<dtypeA, trow, tK, rmd_M, tK>;
  using tile_shapeA_tcorner = TileLeft<dtypeA, trow, tK, rmd_M, rmd_K>;

  using tile_shapeB_trows = TileRight<dtypeB, tK, tcol, tK, rmd_N>;
  using tile_shapeB_tcols = TileRight<dtypeB, tK, tcol, rmd_K, tN>;
  using tile_shapeB_tcorner = TileRight<dtypeB, tK, tcol, rmd_K, rmd_N>;

  using tile_shapeC_trows = TileAcc<float, trow, tcol, tM, rmd_N>;
  using tile_shapeC_tcols = TileAcc<float, trow, tcol, rmd_M, tN>;
  using tile_shapeC_tcorner = TileAcc<float, trow, tcol, rmd_M, rmd_N>;

  using tile_shape_scale_trows = Tile<Location::Scaling, float, tK/128, tcol, BLayout::RowMajor, tK/128, rmd_N, SLayout::NoneBox>;
  using tile_shape_scale_tcols = Tile<Location::Scaling, float, tK/128, tcol, BLayout::RowMajor, rmd_K/128, tN, SLayout::NoneBox>;
  using tile_shape_scale_tcorner = Tile<Location::Scaling, float, tK/128, tcol, BLayout::RowMajor, rmd_K/128, rmd_N, SLayout::NoneBox>;

  using tile_ACCin_trows = Tile<Location::Vec, float, trow, tcol, BLayout::RowMajor, tM, rmd_N>;
  using tile_ACCin_tcols = Tile<Location::Vec, float, trow, tcol, BLayout::RowMajor, rmd_M, tN>;
  using tile_ACCin_tconer = Tile<Location::Vec, float, trow, tcol, BLayout::RowMajor, rmd_M, rmd_N>;

  using tile_shape_dequant_trows = Tile<Location::Vec, float, trow, tcol, BLayout::RowMajor, tM, rmd_N>;
  using tile_shape_dequant_tcols = Tile<Location::Vec, float, trow, tcol, BLayout::RowMajor, rmd_M, tN>;
  using tile_shape_dequant_tcorner = Tile<Location::Vec, float, trow, tcol, BLayout::RowMajor, rmd_M, rmd_N>;
  for(int i=0;i<Mb;i++){
    for(int j=0;j<Nb;j++){
      auto gACC = gACCIter(i, j);

      tile_shapeACC tACC;
      tile_ACCin tACCin;
      tile_shape_dequant tAdder[2];
      TEXPANDS(tAdder[0], 0.);
      int k=0;
      #pragma clang loop unroll(full)
      for(;k<Kb;k++){
        auto gA = gAIter(i,k);
        auto gB = gBIter(k,j);
        auto gS = gScaleIter(k,j);
        tile_shapeA tA;
        tile_shapeB tB;
        tile_shape_scale ts;
        tile_shape_dequant tC_dequant;
        TLOAD(tA, gA);
        TLOAD(tB, gB);
        TLOAD(ts, gS); // [1, tN]

        TMATMUL(tACC, tA, tB);
        TCVT(tACCin, tACC);//[tM, tN] 256->1 , 256 -> 2 scaling factor
        // static_assert(tile_shapeB::ValidCol % (width_factor*128) == 0); // TODO, 暂不考虑padding，假设形状是规整的, 方便处理, taccin*ts_adder=tc_dequant
        dequant_acc_pto(tC_dequant, tACCin, ts, tAdder[k%2]);
        tAdder[(k+1)%2] = tC_dequant;
      }
      k++;
      if constexpr (rmd_K) {
        // to do check acc shape
        auto gA = gAIter(i, Kb);
        auto gB = gBIter(Kb, j);
        auto gS = gScaleIter(Kb,j);
        tile_shapeA_trows tA;
        tile_shapeB_tcols tB;
        tile_shape_scale_tcols ts;
        tile_shape_dequant tC_dequant;

        TLOAD(tA, gA);
        TLOAD(tB, gB);
        TLOAD(ts, gS);

        TMATMUL(tACC, tA, tB);
        TCVT(tACCin, tACC);
        dequant_acc_pto(tC_dequant, tACCin, ts, tAdder[k%2]);
        tAdder[(k+1)%2] = tC_dequant;
      }
      TSTORE(gACC, tAdder[(k+1)%2]);
    }
    // if constexpr (rmd_N) // TODO
  }
  if constexpr (rmd_M) {
    for (int j = 0; j < Nb; ++j) {
      auto gACC = gACCIter(Mb, j);
      tile_shapeC_tcols tACC;
      tile_ACCin_tcols tACCin;
      tile_shape_dequant_tcols tAdder[2];
      TEXPANDS(tAdder[0], 0.);
      int k = 0;
      #pragma clang loop unroll(full)
      for (; k < Kb; ++k) {
        auto gA = gAIter(Mb, k);
        auto gB = gBIter(k, j);
        auto gS = gScaleIter(k,j);

        tile_shapeA_tcols tA;
        tile_shapeB tB;
        tile_shape_scale ts;
        tile_shape_dequant_tcols tC_dequant;
        TLOAD(tA, gA);
        TLOAD(tB, gB);
        TLOAD(ts, gS);
        TMATMUL(tACC, tA, tB);
        TCVT(tACCin, tACC);
        dequant_acc_pto(tC_dequant, tACCin, ts, tAdder[k%2]);
        tAdder[(k+1)%2] = tC_dequant;
      }
      k++;
      if constexpr (rmd_K) {
        auto gA = gAIter(Mb, Kb);
        auto gB = gBIter(Kb, j);
        auto gS = gScaleIter(Kb, j);

        tile_shapeA_tcorner tA;
        tile_shapeB_tcols tB;
        tile_shape_scale_tcols ts;
        tile_shape_dequant tC_dequant;
        TLOAD(tA, gA);
        TLOAD(tB, gB);
        TLOAD(ts, gS);
        TMATMUL(tACC, tA, tB);
        TCVT(tACCin, tACC);
        dequant_acc_pto(tC_dequant, tACCin, ts, tAdder[k%2]);
        tAdder[(k+1)%2] = tC_dequant;
      }
      TSTORE(gACC, tAdder[(k+1)%2]);
    }
    // todo
    // if constexpr (rmd_N) {
    //   auto gC = gCIter(Mb, Nb);

    //   tile_shapeC_tcorner tACC;
    //   if constexpr(Kb>0){
    //     auto gA = gAIter(Mb, 0);
    //     auto gB = gBIter(0, Nb);

    //     tile_shapeA_tcols tA;
    //     tile_shapeB_trows tB;
    //     TLOAD(tA, gA);
    //     TLOAD(tB, gB);
    //     TMATMUL(tACC, tA, tB);
    //   }
    //   #pragma clang loop unroll(full)
    //   for (int k = 1; k < Kb; ++k) {
    //     auto gA = gAIter(Mb, k);
    //     auto gB = gBIter(k, Nb);

    //     tile_shapeA_tcols tA;
    //     tile_shapeB_trows tB;
    //     TLOAD(tA, gA);
    //     TLOAD(tB, gB);
    //     TMATMUL_ACC(tACC, tA, tB);
    //   }
    //   if constexpr (rmd_K) {
    //     auto gA = gAIter(Mb, Kb);
    //     auto gB = gBIter(Kb, Nb);

    //     tile_shapeA_tcorner tA;
    //     tile_shapeB_tcorner tB;
    //     TLOAD(tA, gA);
    //     TLOAD(tB, gB);
    //     if constexpr(Kb>0){
    //       TMATMUL_ACC(tACC, tA, tB);
    //     } else {
    //       TMATMUL(tACC, tA, tB);
    //     }
    //   }
    //   store_acc_tile(gC, tACC);
    // }
  }
}


#endif
