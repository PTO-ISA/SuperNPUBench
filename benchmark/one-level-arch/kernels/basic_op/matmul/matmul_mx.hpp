#ifndef MATMUL_MX_HPP
#define MATMUL_MX_HPP

#include <common/pto_tileop.hpp>
#include <basic_op/utils/layout_transform.hpp>
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

template <typename E_, int R_, int C_, int VR_=R_, int VC_=C_>
using CubeTileA = std::conditional_t<
    (R_ <= 16), CubeTileM16<E_, R_, C_, VR_, VC_>,
    CubeTileM32<E_, R_, C_, VR_, VC_>>;

template <typename E_, int R_, int C_, int VR_=R_, int VC_=C_>
using TileAcc = std::conditional_t<
    (R_ <= 16), CubeAccumulatorM16<E_, R_, C_, VR_, VC_>,
    CubeAccumulatorM32<E_, R_, C_, VR_, VC_>>;

// GM scales retain A [M,K/64] and B [K/64,N]. Local scale
// descriptors are CUBE_M32 [M,K/64] and [N,K/64]. A native scale
// descriptor has at most 32 rows, so each requested N tile consists of
// 32-column operations; their total B, scale-B and result capacities equal
// the original requested tile capacities.
template <const int gM, const int gN, const int gK, const int tM,
          const int tN, const int tK, bool ReuseA = false>
void matmul_hif4x2_mx(float *dst, __fp4_hif4x2 *src0,
                      __fp4_hif4x2 *src1, uint32_t *src0_mx,
                      uint32_t *src1_mx) {
  static_assert(gM % tM == 0 && gN % tN == 0 && gK % tK == 0,
                "HiF4 single-thread matmul currently requires full tiles");
  static_assert(tM == 16 || tM == 32);
  static_assert(gK % 64 == 0 && tK % 64 == 0);
  static_assert(tN >= 32 && tN % 32 == 0,
                "HiF4 Local scale-B uses complete 32-column native operations");
  constexpr uint32_t NativeN = 32;
  constexpr uint32_t PartsN = tN / NativeN;
  constexpr uint32_t ScaleK = tK / 64;
  constexpr uint32_t GlobalScaleK = gK / 64;
  constexpr int Mb = gM / tM, Nb = gN / tN, Kb = gK / tK;
  using A = CubeTileA<__fp4_hif4x2, tM, tK>;
  using B = CubeTileN8<__fp4_hif4x2, tK, NativeN>;
  using C = TileAcc<float, tM, NativeN>;
  using AScale = Tile<Location::Scaling, uint32_t, 32, ScaleK,
                      BLayout::CubeM32, tM, ScaleK>;
  using BScale = Tile<Location::Scaling, uint32_t, 32, ScaleK,
                      BLayout::CubeM32>;
  using ByteOffsets = ElementTile<uint64_t, 32 * ScaleK>;
  static_assert(B::LogicalTileBytes * PartsN == tK * tN / 2);
  static_assert(BScale::LogicalTileBytes * PartsN == ScaleK * tN * 4);
  static_assert(C::LogicalTileBytes * PartsN == tM * tN * 4);

  static_assert(uint64_t(ScaleK) * gN * sizeof(uint32_t) <= 0x7fffffff,
                "scale gather offsets must have a proved signed-B32 lane range");
  TileArray<ElementTile<uint64_t, 32>, 1, ScaleK> offset_parts;
  for (uint32_t group = 0; group < ScaleK; ++group) {
    const uint32_t group_base = group * uint32_t(gN);
    ElementTile<uint64_t, 32> part;
    auto &offsets = TPARTELEMENT(part);
#pragma pto element for
    for (uint32_t element = 0; element < 32; ++element) {
      const uint32_t source_lane = group_base + element;
      const uint64_t byte_offset =
          uint64_t(source_lane) * uint64_t(sizeof(uint32_t));
      offsets[element] = byte_offset;
    }
    TMOV(offset_parts[0][group], part);
  }
  auto scale_offsets = TASSEMBLY<ByteOffsets>(std::move(offset_parts));
  auto load_a = [&](A &a, AScale &scale, int i, int k) {
    global_tensor<__fp4_hif4x2, MatrixLayout<tM, tK, gK, 1>>
        ga(src0 + (i * tM * gK + k * tK) / 2);
    global_tensor<uint32_t, MatrixLayout<tM, ScaleK, GlobalScaleK, 1>>
        gs(src0_mx + i * tM * GlobalScaleK + k * ScaleK);
    TLOAD_CUBE(a, ga);
    TLOAD_CUBE(scale, gs);
  };
  auto load_b = [&](B &b, BScale &scale, int j, int part, int k) {
    const uint64_t column = uint64_t(j) * tN + uint64_t(part) * NativeN;
    global_tensor<__fp4_hif4x2, MatrixLayout<tK, NativeN, gN, 1>>
        gb(src1 + (uint64_t(k) * tK * gN + column) / 2);
    global_tensor<uint32_t, RowMajor<ScaleK, gN>>
        gs(src1_mx + uint64_t(k) * ScaleK * gN + column);
    TLOAD_CUBE(b, gb);
    // Each byte offset selects GM[group, column] into Local[column, group].
    MGATHER(scale, gs, scale_offsets);
  };
#pragma clang loop unroll(full)
  for (int i = 0; i < Mb; ++i) {
    if constexpr (ReuseA) {
      A cached_a[Kb];
      AScale cached_scale[Kb];
#pragma clang loop unroll(full)
      for (int k = 0; k < Kb; ++k)
        load_a(cached_a[k], cached_scale[k], i, k);
#pragma clang loop unroll(full)
      for (int j = 0; j < Nb; ++j) {
        C result[PartsN];
#pragma clang loop unroll(full)
        for (int k = 0; k < Kb; ++k) {
          B b[PartsN];
          BScale scale[PartsN];
          for (uint32_t part = 0; part < PartsN; ++part) {
            load_b(b[part], scale[part], j, part, k);
            if (k == 0)
              TMATMUL_MX(result[part], cached_a[k], cached_scale[k], b[part], scale[part], fixp::keep_acc());
            else
              TMATMUL_MX_ACC(result[part], result[part], cached_a[k], cached_scale[k], b[part], scale[part], fixp::keep_acc());
          }
        }
        for (uint32_t part = 0; part < PartsN; ++part) {
          global_tensor<float, MatrixLayout<tM, NativeN, gN, 1>>
              gc(dst + uint64_t(i) * tM * gN + uint64_t(j) * tN + part * NativeN);
          TSTORE_CUBE(gc, result[part]);
        }
      }
    } else {
#pragma clang loop unroll(full)
      for (int j = 0; j < Nb; ++j) {
        C result[PartsN];
#pragma clang loop unroll(full)
        for (int k = 0; k < Kb; ++k) {
          A a;
          AScale a_scale;
          B b[PartsN];
          BScale b_scale[PartsN];
          load_a(a, a_scale, i, k);
          for (uint32_t part = 0; part < PartsN; ++part) {
            load_b(b[part], b_scale[part], j, part, k);
            if (k == 0)
              TMATMUL_MX(result[part], a, a_scale, b[part], b_scale[part], fixp::keep_acc());
            else
              TMATMUL_MX_ACC(result[part], result[part], a, a_scale, b[part], b_scale[part], fixp::keep_acc());
          }
        }
        for (uint32_t part = 0; part < PartsN; ++part) {
          global_tensor<float, MatrixLayout<tM, NativeN, gN, 1>>
              gc(dst + uint64_t(i) * tM * gN + uint64_t(j) * tN + part * NativeN);
          TSTORE_CUBE(gc, result[part]);
        }
      }
    }
  }
}

// TODO, move to utils.cpp
template <is_global_data_v GmOut, is_tile_data_v TileAcc>
void store_acc_tile(GmOut &Gout, TileAcc &tAcc){
    TSTORE(Gout, tAcc);
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

  using tile_shapeA = CubeTileA<dtypeA, tM, tK, valid_row, tK>;
  using tile_shapeB = CubeTileN8<dtypeB, tK/typeb_wfactor, tN, tK/typeb_wfactor, valid_col>;
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

  using tile_shapeA_trows = CubeTileA<dtypeA, tM, tK,  valid_row, rmd_K>;
  using tile_shapeA_tcols = CubeTileA<dtypeA, tM, tK, rmd_M, tK>;
  using tile_shapeA_tcorner = CubeTileA<dtypeA, tM, tK, rmd_M, rmd_K>;

  using tile_shapeAMX_trows = Tile<Location::Scaling, uint8_t, tM, tK, BLayout::RowMajor, valid_row, rmd_K/smatrix_wfactor, SLayout::RowMajor>;
  using tile_shapeAMX_tcols = Tile<Location::Scaling, uint8_t, tM, tK, BLayout::RowMajor, rmd_M, tK/smatrix_wfactor, SLayout::RowMajor>;
  using tile_shapeAMX_tcorner = Tile<Location::Scaling, uint8_t, tM, tK, BLayout::RowMajor, rmd_M, rmd_K/smatrix_wfactor, SLayout::RowMajor>;

  using tile_shapeB_trows = CubeTileN8<dtypeB, tK, tN, tK, rmd_N>;
  using tile_shapeB_tcols = CubeTileN8<dtypeB, tK, tN, rmd_K, valid_col>;
  using tile_shapeB_tcorner = CubeTileN8<dtypeB, tK, tN, rmd_K, rmd_N>;

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
  using scale_dtype = __fp8_e8m0;
  static const uint32_t valid_row = (tM > gM) ? gM : tM;
  static const uint32_t valid_col = (tN > gN) ? gN : tN;
  using gm_shapeA = global_tensor<dtypeA, RowMajor<gM, gK/typeb_wfactor>>;
  using gm_shapeB = global_tensor<dtypeB, RowMajor<gK/typeb_wfactor, gN>>;
  using gm_shapeC = global_tensor<float, RowMajor<gM, gN>>;

  using tile_shapeA = CubeTileA<dtypeA, tM, tK, valid_row, tK/typeb_wfactor>;
  using tile_shapeB = CubeTileN8<dtypeB, tK/typeb_wfactor, tN, tK/typeb_wfactor, valid_col>;
  using tile_shapeACC = TileAcc<float, tM, tN, valid_row, valid_col>;
  using itA = global_iterator<gm_shapeA, tile_shapeA>;
  using itB = global_iterator<gm_shapeB, tile_shapeB>;
  using itC = global_iterator<gm_shapeC, tile_shapeACC>;

  itA gAIter(src0);
  itB gBIter(src1);
  itC gCIter(dst);

  using gm_shapeAMX = global_tensor<scale_dtype, RowMajor<gM, gK/smatrix_wfactor>>;
  using tile_shapeAMX = Tile<Location::Scaling, scale_dtype, tM, tK, BLayout::RowMajor, valid_row, tK/smatrix_wfactor>;
  using itAMX = global_iterator<gm_shapeAMX, tile_shapeAMX>;
  itAMX gAMXIter(reinterpret_cast<scale_dtype *>(src0_mx));

  using gm_shapeBMX = global_tensor<scale_dtype, RowMajor<gK/smatrix_wfactor, gN>>;
  using tile_shapeBMX = Tile<Location::Scaling, scale_dtype, tK, tN, BLayout::RowMajor, tK/smatrix_wfactor, valid_col>;
  using itBMX = global_iterator<gm_shapeBMX, tile_shapeBMX>;
  itBMX gBMXIter(reinterpret_cast<scale_dtype *>(src1_mx));

  const int Mb = (gM) / tM;
  const int Nb = (gN) / tN;
  const int Kb = (gK) / tK;

  const int rmd_M = gM % tM;
  const int rmd_N = gN % tN;
  const int rmd_K = gK % tK;

  using tile_shapeA_trows = CubeTileA<dtypeA, tM, tK,  valid_row, rmd_K>;
  using tile_shapeA_tcols = CubeTileA<dtypeA, tM, tK, rmd_M, tK>;
  using tile_shapeA_tcorner = CubeTileA<dtypeA, tM, tK, rmd_M, rmd_K>;

  using tile_shapeAMX_trows = Tile<Location::Scaling, scale_dtype, tM, tK, BLayout::RowMajor, valid_row, rmd_K/smatrix_wfactor>;
  using tile_shapeAMX_tcols = Tile<Location::Scaling, scale_dtype, tM, tK, BLayout::RowMajor, rmd_M, tK/smatrix_wfactor>;
  using tile_shapeAMX_tcorner = Tile<Location::Scaling, scale_dtype, tM, tK, BLayout::RowMajor, rmd_M, rmd_K/smatrix_wfactor>;

  using tile_shapeB_trows = CubeTileN8<dtypeB, tK, tN, tK, rmd_N>;
  using tile_shapeB_tcols = CubeTileN8<dtypeB, tK, tN, rmd_K, valid_col>;
  using tile_shapeB_tcorner = CubeTileN8<dtypeB, tK, tN, rmd_K, rmd_N>;

  using tile_shapeBMX_trows = Tile<Location::Scaling, scale_dtype, tK, tN, BLayout::RowMajor, tK/smatrix_wfactor, rmd_N>;
  using tile_shapeBMX_tcols = Tile<Location::Scaling, scale_dtype, tK, tN, BLayout::RowMajor, rmd_K/smatrix_wfactor, valid_col>;
  using tile_shapeBMX_tcorner = Tile<Location::Scaling, scale_dtype, tK, tN, BLayout::RowMajor, rmd_K/smatrix_wfactor, rmd_N>;

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
  using scale_dtype = __fp8_e8m0;
  static const uint32_t valid_row = (tM > gM) ? gM : tM;
  static const uint32_t valid_col = (tN > gN) ? gN : tN;

  using gm_shapeA = global_tensor<dtypeA, RowMajor<gM, gK/typeb_wfactor>>;
  using gm_shapeB = global_tensor<dtypeB, RowMajor<gK/typeb_wfactor, gN>>;
  using gm_shapeC = global_tensor<float, RowMajor<gM, gN>>;

  using tile_shapeA = CubeTileA<dtypeA, tM, tK, valid_row, tK/typeb_wfactor>;
  using tile_shapeB = CubeTileN8<dtypeB, tK/typeb_wfactor, tN, tK/typeb_wfactor, valid_col>;
  using tile_shapeACC = TileAcc<float, tM, tN, valid_row, valid_col>;

  using itA = global_iterator<gm_shapeA, tile_shapeA>;
  using itB = global_iterator<gm_shapeB, tile_shapeB>;
  using itC = global_iterator<gm_shapeC, tile_shapeACC>;

  itA gAIter(src0);
  itB gBIter(src1);
  itC gCIter(dst);

  using gm_shapeAMX = global_tensor<scale_dtype, RowMajor<gM, gK/smatrix_wfactor>>;
  using tile_shapeAMX = Tile<Location::Scaling, scale_dtype, tM, tK, BLayout::RowMajor, valid_row, tK/smatrix_wfactor>;
  using itAMX = global_iterator<gm_shapeAMX, tile_shapeAMX>;
  itAMX gAMXIter(reinterpret_cast<scale_dtype *>(src0_mx));

  using gm_shapeBMX = global_tensor<scale_dtype, RowMajor<gK/smatrix_wfactor, gN>>;
  using tile_shapeBMX = Tile<Location::Scaling, scale_dtype, tK, tN, BLayout::RowMajor, tK/smatrix_wfactor, valid_col>;
  using itBMX = global_iterator<gm_shapeBMX, tile_shapeBMX>;
  itBMX gBMXIter(reinterpret_cast<scale_dtype *>(src1_mx));

  // PTO v0.58 persistent Local CUBE descriptors use a 256 KiB capacity
  // ceiling. Keep the reuse planner consistent with the active TileOP limit.
  constexpr int kLocalTileBytes = 256 * 1024;
  constexpr int kResidentBytes = tile_shapeACC::LogicalTileBytes +
                                 tile_shapeB::LogicalTileBytes +
                                 tile_shapeBMX::LogicalTileBytes;
  constexpr int kReusePairBytes = tile_shapeA::LogicalTileBytes +
                                  tile_shapeAMX::LogicalTileBytes;
  constexpr int MAX_TILE_NUM =
      (kLocalTileBytes - kResidentBytes) / kReusePairBytes;
  static_assert(MAX_TILE_NUM > 0,
                "Local tile capacity cannot hold MX C/B and one A/scale pair");

  const int Mb = (gM) / tM;
  const int Nb = (gN) / tN;
  const int Kb = (gK) / tK;

  const int rmd_M = gM % tM;
  const int rmd_N = gN % tN;
  const int rmd_K = gK % tK;

  using tile_shapeA_trows = CubeTileA<dtypeA, tM, tK,  valid_row, rmd_K>;
  using tile_shapeA_tcols = CubeTileA<dtypeA, tM, tK, rmd_M, tK>;
  using tile_shapeA_tcorner = CubeTileA<dtypeA, tM, tK, rmd_M, rmd_K>;

  using tile_shapeAMX_trows = Tile<Location::Scaling, uint8_t, tM, tK, BLayout::RowMajor, valid_row, rmd_K/smatrix_wfactor>;
  using tile_shapeAMX_tcols = Tile<Location::Scaling, uint8_t, tM, tK, BLayout::RowMajor, rmd_M, tK/smatrix_wfactor>;
  using tile_shapeAMX_tcorner = Tile<Location::Scaling, uint8_t, tM, tK, BLayout::RowMajor, rmd_M, rmd_K/smatrix_wfactor>;

  using tile_shapeB_trows = CubeTileN8<dtypeB, tK, tN, tK, rmd_N>;
  using tile_shapeB_tcols = CubeTileN8<dtypeB, tK, tN, rmd_K, valid_col>;
  using tile_shapeB_tcorner = CubeTileN8<dtypeB, tK, tN, rmd_K, rmd_N>;

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

  using tile_shapeA = CubeTileA<dtypeA, tM, tK/typeb_wfactor>;
  using tile_shapeB = CubeTileN8<dtypeB, tK/typeb_wfactor, tN>;
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

  using tile_shapeA = CubeTileA<dtypeA, tM, tK/typeb_wfactor>;
  using tile_shapeB = CubeTileN8<dtypeB, tK/typeb_wfactor, tN>;
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

// mixed precision matmul dequant(A*B)， fp4x2 width_factor: 2
template <typename dtypeA, typename dtypeB, const int gM, const int gN, const int gK, const int tM, const int tN, const int tK, const int width_factor>
void matmul_mp(float *acc_ptr, dtypeA *a_ptr, dtypeB *b_ptr, float *c_ptr) {
  static_assert(gM % tM == 0 && gN % tN == 0 && gK % tK == 0,
                "A16W4 original profiles use complete matrix tiles");
  static_assert(tK == 128);
  static_assert(tM == 16 && tN == 16 && width_factor == 2,
                "A16W4 native dequantization is defined for 16x16 cells");
  constexpr int HalfN = 8;
  constexpr int Mb = gM / tM;
  constexpr int Nb = gN / tN;
  constexpr int Kb = gK / tK;

  using A = CubeTileA<dtypeA, tM, tK>;
  using B = CubeTileN8<dtypeB, tK, tN>;
  using Acc = CubeAccumulatorM16<float, tM, tN>;
  using AccHalf = CubeAccumulatorM16<float, tM, HalfN>;
  using ScaleHalf =
      Tile<Location::Vec, float, tM, HalfN, BLayout::CubeM16, 1, HalfN>;
  using ResultHalf =
      Tile<Location::Vec, float, tM, HalfN, BLayout::CubeM16>;

  static_assert(Acc::LogicalTileBytes == 1024);
  static_assert(AccHalf::LogicalTileBytes == 512);
  static_assert(ScaleHalf::LogicalTileBytes == 512);
  static_assert(ResultHalf::LogicalTileBytes == 512);

  for (int i = 0; i < Mb; ++i) {
    for (int j = 0; j < Nb; ++j) {
      ResultHalf sums[2][2];
      TEXPANDS(sums[0][0], 0.0f);
      TEXPANDS(sums[0][1], 0.0f);

#pragma clang loop unroll(full)
      for (int k = 0; k < Kb; ++k) {
        A a;
        B b;
        global_tensor<dtypeA, MatrixLayout<tM, tK, gK, 1>>
            a_block(a_ptr + uint64_t(i) * tM * gK + uint64_t(k) * tK);
        global_tensor<dtypeB, MatrixLayout<tK, tN, gN, 1>>
            b_block(b_ptr + (uint64_t(k) * tK * gN + uint64_t(j) * tN) /
                              width_factor);
        TLOAD_CUBE(a, a_block);
        TLOAD_CUBE(b, b_block);

        Acc partial;
        TMATMUL(partial, a, b);
        auto partial_halves = TPARTVIEW<AccHalf, 1, 2>(partial);

        for (int half = 0; half < 2; ++half) {
          ScaleHalf scale;
          global_tensor<float, MatrixLayout<1, HalfN, gN, 1>> scale_block(
              c_ptr + uint64_t(k) * gN + uint64_t(j) * tN + half * HalfN);
          TLOAD(scale, scale_block);

          auto partial_half = partial_halves[0][half];
          ResultHalf scaled;
          TCOLEXPANDMUL(scaled, partial_half, scale);
          TADD(sums[(k + 1) % 2][half], scaled, sums[k % 2][half]);
        }
      }

      for (int half = 0; half < 2; ++half) {
        global_tensor<float, MatrixLayout<tM, HalfN, gN, 1>> out_block(
            acc_ptr + uint64_t(i) * tM * gN + uint64_t(j) * tN +
            half * HalfN);
        TSTORE_CUBE(out_block, sums[Kb % 2][half]);
      }
    }
  }
}

#endif
