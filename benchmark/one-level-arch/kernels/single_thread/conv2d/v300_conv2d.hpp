#ifndef V300_CONV2D_TILEOP_KERNEL_HPP
#define V300_CONV2D_TILEOP_KERNEL_HPP

#include <common/pto_tileop.hpp>
#include <type_traits>

namespace supernpu::conv2d {

using namespace pto;

template <typename Element, int Rows, int Cols,
          int ValidRows = Rows, int ValidCols = Cols>
using AccTile = Tile<Location::Vec, Element, Rows, Cols,
                     BLayout::RowMajor, ValidRows, ValidCols>;

// A 1x1 NCHW convolution is a matrix multiplication:
//   [H * W, C_in] x [C_in, C_out] -> [H * W, C_out].
// ColMajor preserves the NCHW channel-major layout for input and output.
// MBlockStride and m_block_start optionally select a round-robin subset of
// output-spatial tile blocks. Their defaults preserve single-PE execution.
template <typename DType,
          int InChannels, int InHeight, int InWidth, int OutChannels,
          int TileM, int TileN, int TileK, int MBlockStride = 1>
void conv2d_1x1(float *output, DType *input, DType *weight,
                int m_block_start = 0) {
    constexpr int GlobalM = InHeight * InWidth;
    constexpr int GlobalN = OutChannels;
    constexpr int GlobalK = InChannels;

    static_assert(GlobalM % TileM == 0,
                  "H * W must be divisible by TileM");
    static_assert(GlobalN % TileN == 0,
                  "OutChannels must be divisible by TileN");
    static_assert(GlobalK % TileK == 0,
                  "InChannels must be divisible by TileK");
    static_assert(MBlockStride > 0, "MBlockStride must be positive");

    using GlobalInput = global_tensor<DType, ColMajor<GlobalM, GlobalK>>;
    using GlobalWeight = global_tensor<DType, ColMajor<GlobalK, GlobalN>>;
#ifdef CONV2D_TMUL_VERIFY
    using GlobalOutput = global_tensor<float, RowMajor<GlobalM, GlobalN>>;
#else
    using GlobalOutput = global_tensor<float, ColMajor<GlobalM, GlobalN>>;
#endif

    using TileA = std::conditional_t<
        (TileM <= 16), CubeTileM16<DType, TileM, TileK>,
        CubeTileM32<DType, TileM, TileK>>;
    using TileB = CubeTileN8<DType, TileK, TileN>;
    using TileC = std::conditional_t<
        (TileM <= 16), CubeAccumulatorM16<float, TileM, TileN>,
        CubeAccumulatorM32<float, TileM, TileN>>;

    using InputIterator = global_iterator<GlobalInput, TileA>;
    using WeightIterator = global_iterator<GlobalWeight, TileB>;
    using OutputIterator = global_iterator<GlobalOutput, TileC>;

    InputIterator input_iter(input);
    WeightIterator weight_iter(weight);
    OutputIterator output_iter(output);

    constexpr int MBlocks = GlobalM / TileM;
    constexpr int NBlocks = GlobalN / TileN;
    constexpr int KBlocks = GlobalK / TileK;

    for (int m = m_block_start; m < MBlocks; m += MBlockStride) {
        for (int n = 0; n < NBlocks; ++n) {
            auto global_c = output_iter(m, n);
            TileC tile_c;

            auto global_a = input_iter(m, 0);
            auto global_b = weight_iter(0, n);
            TileA tile_a;
            TileB tile_b;
            TLOAD_CUBE(tile_a, global_a);
            TLOAD_CUBE(tile_b, global_b);
            TMATMUL(tile_c, tile_a, tile_b);

#pragma clang loop unroll(full)
            for (int k = 1; k < KBlocks; ++k) {
                auto next_global_a = input_iter(m, k);
                auto next_global_b = weight_iter(k, n);
                TileA next_tile_a;
                TileB next_tile_b;
                TLOAD_CUBE(next_tile_a, next_global_a);
                TLOAD_CUBE(next_tile_b, next_global_b);
                TMATMUL_ACC(tile_c, tile_c, next_tile_a, next_tile_b);
            }

            TSTORE_CUBE(global_c, tile_c);
        }
    }
}

}  // namespace supernpu::conv2d

#endif
