/**
 * @file transpose_tile_isa.hpp
 * @brief 基于纯 Tile ISA 的转置算子实现
 * 
 * 本文件提供两个转置算子：
 * 1. tile_transpose_nd: 通用 N 维任意轴交换，使用 offset 计算 + MGATHER
 * 2. tile_transpose_2d: 2D 矩阵转置，使用硬件原生 TTRANS 指令
 * 
 * 核心设计思想：
 * - 将"标量计算"与"数据并行计算"分离到不同执行路径
 * - 维度循环在标量核心上运行（编译期模板展开）
 * - 每个维度内的坐标提取用 tile 操作并行处理所有元素
 * - 避免使用 __vec__ 的 per-element 标量循环模式
 */

#ifndef SUPERNPU_TRANSPOSE_TILE_ISA_HPP
#define SUPERNPU_TRANSPOSE_TILE_ISA_HPP

#include <common/pto_tileop.hpp>
#include <cstddef>
#include <cstdint>

// TTRANS 指令的 API 兼容性配置
// 旧 API: TTRANS(dst, src)
// 新 API: TTRANS(dst, src, tmp) - 需要显式传入临时 tile
// 根据实际 PTO 版本设置此宏
#ifndef SUPERNPU_PTO_TTRANS_NEEDS_TMP
#define SUPERNPU_PTO_TTRANS_NEEDS_TMP 0
#endif

namespace supernpu::tile_isa {

/**
 * @brief 通用 N 维转置（任意轴交换）
 * 
 * 算法核心：
 * 1. 对输出张量的每个元素，计算其多维坐标
 * 2. 通过轴映射得到对应的输入坐标
 * 3. 用输入 stride 计算 byte offset
 * 4. 用 MGATHER 按 offset 从输入中 gather 数据
 * 
 * 关键优化：
 * - Rank、Axis0、Axis1 是模板参数 → 维度循环编译期完全展开
 * - 每个维度的坐标提取用 tile 操作并行处理 tile 内所有元素
 * - 不是"per-element 串行迭代"，而是"per-dimension 并行处理"
 * 
 * @tparam DType 数据类型
 * @tparam Rank 张量维度数（编译期常量，> 1）
 * @tparam Axis0 转置轴 0（编译期常量，[0, Rank)）
 * @tparam Axis1 转置轴 1（编译期常量，[0, Rank) 且 != Axis0）
 * @tparam Elements 总元素数（编译期常量，> 0）
 * @tparam TileElements 每个 tile 处理的元素数（默认 512）
 * 
 * @param input 输入数据指针（展平为一维）
 * @param output 输出数据指针（展平为一维）
 * @param input_shape 输入张量各维度尺寸（运行时数组）
 * @param output_shape 输出张量各维度尺寸（运行时数组）
 */
template <typename DType, int Rank, int Axis0, int Axis1, int Elements,
          int TileElements = 512>
void tile_transpose_nd(DType *input, DType *output, std::uint32_t *input_shape,
                       std::uint32_t *output_shape) {
  // ========== 编译期安全检查 ==========
  static_assert(Rank > 1, "转置至少需要 2 维");
  static_assert(Axis0 >= 0 && Axis0 < Rank, "Axis0 超出维度范围");
  static_assert(Axis1 >= 0 && Axis1 < Rank, "Axis1 超出维度范围");
  static_assert(Axis0 != Axis1, "转置的两个轴必须不同");
  static_assert(Elements > 0, "元素数必须为正");
  static_assert(TileElements > 0, "TileElements 必须为正");
  // Tile 行必须 32 字节对齐（硬件要求）
  static_assert(TileElements * static_cast<int>(sizeof(DType)) % 32 == 0,
                "RowMajor Tile 的行必须 32 字节对齐");
  // MGATHER 的 offset tile 使用 uint32 byte offset，不能溢出
  static_assert(static_cast<unsigned long long>(Elements) * sizeof(DType) <=
                    0x100000000ULL,
                "MGATHER 的 byte offset 使用 uint32，不能溢出");

  // ========== 分块参数 ==========
  constexpr int kFullTiles = Elements / TileElements;      // 完整 tile 数量
  constexpr int kTailElements = Elements % TileElements;   // 尾部剩余元素数

  // ========== 类型定义 ==========
  // 输入/输出全局张量：展平为一维（转置不改变存储，只改变逻辑索引）
  using InputGlobal = pto::global_tensor<DType, pto::RowMajor<1, Elements>>;
  using OutputGlobal = pto::global_tensor<DType, pto::RowMajor<1, Elements>>;
  
  // 数据 tile：存储实际数据
  using DataTile = pto::Tile<pto::Location::Vec, DType, 1, TileElements,
                             pto::BLayout::RowMajor>;
  // Offset tile：存储每个元素的 byte offset（用于 MGATHER）
  using OffsetTile = pto::Tile<pto::Location::Vec, std::uint32_t, 1,
                               TileElements, pto::BLayout::RowMajor>;
  // 输出迭代器：用于遍历输出张量的 tile
  using OutputIterator = global_iterator<OutputGlobal, DataTile>;

  // ========== 初始化全局张量和迭代器 ==========
  InputGlobal input_global(input);
  OutputIterator output_iter(output);

  // ========== 处理完整 tile ==========
  std::uint32_t output_base = 0;  // 当前 tile 在输出中的起始线性索引
  for (int tile_index = 0; tile_index < kFullTiles; ++tile_index) {
    auto output_global = output_iter(0, tile_index);
    DataTile output_tile;      // 存储 gather 回来的数据
    OffsetTile offset_tile;    // 存储每个元素的 byte offset

    // 临时 tile（用于坐标提取和 offset 计算）
    OffsetTile linear_index;   // 线性索引 [output_base, output_base+1, ...]
    OffsetTile quotient;       // 除法商
    OffsetTile cycle;          // 周期数（用于取模）
    OffsetTile cycle_base;     // 周期基址
    OffsetTile coordinate;     // 当前维度的坐标
    OffsetTile contribution;   // 当前维度对 offset 的贡献

    // Step 1: 生成线性索引序列 [output_base, output_base+1, ..., output_base+TileElements-1]
    TCI(linear_index, output_base);
    
    // Step 2: 初始化 offset 累加器为 0
    TEXPANDSCALAR(offset_tile, static_cast<std::uint32_t>(0));

    // Step 3: 对每个维度计算坐标并累加 offset
    // 关键：这个循环在标量核心上运行，由于 Rank 是模板参数，编译期完全展开
    // 每次迭代用 tile 操作并行处理 tile 内所有元素的当前维度
    std::uint32_t input_stride = 1;  // 输入张量当前维度的 stride（从内层开始）
    for (int input_dim = Rank - 1; input_dim >= 0; --input_dim) {
      // 轴映射：交换 Axis0 和 Axis1
      // 例如：如果 input_dim == Axis0，则对应的 output_dim == Axis1
      int output_dim = input_dim;
      if (input_dim == Axis0) {
        output_dim = Axis1;
      } else if (input_dim == Axis1) {
        output_dim = Axis0;
      }

      // 计算 output_dim 维度的 stride（标量计算，在标量核心上完成）
      // output_stride = output_shape[output_dim+1] * output_shape[output_dim+2] * ... * output_shape[Rank-1]
      std::uint32_t output_stride = 1;
      for (int dim = output_dim + 1; dim < Rank; ++dim) {
        output_stride *= output_shape[dim];
      }

      // 提取当前维度的坐标：coordinate = (linear_index / output_stride) % output_shape[output_dim]
      // 用 tile 操作并行处理所有元素
      
      // quotient = linear_index / output_stride
      // 优化：如果 output_stride == 1，直接拷贝（避免除法）
      if (output_stride == 1) {
        TCVT(quotient, linear_index);
      } else {
        TDIVS(quotient, linear_index, output_stride);
      }

      // 取模实现：a % b = a - (a / b) * b
      // cycle = quotient / output_shape[output_dim]
      TDIVS(cycle, quotient, output_shape[output_dim]);
      // cycle_base = cycle * output_shape[output_dim]
      TMULS(cycle_base, cycle, output_shape[output_dim]);
      // coordinate = quotient - cycle_base （即 quotient % output_shape[output_dim]）
      TSUB(coordinate, quotient, cycle_base);

      // 计算当前维度对 byte offset 的贡献
      // contribution = coordinate * input_stride * sizeof(DType)
      // 注意：coordinate 是 output_coord[output_dim]，由于轴映射，它等于 input_coord[input_dim]
      TMULS(contribution, coordinate,
            input_stride * static_cast<std::uint32_t>(sizeof(DType)));
      
      // 累加到总 offset
      TADD(offset_tile, offset_tile, contribution);

      // 更新 input_stride（从内层向外层累积）
      input_stride *= input_shape[input_dim];
    }

    // Step 4: 按 byte offset 从输入中 gather 数据
    MGATHER(output_tile, input_global, offset_tile);
    
    // Step 5: 写回到输出全局内存
    TSTORE(output_global, output_tile);

    // 更新下一个 tile 的起始索引
    output_base += TileElements;
  }

  // ========== 处理尾部元素 ==========
  // 如果总元素数不是 TileElements 的整数倍，处理剩余元素
  if constexpr (kTailElements != 0) {
    // 尾部 tile 的类型定义（物理尺寸相同，但 valid region 缩小）
    using TailDataTile = pto::Tile<pto::Location::Vec, DType, 1, TileElements,
                                   pto::BLayout::RowMajor, 1, kTailElements>;
    using TailOffsetTile =
        pto::Tile<pto::Location::Vec, std::uint32_t, 1, TileElements,
                  pto::BLayout::RowMajor, 1, kTailElements>;

    auto output_global = output_iter(0, kFullTiles);
    TailDataTile output_tile;
    TailOffsetTile offset_tile;

    TailOffsetTile linear_index;
    TailOffsetTile quotient;
    TailOffsetTile cycle;
    TailOffsetTile cycle_base;
    TailOffsetTile coordinate;
    TailOffsetTile contribution;

    // 尾部 tile 的处理逻辑与完整 tile 完全一致
    // 区别仅在于 valid region 更小（kTailElements 而非 TileElements）
    // tile 操作以 valid region 为迭代域，因此自动只处理有效元素
    TCI(linear_index, output_base);
    TEXPANDSCALAR(offset_tile, static_cast<std::uint32_t>(0));

    std::uint32_t input_stride = 1;
    for (int input_dim = Rank - 1; input_dim >= 0; --input_dim) {
      int output_dim = input_dim;
      if (input_dim == Axis0) {
        output_dim = Axis1;
      } else if (input_dim == Axis1) {
        output_dim = Axis0;
      }

      std::uint32_t output_stride = 1;
      for (int dim = output_dim + 1; dim < Rank; ++dim) {
        output_stride *= output_shape[dim];
      }

      if (output_stride == 1) {
        TCVT(quotient, linear_index);
      } else {
        TDIVS(quotient, linear_index, output_stride);
      }

      TDIVS(cycle, quotient, output_shape[output_dim]);
      TMULS(cycle_base, cycle, output_shape[output_dim]);
      TSUB(coordinate, quotient, cycle_base);
      TMULS(contribution, coordinate,
            input_stride * static_cast<std::uint32_t>(sizeof(DType)));
      TADD(offset_tile, offset_tile, contribution);

      input_stride *= input_shape[input_dim];
    }

    MGATHER(output_tile, input_global, offset_tile);
    TSTORE(output_global, output_tile);
  }
}

/**
 * @brief 2D 矩阵转置（高性能路径）
 * 
 * 使用硬件原生 TTRANS 指令，性能最优。
 * 算法：标准分块转置
 * 1. 将 Rows × Cols 矩阵划分为 TileRows × TileCols 的块
 * 2. 对每个块：TLOAD → TTRANS → TSTORE
 * 3. 输入块 TileRows × TileCols，转置后输出块 TileCols × TileRows
 * 4. 输出位置为 (col_tile, row_tile)（行列互换）
 * 
 * @tparam DType 数据类型
 * @tparam Rows 输入矩阵行数
 * @tparam Cols 输入矩阵列数
 * @tparam TileRows tile 块行数（默认 16）
 * @tparam TileCols tile 块列数（默认 16）
 * 
 * @param input 输入矩阵指针（Rows × Cols）
 * @param output 输出矩阵指针（Cols × Rows）
 */
template <typename DType, int Rows, int Cols, int TileRows = 16,
          int TileCols = 16>
void tile_transpose_2d(DType *input, DType *output) {
  // ========== 编译期安全检查 ==========
  static_assert(Rows > 0 && Cols > 0, "矩阵维度必须为正");
  static_assert(TileRows > 0 && TileCols > 0, "tile 维度必须为正");
  // TTRANS 对行宽有 32 字节对齐要求
  static_assert(TileRows * static_cast<int>(sizeof(DType)) % 32 == 0,
                "TTRANS 输出行宽必须 32 字节对齐");
  static_assert(TileCols * static_cast<int>(sizeof(DType)) % 32 == 0,
                "TTRANS 输入行宽必须 32 字节对齐");

  // ========== 分块参数 ==========
  constexpr int kRowTiles = Rows / TileRows;    // 行方向完整 tile 数
  constexpr int kColTiles = Cols / TileCols;    // 列方向完整 tile 数
  constexpr int kTailRows = Rows % TileRows;    // 行方向尾部元素数
  constexpr int kTailCols = Cols % TileCols;    // 列方向尾部元素数

  // ========== 类型定义 ==========
  using InputGlobal = pto::global_tensor<DType, pto::RowMajor<Rows, Cols>>;
  using OutputGlobal = pto::global_tensor<DType, pto::RowMajor<Cols, Rows>>;
  
  // 输入 tile：TileRows × TileCols
  using SrcTile = pto::Tile<pto::Location::Vec, DType, TileRows, TileCols,
                            pto::BLayout::RowMajor>;
  // 输出 tile：TileCols × TileRows（转置后）
  using DstTile = pto::Tile<pto::Location::Vec, DType, TileCols, TileRows,
                            pto::BLayout::RowMajor>;
  
  using InputIterator = global_iterator<InputGlobal, SrcTile>;
  using OutputIterator = global_iterator<OutputGlobal, DstTile>;

  InputIterator input_iter(input);
  OutputIterator output_iter(output);

  // ========== 处理完整 tile 块 ==========
  for (int row_tile = 0; row_tile < kRowTiles; ++row_tile) {
    for (int col_tile = 0; col_tile < kColTiles; ++col_tile) {
      // 输入位置：(row_tile, col_tile)
      auto input_global = input_iter(row_tile, col_tile);
      // 输出位置：(col_tile, row_tile) - 行列互换
      auto output_global = output_iter(col_tile, row_tile);
      
      SrcTile src_tile;
      DstTile dst_tile;
      
      // 从全局内存加载输入 tile
      TLOAD(src_tile, input_global);
      
      // 硬件转置：TileRows × TileCols → TileCols × TileRows
#if SUPERNPU_PTO_TTRANS_NEEDS_TMP
      SrcTile tmp_tile;
      TTRANS(dst_tile, src_tile, tmp_tile);
#else
      TTRANS(dst_tile, src_tile);
#endif
      
      // 写回到全局内存
      TSTORE(output_global, dst_tile);
    }

    // ========== 处理右侧尾部 tile（列方向） ==========
    if constexpr (kTailCols != 0) {
      // 输入 tile：TileRows × kTailCols（valid region）
      using SrcRightTile =
          pto::Tile<pto::Location::Vec, DType, TileRows, TileCols,
                    pto::BLayout::RowMajor, TileRows, kTailCols>;
      // 输出 tile：kTailCols × TileRows（转置后）
      using DstBottomTile =
          pto::Tile<pto::Location::Vec, DType, TileCols, TileRows,
                    pto::BLayout::RowMajor, kTailCols, TileRows>;
      
      auto input_global = input_iter(row_tile, kColTiles);
      auto output_global = output_iter(kColTiles, row_tile);
      
      SrcRightTile src_tile;
      DstBottomTile dst_tile;
      
      TLOAD(src_tile, input_global);
#if SUPERNPU_PTO_TTRANS_NEEDS_TMP
      SrcRightTile tmp_tile;
      TTRANS(dst_tile, src_tile, tmp_tile);
#else
      TTRANS(dst_tile, src_tile);
#endif
      TSTORE(output_global, dst_tile);
    }
  }

  // ========== 处理底部尾部 tile（行方向） ==========
  if constexpr (kTailRows != 0) {
    for (int col_tile = 0; col_tile < kColTiles; ++col_tile) {
      // 输入 tile：kTailRows × TileCols（valid region）
      using SrcBottomTile =
          pto::Tile<pto::Location::Vec, DType, TileRows, TileCols,
                    pto::BLayout::RowMajor, kTailRows, TileCols>;
      // 输出 tile：TileCols × kTailRows（转置后）
      using DstRightTile =
          pto::Tile<pto::Location::Vec, DType, TileCols, TileRows,
                    pto::BLayout::RowMajor, TileCols, kTailRows>;
      
      auto input_global = input_iter(kRowTiles, col_tile);
      auto output_global = output_iter(col_tile, kRowTiles);
      
      SrcBottomTile src_tile;
      DstRightTile dst_tile;
      
      TLOAD(src_tile, input_global);
#if SUPERNPU_PTO_TTRANS_NEEDS_TMP
      SrcBottomTile tmp_tile;
      TTRANS(dst_tile, src_tile, tmp_tile);
#else
      TTRANS(dst_tile, src_tile);
#endif
      TSTORE(output_global, dst_tile);
    }

    // ========== 处理右下角尾部 tile ==========
    if constexpr (kTailCols != 0) {
      // 输入 tile：kTailRows × kTailCols（valid region）
      using SrcCornerTile =
          pto::Tile<pto::Location::Vec, DType, TileRows, TileCols,
                    pto::BLayout::RowMajor, kTailRows, kTailCols>;
      // 输出 tile：kTailCols × kTailRows（转置后）
      using DstCornerTile =
          pto::Tile<pto::Location::Vec, DType, TileCols, TileRows,
                    pto::BLayout::RowMajor, kTailCols, kTailRows>;
      
      auto input_global = input_iter(kRowTiles, kColTiles);
      auto output_global = output_iter(kColTiles, kRowTiles);
      
      SrcCornerTile src_tile;
      DstCornerTile dst_tile;
      
      TLOAD(src_tile, input_global);
#if SUPERNPU_PTO_TTRANS_NEEDS_TMP
      SrcCornerTile tmp_tile;
      TTRANS(dst_tile, src_tile, tmp_tile);
#else
      TTRANS(dst_tile, src_tile);
#endif
      TSTORE(output_global, dst_tile);
    }
  }
}

} // namespace supernpu::tile_isa

// Global adapter matching the test harness signature
// transpose<DType, MAX_DIM, gIM, gOM, tM, IN_DIM, OUT_DIM, TRANSPOSE_DIM1, TRANSPOSE_DIM0>
template<typename DType, int MAX_DIM, int gIM, int gOM, int tM,
         int IN_DIM, int OUT_DIM, int TRANSPOSE_DIM1, int TRANSPOSE_DIM0>
void transpose(DType *input, DType *output, std::uint32_t *in_shape,
               std::uint32_t *out_shape) {
    ::supernpu::tile_isa::tile_transpose_nd<DType, IN_DIM, TRANSPOSE_DIM1,
                                            TRANSPOSE_DIM0, gIM, tM>(
        input, output, in_shape, out_shape);
}

#endif
