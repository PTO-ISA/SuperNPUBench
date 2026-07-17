// ============================================================================
// Gather 算子 — PTO 一层编程模型
//
// 原始 gather.hpp 策略:
//   对每个输出 tile (tM, tN):
//     1. TLOAD  加载 offset tile (行索引, 1×tM) from GM
//     2. __vec__ gen_offset 计算字节偏移:
//          offset[row,col] = (in_offset[row] * gN + n_base + col) * sizeof(dtype)
//     3. MGATHER  按字节偏移从数据表取数 (旧 MGATHER, 字节偏移语义)
//     4. TSTORE 写回输出
//
// PTO 一层策略:
//   对每个输出 tile (tM, tN):
//     1. TLOAD    加载 offset tile (行索引, 1×tM) from GM
//     2. MGATHER<Coalesce::Row>  按行索引直接取数:
//          dst[r,:] = table[idx[r], :]
//          通过调整 table 指针 (+n_base) 处理列偏移
//          (PTO ISA MGATHER 使用行索引, 非 字节偏移, 无需 gen_offset)
//     3. TSTORE   写回输出
//
//   gen_offset __vec__ 块完全消除: MGATHER<Coalesce::Row> 内部完成
//   行索引 → 行地址 的转换 (tablePtr + idx * tableRowStride)。
//
// ┌─────────────────────────────────────────────────────────────────────────┐
// │                    当前编译器不支持 / 不完整的指令汇总                   │
// ├──────────┬──────────────────┬──────────────────────────────────────────┤
// │ Pto ISA  │ 当前编译器状态   │ 说明                                     │
// │ 指令     │                  │                                          │
// ├──────────┼──────────────────┼──────────────────────────────────────────┤
// │ TLOAD    │ API 有(名不同)， │ PTO ISA 名 TLOAD；当前编译器名 TLOAD；  │
// │          │                  │ 核心参数 (dst, src) 和行为一致            │
// ├──────────┼──────────────────┼──────────────────────────────────────────┤
// │ MGATHER  │ 部分支持         │ template_asm.h 有 MGATHER (asm volatile);│
// │          │                  │ 但缺 Coalesce 模板参数 (无法选 Row/Elem);│
// │          │                  │ 且旧实现按字节偏移取数,                   │
// │          │                  │ PTO ISA Coalesce::Row 按行索引取数        │
// ├──────────┼──────────────────┼──────────────────────────────────────────┤
// │ TSTORE   │ API 有(名不同)， │ PTO ISA 名 TSTORE；当前编译器名 TSTORE；│
// │          │                  │ 核心参数 (dst, src) 和行为一致            │
// └──────────┴──────────────────┴──────────────────────────────────────────┘
//
// PTO ISA 文档签名 (Declared in include/pto/pto_instr.hpp):
//
//   TLOAD:
//     template <typename TileData, typename GlobalData, typename... WaitEvents>
//     PTO_INST RecordEvent TLOAD(TileData &dst, GlobalData &src,
//                                WaitEvents &... events);
//
//   MGATHER (Coalesce::Row 模式):
//     template <Coalesce Mode = Coalesce::Row,
//               GatherOOB Oob  = GatherOOB::Undefined,
//               typename TileDst, typename GlobalData,
//               typename TileInd, typename... WaitEvents>
//     PTO_INST RecordEvent MGATHER(TileDst &dst, GlobalData &src,
//                                  TileInd &indexes, WaitEvents &... events);
//
//     Row 模式语义: dst[r,:] = table[idx[r], :]
//       - idx 为行索引 (非字节偏移)
//       - tablePtr + idx * tableRowStride 定位到行起始
//       - 读取 validCol 个元素到 dst 对应行
//       - 约束 (A2A3): TileIdx::ValidRow==1, TileIdx::ValidCol==TileDst::ValidRow
//       - 约束 (A5):   同上, 或 [R,1] ColMajor; 且 staticShape[4]==ValidCol
//
//   TSTORE:
//     template <typename TileData, typename GlobalData,
//               AtomicType atomicType = AtomicType::AtomicNone,
//               typename... WaitEvents>
//     PTO_INST RecordEvent TSTORE(GlobalData &dst, TileData &src,
//                                 WaitEvents &... events);
// ============================================================================

#include <common/pto_tile.hpp>
#include <common/global_iterator.hpp>
// #include <pto/pto_instr.hpp>            // [!] PTO ISA C++ Intrinsic — 当前编译器未提供

#include <cstdint>
#include <cstdio>

// ----------------------------------------------------------------------------
// gather: 按行索引从数据表中 gather 行
//
// 输入:
//   in_data_ptr   — 数据表, shape (gK, gN), row-major
//   in_offset_ptr — 行索引数组, shape (gM,), 每个元素是数据表的行号
//   out_ptr       — 输出, shape (gM, gN), out[j,:] = in_data[in_offset[j],:]
//
// 模板参数 (与原 gather.hpp 一致):
//   dtype  — 数据类型 (float, half, ...)
//   otype  — 索引类型 (uint32_t, int32_t)
//   gK     — 数据表行数
//   gM     — 输出行数 (= 索引数组长度)
//   gN     — 数据表列数 (= 每行元素数)
//   tM     — tile 行数
//   tN     — tile 列数
// ----------------------------------------------------------------------------
template<typename dtype = float, typename otype = uint32_t,
         size_t gK, size_t gM, size_t gN, size_t tM, size_t tN>
void gather(
    dtype *in_data_ptr,
    otype *in_offset_ptr,
    dtype *out_ptr
    ) {
    const size_t Mb = gM / tM;
    const size_t Nb = gN / tN;
    const size_t rmd_M = gM % tM;
    const size_t rmd_N = gN % tN;

    using gm_shapeInOffset = global_tensor<otype, RowMajor<1, gM>>;
    using gm_shapeIn       = global_tensor<dtype, RowMajor<gK, gN>>;
    using gm_shapeOut      = global_tensor<dtype, RowMajor<gM, gN>>;

    using tile_shapeInOffset     = Tile<Location::Vec, otype,    1,   tM, BLayout::RowMajor>;
    using tile_shapeData         = Tile<Location::Vec, dtype,    tM,  tN, BLayout::RowMajor>;
    using tile_shapeInOffset_rmd_n   = Tile<Location::Vec, otype,    1,   tM, BLayout::RowMajor>;
    using tile_shapeData_rmd_n       = Tile<Location::Vec, dtype,    tM,  tN, BLayout::RowMajor, tM, rmd_N>;
    using tile_shapeInOffset_rmd_mn  = Tile<Location::Vec, otype,    1,   tM, BLayout::RowMajor, 1, rmd_M>;
    using tile_shapeData_rmd_mn      = Tile<Location::Vec, dtype,    tM,  tN, BLayout::RowMajor, rmd_M, rmd_N>;
    using tile_shapeInOffset_rmd_m   = Tile<Location::Vec, otype,    1,   tM, BLayout::RowMajor, 1, rmd_M>;
    using tile_shapeData_rmd_m       = Tile<Location::Vec, dtype,    tM,  tN, BLayout::RowMajor, rmd_M, tN>;

    tile_shapeInOffset inOffsetTile;
    tile_shapeData outTile;
    tile_shapeInOffset_rmd_n inOffsetTile_rmd_n;
    tile_shapeData_rmd_n outTile_rmd_n;
    tile_shapeInOffset_rmd_mn inOffsetTile_rmd_mn;
    tile_shapeData_rmd_mn outTile_rmd_mn;
    tile_shapeInOffset_rmd_m inOffsetTile_rmd_m;
    tile_shapeData_rmd_m outTile_rmd_m;

    using itInOffset = global_iterator<gm_shapeInOffset, tile_shapeInOffset>;
    using itOut      = global_iterator<gm_shapeOut, tile_shapeData>;

    itInOffset gInOffsetIter(in_offset_ptr);
    itOut      gOIter(out_ptr);

    // ---- 主循环: Mb × Nb 个完整 tile ----
    for (int j = 0; j < Mb; ++j) {
        for (int i = 0; i < Nb; ++i) {
            auto gInOffset = gInOffsetIter(0, j);
            auto gO        = gOIter(j, i);
            size_t n_base  = i * tN;

            // TLOAD: 加载行索引 tile (1, tM) from GM
            // [当前编译器] 名为 TLOAD
            TLOAD(inOffsetTile, gInOffset);

            // MGATHER<Coalesce::Row>: 按行索引从数据表取数
            //   dst[r,:] = table[idx[r], :]
            //   table 指针偏移 n_base 个元素, 使取数起始列为 n_base
            //   (tablePtr + idx * gN + n_base 定位到 row idx, col n_base)
            // [当前编译器] template_asm.h 的 MGATHER 无 Coalesce 模板参数;
            //             且按字节偏移取数, 非行索引
            gm_shapeIn adjustedGm(in_data_ptr + n_base);
            MGATHER(outTile, adjustedGm, inOffsetTile);

            // TSTORE: 写回输出 tile (tM, tN) to GM
            // [当前编译器] 名为 TSTORE
            TSTORE(gO, outTile);
        }

        // ---- rmd_N: 最后一个列块不完整 ----
        if constexpr (rmd_N) {
            auto gInOffset = gInOffsetIter(0, j);
            auto gO        = gOIter(j, Nb);
            size_t n_base  = Nb * tN;

            TLOAD(inOffsetTile_rmd_n, gInOffset);
            gm_shapeIn adjustedGm(in_data_ptr + n_base);
            MGATHER(outTile_rmd_n, adjustedGm, inOffsetTile_rmd_n);
            TSTORE(gO, outTile_rmd_n);
        }
    }

    // ---- rmd_M: 最后一个行块不完整 ----
    if constexpr (rmd_M) {
        for (int i = 0; i < Nb; ++i) {
            auto gInOffset = gInOffsetIter(0, Mb);
            auto gO        = gOIter(Mb, i);
            size_t n_base  = i * tN;

            TLOAD(inOffsetTile_rmd_m, gInOffset);
            gm_shapeIn adjustedGm(in_data_ptr + n_base);
            MGATHER(outTile_rmd_m, adjustedGm, inOffsetTile_rmd_m);
            TSTORE(gO, outTile_rmd_m);
        }

        // ---- rmd_M + rmd_N: 右下角不完整 ----
        if constexpr (rmd_N) {
            auto gInOffset = gInOffsetIter(0, Mb);
            auto gO        = gOIter(Mb, Nb);
            size_t n_base  = Nb * tN;

            TLOAD(inOffsetTile_rmd_mn, gInOffset);
            gm_shapeIn adjustedGm(in_data_ptr + n_base);
            MGATHER(outTile_rmd_mn, adjustedGm, inOffsetTile_rmd_mn);
            TSTORE(gO, outTile_rmd_mn);
        }
    }
}
