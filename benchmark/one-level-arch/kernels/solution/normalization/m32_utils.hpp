#ifndef SUPERNPU_NORMALIZATION_M32_UTILS_HPP
#define SUPERNPU_NORMALIZATION_M32_UTILS_HPP

#include <common/pto_tileop.hpp>
#include <common/pto_tile_region_inline_asm.hpp>

namespace normalization_m32 {
// These reductions process one logical row. CUBE reduction outputs retain
// source physical columns; materialize their first CELL before accumulation,
// broadcasting or GM cache round-trips. The FP32 value is unchanged by *1.
template <typename Out, typename In>
__attribute__((always_inline)) inline void row_sum(Out &out, In &in) {
    static_assert(Out::Rows == 1 && Out::Cols == 1);
    static_assert(In::Rows == 1);
    static_assert(In::BFractal == pto::BLayout::CubeM32);
    static_assert(std::is_same_v<typename In::DType, float>);
    static_assert(std::is_same_v<typename Out::DType, float>);
    using Wide = pto::Tile<pto::Location::Vec, float, 1, In::Cols,
                           pto::BLayout::CubeM32, 1, 1>;
    using Row = pto::Tile<pto::Location::Vec, float, 1, 1,
                          pto::BLayout::CubeM32, 1, 1>;
    Wide wide;
    // TCVT/elementwise produce compact M32 FP32 descriptors: one column
    // per CELL, so physical columns equal the runtime valid columns.
    // The current TileOP TROWSUM wrapper instead emits In::Cols into LB2,
    // which is allocation capacity and is wrong for short strips/tails.
    // Emit the same row-reduction instruction with runtime LB2; keep the
    // allocated destination large enough for the maximum supported strip.
    const size_t columns = in.GetValidCol();
    if constexpr (In::ValidCol > 0) {
        asm volatile(
            "BSTART.TEPL 64, %D1\n"
            "B.DATR CUBE_M32, Null\n"
            "B.DIM zero, %c2, ->lb0\n"
            "B.DIM zero, 1, ->lb1\n"
            "B.DIM zero, %c2, ->lb2\n"
            "B.IOT %3, mask=1111, last, ->%0<%Z4>\n"
            : "=Tr"(wide.data())
            : "i"(type_traits<float>::TypeCode), "i"(In::ValidCol),
              "Tr"(in.data()),
              "i"(tile_type_traits<typename Wide::TileDType>::TilesizeCode));
    } else {
    asm volatile(
        "BSTART.TEPL 64, %D1\n"
        "B.DATR CUBE_M32, Null\n"
        "B.DIM %2, 0, ->lb0\n"
        "B.DIM zero, 1, ->lb1\n"
        "B.DIM %2, 0, ->lb2\n"
        "B.IOT %3, mask=1111, last, ->%0<%Z4>\n"
        : "=Tr"(wide.data())
        : "i"(type_traits<float>::TypeCode), "r"(columns), "Tr"(in.data()),
          "i"(tile_type_traits<typename Wide::TileDType>::TilesizeCode));
    }
    auto prefix = pto::TREDUCEPREFIXVIEW<Row>(wide);
    Row compact;
    TMULS(compact, prefix, 1.0f);
    // Only the valid-row type differs for dynamic callers (runtime value 1).
    // Both carriers are one M32 FP32 CELL; preserve the caller's metadata.
    out.assignData(compact.data_);
}
} // namespace normalization_m32
#endif
