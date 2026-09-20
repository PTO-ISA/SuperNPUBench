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
    using Wide = pto::Tile<pto::Location::Vec, float, 1, In::Cols,
                           pto::BLayout::CubeM32, 1, 1>;
    using Row = pto::Tile<pto::Location::Vec, float, 1, 1,
                          pto::BLayout::CubeM32, 1, 1>;
    Wide wide;
    TROWSUM(wide, in);
    auto prefix = pto::TREDUCEPREFIXVIEW<Row>(wide);
    Row compact;
    TMULS(compact, prefix, 1.0f);
    // Only the valid-row type differs for dynamic callers (runtime value 1).
    // Both carriers are one M32 FP32 CELL; preserve the caller's metadata.
    out.assignData(compact.data_);
}
} // namespace normalization_m32
#endif
