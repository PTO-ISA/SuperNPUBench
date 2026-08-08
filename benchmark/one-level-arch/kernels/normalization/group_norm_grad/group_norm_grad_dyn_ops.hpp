// =============================================================================
// group_norm_grad_dyn_ops.hpp — DYN TEPL wrappers for Valid=-1 tiles
// =============================================================================
//
// Some TEPL ops still encode ValidRow/Col as B.DIM immediates; with NTTP -1
// that becomes `B.DIM zero, -1` (Match Instruction Error). These wrappers pass
// GetValid*() in registers — same pattern as rms_norm_binary's custom TADD.
//
// Ops that already use GetValid* (TMUL/TROWSUM/TLOAD/TSTORE/TROWEXPANDMUL/…)
// do not need wrappers here.
// =============================================================================
#ifndef SUPERNPU_GROUP_NORM_GRAD_DYN_OPS_HPP
#define SUPERNPU_GROUP_NORM_GRAD_DYN_OPS_HPP

#include <common/pto_tileop.hpp>

namespace gn_grad_dyn {

template <typename Tile>
inline void tadd(Tile &dst, Tile &src0, Tile &src1) {
    const size_t valid_col = src0.GetValidCol();
    const size_t valid_row = src0.GetValidRow();
    asm volatile(
        "BSTART.TEPL 0, %c1\n"
        "B.DIM %2, 0, ->lb0\n"
        "B.DIM %3, 0, ->lb1\n"
        "B.DIM zero, %c4, ->lb2\n"
        "B.IOT %5, %6, mask=15, last, ->%0<%Z7>\n"
        ""
        : "=Tr"(dst.data())
        : "i"(type_traits<typename Tile::DType>::TypeCode),
          "r"(valid_col),
          "r"(valid_row),
          "i"(Tile::Cols),
          "Tr"(src0.data()),
          "Tr"(src1.data()),
          "i"(tile_type_traits<typename Tile::TileDType>::TilesizeCode));
}

template <typename Tile>
inline void tsub(Tile &dst, Tile &src0, Tile &src1) {
    const size_t valid_col = src0.GetValidCol();
    const size_t valid_row = src0.GetValidRow();
    asm volatile(
        "BSTART.TEPL 1, %c1\n"
        "B.DIM %2, 0, ->lb0\n"
        "B.DIM %3, 0, ->lb1\n"
        "B.DIM zero, %c4, ->lb2\n"
        "B.IOT %5, %6, mask=15, last, ->%0<%Z7>\n"
        ""
        : "=Tr"(dst.data())
        : "i"(type_traits<typename Tile::DType>::TypeCode),
          "r"(valid_col),
          "r"(valid_row),
          "i"(Tile::Cols),
          "Tr"(src0.data()),
          "Tr"(src1.data()),
          "i"(tile_type_traits<typename Tile::TileDType>::TilesizeCode));
}

template <typename Tile>
inline void texpands(Tile &dst, typename Tile::DType s) {
    volatile typename Tile::DType sv = s;
    const size_t valid_col = dst.GetValidCol();
    const size_t valid_row = dst.GetValidRow();
    asm volatile(
        "BSTART.TEPL 59, %c1\n"
        "B.DIM %2, 0, ->lb0\n"
        "B.DIM %3, 0, ->lb1\n"
        "B.DIM zero, %c4, ->lb2\n"
        "B.IOT mask=15, last, ->%0<%Z5>\n"
        "B.IOR [%6],[]\n"
        ""
        : "=Tr"(dst.data())
        : "i"(type_traits<typename Tile::DType>::TypeCode),
          "r"(valid_col),
          "r"(valid_row),
          "i"(Tile::Cols),
          "i"(tile_type_traits<typename Tile::TileDType>::TilesizeCode),
          "r"(sv));
}

template <typename TileOut, typename Tile0, typename Tile1>
inline void trowexpandadd(TileOut &dst, Tile0 &src0, Tile1 &src1) {
    const size_t valid_col = src0.GetValidCol();
    const size_t valid_row = src0.GetValidRow();
    asm volatile(
        "BSTART.TEPL 69, %c1\n"
        "B.DIM %2, 0, ->lb0\n"
        "B.DIM %3, 0, ->lb1\n"
        "B.DIM zero, %c4, ->lb2\n"
        "B.IOT %5, %6, mask=15, last, ->%0<%Z7>\n"
        ""
        : "=Tr"(dst.data())
        : "i"(type_traits<typename Tile0::DType>::TypeCode),
          "r"(valid_col),
          "r"(valid_row),
          "i"(Tile0::Cols),
          "Tr"(src0.data()),
          "Tr"(src1.data()),
          "i"(tile_type_traits<typename TileOut::TileDType>::TilesizeCode));
}

} // namespace gn_grad_dyn

#endif // SUPERNPU_GROUP_NORM_GRAD_DYN_OPS_HPP
