#ifndef JCOREBENCH_TEMPLATE_ASM_HPP
#define JCOREBENCH_TEMPLATE_ASM_HPP

#include <common/pto_tileop.hpp>

using namespace pto;

template <typename TileDType>
struct pto_v057_tile_alloc {
  static constexpr unsigned SizeCode = tile_type_traits<TileDType>::TilesizeCode;
  static_assert(SizeCode >= 3, "v0.57 B.OTA allocation must be at least one 128-byte CELL");
  static constexpr unsigned CellCountM1 = (1u << (SizeCode - 3u)) - 1u;
};

template <is_tile_data_v tile_shape_out, is_tile_data_v tile_shape_offset, is_global_data_v gm_shape>
void MGATHER(tile_shape_out &dst, gm_shape &src, tile_shape_offset &offset) {
  pto::MGATHER(dst, src, offset);
}

template <is_tile_data_v tile_shape_in, is_tile_data_v tile_shape_offset, is_global_data_v gm_shape>
void MSCATTER(gm_shape &dst, tile_shape_in &src, tile_shape_offset &offset) {
  pto::MSCATTER(dst, src, offset);
}

template <is_tile_data_v tile_shape>
void TMAX_TEPL(tile_shape &dst, tile_shape &src0, tile_shape &src1) {
  asm volatile(
    "BSTART.TEPL 0x25, %c1\n"
    "B.DATR Null\n"
    "C.B.DIMI %c2, ->LB0\n"
    "C.B.DIMI %c3, ->LB1\n"
    "C.B.DIMI %c4, ->LB2\n"
    "B.ITP [%5, %6], 0\n"
    "B.OTA ->%0<%c7>, last, 0\n"
    ""
    : "=Tr"(dst.raw())
    : "i"(type_traits<typename tile_shape::DType>::TypeCode),
      "i"(tile_shape::ValidCol),
      "i"(tile_shape::ValidRow),
      "i"(tile_shape::Cols),
      "Tr"(src0.raw()),
      "Tr"(src1.raw()),
      "i"(pto_v057_tile_alloc<typename tile_shape::TileDType>::CellCountM1)
  );
}

template <is_tile_data_v tile_shape>
void TMULS_TEPL(tile_shape &dst, tile_shape &src0, typename tile_shape::DType s) {
  asm volatile(
    "BSTART.TEPL 0x2B, %c1\n"
    "B.DATR Null\n"
    "C.B.DIMI %c2, ->LB0\n"
    "C.B.DIMI %c3, ->LB1\n"
    "C.B.DIMI %c4, ->LB2\n"
    "B.ITP [%5], 0\n"
    "B.OTA ->%0<%c6>, last, 0\n"
    "B.IOR [%7],[]\n"
    ""
    : "=Tr"(dst.raw())
    : "i"(type_traits<typename tile_shape::DType>::TypeCode),
      "i"(tile_shape::ValidCol),
      "i"(tile_shape::ValidRow),
      "i"(tile_shape::Cols),
      "Tr"(src0.raw()),
      "i"(pto_v057_tile_alloc<typename tile_shape::TileDType>::CellCountM1),
      "r"(s)
  );
}

template <is_tile_data_v tile_shape_out, is_tile_data_v tile_shape_in>
void TROWMAX_TEPL(tile_shape_out &dst, tile_shape_in &src) {
  asm volatile(
    "BSTART.TEPL 0x47, %c1\n"
    "B.DATR Null\n"
    "C.B.DIMI %c2, ->LB0\n"
    "C.B.DIMI %c3, ->LB1\n"
    "C.B.DIMI %c4, ->LB2\n"
    "B.ITP [%5], 0\n"
    "B.OTA ->%0<%c6>, last, 0\n"
    ""
    : "=Tr"(dst.raw())
    : "i"(type_traits<typename tile_shape_in::DType>::TypeCode),
      "i"(tile_shape_in::ValidCol),
      "i"(tile_shape_in::ValidRow),
      "i"(tile_shape_in::Cols),
      "Tr"(src.raw()),
      "i"(pto_v057_tile_alloc<typename tile_shape_out::TileDType>::CellCountM1)
  );
}

template <is_tile_data_v tile_shape>
void TSUB_TEPL(tile_shape &dst, tile_shape &src0, tile_shape &src1) {
  asm volatile(
    "BSTART.TEPL 0x55, %c1\n"
    "B.DATR Null\n"
    "C.B.DIMI %c2, ->LB0\n"
    "C.B.DIMI %c3, ->LB1\n"
    "C.B.DIMI %c4, ->LB2\n"
    "B.ITP [%5, %6], 0\n"
    "B.OTA ->%0<%c7>, last, 0\n"
    ""
    : "=Tr"(dst.raw())
    : "i"(type_traits<typename tile_shape::DType>::TypeCode),
      "i"(tile_shape::ValidCol),
      "i"(tile_shape::ValidRow),
      "i"(tile_shape::Cols),
      "Tr"(src0.raw()),
      "Tr"(src1.raw()),
      "i"(pto_v057_tile_alloc<typename tile_shape::TileDType>::CellCountM1)
  );
}

template <is_tile_data_v tile_shape>
void TEXP_TEPL(tile_shape &dst, tile_shape &src) {
  asm volatile(
    "BSTART.TEPL 0x1C, %c1\n"
    "B.DATR Null\n"
    "C.B.DIMI %c2, ->LB0\n"
    "C.B.DIMI %c3, ->LB1\n"
    "C.B.DIMI %c4, ->LB2\n"
    "B.ITP [%5], 0\n"
    "B.OTA ->%0<%c6>, last, 0\n"
    ""
    : "=Tr"(dst.raw())
    : "i"(type_traits<typename tile_shape::DType>::TypeCode),
      "i"(tile_shape::ValidCol),
      "i"(tile_shape::ValidRow),
      "i"(tile_shape::Cols),
      "Tr"(src.raw()),
      "i"(pto_v057_tile_alloc<typename tile_shape::TileDType>::CellCountM1)
  );
}

template <is_tile_data_v tile_shape>
void TMUL_TEPL(tile_shape &dst, tile_shape &src0, tile_shape &src1) {
  asm volatile(
    "BSTART.TEPL 0x2A, %c1\n"
    "B.DATR Null\n"
    "C.B.DIMI %c2, ->LB0\n"
    "C.B.DIMI %c3, ->LB1\n"
    "C.B.DIMI %c4, ->LB2\n"
    "B.ITP [%5, %6], 0\n"
    "B.OTA ->%0<%c7>, last, 0\n"
    ""
    : "=Tr"(dst.raw())
    : "i"(type_traits<typename tile_shape::DType>::TypeCode),
      "i"(tile_shape::ValidCol),
      "i"(tile_shape::ValidRow),
      "i"(tile_shape::Cols),
      "Tr"(src0.raw()),
      "Tr"(src1.raw()),
      "i"(pto_v057_tile_alloc<typename tile_shape::TileDType>::CellCountM1)
  );
}

template <is_tile_data_v tile_shape_out, is_tile_data_v tile_shape_in>
void TROWSUM_TEPL(tile_shape_out &dst, tile_shape_in &src) {
  asm volatile(
    "BSTART.TEPL 0x4A, %c1\n"
    "B.DATR Null\n"
    "C.B.DIMI %c2, ->LB0\n"
    "C.B.DIMI %c3, ->LB1\n"
    "C.B.DIMI %c4, ->LB2\n"
    "B.ITP [%5], 0\n"
    "B.OTA ->%0<%c6>, last, 0\n"
    ""
    : "=Tr"(dst.raw())
    : "i"(type_traits<typename tile_shape_in::DType>::TypeCode),
      "i"(tile_shape_in::ValidCol),
      "i"(tile_shape_in::ValidRow),
      "i"(tile_shape_in::Cols),
      "Tr"(src.raw()),
      "i"(pto_v057_tile_alloc<typename tile_shape_out::TileDType>::CellCountM1)
  );
}

template <is_tile_data_v tile_shape>
void TADD_TEPL(tile_shape &dst, tile_shape &src0, tile_shape &src1) {
  asm volatile(
    "BSTART.TEPL 0x01, %c1\n"
    "B.DATR Null\n"
    "C.B.DIMI %c2, ->LB0\n"
    "C.B.DIMI %c3, ->LB1\n"
    "C.B.DIMI %c4, ->LB2\n"
    "B.ITP [%5, %6], 0\n"
    "B.OTA ->%0<%c7>, last, 0\n"
    ""
    : "=Tr"(dst.raw())
    : "i"(type_traits<typename tile_shape::DType>::TypeCode),
      "i"(tile_shape::ValidCol),
      "i"(tile_shape::ValidRow),
      "i"(tile_shape::Cols),
      "Tr"(src0.raw()),
      "Tr"(src1.raw()),
      "i"(pto_v057_tile_alloc<typename tile_shape::TileDType>::CellCountM1)
  );
}

template <is_tile_data_v tile_shape>
void TRECIP_TEPL(tile_shape &dst, tile_shape &src) {
  asm volatile(
    "BSTART.TEPL 0x39, %c1\n"
    "B.DATR Null\n"
    "C.B.DIMI %c2, ->LB0\n"
    "C.B.DIMI %c3, ->LB1\n"
    "C.B.DIMI %c4, ->LB2\n"
    "B.ITP [%5], 0\n"
    "B.OTA ->%0<%c6>, last, 0\n"
    ""
    : "=Tr"(dst.raw())
    : "i"(type_traits<typename tile_shape::DType>::TypeCode),
      "i"(tile_shape::ValidCol),
      "i"(tile_shape::ValidRow),
      "i"(tile_shape::Cols),
      "Tr"(src.raw()),
      "i"(pto_v057_tile_alloc<typename tile_shape::TileDType>::CellCountM1)
  );
}

template <is_tile_data_v tile_shape_out, is_tile_data_v tile_shape_in>
void TCAST_TEPL(tile_shape_out &dst, tile_shape_in &src) {
  asm volatile(
    "BSTART.TEPL 0x19, %c1\n"
    "B.DATR %c2, RNONE\n"
    "C.B.DIMI %c3, ->LB0\n"
    "C.B.DIMI %c4, ->LB1\n"

    "B.ITP [%5], 0\n"
    "B.OTA ->%0<%c6>, last, 0\n"
    ""
    : "=Tr"(dst.raw())
    : "i"(type_traits<typename tile_shape_in::DType>::TypeCode),
      "i"(type_traits<typename tile_shape_out::DType>::TypeCode),
      "i"(tile_shape_in::ValidCol),
      "i"(tile_shape_in::ValidRow),
      "Tr"(src.raw()),
      "i"(pto_v057_tile_alloc<typename tile_shape_out::TileDType>::CellCountM1)
  );
}

template <is_tile_data_v tile_shape>
void TEXPANDSCALAR_TEPL(tile_shape &dst, typename tile_shape::DType s) {
  asm volatile(
    "BSTART.TEPL 0x1D, %c1\n"
    "B.DATR Null\n"
    "C.B.DIMI %c2, ->LB0\n"
    "C.B.DIMI %c3, ->LB1\n"
    "C.B.DIMI %c4, ->LB2\n"
    "B.OTA ->%0<%c5>, last, 0\n"
    "B.IOR [%6],[]\n"
    ""
    : "=Tr"(dst.raw())
    : "i"(type_traits<typename tile_shape::DType>::TypeCode),
      "i"(tile_shape::ValidCol),
      "i"(tile_shape::ValidRow),
      "i"(tile_shape::Cols),
      "i"(pto_v057_tile_alloc<typename tile_shape::TileDType>::CellCountM1),
      "r"(s)
  );
}

template <is_tile_data_v tile_shape_out, is_tile_data_v tile_shape_in>
void TROWEXPAND_TEPL(tile_shape_out &dst, tile_shape_in &src) {
  asm volatile(
    "BSTART.TEPL 0x3F, %c1\n"
    "C.B.DIMI %c2, ->LB0\n"
    "C.B.DIMI %c3, ->LB1\n"
    "C.B.DIMI %c4, ->LB2\n"
    "B.ITP [%5], 0\n"
    "B.OTA ->%0<%c6>, last, 0\n"
    ""
    : "=Tr"(dst.raw())
    : "i"(type_traits<typename tile_shape_in::DType>::TypeCode),
      "i"(tile_shape_in::ValidCol),
      "i"(tile_shape_in::ValidRow),
      "i"(tile_shape_in::Cols),
      "Tr"(src.raw()),
      "i"(pto_v057_tile_alloc<typename tile_shape_out::TileDType>::CellCountM1)
  );
}

template <is_tile_data_v tile_shape_out, is_tile_data_v tile_shape_in>
void TCOLMAX_TEPL(tile_shape_out &dst, tile_shape_in &src) {
  asm volatile(
    "BSTART.TEPL 0x15, %c1\n"
    "B.DATR Null\n"
    "C.B.DIMI %c2, ->LB0\n"
    "C.B.DIMI %c3, ->LB1\n"
    "C.B.DIMI %c4, ->LB2\n"
    "B.ITP [%5], 0\n"
    "B.OTA ->%0<%c6>, last, 0\n"
    ""
    : "=Tr"(dst.raw())
    : "i"(type_traits<typename tile_shape_in::DType>::TypeCode),
      "i"(tile_shape_in::ValidCol),
      "i"(tile_shape_in::ValidRow),
      "i"(tile_shape_in::Cols),
      "Tr"(src.raw()),
      "i"(pto_v057_tile_alloc<typename tile_shape_out::TileDType>::CellCountM1)
  );
}

template <is_tile_data_v tile_shape_out, is_tile_data_v tile_shape_in>
void TCOLSUM_TEPL(tile_shape_out &dst, tile_shape_in &src) {
  asm volatile(
    "BSTART.TEPL 0x18, %c1\n"
    "B.DATR Null\n"
    "C.B.DIMI %c2, ->LB0\n"
    "C.B.DIMI %c3, ->LB1\n"
    "C.B.DIMI %c4, ->LB2\n"
    "B.ITP [%5], 0\n"
    "B.OTA ->%0<%c6>, last, 0\n"
    ""
    : "=Tr"(dst.raw())
    : "i"(type_traits<typename tile_shape_in::DType>::TypeCode),
      "i"(tile_shape_in::ValidCol),
      "i"(tile_shape_in::ValidRow),
      "i"(tile_shape_in::Cols),
      "Tr"(src.raw()),
      "i"(pto_v057_tile_alloc<typename tile_shape_out::TileDType>::CellCountM1)
  );
}

template <is_tile_data_v tile_shape_out, is_tile_data_v tile_shape_in>
void TCOLEXPANDSUB_TEPL(tile_shape_out &dst, tile_shape_out &src0, tile_shape_in &src1) {
  asm volatile(
    "BSTART.TEPL 0x14, %c1\n"
    "B.DATR Null\n"
    "C.B.DIMI %c2, ->LB0\n"
    "C.B.DIMI %c3, ->LB1\n"
    "C.B.DIMI %c4, ->LB2\n"
    "B.ITP [%5, %6], 0\n"
    "B.OTA ->%0<%c7>, last, 0\n"
    ""
    : "=Tr"(dst.raw())
    : "i"(type_traits<typename tile_shape_out::DType>::TypeCode),
      "i"(tile_shape_out::ValidCol),
      "i"(tile_shape_out::ValidRow),
      "i"(tile_shape_out::Cols),
      "Tr"(src0.raw()),
      "Tr"(src1.raw()),
      "i"(pto_v057_tile_alloc<typename tile_shape_out::TileDType>::CellCountM1)
  );
}

template <is_tile_data_v tile_shape_out, is_tile_data_v tile_shape_in>
void TCOLEXPANDMUL_TEPL(tile_shape_out &dst, tile_shape_out &src0, tile_shape_in &src1) {
  asm volatile(
    "BSTART.TEPL 0x13, %c1\n"
    "B.DATR Null\n"
    "C.B.DIMI %c2, ->LB0\n"
    "C.B.DIMI %c3, ->LB1\n"
    "C.B.DIMI %c4, ->LB2\n"
    "B.ITP [%5, %6], 0\n"
    "B.OTA ->%0<%c7>, last, 0\n"
    ""
    : "=Tr"(dst.raw())
    : "i"(type_traits<typename tile_shape_out::DType>::TypeCode),
      "i"(tile_shape_out::ValidCol),
      "i"(tile_shape_out::ValidRow),
      "i"(tile_shape_out::Cols),
      "Tr"(src0.raw()),
      "Tr"(src1.raw()),
      "i"(pto_v057_tile_alloc<typename tile_shape_out::TileDType>::CellCountM1)
  );
}

#endif
