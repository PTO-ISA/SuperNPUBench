#ifndef JCOREBENCH_TEMPLATE_ASM_HPP
// 2026-09-24：已删除旧版 MGATHER/MSCATTER 封装（BSTART.TMA 4/5 助记符
// 已废弃——"Match Instruction Error"，且与 TileOP 新版
// BSTART.TLSU MGATHER 同名冲突、重载决议更特化会拦截调用）。
#define JCOREBENCH_TEMPLATE_ASM_HPP

#include <common/pto_tileop.hpp>

template <is_tile_data_v tile_shape>
void TMAX_TEPL(tile_shape &dst, tile_shape &src0, tile_shape &src1) {
  asm volatile(
    "BSTART.TEPL 11, %c1\n"
    "B.DATR Null\n"
    "C.B.DIMI %c2, ->LB0\n"
    "C.B.DIMI %c3, ->LB1\n"
    "C.B.DIMI %c4, ->LB2\n"
    "B.IOT [%5, %6], last, ->%0<%c7>\n"
    ""
    : "=Tr"(dst.data())
    : "i"(type_traits<typename tile_shape::DType>::TypeCode),
      "i"(tile_shape::ValidCol),
      "i"(tile_shape::ValidRow),
      "i"(tile_shape::Cols),
      "Tr"(src0.data()),
      "Tr"(src1.data()),
      "i"(tile_type_traits<typename tile_shape::TileDType>::TilesizeCode)
  );
}

template <is_tile_data_v tile_shape>
void TMULS_TEPL(tile_shape &dst, tile_shape &src0, typename tile_shape::DType s) {
  asm volatile(
    "BSTART.TEPL 0b0100010, %c1\n"
    "B.DATR Null\n"
    "C.B.DIMI %c2, ->LB0\n"
    "C.B.DIMI %c3, ->LB1\n"
    "C.B.DIMI %c4, ->LB2\n"
    "B.IOT [%5], last, ->%0<%c6>\n"
    "B.IOR [%7],[]\n"
    ""
    : "=Tr"(dst.data())
    : "i"(type_traits<typename tile_shape::DType>::TypeCode),
      "i"(tile_shape::ValidCol),
      "i"(tile_shape::ValidRow),
      "i"(tile_shape::Cols),
      "Tr"(src0.data()),
      "i"(tile_type_traits<typename tile_shape::TileDType>::TilesizeCode),
      "r"(s)
  );
}

template <is_tile_data_v tile_shape_out, is_tile_data_v tile_shape_in>
void TROWMAX_TEPL(tile_shape_out &dst, tile_shape_in &src) {
  asm volatile(
    "BSTART.TEPL 0b1000001, %c1\n"
    "B.DATR Null\n"
    "C.B.DIMI %c2, ->LB0\n"
    "C.B.DIMI %c3, ->LB1\n"
    "C.B.DIMI %c4, ->LB2\n"
    "B.IOT [%5], last, ->%0<%c6>\n"
    ""
    : "=Tr"(dst.data())
    : "i"(type_traits<typename tile_shape_in::DType>::TypeCode),
      "i"(tile_shape_in::ValidCol),
      "i"(tile_shape_in::ValidRow),
      "i"(tile_shape_in::Cols),
      "Tr"(src.data()),
      "i"(tile_type_traits<typename tile_shape_out::TileDType>::TilesizeCode)
  );
}

template <is_tile_data_v tile_shape>
void TSUB_TEPL(tile_shape &dst, tile_shape &src0, tile_shape &src1) {
  asm volatile(
    "BSTART.TEPL 1, %c1\n"
    "B.DATR Null\n"
    "C.B.DIMI %c2, ->LB0\n"
    "C.B.DIMI %c3, ->LB1\n"
    "C.B.DIMI %c4, ->LB2\n"
    "B.IOT [%5, %6], last, ->%0<%c7>\n"
    ""
    : "=Tr"(dst.data())
    : "i"(type_traits<typename tile_shape::DType>::TypeCode),
      "i"(tile_shape::ValidCol),
      "i"(tile_shape::ValidRow),
      "i"(tile_shape::Cols),
      "Tr"(src0.data()),
      "Tr"(src1.data()),
      "i"(tile_type_traits<typename tile_shape::TileDType>::TilesizeCode)
  );
}

template <is_tile_data_v tile_shape>
void TEXP_TEPL(tile_shape &dst, tile_shape &src) {
  asm volatile(
    "BSTART.TEPL 18, %c1\n"
    "B.DATR Null\n"
    "C.B.DIMI %c2, ->LB0\n"
    "C.B.DIMI %c3, ->LB1\n"
    "C.B.DIMI %c4, ->LB2\n"
    "B.IOT [%5], last, ->%0<%c6>\n"
    ""
    : "=Tr"(dst.data())
    : "i"(type_traits<typename tile_shape::DType>::TypeCode),
      "i"(tile_shape::ValidCol),
      "i"(tile_shape::ValidRow),
      "i"(tile_shape::Cols),
      "Tr"(src.data()),
      "i"(tile_type_traits<typename tile_shape::TileDType>::TilesizeCode)
  );
}

template <is_tile_data_v tile_shape>
void TMUL_TEPL(tile_shape &dst, tile_shape &src0, tile_shape &src1) {
  asm volatile(
    "BSTART.TEPL 2, %c1\n"
    "B.DATR Null\n"
    "C.B.DIMI %c2, ->LB0\n"
    "C.B.DIMI %c3, ->LB1\n"
    "C.B.DIMI %c4, ->LB2\n"
    "B.IOT [%5, %6], last, ->%0<%c7>\n"
    ""
    : "=Tr"(dst.data())
    : "i"(type_traits<typename tile_shape::DType>::TypeCode),
      "i"(tile_shape::ValidCol),
      "i"(tile_shape::ValidRow),
      "i"(tile_shape::Cols),
      "Tr"(src0.data()),
      "Tr"(src1.data()),
      "i"(tile_type_traits<typename tile_shape::TileDType>::TilesizeCode)
  );
}

template <is_tile_data_v tile_shape_out, is_tile_data_v tile_shape_in>
void TROWSUM_TEPL(tile_shape_out &dst, tile_shape_in &src) {
  asm volatile(
    "BSTART.TEPL 0b1000000, %c1\n"
    "B.DATR Null\n"
    "C.B.DIMI %c2, ->LB0\n"
    "C.B.DIMI %c3, ->LB1\n"
    "C.B.DIMI %c4, ->LB2\n"
    "B.IOT [%5], last, ->%0<%c6>\n"
    ""
    : "=Tr"(dst.data())
    : "i"(type_traits<typename tile_shape_in::DType>::TypeCode),
      "i"(tile_shape_in::ValidCol),
      "i"(tile_shape_in::ValidRow),
      "i"(tile_shape_in::Cols),
      "Tr"(src.data()),
      "i"(tile_type_traits<typename tile_shape_out::TileDType>::TilesizeCode)
  );
}

template <is_tile_data_v tile_shape>
void TADD_TEPL(tile_shape &dst, tile_shape &src0, tile_shape &src1) {
  asm volatile(
    "BSTART.TEPL 0, %c1\n"
    "B.DATR Null\n"
    "C.B.DIMI %c2, ->LB0\n"
    "C.B.DIMI %c3, ->LB1\n"
    "C.B.DIMI %c4, ->LB2\n"
    "B.IOT [%5, %6], last, ->%0<%c7>\n"
    ""
    : "=Tr"(dst.data())
    : "i"(type_traits<typename tile_shape::DType>::TypeCode),
      "i"(tile_shape::ValidCol),
      "i"(tile_shape::ValidRow),
      "i"(tile_shape::Cols),
      "Tr"(src0.data()),
      "Tr"(src1.data()),
      "i"(tile_type_traits<typename tile_shape::TileDType>::TilesizeCode)
  );
}

template <is_tile_data_v tile_shape>
void TRECIP_TEPL(tile_shape &dst, tile_shape &src) {
  asm volatile(
    "BSTART.TEPL 20, %c1\n"
    "B.DATR Null\n"
    "C.B.DIMI %c2, ->LB0\n"
    "C.B.DIMI %c3, ->LB1\n"
    "C.B.DIMI %c4, ->LB2\n"
    "B.IOT [%5], last, ->%0<%c6>\n"
    ""
    : "=Tr"(dst.data())
    : "i"(type_traits<typename tile_shape::DType>::TypeCode),
      "i"(tile_shape::ValidCol),
      "i"(tile_shape::ValidRow),
      "i"(tile_shape::Cols),
      "Tr"(src.data()),
      "i"(tile_type_traits<typename tile_shape::TileDType>::TilesizeCode)
  );
}

template <is_tile_data_v tile_shape_out, is_tile_data_v tile_shape_in>
void TCAST_TEPL(tile_shape_out &dst, tile_shape_in &src) {
  asm volatile(
    "BSTART.TEPL 27, %c1\n"
    "B.DATR %c2, RNONE\n"
    "C.B.DIMI %c3, ->LB0\n"
    "C.B.DIMI %c4, ->LB1\n"

    "B.IOT [%5], last, ->%0<%c6>\n"
    ""
    : "=Tr"(dst.data())
    : "i"(type_traits<typename tile_shape_in::DType>::TypeCode),
      "i"(type_traits<typename tile_shape_out::DType>::TypeCode),
      "i"(tile_shape_in::ValidCol),
      "i"(tile_shape_in::ValidRow),
      "Tr"(src.data()),
      "i"(tile_type_traits<typename tile_shape_out::TileDType>::TilesizeCode)
  );
}

template <is_tile_data_v tile_shape>
void TEXPANDSCALAR_TEPL(tile_shape &dst, typename tile_shape::DType s) {
  asm volatile(
    "BSTART.TEPL 0b0111011, %c1\n"
    "B.DATR Null\n"
    "C.B.DIMI %c2, ->LB0\n"
    "C.B.DIMI %c3, ->LB1\n"
    "C.B.DIMI %c4, ->LB2\n"
    "B.IOT [], last, ->%0<%c5>\n"
    "B.IOR [%6],[]\n"
    ""
    : "=Tr"(dst.data())
    : "i"(type_traits<typename tile_shape::DType>::TypeCode),
      "i"(tile_shape::ValidCol),
      "i"(tile_shape::ValidRow),
      "i"(tile_shape::Cols),
      "i"(tile_type_traits<typename tile_shape::TileDType>::TilesizeCode),
      "r"(s)
  );
}

template <is_tile_data_v tile_shape_out, is_tile_data_v tile_shape_in>
void TROWEXPAND_TEPL(tile_shape_out &dst, tile_shape_in &src) {
  asm volatile(
    "BSTART.TEPL 0b1000100, %c1\n"
    "C.B.DIMI %c2, ->LB0\n"
    "C.B.DIMI %c3, ->LB1\n"
    "C.B.DIMI %c4, ->LB2\n"
    "B.IOT [%5], last, ->%0<%c6>\n"
    ""
    : "=Tr"(dst.data())
    : "i"(type_traits<typename tile_shape_in::DType>::TypeCode),
      "i"(tile_shape_in::ValidCol),
      "i"(tile_shape_in::ValidRow),
      "i"(tile_shape_in::Cols),
      "Tr"(src.data()),
      "i"(tile_type_traits<typename tile_shape_out::TileDType>::TilesizeCode)
  );
}

template <is_tile_data_v tile_shape_out, is_tile_data_v tile_shape_in>
void TCOLMAX_TEPL(tile_shape_out &dst, tile_shape_in &src) {
  asm volatile(
    "BSTART.TEPL 0b1010001, %c1\n"
    "B.DATR Null\n"
    "C.B.DIMI %c2, ->LB0\n"
    "C.B.DIMI %c3, ->LB1\n"
    "C.B.DIMI %c4, ->LB2\n"
    "B.IOT [%5], last, ->%0<%c6>\n"
    ""
    : "=Tr"(dst.data())
    : "i"(type_traits<typename tile_shape_in::DType>::TypeCode),
      "i"(tile_shape_in::ValidCol),
      "i"(tile_shape_in::ValidRow),
      "i"(tile_shape_in::Cols),
      "Tr"(src.data()),
      "i"(tile_type_traits<typename tile_shape_out::TileDType>::TilesizeCode)
  );
}

template <is_tile_data_v tile_shape_out, is_tile_data_v tile_shape_in>
void TCOLSUM_TEPL(tile_shape_out &dst, tile_shape_in &src) {
  asm volatile(
    "BSTART.TEPL 0b1010000, %c1\n"
    "B.DATR Null\n"
    "C.B.DIMI %c2, ->LB0\n"
    "C.B.DIMI %c3, ->LB1\n"
    "C.B.DIMI %c4, ->LB2\n"
    "B.IOT [%5], last, ->%0<%c6>\n"
    ""
    : "=Tr"(dst.data())
    : "i"(type_traits<typename tile_shape_in::DType>::TypeCode),
      "i"(tile_shape_in::ValidCol),
      "i"(tile_shape_in::ValidRow),
      "i"(tile_shape_in::Cols),
      "Tr"(src.data()),
      "i"(tile_type_traits<typename tile_shape_out::TileDType>::TilesizeCode)
  );
}

template <is_tile_data_v tile_shape_out, is_tile_data_v tile_shape_in>
void TCOLEXPANDSUB_TEPL(tile_shape_out &dst, tile_shape_out &src0, tile_shape_in &src1) {
  asm volatile(
    "BSTART.TEPL 0b1010110, %c1\n"
    "B.DATR Null\n"
    "C.B.DIMI %c2, ->LB0\n"
    "C.B.DIMI %c3, ->LB1\n"
    "C.B.DIMI %c4, ->LB2\n"
    "B.IOT [%5, %6], last, ->%0<%c7>\n"
    ""
    : "=Tr"(dst.data())
    : "i"(type_traits<typename tile_shape_out::DType>::TypeCode),
      "i"(tile_shape_out::ValidCol),
      "i"(tile_shape_out::ValidRow),
      "i"(tile_shape_out::Cols),
      "Tr"(src0.data()),
      "Tr"(src1.data()),
      "i"(tile_type_traits<typename tile_shape_out::TileDType>::TilesizeCode)
  );
}

template <is_tile_data_v tile_shape_out, is_tile_data_v tile_shape_in>
void TCOLEXPANDMUL_TEPL(tile_shape_out &dst, tile_shape_out &src0, tile_shape_in &src1) {
  asm volatile(
    "BSTART.TEPL 0b1010111, %c1\n"
    "B.DATR Null\n"
    "C.B.DIMI %c2, ->LB0\n"
    "C.B.DIMI %c3, ->LB1\n"
    "C.B.DIMI %c4, ->LB2\n"
    "B.IOT [%5, %6], last, ->%0<%c7>\n"
    ""
    : "=Tr"(dst.data())
    : "i"(type_traits<typename tile_shape_out::DType>::TypeCode),
      "i"(tile_shape_out::ValidCol),
      "i"(tile_shape_out::ValidRow),
      "i"(tile_shape_out::Cols),
      "Tr"(src0.data()),
      "Tr"(src1.data()),
      "i"(tile_type_traits<typename tile_shape_out::TileDType>::TilesizeCode)
  );
}

#endif
