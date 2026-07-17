#ifndef JCOREBENCH_TEMPLATE_ASM_HPP
#define JCOREBENCH_TEMPLATE_ASM_HPP

#include <common/pto_tileop.hpp>

#if !defined(__linx)

namespace supernpubench_v057_test_fallback {
template <typename T>
constexpr int tile_numel_v = T::Rows * T::Cols;

template <typename OutTile, typename InTile>
inline void copy_tile(OutTile &dst, const InTile &src) {
  constexpr int OutN = tile_numel_v<OutTile>;
  constexpr int InN = tile_numel_v<InTile>;
  constexpr int N = OutN < InN ? OutN : InN;
  for (int i = 0; i < N; ++i)
    dst.data()[i] = static_cast<typename OutTile::DType>(src.data()[i]);
}

template <typename OutTile>
inline void fill_tile(OutTile &dst, typename OutTile::DType value) {
  for (int i = 0; i < tile_numel_v<OutTile>; ++i)
    dst.data()[i] = value;
}

template <typename OutTile, typename OffTile, typename GmShape>
inline void gather_bytes(OutTile &dst, GmShape &src, OffTile &offset) {
  using DType = typename OutTile::DType;
  constexpr int OutN = tile_numel_v<OutTile>;
  constexpr int OffN = tile_numel_v<OffTile>;
  constexpr int N = OutN < OffN ? OutN : OffN;
  const auto *base = src.data();
  for (int i = 0; i < N; ++i) {
    long long byte_off = static_cast<long long>(offset.data()[i]);
    dst.data()[i] = static_cast<DType>(
        *(reinterpret_cast<const DType *>(
            reinterpret_cast<const char *>(base) + byte_off)));
  }
}

template <typename GmShape, typename InTile, typename OffTile>
inline void scatter_bytes(GmShape &dst, InTile &src, OffTile &offset) {
  using DType = typename InTile::DType;
  constexpr int SrcN = tile_numel_v<InTile>;
  constexpr int OffN = tile_numel_v<OffTile>;
  constexpr int N = SrcN < OffN ? SrcN : OffN;
  auto *base = dst.data();
  for (int i = 0; i < N; ++i) {
    long long byte_off = static_cast<long long>(offset.data()[i]);
    *(reinterpret_cast<DType *>(reinterpret_cast<char *>(base) + byte_off)) =
        src.data()[i];
  }
}
} // namespace supernpubench_v057_test_fallback

template <is_tile_data_v tile_shape>
void TXOR(tile_shape &dst, tile_shape &src0, tile_shape &src1) {
  for (int i = 0; i < supernpubench_v057_test_fallback::tile_numel_v<tile_shape>; ++i)
    dst.data()[i] = static_cast<typename tile_shape::DType>(src0.data()[i] ^ src1.data()[i]);
}

template <is_tile_data_v tile_shape>
void TSHLS(tile_shape &dst, tile_shape &src, unsigned shift) {
  for (int i = 0; i < supernpubench_v057_test_fallback::tile_numel_v<tile_shape>; ++i)
    dst.data()[i] = static_cast<typename tile_shape::DType>(src.data()[i] << shift);
}

template <is_tile_data_v tile_shape>
void TSHRS(tile_shape &dst, tile_shape &src, unsigned shift) {
  using Unsigned = decltype(static_cast<unsigned long long>(src.data()[0]));
  for (int i = 0; i < supernpubench_v057_test_fallback::tile_numel_v<tile_shape>; ++i)
    dst.data()[i] = static_cast<typename tile_shape::DType>(
        static_cast<Unsigned>(src.data()[i]) >> shift);
}

template <is_tile_data_v tile_shape, is_tile_data_v tile_shape_index>
void TSEL(tile_shape &dst, tile_shape_index &cond, tile_shape &src) {
  constexpr int DstN = supernpubench_v057_test_fallback::tile_numel_v<tile_shape>;
  constexpr int CondN = supernpubench_v057_test_fallback::tile_numel_v<tile_shape_index>;
  constexpr int N = DstN < CondN ? DstN : CondN;
  for (int i = 0; i < N; ++i)
    if (cond.data()[i])
      dst.data()[i] = src.data()[i];
}

template <is_tile_data_v tile_shape_out, is_tile_data_v tile_shape_offset,
          is_global_data_v gm_shape>
void MGATHER(tile_shape_out &dst, gm_shape &src, tile_shape_offset &offset) {
  supernpubench_v057_test_fallback::gather_bytes(dst, src, offset);
}

template <is_tile_data_v tile_shape_in, is_tile_data_v tile_shape_offset,
          is_global_data_v gm_shape>
void MSCATTER(gm_shape &dst, tile_shape_in &src, tile_shape_offset &offset) {
  supernpubench_v057_test_fallback::scatter_bytes(dst, src, offset);
}

template <is_tile_data_v tile_shape>
void TMAX_TEPL(tile_shape &dst, tile_shape &src0, tile_shape &src1) {
  for (int i = 0; i < supernpubench_v057_test_fallback::tile_numel_v<tile_shape>; ++i)
    dst.data()[i] = src0.data()[i] > src1.data()[i] ? src0.data()[i] : src1.data()[i];
}

template <is_tile_data_v tile_shape>
void TMULS_TEPL(tile_shape &dst, tile_shape &src0, typename tile_shape::DType s) {
  for (int i = 0; i < supernpubench_v057_test_fallback::tile_numel_v<tile_shape>; ++i)
    dst.data()[i] = static_cast<typename tile_shape::DType>(src0.data()[i] * s);
}

template <is_tile_data_v tile_shape_out, is_tile_data_v tile_shape_in>
void TROWMAX_TEPL(tile_shape_out &dst, tile_shape_in &src) {
  using DType = typename tile_shape_out::DType;
  for (int row = 0; row < tile_shape_in::Rows && row < tile_shape_out::Rows; ++row) {
    DType max_value{};
    bool init = false;
    for (int col = 0; col < tile_shape_in::Cols; ++col) {
      auto value = src.data()[row * tile_shape_in::Cols + col];
      if (!init || value > max_value) {
        max_value = static_cast<DType>(value);
        init = true;
      }
    }
    for (int col = 0; col < tile_shape_out::Cols; ++col)
      dst.data()[row * tile_shape_out::Cols + col] = max_value;
  }
}

template <is_tile_data_v tile_shape>
void TSUB_TEPL(tile_shape &dst, tile_shape &src0, tile_shape &src1) {
  for (int i = 0; i < supernpubench_v057_test_fallback::tile_numel_v<tile_shape>; ++i)
    dst.data()[i] = static_cast<typename tile_shape::DType>(src0.data()[i] - src1.data()[i]);
}

template <is_tile_data_v tile_shape>
void TEXP_TEPL(tile_shape &dst, tile_shape &src) {
  supernpubench_v057_test_fallback::copy_tile(dst, src);
}

template <is_tile_data_v tile_shape>
void TMUL_TEPL(tile_shape &dst, tile_shape &src0, tile_shape &src1) {
  for (int i = 0; i < supernpubench_v057_test_fallback::tile_numel_v<tile_shape>; ++i)
    dst.data()[i] = static_cast<typename tile_shape::DType>(src0.data()[i] * src1.data()[i]);
}

template <is_tile_data_v tile_shape_out, is_tile_data_v tile_shape_in>
void TROWSUM_TEPL(tile_shape_out &dst, tile_shape_in &src) {
  using DType = typename tile_shape_out::DType;
  for (int row = 0; row < tile_shape_in::Rows && row < tile_shape_out::Rows; ++row) {
    DType sum{};
    for (int col = 0; col < tile_shape_in::Cols; ++col)
      sum = static_cast<DType>(sum + src.data()[row * tile_shape_in::Cols + col]);
    for (int col = 0; col < tile_shape_out::Cols; ++col)
      dst.data()[row * tile_shape_out::Cols + col] = sum;
  }
}

template <is_tile_data_v tile_shape>
void TADD_TEPL(tile_shape &dst, tile_shape &src0, tile_shape &src1) {
  for (int i = 0; i < supernpubench_v057_test_fallback::tile_numel_v<tile_shape>; ++i)
    dst.data()[i] = static_cast<typename tile_shape::DType>(src0.data()[i] + src1.data()[i]);
}

template <is_tile_data_v tile_shape>
void TRECIP_TEPL(tile_shape &dst, tile_shape &src) {
  supernpubench_v057_test_fallback::copy_tile(dst, src);
}

template <is_tile_data_v tile_shape_out, is_tile_data_v tile_shape_in>
void TCAST_TEPL(tile_shape_out &dst, tile_shape_in &src) {
  supernpubench_v057_test_fallback::copy_tile(dst, src);
}

template <is_tile_data_v tile_shape>
void TEXPANDSCALAR_TEPL(tile_shape &dst, typename tile_shape::DType s) {
  supernpubench_v057_test_fallback::fill_tile(dst, s);
}

template <is_tile_data_v tile_shape_out, is_tile_data_v tile_shape_in>
void TROWEXPAND_TEPL(tile_shape_out &dst, tile_shape_in &src) {
  for (int row = 0; row < tile_shape_out::Rows; ++row) {
    auto value = src.data()[row < tile_shape_in::Rows ? row * tile_shape_in::Cols : 0];
    for (int col = 0; col < tile_shape_out::Cols; ++col)
      dst.data()[row * tile_shape_out::Cols + col] =
          static_cast<typename tile_shape_out::DType>(value);
  }
}

template <is_tile_data_v tile_shape_out, is_tile_data_v tile_shape_in>
void TCOLMAX_TEPL(tile_shape_out &dst, tile_shape_in &src) {
  using DType = typename tile_shape_out::DType;
  for (int col = 0; col < tile_shape_in::Cols && col < tile_shape_out::Cols; ++col) {
    DType max_value{};
    bool init = false;
    for (int row = 0; row < tile_shape_in::Rows; ++row) {
      auto value = src.data()[row * tile_shape_in::Cols + col];
      if (!init || value > max_value) {
        max_value = static_cast<DType>(value);
        init = true;
      }
    }
    for (int row = 0; row < tile_shape_out::Rows; ++row)
      dst.data()[row * tile_shape_out::Cols + col] = max_value;
  }
}

template <is_tile_data_v tile_shape_out, is_tile_data_v tile_shape_in>
void TCOLSUM_TEPL(tile_shape_out &dst, tile_shape_in &src) {
  using DType = typename tile_shape_out::DType;
  for (int col = 0; col < tile_shape_in::Cols && col < tile_shape_out::Cols; ++col) {
    DType sum{};
    for (int row = 0; row < tile_shape_in::Rows; ++row)
      sum = static_cast<DType>(sum + src.data()[row * tile_shape_in::Cols + col]);
    for (int row = 0; row < tile_shape_out::Rows; ++row)
      dst.data()[row * tile_shape_out::Cols + col] = sum;
  }
}

template <is_tile_data_v tile_shape_out, is_tile_data_v tile_shape_in>
void TCOLEXPANDSUB_TEPL(tile_shape_out &dst, tile_shape_out &src0,
                        tile_shape_in &src1) {
  constexpr int OutN = supernpubench_v057_test_fallback::tile_numel_v<tile_shape_out>;
  constexpr int InN = supernpubench_v057_test_fallback::tile_numel_v<tile_shape_in>;
  constexpr int N = OutN < InN ? OutN : InN;
  for (int i = 0; i < N; ++i)
    dst.data()[i] = static_cast<typename tile_shape_out::DType>(src0.data()[i] - src1.data()[i]);
}

template <is_tile_data_v tile_shape_out, is_tile_data_v tile_shape_in>
void TCOLEXPANDMUL_TEPL(tile_shape_out &dst, tile_shape_out &src0,
                        tile_shape_in &src1) {
  constexpr int OutN = supernpubench_v057_test_fallback::tile_numel_v<tile_shape_out>;
  constexpr int InN = supernpubench_v057_test_fallback::tile_numel_v<tile_shape_in>;
  constexpr int N = OutN < InN ? OutN : InN;
  for (int i = 0; i < N; ++i)
    dst.data()[i] = static_cast<typename tile_shape_out::DType>(src0.data()[i] * src1.data()[i]);
}

#else

template <is_tile_data_v tile_shape_out, is_tile_data_v tile_shape_offset, is_global_data_v gm_shape>
void MGATHER(tile_shape_out &dst, gm_shape &src, tile_shape_offset &offset) {
  asm volatile(
    "BSTART.MGATHER %c[DataType]\n"
    "B.DIM zero, %c[VCOL], ->lb0\n"
    "B.DIM zero, %c[VROW], ->lb1\n"
    "B.IOT %[s1], last, ->%q[d0]<%c[TileSize]>\n"
    "B.IOR [%[s0]], []\n"
    : [d0]"=r"(dst.data())
    : [s0]"r"(src.data()),
      [s1]"r"(offset.data()),
      [DataType]"i"(type_traits<typename tile_shape_out::DType>::TypeCode),
      [TileSize]"i"(tile_type_traits<typename tile_shape_out::TileDType>::TilesizeCode),
      [VCOL]"i"(tile_shape_offset::ValidCol), [VROW]"i"(tile_shape_offset::ValidRow)
  );
}

template <is_tile_data_v tile_shape_in, is_tile_data_v tile_shape_offset, is_global_data_v gm_shape>
void MSCATTER(gm_shape &dst, tile_shape_in &src, tile_shape_offset &offset) {
  asm volatile(
    "BSTART.MSCATTER %c[SrcType]\n"
    "B.DIM zero, %c[VCOL], ->lb0\n"
    "B.DIM zero, %c[VROW], ->lb1\n"
    "B.IOT %[s0], %[s1], last\n"
    "B.IOR [%[d0]], []\n"
    :
    : [d0]"r"(dst.data()), [s0]"r"(src.data()),
      [s1]"r"(offset.data()),
      [SrcType]"i"(type_traits<typename tile_shape_in::DType>::TypeCode),
      [VCOL]"i"(tile_shape_offset::ValidCol), [VROW]"i"(tile_shape_offset::ValidRow)
  );
}

template <is_tile_data_v tile_shape>
void TMAX_TEPL(tile_shape &dst, tile_shape &src0, tile_shape &src1) {
  asm volatile(
    "BSTART.TMAX %c1\n"
    "B.DATR Null\n"
    "C.B.DIMI %c2, ->LB0\n"
    "C.B.DIMI %c3, ->LB1\n"
    "C.B.DIMI %c4, ->LB2\n"
    "B.IOT %5, %6, last, ->%q0<%c7>\n"
    ""
    : "=r"(dst.data())
    : "i"(type_traits<typename tile_shape::DType>::TypeCode),
      "i"(tile_shape::ValidCol),
      "i"(tile_shape::ValidRow),
      "i"(tile_shape::Cols),
      "r"(src0.data()),
      "r"(src1.data()),
      "i"(tile_type_traits<typename tile_shape::TileDType>::TilesizeCode)
  );
}

template <is_tile_data_v tile_shape>
void TMULS_TEPL(tile_shape &dst, tile_shape &src0, typename tile_shape::DType s) {
  asm volatile(
    "BSTART.TMULS %c1\n"
    "B.DATR Null\n"
    "C.B.DIMI %c2, ->LB0\n"
    "C.B.DIMI %c3, ->LB1\n"
    "C.B.DIMI %c4, ->LB2\n"
    "B.IOT %5, last, ->%q0<%c6>\n"
    "B.IOR [%7],[]\n"
    ""
    : "=r"(dst.data())
    : "i"(type_traits<typename tile_shape::DType>::TypeCode),
      "i"(tile_shape::ValidCol),
      "i"(tile_shape::ValidRow),
      "i"(tile_shape::Cols),
      "r"(src0.data()),
      "i"(tile_type_traits<typename tile_shape::TileDType>::TilesizeCode),
      "r"(s)
  );
}

template <is_tile_data_v tile_shape_out, is_tile_data_v tile_shape_in>
void TROWMAX_TEPL(tile_shape_out &dst, tile_shape_in &src) {
  asm volatile(
    "BSTART.TROWMAX %c1\n"
    "B.DATR Null\n"
    "C.B.DIMI %c2, ->LB0\n"
    "C.B.DIMI %c3, ->LB1\n"
    "C.B.DIMI %c4, ->LB2\n"
    "B.IOT %5, last, ->%q0<%c6>\n"
    ""
    : "=r"(dst.data())
    : "i"(type_traits<typename tile_shape_in::DType>::TypeCode),
      "i"(tile_shape_in::ValidCol),
      "i"(tile_shape_in::ValidRow),
      "i"(tile_shape_in::Cols),
      "r"(src.data()),
      "i"(tile_type_traits<typename tile_shape_out::TileDType>::TilesizeCode)
  );
}

template <is_tile_data_v tile_shape>
void TSUB_TEPL(tile_shape &dst, tile_shape &src0, tile_shape &src1) {
  asm volatile(
    "BSTART.TSUB %c1\n"
    "B.DATR Null\n"
    "C.B.DIMI %c2, ->LB0\n"
    "C.B.DIMI %c3, ->LB1\n"
    "C.B.DIMI %c4, ->LB2\n"
    "B.IOT %5, %6, last, ->%q0<%c7>\n"
    ""
    : "=r"(dst.data())
    : "i"(type_traits<typename tile_shape::DType>::TypeCode),
      "i"(tile_shape::ValidCol),
      "i"(tile_shape::ValidRow),
      "i"(tile_shape::Cols),
      "r"(src0.data()),
      "r"(src1.data()),
      "i"(tile_type_traits<typename tile_shape::TileDType>::TilesizeCode)
  );
}

template <is_tile_data_v tile_shape>
void TEXP_TEPL(tile_shape &dst, tile_shape &src) {
  asm volatile(
    "BSTART.TEXP %c1\n"
    "B.DATR Null\n"
    "C.B.DIMI %c2, ->LB0\n"
    "C.B.DIMI %c3, ->LB1\n"
    "C.B.DIMI %c4, ->LB2\n"
    "B.IOT %5, last, ->%q0<%c6>\n"
    ""
    : "=r"(dst.data())
    : "i"(type_traits<typename tile_shape::DType>::TypeCode),
      "i"(tile_shape::ValidCol),
      "i"(tile_shape::ValidRow),
      "i"(tile_shape::Cols),
      "r"(src.data()),
      "i"(tile_type_traits<typename tile_shape::TileDType>::TilesizeCode)
  );
}

template <is_tile_data_v tile_shape>
void TMUL_TEPL(tile_shape &dst, tile_shape &src0, tile_shape &src1) {
  asm volatile(
    "BSTART.TMUL %c1\n"
    "B.DATR Null\n"
    "C.B.DIMI %c2, ->LB0\n"
    "C.B.DIMI %c3, ->LB1\n"
    "C.B.DIMI %c4, ->LB2\n"
    "B.IOT %5, %6, last, ->%q0<%c7>\n"
    ""
    : "=r"(dst.data())
    : "i"(type_traits<typename tile_shape::DType>::TypeCode),
      "i"(tile_shape::ValidCol),
      "i"(tile_shape::ValidRow),
      "i"(tile_shape::Cols),
      "r"(src0.data()),
      "r"(src1.data()),
      "i"(tile_type_traits<typename tile_shape::TileDType>::TilesizeCode)
  );
}

template <is_tile_data_v tile_shape_out, is_tile_data_v tile_shape_in>
void TROWSUM_TEPL(tile_shape_out &dst, tile_shape_in &src) {
  asm volatile(
    "BSTART.TROWSUM %c1\n"
    "B.DATR Null\n"
    "C.B.DIMI %c2, ->LB0\n"
    "C.B.DIMI %c3, ->LB1\n"
    "C.B.DIMI %c4, ->LB2\n"
    "B.IOT %5, last, ->%q0<%c6>\n"
    ""
    : "=r"(dst.data())
    : "i"(type_traits<typename tile_shape_in::DType>::TypeCode),
      "i"(tile_shape_in::ValidCol),
      "i"(tile_shape_in::ValidRow),
      "i"(tile_shape_in::Cols),
      "r"(src.data()),
      "i"(tile_type_traits<typename tile_shape_out::TileDType>::TilesizeCode)
  );
}

template <is_tile_data_v tile_shape>
void TADD_TEPL(tile_shape &dst, tile_shape &src0, tile_shape &src1) {
  asm volatile(
    "BSTART.TADD %c1\n"
    "B.DATR Null\n"
    "C.B.DIMI %c2, ->LB0\n"
    "C.B.DIMI %c3, ->LB1\n"
    "C.B.DIMI %c4, ->LB2\n"
    "B.IOT %5, %6, last, ->%q0<%c7>\n"
    ""
    : "=r"(dst.data())
    : "i"(type_traits<typename tile_shape::DType>::TypeCode),
      "i"(tile_shape::ValidCol),
      "i"(tile_shape::ValidRow),
      "i"(tile_shape::Cols),
      "r"(src0.data()),
      "r"(src1.data()),
      "i"(tile_type_traits<typename tile_shape::TileDType>::TilesizeCode)
  );
}

template <is_tile_data_v tile_shape>
void TRECIP_TEPL(tile_shape &dst, tile_shape &src) {
  asm volatile(
    "BSTART.TRECIP %c1\n"
    "B.DATR Null\n"
    "C.B.DIMI %c2, ->LB0\n"
    "C.B.DIMI %c3, ->LB1\n"
    "C.B.DIMI %c4, ->LB2\n"
    "B.IOT %5, last, ->%q0<%c6>\n"
    ""
    : "=r"(dst.data())
    : "i"(type_traits<typename tile_shape::DType>::TypeCode),
      "i"(tile_shape::ValidCol),
      "i"(tile_shape::ValidRow),
      "i"(tile_shape::Cols),
      "r"(src.data()),
      "i"(tile_type_traits<typename tile_shape::TileDType>::TilesizeCode)
  );
}

template <is_tile_data_v tile_shape_out, is_tile_data_v tile_shape_in>
void TCAST_TEPL(tile_shape_out &dst, tile_shape_in &src) {
  asm volatile(
    "BSTART.TCVT %c1\n"
    "B.DATR %c2, RNONE\n"
    "C.B.DIMI %c3, ->LB0\n"
    "C.B.DIMI %c4, ->LB1\n"

    "B.IOT %5, last, ->%q0<%c6>\n"
    ""
    : "=r"(dst.data())
    : "i"(type_traits<typename tile_shape_in::DType>::TypeCode),
      "i"(type_traits<typename tile_shape_out::DType>::TypeCode),
      "i"(tile_shape_in::ValidCol),
      "i"(tile_shape_in::ValidRow),
      "r"(src.data()),
      "i"(tile_type_traits<typename tile_shape_out::TileDType>::TilesizeCode)
  );
}

template <is_tile_data_v tile_shape>
void TEXPANDSCALAR_TEPL(tile_shape &dst, typename tile_shape::DType s) {
  asm volatile(
    "BSTART.TEXPANDS %c1\n"
    "B.DATR Null\n"
    "C.B.DIMI %c2, ->LB0\n"
    "C.B.DIMI %c3, ->LB1\n"
    "C.B.DIMI %c4, ->LB2\n"
    "B.IOT last, ->%q0<%c5>\n"
    "B.IOR [%6],[]\n"
    ""
    : "=r"(dst.data())
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
    "BSTART.TROWEXPAND %c1\n"
    "C.B.DIMI %c2, ->LB0\n"
    "C.B.DIMI %c3, ->LB1\n"
    "C.B.DIMI %c4, ->LB2\n"
    "B.IOT %5, last, ->%q0<%c6>\n"
    ""
    : "=r"(dst.data())
    : "i"(type_traits<typename tile_shape_in::DType>::TypeCode),
      "i"(tile_shape_in::ValidCol),
      "i"(tile_shape_in::ValidRow),
      "i"(tile_shape_in::Cols),
      "r"(src.data()),
      "i"(tile_type_traits<typename tile_shape_out::TileDType>::TilesizeCode)
  );
}

template <is_tile_data_v tile_shape_out, is_tile_data_v tile_shape_in>
void TCOLMAX_TEPL(tile_shape_out &dst, tile_shape_in &src) {
  asm volatile(
    "BSTART.TCOLMAX %c1\n"
    "B.DATR Null\n"
    "C.B.DIMI %c2, ->LB0\n"
    "C.B.DIMI %c3, ->LB1\n"
    "C.B.DIMI %c4, ->LB2\n"
    "B.IOT %5, last, ->%q0<%c6>\n"
    ""
    : "=r"(dst.data())
    : "i"(type_traits<typename tile_shape_in::DType>::TypeCode),
      "i"(tile_shape_in::ValidCol),
      "i"(tile_shape_in::ValidRow),
      "i"(tile_shape_in::Cols),
      "r"(src.data()),
      "i"(tile_type_traits<typename tile_shape_out::TileDType>::TilesizeCode)
  );
}

template <is_tile_data_v tile_shape_out, is_tile_data_v tile_shape_in>
void TCOLSUM_TEPL(tile_shape_out &dst, tile_shape_in &src) {
  asm volatile(
    "BSTART.TCOLSUM %c1\n"
    "B.DATR Null\n"
    "C.B.DIMI %c2, ->LB0\n"
    "C.B.DIMI %c3, ->LB1\n"
    "C.B.DIMI %c4, ->LB2\n"
    "B.IOT %5, last, ->%q0<%c6>\n"
    ""
    : "=r"(dst.data())
    : "i"(type_traits<typename tile_shape_in::DType>::TypeCode),
      "i"(tile_shape_in::ValidCol),
      "i"(tile_shape_in::ValidRow),
      "i"(tile_shape_in::Cols),
      "r"(src.data()),
      "i"(tile_type_traits<typename tile_shape_out::TileDType>::TilesizeCode)
  );
}

template <is_tile_data_v tile_shape_out, is_tile_data_v tile_shape_in>
void TCOLEXPANDSUB_TEPL(tile_shape_out &dst, tile_shape_out &src0, tile_shape_in &src1) {
  asm volatile(
    "BSTART.TCOLEXPANDSUB %c1\n"
    "B.DATR Null\n"
    "C.B.DIMI %c2, ->LB0\n"
    "C.B.DIMI %c3, ->LB1\n"
    "C.B.DIMI %c4, ->LB2\n"
    "B.IOT %5, %6, last, ->%q0<%c7>\n"
    ""
    : "=r"(dst.data())
    : "i"(type_traits<typename tile_shape_out::DType>::TypeCode),
      "i"(tile_shape_out::ValidCol),
      "i"(tile_shape_out::ValidRow),
      "i"(tile_shape_out::Cols),
      "r"(src0.data()),
      "r"(src1.data()),
      "i"(tile_type_traits<typename tile_shape_out::TileDType>::TilesizeCode)
  );
}

template <is_tile_data_v tile_shape_out, is_tile_data_v tile_shape_in>
void TCOLEXPANDMUL_TEPL(tile_shape_out &dst, tile_shape_out &src0, tile_shape_in &src1) {
  asm volatile(
    "BSTART.TCOLEXPANDMUL %c1\n"
    "B.DATR Null\n"
    "C.B.DIMI %c2, ->LB0\n"
    "C.B.DIMI %c3, ->LB1\n"
    "C.B.DIMI %c4, ->LB2\n"
    "B.IOT %5, %6, last, ->%q0<%c7>\n"
    ""
    : "=r"(dst.data())
    : "i"(type_traits<typename tile_shape_out::DType>::TypeCode),
      "i"(tile_shape_out::ValidCol),
      "i"(tile_shape_out::ValidRow),
      "i"(tile_shape_out::Cols),
      "r"(src0.data()),
      "r"(src1.data()),
      "i"(tile_type_traits<typename tile_shape_out::TileDType>::TilesizeCode)
  );
}

#endif

#endif
