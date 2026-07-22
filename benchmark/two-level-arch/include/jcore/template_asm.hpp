#ifndef TEMPLATE_ASM_HPP
#define TEMPLATE_ASM_HPP

#include "common/pto_tile.hpp"

// This header defines the historical global PTO wrapper surface while its
// type traits live in namespace pto. Keep it self-contained for compile-only
// conformance tests instead of relying on include-order namespace imports.
using namespace pto;

#if !defined(__linx)

namespace supernpubench_v057_fallback {
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

template <typename OutTile, typename InTile>
inline void gather_bytes(OutTile &dst, const auto &src, const InTile &offset) {
  using DType = typename OutTile::DType;
  constexpr int OutN = tile_numel_v<OutTile>;
  constexpr int OffN = tile_numel_v<InTile>;
  constexpr int N = OutN < OffN ? OutN : OffN;
  const auto *base = src.data();
  for (int i = 0; i < N; ++i) {
    long long byte_off = static_cast<long long>(offset.data()[i]);
    dst.data()[i] = static_cast<DType>(
        *(reinterpret_cast<const DType *>(
            reinterpret_cast<const char *>(base) + byte_off)));
  }
}

template <typename InTile, typename OffTile>
inline void scatter_bytes(auto &dst, const InTile &src, const OffTile &offset) {
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

template <typename Tile, typename Gm>
inline void load_tile(Tile &dst, Gm &src) {
  const int rows = dst.GetValidRow();
  const int cols = dst.GetValidCol();
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      const int gm_index =
          Gm::isRowMajor ? row * Gm::RowStride + col
                         : col * Gm::ColStride + row;
      const int tile_index =
          Tile::isRowMajor ? row * Tile::RowStride + col
                           : col * Tile::ColStride + row;
      dst.data()[tile_index] = src.data()[gm_index];
    }
  }
}

template <typename Gm, typename Tile>
inline void store_tile(Gm &dst, Tile &src) {
  const int rows = src.GetValidRow();
  const int cols = src.GetValidCol();
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      const int gm_index =
          Gm::isRowMajor ? row * Gm::RowStride + col
                         : col * Gm::ColStride + row;
      const int tile_index =
          Tile::isRowMajor ? row * Tile::RowStride + col
                           : col * Tile::ColStride + row;
      dst.data()[gm_index] = src.data()[tile_index];
    }
  }
}
} // namespace supernpubench_v057_fallback

enum class TmaPadValue : int {
  Zero = 0,
  Max = 1,
  Min = 2,
  Null = 3,
};

template <is_tile_data_v tile_shape_out, is_tile_data_v tile_shape_in>
void ACCSCALE_T(tile_shape_out &dst, tile_shape_in &src,
                typename tile_shape_in::DType s) {
  (void)s;
  supernpubench_v057_fallback::copy_tile(dst, src);
}

template <is_tile_data_v tile_shape_out, is_tile_data_v tile_shape_in>
void ACCSCALE_NZ2DN(tile_shape_out &dst, tile_shape_in &src,
                    typename tile_shape_in::DType s) {
  ACCSCALE_T(dst, src, s);
}

template <is_tile_data_v tile_shape_max, is_tile_data_v tile_shape_out,
          is_tile_data_v tile_shape_in>
void ACCCVT_RMAX_SCALE_NZ2DN(tile_shape_max &row_max, tile_shape_out &dst,
                             tile_shape_in &src,
                             typename tile_shape_in::DType s) {
  (void)s;
  supernpubench_v057_fallback::copy_tile(dst, src);
  supernpubench_v057_fallback::fill_tile(row_max, typename tile_shape_max::DType{});
}

template <is_tile_data_v tile_shape_out, is_tile_data_v tile_shape_in0,
          is_tile_data_v tile_shape_in1>
void TMAX_T(tile_shape_out &dst, tile_shape_in0 &src0, tile_shape_in1 &src1) {
  constexpr int N0 = supernpubench_v057_fallback::tile_numel_v<tile_shape_in0>;
  constexpr int N1 = supernpubench_v057_fallback::tile_numel_v<tile_shape_in1>;
  constexpr int NO = supernpubench_v057_fallback::tile_numel_v<tile_shape_out>;
  constexpr int N = (NO < (N0 < N1 ? N0 : N1)) ? NO : (N0 < N1 ? N0 : N1);
  for (int i = 0; i < N; ++i)
    dst.data()[i] = src0.data()[i] > src1.data()[i]
                        ? static_cast<typename tile_shape_out::DType>(src0.data()[i])
                        : static_cast<typename tile_shape_out::DType>(src1.data()[i]);
}

template <is_tile_data_v tile_shape_out, is_tile_data_v tile_shape_in0,
          is_tile_data_v tile_shape_in1>
void TSUB_EXP_EXPAND_T(tile_shape_out &dst, tile_shape_in0 &src0,
                       tile_shape_in1 &src1) {
  constexpr int N0 = supernpubench_v057_fallback::tile_numel_v<tile_shape_in0>;
  constexpr int N1 = supernpubench_v057_fallback::tile_numel_v<tile_shape_in1>;
  constexpr int NO = supernpubench_v057_fallback::tile_numel_v<tile_shape_out>;
  constexpr int N = (NO < (N0 < N1 ? N0 : N1)) ? NO : (N0 < N1 ? N0 : N1);
  for (int i = 0; i < N; ++i)
    dst.data()[i] = static_cast<typename tile_shape_out::DType>(src0.data()[i] - src1.data()[i]);
}

template <is_tile_data_v tile_shape_out, is_tile_data_v tile_shape_in0,
          is_tile_data_v tile_shape_in1, is_tile_data_v tile_shape_in2>
void TMUL_ADD_ROWSUM_T(tile_shape_out &dst, tile_shape_in0 &src0,
                       tile_shape_in1 &src1, tile_shape_in2 &src2) {
  constexpr int N0 = supernpubench_v057_fallback::tile_numel_v<tile_shape_in0>;
  constexpr int N1 = supernpubench_v057_fallback::tile_numel_v<tile_shape_in1>;
  constexpr int N2 = supernpubench_v057_fallback::tile_numel_v<tile_shape_in2>;
  constexpr int NO = supernpubench_v057_fallback::tile_numel_v<tile_shape_out>;
  constexpr int NI = (N0 < N1 ? N0 : N1) < N2 ? (N0 < N1 ? N0 : N1) : N2;
  constexpr int N = NO < NI ? NO : NI;
  for (int i = 0; i < N; ++i)
    dst.data()[i] = static_cast<typename tile_shape_out::DType>(src0.data()[i] * src1.data()[i] + src2.data()[i]);
}

template <is_tile_data_v tile_shape_out, is_tile_data_v tile_shape_in0,
          is_tile_data_v tile_shape_in1, is_tile_data_v tile_shape_in2>
void TADD_MUL_EXPAND_T(tile_shape_out &dst, tile_shape_in0 &src0,
                       tile_shape_in1 &src1, tile_shape_in2 &src2) {
  constexpr int N0 = supernpubench_v057_fallback::tile_numel_v<tile_shape_in0>;
  constexpr int N1 = supernpubench_v057_fallback::tile_numel_v<tile_shape_in1>;
  constexpr int N2 = supernpubench_v057_fallback::tile_numel_v<tile_shape_in2>;
  constexpr int NO = supernpubench_v057_fallback::tile_numel_v<tile_shape_out>;
  constexpr int NI = (N0 < N1 ? N0 : N1) < N2 ? (N0 < N1 ? N0 : N1) : N2;
  constexpr int N = NO < NI ? NO : NI;
  for (int i = 0; i < N; ++i)
    dst.data()[i] = static_cast<typename tile_shape_out::DType>((src0.data()[i] + src1.data()[i]) * src2.data()[i]);
}

template <is_tile_data_v tile_shape_out, is_tile_data_v tile_shape_in>
void TCVT_T(tile_shape_out &dst, tile_shape_in &src) {
  supernpubench_v057_fallback::copy_tile(dst, src);
}

#define DEFINE_TMOV_LAYOUT(LAYOUT_NAME)                                          \
template <is_tile_data_v tile_shape_out, is_tile_data_v tile_shape_in>           \
void TMOV_##LAYOUT_NAME(tile_shape_out &dst, tile_shape_in &src) {               \
  supernpubench_v057_fallback::copy_tile(dst, src);                              \
}

DEFINE_TMOV_LAYOUT(ND2NZ)
DEFINE_TMOV_LAYOUT(NZ2ND)
DEFINE_TMOV_LAYOUT(ND2ZN)
DEFINE_TMOV_LAYOUT(DN2ZN)
DEFINE_TMOV_LAYOUT(DN2NZ)
DEFINE_TMOV_LAYOUT(NZ2DN)
DEFINE_TMOV_LAYOUT(NZ2ZN)
DEFINE_TMOV_LAYOUT(ZN2NZ)
DEFINE_TMOV_LAYOUT(NORM)
#undef DEFINE_TMOV_LAYOUT

template <is_tile_data_v tile_shape_out, is_tile_data_v tile_shape_in>
void TMOV_DN2NZ_DYN(tile_shape_out &dst, tile_shape_in &src) {
  supernpubench_v057_fallback::copy_tile(dst, src);
}

template <is_tile_data_v tile_shape_out, is_tile_data_v tile_shape_in>
void THISTOGRAM(tile_shape_out &dst, tile_shape_in &src, tile_shape_in &Idx,
                int ByteId) {
  (void)src;
  (void)Idx;
  (void)ByteId;
  supernpubench_v057_fallback::fill_tile(dst, typename tile_shape_out::DType{});
}

template <is_tile_data_v tile_shape, is_global_data_v gm_shape>
void TLOAD2_ND2NZ(tile_shape &dst1, tile_shape &dst0, gm_shape &src) {
  supernpubench_v057_fallback::load_tile(dst0, src);
  supernpubench_v057_fallback::load_tile(dst1, src);
}

template <is_tile_data_v tile_shape, is_global_data_v gm_shape>
void TLOAD2_ND2ZN(tile_shape &dst1, tile_shape &dst0, gm_shape &src) {
  TLOAD2_ND2NZ(dst1, dst0, src);
}

template <is_tile_data_v tile_shape, is_global_data_v gm_shape>
void TLOAD2_DN2ZN(tile_shape &dst1, tile_shape &dst0, gm_shape &src) {
  TLOAD2_ND2NZ(dst1, dst0, src);
}

template <is_tile_data_v tile_shape, is_global_data_v gm_shape>
void TSTORE2_DN2DN(gm_shape &dst, tile_shape &src1, tile_shape &src0) {
  supernpubench_v057_fallback::store_tile(dst, src0);
  (void)src1;
}

template <is_tile_data_v tile_shape, is_global_data_v gm_shape>
void TLOAD4_ND2NZ(tile_shape &dst3, tile_shape &dst2, tile_shape &dst1,
                  tile_shape &dst0, gm_shape &src) {
  supernpubench_v057_fallback::load_tile(dst0, src);
  supernpubench_v057_fallback::load_tile(dst1, src);
  supernpubench_v057_fallback::load_tile(dst2, src);
  supernpubench_v057_fallback::load_tile(dst3, src);
}

template <is_tile_data_v tile_shape, is_global_data_v gm_shape>
void TLOAD4_ND2ZN(tile_shape &dst3, tile_shape &dst2, tile_shape &dst1,
                  tile_shape &dst0, gm_shape &src) {
  TLOAD4_ND2NZ(dst3, dst2, dst1, dst0, src);
}

template <is_tile_data_v tile_shape, is_global_data_v gm_shape>
void TLOAD4_DN2ZN(tile_shape &dst3, tile_shape &dst2, tile_shape &dst1,
                  tile_shape &dst0, gm_shape &src) {
  TLOAD4_ND2NZ(dst3, dst2, dst1, dst0, src);
}

template <typename tile_shape_out, typename tile_shape_offset, typename gm_shape,
          TmaPadValue Pad = TmaPadValue::Null>
inline void MGATHER(tile_shape_out &dst, const gm_shape &src,
                    const tile_shape_offset &offset) {
  (void)Pad;
  supernpubench_v057_fallback::gather_bytes(dst, src, offset);
}

template <typename tile_shape_in, typename tile_shape_offset, typename gm_shape>
inline void MSCATTER(gm_shape &dst, const tile_shape_in &src,
                     const tile_shape_offset &offset) {
  supernpubench_v057_fallback::scatter_bytes(dst, src, offset);
}

template <typename tile_shape, typename gm_shape>
inline void TPREFETCH(const gm_shape &src) {
  (void)sizeof(tile_shape);
  (void)src;
}

template <typename tile_shape_out, typename tile_shape_offset,
          typename tile_shape_mask, typename gm_shape,
          TmaPadValue Pad = TmaPadValue::Null>
inline void MGATHER_MASK(tile_shape_out &dst, const gm_shape &src,
                         const tile_shape_offset &offset,
                         const tile_shape_mask &mask) {
  (void)Pad;
  supernpubench_v057_fallback::gather_bytes(dst, src, offset);
  for (int i = 0; i < supernpubench_v057_fallback::tile_numel_v<tile_shape_out> &&
                  i < supernpubench_v057_fallback::tile_numel_v<tile_shape_mask>; ++i)
    if (!mask.data()[i])
      dst.data()[i] = typename tile_shape_out::DType{};
}

template <typename tile_shape_out, typename tile_shape_offset,
          typename tile_shape_expected, typename tile_shape_desired,
          typename gm_shape, TmaPadValue Pad = TmaPadValue::Null>
inline void MGATHER_CAS(tile_shape_out &dst, const gm_shape &src,
                        const tile_shape_offset &offset,
                        const tile_shape_expected &expected,
                        const tile_shape_desired &desired) {
  (void)Pad;
  using DType = typename tile_shape_out::DType;
  auto *base = const_cast<typename gm_shape::DType *>(src.data());
  constexpr int DstN = supernpubench_v057_fallback::tile_numel_v<tile_shape_out>;
  constexpr int OffN =
      supernpubench_v057_fallback::tile_numel_v<tile_shape_offset>;
  constexpr int ExpN =
      supernpubench_v057_fallback::tile_numel_v<tile_shape_expected>;
  constexpr int DesN =
      supernpubench_v057_fallback::tile_numel_v<tile_shape_desired>;
  constexpr int N0 = DstN < OffN ? DstN : OffN;
  constexpr int N1 = ExpN < DesN ? ExpN : DesN;
  constexpr int N = N0 < N1 ? N0 : N1;
  for (int i = 0; i < N; ++i) {
    const long long byte_off = static_cast<long long>(offset.data()[i]);
    auto *ptr =
        reinterpret_cast<DType *>(reinterpret_cast<char *>(base) + byte_off);
    const DType old = *ptr;
    if (old == expected.data()[i])
      *ptr = desired.data()[i];
    dst.data()[i] = old;
  }
}

template <typename tile_shape_in, typename tile_shape_offset,
          typename tile_shape_mask, typename gm_shape>
inline void MSCATTER_MASK(gm_shape &dst, const tile_shape_in &src,
                          const tile_shape_offset &offset,
                          const tile_shape_mask &mask) {
  using DType = typename tile_shape_in::DType;
  auto *base = dst.data();
  constexpr int SrcN = supernpubench_v057_fallback::tile_numel_v<tile_shape_in>;
  constexpr int OffN = supernpubench_v057_fallback::tile_numel_v<tile_shape_offset>;
  constexpr int MaskN = supernpubench_v057_fallback::tile_numel_v<tile_shape_mask>;
  constexpr int N = (SrcN < OffN ? SrcN : OffN) < MaskN ? (SrcN < OffN ? SrcN : OffN) : MaskN;
  for (int i = 0; i < N; ++i) {
    if (!mask.data()[i])
      continue;
    long long byte_off = static_cast<long long>(offset.data()[i]);
    *(reinterpret_cast<DType *>(reinterpret_cast<char *>(base) + byte_off)) =
        src.data()[i];
  }
}

#else

template <is_tile_data_v tile_shape_out, is_tile_data_v tile_shape_in>
void ACCSCALE_T(tile_shape_out &dst, tile_shape_in &src, typename tile_shape_in::DType s) {
  asm volatile(
    "BSTART.CUBE ACCCVT, %c1\n"
    "B.DATR NZ2ND.normal, %c1, Null\n"
    "B.IOT last, ->%q0<%c2>\n"
    "B.IOR [%3],[]\n"
    "C.B.DIMI %c4, ->lb0\n"
    "C.B.DIMI %c5, ->lb1\n"

    ""
    : "=r"(dst.data())
    : "i"(type_traits<typename tile_shape_in::DType>::TypeCode),
      "i"(tile_type_traits<typename tile_shape_out::TileDType>::TilesizeCode),
      "r"(s), "i"(tile_shape_in::ValidCol), "i"(tile_shape_in::ValidRow)
  );
}

template <is_tile_data_v tile_shape_out, is_tile_data_v tile_shape_in>
void ACCSCALE_NZ2DN(tile_shape_out &dst, tile_shape_in &src, typename tile_shape_in::DType s) {
  asm volatile(
    "BSTART.CUBE ACCCVT, %c1\n"
    "B.DATR NZ2DN.normal, %c1, Null\n"
    "B.IOT last, ->%q0<%c2>\n"
    "B.IOR [%3],[]\n"
    "C.B.DIMI %c4, ->lb0\n"
    "C.B.DIMI %c5, ->lb1\n"

    ""
    : "=r"(dst.data())
    : "i"(type_traits<typename tile_shape_in::DType>::TypeCode),
      "i"(tile_type_traits<typename tile_shape_out::TileDType>::TilesizeCode),
      "r"(s), "i"(tile_shape_in::ValidCol), "i"(tile_shape_in::ValidRow)
  );
}

template <is_tile_data_v tile_shape_max, is_tile_data_v tile_shape_out, is_tile_data_v tile_shape_in>
void ACCCVT_RMAX_SCALE_NZ2DN(tile_shape_max &row_max, tile_shape_out &dst, tile_shape_in &src, typename tile_shape_in::DType s) {
  asm volatile(
    "BSTART.CUBE ACCCVT, %c[__pto_SrcType]\n"
    "B.DATR NZ2DN.normal, %c[__pto_DstType], Null\n"
    "B.IOT , ->%q[__pto_dout]<%c[__pto_DstTileSize]>\n"
    "B.IOT last, ->%q[__pto_rmax]<%c[__pto_RmaxTileSize]>\n"
    "B.IOR [%[__pto_scale]],[]\n"
    "C.B.DIMI %c[__pto_VCOL], ->lb0\n"
    "C.B.DIMI %c[__pto_VROW], ->lb1\n"

    ""
    : [__pto_dout]"=r"(dst.data()),  [__pto_rmax]"=r"(row_max.data())
    : [__pto_SrcType]"i"(type_traits<typename tile_shape_in::DType>::TypeCode),
      [__pto_DstType]"i"(type_traits<typename tile_shape_out::DType>::TypeCode),
      [__pto_DstTileSize]"i"(tile_type_traits<typename tile_shape_out::TileDType>::TilesizeCode),
      [__pto_RmaxTileSize]"i"(tile_type_traits<typename tile_shape_max::TileDType>::TilesizeCode),
      [__pto_scale]"r"(s), [__pto_VCOL]"i"(tile_shape_in::ValidCol), [__pto_VROW]"i"(tile_shape_in::ValidRow)
  );
}

template <is_tile_data_v tile_shape_out, is_tile_data_v tile_shape_in0, is_tile_data_v tile_shape_in1>
void TMAX_T(tile_shape_out &dst, tile_shape_in0 &src0, tile_shape_in1 &src1) {
  asm volatile(
    "BSTART.VPAR 0b0000100011, %c3\n"
    "B.IOT %1,%2, last, ->%q0<%c4>\n"
    "C.B.DIMI %c5, ->lb0\n"
    "C.B.DIMI %c6, ->lb1\n"

    ""
    : "=r"(dst.data())
    : "r"(src0.data()), "r"(src1.data()), \
      "i"(type_traits<typename tile_shape_in0::DType>::TypeCode),
      "i"(tile_type_traits<typename tile_shape_out::TileDType>::TilesizeCode),
      "i"(tile_shape_in0::ValidCol), "i"(tile_shape_in0::ValidRow)
  );
}

template <is_tile_data_v tile_shape_out, is_tile_data_v tile_shape_in0, is_tile_data_v tile_shape_in1>
void TSUB_EXP_EXPAND_T(tile_shape_out &dst, tile_shape_in0 &src0, tile_shape_in1 &src1) {
  asm volatile(
    "BSTART.VPAR 0b0001000011, %c3\n"
    "B.IOT %1,%2, last, ->%q0<%c4>\n"
    "C.B.DIMI %c5, ->lb0\n"
    "C.B.DIMI %c6, ->lb1\n"

    ""
    : "=r"(dst.data())
    : "r"(src0.data()), "r"(src1.data()), \
      "i"(type_traits<typename tile_shape_in0::DType>::TypeCode),
      "i"(tile_type_traits<typename tile_shape_out::TileDType>::TilesizeCode),
      "i"(tile_shape_in0::ValidCol), "i"(tile_shape_in0::ValidRow)
  );
}

template <is_tile_data_v tile_shape_out, is_tile_data_v tile_shape_in0, is_tile_data_v tile_shape_in1, is_tile_data_v tile_shape_in2>
void TMUL_ADD_ROWSUM_T(tile_shape_out &dst, tile_shape_in0 &src0, tile_shape_in1 &src1, tile_shape_in2 &src2) {
  asm volatile(
    "BSTART.VPAR 0b0001100011, %c4\n"
    "B.IOT %1,%2, last, ->%q0<%c5>\n"
    "B.IOT %3, 1\n"
    "C.B.DIMI %c6, ->lb0\n"
    "C.B.DIMI %c7, ->lb1\n"

    ""
    : "=r"(dst.data())
    : "r"(src0.data()), "r"(src1.data()), "r"(src2.data()),
      "i"(type_traits<typename tile_shape_in0::DType>::TypeCode),
      "i"(tile_type_traits<typename tile_shape_out::TileDType>::TilesizeCode),
      "i"(tile_shape_in0::ValidCol), "i"(tile_shape_in0::ValidRow)
  );
}

template <is_tile_data_v tile_shape_out, is_tile_data_v tile_shape_in0, is_tile_data_v tile_shape_in1, is_tile_data_v tile_shape_in2>
void TADD_MUL_EXPAND_T(tile_shape_out &dst, tile_shape_in0 &src0, tile_shape_in1 &src1, tile_shape_in2 &src2) {
  asm volatile(
    "BSTART.VPAR 0b0010000011, %c4\n"
    "B.IOT %1,%2, last, ->%q0<%c5>\n"
    "B.IOT %3, 1\n"
    "C.B.DIMI %c6, ->lb0\n"
    "C.B.DIMI %c7, ->lb1\n"

    ""
    : "=r"(dst.data())
    : "r"(src0.data()), "r"(src1.data()), "r"(src2.data()),
      "i"(type_traits<typename tile_shape_in0::DType>::TypeCode),
      "i"(tile_type_traits<typename tile_shape_out::TileDType>::TilesizeCode),
      "i"(tile_shape_in0::ValidCol), "i"(tile_shape_in0::ValidRow)
  );
}

template <is_tile_data_v tile_shape_out, is_tile_data_v tile_shape_in>
void TCVT_T(tile_shape_out &dst,  tile_shape_in &src) {
  asm volatile(
    "BSTART.TCVT %c1\n"
    "B.DATR %c2, RNone\n"
    "B.IOT %3, last, ->%q0<%c4>\n"
    "B.DIM zero, %c5, ->lb0\n"
    "B.DIM zero, %c6, ->lb1\n"
    : "=r"(dst.data())
    : "i"(type_traits<typename tile_shape_in::DType>::TypeCode),
      "i"(type_traits<typename tile_shape_out::DType>::TypeCode),
      "r"(src.data()),
      "i"(tile_type_traits<typename tile_shape_out::TileDType>::TilesizeCode),
      "i"(tile_shape_in::ValidCol),
      "i"(tile_shape_in::ValidRow)
  );
}

#define DEFINE_TMOV_LAYOUT(LAYOUT_NAME)                                          \
template <is_tile_data_v tile_shape_out, is_tile_data_v tile_shape_in>           \
void TMOV_##LAYOUT_NAME(tile_shape_out &dst, tile_shape_in &src) {               \
  asm volatile(                                                                  \
    "BSTART.TMOV %c2\n"                                                          \
    "B.DATR " #LAYOUT_NAME ".normal, Null\n"                                     \
    "B.IOT %1, last, ->%q0<%c3>\n"                                              \
    "B.DIM zero, %c4, ->lb0\n"                                                   \
    "B.DIM zero, %c5, ->lb1\n"                                                   \
    : "=r"(dst.data())                                                          \
    : "r"(src.data()),                                                          \
      "i"(type_traits<typename tile_shape_in::DType>::TypeCode),                 \
      "i"(tile_type_traits<typename tile_shape_out::TileDType>::TilesizeCode),   \
      "i"(tile_shape_in::ValidCol),                                              \
      "i"(tile_shape_in::ValidRow)                                               \
  );                                                                             \
}

DEFINE_TMOV_LAYOUT(ND2NZ)
DEFINE_TMOV_LAYOUT(NZ2ND)
DEFINE_TMOV_LAYOUT(ND2ZN)
DEFINE_TMOV_LAYOUT(DN2ZN)
DEFINE_TMOV_LAYOUT(DN2NZ)
DEFINE_TMOV_LAYOUT(NZ2DN)
DEFINE_TMOV_LAYOUT(NZ2ZN)
DEFINE_TMOV_LAYOUT(ZN2NZ)
DEFINE_TMOV_LAYOUT(NORM)

template <is_tile_data_v tile_shape_out, is_tile_data_v tile_shape_in>
void TMOV_DN2NZ_DYN(tile_shape_out &dst, tile_shape_in &src) {
  asm volatile(
    "BSTART.TMOV %c2\n"
    "B.DATR DN2NZ.normal, Null\n"
    "B.IOT %1, last, ->%q0<%c3>\n"
    "B.DIM %4, 0, ->lb0\n"
    "B.DIM %5, 0, ->lb1\n"

    : "=r"(dst.data())
    : "r"(src.data()),
      "i"(type_traits<typename tile_shape_in::DType>::TypeCode),
      "i"(tile_type_traits<typename tile_shape_out::TileDType>::TilesizeCode),
      "r"(src.GetValidCol()),
      "r"(src.GetValidRow())
  );
}

template <is_tile_data_v tile_shape_out, is_tile_data_v tile_shape_in>
void THISTOGRAM(tile_shape_out &dst, tile_shape_in &src, tile_shape_in &Idx, int ByteId) {
#define THISTOGRAM_ASM(BYTE_NAME)                                      \
  asm volatile(                                                        \
    "BSTART.THISTOGRAM %c1\n"                                     \
    "B.DATR %c2," BYTE_NAME ",Null\n"                                  \
    "B.DIM zero, %c3, ->LB0\n"                                         \
    "B.DIM zero, %c4, ->LB1\n"                                         \
    "B.DIM zero, %c4, ->LB2\n"                                         \
    "B.IOT %5, %6, last, ->%q0<%c7>\n"                                \
    ""                                                                 \
    : "=r"(dst.data())                                                \
    : "i"(type_traits<typename tile_shape_in::DType>::TypeCode),       \
      "i"(type_traits<typename tile_shape_out::DType>::TypeCode),      \
      "i"(tile_shape_in::ValidCol),                                    \
      "i"(tile_shape_in::ValidRow),                                    \
      "i"(tile_shape_in::Cols),                                        \
      "r"(src.data()),                                                \
      "r"(Idx.data()),                                                \
      "i"(tile_type_traits<typename tile_shape_out::TileDType>::TilesizeCode))

  switch (ByteId) {
    case 0:
      THISTOGRAM_ASM("Byte0");
      break;
    case 1:
      THISTOGRAM_ASM("Byte1");
      break;
    case 2:
      THISTOGRAM_ASM("Byte2");
      break;
    case 3:
      THISTOGRAM_ASM("Byte3");
      break;
    default:
      return;  // ByteId > 3 或 < 0，无效
  }

#undef THISTOGRAM_ASM
}


template <is_tile_data_v tile_shape, is_global_data_v gm_shape>
void TLOAD2_ND2NZ(tile_shape &dst1, tile_shape &dst0, gm_shape &src) {
  static_assert(gm_shape::isRowMajor && is_Nz_layout<tile_shape>::value,
                    "GM_SHAPE should ND and TILE_SHAPE should be Nz ");
  asm volatile(
    "BSTART.TLOAD %c[__pto_SrcType]\n"
    "B.DATR ND2NZ.normal, %c[__pto_DstType], Null\n"
    "B.DIM zero, %[__pto_VCOL], ->lb0\n"
    "B.DIM zero, %[__pto_VROW], ->lb1\n"
    "B.DIM zero, %[__pto_COL], ->lb2\n"
    "B.IOT , ->%q[__pto_d0]<%c[__pto_TileSize]>\n"
    "B.IOT last, ->%q[__pto_d1]<%c[__pto_TileSize]>\n"
    "B.IOR [%[__pto_s0],%[__pto_GmStride]], []\n"
    : [__pto_d0]"=r"(dst0.data()),[__pto_d1]"=r"(dst1.data())
    : [__pto_s0]"r"(src.data()),
      [__pto_DstType]"i"(type_traits<typename tile_shape::DType>::TypeCode),
      [__pto_SrcType]"i"(type_traits<typename gm_shape::DType>::TypeCode),
      [__pto_TileSize]"i"(tile_type_traits<typename tile_shape::TileDType>::TilesizeCode),
      [__pto_VCOL]"i"(tile_shape::ValidCol*2), [__pto_VROW]"i"(tile_shape::ValidRow), [__pto_COL]"i"(tile_shape::Cols*2),
      [__pto_GmStride]"r"(gm_shape::RowStride * sizeof(typename gm_shape::DType))
  );
}

template <is_tile_data_v tile_shape, is_global_data_v gm_shape>
void TLOAD2_ND2ZN(tile_shape &dst1, tile_shape &dst0, gm_shape &src) {
  static_assert(gm_shape::isRowMajor && is_Zn_layout<tile_shape>::value,
                    "GM_SHAPE should ND and TILE_SHAPE should be Zn ");
  asm volatile(
    "BSTART.TLOAD %c[__pto_SrcType]\n"
    "B.DATR ND2ZN.normal, %c[__pto_DstType], Null\n"
    "B.DIM zero, %[__pto_VCOL], ->lb0\n"
    "B.DIM zero, %[__pto_VROW], ->lb1\n"
    "B.DIM zero, %[__pto_COL], ->lb2\n"
    "B.IOT , ->%q[__pto_d0]<%c[__pto_TileSize]>\n"
    "B.IOT last, ->%q[__pto_d1]<%c[__pto_TileSize]>\n"
    "B.IOR [%[__pto_s0],%[__pto_GmStride]], []\n"
    : [__pto_d0]"=r"(dst0.data()),[__pto_d1]"=r"(dst1.data())
    : [__pto_s0]"r"(src.data()),
      [__pto_DstType]"i"(type_traits<typename tile_shape::DType>::TypeCode),
      [__pto_SrcType]"i"(type_traits<typename gm_shape::DType>::TypeCode),
      [__pto_TileSize]"i"(tile_type_traits<typename tile_shape::TileDType>::TilesizeCode),
      [__pto_VCOL]"i"(tile_shape::ValidCol*2), [__pto_VROW]"i"(tile_shape::ValidRow), [__pto_COL]"i"(tile_shape::Cols*2),
      [__pto_GmStride]"r"(gm_shape::RowStride * sizeof(typename gm_shape::DType))
  );
}

template <is_tile_data_v tile_shape, is_global_data_v gm_shape>
void TLOAD2_DN2ZN(tile_shape &dst1, tile_shape &dst0, gm_shape &src) {
  static_assert(!gm_shape::isRowMajor && is_Nz_layout<tile_shape>::value,
                    "GM_SHAPE should DN and TILE_SHAPE should be Zn ");
  asm volatile(
    "BSTART.TLOAD %c[__pto_SrcType]\n"
    "B.DATR DN2ZN.normal, %c[__pto_DstType], Null\n"
    "B.DIM zero, %[__pto_VCOL], ->lb0\n"
    "B.DIM zero, %[__pto_VROW], ->lb1\n"
    "B.DIM zero, %[__pto_COL], ->lb2\n"
    "B.IOT , ->%q[__pto_d0]<%c[__pto_TileSize]>\n"
    "B.IOT last, ->%q[__pto_d1]<%c[__pto_TileSize]>\n"
    "B.IOR [%[__pto_s0],%[__pto_GmStride]], []\n"
    : [__pto_d0]"=r"(dst0.data()),[__pto_d1]"=r"(dst1.data())
    : [__pto_s0]"r"(src.data()),
      [__pto_DstType]"i"(type_traits<typename tile_shape::DType>::TypeCode),
      [__pto_SrcType]"i"(type_traits<typename gm_shape::DType>::TypeCode),
      [__pto_TileSize]"i"(tile_type_traits<typename tile_shape::TileDType>::TilesizeCode),
      [__pto_VCOL]"i"(tile_shape::ValidCol), [__pto_VROW]"i"(tile_shape::ValidRow*2), [__pto_COL]"i"(tile_shape::Cols),
      [__pto_GmStride]"r"(gm_shape::ColStride * sizeof(typename gm_shape::DType))
  );
}

template <is_tile_data_v tile_shape, is_global_data_v gm_shape>
void TSTORE2_DN2DN(gm_shape &dst, tile_shape &src1, tile_shape &src0) {
  static_assert(!gm_shape::isRowMajor && !tile_shape::isRowMajor,
                    "GM_SHAPE should DN and TILE_SHAPE should be DN");
  asm volatile(
    "BSTART.TSTORE %c[__pto_SrcType]\n"
    "B.DATR NORM.normal, %c[__pto_DstType], Null\n"
    "B.DIM zero, %[__pto_VCOL], ->lb0\n"
    "B.DIM zero, %[__pto_VROW], ->lb1\n"
    "B.DIM zero, %[__pto_COL], ->lb2\n"
    "B.IOT %[__pto_s0], %[s1], last\n"
    "B.IOR [%[__pto_d0],%[__pto_GmStride]], []\n"
    :
    : [__pto_d0]"r"(dst.data()), [__pto_s0]"r"(src0.data()), [s1]"r"(src1.data()),
      [__pto_DstType]"i"(type_traits<typename gm_shape::DType>::TypeCode),
      [__pto_SrcType]"i"(type_traits<typename tile_shape::DType>::TypeCode),
      [__pto_VCOL]"i"(tile_shape::ValidRow*2), [__pto_VROW]"i"(tile_shape::ValidCol), [__pto_COL]"i"(tile_shape::Rows*2),
      [__pto_GmStride]"r"(gm_shape::ColStride * sizeof(typename gm_shape::DType))
  );
}

template <is_tile_data_v tile_shape, is_global_data_v gm_shape>
void TLOAD4_ND2NZ(tile_shape &dst3, tile_shape &dst2, tile_shape &dst1, tile_shape &dst0, gm_shape &src) {
  static_assert(gm_shape::isRowMajor && is_Nz_layout<tile_shape>::value,
                    "GM_SHAPE should ND and TILE_SHAPE should be Nz ");
  asm volatile(
    "BSTART.TLOAD %c[__pto_SrcType]\n"
    "B.DATR ND2NZ.normal, %c[__pto_DstType], Null\n"
    "B.DIM zero, %[__pto_VCOL], ->lb0\n"
    "B.DIM zero, %[__pto_VROW], ->lb1\n"
    "B.DIM zero, %[__pto_COL], ->lb2\n"
    "B.IOT , ->%q[__pto_d0]<%c[__pto_TileSize]>\n"
    "B.IOT , ->%q[__pto_d1]<%c[__pto_TileSize]>\n"
    "B.IOT , ->%q[d2]<%c[__pto_TileSize]>\n"
    "B.IOT last, ->%q[d3]<%c[__pto_TileSize]>\n"
    "B.IOR [%[__pto_s0],%[__pto_GmStride]], []\n"
    : [__pto_d0]"=r"(dst0.data()),[__pto_d1]"=r"(dst1.data()),[d2]"=r"(dst2.data()),[d3]"=r"(dst3.data())
    : [__pto_s0]"r"(src.data()),
      [__pto_DstType]"i"(type_traits<typename tile_shape::DType>::TypeCode),
      [__pto_SrcType]"i"(type_traits<typename gm_shape::DType>::TypeCode),
      [__pto_TileSize]"i"(tile_type_traits<typename tile_shape::TileDType>::TilesizeCode),
      [__pto_VCOL]"i"(tile_shape::ValidCol*4), [__pto_VROW]"i"(tile_shape::ValidRow), [__pto_COL]"i"(tile_shape::Cols*4),
      [__pto_GmStride]"r"(gm_shape::RowStride * sizeof(typename gm_shape::DType))
  );
}

template <is_tile_data_v tile_shape, is_global_data_v gm_shape>
void TLOAD4_ND2ZN(tile_shape &dst3, tile_shape &dst2, tile_shape &dst1, tile_shape &dst0, gm_shape &src) {
  static_assert(gm_shape::isRowMajor && is_Zn_layout<tile_shape>::value,
                    "GM_SHAPE should ND and TILE_SHAPE should be Nz ");
  asm volatile(
    "BSTART.TLOAD %c[__pto_SrcType]\n"
    "B.DATR ND2ZN.normal, %c[__pto_DstType], Null\n"
    "B.DIM zero, %[__pto_VCOL], ->lb0\n"
    "B.DIM zero, %[__pto_VROW], ->lb1\n"
    "B.DIM zero, %[__pto_COL], ->lb2\n"
    "B.IOT , ->%q[__pto_d0]<%c[__pto_TileSize]>\n"
    "B.IOT , ->%q[__pto_d1]<%c[__pto_TileSize]>\n"
    "B.IOT , ->%q[d2]<%c[__pto_TileSize]>\n"
    "B.IOT last, ->%q[d3]<%c[__pto_TileSize]>\n"
    "B.IOR [%[__pto_s0],%[__pto_GmStride]], []\n"
    : [__pto_d0]"=r"(dst0.data()),[__pto_d1]"=r"(dst1.data()),[d2]"=r"(dst2.data()),[d3]"=r"(dst3.data())
    : [__pto_s0]"r"(src.data()),
      [__pto_DstType]"i"(type_traits<typename tile_shape::DType>::TypeCode),
      [__pto_SrcType]"i"(type_traits<typename gm_shape::DType>::TypeCode),
      [__pto_TileSize]"i"(tile_type_traits<typename tile_shape::TileDType>::TilesizeCode),
      [__pto_VCOL]"i"(tile_shape::ValidRow*4), [__pto_VROW]"i"(tile_shape::ValidCol), [__pto_COL]"i"(tile_shape::Rows*4),
      [__pto_GmStride]"r"(gm_shape::RowStride * sizeof(typename gm_shape::DType))
  );
}

template <is_tile_data_v tile_shape, is_global_data_v gm_shape>
void TLOAD4_DN2ZN(tile_shape &dst3, tile_shape &dst2, tile_shape &dst1, tile_shape &dst0, gm_shape &src) {
  static_assert(!gm_shape::isRowMajor && is_Zn_layout<tile_shape>::value,
                    "GM_SHAPE should DN and TILE_SHAPE should be Zn ");
  asm volatile(
    "BSTART.TLOAD %c[__pto_SrcType]\n"
    "B.DATR DN2ZN.normal, %c[__pto_DstType], Null\n"
    "B.DIM zero, %[__pto_VCOL], ->lb0\n"
    "B.DIM zero, %[__pto_VROW], ->lb1\n"
    "B.DIM zero, %[__pto_COL], ->lb2\n"
    "B.IOT , ->%q[__pto_d0]<%c[__pto_TileSize]>\n"
    "B.IOT , ->%q[__pto_d1]<%c[__pto_TileSize]>\n"
    "B.IOT , ->%q[d2]<%c[__pto_TileSize]>\n"
    "B.IOT last, ->%q[d3]<%c[__pto_TileSize]>\n"
    "B.IOR [%[__pto_s0],%[__pto_GmStride]], []\n"
    : [__pto_d0]"=r"(dst0.data()),[__pto_d1]"=r"(dst1.data()),[d2]"=r"(dst2.data()),[d3]"=r"(dst3.data())
    : [__pto_s0]"r"(src.data()),
      [__pto_DstType]"i"(type_traits<typename tile_shape::DType>::TypeCode),
      [__pto_SrcType]"i"(type_traits<typename gm_shape::DType>::TypeCode),
      [__pto_TileSize]"i"(tile_type_traits<typename tile_shape::TileDType>::TilesizeCode),
      [__pto_VCOL]"i"(tile_shape::ValidCol*4), [__pto_VROW]"i"(tile_shape::ValidRow), [__pto_COL]"i"(tile_shape::Cols*4),
      [__pto_GmStride]"r"(gm_shape::ColStride * sizeof(typename gm_shape::DType))
  );
}

enum class TmaPadValue : int {
  Zero = 0,
  Max = 1,
  Min = 2,
  Null = 3,
};

template <typename tile_shape_out, typename tile_shape_offset, typename gm_shape,
          TmaPadValue Pad = TmaPadValue::Null>
inline void MGATHER(tile_shape_out &dst, const gm_shape &src,
                    const tile_shape_offset &offset) {
  static_assert(tile_shape_offset::ValidCol <= tile_shape_offset::Cols, "");
  asm volatile(
      "BSTART.MGATHER %c[DataType]\n"
      "B.DATR %c[PadValue]\n"
      "B.DIM zero, %c[ValidCol], ->LB0\n"
      "B.DIM zero, %c[ValidRow], ->LB1\n"
      "B.DIM zero, %c[Col], ->LB2\n"
      "B.IOT %[off], last, ->%q[dst]<%c[TileSize]>\n"
      "B.IOR [%[base]], []\n"
      : [dst] "=r"(dst.data())
      : [base] "r"(src.data()), [off] "r"(offset.data()),
        [DataType] "i"(type_traits<typename tile_shape_out::DType>::TypeCode),
        [PadValue] "i"(static_cast<int>(Pad)),
        [TileSize] "i"(
            tile_type_traits<typename tile_shape_out::TileDType>::TilesizeCode),
        [ValidCol] "i"(tile_shape_offset::ValidCol),
        [ValidRow] "i"(tile_shape_offset::ValidRow),
        [Col] "i"(tile_shape_offset::Cols)
      : "memory");
}

template <typename tile_shape_in, typename tile_shape_offset, typename gm_shape>
inline void MSCATTER(gm_shape &dst, const tile_shape_in &src,
                     const tile_shape_offset &offset) {
  static_assert(tile_shape_offset::ValidCol <= tile_shape_offset::Cols, "");
  asm volatile(
      "BSTART.MSCATTER %c[DataType]\n"
      "B.DIM zero, %c[ValidCol], ->LB0\n"
      "B.DIM zero, %c[ValidRow], ->LB1\n"
      "B.DIM zero, %c[Col], ->LB2\n"
      "B.IOT %[src], %[off], last\n"
      "B.IOR [%[base]], []\n"
      :
      : [base] "r"(dst.data()), [src] "r"(src.data()),
        [off] "r"(offset.data()),
        [DataType] "i"(type_traits<typename tile_shape_in::DType>::TypeCode),
        [ValidCol] "i"(tile_shape_offset::ValidCol),
        [ValidRow] "i"(tile_shape_offset::ValidRow),
        [Col] "i"(tile_shape_offset::Cols)
      : "memory");
}

template <typename tile_shape, typename gm_shape>
inline void TPREFETCH(const gm_shape &src) {
  asm volatile(
      "BSTART.TPREFETCH %c[DataType]\n"
      "B.DATR NORM.normal, %c[DataType], Null\n"
      "B.DIM zero, %c[ValidCol], ->LB0\n"
      "B.DIM zero, %c[ValidRow], ->LB1\n"
      "B.DIM zero, %c[Col], ->LB2\n"
      "B.IOR [%[base]], []\n"
      :
      : [base] "r"(src.data()),
        [DataType] "i"(type_traits<typename tile_shape::DType>::TypeCode),
        [ValidCol] "i"(tile_shape::ValidCol),
        [ValidRow] "i"(tile_shape::ValidRow),
        [Col] "i"(tile_shape::Cols)
      : "memory");
}

template <typename tile_shape_out, typename tile_shape_offset,
          typename tile_shape_mask, typename gm_shape,
          TmaPadValue Pad = TmaPadValue::Null>
inline void MGATHER_MASK(tile_shape_out &dst, const gm_shape &src,
                         const tile_shape_offset &offset,
                         const tile_shape_mask &mask) {
  asm volatile(
      "BSTART.MGATHER.MASK %c[DataType]\n"
      "B.DATR %c[PadValue]\n"
      "B.DIM zero, %c[Col], ->LB0\n"
      "B.DIM zero, %c[Row], ->LB1\n"
      "B.IOT %[off], ->%q[dst]<%c[TileSize]>\n"
      "B.IOT %[mask], last\n"
      "B.IOR [%[base]], []\n"
      : [dst] "=r"(dst.data())
      : [base] "r"(src.data()), [off] "r"(offset.data()),
        [mask] "r"(mask.data()),
        [DataType] "i"(type_traits<typename tile_shape_out::DType>::TypeCode),
        [PadValue] "i"(static_cast<int>(Pad)),
        [TileSize] "i"(
            tile_type_traits<typename tile_shape_out::TileDType>::TilesizeCode),
        [Col] "i"(tile_shape_offset::Cols), [Row] "i"(tile_shape_offset::Rows)
      : "memory");
}

template <typename tile_shape_out, typename tile_shape_offset,
          typename tile_shape_expected, typename tile_shape_desired,
          typename gm_shape, TmaPadValue Pad = TmaPadValue::Null>
inline void MGATHER_CAS(tile_shape_out &dst, const gm_shape &src,
                        const tile_shape_offset &offset,
                        const tile_shape_expected &expected,
                        const tile_shape_desired &desired) {
  static_assert(tile_shape_offset::ValidCol <= tile_shape_offset::Cols, "");
  static_assert(tile_shape_expected::Rows == tile_shape_desired::Rows, "");
  static_assert(tile_shape_expected::Cols == tile_shape_desired::Cols, "");
  asm volatile(
      "BSTART.MGATHER.CAS %c[DataType]\n"
      "B.DATR %c[PadValue]\n"
      "B.DIM zero, %c[ValidCol], ->LB0\n"
      "B.DIM zero, %c[ValidRow], ->LB1\n"
      "B.DIM zero, %c[Col], ->LB2\n"
      "B.IOT %[off], %[expected]\n"
      "B.IOT %[desired], last, ->%q[dst]<%c[TileSize]>\n"
      "B.IOR [%[base]], []\n"
      : [dst] "=r"(dst.data())
      : [base] "r"(src.data()), [off] "r"(offset.data()),
        [expected] "r"(expected.data()), [desired] "r"(desired.data()),
        [DataType] "i"(type_traits<typename tile_shape_out::DType>::TypeCode),
        [PadValue] "i"(static_cast<int>(Pad)),
        [TileSize] "i"(
            tile_type_traits<typename tile_shape_out::TileDType>::TilesizeCode),
        [ValidCol] "i"(tile_shape_offset::ValidCol),
        [ValidRow] "i"(tile_shape_offset::ValidRow),
        [Col] "i"(tile_shape_offset::Cols)
      : "memory");
}

template <typename tile_shape_in, typename tile_shape_offset,
          typename tile_shape_mask, typename gm_shape>
inline void MSCATTER_MASK(gm_shape &dst, const tile_shape_in &src,
                          const tile_shape_offset &offset,
                          const tile_shape_mask &mask) {
  asm volatile(
      "BSTART.MSCATTER.MASK %c[DataType]\n"
      "B.DIM zero, %c[Col], ->LB0\n"
      "B.DIM zero, %c[Row], ->LB1\n"
      "B.IOT %[src], %[off]\n"
      "B.IOT %[mask], last\n"
      "B.IOR [%[base]], []\n"
      :
      : [base] "r"(dst.data()), [src] "r"(src.data()),
        [off] "r"(offset.data()), [mask] "r"(mask.data()),
        [DataType] "i"(type_traits<typename tile_shape_in::DType>::TypeCode),
        [Col] "i"(tile_shape_offset::Cols), [Row] "i"(tile_shape_offset::Rows)
      : "memory");
}

template <is_tile_data_v tile_shape>
inline void TXOR(tile_shape &dst, tile_shape &src0, tile_shape &src1) {
  asm volatile(
      "BSTART.TXOR %c[DataType]\n"
      "B.DATR Null\n"
      "C.B.DIMI %c[ValidCol], ->LB0\n"
      "C.B.DIMI %c[ValidRow], ->LB1\n"
      "C.B.DIMI %c[Cols], ->LB2\n"
      "B.IOT %[src0], %[src1], last, ->%q[dst]<%c[TileSize]>\n"
      : [dst] "=r"(dst.data())
      : [src0] "r"(src0.data()), [src1] "r"(src1.data()),
        [DataType] "i"(type_traits<typename tile_shape::DType>::TypeCode),
        [ValidCol] "i"(tile_shape::ValidCol),
        [ValidRow] "i"(tile_shape::ValidRow), [Cols] "i"(tile_shape::Cols),
        [TileSize] "i"(
            tile_type_traits<typename tile_shape::TileDType>::TilesizeCode));
}

template <is_tile_data_v tile_shape>
inline void TSHLS(tile_shape &dst, tile_shape &src, unsigned shift) {
  asm volatile(
      "BSTART.TSHLS %c[DataType]\n"
      "B.DATR Null\n"
      "C.B.DIMI %c[ValidCol], ->LB0\n"
      "C.B.DIMI %c[ValidRow], ->LB1\n"
      "C.B.DIMI %c[Cols], ->LB2\n"
      "B.IOT %[src], last, ->%q[dst]<%c[TileSize]>\n"
      "B.IOR [%[shift]], []\n"
      : [dst] "=r"(dst.data())
      : [src] "r"(src.data()), [shift] "r"(shift),
        [DataType] "i"(type_traits<typename tile_shape::DType>::TypeCode),
        [ValidCol] "i"(tile_shape::ValidCol),
        [ValidRow] "i"(tile_shape::ValidRow), [Cols] "i"(tile_shape::Cols),
        [TileSize] "i"(
            tile_type_traits<typename tile_shape::TileDType>::TilesizeCode));
}

template <is_tile_data_v tile_shape>
inline void TSHRS(tile_shape &dst, tile_shape &src, unsigned shift) {
  asm volatile(
      "BSTART.TSHRS %c[DataType]\n"
      "B.DATR Null\n"
      "C.B.DIMI %c[ValidCol], ->LB0\n"
      "C.B.DIMI %c[ValidRow], ->LB1\n"
      "C.B.DIMI %c[Cols], ->LB2\n"
      "B.IOT %[src], last, ->%q[dst]<%c[TileSize]>\n"
      "B.IOR [%[shift]], []\n"
      : [dst] "=r"(dst.data())
      : [src] "r"(src.data()), [shift] "r"(shift),
        [DataType] "i"(type_traits<typename tile_shape::DType>::TypeCode),
        [ValidCol] "i"(tile_shape::ValidCol),
        [ValidRow] "i"(tile_shape::ValidRow), [Cols] "i"(tile_shape::Cols),
        [TileSize] "i"(
            tile_type_traits<typename tile_shape::TileDType>::TilesizeCode));
}

template <is_tile_data_v tile_shape, is_tile_data_v tile_shape_index>
inline void TSELECT_V057(tile_shape &dst, tile_shape_index &cond,
                         tile_shape &src0, tile_shape &src1) {
  typename tile_shape::TileDType tmp;
  asm volatile(
      "BSTART.TSEL %c[DataType]\n"
      "B.DATR Null\n"
      "C.B.DIMI %c[ValidCol], ->LB0\n"
      "C.B.DIMI %c[ValidRow], ->LB1\n"
      "C.B.DIMI %c[Cols], ->LB2\n"
      "B.IOT %[cond], %[src0], ->%q[dst]<%c[TileSize]>\n"
      "B.IOT %[src1], last, ->%q[tmp]<%c[TileSize]>\n"
      : [dst] "=r"(dst.data()), [tmp] "=r"(tmp)
      : [cond] "r"(cond.data()), [src0] "r"(src0.data()),
        [src1] "r"(src1.data()),
        [DataType] "i"(type_traits<typename tile_shape::DType>::TypeCode),
        [ValidCol] "i"(tile_shape::ValidCol),
        [ValidRow] "i"(tile_shape::ValidRow), [Cols] "i"(tile_shape::Cols),
        [TileSize] "i"(
            tile_type_traits<typename tile_shape::TileDType>::TilesizeCode));
}

// v0.57 named TEPL/TMA surface.  These wrappers replace the retired vendor
// tile_size-based per-element implementation: tile values stay opaque and are
// produced/consumed only by architectural tile blocks.
#define SUPERNPUBENCH_V057_TEPL_BINARY(FUNCTION, WRAPPER)                       \
  template <is_tile_data_v tile_shape>                                         \
  inline void WRAPPER(tile_shape &dst, tile_shape &src0, tile_shape &src1) {    \
    asm volatile(                                                               \
        "BSTART." #FUNCTION " %c[DataType]\n"                            \
        "B.DATR Null\n"                                                        \
        "C.B.DIMI %c[ValidCol], ->LB0\n"                                       \
        "C.B.DIMI %c[ValidRow], ->LB1\n"                                       \
        "C.B.DIMI %c[Cols], ->LB2\n"                                           \
        "B.IOT %[src0], %[src1], last, ->%q[dst]<%c[TileSize]>\n"             \
        : [dst] "=r"(dst.data())                                                \
        : [src0] "r"(src0.data()), [src1] "r"(src1.data()),                    \
          [DataType] "i"(type_traits<typename tile_shape::DType>::TypeCode),   \
          [ValidCol] "i"(tile_shape::ValidCol),                                \
          [ValidRow] "i"(tile_shape::ValidRow), [Cols] "i"(tile_shape::Cols),  \
          [TileSize] "i"(                                                      \
              tile_type_traits<typename tile_shape::TileDType>::TilesizeCode));\
  }

SUPERNPUBENCH_V057_TEPL_BINARY(TADD, TADD_V057)
SUPERNPUBENCH_V057_TEPL_BINARY(TSUB, TSUB_V057)
SUPERNPUBENCH_V057_TEPL_BINARY(TMUL, TMUL_V057)
SUPERNPUBENCH_V057_TEPL_BINARY(TAND, TAND_V057)
SUPERNPUBENCH_V057_TEPL_BINARY(TOR, TOR_V057)
SUPERNPUBENCH_V057_TEPL_BINARY(TREM, TREM_V057)
#undef SUPERNPUBENCH_V057_TEPL_BINARY

#define SUPERNPUBENCH_V057_TEPL_SCALAR(FUNCTION, WRAPPER)                       \
  template <is_tile_data_v tile_shape>                                         \
  inline void WRAPPER(tile_shape &dst, tile_shape &src,                         \
                      typename tile_shape::DType scalar) {                      \
    asm volatile(                                                               \
        "BSTART." #FUNCTION " %c[DataType]\n"                            \
        "B.DATR Null\n"                                                        \
        "C.B.DIMI %c[ValidCol], ->LB0\n"                                       \
        "C.B.DIMI %c[ValidRow], ->LB1\n"                                       \
        "C.B.DIMI %c[Cols], ->LB2\n"                                           \
        "B.IOT %[src], last, ->%q[dst]<%c[TileSize]>\n"                       \
        "B.IOR [%[scalar]], []\n"                                              \
        : [dst] "=r"(dst.data())                                                \
        : [src] "r"(src.data()), [scalar] "r"(scalar),                         \
          [DataType] "i"(type_traits<typename tile_shape::DType>::TypeCode),   \
          [ValidCol] "i"(tile_shape::ValidCol),                                \
          [ValidRow] "i"(tile_shape::ValidRow), [Cols] "i"(tile_shape::Cols),  \
          [TileSize] "i"(                                                      \
              tile_type_traits<typename tile_shape::TileDType>::TilesizeCode));\
  }

SUPERNPUBENCH_V057_TEPL_SCALAR(TADDS, TADDS_V057)
SUPERNPUBENCH_V057_TEPL_SCALAR(TMULS, TMULS_V057)
#undef SUPERNPUBENCH_V057_TEPL_SCALAR

template <is_tile_data_v tile_shape>
inline void TEXPANDS_V057(tile_shape &dst,
                          typename tile_shape::DType scalar) {
  asm volatile(
      "BSTART.TEXPANDS %c[DataType]\n"
      "B.DATR Null\n"
      "C.B.DIMI %c[ValidCol], ->LB0\n"
      "C.B.DIMI %c[ValidRow], ->LB1\n"
      "C.B.DIMI %c[Cols], ->LB2\n"
      "B.IOT last, ->%q[dst]<%c[TileSize]>\n"
      "B.IOR [%[scalar]], []\n"
      : [dst] "=r"(dst.data())
      : [scalar] "r"(scalar),
        [DataType] "i"(type_traits<typename tile_shape::DType>::TypeCode),
        [ValidCol] "i"(tile_shape::ValidCol),
        [ValidRow] "i"(tile_shape::ValidRow), [Cols] "i"(tile_shape::Cols),
        [TileSize] "i"(
            tile_type_traits<typename tile_shape::TileDType>::TilesizeCode));
}

template <is_tile_data_v tile_shape_out, is_tile_data_v tile_shape_in>
inline void TCVT_V057(tile_shape_out &dst, tile_shape_in &src) {
  asm volatile(
      "BSTART.TCVT %c[SrcType]\n"
      "B.DATR %c[DstType], RNONE\n"
      "C.B.DIMI %c[ValidCol], ->LB0\n"
      "C.B.DIMI %c[ValidRow], ->LB1\n"
      "C.B.DIMI %c[Cols], ->LB2\n"
      "B.IOT %[src], last, ->%q[dst]<%c[TileSize]>\n"
      : [dst] "=r"(dst.data())
      : [src] "r"(src.data()),
        [SrcType] "i"(type_traits<typename tile_shape_in::DType>::TypeCode),
        [DstType] "i"(type_traits<typename tile_shape_out::DType>::TypeCode),
        [ValidCol] "i"(tile_shape_in::ValidCol),
        [ValidRow] "i"(tile_shape_in::ValidRow),
        [Cols] "i"(tile_shape_in::Cols),
        [TileSize] "i"(
            tile_type_traits<typename tile_shape_out::TileDType>::TilesizeCode));
}

#define SUPERNPUBENCH_V057_TCMP_CASE(MODE, ASM_MODE, LEFT, RIGHT)               \
  case CmpMode::MODE:                                                          \
    asm volatile(                                                               \
        "BSTART.TCMP %c[DataType]\n"                                    \
        "B.DATR Null, " #ASM_MODE "\n"                                        \
        "C.B.DIMI %c[ValidCol], ->LB0\n"                                       \
        "C.B.DIMI %c[ValidRow], ->LB1\n"                                       \
        "C.B.DIMI %c[Cols], ->LB2\n"                                           \
        "B.IOT %[left], %[right], last, ->%q[dst]<%c[TileSize]>\n"            \
        : [dst] "=r"(dst.data())                                                \
        : [left] "r"(LEFT.data()), [right] "r"(RIGHT.data()),                  \
          [DataType] "i"(type_traits<typename tile_shape_in::DType>::TypeCode),\
          [ValidCol] "i"(tile_shape_in::ValidCol),                             \
          [ValidRow] "i"(tile_shape_in::ValidRow),                             \
          [Cols] "i"(tile_shape_in::Cols),                                     \
          [TileSize] "i"(                                                      \
              tile_type_traits<typename tile_shape_out::TileDType>::TilesizeCode));\
    return

template <is_tile_data_v tile_shape_out, is_tile_data_v tile_shape_in>
inline void TCMP_V057(tile_shape_out &dst, tile_shape_in &src0,
                      tile_shape_in &src1, CmpMode mode) {
  switch (mode) {
    SUPERNPUBENCH_V057_TCMP_CASE(EQ, EQ, src0, src1);
    SUPERNPUBENCH_V057_TCMP_CASE(NE, NE, src0, src1);
    SUPERNPUBENCH_V057_TCMP_CASE(LT, LT, src0, src1);
    SUPERNPUBENCH_V057_TCMP_CASE(GE, GE, src0, src1);
    SUPERNPUBENCH_V057_TCMP_CASE(GT, LT, src1, src0);
    SUPERNPUBENCH_V057_TCMP_CASE(LE, GE, src1, src0);
  }
}
#undef SUPERNPUBENCH_V057_TCMP_CASE

template <is_tile_data_v tile_shape, is_global_data_v gm_shape>
inline void TLOAD_V057(tile_shape &dst, gm_shape &src) {
  static_assert(tile_shape::isRowMajor && gm_shape::isRowMajor,
                "v0.57 direct TLOAD currently requires row-major tensors");
  asm volatile(
      "BSTART.TLOAD %c[SrcType]\n"
      "B.DATR NORM.normal, %c[DstType], Null\n"
      "C.B.DIMI %c[ValidCol], ->LB0\n"
      "C.B.DIMI %c[ValidRow], ->LB1\n"
      "C.B.DIMI %c[Cols], ->LB2\n"
      "B.IOT last, ->%q[dst]<%c[TileSize]>\n"
      "B.IOR [%[base], %[stride]], []\n"
      : [dst] "=r"(dst.data())
      : [base] "r"(src.data()),
        [stride] "r"(gm_shape::RowStride * sizeof(typename gm_shape::DType)),
        [SrcType] "i"(type_traits<typename gm_shape::DType>::TypeCode),
        [DstType] "i"(type_traits<typename tile_shape::DType>::TypeCode),
        [ValidCol] "i"(tile_shape::ValidCol),
        [ValidRow] "i"(tile_shape::ValidRow), [Cols] "i"(tile_shape::Cols),
        [TileSize] "i"(
            tile_type_traits<typename tile_shape::TileDType>::TilesizeCode)
      : "memory");
}

template <is_global_data_v gm_shape, is_tile_data_v tile_shape>
inline void TSTORE_V057(gm_shape &dst, tile_shape &src) {
  static_assert(tile_shape::isRowMajor && gm_shape::isRowMajor,
                "v0.57 direct TSTORE currently requires row-major tensors");
  asm volatile(
      "BSTART.TSTORE %c[SrcType]\n"
      "B.DATR NORM.normal, %c[DstType], Null\n"
      "C.B.DIMI %c[ValidCol], ->LB0\n"
      "C.B.DIMI %c[ValidRow], ->LB1\n"
      "C.B.DIMI %c[Cols], ->LB2\n"
      "B.IOT %[src], last, ->%q[src]<%c[TileSize]>\n"
      "B.IOR [%[base], %[stride]], []\n"
      :
      : [base] "r"(dst.data()), [src] "r"(src.data()),
        [stride] "r"(gm_shape::RowStride * sizeof(typename gm_shape::DType)),
        [SrcType] "i"(type_traits<typename tile_shape::DType>::TypeCode),
        [DstType] "i"(type_traits<typename gm_shape::DType>::TypeCode),
        [ValidCol] "i"(tile_shape::ValidCol),
        [ValidRow] "i"(tile_shape::ValidRow), [Cols] "i"(tile_shape::Cols),
        [TileSize] "i"(
            tile_type_traits<typename tile_shape::TileDType>::TilesizeCode)
      : "memory");
}

#endif

#if !defined(__linx)

#ifndef LINX_CVT_INLINE
#define LINX_CVT_INLINE __attribute__((always_inline)) inline
#endif

#ifndef SIMPLE_STORAGE
#define SIMPLE_STORAGE(d) (d)
#endif

// v0.57 exposes only architectural GPR inline-assembly operands.  The prior
// toolchain supplied private *_STORAGE accessors alongside its vector-register
// constraints; keep the public value types usable by mapping those accessors to
// their scalar or packed value representation.
#define __tf32_STORAGE(d) (d)
#define __hf32_STORAGE(d) (d)
#define __blkc_bf16_STORAGE(d) (d)
#define __hif8_STORAGE(d) (d)
#define __fp8_e4m3_STORAGE(d) (d)
#define __fp8_e5m2_STORAGE(d) (d)
#define __fp6_e3m2_STORAGE(d) (d)
#define __fp6_e2m3_STORAGE(d) (d)
#define __fp8_e8m0_STORAGE(d) (d)
#define __fp8_e6m2_STORAGE(d) (d)
#define __fp4_e2m1x2_STORAGE(d) (d)
#define __fp4_e1m2x2_STORAGE(d) (d)
#define __fp4_hif4x2_STORAGE(d) (d)
#define __fp8_e6m2x2_STORAGE(d) (d)
#define __fp8_e4m3x2_STORAGE(d) (d)
#define __fp8_e5m2x2_STORAGE(d) (d)
#define __fp16x2_STORAGE(d) (d)
#define __bf16x2_STORAGE(d) (d)
#define __blkc_bf16x2_STORAGE(d) (d)
#define __uint4x2_STORAGE(d) (d)
#define __uint16x2_STORAGE(d) (d)
#define __int4x2_STORAGE(d) (d)
#define __int16x2_STORAGE(d) (d)
#define __fp8_e4m3x4_STORAGE(d) (d)
#define __fp8_e5m2x4_STORAGE(d) (d)
#define __uint8x4_STORAGE(d) (d)
#define __int8x4_STORAGE(d) (d)

// 默认生成：
//   v.cvt.xxx2yyy %1.src, ->%0.dst, RMode, sat
//
// 如果你的汇编器支持 "-> %0."，可以在 include 前改成：
//   #define LINX_CVT_DST_PREFIX ", -> %0."
//
// 如果你的汇编器完全不需要 "->"，可以在 include 前改成：
//   #define LINX_CVT_DST_PREFIX ", %0."
#ifndef LINX_CVT_DST_PREFIX
#define LINX_CVT_DST_PREFIX ", ->%0."
#endif

enum LinxRMode {
  LINX_RNONE = 0,
  LINX_RNE   = 1,
  LINX_RTZ   = 2,
  LINX_RDN   = 3,
  LINX_RUP   = 4,
  LINX_RNA   = 5,
  LINX_RTO   = 6,
  LINX_RHB   = 7,
};

enum LinxSat {
  LINX_NOSAT = 0,
  LINX_SAT   = 1,
};

template <class...>
struct linx_cvt_false {
  enum { value = 0 };
};

template <int RMode>
struct linx_valid_rmode {
  enum { value = RMode >= LINX_RNONE && RMode <= LINX_RHB };
};

template <int Sat>
struct linx_valid_sat {
  enum { value = Sat == LINX_NOSAT || Sat == LINX_SAT };
};

#define LINX_CVT_ASM(SRC_TYPE, DST_TYPE, SRC_REG, DST_REG, RMODE, SAT)        \
  "v.cvt." SRC_TYPE "2" DST_TYPE " %1." SRC_REG LINX_CVT_DST_PREFIX DST_REG   \
  ", " RMODE ", " SAT "\n"

#define LINX_CVT_PACKED_ASM(SRC_TYPE, DST_TYPE, SRC_REG, DST_REG, RMODE, SAT) \
  "v.cvt." SRC_TYPE "2" DST_TYPE " %1." SRC_REG ", %2." SRC_REG               \
  LINX_CVT_DST_PREFIX DST_REG ", " RMODE ", " SAT "\n"

#define LINX_CVT_EMIT_NORMAL(SRC_TYPE, DST_TYPE, SRC_REG, DST_REG,            \
                             DST_STORAGE, SRC_STORAGE, DST, SRC,              \
                             RMODE_STR, SAT_STR)                              \
  asm volatile(LINX_CVT_ASM(SRC_TYPE, DST_TYPE, SRC_REG, DST_REG,             \
                            RMODE_STR, SAT_STR)                               \
               : "=r"(DST_STORAGE(DST))                                       \
               : "r"(SRC_STORAGE(SRC)))

#define LINX_CVT_EMIT_PACKED(SRC_TYPE, DST_TYPE, SRC_REG, DST_REG,            \
                             DST_STORAGE, SRC_STORAGE, DST, SRC0, SRC1,        \
                             RMODE_STR, SAT_STR)                              \
  asm volatile(LINX_CVT_PACKED_ASM(SRC_TYPE, DST_TYPE, SRC_REG, DST_REG,      \
                                   RMODE_STR, SAT_STR)                        \
               : "=r"(DST_STORAGE(DST))                                       \
               : "r"(SRC_STORAGE(SRC0)),                                      \
                 "r"(SRC_STORAGE(SRC1)))

#define LINX_CVT_DISPATCH_NORMAL(RMODE, SAT, SRC_TYPE, DST_TYPE, SRC_REG,     \
                                 DST_REG, DST_STORAGE, SRC_STORAGE, DST, SRC)  \
  do {                                                                        \
    static_assert(linx_valid_rmode<RMODE>::value, "invalid cvt RMode");       \
    static_assert(linx_valid_sat<SAT>::value, "invalid cvt Sat");             \
    if constexpr ((RMODE) == LINX_RNONE && (SAT) == LINX_NOSAT) {             \
      LINX_CVT_EMIT_NORMAL(SRC_TYPE, DST_TYPE, SRC_REG, DST_REG,              \
                           DST_STORAGE, SRC_STORAGE, DST, SRC,                \
                           "RNONE", "nosat");                                 \
    } else if constexpr ((RMODE) == LINX_RNONE && (SAT) == LINX_SAT) {        \
      LINX_CVT_EMIT_NORMAL(SRC_TYPE, DST_TYPE, SRC_REG, DST_REG,              \
                           DST_STORAGE, SRC_STORAGE, DST, SRC,                \
                           "RNONE", "sat");                                   \
    } else if constexpr ((RMODE) == LINX_RNE && (SAT) == LINX_NOSAT) {        \
      LINX_CVT_EMIT_NORMAL(SRC_TYPE, DST_TYPE, SRC_REG, DST_REG,              \
                           DST_STORAGE, SRC_STORAGE, DST, SRC,                \
                           "RNE", "nosat");                                   \
    } else if constexpr ((RMODE) == LINX_RNE && (SAT) == LINX_SAT) {          \
      LINX_CVT_EMIT_NORMAL(SRC_TYPE, DST_TYPE, SRC_REG, DST_REG,              \
                           DST_STORAGE, SRC_STORAGE, DST, SRC,                \
                           "RNE", "sat");                                     \
    } else if constexpr ((RMODE) == LINX_RTZ && (SAT) == LINX_NOSAT) {        \
      LINX_CVT_EMIT_NORMAL(SRC_TYPE, DST_TYPE, SRC_REG, DST_REG,              \
                           DST_STORAGE, SRC_STORAGE, DST, SRC,                \
                           "RTZ", "nosat");                                   \
    } else if constexpr ((RMODE) == LINX_RTZ && (SAT) == LINX_SAT) {          \
      LINX_CVT_EMIT_NORMAL(SRC_TYPE, DST_TYPE, SRC_REG, DST_REG,              \
                           DST_STORAGE, SRC_STORAGE, DST, SRC,                \
                           "RTZ", "sat");                                     \
    } else if constexpr ((RMODE) == LINX_RDN && (SAT) == LINX_NOSAT) {        \
      LINX_CVT_EMIT_NORMAL(SRC_TYPE, DST_TYPE, SRC_REG, DST_REG,              \
                           DST_STORAGE, SRC_STORAGE, DST, SRC,                \
                           "RDN", "nosat");                                   \
    } else if constexpr ((RMODE) == LINX_RDN && (SAT) == LINX_SAT) {          \
      LINX_CVT_EMIT_NORMAL(SRC_TYPE, DST_TYPE, SRC_REG, DST_REG,              \
                           DST_STORAGE, SRC_STORAGE, DST, SRC,                \
                           "RDN", "sat");                                     \
    } else if constexpr ((RMODE) == LINX_RUP && (SAT) == LINX_NOSAT) {        \
      LINX_CVT_EMIT_NORMAL(SRC_TYPE, DST_TYPE, SRC_REG, DST_REG,              \
                           DST_STORAGE, SRC_STORAGE, DST, SRC,                \
                           "RUP", "nosat");                                   \
    } else if constexpr ((RMODE) == LINX_RUP && (SAT) == LINX_SAT) {          \
      LINX_CVT_EMIT_NORMAL(SRC_TYPE, DST_TYPE, SRC_REG, DST_REG,              \
                           DST_STORAGE, SRC_STORAGE, DST, SRC,                \
                           "RUP", "sat");                                     \
    } else if constexpr ((RMODE) == LINX_RNA && (SAT) == LINX_NOSAT) {        \
      LINX_CVT_EMIT_NORMAL(SRC_TYPE, DST_TYPE, SRC_REG, DST_REG,              \
                           DST_STORAGE, SRC_STORAGE, DST, SRC,                \
                           "RNA", "nosat");                                   \
    } else if constexpr ((RMODE) == LINX_RNA && (SAT) == LINX_SAT) {          \
      LINX_CVT_EMIT_NORMAL(SRC_TYPE, DST_TYPE, SRC_REG, DST_REG,              \
                           DST_STORAGE, SRC_STORAGE, DST, SRC,                \
                           "RNA", "sat");                                     \
    } else if constexpr ((RMODE) == LINX_RTO && (SAT) == LINX_NOSAT) {        \
      LINX_CVT_EMIT_NORMAL(SRC_TYPE, DST_TYPE, SRC_REG, DST_REG,              \
                           DST_STORAGE, SRC_STORAGE, DST, SRC,                \
                           "RTO", "nosat");                                   \
    } else if constexpr ((RMODE) == LINX_RTO && (SAT) == LINX_SAT) {          \
      LINX_CVT_EMIT_NORMAL(SRC_TYPE, DST_TYPE, SRC_REG, DST_REG,              \
                           DST_STORAGE, SRC_STORAGE, DST, SRC,                \
                           "RTO", "sat");                                     \
    } else if constexpr ((RMODE) == LINX_RHB && (SAT) == LINX_NOSAT) {        \
      LINX_CVT_EMIT_NORMAL(SRC_TYPE, DST_TYPE, SRC_REG, DST_REG,              \
                           DST_STORAGE, SRC_STORAGE, DST, SRC,                \
                           "RHB", "nosat");                                   \
    } else {                                                                  \
      LINX_CVT_EMIT_NORMAL(SRC_TYPE, DST_TYPE, SRC_REG, DST_REG,              \
                           DST_STORAGE, SRC_STORAGE, DST, SRC,                \
                           "RHB", "sat");                                     \
    }                                                                         \
  } while (0)

#define LINX_CVT_DISPATCH_PACKED(RMODE, SAT, SRC_TYPE, DST_TYPE, SRC_REG,     \
                                 DST_REG, DST_STORAGE, SRC_STORAGE, DST,       \
                                 SRC0, SRC1)                                  \
  do {                                                                        \
    static_assert(linx_valid_rmode<RMODE>::value, "invalid cvt RMode");       \
    static_assert(linx_valid_sat<SAT>::value, "invalid cvt Sat");             \
    if constexpr ((RMODE) == LINX_RNONE && (SAT) == LINX_NOSAT) {             \
      LINX_CVT_EMIT_PACKED(SRC_TYPE, DST_TYPE, SRC_REG, DST_REG,              \
                           DST_STORAGE, SRC_STORAGE, DST, SRC0, SRC1,         \
                           "RNONE", "nosat");                                 \
    } else if constexpr ((RMODE) == LINX_RNONE && (SAT) == LINX_SAT) {        \
      LINX_CVT_EMIT_PACKED(SRC_TYPE, DST_TYPE, SRC_REG, DST_REG,              \
                           DST_STORAGE, SRC_STORAGE, DST, SRC0, SRC1,         \
                           "RNONE", "sat");                                   \
    } else if constexpr ((RMODE) == LINX_RNE && (SAT) == LINX_NOSAT) {        \
      LINX_CVT_EMIT_PACKED(SRC_TYPE, DST_TYPE, SRC_REG, DST_REG,              \
                           DST_STORAGE, SRC_STORAGE, DST, SRC0, SRC1,         \
                           "RNE", "nosat");                                   \
    } else if constexpr ((RMODE) == LINX_RNE && (SAT) == LINX_SAT) {          \
      LINX_CVT_EMIT_PACKED(SRC_TYPE, DST_TYPE, SRC_REG, DST_REG,              \
                           DST_STORAGE, SRC_STORAGE, DST, SRC0, SRC1,         \
                           "RNE", "sat");                                     \
    } else if constexpr ((RMODE) == LINX_RTZ && (SAT) == LINX_NOSAT) {        \
      LINX_CVT_EMIT_PACKED(SRC_TYPE, DST_TYPE, SRC_REG, DST_REG,              \
                           DST_STORAGE, SRC_STORAGE, DST, SRC0, SRC1,         \
                           "RTZ", "nosat");                                   \
    } else if constexpr ((RMODE) == LINX_RTZ && (SAT) == LINX_SAT) {          \
      LINX_CVT_EMIT_PACKED(SRC_TYPE, DST_TYPE, SRC_REG, DST_REG,              \
                           DST_STORAGE, SRC_STORAGE, DST, SRC0, SRC1,         \
                           "RTZ", "sat");                                     \
    } else if constexpr ((RMODE) == LINX_RDN && (SAT) == LINX_NOSAT) {        \
      LINX_CVT_EMIT_PACKED(SRC_TYPE, DST_TYPE, SRC_REG, DST_REG,              \
                           DST_STORAGE, SRC_STORAGE, DST, SRC0, SRC1,         \
                           "RDN", "nosat");                                   \
    } else if constexpr ((RMODE) == LINX_RDN && (SAT) == LINX_SAT) {          \
      LINX_CVT_EMIT_PACKED(SRC_TYPE, DST_TYPE, SRC_REG, DST_REG,              \
                           DST_STORAGE, SRC_STORAGE, DST, SRC0, SRC1,         \
                           "RDN", "sat");                                     \
    } else if constexpr ((RMODE) == LINX_RUP && (SAT) == LINX_NOSAT) {        \
      LINX_CVT_EMIT_PACKED(SRC_TYPE, DST_TYPE, SRC_REG, DST_REG,              \
                           DST_STORAGE, SRC_STORAGE, DST, SRC0, SRC1,         \
                           "RUP", "nosat");                                   \
    } else if constexpr ((RMODE) == LINX_RUP && (SAT) == LINX_SAT) {          \
      LINX_CVT_EMIT_PACKED(SRC_TYPE, DST_TYPE, SRC_REG, DST_REG,              \
                           DST_STORAGE, SRC_STORAGE, DST, SRC0, SRC1,         \
                           "RUP", "sat");                                     \
    } else if constexpr ((RMODE) == LINX_RNA && (SAT) == LINX_NOSAT) {        \
      LINX_CVT_EMIT_PACKED(SRC_TYPE, DST_TYPE, SRC_REG, DST_REG,              \
                           DST_STORAGE, SRC_STORAGE, DST, SRC0, SRC1,         \
                           "RNA", "nosat");                                   \
    } else if constexpr ((RMODE) == LINX_RNA && (SAT) == LINX_SAT) {          \
      LINX_CVT_EMIT_PACKED(SRC_TYPE, DST_TYPE, SRC_REG, DST_REG,              \
                           DST_STORAGE, SRC_STORAGE, DST, SRC0, SRC1,         \
                           "RNA", "sat");                                     \
    } else if constexpr ((RMODE) == LINX_RTO && (SAT) == LINX_NOSAT) {        \
      LINX_CVT_EMIT_PACKED(SRC_TYPE, DST_TYPE, SRC_REG, DST_REG,              \
                           DST_STORAGE, SRC_STORAGE, DST, SRC0, SRC1,         \
                           "RTO", "nosat");                                   \
    } else if constexpr ((RMODE) == LINX_RTO && (SAT) == LINX_SAT) {          \
      LINX_CVT_EMIT_PACKED(SRC_TYPE, DST_TYPE, SRC_REG, DST_REG,              \
                           DST_STORAGE, SRC_STORAGE, DST, SRC0, SRC1,         \
                           "RTO", "sat");                                     \
    } else if constexpr ((RMODE) == LINX_RHB && (SAT) == LINX_NOSAT) {        \
      LINX_CVT_EMIT_PACKED(SRC_TYPE, DST_TYPE, SRC_REG, DST_REG,              \
                           DST_STORAGE, SRC_STORAGE, DST, SRC0, SRC1,         \
                           "RHB", "nosat");                                   \
    } else {                                                                  \
      LINX_CVT_EMIT_PACKED(SRC_TYPE, DST_TYPE, SRC_REG, DST_REG,              \
                           DST_STORAGE, SRC_STORAGE, DST, SRC0, SRC1,         \
                           "RHB", "sat");                                     \
    }                                                                         \
  } while (0)

template <int RMode = LINX_RNONE, int Sat = LINX_NOSAT, class Dst, class Src>
LINX_CVT_INLINE void linx_cvt(Dst &, const Src &) {
  static_assert(linx_cvt_false<Dst, Src>::value,
                "unsupported linx_cvt(dst, src) type pair");
}

template <int RMode = LINX_RNONE, int Sat = LINX_NOSAT,
          class Dst, class Src0, class Src1>
LINX_CVT_INLINE void linx_cvt_packed(Dst &, const Src0 &, const Src1 &) {
  static_assert(linx_cvt_false<Dst, Src0, Src1>::value,
                "unsupported linx_cvt_packed(dst, src0, src1) type pair");
}

#define LINX_DEFINE_CVT(DST_CPP, DST_TYPE, DST_REG, DST_STORAGE,              \
                        SRC_CPP, SRC_TYPE, SRC_REG, SRC_STORAGE)              \
  template <int RMode = LINX_RNONE, int Sat = LINX_NOSAT>                     \
  LINX_CVT_INLINE void linx_cvt(DST_CPP &dst, const SRC_CPP &src) {           \
    LINX_CVT_DISPATCH_NORMAL(RMode, Sat, SRC_TYPE, DST_TYPE, SRC_REG,         \
                             DST_REG, DST_STORAGE, SRC_STORAGE, dst, src);    \
  }

#define LINX_DEFINE_CVT_PACKED(DST_CPP, DST_TYPE, DST_REG, DST_STORAGE,       \
                               SRC_CPP, SRC_TYPE, SRC_REG, SRC_STORAGE)       \
  template <int RMode = LINX_RNONE, int Sat = LINX_NOSAT>                     \
  LINX_CVT_INLINE void linx_cvt_packed(DST_CPP &dst,                          \
                                       const SRC_CPP &src0,                   \
                                       const SRC_CPP &src1) {                 \
    LINX_CVT_DISPATCH_PACKED(RMode, Sat, SRC_TYPE, DST_TYPE, SRC_REG,         \
                             DST_REG, DST_STORAGE, SRC_STORAGE, dst,          \
                             src0, src1);                                     \
  }

#define LINX_FOR_EACH_NORMAL_SRC(M, DST_CPP, DST_TYPE, DST_REG, DST_STORAGE)  \
  M(DST_CPP, DST_TYPE, DST_REG, DST_STORAGE, double, "fp64", "fd",            \
    SIMPLE_STORAGE)                                                           \
  M(DST_CPP, DST_TYPE, DST_REG, DST_STORAGE, float, "fp32", "fs",             \
    SIMPLE_STORAGE)                                                           \
  M(DST_CPP, DST_TYPE, DST_REG, DST_STORAGE, __tf32, "tf32", "fs",            \
    __tf32_STORAGE)                                                           \
  M(DST_CPP, DST_TYPE, DST_REG, DST_STORAGE, __hf32, "hf32", "fs",            \
    __hf32_STORAGE)                                                           \
  M(DST_CPP, DST_TYPE, DST_REG, DST_STORAGE, __half, "fp16", "fh",            \
    SIMPLE_STORAGE)                                                           \
  M(DST_CPP, DST_TYPE, DST_REG, DST_STORAGE, __bf16, "bf16", "fh",            \
    __blkc_bf16_STORAGE)                                                      \
  M(DST_CPP, DST_TYPE, DST_REG, DST_STORAGE, __hif8, "hif8", "fb",            \
    __hif8_STORAGE)                                                           \
  M(DST_CPP, DST_TYPE, DST_REG, DST_STORAGE, __fp8_e4m3, "e4m3", "fb",        \
    __fp8_e4m3_STORAGE)                                                       \
  M(DST_CPP, DST_TYPE, DST_REG, DST_STORAGE, __fp8_e5m2, "e5m2", "fb",        \
    __fp8_e5m2_STORAGE)                                                       \
  M(DST_CPP, DST_TYPE, DST_REG, DST_STORAGE, __fp6_e3m2, "e3m2", "fb",        \
    __fp6_e3m2_STORAGE)                                                       \
  M(DST_CPP, DST_TYPE, DST_REG, DST_STORAGE, __fp6_e2m3, "e2m3", "fb",        \
    __fp6_e2m3_STORAGE)                                                       \
  M(DST_CPP, DST_TYPE, DST_REG, DST_STORAGE, __fp8_e8m0, "e8m0", "fb",        \
    __fp8_e8m0_STORAGE)                                                       \
  M(DST_CPP, DST_TYPE, DST_REG, DST_STORAGE, __fp8_e6m2, "e6m2", "fb",        \
    __fp8_e6m2_STORAGE)                                                       \
  M(DST_CPP, DST_TYPE, DST_REG, DST_STORAGE, long, "s64", "sd",               \
    SIMPLE_STORAGE)                                                           \
  M(DST_CPP, DST_TYPE, DST_REG, DST_STORAGE, int, "s32", "sw",                \
    SIMPLE_STORAGE)                                                           \
  M(DST_CPP, DST_TYPE, DST_REG, DST_STORAGE, short, "s16", "sh",              \
    SIMPLE_STORAGE)                                                           \
  M(DST_CPP, DST_TYPE, DST_REG, DST_STORAGE, char, "s8", "sb",                \
    SIMPLE_STORAGE)                                                           \
  M(DST_CPP, DST_TYPE, DST_REG, DST_STORAGE, unsigned long, "u64", "ud",      \
    SIMPLE_STORAGE)                                                           \
  M(DST_CPP, DST_TYPE, DST_REG, DST_STORAGE, unsigned int, "u32", "uw",       \
    SIMPLE_STORAGE)                                                           \
  M(DST_CPP, DST_TYPE, DST_REG, DST_STORAGE, unsigned short, "u16", "uh",     \
    SIMPLE_STORAGE)                                                           \
  M(DST_CPP, DST_TYPE, DST_REG, DST_STORAGE, unsigned char, "u8", "ub",       \
    SIMPLE_STORAGE)

#define LINX_FOR_EACH_NORMAL_DST(M)                                           \
  M(double, "fp64", "d", SIMPLE_STORAGE)                                      \
  M(float, "fp32", "w", SIMPLE_STORAGE)                                       \
  M(__tf32, "tf32", "w", __tf32_STORAGE)                                      \
  M(__hf32, "hf32", "w", __hf32_STORAGE)                                      \
  M(__half, "fp16", "h", SIMPLE_STORAGE)                                      \
  M(__bf16, "bf16", "h", __blkc_bf16_STORAGE)                                \
  M(__hif8, "hif8", "b", __hif8_STORAGE)                                      \
  M(__fp8_e4m3, "e4m3", "b", __fp8_e4m3_STORAGE)                             \
  M(__fp8_e5m2, "e5m2", "b", __fp8_e5m2_STORAGE)                             \
  M(__fp6_e3m2, "e3m2", "b", __fp6_e3m2_STORAGE)                             \
  M(__fp6_e2m3, "e2m3", "b", __fp6_e2m3_STORAGE)                             \
  M(__fp8_e8m0, "e8m0", "b", __fp8_e8m0_STORAGE)                             \
  M(__fp8_e6m2, "e6m2", "b", __fp8_e6m2_STORAGE)                             \
  M(long, "s64", "d", SIMPLE_STORAGE)                                        \
  M(int, "s32", "w", SIMPLE_STORAGE)                                         \
  M(short, "s16", "h", SIMPLE_STORAGE)                                       \
  M(char, "s8", "b", SIMPLE_STORAGE)                                         \
  M(unsigned long, "u64", "d", SIMPLE_STORAGE)                               \
  M(unsigned int, "u32", "w", SIMPLE_STORAGE)                                \
  M(unsigned short, "u16", "h", SIMPLE_STORAGE)                              \
  M(unsigned char, "u8", "b", SIMPLE_STORAGE)

#define LINX_FOR_EACH_PACKED_ONLY_SRC(M, DST_CPP, DST_TYPE, DST_REG,          \
                                      DST_STORAGE)                            \
  M(DST_CPP, DST_TYPE, DST_REG, DST_STORAGE, __fp4_e2m1x2, "e2m1x2", "fb",    \
    __fp4_e2m1x2_STORAGE)                                                     \
  M(DST_CPP, DST_TYPE, DST_REG, DST_STORAGE, __fp4_e1m2x2, "e1m2x2", "fb",    \
    __fp4_e1m2x2_STORAGE)                                                     \
  M(DST_CPP, DST_TYPE, DST_REG, DST_STORAGE, __fp4_hif4x2, "hif4x2", "fb",    \
    __fp4_hif4x2_STORAGE)                                                     \
  M(DST_CPP, DST_TYPE, DST_REG, DST_STORAGE, __fp8_e6m2x2, "e6m2x2", "fh",    \
    __fp8_e6m2x2_STORAGE)                                                     \
  M(DST_CPP, DST_TYPE, DST_REG, DST_STORAGE, __fp8_e4m3x2, "e4m3x2", "fh",    \
    __fp8_e4m3x2_STORAGE)                                                     \
  M(DST_CPP, DST_TYPE, DST_REG, DST_STORAGE, __fp8_e5m2x2, "e5m2x2", "fh",    \
    __fp8_e5m2x2_STORAGE)                                                     \
  M(DST_CPP, DST_TYPE, DST_REG, DST_STORAGE, __fp16x2, "fp16x2", "fh",        \
    __fp16x2_STORAGE)                                                         \
  M(DST_CPP, DST_TYPE, DST_REG, DST_STORAGE, __uint4x2, "u4x2", "ub",         \
    __uint4x2_STORAGE)                                                        \
  M(DST_CPP, DST_TYPE, DST_REG, DST_STORAGE, __int4x2, "s4x2", "sb",          \
    __int4x2_STORAGE)


#define LINX_DEFINE_CVT_TO(DST_CPP, DST_TYPE, DST_REG, DST_STORAGE)           \
  LINX_FOR_EACH_NORMAL_SRC(LINX_DEFINE_CVT, DST_CPP, DST_TYPE, DST_REG,       \
                           DST_STORAGE)

LINX_FOR_EACH_NORMAL_DST(LINX_DEFINE_CVT_TO)

// 明确禁止：packed/x2 source -> normal scalar/vector destination。
// 例如禁止生成：
//   v.cvt.e1m2x22s8 ...
//   v.cvt.fp16x22fp32 ...
//   v.cvt.u4x22s8 ...
#define LINX_DELETE_CVT(DST_CPP, DST_TYPE, DST_REG, DST_STORAGE,              \
                        SRC_CPP, SRC_TYPE, SRC_REG, SRC_STORAGE)              \
  template <int RMode = LINX_RNONE, int Sat = LINX_NOSAT>                     \
  void linx_cvt(DST_CPP &, const SRC_CPP &) = delete;

#define LINX_DELETE_PACKED_SRC_TO_NORMAL_DST(DST_CPP, DST_TYPE, DST_REG,      \
                                             DST_STORAGE)                     \
  LINX_FOR_EACH_PACKED_ONLY_SRC(LINX_DELETE_CVT, DST_CPP, DST_TYPE, DST_REG,  \
                                DST_STORAGE)

LINX_FOR_EACH_NORMAL_DST(LINX_DELETE_PACKED_SRC_TO_NORMAL_DST)

//===----------------------------------------------------------------------===//
// Packed-to-Packed CVT
// one packed src -> one packed dst
//
// Supports:
//   x2 -> x2:
//     float x2: e2m1x2, e1m2x2, hif4x2, fp16x2, bf16x2
//     int   x2: u4x2, u16x2, s4x2, s16x2
//
//   x4 -> x4:
//     float x4: e4m3x4, e5m2x4
//     int   x4: u8x4, s8x4
//===----------------------------------------------------------------------===//

#define LINX_FOR_EACH_P2P_X2_SRC(M, DST_CPP, DST_TYPE, DST_REG, DST_STORAGE)  \
  M(DST_CPP, DST_TYPE, DST_REG, DST_STORAGE, __fp4_e2m1x2, "e2m1x2", "fb",    \
    __fp4_e2m1x2_STORAGE)                                                     \
  M(DST_CPP, DST_TYPE, DST_REG, DST_STORAGE, __fp4_e1m2x2, "e1m2x2", "fb",    \
    __fp4_e1m2x2_STORAGE)                                                     \
  M(DST_CPP, DST_TYPE, DST_REG, DST_STORAGE, __fp4_hif4x2, "hif4x2", "fb",    \
    __fp4_hif4x2_STORAGE)                                                     \
  M(DST_CPP, DST_TYPE, DST_REG, DST_STORAGE, __fp8_e6m2x2, "e6m2x2", "fh",    \
    __fp8_e6m2x2_STORAGE)                                                     \
  M(DST_CPP, DST_TYPE, DST_REG, DST_STORAGE, __fp8_e4m3x2, "e4m3x2", "fh",    \
    __fp8_e4m3x2_STORAGE)                                                     \
  M(DST_CPP, DST_TYPE, DST_REG, DST_STORAGE, __fp8_e5m2x2, "e5m2x2", "fh",    \
    __fp8_e5m2x2_STORAGE)                                                     \
  M(DST_CPP, DST_TYPE, DST_REG, DST_STORAGE, __fp16x2, "fp16x2", "fh",        \
    __fp16x2_STORAGE)                                                         \
  M(DST_CPP, DST_TYPE, DST_REG, DST_STORAGE, __bf16x2, "bf16x2", "fh",        \
    __bf16x2_STORAGE)                                                         \
  M(DST_CPP, DST_TYPE, DST_REG, DST_STORAGE, __uint4x2, "u4x2", "ub",         \
    __uint4x2_STORAGE)                                                        \
  M(DST_CPP, DST_TYPE, DST_REG, DST_STORAGE, __uint16x2, "u16x2", "uh",       \
    __uint16x2_STORAGE)                                                       \
  M(DST_CPP, DST_TYPE, DST_REG, DST_STORAGE, __int4x2, "s4x2", "sb",          \
    __int4x2_STORAGE)                                                         \
  M(DST_CPP, DST_TYPE, DST_REG, DST_STORAGE, __int16x2, "s16x2", "sh",        \
    __int16x2_STORAGE)


#define LINX_FOR_EACH_P2P_X2_DST(M)                                           \
  M(__fp4_e2m1x2, "e2m1x2", "b", __fp4_e2m1x2_STORAGE)                       \
  M(__fp4_e1m2x2, "e1m2x2", "b", __fp4_e1m2x2_STORAGE)                       \
  M(__fp4_hif4x2, "hif4x2", "b", __fp4_hif4x2_STORAGE)                       \
  M(__fp8_e6m2x2, "e6m2x2", "h", __fp8_e6m2x2_STORAGE)                       \
  M(__fp8_e4m3x2, "e4m3x2", "h", __fp8_e4m3x2_STORAGE)                       \
  M(__fp8_e5m2x2, "e5m2x2", "h", __fp8_e5m2x2_STORAGE)                       \
  M(__fp16x2, "fp16x2", "w", __fp16x2_STORAGE)                               \
  M(__bf16x2, "bf16x2", "w", __bf16x2_STORAGE)                               \
  M(__uint4x2, "u4x2", "b", __uint4x2_STORAGE)                               \
  M(__uint16x2, "u16x2", "w", __uint16x2_STORAGE)                            \
  M(__int4x2, "s4x2", "b", __int4x2_STORAGE)                                 \
  M(__int16x2, "s16x2", "w", __int16x2_STORAGE)


#define LINX_FOR_EACH_P2P_X4_SRC(M, DST_CPP, DST_TYPE, DST_REG, DST_STORAGE)  \
  M(DST_CPP, DST_TYPE, DST_REG, DST_STORAGE, __fp8_e4m3x4, "e4m3x4", "fb",    \
    __fp8_e4m3x4_STORAGE)                                                     \
  M(DST_CPP, DST_TYPE, DST_REG, DST_STORAGE, __fp8_e5m2x4, "e5m2x4", "fb",    \
    __fp8_e5m2x4_STORAGE)                                                     \
  M(DST_CPP, DST_TYPE, DST_REG, DST_STORAGE, __uint8x4, "u8x4", "ub",         \
    __uint8x4_STORAGE)                                                        \
  M(DST_CPP, DST_TYPE, DST_REG, DST_STORAGE, __int8x4, "s8x4", "sb",          \
    __int8x4_STORAGE)

#define LINX_FOR_EACH_P2P_X4_DST(M)                                           \
  M(__fp8_e4m3x4, "e4m3x4", "w", __fp8_e4m3x4_STORAGE)                       \
  M(__fp8_e5m2x4, "e5m2x4", "w", __fp8_e5m2x4_STORAGE)                       \
  M(__uint8x4, "u8x4", "w", __uint8x4_STORAGE)                               \
  M(__int8x4, "s8x4", "w", __int8x4_STORAGE)

#define LINX_DEFINE_P2P_X2_CVT_TO(DST_CPP, DST_TYPE, DST_REG, DST_STORAGE)    \
  LINX_FOR_EACH_P2P_X2_SRC(LINX_DEFINE_CVT, DST_CPP, DST_TYPE, DST_REG,       \
                           DST_STORAGE)

#define LINX_DEFINE_P2P_X4_CVT_TO(DST_CPP, DST_TYPE, DST_REG, DST_STORAGE)    \
  LINX_FOR_EACH_P2P_X4_SRC(LINX_DEFINE_CVT, DST_CPP, DST_TYPE, DST_REG,       \
                           DST_STORAGE)

LINX_FOR_EACH_P2P_X2_DST(LINX_DEFINE_P2P_X2_CVT_TO)
LINX_FOR_EACH_P2P_X4_DST(LINX_DEFINE_P2P_X4_CVT_TO)

// packed destination 只支持：
//   normal src0 + normal src1 -> packed dst
//
// 不支持：
//   packed src -> normal dst
//   packed src0 + packed src1 -> packed dst
#define LINX_FOR_EACH_PACKED_SRC(M, DST_CPP, DST_TYPE, DST_REG, DST_STORAGE)  \
  LINX_FOR_EACH_NORMAL_SRC(M, DST_CPP, DST_TYPE, DST_REG, DST_STORAGE)

#define LINX_FOR_EACH_PACKED_DST(M)                                           \
  M(__fp4_e2m1x2, "e2m1x2", "b", __fp4_e2m1x2_STORAGE)                       \
  M(__fp4_e1m2x2, "e1m2x2", "b", __fp4_e1m2x2_STORAGE)                       \
  M(__fp4_hif4x2, "hif4x2", "b", __fp4_hif4x2_STORAGE)                       \
  M(__fp8_e6m2x2, "e6m2x2", "h", __fp8_e6m2x2_STORAGE)                       \
  M(__fp8_e4m3x2, "e4m3x2", "h", __fp8_e4m3x2_STORAGE)                       \
  M(__fp8_e5m2x2, "e5m2x2", "h", __fp8_e5m2x2_STORAGE)                       \
  M(__fp16x2, "fp16x2", "w", __fp16x2_STORAGE)                               \
  M(__bf16x2, "bf16x2", "w", __bf16x2_STORAGE)                               \
  M(__uint4x2, "u4x2", "b", __uint4x2_STORAGE)                               \
  M(__int4x2, "s4x2", "b", __int4x2_STORAGE)


#define LINX_DEFINE_PACKED_CVT_TO(DST_CPP, DST_TYPE, DST_REG, DST_STORAGE)    \
  LINX_FOR_EACH_PACKED_SRC(LINX_DEFINE_CVT_PACKED, DST_CPP, DST_TYPE,         \
                           DST_REG, DST_STORAGE)

LINX_FOR_EACH_PACKED_DST(LINX_DEFINE_PACKED_CVT_TO)

typedef __attribute__((address_space(6))) __fp4_e1m2x2
    __fp4_e1m2x2_as6;

typedef __attribute__((address_space(6))) __bf16x2
    __bf16x2_as6;

typedef __attribute__((address_space(6))) __fp8_e6m2x2
    __fp8_e6m2x2_as6;

typedef __attribute__((address_space(6))) __fp16x2
    __fp16x2_as6;

// __bf16x2 -> address_space(6) __fp8_e6m2x2
template <int RMode = LINX_RNONE, int Sat = LINX_NOSAT>
LINX_CVT_INLINE void linx_cvt(__fp8_e6m2x2_as6 &dst,
                              const __bf16x2 &src) {
  LINX_CVT_DISPATCH_NORMAL(RMode, Sat,
                           "bf16x2", "e6m2x2",
                           "fh", "h",
                           __fp8_e6m2x2_STORAGE,
                           __bf16x2_STORAGE,
                           dst, src);
}


// address_space(6) __bf16x2 -> __fp8_e6m2x2
template <int RMode = LINX_RNONE, int Sat = LINX_NOSAT>
LINX_CVT_INLINE void linx_cvt(__fp8_e6m2x2 &dst,
                              const __bf16x2_as6 &src) {
  LINX_CVT_DISPATCH_NORMAL(RMode, Sat,
                           "bf16x2", "e6m2x2",
                           "fh", "h",
                           __fp8_e6m2x2_STORAGE,
                           __bf16x2_STORAGE,
                           dst, src);
}


// address_space(6) __bf16x2 -> address_space(6) __fp8_e6m2x2
template <int RMode = LINX_RNONE, int Sat = LINX_NOSAT>
LINX_CVT_INLINE void linx_cvt(__fp8_e6m2x2_as6 &dst,
                              const __bf16x2_as6 &src) {
  LINX_CVT_DISPATCH_NORMAL(RMode, Sat,
                           "bf16x2", "e6m2x2",
                           "fh", "h",
                           __fp8_e6m2x2_STORAGE,
                           __bf16x2_STORAGE,
                           dst, src);
}

// bf16x2 -> address_space(6) __fp4_e1m2x2
template <int RMode = LINX_RNONE, int Sat = LINX_NOSAT>
LINX_CVT_INLINE void linx_cvt(__fp4_e1m2x2_as6 &dst,
                              const __bf16x2 &src) {
  LINX_CVT_DISPATCH_NORMAL(RMode, Sat,
                           "bf16x2", "e1m2x2",
                           "fh", "b",
                           __fp4_e1m2x2_STORAGE,
                           __bf16x2_STORAGE,
                           dst, src);
}


// raw __blkc_bf16 + raw __blkc_bf16 -> address_space(6) __fp4_e1m2x2
template <int RMode = LINX_RNONE, int Sat = LINX_NOSAT>
LINX_CVT_INLINE void linx_cvt_packed(__fp4_e1m2x2_as6 &dst,
                                     const __blkc_bf16 &src0,
                                     const __blkc_bf16 &src1) {
  LINX_CVT_DISPATCH_PACKED(RMode, Sat,
                           "bf16", "e1m2x2",
                           "fh", "b",
                           __fp4_e1m2x2_STORAGE,
                           SIMPLE_STORAGE,
                           dst, src0, src1);
}


// float + float -> address_space(6) __bf16x2
template <int RMode = LINX_RNONE, int Sat = LINX_NOSAT>
LINX_CVT_INLINE void linx_cvt_packed(__bf16x2_as6 &dst,
                                     const float &src0,
                                     const float &src1) {
  LINX_CVT_DISPATCH_PACKED(RMode, Sat,
                           "fp32", "bf16x2",
                           "fs", "w",
                           __bf16x2_STORAGE,
                           SIMPLE_STORAGE,
                           dst, src0, src1);
}

typedef __attribute__((address_space(6))) float float_as6;


// address_space(6) float -> raw __blkc_bf16
template <int RMode = LINX_RNONE, int Sat = LINX_NOSAT>
LINX_CVT_INLINE void linx_cvt(__blkc_bf16 &dst,
                              const float_as6 &src) {
  LINX_CVT_DISPATCH_NORMAL(RMode, Sat,
                           "fp32", "bf16",
                           "fs", "h",
                           SIMPLE_STORAGE,
                           SIMPLE_STORAGE,
                           dst, src);
}

// __fp16x2 -> address_space(6) __fp8_e6m2x2
template <int RMode = LINX_RNONE, int Sat = LINX_NOSAT>
LINX_CVT_INLINE void linx_cvt(__fp8_e6m2x2_as6 &dst,
                              const __fp16x2 &src) {
  LINX_CVT_DISPATCH_NORMAL(RMode, Sat,
                           "fp16x2", "e6m2x2",
                           "fh", "h",
                           __fp8_e6m2x2_STORAGE,
                           __fp16x2_STORAGE,
                           dst, src);
}


// address_space(6) __fp8_e6m2x2 -> __fp16x2
template <int RMode = LINX_RNONE, int Sat = LINX_NOSAT>
LINX_CVT_INLINE void linx_cvt(__fp16x2 &dst,
                              const __fp8_e6m2x2_as6 &src) {
  LINX_CVT_DISPATCH_NORMAL(RMode, Sat,
                           "e6m2x2", "fp16x2",
                           "fh", "w",
                           __fp16x2_STORAGE,
                           __fp8_e6m2x2_STORAGE,
                           dst, src);
}


// address_space(6) __fp16x2 -> __fp8_e6m2x2
template <int RMode = LINX_RNONE, int Sat = LINX_NOSAT>
LINX_CVT_INLINE void linx_cvt(__fp8_e6m2x2 &dst,
                              const __fp16x2_as6 &src) {
  LINX_CVT_DISPATCH_NORMAL(RMode, Sat,
                           "fp16x2", "e6m2x2",
                           "fh", "h",
                           __fp8_e6m2x2_STORAGE,
                           __fp16x2_STORAGE,
                           dst, src);
}


// address_space(6) __fp16x2 -> address_space(6) __fp8_e6m2x2
template <int RMode = LINX_RNONE, int Sat = LINX_NOSAT>
LINX_CVT_INLINE void linx_cvt(__fp8_e6m2x2_as6 &dst,
                              const __fp16x2_as6 &src) {
  LINX_CVT_DISPATCH_NORMAL(RMode, Sat,
                           "fp16x2", "e6m2x2",
                           "fh", "h",
                           __fp8_e6m2x2_STORAGE,
                           __fp16x2_STORAGE,
                           dst, src);
}

typedef __attribute__((address_space(6))) __blkc_bf16
    __blkc_bf16_as6;


// address_space(6) __blkc_bf16 + address_space(6) __blkc_bf16 -> __bf16x2
template <int RMode = LINX_RNONE, int Sat = LINX_NOSAT>
LINX_CVT_INLINE void linx_cvt_packed(__bf16x2 &dst,
                                     const __blkc_bf16_as6 &src0,
                                     const __blkc_bf16_as6 &src1) {
  LINX_CVT_DISPATCH_PACKED(RMode, Sat,
                           "bf16", "bf16x2",
                           "fh", "w",
                           __bf16x2_STORAGE,
                           SIMPLE_STORAGE,
                           dst, src0, src1);
}

template <class Dst, class Src>
LINX_CVT_INLINE Dst linx_cvt_as(const Src &src) {
  Dst dst;
  linx_cvt(dst, src);
  return dst;
}

template <int RMode, int Sat, class Dst, class Src>
LINX_CVT_INLINE Dst linx_cvt_as(const Src &src) {
  Dst dst;
  linx_cvt<RMode, Sat>(dst, src);
  return dst;
}

template <class Dst, class Src>
LINX_CVT_INLINE void linx_cvt_package(Dst &dst,
                                      const Src &src0,
                                      const Src &src1) {
  linx_cvt_packed(dst, src0, src1);
}

template <int RMode, int Sat, class Dst, class Src>
LINX_CVT_INLINE void linx_cvt_package(Dst &dst,
                                      const Src &src0,
                                      const Src &src1) {
  linx_cvt_packed<RMode, Sat>(dst, src0, src1);
}

template <class Dst, class Src>
LINX_CVT_INLINE Dst linx_cvt_package_as(const Src &src0, const Src &src1) {
  Dst dst;
  linx_cvt_package(dst, src0, src1);
  return dst;
}

template <int RMode, int Sat, class Dst, class Src>
LINX_CVT_INLINE Dst linx_cvt_package_as(const Src &src0, const Src &src1) {
  Dst dst;
  linx_cvt_package<RMode, Sat>(dst, src0, src1);
  return dst;
}


#ifndef BLKV_BF16_OPS_HPP
#define BLKV_BF16_OPS_HPP

#ifndef BLKV_BF16_INLINE
#define BLKV_BF16_INLINE __attribute__((always_inline)) inline
#endif

#ifndef BLKV_BF16_STORAGE
#ifdef __blkc_bf16_STORAGE
#define BLKV_BF16_STORAGE(d) __blkc_bf16_STORAGE(d)
#else
#define BLKV_BF16_STORAGE(d) ((d).data)
#endif
#endif

// 默认生成：", -> %0.h"
// 如果你的 asm parser 使用新格式 ", %0.h"，可在 include 前定义：
// #define BLKV_BF16_DST_PREFIX ", %0.h"
#ifndef BLKV_BF16_DST_PREFIX
#define BLKV_BF16_DST_PREFIX ", -> %0.h"
#endif

// 如果后端实际 mnemonic 是 v.musb，可在 include 前定义：
// #define BLKV_BF16_FMSUB_MNEMONIC "musb"
#ifndef BLKV_BF16_FMSUB_MNEMONIC
#define BLKV_BF16_FMSUB_MNEMONIC "fmsub"
#endif

#ifndef BLKV_BF16X2_FMSUB_MNEMONIC
#define BLKV_BF16X2_FMSUB_MNEMONIC BLKV_BF16_FMSUB_MNEMONIC
#endif

enum BlkvBf16RMode {
  BLKV_RNONE = 0,
  BLKV_RNE   = 1,
  BLKV_RTZ   = 2,
  BLKV_RDN   = 3,
  BLKV_RUP   = 4,
  BLKV_RNA   = 5,
  BLKV_RTO   = 6,
  BLKV_RHB   = 7,
};

enum BlkvBf16Sat {
  BLKV_NOSAT = 0,
  BLKV_SAT   = 1,
};

#define BLKV_BF16_EMIT_UNARY(OP, DST, SRC, RMODE_STR, SAT_STR)               \
  asm volatile("v." OP " %1.bf" BLKV_BF16_DST_PREFIX ", " RMODE_STR           \
               ", " SAT_STR "\n"                                              \
               : "=r"(BLKV_BF16_STORAGE(DST))                                \
               : "r"(BLKV_BF16_STORAGE(SRC)))

#define BLKV_BF16_EMIT_BINARY(OP, DST, SRC_L, SRC_R, RMODE_STR, SAT_STR)      \
  asm volatile("v." OP " %1.bf, %2.bf" BLKV_BF16_DST_PREFIX ", " RMODE_STR    \
               ", " SAT_STR "\n"                                              \
               : "=r"(BLKV_BF16_STORAGE(DST))                                \
               : "r"(BLKV_BF16_STORAGE(SRC_L)),                               \
                 "r"(BLKV_BF16_STORAGE(SRC_R)))

#define BLKV_BF16_EMIT_TERNARY(OP, DST, SRC_L, SRC_R, SRC_A,                 \
                               RMODE_STR, SAT_STR)                           \
  asm volatile("v." OP " %1.bf, %2.bf, %3.bf" BLKV_BF16_DST_PREFIX ", "       \
               RMODE_STR ", " SAT_STR "\n"                                    \
               : "=r"(BLKV_BF16_STORAGE(DST))                                \
               : "r"(BLKV_BF16_STORAGE(SRC_L)),                               \
                 "r"(BLKV_BF16_STORAGE(SRC_R)),                               \
                 "r"(BLKV_BF16_STORAGE(SRC_A)))

#define BLKV_BF16_EMIT_FMAX(DST, SRC_L, SRC_R)                                \
  asm volatile("v.fmax %1.bf, %2.bf" BLKV_BF16_DST_PREFIX "\n"                \
               : "=r"(BLKV_BF16_STORAGE(DST))                                \
               : "r"(BLKV_BF16_STORAGE(SRC_L)),                               \
                 "r"(BLKV_BF16_STORAGE(SRC_R)))

#define BLKV_BF16_DISPATCH_RMODE_SAT(RMODE, SAT, EMIT, ...)                  \
  do {                                                                        \
    static_assert((RMODE) >= 0 && (RMODE) <= 7, "invalid bf16 RMode");        \
    static_assert((SAT) == 0 || (SAT) == 1, "invalid bf16 Sat");              \
    if constexpr ((RMODE) == 0 && (SAT) == 0) {                               \
      EMIT(__VA_ARGS__, "RNONE", "nosat");                                    \
    } else if constexpr ((RMODE) == 0 && (SAT) == 1) {                        \
      EMIT(__VA_ARGS__, "RNONE", "sat");                                      \
    } else if constexpr ((RMODE) == 1 && (SAT) == 0) {                        \
      EMIT(__VA_ARGS__, "RNE", "nosat");                                      \
    } else if constexpr ((RMODE) == 1 && (SAT) == 1) {                        \
      EMIT(__VA_ARGS__, "RNE", "sat");                                        \
    } else if constexpr ((RMODE) == 2 && (SAT) == 0) {                        \
      EMIT(__VA_ARGS__, "RTZ", "nosat");                                      \
    } else if constexpr ((RMODE) == 2 && (SAT) == 1) {                        \
      EMIT(__VA_ARGS__, "RTZ", "sat");                                        \
    } else if constexpr ((RMODE) == 3 && (SAT) == 0) {                        \
      EMIT(__VA_ARGS__, "RDN", "nosat");                                      \
    } else if constexpr ((RMODE) == 3 && (SAT) == 1) {                        \
      EMIT(__VA_ARGS__, "RDN", "sat");                                        \
    } else if constexpr ((RMODE) == 4 && (SAT) == 0) {                        \
      EMIT(__VA_ARGS__, "RUP", "nosat");                                      \
    } else if constexpr ((RMODE) == 4 && (SAT) == 1) {                        \
      EMIT(__VA_ARGS__, "RUP", "sat");                                        \
    } else if constexpr ((RMODE) == 5 && (SAT) == 0) {                        \
      EMIT(__VA_ARGS__, "RNA", "nosat");                                      \
    } else if constexpr ((RMODE) == 5 && (SAT) == 1) {                        \
      EMIT(__VA_ARGS__, "RNA", "sat");                                        \
    } else if constexpr ((RMODE) == 6 && (SAT) == 0) {                        \
      EMIT(__VA_ARGS__, "RTO", "nosat");                                      \
    } else if constexpr ((RMODE) == 6 && (SAT) == 1) {                        \
      EMIT(__VA_ARGS__, "RTO", "sat");                                        \
    } else if constexpr ((RMODE) == 7 && (SAT) == 0) {                        \
      EMIT(__VA_ARGS__, "RHB", "nosat");                                      \
    } else {                                                                  \
      EMIT(__VA_ARGS__, "RHB", "sat");                                        \
    }                                                                         \
  } while (0)

#define BLKV_DEFINE_BF16_UNARY(NAME, OP)                                      \
  template <int RMode = BLKV_RNONE, int Sat = BLKV_NOSAT, class Src>         \
  BLKV_BF16_INLINE void blkv_bf16_##NAME(__bf16 &dst,                         \
                                         const Src &src) {                    \
    BLKV_BF16_DISPATCH_RMODE_SAT(RMode, Sat, BLKV_BF16_EMIT_UNARY,            \
                                 OP, dst, src);                               \
  }                                                                           \
                                                                              \
  template <int RMode = BLKV_RNONE, int Sat = BLKV_NOSAT, class Src>         \
  BLKV_BF16_INLINE __bf16 blkv_bf16_##NAME(const Src &src) {                  \
    __bf16 dst;                                                               \
    blkv_bf16_##NAME<RMode, Sat>(dst, src);                                   \
    return dst;                                                               \
  }

#define BLKV_DEFINE_BF16_BINARY(NAME, OP)                                     \
  template <int RMode = BLKV_RNONE, int Sat = BLKV_NOSAT,                    \
            class SrcL, class SrcR>                                           \
  BLKV_BF16_INLINE void blkv_bf16_##NAME(__bf16 &dst,                         \
                                         const SrcL &src_l,                   \
                                         const SrcR &src_r) {                 \
    BLKV_BF16_DISPATCH_RMODE_SAT(RMode, Sat, BLKV_BF16_EMIT_BINARY,           \
                                 OP, dst, src_l, src_r);                      \
  }                                                                           \
                                                                              \
  template <int RMode = BLKV_RNONE, int Sat = BLKV_NOSAT,                    \
            class SrcL, class SrcR>                                           \
  BLKV_BF16_INLINE __bf16 blkv_bf16_##NAME(const SrcL &src_l,                 \
                                           const SrcR &src_r) {               \
    __bf16 dst;                                                               \
    blkv_bf16_##NAME<RMode, Sat>(dst, src_l, src_r);                          \
    return dst;                                                               \
  }

#define BLKV_DEFINE_BF16_TERNARY(NAME, OP)                                    \
  template <int RMode = BLKV_RNONE, int Sat = BLKV_NOSAT,                    \
            class SrcL, class SrcR, class SrcA>                               \
  BLKV_BF16_INLINE void blkv_bf16_##NAME(__bf16 &dst,                         \
                                         const SrcL &src_l,                   \
                                         const SrcR &src_r,                   \
                                         const SrcA &src_a) {                 \
    BLKV_BF16_DISPATCH_RMODE_SAT(RMode, Sat, BLKV_BF16_EMIT_TERNARY,          \
                                 OP, dst, src_l, src_r, src_a);               \
  }                                                                           \
                                                                              \
  template <int RMode = BLKV_RNONE, int Sat = BLKV_NOSAT,                    \
            class SrcL, class SrcR, class SrcA>                               \
  BLKV_BF16_INLINE __bf16 blkv_bf16_##NAME(const SrcL &src_l,                 \
                                           const SrcR &src_r,                 \
                                           const SrcA &src_a) {               \
    __bf16 dst;                                                               \
    blkv_bf16_##NAME<RMode, Sat>(dst, src_l, src_r, src_a);                   \
    return dst;                                                               \
  }


#define BLKV_DEFINE_BF16_UNARY_ALIAS(ALIAS, TARGET)                           \
  template <int RMode = BLKV_RNONE, int Sat = BLKV_NOSAT>                    \
  BLKV_BF16_INLINE void blkv_bf16_##ALIAS(__bf16 &dst,                        \
                                          const __bf16 &src) {                \
    blkv_bf16_##TARGET<RMode, Sat>(dst, src);                                 \
  }                                                                           \
                                                                              \
  template <int RMode = BLKV_RNONE, int Sat = BLKV_NOSAT>                    \
  BLKV_BF16_INLINE __bf16 blkv_bf16_##ALIAS(const __bf16 &src) {              \
    return blkv_bf16_##TARGET<RMode, Sat>(src);                               \
  }

#define BLKV_DEFINE_BF16_BINARY_ALIAS(ALIAS, TARGET)                          \
  template <int RMode = BLKV_RNONE, int Sat = BLKV_NOSAT>                    \
  BLKV_BF16_INLINE void blkv_bf16_##ALIAS(__bf16 &dst,                        \
                                          const __bf16 &src_l,                \
                                          const __bf16 &src_r) {              \
    blkv_bf16_##TARGET<RMode, Sat>(dst, src_l, src_r);                        \
  }                                                                           \
                                                                              \
  template <int RMode = BLKV_RNONE, int Sat = BLKV_NOSAT>                    \
  BLKV_BF16_INLINE __bf16 blkv_bf16_##ALIAS(const __bf16 &src_l,              \
                                            const __bf16 &src_r) {            \
    return blkv_bf16_##TARGET<RMode, Sat>(src_l, src_r);                      \
  }

#define BLKV_DEFINE_BF16_TERNARY_ALIAS(ALIAS, TARGET)                         \
  template <int RMode = BLKV_RNONE, int Sat = BLKV_NOSAT>                    \
  BLKV_BF16_INLINE void blkv_bf16_##ALIAS(__bf16 &dst,                        \
                                          const __bf16 &src_l,                \
                                          const __bf16 &src_r,                \
                                          const __bf16 &src_a) {              \
    blkv_bf16_##TARGET<RMode, Sat>(dst, src_l, src_r, src_a);                 \
  }                                                                           \
                                                                              \
  template <int RMode = BLKV_RNONE, int Sat = BLKV_NOSAT>                    \
  BLKV_BF16_INLINE __bf16 blkv_bf16_##ALIAS(const __bf16 &src_l,              \
                                            const __bf16 &src_r,              \
                                            const __bf16 &src_a) {            \
    return blkv_bf16_##TARGET<RMode, Sat>(src_l, src_r, src_a);               \
  }

// v.fadd / v.fsub / v.fmul / v.fdiv
BLKV_DEFINE_BF16_BINARY(fadd, "fadd")
BLKV_DEFINE_BF16_BINARY(fsub, "fsub")
BLKV_DEFINE_BF16_BINARY(fmul, "fmul")
BLKV_DEFINE_BF16_BINARY(fdiv, "fdiv")

// v.fexp
BLKV_DEFINE_BF16_UNARY(fexp, "fexp")

// v.fsqrt
BLKV_DEFINE_BF16_UNARY(fsqrt, "fsqrt")

// v.fmadd / v.fmsub
BLKV_DEFINE_BF16_TERNARY(fmadd, "fmadd")
BLKV_DEFINE_BF16_TERNARY(fmsub, BLKV_BF16_FMSUB_MNEMONIC)

// v.fmax: no rmode / sat
template <class SrcL, class SrcR>
BLKV_BF16_INLINE void blkv_bf16_fmax(__bf16 &dst,
                                     const SrcL &src_l,
                                     const SrcR &src_r) {
  BLKV_BF16_EMIT_FMAX(dst, src_l, src_r);
}

template <class SrcL, class SrcR>
BLKV_BF16_INLINE __bf16 blkv_bf16_fmax(const SrcL &src_l,
                                       const SrcR &src_r) {
  __bf16 dst;
  blkv_bf16_fmax(dst, src_l, src_r);
  return dst;
}


// short aliases
BLKV_DEFINE_BF16_BINARY_ALIAS(add, fadd)
BLKV_DEFINE_BF16_BINARY_ALIAS(sub, fsub)
BLKV_DEFINE_BF16_BINARY_ALIAS(mul, fmul)
BLKV_DEFINE_BF16_BINARY_ALIAS(div, fdiv)

BLKV_DEFINE_BF16_UNARY_ALIAS(exp, fexp)
BLKV_DEFINE_BF16_UNARY_ALIAS(sqrt, fsqrt)

BLKV_DEFINE_BF16_TERNARY_ALIAS(madd, fmadd)
BLKV_DEFINE_BF16_TERNARY_ALIAS(fmmad, fmadd)
BLKV_DEFINE_BF16_TERNARY_ALIAS(msub, fmsub)

BLKV_BF16_INLINE void blkv_bf16_max(__bf16 &dst,
                                    const __bf16 &src_l,
                                    const __bf16 &src_r) {
  blkv_bf16_fmax(dst, src_l, src_r);
}

BLKV_BF16_INLINE __bf16 blkv_bf16_max(const __bf16 &src_l,
                                      const __bf16 &src_r) {
  return blkv_bf16_fmax(src_l, src_r);
}

//===----------------------------------------------------------------------===//
// bf16x2 ops
//===----------------------------------------------------------------------===//

#ifndef BLKV_BF16X2_TYPE
#define BLKV_BF16X2_TYPE __bf16x2
#endif

#ifndef BLKV_BF16X2_STORAGE
#ifdef __blkc_bf16x2_STORAGE
#define BLKV_BF16X2_STORAGE(d) __blkc_bf16x2_STORAGE(d)
#else
#define BLKV_BF16X2_STORAGE(d) ((d).data)
#endif
#endif

// 默认生成：", -> %0.h"
// 如果你的 asm parser 使用新格式 ", %0.h"，可在 include 前定义：
// #define BLKV_BF16X2_DST_PREFIX ", %0.h"
// 如果目的类型也需要 x2，比如 ", -> %0.hx2"，也可以在 include 前改。
#ifndef BLKV_BF16X2_DST_PREFIX
#define BLKV_BF16X2_DST_PREFIX ", -> %0.w"
#endif

#define BLKV_BF16X2_EMIT_UNARY(OP, DST, SRC, RMODE_STR, SAT_STR)             \
  asm volatile("v." OP " %1.bfx2" BLKV_BF16X2_DST_PREFIX ", " RMODE_STR       \
               ", " SAT_STR "\n"                                              \
               : "=r"(BLKV_BF16X2_STORAGE(DST))                              \
               : "r"(BLKV_BF16X2_STORAGE(SRC)))

#define BLKV_BF16X2_EMIT_BINARY(OP, DST, SRC_L, SRC_R, RMODE_STR, SAT_STR)    \
  asm volatile("v." OP " %1.bfx2, %2.bfx2" BLKV_BF16X2_DST_PREFIX ", "        \
               RMODE_STR ", " SAT_STR "\n"                                    \
               : "=r"(BLKV_BF16X2_STORAGE(DST))                              \
               : "r"(BLKV_BF16X2_STORAGE(SRC_L)),                             \
                 "r"(BLKV_BF16X2_STORAGE(SRC_R)))

#define BLKV_BF16X2_EMIT_TERNARY(OP, DST, SRC_L, SRC_R, SRC_A,               \
                                  RMODE_STR, SAT_STR)                       \
  asm volatile("v." OP " %1.bfx2, %2.bfx2, %3.bfx2"                          \
               BLKV_BF16X2_DST_PREFIX ", " RMODE_STR ", " SAT_STR "\n"       \
               : "=r"(BLKV_BF16X2_STORAGE(DST))                              \
               : "r"(BLKV_BF16X2_STORAGE(SRC_L)),                             \
                 "r"(BLKV_BF16X2_STORAGE(SRC_R)),                             \
                 "r"(BLKV_BF16X2_STORAGE(SRC_A)))

#define BLKV_BF16X2_EMIT_FMAX(DST, SRC_L, SRC_R)                              \
  asm volatile("v.fmax %1.bfx2, %2.bfx2" BLKV_BF16X2_DST_PREFIX "\n"          \
               : "=r"(BLKV_BF16X2_STORAGE(DST))                              \
               : "r"(BLKV_BF16X2_STORAGE(SRC_L)),                             \
                 "r"(BLKV_BF16X2_STORAGE(SRC_R)))

#define BLKV_DEFINE_BF16X2_UNARY(NAME, OP)                                    \
  template <int RMode = BLKV_RNONE, int Sat = BLKV_NOSAT, class Src>         \
  BLKV_BF16_INLINE void blkv_bf16x2_##NAME(BLKV_BF16X2_TYPE &dst,             \
                                           const Src &src) {                  \
    BLKV_BF16_DISPATCH_RMODE_SAT(RMode, Sat, BLKV_BF16X2_EMIT_UNARY,          \
                                 OP, dst, src);                               \
  }                                                                           \
                                                                              \
  template <int RMode = BLKV_RNONE, int Sat = BLKV_NOSAT, class Src>         \
  BLKV_BF16_INLINE BLKV_BF16X2_TYPE blkv_bf16x2_##NAME(const Src &src) {      \
    BLKV_BF16X2_TYPE dst;                                                     \
    blkv_bf16x2_##NAME<RMode, Sat>(dst, src);                                 \
    return dst;                                                               \
  }

#define BLKV_DEFINE_BF16X2_BINARY(NAME, OP)                                   \
  template <int RMode = BLKV_RNONE, int Sat = BLKV_NOSAT,                    \
            class SrcL, class SrcR>                                           \
  BLKV_BF16_INLINE void blkv_bf16x2_##NAME(BLKV_BF16X2_TYPE &dst,             \
                                           const SrcL &src_l,                 \
                                           const SrcR &src_r) {               \
    BLKV_BF16_DISPATCH_RMODE_SAT(RMode, Sat, BLKV_BF16X2_EMIT_BINARY,         \
                                 OP, dst, src_l, src_r);                      \
  }                                                                           \
                                                                              \
  template <int RMode = BLKV_RNONE, int Sat = BLKV_NOSAT,                    \
            class SrcL, class SrcR>                                           \
  BLKV_BF16_INLINE BLKV_BF16X2_TYPE blkv_bf16x2_##NAME(const SrcL &src_l,     \
                                                       const SrcR &src_r) {   \
    BLKV_BF16X2_TYPE dst;                                                     \
    blkv_bf16x2_##NAME<RMode, Sat>(dst, src_l, src_r);                        \
    return dst;                                                               \
  }

#define BLKV_DEFINE_BF16X2_TERNARY(NAME, OP)                                 \
  template <int RMode = BLKV_RNONE, int Sat = BLKV_NOSAT,                    \
            class SrcL, class SrcR, class SrcA>                               \
  BLKV_BF16_INLINE void blkv_bf16x2_##NAME(BLKV_BF16X2_TYPE &dst,             \
                                           const SrcL &src_l,                 \
                                           const SrcR &src_r,                 \
                                           const SrcA &src_a) {               \
    BLKV_BF16_DISPATCH_RMODE_SAT(RMode, Sat, BLKV_BF16X2_EMIT_TERNARY,        \
                                 OP, dst, src_l, src_r, src_a);               \
  }                                                                           \
                                                                              \
  template <int RMode = BLKV_RNONE, int Sat = BLKV_NOSAT,                    \
            class SrcL, class SrcR, class SrcA>                               \
  BLKV_BF16_INLINE BLKV_BF16X2_TYPE                                           \
  blkv_bf16x2_##NAME(const SrcL &src_l,                                       \
                     const SrcR &src_r,                                       \
                     const SrcA &src_a) {                                     \
    BLKV_BF16X2_TYPE dst;                                                     \
    blkv_bf16x2_##NAME<RMode, Sat>(dst, src_l, src_r, src_a);                 \
    return dst;                                                               \
  }

#define BLKV_DEFINE_BF16X2_UNARY_ALIAS(ALIAS, TARGET)                         \
  template <int RMode = BLKV_RNONE, int Sat = BLKV_NOSAT>                    \
  BLKV_BF16_INLINE void blkv_bf16x2_##ALIAS(BLKV_BF16X2_TYPE &dst,            \
                                            const BLKV_BF16X2_TYPE &src) {    \
    blkv_bf16x2_##TARGET<RMode, Sat>(dst, src);                               \
  }                                                                           \
                                                                              \
  template <int RMode = BLKV_RNONE, int Sat = BLKV_NOSAT>                    \
  BLKV_BF16_INLINE BLKV_BF16X2_TYPE                                           \
  blkv_bf16x2_##ALIAS(const BLKV_BF16X2_TYPE &src) {                          \
    return blkv_bf16x2_##TARGET<RMode, Sat>(src);                             \
  }

#define BLKV_DEFINE_BF16X2_BINARY_ALIAS(ALIAS, TARGET)                        \
  template <int RMode = BLKV_RNONE, int Sat = BLKV_NOSAT>                    \
  BLKV_BF16_INLINE void blkv_bf16x2_##ALIAS(BLKV_BF16X2_TYPE &dst,            \
                                            const BLKV_BF16X2_TYPE &src_l,    \
                                            const BLKV_BF16X2_TYPE &src_r) {  \
    blkv_bf16x2_##TARGET<RMode, Sat>(dst, src_l, src_r);                      \
  }                                                                           \
                                                                              \
  template <int RMode = BLKV_RNONE, int Sat = BLKV_NOSAT>                    \
  BLKV_BF16_INLINE BLKV_BF16X2_TYPE                                           \
  blkv_bf16x2_##ALIAS(const BLKV_BF16X2_TYPE &src_l,                          \
                      const BLKV_BF16X2_TYPE &src_r) {                        \
    return blkv_bf16x2_##TARGET<RMode, Sat>(src_l, src_r);                    \
  }

#define BLKV_DEFINE_BF16X2_TERNARY_ALIAS(ALIAS, TARGET)                      \
  template <int RMode = BLKV_RNONE, int Sat = BLKV_NOSAT>                    \
  BLKV_BF16_INLINE void blkv_bf16x2_##ALIAS(BLKV_BF16X2_TYPE &dst,            \
                                            const BLKV_BF16X2_TYPE &src_l,    \
                                            const BLKV_BF16X2_TYPE &src_r,    \
                                            const BLKV_BF16X2_TYPE &src_a) {  \
    blkv_bf16x2_##TARGET<RMode, Sat>(dst, src_l, src_r, src_a);               \
  }                                                                           \
                                                                              \
  template <int RMode = BLKV_RNONE, int Sat = BLKV_NOSAT>                    \
  BLKV_BF16_INLINE BLKV_BF16X2_TYPE                                           \
  blkv_bf16x2_##ALIAS(const BLKV_BF16X2_TYPE &src_l,                          \
                      const BLKV_BF16X2_TYPE &src_r,                          \
                      const BLKV_BF16X2_TYPE &src_a) {                        \
    return blkv_bf16x2_##TARGET<RMode, Sat>(src_l, src_r, src_a);             \
  }

// v.fadd / v.fsub / v.fmul / v.fdiv
BLKV_DEFINE_BF16X2_BINARY(fadd, "fadd")
BLKV_DEFINE_BF16X2_BINARY(fsub, "fsub")
BLKV_DEFINE_BF16X2_BINARY(fmul, "fmul")
BLKV_DEFINE_BF16X2_BINARY(fdiv, "fdiv")

// v.fexp
BLKV_DEFINE_BF16X2_UNARY(fexp, "fexp")

// v.fsqrt
BLKV_DEFINE_BF16X2_UNARY(fsqrt, "fsqrt")

// v.fmadd / v.fmsub
BLKV_DEFINE_BF16X2_TERNARY(fmadd, "fmadd")
BLKV_DEFINE_BF16X2_TERNARY(fmsub, BLKV_BF16X2_FMSUB_MNEMONIC)

// v.fmax: no rmode / sat
template <class SrcL, class SrcR>
BLKV_BF16_INLINE void blkv_bf16x2_fmax(BLKV_BF16X2_TYPE &dst,
                                       const SrcL &src_l,
                                       const SrcR &src_r) {
  BLKV_BF16X2_EMIT_FMAX(dst, src_l, src_r);
}

template <class SrcL, class SrcR>
BLKV_BF16_INLINE BLKV_BF16X2_TYPE blkv_bf16x2_fmax(const SrcL &src_l,
                                                   const SrcR &src_r) {
  BLKV_BF16X2_TYPE dst;
  blkv_bf16x2_fmax(dst, src_l, src_r);
  return dst;
}

// short aliases
BLKV_DEFINE_BF16X2_BINARY_ALIAS(add, fadd)
BLKV_DEFINE_BF16X2_BINARY_ALIAS(sub, fsub)
BLKV_DEFINE_BF16X2_BINARY_ALIAS(mul, fmul)
BLKV_DEFINE_BF16X2_BINARY_ALIAS(div, fdiv)

BLKV_DEFINE_BF16X2_UNARY_ALIAS(exp, fexp)
BLKV_DEFINE_BF16X2_UNARY_ALIAS(sqrt, fsqrt)

BLKV_DEFINE_BF16X2_TERNARY_ALIAS(madd, fmadd)
BLKV_DEFINE_BF16X2_TERNARY_ALIAS(fmmad, fmadd)
BLKV_DEFINE_BF16X2_TERNARY_ALIAS(msub, fmsub)

BLKV_BF16_INLINE void blkv_bf16x2_max(BLKV_BF16X2_TYPE &dst,
                                      const BLKV_BF16X2_TYPE &src_l,
                                      const BLKV_BF16X2_TYPE &src_r) {
  blkv_bf16x2_fmax(dst, src_l, src_r);
}

BLKV_BF16_INLINE BLKV_BF16X2_TYPE
blkv_bf16x2_max(const BLKV_BF16X2_TYPE &src_l,
                const BLKV_BF16X2_TYPE &src_r) {
  return blkv_bf16x2_fmax(src_l, src_r);
}

#endif // BLKV_BF16_OPS_HPP


#endif
#endif
