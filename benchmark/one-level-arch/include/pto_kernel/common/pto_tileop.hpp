#ifndef PTO_COMMON_PTO_TILEOP_HPP
#define PTO_COMMON_PTO_TILEOP_HPP

#include <stdint.h>

#include <pto_kernel/pto/linx/impl/backend.hpp>

namespace pto {

constexpr int DYNAMIC = -1;

enum class Location : uint8_t {
  Vec,
  Left,
  Right,
  Acc,
};

enum class BLayout : uint8_t {
  RowMajor = 0,
  ColMajor = 1,
};

template <int Rows_, int Cols_> struct RowMajor {
  static constexpr int Rows = Rows_;
  static constexpr int Cols = Cols_;
  static constexpr bool IsRowMajor = true;
};

template <int Rows_, int Cols_> struct ColMajor {
  static constexpr int Rows = Rows_;
  static constexpr int Cols = Cols_;
  static constexpr bool IsRowMajor = false;
};

template <typename Element_, typename Layout_> struct global_tensor {
  using DType = Element_;
  using Layout = Layout_;
};

namespace detail {

using ptrdiff_builtin_t = __PTRDIFF_TYPE__;

template <typename... Ts> using void_t = void;

template <int Value> struct StaticIndex {
  static constexpr int value = Value;
  constexpr operator int() const { return Value; }
};

template <int Begin, int End, typename Fn>
inline __attribute__((always_inline)) void static_for(Fn &&fn) {
  if constexpr (Begin < End) {
    fn(StaticIndex<Begin>{});
    static_for<Begin + 1, End>(fn);
  }
}

// TMA format selectors used by B.ARG in canonical v0.57.
constexpr long long kLayoutNorm = 0ll;  // NORM.normal
constexpr long long kLayoutND2NZ = 2ll; // ND2NZ.normal
constexpr long long kLayoutND2ZN = 3ll; // ND2ZN.normal
constexpr long long kLayoutDN2ZN = 8ll; // DN2ZN.normal
constexpr long long kLayoutDN2NZ = 9ll; // DN2NZ.normal

template <typename TileT> constexpr unsigned tileBytes() {
  constexpr int rows = TileT::Rows;
  constexpr int cols = TileT::Cols;
  constexpr unsigned bytes =
      static_cast<unsigned>(rows * cols * sizeof(typename TileT::DType));
  static_assert(bytes > 0u,
                "PTO Linx canonical v0.57: tile bytes must be positive");
  return bytes;
}

template <typename TileT> constexpr unsigned tileSizeCode() {
  static_assert(tileBytes<TileT>() <= linx::detail::kMaxTileBytes,
                "PTO Linx canonical v0.57: tile size exceeds 4KB");
  // Keep one 4 KiB carrier across data types. TCVT changes the element type,
  // but not the architectural tile-register capacity or its SSA identity.
  return 8u;
}

template <typename TileT> constexpr unsigned tileDTypeCode() {
  return linx::detail::DTypeCode<typename TileT::DType>::value;
}

template <typename TileT> constexpr long long tileLayoutCode() {
  return TileT::LayoutTag == BLayout::RowMajor ? 0ll : 1ll;
}

template <typename GTensor> constexpr long long gmStrideBytes() {
  constexpr long long elemBytes =
      static_cast<long long>(sizeof(typename GTensor::DType));
  if constexpr (GTensor::Layout::IsRowMajor)
    return static_cast<long long>(GTensor::Layout::Cols) * elemBytes;
  return static_cast<long long>(GTensor::Layout::Rows) * elemBytes;
}

template <typename GTensor, typename TileT>
constexpr long long tensorTileLayoutCode() {
  if constexpr (TileT::Loc == Location::Left || TileT::Loc == Location::Acc) {
    return GTensor::Layout::IsRowMajor ? kLayoutND2ZN : kLayoutDN2ZN;
  }
  if constexpr (TileT::Loc == Location::Right) {
    return GTensor::Layout::IsRowMajor ? kLayoutND2NZ : kLayoutDN2NZ;
  }
  return kLayoutNorm;
}

template <typename TileT> constexpr long long tileLB0() {
  return TileT::ColValid > 0 ? static_cast<long long>(TileT::ColValid)
                             : static_cast<long long>(TileT::Cols);
}

template <typename TileT> constexpr long long tileLB1() {
  return TileT::RowValid > 0 ? static_cast<long long>(TileT::RowValid)
                             : static_cast<long long>(TileT::Rows);
}

template <typename GTensor, typename TileT>
inline ptrdiff_builtin_t tileOffset(int tileRow, int tileCol) {
  const int row = tileRow * TileT::Rows;
  const int col = tileCol * TileT::Cols;
  if constexpr (GTensor::Layout::IsRowMajor) {
    return static_cast<ptrdiff_builtin_t>(row) * GTensor::Layout::Cols + col;
  }
  return static_cast<ptrdiff_builtin_t>(col) * GTensor::Layout::Rows + row;
}

template <typename AddressLike>
inline auto addressPtr(const AddressLike &addr) -> decltype(addr.ptr()) {
  return addr.ptr();
}

template <typename T> inline T *addressPtr(T *addr) { return addr; }

template <typename T> inline const T *addressPtr(const T *addr) { return addr; }

template <typename AddressLike, typename TileT, typename = void>
struct AddressDesc {
  static constexpr long long Layout = tileLayoutCode<TileT>();
  static constexpr long long LB0 = tileLB0<TileT>();
  static constexpr long long LB1 = tileLB1<TileT>();
  static constexpr long long StrideBytes = 0ll;
};

template <typename AddressLike, typename TileT>
struct AddressDesc<
    AddressLike, TileT,
    void_t<decltype(AddressLike::kLayoutCode), decltype(AddressLike::kLB0),
           decltype(AddressLike::kLB1), decltype(AddressLike::kStrideBytes)>> {
  static constexpr long long Layout = AddressLike::kLayoutCode;
  static constexpr long long LB0 = AddressLike::kLB0;
  static constexpr long long LB1 = AddressLike::kLB1;
  static constexpr long long StrideBytes = AddressLike::kStrideBytes;
};

template <typename AddressLike, typename TileT>
constexpr long long addressLayoutCode() {
  return AddressDesc<AddressLike, TileT>::Layout;
}

template <typename AddressLike, typename TileT>
constexpr long long addressLB0() {
  return AddressDesc<AddressLike, TileT>::LB0;
}

template <typename AddressLike, typename TileT>
constexpr long long addressLB1() {
  return AddressDesc<AddressLike, TileT>::LB1;
}

template <typename AddressLike, typename TileT>
constexpr long long addressStrideBytes() {
  return AddressDesc<AddressLike, TileT>::StrideBytes;
}

} // namespace detail

template <Location Loc_, typename Element_, int Rows_, int Cols_,
          BLayout Layout_ = BLayout::RowMajor, int RowValid_ = Rows_,
          int ColValid_ = Cols_>
struct Tile {
  using DType = Element_;
  using RawTile = linx::detail::RawTile;
  using TileDType = Tile *;
  using ConstTileDType = const Tile *;

  static constexpr Location Loc = Loc_;
  static constexpr int Rows = Rows_;
  static constexpr int Cols = Cols_;
  static constexpr int RowValid = RowValid_;
  static constexpr int ColValid = ColValid_;
  static constexpr int ValidRow = RowValid_;
  static constexpr int ValidCol = ColValid_;
  static constexpr BLayout LayoutTag = Layout_;
  static constexpr int RowStride = LayoutTag == BLayout::RowMajor ? Cols_ : 1;
  static constexpr int ColStride = LayoutTag == BLayout::RowMajor ? 1 : Rows_;

  static_assert(RowValid_ == DYNAMIC || (RowValid_ > 0 && RowValid_ <= Rows_),
                "PTO Linx: valid rows must fit the physical tile");
  static_assert(ColValid_ == DYNAMIC || (ColValid_ > 0 && ColValid_ <= Cols_),
                "PTO Linx: valid columns must fit the physical tile");

  Tile()
      : valid_rows_(RowValid_ == DYNAMIC ? Rows_ : RowValid_),
        valid_cols_(ColValid_ == DYNAMIC ? Cols_ : ColValid_) {}

  Tile(int validRows, int validCols)
      : valid_rows_(validRows), valid_cols_(validCols) {}

  template <typename Scalar> explicit Tile(Scalar scalar) {
    raw_ = linx::detail::teplSplat<0x019u, detail::tileSizeCode<Tile>(),
                                   detail::tileDTypeCode<Tile>(), 2u>(
        scalar, GetValidCol(), GetValidRow(), Cols);
  }

  int GetValidRow() const { return valid_rows_; }
  int GetValidCol() const { return valid_cols_; }

  void SetValidShape(int validRows, int validCols) {
    valid_rows_ = validRows;
    valid_cols_ = validCols;
  }

  RawTile &raw() { return raw_; }
  const RawTile &raw() const { return raw_; }
  TileDType data() { return this; }
  ConstTileDType data() const { return this; }

private:
  RawTile raw_{};
  int valid_rows_ = RowValid_ == DYNAMIC ? Rows_ : RowValid_;
  int valid_cols_ = ColValid_ == DYNAMIC ? Cols_ : ColValid_;
};

template <typename Element_, int Rows_, int Cols_, int RowValid_ = Rows_,
          int ColValid_ = Cols_>
using TileLeft = Tile<Location::Left, Element_, Rows_, Cols_, BLayout::ColMajor,
                      RowValid_, ColValid_>;

template <typename Element_, int Rows_, int Cols_, int RowValid_ = Rows_,
          int ColValid_ = Cols_>
using TileRight = Tile<Location::Right, Element_, Rows_, Cols_,
                       BLayout::RowMajor, RowValid_, ColValid_>;

template <typename Element_, int Rows_, int Cols_, int RowValid_ = Rows_,
          int ColValid_ = Cols_>
using TileAcc = Tile<Location::Acc, Element_, Rows_, Cols_, BLayout::ColMajor,
                     RowValid_, ColValid_>;

template <typename GTensor, typename TileT> class global_iterator {
public:
  using Element = typename GTensor::DType;

  explicit global_iterator(Element *base) : base_(base) {}

  struct tile_address {
    using TensorType = GTensor;
    using TileType = TileT;
    static constexpr long long kLayoutCode =
        detail::tensorTileLayoutCode<GTensor, TileT>();
    // TMA contract: LB0/LB1 are GM-side inner/outer counts.
    // ND(row-major): inner=cols, outer=rows; DN(column-major): inner=rows,
    // outer=cols.
    static constexpr long long kLB0 = GTensor::Layout::IsRowMajor
                                          ? detail::tileLB1<TileT>()
                                          : detail::tileLB0<TileT>();
    static constexpr long long kLB1 = GTensor::Layout::IsRowMajor
                                          ? detail::tileLB0<TileT>()
                                          : detail::tileLB1<TileT>();
    static constexpr long long kStrideBytes = detail::gmStrideBytes<GTensor>();

    Element *base;
    int tileRow;
    int tileCol;

    Element *ptr() const {
      return base + detail::tileOffset<GTensor, TileT>(tileRow, tileCol);
    }
  };

  tile_address operator()(int tileRow, int tileCol) const {
    return tile_address{base_, tileRow, tileCol};
  }

private:
  Element *base_;
};

namespace tepl {
constexpr unsigned TADD = 0x000u;
constexpr unsigned TSUB = 0x001u;
constexpr unsigned TMUL = 0x002u;
constexpr unsigned TDIV = 0x003u;
constexpr unsigned TMAX = 0x004u;
constexpr unsigned TMIN = 0x005u;
constexpr unsigned TAND = 0x006u;
constexpr unsigned TOR = 0x007u;
constexpr unsigned TXOR = 0x008u;
constexpr unsigned TSHL = 0x009u;
constexpr unsigned TSHR = 0x00au;
constexpr unsigned TRELU = 0x00bu;
constexpr unsigned TPRELU = 0x00cu;
constexpr unsigned TCVT = 0x00du;
constexpr unsigned TEXP = 0x00eu;
constexpr unsigned TLOG = 0x00fu;
constexpr unsigned TSQRT = 0x010u;
constexpr unsigned TRSQRT = 0x011u;
constexpr unsigned TROWMAX = 0x012u;
constexpr unsigned TROWMIN = 0x013u;
constexpr unsigned TROWSUM = 0x014u;
constexpr unsigned TCOLMAX = 0x015u;
constexpr unsigned TCOLMIN = 0x016u;
constexpr unsigned TCOLSUM = 0x017u;
constexpr unsigned TRECIP = 0x018u;
constexpr unsigned TEXPANDS = 0x019u;
constexpr unsigned TGATHER = 0x01au;
constexpr unsigned TSCATTER = 0x01bu;
constexpr unsigned TRESHAPE = 0x01cu;
constexpr unsigned TTRANSPOSE = 0x01du;
constexpr unsigned TCOLEXPAND = 0x01eu;
constexpr unsigned TROWEXPAND = 0x01fu;
constexpr unsigned TADDS = 0x020u;
constexpr unsigned TSUBS = 0x021u;
constexpr unsigned TMULS = 0x022u;
constexpr unsigned TDIVS = 0x023u;
constexpr unsigned TMAXS = 0x024u;
constexpr unsigned TMINS = 0x025u;
constexpr unsigned TANDS = 0x026u;
constexpr unsigned TORS = 0x027u;
constexpr unsigned TXORS = 0x028u;
constexpr unsigned TSHLS = 0x029u;
constexpr unsigned TSHRS = 0x02au;
constexpr unsigned TCMP = 0x02bu;
constexpr unsigned TSEL = 0x02cu;
constexpr unsigned TABS = 0x02du;
constexpr unsigned TNOT = 0x02eu;
constexpr unsigned TCMPS = 0x033u;
constexpr unsigned TSELS = 0x034u;
constexpr unsigned TCONCAT = 0x087u;
constexpr unsigned TSORT = 0x0c0u;
constexpr unsigned TMRGSORT = 0x0c1u;
constexpr unsigned THISTOGRAM = 0x0c2u;
constexpr unsigned TPARTADD = 0x0c3u;
constexpr unsigned TPARTMUL = 0x0c4u;
constexpr unsigned TPARTMAX = 0x0c5u;
constexpr unsigned TPARTMIN = 0x0c6u;
constexpr unsigned TPARTARGMAX = 0x0c7u;
constexpr unsigned TPARTARGMIN = 0x0c8u;
} // namespace tepl

// Core tile ops used by PR5 FlashAttention bring-up.
template <typename DstTile, typename SrcAddress>
inline void TLOAD(DstTile &dst, const SrcAddress &src) {
  dst.raw() =
      linx::detail::tileTLoad<detail::tileSizeCode<DstTile>(),
                              detail::tileDTypeCode<DstTile>(),
                              detail::addressLayoutCode<SrcAddress, DstTile>()>(
          reinterpret_cast<const void *>(detail::addressPtr(src)),
          dst.GetValidCol(), dst.GetValidRow(), DstTile::Cols,
          detail::addressStrideBytes<SrcAddress, DstTile>());
}

template <typename DstAddress, typename SrcTile>
inline void TSTORE(const DstAddress &dst, SrcTile &src) {
  linx::detail::tileTStore<detail::tileSizeCode<SrcTile>(),
                           detail::tileDTypeCode<SrcTile>(),
                           detail::addressLayoutCode<DstAddress, SrcTile>()>(
      reinterpret_cast<void *>(detail::addressPtr(dst)), src.raw(),
      src.GetValidCol(), src.GetValidRow(), SrcTile::Cols,
      detail::addressStrideBytes<DstAddress, SrcTile>());
}

template <typename DstTile, typename SrcTile>
inline void TMOV(DstTile &dst, const SrcTile &src, unsigned mode = 0u) {
  dst.SetValidShape(src.GetValidRow(), src.GetValidCol());
  if (mode == 1u) {
    dst.raw() =
        linx::detail::tileTMov<detail::tileSizeCode<DstTile>(),
                               detail::tileDTypeCode<DstTile>(),
                               detail::tileLayoutCode<DstTile>(), 1u, 1u>(
            src.raw());
  } else {
    dst.raw() =
        linx::detail::tileTMov<detail::tileSizeCode<DstTile>(),
                               detail::tileDTypeCode<DstTile>(),
                               detail::tileLayoutCode<DstTile>(), 1u, 0u>(
            src.raw());
  }
}

template <typename TileRes, typename TileLeft_, typename TileRight_>
inline void TMATMUL(TileRes &dst, const TileLeft_ &lhs, const TileRight_ &rhs) {
  // Canonical v0.57 compiler policy:
  // tile_bytes = ceil(m*n*k*elem_bits/8) must fit <=4KB
  // (m=Rows, n=Cols, k=lhs.Cols).
  constexpr unsigned M = static_cast<unsigned>(TileRes::Rows);
  constexpr unsigned N = static_cast<unsigned>(TileRes::Cols);
  constexpr unsigned K = static_cast<unsigned>(TileLeft_::Cols);
  dst.raw() = linx::detail::cubeMamulb<M, N, K>(lhs.raw(), rhs.raw());
}

template <typename TileRes, typename TileLeft_, typename TileRight_>
inline void TMATMUL_ACC(TileRes &dst, TileRes &acc, const TileLeft_ &lhs,
                        const TileRight_ &rhs) {
  constexpr unsigned M = static_cast<unsigned>(TileRes::Rows);
  constexpr unsigned N = static_cast<unsigned>(TileRes::Cols);
  constexpr unsigned K = static_cast<unsigned>(TileLeft_::Cols);
  dst.raw() =
      linx::detail::cubeMamulbAcc<M, N, K>(acc.raw(), lhs.raw(), rhs.raw());
}

template <typename TileRes, typename TileLeft_, typename TileRight_>
inline void MATMACC(TileRes &dst, const TileLeft_ &lhs, const TileRight_ &rhs) {
  // Keep strict CUBE accumulator-chain legality: materialize the product with
  // TMATMUL, then accumulate explicitly with TEPL add.
  TileRes product;
  TMATMUL(product, lhs, rhs);
  TADD(dst, dst, product);
}

template <typename DstTile, typename SrcTile>
inline void TCVT(DstTile &dst, const SrcTile &src) {
  dst.SetValidShape(src.GetValidRow(), src.GetValidCol());
  dst.raw() =
      linx::detail::teplUnary<tepl::TCVT, detail::tileSizeCode<DstTile>(),
                              detail::tileDTypeCode<DstTile>(),
                              detail::tileDTypeCode<SrcTile>()>(
          src.raw(), dst.GetValidCol(), dst.GetValidRow(), DstTile::Cols);
}

template <typename DstTile, typename SrcTile0, typename SrcTile1>
inline void TADD(DstTile &dst, const SrcTile0 &src0, const SrcTile1 &src1) {
  dst.SetValidShape(src0.GetValidRow(), src0.GetValidCol());
  dst.raw() =
      linx::detail::teplBinary<tepl::TADD, detail::tileSizeCode<DstTile>(),
                               detail::tileDTypeCode<DstTile>()>(
          src0.raw(), src1.raw(), dst.GetValidCol(), dst.GetValidRow(),
          DstTile::Cols);
}

template <typename DstTile, typename SrcTile0, typename SrcTile1>
inline void TSUB(DstTile &dst, const SrcTile0 &src0, const SrcTile1 &src1) {
  dst.SetValidShape(src0.GetValidRow(), src0.GetValidCol());
  dst.raw() =
      linx::detail::teplBinary<tepl::TSUB, detail::tileSizeCode<DstTile>(),
                               detail::tileDTypeCode<DstTile>()>(
          src0.raw(), src1.raw(), dst.GetValidCol(), dst.GetValidRow(),
          DstTile::Cols);
}

template <typename DstTile, typename SrcTile0, typename SrcTile1>
inline void TMUL(DstTile &dst, const SrcTile0 &src0, const SrcTile1 &src1) {
  dst.SetValidShape(src0.GetValidRow(), src0.GetValidCol());
  dst.raw() =
      linx::detail::teplBinary<tepl::TMUL, detail::tileSizeCode<DstTile>(),
                               detail::tileDTypeCode<DstTile>()>(
          src0.raw(), src1.raw(), dst.GetValidCol(), dst.GetValidRow(),
          DstTile::Cols);
}

template <typename DstTile, typename SrcTile0, typename SrcTile1>
inline void TMAX(DstTile &dst, const SrcTile0 &src0, const SrcTile1 &src1) {
  dst.SetValidShape(src0.GetValidRow(), src0.GetValidCol());
  dst.raw() =
      linx::detail::teplBinary<tepl::TMAX, detail::tileSizeCode<DstTile>(),
                               detail::tileDTypeCode<DstTile>()>(
          src0.raw(), src1.raw(), dst.GetValidCol(), dst.GetValidRow(),
          DstTile::Cols);
}

template <typename DstTile, typename SrcTile0, typename SrcTile1>
inline void TDIV(DstTile &dst, const SrcTile0 &src0, const SrcTile1 &src1) {
  dst.SetValidShape(src0.GetValidRow(), src0.GetValidCol());
  dst.raw() =
      linx::detail::teplBinary<tepl::TDIV, detail::tileSizeCode<DstTile>(),
                               detail::tileDTypeCode<DstTile>()>(
          src0.raw(), src1.raw(), dst.GetValidCol(), dst.GetValidRow(),
          DstTile::Cols);
}

template <typename DstTile, typename SrcTile0, typename SrcTile1>
inline void TMIN(DstTile &dst, const SrcTile0 &src0, const SrcTile1 &src1) {
  dst.SetValidShape(src0.GetValidRow(), src0.GetValidCol());
  dst.raw() =
      linx::detail::teplBinary<tepl::TMIN, detail::tileSizeCode<DstTile>(),
                               detail::tileDTypeCode<DstTile>()>(
          src0.raw(), src1.raw(), dst.GetValidCol(), dst.GetValidRow(),
          DstTile::Cols);
}

template <typename DstTile, typename SrcTile0, typename SrcTile1>
inline void TAND(DstTile &dst, const SrcTile0 &src0, const SrcTile1 &src1) {
  dst.SetValidShape(src0.GetValidRow(), src0.GetValidCol());
  dst.raw() =
      linx::detail::teplBinary<tepl::TAND, detail::tileSizeCode<DstTile>(),
                               detail::tileDTypeCode<DstTile>()>(
          src0.raw(), src1.raw(), dst.GetValidCol(), dst.GetValidRow(),
          DstTile::Cols);
}

template <typename DstTile, typename SrcTile0, typename SrcTile1>
inline void TOR(DstTile &dst, const SrcTile0 &src0, const SrcTile1 &src1) {
  dst.SetValidShape(src0.GetValidRow(), src0.GetValidCol());
  dst.raw() =
      linx::detail::teplBinary<tepl::TOR, detail::tileSizeCode<DstTile>(),
                               detail::tileDTypeCode<DstTile>()>(
          src0.raw(), src1.raw(), dst.GetValidCol(), dst.GetValidRow(),
          DstTile::Cols);
}

template <typename DstTile, typename SrcTile0, typename SrcTile1>
inline void TXOR(DstTile &dst, const SrcTile0 &src0, const SrcTile1 &src1) {
  dst.SetValidShape(src0.GetValidRow(), src0.GetValidCol());
  dst.raw() =
      linx::detail::teplBinary<tepl::TXOR, detail::tileSizeCode<DstTile>(),
                               detail::tileDTypeCode<DstTile>()>(
          src0.raw(), src1.raw(), dst.GetValidCol(), dst.GetValidRow(),
          DstTile::Cols);
}

template <typename DstTile, typename SrcTile0, typename SrcTile1>
inline void TSHL(DstTile &dst, const SrcTile0 &src0, const SrcTile1 &src1) {
  dst.SetValidShape(src0.GetValidRow(), src0.GetValidCol());
  dst.raw() =
      linx::detail::teplBinary<tepl::TSHL, detail::tileSizeCode<DstTile>(),
                               detail::tileDTypeCode<DstTile>()>(
          src0.raw(), src1.raw(), dst.GetValidCol(), dst.GetValidRow(),
          DstTile::Cols);
}

template <typename DstTile, typename SrcTile0, typename SrcTile1>
inline void TSHR(DstTile &dst, const SrcTile0 &src0, const SrcTile1 &src1) {
  dst.SetValidShape(src0.GetValidRow(), src0.GetValidCol());
  dst.raw() =
      linx::detail::teplBinary<tepl::TSHR, detail::tileSizeCode<DstTile>(),
                               detail::tileDTypeCode<DstTile>()>(
          src0.raw(), src1.raw(), dst.GetValidCol(), dst.GetValidRow(),
          DstTile::Cols);
}

template <typename DstTile, typename SrcTile>
inline void TROWMAX(DstTile &dst, const SrcTile &src) {
  dst.SetValidShape(src.GetValidRow(), src.GetValidCol());
  dst.raw() =
      linx::detail::teplUnary<tepl::TROWMAX, detail::tileSizeCode<DstTile>(),
                              detail::tileDTypeCode<DstTile>()>(
          src.raw(), dst.GetValidCol(), dst.GetValidRow(), DstTile::Cols);
  dst.SetValidShape(src.GetValidRow(), 1);
}

template <typename DstTile, typename SrcTile>
inline void TROWMIN(DstTile &dst, const SrcTile &src) {
  dst.SetValidShape(src.GetValidRow(), src.GetValidCol());
  dst.raw() =
      linx::detail::teplUnary<tepl::TROWMIN, detail::tileSizeCode<DstTile>(),
                              detail::tileDTypeCode<DstTile>()>(
          src.raw(), dst.GetValidCol(), dst.GetValidRow(), DstTile::Cols);
  dst.SetValidShape(src.GetValidRow(), 1);
}

template <typename DstTile, typename SrcTile>
inline void TROWSUM(DstTile &dst, const SrcTile &src) {
  dst.SetValidShape(src.GetValidRow(), src.GetValidCol());
  dst.raw() =
      linx::detail::teplUnary<tepl::TROWSUM, detail::tileSizeCode<DstTile>(),
                              detail::tileDTypeCode<DstTile>()>(
          src.raw(), dst.GetValidCol(), dst.GetValidRow(), DstTile::Cols);
  dst.SetValidShape(src.GetValidRow(), 1);
}

template <typename DstTile, typename SrcTile>
inline void TCOLMAX(DstTile &dst, const SrcTile &src) {
  dst.SetValidShape(src.GetValidRow(), src.GetValidCol());
  dst.raw() =
      linx::detail::teplUnary<tepl::TCOLMAX, detail::tileSizeCode<DstTile>(),
                              detail::tileDTypeCode<DstTile>()>(
          src.raw(), dst.GetValidCol(), dst.GetValidRow(), DstTile::Cols);
  dst.SetValidShape(1, src.GetValidCol());
}

template <typename DstTile, typename SrcTile>
inline void TCOLMIN(DstTile &dst, const SrcTile &src) {
  dst.SetValidShape(src.GetValidRow(), src.GetValidCol());
  dst.raw() =
      linx::detail::teplUnary<tepl::TCOLMIN, detail::tileSizeCode<DstTile>(),
                              detail::tileDTypeCode<DstTile>()>(
          src.raw(), dst.GetValidCol(), dst.GetValidRow(), DstTile::Cols);
  dst.SetValidShape(1, src.GetValidCol());
}

template <typename DstTile, typename SrcTile>
inline void TCOLSUM(DstTile &dst, const SrcTile &src) {
  dst.SetValidShape(src.GetValidRow(), src.GetValidCol());
  dst.raw() =
      linx::detail::teplUnary<tepl::TCOLSUM, detail::tileSizeCode<DstTile>(),
                              detail::tileDTypeCode<DstTile>()>(
          src.raw(), dst.GetValidCol(), dst.GetValidRow(), DstTile::Cols);
  dst.SetValidShape(1, src.GetValidCol());
}

template <typename DstTile, typename SrcTile>
inline void TRELU(DstTile &dst, const SrcTile &src) {
  dst.SetValidShape(src.GetValidRow(), src.GetValidCol());
  dst.raw() =
      linx::detail::teplUnary<tepl::TRELU, detail::tileSizeCode<DstTile>(),
                              detail::tileDTypeCode<DstTile>()>(
          src.raw(), dst.GetValidCol(), dst.GetValidRow(), DstTile::Cols);
}

template <typename DstTile, typename SrcTile>
inline void TEXP(DstTile &dst, const SrcTile &src) {
  dst.SetValidShape(src.GetValidRow(), src.GetValidCol());
  dst.raw() =
      linx::detail::teplUnary<tepl::TEXP, detail::tileSizeCode<DstTile>(),
                              detail::tileDTypeCode<DstTile>()>(
          src.raw(), dst.GetValidCol(), dst.GetValidRow(), DstTile::Cols);
}

template <typename DstTile, typename SrcTile>
inline void TLOG(DstTile &dst, const SrcTile &src) {
  dst.SetValidShape(src.GetValidRow(), src.GetValidCol());
  dst.raw() =
      linx::detail::teplUnary<tepl::TLOG, detail::tileSizeCode<DstTile>(),
                              detail::tileDTypeCode<DstTile>()>(
          src.raw(), dst.GetValidCol(), dst.GetValidRow(), DstTile::Cols);
}

template <typename DstTile, typename SrcTile>
inline void TSQRT(DstTile &dst, const SrcTile &src) {
  dst.SetValidShape(src.GetValidRow(), src.GetValidCol());
  dst.raw() =
      linx::detail::teplUnary<tepl::TSQRT, detail::tileSizeCode<DstTile>(),
                              detail::tileDTypeCode<DstTile>()>(
          src.raw(), dst.GetValidCol(), dst.GetValidRow(), DstTile::Cols);
}

template <typename DstTile, typename SrcTile>
inline void TRSQRT(DstTile &dst, const SrcTile &src) {
  dst.SetValidShape(src.GetValidRow(), src.GetValidCol());
  dst.raw() =
      linx::detail::teplUnary<tepl::TRSQRT, detail::tileSizeCode<DstTile>(),
                              detail::tileDTypeCode<DstTile>()>(
          src.raw(), dst.GetValidCol(), dst.GetValidRow(), DstTile::Cols);
}

template <typename DstTile, typename SrcTile>
inline void TRECIP(DstTile &dst, const SrcTile &src) {
  dst.SetValidShape(src.GetValidRow(), src.GetValidCol());
  dst.raw() =
      linx::detail::teplUnary<tepl::TRECIP, detail::tileSizeCode<DstTile>(),
                              detail::tileDTypeCode<DstTile>()>(
          src.raw(), dst.GetValidCol(), dst.GetValidRow(), DstTile::Cols);
}

template <typename DstTile, typename SrcTile>
inline void TABS(DstTile &dst, const SrcTile &src) {
  dst.SetValidShape(src.GetValidRow(), src.GetValidCol());
  dst.raw() =
      linx::detail::teplUnary<tepl::TABS, detail::tileSizeCode<DstTile>(),
                              detail::tileDTypeCode<DstTile>()>(
          src.raw(), dst.GetValidCol(), dst.GetValidRow(), DstTile::Cols);
}

template <typename DstTile, typename SrcTile>
inline void TNOT(DstTile &dst, const SrcTile &src) {
  dst.SetValidShape(src.GetValidRow(), src.GetValidCol());
  dst.raw() =
      linx::detail::teplUnary<tepl::TNOT, detail::tileSizeCode<DstTile>(),
                              detail::tileDTypeCode<DstTile>()>(
          src.raw(), dst.GetValidCol(), dst.GetValidRow(), DstTile::Cols);
}

template <typename DstTile, typename SrcTile>
inline void TRESHAPE(DstTile &dst, const SrcTile &src) {
  dst.SetValidShape(src.GetValidRow(), src.GetValidCol());
  dst.raw() =
      linx::detail::teplUnary<tepl::TRESHAPE, detail::tileSizeCode<DstTile>(),
                              detail::tileDTypeCode<DstTile>()>(
          src.raw(), dst.GetValidCol(), dst.GetValidRow(), DstTile::Cols);
}

template <typename DstTile, typename SrcTile>
inline void TTRANSPOSE(DstTile &dst, const SrcTile &src) {
  dst.SetValidShape(src.GetValidRow(), src.GetValidCol());
  dst.raw() =
      linx::detail::teplUnary<tepl::TTRANSPOSE, detail::tileSizeCode<DstTile>(),
                              detail::tileDTypeCode<DstTile>()>(
          src.raw(), dst.GetValidCol(), dst.GetValidRow(), DstTile::Cols);
  dst.SetValidShape(src.GetValidCol(), src.GetValidRow());
}

template <typename DstTile, typename SrcTile>
inline void TSORT(DstTile &dst, const SrcTile &src) {
  dst.SetValidShape(src.GetValidRow(), src.GetValidCol());
  dst.raw() =
      linx::detail::teplUnary<tepl::TSORT, detail::tileSizeCode<DstTile>(),
                              detail::tileDTypeCode<DstTile>()>(
          src.raw(), dst.GetValidCol(), dst.GetValidRow(), DstTile::Cols);
}

template <typename DstTile, typename SrcTile>
inline void THISTOGRAM(DstTile &dst, const SrcTile &src) {
  dst.SetValidShape(src.GetValidRow(), src.GetValidCol());
  dst.raw() =
      linx::detail::teplUnary<tepl::THISTOGRAM, detail::tileSizeCode<DstTile>(),
                              detail::tileDTypeCode<DstTile>()>(
          src.raw(), dst.GetValidCol(), dst.GetValidRow(), DstTile::Cols);
  dst.SetValidShape(1, src.GetValidCol());
}

template <typename DstTile, typename SrcTile0, typename SrcTile1>
inline void TGATHER(DstTile &dst, const SrcTile0 &src0, const SrcTile1 &src1) {
  dst.SetValidShape(src0.GetValidRow(), src0.GetValidCol());
  dst.raw() =
      linx::detail::teplBinary<tepl::TGATHER, detail::tileSizeCode<DstTile>(),
                               detail::tileDTypeCode<DstTile>()>(
          src0.raw(), src1.raw(), dst.GetValidCol(), dst.GetValidRow(),
          DstTile::Cols);
}

template <typename DstTile, typename SrcTile0, typename SrcTile1>
inline void TSCATTER(DstTile &dst, const SrcTile0 &src0, const SrcTile1 &src1) {
  dst.SetValidShape(src0.GetValidRow(), src0.GetValidCol());
  dst.raw() =
      linx::detail::teplBinary<tepl::TSCATTER, detail::tileSizeCode<DstTile>(),
                               detail::tileDTypeCode<DstTile>()>(
          src0.raw(), src1.raw(), dst.GetValidCol(), dst.GetValidRow(),
          DstTile::Cols);
}

template <typename DstTile, typename SrcTile, typename Scalar>
inline void TMULS(DstTile &dst, const SrcTile &src, Scalar scalar) {
  dst.SetValidShape(src.GetValidRow(), src.GetValidCol());
  dst.raw() =
      linx::detail::teplBinaryScalar<tepl::TMULS,
                                     detail::tileSizeCode<DstTile>(),
                                     detail::tileDTypeCode<DstTile>(), 1u>(
          src.raw(), scalar, dst.GetValidCol(), dst.GetValidRow(),
          DstTile::Cols);
}

template <typename DstTile, typename SrcTile, typename Scalar>
inline void TADDS(DstTile &dst, const SrcTile &src, Scalar scalar) {
  dst.SetValidShape(src.GetValidRow(), src.GetValidCol());
  dst.raw() =
      linx::detail::teplBinaryScalar<tepl::TADDS,
                                     detail::tileSizeCode<DstTile>(),
                                     detail::tileDTypeCode<DstTile>(), 1u>(
          src.raw(), scalar, dst.GetValidCol(), dst.GetValidRow(),
          DstTile::Cols);
}

template <typename DstTile, typename SrcTile, typename Scalar>
inline void TSUBS(DstTile &dst, const SrcTile &src, Scalar scalar) {
  dst.SetValidShape(src.GetValidRow(), src.GetValidCol());
  dst.raw() =
      linx::detail::teplBinaryScalar<tepl::TSUBS,
                                     detail::tileSizeCode<DstTile>(),
                                     detail::tileDTypeCode<DstTile>(), 1u>(
          src.raw(), scalar, dst.GetValidCol(), dst.GetValidRow(),
          DstTile::Cols);
}

template <typename DstTile, typename SrcTile, typename Scalar>
inline void TDIVS(DstTile &dst, const SrcTile &src, Scalar scalar) {
  dst.SetValidShape(src.GetValidRow(), src.GetValidCol());
  dst.raw() =
      linx::detail::teplBinaryScalar<tepl::TDIVS,
                                     detail::tileSizeCode<DstTile>(),
                                     detail::tileDTypeCode<DstTile>(), 1u>(
          src.raw(), scalar, dst.GetValidCol(), dst.GetValidRow(),
          DstTile::Cols);
}

template <typename DstTile, typename SrcTile, typename Scalar>
inline void TMAXS(DstTile &dst, const SrcTile &src, Scalar scalar) {
  dst.SetValidShape(src.GetValidRow(), src.GetValidCol());
  dst.raw() =
      linx::detail::teplBinaryScalar<tepl::TMAXS,
                                     detail::tileSizeCode<DstTile>(),
                                     detail::tileDTypeCode<DstTile>(), 1u>(
          src.raw(), scalar, dst.GetValidCol(), dst.GetValidRow(),
          DstTile::Cols);
}

template <typename DstTile, typename SrcTile, typename Scalar>
inline void TMINS(DstTile &dst, const SrcTile &src, Scalar scalar) {
  dst.SetValidShape(src.GetValidRow(), src.GetValidCol());
  dst.raw() =
      linx::detail::teplBinaryScalar<tepl::TMINS,
                                     detail::tileSizeCode<DstTile>(),
                                     detail::tileDTypeCode<DstTile>(), 1u>(
          src.raw(), scalar, dst.GetValidCol(), dst.GetValidRow(),
          DstTile::Cols);
}

template <typename DstTile, typename SrcTile, typename Scalar>
inline void TXORS(DstTile &dst, const SrcTile &src, Scalar scalar) {
  dst.SetValidShape(src.GetValidRow(), src.GetValidCol());
  dst.raw() =
      linx::detail::teplBinaryScalar<tepl::TXORS,
                                     detail::tileSizeCode<DstTile>(),
                                     detail::tileDTypeCode<DstTile>(), 1u>(
          src.raw(), scalar, dst.GetValidCol(), dst.GetValidRow(),
          DstTile::Cols);
}

template <typename DstTile, typename SrcTile, typename Scalar>
inline void TSHLS(DstTile &dst, const SrcTile &src, Scalar scalar) {
  dst.SetValidShape(src.GetValidRow(), src.GetValidCol());
  dst.raw() =
      linx::detail::teplBinaryScalar<tepl::TSHLS,
                                     detail::tileSizeCode<DstTile>(),
                                     detail::tileDTypeCode<DstTile>(), 1u>(
          src.raw(), scalar, dst.GetValidCol(), dst.GetValidRow(),
          DstTile::Cols);
}

template <typename DstTile, typename SrcTile, typename Scalar>
inline void TSHRS(DstTile &dst, const SrcTile &src, Scalar scalar) {
  dst.SetValidShape(src.GetValidRow(), src.GetValidCol());
  dst.raw() =
      linx::detail::teplBinaryScalar<tepl::TSHRS,
                                     detail::tileSizeCode<DstTile>(),
                                     detail::tileDTypeCode<DstTile>(), 1u>(
          src.raw(), scalar, dst.GetValidCol(), dst.GetValidRow(),
          DstTile::Cols);
}

template <typename DstTile, typename Scalar>
inline void TEXPANDS(DstTile &dst, Scalar scalar) {
  dst.raw() =
      linx::detail::teplSplat<tepl::TEXPANDS, detail::tileSizeCode<DstTile>(),
                              detail::tileDTypeCode<DstTile>(), 2u>(
          scalar, dst.GetValidCol(), dst.GetValidRow(), DstTile::Cols);
}

template <typename DstTile, typename SrcTile>
inline void TCOLEXPAND(DstTile &dst, const SrcTile &src) {
  dst.raw() =
      linx::detail::teplUnary<tepl::TCOLEXPAND, detail::tileSizeCode<DstTile>(),
                              detail::tileDTypeCode<DstTile>()>(
          src.raw(), dst.GetValidCol(), dst.GetValidRow(), DstTile::Cols);
}

template <typename DstTile, typename SrcTile>
inline void TROWEXPAND(DstTile &dst, const SrcTile &src) {
  dst.raw() =
      linx::detail::teplUnary<tepl::TROWEXPAND, detail::tileSizeCode<DstTile>(),
                              detail::tileDTypeCode<DstTile>()>(
          src.raw(), dst.GetValidCol(), dst.GetValidRow(), DstTile::Cols);
}

template <typename DstTile, typename SrcTile>
inline void TEXPANDCOL(DstTile &dst, const SrcTile &src) {
  TCOLEXPAND(dst, src);
}

} // namespace pto

#endif // PTO_COMMON_PTO_TILEOP_HPP
