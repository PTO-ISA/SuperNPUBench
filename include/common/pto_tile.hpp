#ifndef PTO_TILE_HPP
#define PTO_TILE_HPP

#include "common/layout.hpp"
#include <common/type.hpp>

namespace pto {

/// Padding Value : keep SAME with asm encoding
enum class PadValue {
  Zero = 0,
  Max = 1,
  Min = 2,
  Null = 3,
};

/// Layout for GlobalTensor
/// ND: lower tow dimensions are arranged in RowMajor order;
/// DN: lower tow dimensions are arranged in ColMajor order;
/// NZ: lower tow dimensions are arranged in RowMajor order;
///     third and fourth dimensions are arranged in ColMajor order.
/// SCALE: 
/// MAX: 
enum class Layout {
  ND,
  DN,
  NZ,
  SCALE,
  MAX,
};

constexpr int DYNAMIC = -1;

template <int N1 = DYNAMIC, int N2 = DYNAMIC, int N3 = DYNAMIC, int N4 = DYNAMIC, int N5 = DYNAMIC>
struct Shape {
  static constexpr int staticShape[5] = {N1, N2, N3, N4, N5};

  inline Shape() {
    static_assert(
      (N1 == DYNAMIC) + (N2 == DYNAMIC) + (N3 == DYNAMIC) + (N4 == DYNAMIC) + (N5 == DYNAMIC) == 0,
      "0-parameter constructors is only applicable to Stride with 0 dynamic dimension."
    );
  }

  inline Shape(int n) {
    static_assert(
      (N1 == DYNAMIC) + (N2 == DYNAMIC) + (N3 == DYNAMIC) + (N4 == DYNAMIC) + (N5 == DYNAMIC) == 1,
      "1-parameter constructors is only applicable to Stride with 1 dynamic dimension."
    );
    if constexpr (N1 == DYNAMIC) shape[0] = n;
    if constexpr (N2 == DYNAMIC) shape[1] = n;
    if constexpr (N3 == DYNAMIC) shape[2] = n;
    if constexpr (N4 == DYNAMIC) shape[3] = n;
    if constexpr (N5 == DYNAMIC) shape[4] = n;
  }

  inline Shape(int n1, int n2) {
    static_assert(
      (N1 == DYNAMIC) + (N2 == DYNAMIC) + (N3 == DYNAMIC) + (N4 == DYNAMIC) + (N5 == DYNAMIC) == 2,
      "2-parameter constructors is only applicable to Stride with 2 dynamic dimension."
    );

    int idx = 0;
    const int vals[] = {n1, n2};
    if constexpr (N1 == DYNAMIC) shape[0] = vals[idx++];
    if constexpr (N2 == DYNAMIC) shape[1] = vals[idx++];
    if constexpr (N3 == DYNAMIC) shape[2] = vals[idx++];
    if constexpr (N4 == DYNAMIC) shape[3] = vals[idx++];
    if constexpr (N5 == DYNAMIC) shape[4] = vals[idx++];
  }

  inline Shape(int n1, int n2, int n3) {
    static_assert(
      (N1 == DYNAMIC) + (N2 == DYNAMIC) + (N3 == DYNAMIC) + (N4 == DYNAMIC) + (N5 == DYNAMIC) == 3,
      "3-parameter constructors is only applicable to Stride with 3 dynamic dimension."
    );
    int idx = 0;
    const int vals[] = {n1, n2, n3};
    if constexpr (N1 == DYNAMIC) shape[0] = vals[idx++];
    if constexpr (N2 == DYNAMIC) shape[1] = vals[idx++];
    if constexpr (N3 == DYNAMIC) shape[2] = vals[idx++];
    if constexpr (N4 == DYNAMIC) shape[3] = vals[idx++];
    if constexpr (N5 == DYNAMIC) shape[4] = vals[idx++];
  }

  inline Shape(int n1, int n2, int n3, int n4) {
    static_assert(
      (N1 == DYNAMIC) + (N2 == DYNAMIC) + (N3 == DYNAMIC) + (N4 == DYNAMIC) + (N5 == DYNAMIC) == 4,
      "4-parameter constructors is only applicable to Stride with 4 dynamic dimension."
    );
    int idx = 0;
    const int vals[] = {n1, n2, n3, n4};
    if constexpr (N1 == DYNAMIC) shape[0] = vals[idx++];
    if constexpr (N2 == DYNAMIC) shape[1] = vals[idx++];
    if constexpr (N3 == DYNAMIC) shape[2] = vals[idx++];
    if constexpr (N4 == DYNAMIC) shape[3] = vals[idx++];
    if constexpr (N5 == DYNAMIC) shape[4] = vals[idx++];
  }

  inline Shape(int n1, int n2, int n3, int n4, int n5)
  {
    static_assert(
      (N1 == DYNAMIC) + (N2 == DYNAMIC) + (N3 == DYNAMIC) + (N4 == DYNAMIC) + (N5 == DYNAMIC) == 5,
      "5-parameter constructors is only applicable to Stride with 5 dynamic dimension."
    );
    if constexpr (N1 == DYNAMIC) shape[0] = n1;
    if constexpr (N2 == DYNAMIC) shape[1] = n2;
    if constexpr (N3 == DYNAMIC) shape[2] = n3;
    if constexpr (N4 == DYNAMIC) shape[3] = n4;
    if constexpr (N5 == DYNAMIC) shape[4] = n5;
  }

public:
  int shape[5] = {1};
};

template <int SN1 = DYNAMIC, int SN2 = DYNAMIC, int SN3 = DYNAMIC, int SN4 = DYNAMIC, int SN5 = DYNAMIC>
struct Stride {
  static constexpr int staticStride[5] = {SN1, SN2, SN3, SN4, SN5};

  inline Stride() {  }

  inline Stride(int n) {
    static_assert(
      (SN1 == DYNAMIC) + (SN2 == DYNAMIC) + (SN3 == DYNAMIC) + (SN4 == DYNAMIC) + (SN5 == DYNAMIC) == 1,
      "1-parameter constructors is only applicable to Stride with 1 dynamic dimension."
    );

    if constexpr (SN1 == DYNAMIC) stride[0] = n;
    if constexpr (SN2 == DYNAMIC) stride[1] = n;
    if constexpr (SN3 == DYNAMIC) stride[2] = n;
    if constexpr (SN4 == DYNAMIC) stride[3] = n;
    if constexpr (SN5 == DYNAMIC) stride[4] = n;
  }

  inline Stride(int n1, int n2) {
    static_assert(
      (SN1 == DYNAMIC) + (SN2 == DYNAMIC) + (SN3 == DYNAMIC) + (SN4 == DYNAMIC) + (SN5 == DYNAMIC) == 2,
      "2-parameter constructors is only applicable to Stride with 2 dynamic dimension."
    );
    int idx = 0;
    const int vals[] = {n1, n2};
    if constexpr (SN1 == DYNAMIC) stride[0] = vals[idx++];
    if constexpr (SN2 == DYNAMIC) stride[1] = vals[idx++];
    if constexpr (SN3 == DYNAMIC) stride[2] = vals[idx++];
    if constexpr (SN4 == DYNAMIC) stride[3] = vals[idx++];
    if constexpr (SN5 == DYNAMIC) stride[4] = vals[idx++];
  }

  inline Stride(int n1, int n2, int n3) {
    static_assert(
      (SN1 == DYNAMIC) + (SN2 == DYNAMIC) + (SN3 == DYNAMIC) + (SN4 == DYNAMIC) + (SN5 == DYNAMIC) == 3,
      "3-parameter constructors is only applicable to Stride with 3 dynamic dimension."
    );
    int idx = 0;
    const int vals[] = {n1, n2, n3};
    if constexpr (SN1 == DYNAMIC) stride[0] = vals[idx++];
    if constexpr (SN2 == DYNAMIC) stride[1] = vals[idx++];
    if constexpr (SN3 == DYNAMIC) stride[2] = vals[idx++];
    if constexpr (SN4 == DYNAMIC) stride[3] = vals[idx++];
    if constexpr (SN5 == DYNAMIC) stride[4] = vals[idx++];
  }

  inline Stride(int n1, int n2, int n3, int n4) {
    static_assert(
      (SN1 == DYNAMIC) + (SN2 == DYNAMIC) + (SN3 == DYNAMIC) + (SN4 == DYNAMIC) + (SN5 == DYNAMIC) == 4,
      "4-parameter constructors is only applicable to Stride with 4 dynamic dimension."
    );
    int idx = 0;
    const int vals[] = {n1, n2, n3, n4};
    if constexpr (SN1 == DYNAMIC) stride[0] = vals[idx++];
    if constexpr (SN2 == DYNAMIC) stride[1] = vals[idx++];
    if constexpr (SN3 == DYNAMIC) stride[2] = vals[idx++];
    if constexpr (SN4 == DYNAMIC) stride[3] = vals[idx++];
    if constexpr (SN5 == DYNAMIC) stride[4] = vals[idx++];
  }

  inline Stride(int n1, int n2, int n3, int n4, int n5)
  {
    static_assert(
      (SN1 == DYNAMIC) + (SN2 == DYNAMIC) + (SN3 == DYNAMIC) + (SN4 == DYNAMIC) + (SN5 == DYNAMIC) == 5,
      "5-parameter constructors is only applicable to Stride with 5 dynamic dimension."
    );
    if constexpr (SN1 == DYNAMIC) stride[0] = n1;
    if constexpr (SN2 == DYNAMIC) stride[1] = n2;
    if constexpr (SN3 == DYNAMIC) stride[2] = n3;
    if constexpr (SN4 == DYNAMIC) stride[3] = n4;
    if constexpr (SN5 == DYNAMIC) stride[4] = n5;
  }

public:
  int stride[5] = {1};
};

template <typename Element_, typename Shape_, typename Stride_, Layout Layout_ = Layout::ND>
struct GlobalTensor {
  using Shape = Shape_;
  using Stride = Stride_;
  using DType = Element_;
  static constexpr Layout layout = Layout_;

  static const Shape defaultShape;
  static const Stride defaultStride;

  static constexpr int staticShape[5] = {Shape::staticShape[0], Shape::staticShape[1],
                                         Shape::staticShape[2], Shape::staticShape[3],
                                         Shape::staticShape[4]};
  static constexpr int staticStride[5] = {Stride::staticStride[0], Stride::staticStride[1],
                                          Stride::staticStride[2], Stride::staticStride[3],
                                          Stride::staticStride[4]};

  inline GlobalTensor(DType *data, const Shape &shape = defaultShape, const Stride &stride = defaultStride)
  {
    data_ = data;

    if constexpr (staticShape[0] == DYNAMIC) shape_.shape[0] = shape.shape[0];
    if constexpr (staticShape[1] == DYNAMIC) shape_.shape[1] = shape.shape[1];
    if constexpr (staticShape[2] == DYNAMIC) shape_.shape[2] = shape.shape[2];
    if constexpr (staticShape[3] == DYNAMIC) shape_.shape[3] = shape.shape[3];
    if constexpr (staticShape[4] == DYNAMIC) shape_.shape[4] = shape.shape[4];

    if constexpr (staticStride[0] == DYNAMIC) stride_.stride[0] = stride.stride[0];
    if constexpr (staticStride[1] == DYNAMIC) stride_.stride[1] = stride.stride[1];
    if constexpr (staticStride[2] == DYNAMIC) stride_.stride[2] = stride.stride[2];
    if constexpr (staticStride[3] == DYNAMIC) stride_.stride[3] = stride.stride[3];
    if constexpr (staticStride[4] == DYNAMIC) stride_.stride[4] = stride.stride[4];
  }

  inline int GetShape(const int dim)
  {
    switch (dim) {
      case 0: return GetShapeSize<staticShape[0]>(dim);
      case 1: return GetShapeSize<staticShape[1]>(dim);
      case 2: return GetShapeSize<staticShape[2]>(dim);
      case 3: return GetShapeSize<staticShape[3]>(dim);
      case 4: return GetShapeSize<staticShape[4]>(dim);
      default: return -1;
    }
  }

  inline int GetStride(const int dim)
  {
    switch (dim) {
      case 0: return GetStrideSize<staticStride[0]>(dim);
      case 1: return GetStrideSize<staticStride[1]>(dim);
      case 2: return GetStrideSize<staticStride[2]>(dim);
      case 3: return GetStrideSize<staticStride[3]>(dim);
      case 4: return GetStrideSize<staticStride[4]>(dim);
      default: return -1;
    }
  }

  DType *data() { return data_; }
  const DType *data() const { return data_; }

private:
  template <int StaticShape>
  inline int GetShapeSize(const int dim)
  {
    if constexpr (StaticShape == DYNAMIC)
      return shape_.shape[dim];
    else
      return StaticShape;
  }

  template <int StaticStride>
  inline int GetStrideSize(const int dim)
  {
    if constexpr (StaticStride == DYNAMIC)
      return stride_.stride[dim];
    else
      return StaticStride;
  }

  DType *data_;
  Shape shape_ = defaultShape;
  Stride stride_ = defaultStride;
};

template <typename Element_, typename Shape_, typename Stride_, Layout Layout_>
const typename GlobalTensor<Element_, Shape_, Stride_, Layout_>::Shape
GlobalTensor<Element_, Shape_, Stride_, Layout_>::defaultShape{ };

template <typename Element_, typename Shape_, typename Stride_, Layout Layout_>
const typename GlobalTensor<Element_, Shape_, Stride_, Layout_>::Stride
GlobalTensor<Element_, Shape_, Stride_, Layout_>::defaultStride{ };

template <Location Loc_, typename Element_, const int Rows_, const int Cols_,
          const BLayout BFractal_ = BLayout::RowMajor,
          const int RowValid_ = Rows_, const int ColValid_ = Cols_,
          const SLayout SFractal_ = SLayout::NoneBox,
          const int SFractalSize_ = 512,
          const PadValue PadVal_ = PadValue::Null>
struct Tile {
public:
  using DType = Element_;

  static constexpr int getInnerRow() {
    if constexpr (SFractalSize_ == 1024) { // output/acc
      static_assert(sizeof(DType) == 4, "Size of datatype != 4");
      return 16;
    } else {
      return isBoxedLayout
                ? (isInnerRowMajor ? 16 : byteSize / sizeof(DType))
                : 1;
    }
  }

  static constexpr int getInnerCol() {
      if constexpr (SFractalSize_ == 1024) { // output/acc
        static_assert(sizeof(DType) == 4, "Size of datatype != 4");
        return 16;
      } else {
        return isBoxedLayout
                  ? (isInnerRowMajor ? byteSize / sizeof(DType) : 16)
                  : 1;
      }
  }

  static constexpr Location Loc = Loc_;
  static constexpr int Rows = Rows_;
  static constexpr int Cols = Cols_;
  static constexpr int RowStride = BFractal_ == BLayout::RowMajor ? Cols : 1;
  static constexpr int ColStride = BFractal_ == BLayout::RowMajor ? 1 : Rows;

  static constexpr int kBytes = Rows_ * Cols_ * sizeof(DType);
  static_assert(kBytes % 512 == 0, "Tile size must be 512 bytes aligned");
  static_assert(((kBytes / 512 - 1) & (kBytes / 512)) == 0, "Tile size must by (512 * 2 ^ n) Bytes");
  //static_assert(kBytes >= 512 && kBytes <= 32 * 1024, "Tile size must be in [512B, 32kB]");


  static constexpr int ValidRow = RowValid_;
  static constexpr int ValidCol = ColValid_;
  static_assert(Rows > 0 && ValidRow <= Rows && Cols > 0 && ValidCol <= Cols,
                "Invalid Tile Layout.");

  static constexpr BLayout BFractal = BFractal_;
  static constexpr SLayout SFractal = SFractal_;
  static constexpr int Numel = Rows * Cols;
  static constexpr bool isRowMajor = BFractal_ == BLayout::RowMajor;

  static constexpr int SFractalSize = SFractalSize_;
  static constexpr PadValue PadVal = PadVal_;

  // constructor for static shape
  Tile() { };
  template <int RowMask = ValidRow, int ColMask = ValidCol>
  Tile(std::enable_if_t<(RowMask > 0) && (ColMask > 0), DType> val);

  // constructor for both dimensions are runtime variables
  template <int RowMask = ValidRow, int ColMask = ValidCol>
  Tile(std::enable_if_t<RowMask == -1 && ColMask == -1, size_t> VR,
       std::enable_if_t<RowMask == -1 && ColMask == -1, size_t> VC);
  template <int RowMask = ValidRow, int ColMask = ValidCol>
  Tile(DType val, std::enable_if_t<RowMask == -1 && ColMask == -1, size_t> VR,
                  std::enable_if_t<RowMask == -1 && ColMask == -1, size_t> VC);

  // constructor for row dimension is runtime variables
  template <int RowMask = ValidRow, int ColMask = ValidCol>
  Tile(std::enable_if_t<(RowMask == -1) && (ColMask > 0), size_t> VR);
  template <int RowMask = ValidRow, int ColMask = ValidCol>
  Tile(DType val, std::enable_if_t<(RowMask == -1) && (ColMask > 0), size_t> VR);

  // constructor for col dimension is runtime variables
  template <int RowMask = ValidRow, int ColMask = ValidCol>
  Tile(std::enable_if_t<(RowMask > 0) && (ColMask == -1), size_t> VC);
  template <int RowMask = ValidRow, int ColMask = ValidCol>
  Tile(DType val, std::enable_if_t<(RowMask > 0) && (ColMask == -1), size_t> VC);

  static constexpr int byteSize = 32;
  static constexpr bool isBoxedLayout = (SFractal != SLayout::NoneBox);
  static constexpr bool isInnerRowMajor = (SFractal == SLayout::RowMajor);
  static constexpr bool isInnerColMajor = (SFractal == SLayout::ColMajor);

  static constexpr int InnerRows = getInnerRow();
  static constexpr int InnerCols = getInnerCol();

  static constexpr int InnerNumel = InnerRows * InnerCols;

  static_assert(Rows % InnerRows == 0,
                "Layout rows must be divisible by inner box rows");
  static_assert(Cols % InnerCols == 0,
                "Layout cols must be divisible by inner box cols");

  // static_assert(
  //     (BFractal_ == BLayout::RowMajor && SFractal_ == SLayout::NoneBox && Cols * sizeof(DType) % 32 == 0) ||
  //     (BFractal_ == BLayout::ColMajor && SFractal_ == SLayout::NoneBox && Rows * sizeof(DType) % 32 == 0) ||
  //     (SFractal_ != SLayout::NoneBox) && (Rows % InnerRows == 0 && Cols % InnerCols == 0),
  //     "BFractal_ is RowMajor and SFractal_ is NoneBox: Rows must be 32 bytes align, \
  //       BFractal_ is ColMajor and SFractal_ is NoneBox: Cols must be 32 bytes align, \
  //       SFractal_ in not NoneBox: Rows/Cols must be integer multiple of InnerRows/InnerCols."
  //       );

  static_assert(SFractalSize_ == 512 || SFractalSize_ == 1024,
                "SFractalSize_ illegal");

#ifdef __linx
  using TileDType = DType tile_size(Rows *Cols);
#else
  using TileDType = DType[Rows * Cols];
#endif

  TileDType &data() { return data_; }
  const TileDType &data() const { return data_; }

  // record dynamic shape info
  int RowMaskInternal;
  int ColMaskInternal;

  template <int RowMask = ValidRow>
  static constexpr std::enable_if_t<(RowMask > 0), int> GetValidRow() {
    return RowMask;
  }

  template <int RowMask = ValidRow>
  std::enable_if_t<RowMask == -1, int> GetValidRow() const {
    return RowMaskInternal;
  }

  template <int ColMask = ValidCol>
  static constexpr std::enable_if_t<(ColMask > 0), int> GetValidCol() {
    return ColMask;
  }

  template <int ColMask = ValidCol>
  std::enable_if_t<ColMask == -1, int> GetValidCol() const {
    return ColMaskInternal;
  }

  void assignData(TileDType data) { data_ = data; }

  TileDType data_;
};

template <typename Element_, const int Rows_, const int Cols_,
          const int RowValid_ = Rows_, const int ColValid_ = Cols_>
using TileLeft =
  Tile<Location::Left, Element_, Rows_, Cols_, BLayout::ColMajor,
       RowValid_, ColValid_, SLayout::RowMajor, 512>;

template <typename Element_, const int Rows_, const int Cols_,
          const int RowValid_ = Rows_, const int ColValid_ = Cols_>
using TileRight =
  Tile<Location::Right, Element_, Rows_, Cols_, BLayout::RowMajor,
       RowValid_, ColValid_, SLayout::ColMajor, 512>;

template <typename Element_, const int Rows_, const int Cols_,
          const int RowValid_ = Rows_, const int ColValid_ = Cols_>
using TileAcc =
  Tile<Location::Acc, Element_, Rows_, Cols_, BLayout::ColMajor,
       RowValid_, ColValid_, SLayout::RowMajor, 1024>;

template <int Rows, int Cols, bool RowMajor>
struct stride_selector;

template <>
struct stride_selector<DYNAMIC, DYNAMIC, true> {
    using type = Stride<1, 1, DYNAMIC, DYNAMIC, DYNAMIC>;
};

template <int Cols>
struct stride_selector<DYNAMIC, Cols, true> {
    using type = Stride<1, 1, -1, Cols, 1>;
};

template <int Cols>
struct stride_selector<DYNAMIC, Cols, false> {
    using type = Stride<1, 1, -1, 1, -1>;
};

template <int Rows>
struct stride_selector<Rows, DYNAMIC, true> {
    using type = Stride<1, 1, -1, -1, 1>;
};
template <int Rows>
struct stride_selector<Rows, DYNAMIC, false> {
    using type = Stride<1, 1, -1, 1, Rows>;
};

template <int Rows, int Cols, bool RowMajor>
struct stride_selector {
    using type = std::conditional_t<RowMajor,
                      Stride<1, 1, Rows * Cols, Cols, 1>,
                      Stride<1, 1, Rows * Cols, 1, Rows>>;
};

template <typename Element_, typename MLayout_>
struct global_tensor {
public:
  using DType = Element_;
  using MLayout = MLayout_;

  static constexpr int kNumel = get_numel<MLayout>;
  static constexpr int kRows = num_rows<MLayout>;
  static constexpr int kCols = num_cols<MLayout>;
  static constexpr int kRowStride = row_stride<MLayout>;
  static constexpr int kColStride = col_stride<MLayout>;
  static constexpr LayoutEnum kType = layout_type<MLayout>;

  static constexpr bool isRowMajor = kType == LayoutEnum::kRowMajor;

  using shape_t = Shape<1, 1, 1, 1, 1>;
  using stride_t = typename stride_selector<kRows, kCols, isRowMajor>::type;
  static constexpr Layout layout_t = isRowMajor ? Layout::ND : Layout::DN;

  using Impl = GlobalTensor<DType, shape_t, stride_t, layout_t>;

  static constexpr int staticShape[5] = {
      Impl::staticShape[0], Impl::staticShape[1], Impl::staticShape[2],
      Impl::staticShape[3], Impl::staticShape[4]
  };

  static constexpr int staticStride[5] = {
      Impl::staticStride[0], Impl::staticStride[1], Impl::staticStride[2],
      Impl::staticStride[3], Impl::staticStride[4]
  };

  template <typename T = void,
            typename = std::enable_if_t<(kRows != DYNAMIC && kCols != DYNAMIC), T>>
  global_tensor(DType* data)
            : impl_(data), layout_(MLayout{}) {}

  template <typename T = void,
            typename = std::enable_if_t<(kRows == DYNAMIC && kCols != DYNAMIC) || (kRows != DYNAMIC && kCols == DYNAMIC), T>>
  global_tensor(DType* data, int stride) {
    if constexpr ((kRows == DYNAMIC && kCols != DYNAMIC && isRowMajor) || (kRows != DYNAMIC && kCols == DYNAMIC && !isRowMajor)) {
      impl_ = Impl(data, shape_t{}, stride_t(stride)), layout_(MLayout{});
    } else {
      impl_ = Impl(data, shape_t{}, stride_t(stride, stride)), layout_(MLayout{});
    }
  }

  template <typename T = void,
            typename = std::enable_if_t<(kRows == DYNAMIC && kCols == DYNAMIC), T>>
  global_tensor(DType* data, int dynamicRow, int dynamicCol)
            : impl_(data, shape_t{}, stride_t(dynamicRow*dynamicCol, dynamicCol, dynamicRow)), layout_(MLayout{}) {}

  const DType *data() const { return impl_.data(); }
  DType *data() { return impl_.data(); }

  int GetShape(int dim) const { return impl_.GetShape(dim); }
  int GetStride(int dim) const { return impl_.GetStride(dim); }

private:
  Impl impl_;
  MLayout layout_;
};

template <typename T> struct is_global : std::false_type {};
template <typename T> struct is_tile : std::false_type {
  static constexpr SLayout layout_enum = SLayout::NoneBox;
};

template <typename Element_, typename Layout_, typename Stride_>
struct is_global<GlobalTensor<Element_, Layout_, Stride_>> : std::true_type {};

template <typename Element_, typename Layout_>
struct is_global<global_tensor<Element_, Layout_>> : std::true_type {};

template <Location Loc_, typename Element_, const int Rows_, const int Cols_,
          const BLayout BFractal_, const int RowValid_, const int ColValid_,
          const SLayout SFractal_, const int SFractalSize_>
struct is_tile<Tile<Loc_, Element_, Rows_, Cols_, BFractal_, RowValid_,
                    ColValid_, SFractal_, SFractalSize_>> : std::true_type {
  static constexpr SLayout layout_enum = SFractal_;
};

template <typename T>
constexpr bool is_boxed_tile =
  is_tile<T>::value && (is_tile<T>::layout_enum != SLayout::NoneBox);

template <typename tile_shape> struct is_Nz_layout {
  static constexpr bool value = !tile_shape::isRowMajor &&
                                tile_shape::isBoxedLayout &&
                                tile_shape::isInnerRowMajor;
};

template <typename tile_shape> struct is_Zn_layout {
  static constexpr bool value = tile_shape::isRowMajor &&
                                tile_shape::isBoxedLayout &&
                                tile_shape::isInnerColMajor;
};

template <typename T> concept is_global_data_v = is_global<T>::value;

template <typename T> concept is_tile_data_v = is_tile<T>::value;

template <typename T> concept is_boxed_data_v = is_boxed_tile<T>;

template <typename shape> int index(int i, int j) {
  if constexpr (is_global_data_v<shape>) {
    return i * shape::kRowStride + j * shape::kColStride;
  } else if constexpr (is_tile_data_v<shape>) {
    if constexpr (is_boxed_data_v<shape>) {
      int sub_tile_i = i / shape::InnerRows;
      int sub_tile_j = j / shape::InnerCols;
      int idx_i = i % shape::InnerRows;
      int idx_j = j % shape::InnerCols;
      if constexpr (is_Nz_layout<shape>::value) {
        return sub_tile_j * shape::Rows * shape::InnerCols +
               sub_tile_i * shape::InnerNumel + idx_i * shape::InnerCols +
               idx_j;
      } else if constexpr (is_Zn_layout<shape>::value) {
        return sub_tile_i * shape::Cols * shape::InnerRows +
               sub_tile_j * shape::InnerNumel + idx_i +
               idx_j * shape::InnerRows;
      } else {
        static_assert((is_Nz_layout<shape>::value) ||
                          (is_Zn_layout<shape>::value),
                      "illegal layout");
      }
    } else {
      return i * shape::RowStride + j * shape::ColStride;
    }
  }
}

template <typename tile_shape>
static inline __attribute__((always_inline))
const char* get_layout_str() {
  if constexpr (!tile_shape::isBoxedLayout) {
    if constexpr (tile_shape::isRowMajor)
      return "RowMajor";
    return "ColMajor";
  }
  if constexpr (is_Nz_layout<tile_shape>::value)
    return "NzLayout";
  if constexpr (is_Zn_layout<tile_shape>::value)
    return "ZnLayout";
  return "Other";
}

template <typename tile_shape>
static inline __attribute__((always_inline))
void print_tile_info() {
  std::cout << "Tile Rows Number: " << tile_shape::Rows << std::endl;
  std::cout << "Tile Columns Number: " << tile_shape::Cols << std::endl;
  std::cout << "Tile Active Rows Number: " << tile_shape::ValidRow << std::endl;
  std::cout << "Tile Active Columns Number: " << tile_shape::ValidCol << std::endl;
  if constexpr (tile_shape::isBoxedLayout) {
    std::cout << "Tile Fractal Inner Rows Number: " << tile_shape::InnerRows << std::endl;
    std::cout << "Tile Fractal Inner Columns Number: " << tile_shape::InnerCols << std::endl;
  }
  std::cout << "Tile Size: " << tile_shape::Numel << std::endl;
  std::cout << "Tile Layout: " << get_layout_str<tile_shape>() << std::endl;
  std::cout << "Tile Data Dump: " << std::endl;
}

} // namespace pto

#endif
