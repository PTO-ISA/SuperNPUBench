#ifndef TILE_TENSOR_IMPL
#define TILE_TENSOR_IMPL

#include "common/pto_tile.hpp"
#include "common/tileop_api.hpp"

using namespace pto;

template <Location Loc_, typename Element_, const int Rows_, const int Cols_,
          const BLayout B_Fractal_, const int RowValid_, const int ColValid_,
          const SLayout S_Fractal_, const int S_FractalSize_, const PadValue PadVal_>
template <int RowMask, int ColMask>
inline Tile<Loc_, Element_, Rows_, Cols_, B_Fractal_, RowValid_, ColValid_,
            S_Fractal_, S_FractalSize_, PadVal_>::
  Tile(std::enable_if_t<(RowMask > 0) && (ColMask > 0), DType> s) {
  TEXPANDSCALAR(*this, s);
}

template <Location Loc_, typename Element_, const int Rows_, const int Cols_,
          const BLayout B_Fractal_, const int RowValid_, const int ColValid_,
          const SLayout S_Fractal_, const int S_FractalSize_, const PadValue PadVal_>
template <int RowMask, int ColMask>
inline Tile<Loc_, Element_, Rows_, Cols_, B_Fractal_, RowValid_, ColValid_,
            S_Fractal_, S_FractalSize_, PadVal_>::
  Tile(typename Tile::DType s,
       std::enable_if_t<RowMask == -1 && ColMask == -1, size_t> ValidRow,
       std::enable_if_t<RowMask == -1 && ColMask == -1, size_t> ValidCol) {
  RowMaskInternal = ValidRow;
  ColMaskInternal = ValidCol;
  TEXPANDSCALAR(*this, s);
}

template <Location Loc_, typename Element_, const int Rows_, const int Cols_,
          const BLayout B_Fractal_, const int RowValid_, const int ColValid_,
          const SLayout S_Fractal_, const int S_FractalSize_, const PadValue PadVal_>
template <int RowMask, int ColMask>
inline Tile<Loc_, Element_, Rows_, Cols_, B_Fractal_, RowValid_, ColValid_,
            S_Fractal_, S_FractalSize_, PadVal_>::
  Tile(std::enable_if_t<RowMask == -1 && ColMask == -1, size_t> ValidRow,
       std::enable_if_t<RowMask == -1 && ColMask == -1, size_t> ValidCol) {
  RowMaskInternal = ValidRow;
  ColMaskInternal = ValidCol;
}

template <Location Loc_, typename Element_, const int Rows_, const int Cols_,
          const BLayout B_Fractal_, const int RowValid_, const int ColValid_,
          const SLayout S_Fractal_, const int S_FractalSize_, const PadValue PadVal_>
template <int RowMask, int ColMask>
inline Tile<Loc_, Element_, Rows_, Cols_, B_Fractal_, RowValid_, ColValid_,
            S_Fractal_, S_FractalSize_, PadVal_>::
  Tile(typename Tile::DType s,
       std::enable_if_t<(RowMask == -1) && (ColMask > 0), size_t> ValidRow) {
  RowMaskInternal = ValidRow;
  TEXPANDSCALAR(*this, s);
}

template <Location Loc_, typename Element_, const int Rows_, const int Cols_,
          const BLayout B_Fractal_, const int RowValid_, const int ColValid_,
          const SLayout S_Fractal_, const int S_FractalSize_, const PadValue PadVal_>
template <int RowMask, int ColMask>
inline Tile<Loc_, Element_, Rows_, Cols_, B_Fractal_, RowValid_, ColValid_,
            S_Fractal_, S_FractalSize_, PadVal_>::
  Tile(std::enable_if_t<(RowMask == -1) && (ColMask > 0), size_t> ValidRow) {
  RowMaskInternal = ValidRow;
}

template <Location Loc_, typename Element_, const int Rows_, const int Cols_,
          const BLayout B_Fractal_, const int RowValid_, const int ColValid_,
          const SLayout S_Fractal_, const int S_FractalSize_, const PadValue PadVal_>
template <int RowMask, int ColMask>
inline Tile<Loc_, Element_, Rows_, Cols_, B_Fractal_, RowValid_, ColValid_,
            S_Fractal_, S_FractalSize_, PadVal_>::
  Tile(DType s,
       std::enable_if_t<(RowMask > 0) && (ColMask == -1), size_t> ValidCol) {
  ColMaskInternal = ValidCol;
  TEXPANDSCALAR(*this, s);
}

template <Location Loc_, typename Element_, const int Rows_, const int Cols_,
          const BLayout B_Fractal_, const int RowValid_, const int ColValid_,
          const SLayout S_Fractal_, const int S_FractalSize_, const PadValue PadVal_>
template <int RowMask, int ColMask>
inline Tile<Loc_, Element_, Rows_, Cols_, B_Fractal_, RowValid_, ColValid_,
            S_Fractal_, S_FractalSize_, PadVal_>::
  Tile(std::enable_if_t<(RowMask > 0) && (ColMask == -1), size_t> ValidCol) {
  ColMaskInternal = ValidCol;
}

#endif