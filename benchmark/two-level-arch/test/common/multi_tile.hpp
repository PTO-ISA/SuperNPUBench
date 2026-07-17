#ifndef MULTI_TILE_HPP
#define MULTI_TILE_HPP

#include <common/pto_tileop.hpp>

namespace pto {
template <int N, class T>
struct MultiTile {
  static constexpr int NumTiles = N;
  using TileType = T;
  TileType Tiles[N];
  MultiTile() = default;
  MultiTile(T t) {
    for (int i = 0; i < NumTiles; ++i)
      Tiles[i] = t;
  }
  MultiTile(int n){
    for(int i=0;i<NumTiles;i++){
      TileType tmp(n);
      Tiles[i] = tmp;
    }
  }
  MultiTile(int n1, int n2){
    for(int i=0;i<NumTiles;i++){
      TileType tmp(n1, n2);
      Tiles[i] = tmp;
    }
  }
};

template <class MT>
concept is_multi_tile = requires (MT mt) {
  MT::NumTiles;
  mt.Tiles[0];
  requires (is_tile_data_v<typename MT::TileType>);
};

template <class Fn>
concept is_gm_iter = requires (Fn fn) {
  fn(0);
};

// template<typename glb_tensor, is_multi_tile mtile>
// struct global_iterator {
//   using DType = typename glb_tensor::DType;
//   using tile = typename mtile::TileType;
//   global_iterator(DType *data) : data_(data) {}

//   auto operator()(int i, int j) {
//     // using tile_layout =
//     //   MatrixLayout<tile::Rows, tile::Cols, glb_tensor::kRowStride, glb_tensor::kColStride>;

//     // using new_gm = global_tensor<DType, tile_layout>;

//     glb_tensor gms[mtile::NumTiles];
//     for (; i < mtile::NumTiles; ++i) {
//       int offset = glb_tensor::isRowMajor
//           ? i * glb_tensor::kCols * tile::Rows + j * tile::Cols
//           : i * tile::Rows + j * glb_tensor::kRows * tile::Cols;
//       gms[i] = glb_tensor(data_ + offset);
//     }
//     return gms;
//   }
// }

}

template <is_multi_tile tile_shape_A, is_multi_tile tile_shape_B,
          is_multi_tile tile_shape_C, is_multi_tile tile_shape_ACC>
void MATMACC(tile_shape_C &dst, tile_shape_A &src0, tile_shape_B &src1, tile_shape_ACC &acc) {
  #pragma clang loop unroll(full)
  for (int i = 0; i < tile_shape_A::NumTiles; ++i) {
    MATMACC(acc.Tiles[i], src0.Tiles[i], src1.Tiles[i]);
    TCVT(dst.Tiles[i], acc.Tiles[i]);
  }
}

template <is_multi_tile tile_shape_A, is_multi_tile tile_shape_B,
          is_multi_tile tile_shape_C>
void MATMUL(tile_shape_C &dst, tile_shape_A &src0, tile_shape_B &src1) {
  #pragma clang loop unroll(full)
  for (int i = 0; i < tile_shape_A::NumTiles; ++i) {
    using tA = typename tile_shape_A::TileType;
    using tB = typename tile_shape_B::TileType;
    using tC = typename tile_shape_C::TileType;
    using tACC = TileAcc<float, tA::ValidRow, tB::ValidCol>;
    tACC acc;
    MATMUL(acc, src0.Tiles[i], src1.Tiles[i]);
    TCVT(dst.Tiles[i], acc);
  }
}

template <is_multi_tile tile_shape_A, is_multi_tile tile_shape_B,
          is_multi_tile tile_shape_C>
void MATMUL_DYN(tile_shape_C &dst, tile_shape_A &src0, tile_shape_B &src1) {
  #pragma clang loop unroll(full)
  for (int i = 0; i < tile_shape_A::NumTiles; ++i) {
    using tA = typename tile_shape_A::TileType;
    using tB = typename tile_shape_B::TileType;
    using tC = typename tile_shape_C::TileType;
    using tACC = TileAcc<float, tA::Rows, tB::Cols, -1, -1>;
    tACC acc(src0.Tiles[i].GetValidRow(), src1.Tiles[i].GetValidCol());
    MATMUL(acc, src0.Tiles[i], src1.Tiles[i]);
    TCVT(dst.Tiles[i], acc);
  }
}

template <is_multi_tile tile_shape_A, is_multi_tile tile_shape_B,
          is_multi_tile tile_shape_C>
void MATMUL_SCALE(tile_shape_C &dst, tile_shape_A &src0, tile_shape_B &src1, const float &scale) {
  #pragma clang loop unroll(full)
  for (int i = 0; i < tile_shape_A::NumTiles; ++i) {
    using tA = typename tile_shape_A::TileType;
    using tB = typename tile_shape_B::TileType;
    using tC = typename tile_shape_C::TileType;
    using tACC = TileAcc<float, tA::ValidRow, tB::ValidCol>;
    tACC acc;
    MATMUL(acc, src0.Tiles[i], src1.Tiles[i]);
    ACCSCALE_NZ2DN(dst.Tiles[i], acc, scale);
  }
}

template <is_multi_tile tO, is_multi_tile tA>
void TCAST(tO &o, tA &a) {
  #pragma clang loop unroll(full)
  for (int i = 0; i < tA::NumTiles; ++i) {
    TCAST(o.Tiles[i], a.Tiles[i]);
  }
}

template <is_multi_tile tile_shape, is_gm_iter itfn>
void TLOAD(tile_shape &dst, itfn it) {
  #pragma clang loop unroll(full)
  for (int i = 0; i < tile_shape::NumTiles; ++i) {
    auto gm = it(i);
    TLOAD(dst.Tiles[i], gm);
  }
}

template <is_multi_tile tile_shape, is_global_data_v gm_shape>
void TLOAD(tile_shape &dst, gm_shape &src) {
#ifdef MULTI_REUSE
  typename tile_shape::TileType t;
  TLOAD(t, src);
  #pragma clang loop unroll(full)
  for (int i = 0; i < tile_shape::NumTiles; ++i) {
    dst.Tiles[i] = t;
  }
#else
  #pragma clang loop unroll(full)
  for (int i = 0; i < tile_shape::NumTiles; ++i) {
    TLOAD(dst.Tiles[i], src);
  }
#endif
}

template <class itfn, is_multi_tile tile_shape>
void TSTORE(itfn it, tile_shape &src) {
  #pragma clang loop unroll(full)
  for (int i = 0; i < tile_shape::NumTiles; ++i) {
    auto gm = it(i);
    TSTORE(gm, src.Tiles[i]);
  }
}

template <is_multi_tile tile_shape_out, is_multi_tile tile_shape_in>
void TCVT(tile_shape_out &dst, tile_shape_in &src) {
  #pragma clang loop unroll(full)
  for (int i = 0; i < tile_shape_out::NumTiles; ++i) {
    TCVT(dst.Tiles[i], src.Tiles[i]);
  }
}

template <is_multi_tile tile_shape_out, is_multi_tile tile_shape_in>
void TCVT_DN2NZ(tile_shape_out &dst, tile_shape_in &src) {
  #pragma clang loop unroll(full)
  for (int i = 0; i < tile_shape_out::NumTiles; ++i) {
    TCVT_DN2NZ(dst.Tiles[i], src.Tiles[i]);
  }
}

template <is_multi_tile tile_shape_out, is_multi_tile tile_shape_in>
void TCVT_DN2NZ_DYN(tile_shape_out &dst, tile_shape_in &src) {
  #pragma clang loop unroll(full)
  for (int i = 0; i < tile_shape_out::NumTiles; ++i) {
    TCVT_DN2NZ_DYN(dst.Tiles[i], src.Tiles[i]);
  }
}

template <is_multi_tile tile_shape_out, is_multi_tile tile_shape_in>
void TEXPANDCOL(tile_shape_out &dst, tile_shape_in &src) {
  #pragma clang loop unroll(full)
  for (int i = 0; i < tile_shape_out::NumTiles; ++i) {
    TEXPANDCOL(dst.Tiles[i], src.Tiles[i]);
  }
}

template <is_multi_tile tile_shape>
void TEXPANDSCALAR(tile_shape &dst, typename tile_shape::TileType::DType s) {
#ifdef MULTI_REUSE
  typename tile_shape::TileType t;
  TEXPANDSCALAR(t, s);
  #pragma clang loop unroll(full)
  for (int i = 0; i < tile_shape::NumTiles; ++i) {
    dst.Tiles[i] = t;
  }
#else
  #pragma clang loop unroll(full)
  for (int i = 0; i < tile_shape::NumTiles; ++i) {
    TEXPANDSCALAR(dst.Tiles[i], s);
  }
#endif
}

template <is_multi_tile tile_shape>
void TMUL(tile_shape &dst, tile_shape &lhs, tile_shape &rhs) {
  #pragma clang loop unroll(full)
  for (int i = 0; i < tile_shape::NumTiles; ++i) {
    TMUL(dst.Tiles[i], lhs.Tiles[i], rhs.Tiles[i]);
  }
}

template <is_multi_tile tile_shape>
void TADD(tile_shape &dst, tile_shape &lhs, tile_shape &rhs) {
  #pragma clang loop unroll(full)
  for (int i = 0; i < tile_shape::NumTiles; ++i) {
    TADD(dst.Tiles[i], lhs.Tiles[i], rhs.Tiles[i]);
  }
}

template <is_multi_tile tile_shape>
void TRECIP(tile_shape &dst, tile_shape &src) {
  #pragma clang loop unroll(full)
  for (int i = 0; i < tile_shape::NumTiles; ++i) {
    TRECIP(dst.Tiles[i], src.Tiles[i]);
  }
}

#endif

