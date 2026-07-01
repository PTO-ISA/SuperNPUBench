#ifndef SUPERNPUBENCH_PTOISA_GELU_HPP
#define SUPERNPUBENCH_PTOISA_GELU_HPP

#include <common/pto_tileop.hpp>

#include <cstddef>

using namespace pto;

template <typename TileT>
inline void gelu_tile(TileT &dst, TileT &src) {
  TileT t;
  TileT t2;
  TileT p;
  TileT tmp;
  TileT denom;
  TileT recip;

  TMAXS(t, src, -5.75f);
  TMINS(t, t, 5.75f);
  TMUL(t2, t, t);

  TMULS(p, t2, -3.5123395303315874e-09f);
  TADDS(p, p, 2.6452661927578447e-07f);
  TMUL(p, p, t2);
  TADDS(p, p, -7.9294877650681883e-06f);
  TMUL(p, p, t2);
  TADDS(p, p, 1.1061238183174282e-04f);
  TMUL(p, p, t2);
  TADDS(p, p, 6.5189960878342390e-05f);
  TMUL(p, p, t2);
  TADDS(p, p, -7.2666168212890625e-02f);
  TMUL(p, p, t2);
  TADDS(p, p, -1.5957698822021484e+00f);

  TMUL(tmp, t, p);
  TEXP(tmp, tmp);
  TADDS(denom, tmp, 1.0f);
  TRECIP(recip, denom);
  TMUL(dst, src, recip);
}

template <typename dtype, int gM, int tM>
void gelu(dtype *in_ptr, dtype *out_ptr, bool = false) {
  constexpr int kFullTiles = gM / tM;
  constexpr int kTail = gM % tM;

  using Global = global_tensor<dtype, RowMajor<1, gM>>;
  using TileT = Tile<Location::Vec, dtype, 1, tM, BLayout::RowMajor>;
  using Iterator = global_iterator<Global, TileT>;

  Iterator input_iter(in_ptr);
  Iterator output_iter(out_ptr);

  for (int tile = 0; tile < kFullTiles; ++tile) {
    TileT input_tile;
    TileT output_tile;
    TLOAD(input_tile, input_iter(0, tile));
    gelu_tile(output_tile, input_tile);
    TSTORE(output_iter(0, tile), output_tile);
  }

  if constexpr (kTail != 0) {
    using TailTile =
        Tile<Location::Vec, dtype, 1, tM, BLayout::RowMajor, 1, kTail>;
    using TailIterator = global_iterator<Global, TailTile>;
    TailIterator tail_input_iter(in_ptr);
    TailIterator tail_output_iter(out_ptr);
    TailTile input_tile;
    TailTile output_tile;
    TLOAD(input_tile, tail_input_iter(0, kFullTiles));
    gelu_tile(output_tile, input_tile);
    TSTORE(tail_output_iter(0, kFullTiles), output_tile);
  }
}

#endif // SUPERNPUBENCH_PTOISA_GELU_HPP
