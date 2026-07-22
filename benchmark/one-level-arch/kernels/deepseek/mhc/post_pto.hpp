// =============================================================================
// post_pto.hpp — MHC post mix fwd/bwd
// =============================================================================
//
// Source: TileKernels/tile_kernels/mhc/post_kernel.py @ 36d9e45.
// Fwd: x[o,h] = c[o] * d[h] + sum_i a[i,o] * b[i,h].
// Bwd computes source-level gradients for a/b/c/d using tile reductions.
// =============================================================================
#ifndef SUPERNPU_MHC_POST_PTO_HPP
#define SUPERNPU_MHC_POST_PTO_HPP

#include <common/pto_tileop.hpp>
#include <cstddef>
#include <cstdint>

namespace supernpu::tile_isa {

template <int NumTokens, int Mhc, int Hidden, int TileH = 64>
void mhc_post_fwd(float *a,
                  __bf16 *b,
                  float *c,
                  __bf16 *d,
                  __bf16 *x) {
    static_assert(Mhc <= 8 && TileH >= 64 && TileH % 8 == 0 && Hidden % TileH == 0, "valid tile shape required");
    constexpr int kTiles = Hidden / TileH;

    using namespace pto;
    using gm_a = global_tensor<float, RowMajor<NumTokens * Mhc * Mhc, 1>>;
    using gm_c = global_tensor<float, RowMajor<NumTokens * Mhc, 1>>;
    using gm_b = global_tensor<__bf16, RowMajor<NumTokens * Mhc, Hidden>>;
    using gm_d = global_tensor<__bf16, RowMajor<NumTokens, Hidden>>;
    using tile_s = Tile<Location::Vec, float, 4, 8, BLayout::RowMajor, 1, 1>;
    using tile_b = Tile<Location::Vec, __bf16, 1, TileH, BLayout::RowMajor>;
    using tile_f = Tile<Location::Vec, float, 1, TileH, BLayout::RowMajor>;
    static_assert(tile_s::Rows * tile_s::Cols * static_cast<int>(sizeof(float)) >= 128);
    static_assert(tile_b::Rows * tile_b::Cols * static_cast<int>(sizeof(__bf16)) >= 128);
    static_assert(tile_f::Rows * tile_f::Cols * static_cast<int>(sizeof(float)) >= 128);

    using it_a = global_iterator<gm_a, tile_s>;
    using it_c = global_iterator<gm_c, tile_s>;
    using it_b = global_iterator<gm_b, tile_b>;
    using it_d = global_iterator<gm_d, tile_b>;
    it_a a_iter(a);
    it_c c_iter(c);
    it_b b_iter(b), x_iter(x);
    it_d d_iter(d);

    for (int n = 0; n < NumTokens; ++n) {
        for (int ohead = 0; ohead < Mhc; ++ohead) {
            auto gc = c_iter(n * Mhc + ohead, 0);
            tile_s cscalar;
            TLOAD(cscalar, gc);
            for (int th = 0; th < kTiles; ++th) {
                auto gd = d_iter(n, th);
                auto gx = x_iter(n * Mhc + ohead, th);
                tile_b db, xb;
                tile_f df, out, full;
                TLOAD(db, gd);
                TCVT(df, db);
                TROWEXPAND(full, cscalar);
                TMUL(out, df, full);
                for (int ihead = 0; ihead < Mhc; ++ihead) {
                    auto ga = a_iter(n * Mhc * Mhc + ihead * Mhc + ohead, 0);
                    auto gb = b_iter(n * Mhc + ihead, th);
                    tile_s ascalar;
                    tile_b bb;
                    tile_f bf, prod;
                    TLOAD(ascalar, ga);
                    TLOAD(bb, gb);
                    TCVT(bf, bb);
                    TROWEXPAND(full, ascalar);
                    TMUL(prod, bf, full);
                    TADD(out, out, prod);
                }
                TCVT(xb, out);
                TSTORE(gx, xb);
            }
        }
    }
}

template <int NumTokens, int Mhc, int Hidden, int TileH = 64>
void mhc_post_bwd(__bf16 *dx,
                  float *a,
                  __bf16 *b,
                  float *c,
                  __bf16 *d,
                  float *da,
                  __bf16 *db,
                  float *dc,
                  __bf16 *dd) {
    static_assert(Mhc <= 8 && TileH >= 64 && TileH % 8 == 0 && Hidden % TileH == 0, "valid tile shape required");
    constexpr int kTiles = Hidden / TileH;

    using namespace pto;
    using gm_a = global_tensor<float, RowMajor<NumTokens * Mhc * Mhc, 1>>;
    using gm_c = global_tensor<float, RowMajor<NumTokens * Mhc, 1>>;
    using gm_b = global_tensor<__bf16, RowMajor<NumTokens * Mhc, Hidden>>;
    using gm_d = global_tensor<__bf16, RowMajor<NumTokens, Hidden>>;
    using tile_s = Tile<Location::Vec, float, 4, 8, BLayout::RowMajor, 1, 1>;
    using tile_b = Tile<Location::Vec, __bf16, 1, TileH, BLayout::RowMajor>;
    using tile_f = Tile<Location::Vec, float, 1, TileH, BLayout::RowMajor>;
    using it_a = global_iterator<gm_a, tile_s>;
    using it_c = global_iterator<gm_c, tile_s>;
    using it_b = global_iterator<gm_b, tile_b>;
    using it_d = global_iterator<gm_d, tile_b>;
    it_a a_iter(a), da_iter(da);
    it_c c_iter(c), dc_iter(dc);
    it_b dx_iter(dx), b_iter(b), db_iter(db);
    it_d d_iter(d), dd_iter(dd);

    for (int n = 0; n < NumTokens; ++n) {
        for (int ihead = 0; ihead < Mhc; ++ihead) {
            for (int th = 0; th < kTiles; ++th) {
                tile_f db_acc;
                TEXPANDS(db_acc, 0.0f);
                for (int ohead = 0; ohead < Mhc; ++ohead) {
                    auto ga = a_iter(n * Mhc * Mhc + ihead * Mhc + ohead, 0);
                    auto gdx = dx_iter(n * Mhc + ohead, th);
                    tile_s ascalar;
                    tile_b dxb;
                    tile_f dxf, full, prod;
                    TLOAD(ascalar, ga);
                    TLOAD(dxb, gdx);
                    TCVT(dxf, dxb);
                    TROWEXPAND(full, ascalar);
                    TMUL(prod, dxf, full);
                    TADD(db_acc, db_acc, prod);
                }
                auto gdb = db_iter(n * Mhc + ihead, th);
                tile_b dbb;
                TCVT(dbb, db_acc);
                TSTORE(gdb, dbb);
            }
        }

        for (int ohead = 0; ohead < Mhc; ++ohead) {
            tile_s dc_acc;
            TEXPANDS(dc_acc, 0.0f);
            for (int th = 0; th < kTiles; ++th) {
                auto gd = d_iter(n, th);
                auto gdx = dx_iter(n * Mhc + ohead, th);
                tile_b dbf, dxb;
                tile_f df, dxf, prod;
                tile_s part;
                TLOAD(dbf, gd);
                TLOAD(dxb, gdx);
                TCVT(df, dbf);
                TCVT(dxf, dxb);
                TMUL(prod, df, dxf);
                TROWSUM(part, prod);
                TADD(dc_acc, dc_acc, part);
            }
            auto gdc = dc_iter(n * Mhc + ohead, 0);
            TSTORE(gdc, dc_acc);
        }

        for (int th = 0; th < kTiles; ++th) {
            tile_f dd_acc;
            TEXPANDS(dd_acc, 0.0f);
            for (int ohead = 0; ohead < Mhc; ++ohead) {
                auto gc = c_iter(n * Mhc + ohead, 0);
                auto gdx = dx_iter(n * Mhc + ohead, th);
                tile_s cscalar;
                tile_b dxb;
                tile_f dxf, full, prod;
                TLOAD(cscalar, gc);
                TLOAD(dxb, gdx);
                TCVT(dxf, dxb);
                TROWEXPAND(full, cscalar);
                TMUL(prod, dxf, full);
                TADD(dd_acc, dd_acc, prod);
            }
            auto gdd = dd_iter(n, th);
            tile_b ddb;
            TCVT(ddb, dd_acc);
            TSTORE(gdd, ddb);
        }

        for (int ihead = 0; ihead < Mhc; ++ihead) {
            for (int ohead = 0; ohead < Mhc; ++ohead) {
                tile_s da_acc;
                TEXPANDS(da_acc, 0.0f);
                for (int th = 0; th < kTiles; ++th) {
                    auto gb = b_iter(n * Mhc + ihead, th);
                    auto gdx = dx_iter(n * Mhc + ohead, th);
                    tile_b bb, dxb;
                    tile_f bf, dxf, prod;
                    tile_s part;
                    TLOAD(bb, gb);
                    TLOAD(dxb, gdx);
                    TCVT(bf, bb);
                    TCVT(dxf, dxb);
                    TMUL(prod, bf, dxf);
                    TROWSUM(part, prod);
                    TADD(da_acc, da_acc, part);
                }
                auto gda = da_iter(n * Mhc * Mhc + ihead * Mhc + ohead, 0);
                TSTORE(gda, da_acc);
            }
        }
    }
}

} // namespace supernpu::tile_isa
#endif
