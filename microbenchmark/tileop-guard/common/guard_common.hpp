#ifndef TILEOP_GUARD_COMMON_HPP
#define TILEOP_GUARD_COMMON_HPP

// ---------------------------------------------------------------------------
// TileOP-API v0.58 documentation guard demos.
//
// GROUND RULE: every demo is written STRICTLY from the prose/examples in
//   Linx-TileOP-API/docs/tileop-usage/*.md
// We do NOT read the intrinsic source headers to discover usage; the whole
// point is to test whether the documentation alone is sufficient to drive a
// correct call. Canonical v0.58 uppercase names are used (TADD, TMATMUL, ...),
// never the historical PTO-0.57 shims (TSELECT / TEXPANDSCALAR / ...).
//
// A demo "passes" when it compiles to an .elf and gfrun runs it to normal
// exit (no crash / assert). No numeric golden check -- this is a toolchain +
// simulator stability guard, per task scope.
// ---------------------------------------------------------------------------

#include <common/pto_tileop.hpp>
#include <cstdint>
#include "benchmark.h"

using namespace pto;

// ---- data init helpers (host arrays feeding TLOAD) ----
template <typename T>
static inline void gfill_seq(T *p, int n, T base = (T)0) {
    for (int i = 0; i < n; ++i) p[i] = (T)(base + (T)i * (T)0.1);
}
template <typename T>
static inline void gfill_const(T *p, int n, T v) {
    for (int i = 0; i < n; ++i) p[i] = v;
}
template <typename T>
static inline void gzero(T *p, int n) { gfill_const(p, n, (T)0); }
// index / offset fill for gather/scatter style ops
template <typename T>
static inline void gfill_idx(T *p, int n, T base = (T)0) {
    for (int i = 0; i < n; ++i) p[i] = (T)((i * 7) % n) + base;
}

// ---- common tile / global aliases (RowMajor, VEC location) ----
template <typename D, int M, int N>
using gm_t = global_tensor<D, RowMajor<M, N>>;
template <typename D, int M, int N>
using vtile_t = Tile<Location::Vec, D, M, N, BLayout::RowMajor>;
template <typename D, int M, int N>
using iter_t = global_iterator<gm_t<D, M, N>, vtile_t<D, M, N>>;

// ---------------------------------------------------------------------------
// Driver templates. Only the plumbing (TLOAD -> op -> TSTORE, documented in
// tlsu.md) is templated; the actual intrinsic call arrives as a lambda that
// each case writes STRICTLY from the docs. dtypes may differ between src and
// dst (e.g. TCVT, TCMP), so the load/store tiles are typed independently.
// ---------------------------------------------------------------------------

// dst = op(src0) ; same in/out dtype
template <typename D, int M, int N, typename Op>
void g_unary(D *c, const D *a, Op op) {
    iter_t<D, M, N> gA((D *)a), gC(c);
    auto gA0 = gA(0, 0), gC0 = gC(0, 0);
    vtile_t<D, M, N> tA, tC;
    TLOAD(tA, gA0);
    op(tC, tA);
    TSTORE(gC0, tC);
}

// dst = op(src0, src1) ; same dtype
template <typename D, int M, int N, typename Op>
void g_binary(D *c, const D *a, const D *b, Op op) {
    iter_t<D, M, N> gA((D *)a), gB((D *)b), gC(c);
    auto gA0 = gA(0, 0), gB0 = gB(0, 0), gC0 = gC(0, 0);
    vtile_t<D, M, N> tA, tB, tC;
    TLOAD(tA, gA0);
    TLOAD(tB, gB0);
    op(tC, tA, tB);
    TSTORE(gC0, tC);
}

// dst = op(src0, src1, src2) ; same dtype
template <typename D, int M, int N, typename Op>
void g_ternary(D *c, const D *a, const D *b, const D *d, Op op) {
    iter_t<D, M, N> gA((D *)a), gB((D *)b), gD((D *)d), gC(c);
    auto gA0 = gA(0, 0), gB0 = gB(0, 0), gD0 = gD(0, 0), gC0 = gC(0, 0);
    vtile_t<D, M, N> tA, tB, tD, tC;
    TLOAD(tA, gA0);
    TLOAD(tB, gB0);
    TLOAD(tD, gD0);
    op(tC, tA, tB, tD);
    TSTORE(gC0, tC);
}

// dst = op(src0, scalar)
template <typename D, int M, int N, typename Op>
void g_scalar(D *c, const D *a, D s, Op op) {
    iter_t<D, M, N> gA((D *)a), gC(c);
    auto gA0 = gA(0, 0), gC0 = gC(0, 0);
    vtile_t<D, M, N> tA, tC;
    TLOAD(tA, gA0);
    op(tC, tA, s);
    TSTORE(gC0, tC);
}

// dst = op(src0)  with distinct out dtype (TCVT / TCMP-style)
template <typename DIn, typename DOut, int M, int N, typename Op>
void g_unary_cvt(DOut *c, const DIn *a, Op op) {
    iter_t<DIn, M, N> gA((DIn *)a);
    iter_t<DOut, M, N> gC(c);
    auto gA0 = gA(0, 0);
    auto gC0 = gC(0, 0);
    vtile_t<DIn, M, N> tA;
    vtile_t<DOut, M, N> tC;
    TLOAD(tA, gA0);
    op(tC, tA);
    TSTORE(gC0, tC);
}

// dst = op(src0, src1)  with distinct out dtype (TCMP: bool/int out)
template <typename DIn, typename DOut, int M, int N, typename Op>
void g_binary_cvt(DOut *c, const DIn *a, const DIn *b, Op op) {
    iter_t<DIn, M, N> gA((DIn *)a), gB((DIn *)b);
    iter_t<DOut, M, N> gC(c);
    auto gA0 = gA(0, 0);
    auto gB0 = gB(0, 0);
    auto gC0 = gC(0, 0);
    vtile_t<DIn, M, N> tA, tB;
    vtile_t<DOut, M, N> tC;
    TLOAD(tA, gA0);
    TLOAD(tB, gB0);
    op(tC, tA, tB);
    TSTORE(gC0, tC);
}

// dst = op(src0, scalar)  with distinct out dtype (TCMPS)
template <typename DIn, typename DOut, int M, int N, typename Op>
void g_scalar_cvt(DOut *c, const DIn *a, DIn s, Op op) {
    iter_t<DIn, M, N> gA((DIn *)a);
    iter_t<DOut, M, N> gC(c);
    auto gA0 = gA(0, 0);
    auto gC0 = gC(0, 0);
    vtile_t<DIn, M, N> tA;
    vtile_t<DOut, M, N> tC;
    TLOAD(tA, gA0);
    op(tC, tA, s);
    TSTORE(gC0, tC);
}

// ---------------------------------------------------------------------------
// Reduce / expand drivers. Row-reduce collapses columns -> M x 1 output;
// col-reduce collapses rows -> 1 x N output. Output valid dim is 1 on the
// reduced axis (matches the reference tree's bench_reduce ValidCol==1 rule).
// Docs give NO signatures for the TROW*/TCOL* family; shapes here are inferred
// and corrected from compiler / gfrun feedback (recorded in REPORT.md).
// ---------------------------------------------------------------------------

// row reduce: src M x N  ->  dst GENUINE M x 1 (Cols==1).
// ops-20260904 TileOP-API (f8fb894) added a static_assert requiring the
// row-reduce destination to be a real single-column tile
// (ValidCol==1 && Cols==1); the historical "physical M x N + ValidCol=1"
// workaround now fails to compile. The paired model lifted the old fp32
// Cols%8 store constraint for reduce outputs.
template <typename D, int M, int N, typename Op>
void g_rowreduce(D *c, const D *a, Op op) {
    // ops-20260904 TileOP-API (f8fb894) requires the row-reduce dst to be a
    // GENUINE single-column tile (ValidCol==1 && Cols==1); the historical
    // physical M x N + ValidCol=1 workaround now fails to compile. Mirror the
    // release reference kernel (reducemax_rowvec.hpp): static M x N source,
    // genuine M x 1 dst, M x 1 output global.
    using OutTile = Tile<Location::Vec, D, M, 1, BLayout::RowMajor, M, 1>;
    iter_t<D, M, N> gA((D *)a);
    global_iterator<gm_t<D, M, 1>, OutTile> gC(c);
    auto gA0 = gA(0, 0);
    auto gC0 = gC(0, 0);
    vtile_t<D, M, N> tA;
    OutTile tC;
    TLOAD(tA, gA0);
    op(tC, tA);
    TSTORE(gC0, tC);
}

// col reduce: src M x N  ->  dst GENUINE 1 x N (Rows==1).
// ops-20260904 model writes col reductions into a real single-row 1 x N tile
// (release reference reducemax_colvec.hpp: dst Tile<Vec,dtype,1,tN,RowMajor>,
// output global RowMajor<1,gIN>). The historical physical M x N + ValidRow=1
// workaround made the harness read stale row-0 bytes (got=1.0), so use the
// genuine 1 x N geometry.
template <typename D, int M, int N, typename Op>
void g_colreduce(D *c, const D *a, Op op) {
    using OutTile = Tile<Location::Vec, D, 1, N, BLayout::RowMajor>;
    iter_t<D, M, N> gA((D *)a);
    global_iterator<gm_t<D, 1, N>, OutTile> gC(c);
    auto gA0 = gA(0, 0);
    auto gC0 = gC(0, 0);
    vtile_t<D, M, N> tA;
    OutTile tC;
    TLOAD(tA, gA0);
    op(tC, tA);
    TSTORE(gC0, tC);
}

// ---------------------------------------------------------------------------
// Expand-arith drivers. TROWEXPAND*/TCOLEXPAND* fuse a broadcast + binary op.
// gfrun asserts the broadcast source shape:
//   row-expand: srcs[2] must be a validRow x 1 (M x 1) column broadcast source.
//   col-expand: srcs[2] must be a 1 x validCol (1 x N) row broadcast source.
// Docs give NO signature or shape rule for these; inferred from gfrun feedback
// (recorded in REPORT.md as a documentation gap).
// dst = op(src0 M x N, src1_broadcast) -> M x N.
// ---------------------------------------------------------------------------

// row-expand: src1 must be a PHYSICAL M x 1 column broadcast source. gfrun
// asserts physicalCol==1 (IsCompatibleDataTile(srcs[2],...,validRow,1,1,...)),
// so a physical M x N tile with ValidCol=1 is REJECTED here (unlike col-expand,
// which accepts a physical M x N ValidRow=1 source). Asymmetry recorded in REPORT.
template <typename D, int M, int N, typename Op>
void g_rowexpand(D *c, const D *a, const D *b, Op op) {
    using BcastTile = vtile_t<D, M, 1>;
    iter_t<D, M, N> gA((D *)a), gC(c);
    iter_t<D, M, 1> gB((D *)b);
    auto gA0 = gA(0, 0);
    auto gB0 = gB(0, 0);
    auto gC0 = gC(0, 0);
    vtile_t<D, M, N> tA, tC;
    BcastTile tB;
    TLOAD(tA, gA0);
    TLOAD(tB, gB0);
    op(tC, tA, tB);
    TSTORE(gC0, tC);
}

// col-expand: src1 is a 1 x N row broadcast source (physical M x N, ValidRow=1).
template <typename D, int M, int N, typename Op>
void g_colexpand(D *c, const D *a, const D *b, Op op) {
    using BcastTile = Tile<Location::Vec, D, M, N, BLayout::RowMajor, 1, N>;
    iter_t<D, M, N> gA((D *)a), gC(c);
    global_iterator<gm_t<D, M, N>, BcastTile> gB((D *)b);
    auto gA0 = gA(0, 0);
    auto gB0 = gB(0, 0);
    auto gC0 = gC(0, 0);
    vtile_t<D, M, N> tA, tC;
    BcastTile tB;
    TLOAD(tA, gA0);
    TLOAD(tB, gB0);
    op(tC, tA, tB);
    TSTORE(gC0, tC);
}

#endif  // TILEOP_GUARD_COMMON_HPP
