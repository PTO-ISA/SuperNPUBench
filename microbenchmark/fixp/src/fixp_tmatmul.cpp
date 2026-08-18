// TMATMUL fixp microbenchmark.
//
// One source, many binaries: each build selects exactly one fixp mode via a
// -D define (see ../Makefile / ../compile.all). Every binary performs a single
// TMATMUL(C = A*B) with the corresponding B.FPATR configuration and auxiliary
// operand stream, so the ELF/.diss can be checked against the PTO-ISA v0.58
// fixp matrix contract.
//
// New-style interface: TMATMUL(dst, a, b, fixp::Options) with no .FIXP suffix.
//   - parameter-free conversion:  fixp::keep_acc() / fixp::f16() / fixp::bf16()
//   - plain ReLU:                 .relu()
//   - scalar quant descriptor:    fixp::scalar<Mode>(desc), fixp::s8(desc)
//   - vector quant tile:          fixp::vector<Mode>(tile), fixp::s8(tile)
//   - LReLU / PReLU:              .lrelu(desc) / .prelu(tile)
//   - max reductions:             .row_max(out) / .row_max(in,out)
//                                 .group_max<GroupN>(out) / .max_abs()
#include <common/pto_tileop.hpp>
#include <cstdint>

#include "benchmark.h"

#ifndef TM
#define TM 32
#endif

#ifndef TN
#define TN 32
#endif

#ifndef TK
#define TK 32
#endif

using namespace pto;

// B.IOR SrcReg0 scalar quant descriptor layout:
//   FP19 scale in [31:13]; signed offset (width S4/S8/S16) starting at bit 37.
constexpr uint64_t mk_desc(uint32_t fp19_scale, uint64_t offset,
                           uint32_t off_bits) {
  uint64_t d = (static_cast<uint64_t>(fp19_scale) & 0x7ffffull) << 13;
  if (off_bits != 0)
    d |= (offset & ((1ull << off_bits) - 1ull)) << 37;
  return d;
}

// Vector quant / PReLU parameter Tile: physical 2 x N uint64 (512 B),
// valid 1 x N.
template <int N>
using par_tile_t = Tile<Location::Vec, uint64_t, 2, N, BLayout::RowMajor, 1, N>;

// RowMax Tile: physical M x 8 FP32 (1 KB), valid M x 1.
template <int M>
using row_max_tile_t =
    Tile<Location::Vec, float, M, 8, BLayout::RowMajor, M, 1>;

// GroupMax Tile: physical M x 8 FP32 (1 KB), valid M x ceil(N/GroupN).
template <int M, int GCols>
using group_max_tile_t =
    Tile<Location::Vec, float, M, 8, BLayout::RowMajor, M, GCols>;

// --- Auxiliary operand tiles for the operation-family modes ---------------
// (Only PostProcess parameter tiles carry the >=512 B minimum; the math
// operands below are ordinary data tiles. 1-row tiles are still padded to
// >=512 B physical per the tile-register contract, like par_tile_t above.)

// Bias tile (TMATMUL_BIAS / TGEMV_BIAS): valid 1 x N float, padded to >=512 B.
template <int N>
using bias_tile_t = Tile<Location::Vec, float, 4, N, BLayout::RowMajor, 1, N>;

// FP32 accumulator C tile for ACC ops: full M x N (>=512 B at 32x32).
template <int M, int N>
using acc_tile_t = Tile<Location::Vec, float, M, N>;

// MX scale tiles (per-element, shape matching A / B): ScaleA = M x K,
// ScaleB = K x N. Dtype __half (pairs with the FP16 operand).
template <int M, int K>
using scale_a_tile_t = Tile<Location::Vec, __half, M, K>;
template <int K, int N>
using scale_b_tile_t = Tile<Location::Vec, __half, K, N>;
// TGEMV_MX scale-vec: valid 1 x K __half, padded to >=512 B.
template <int K>
using scale_vec_tile_t =
    Tile<Location::Vec, __half, 16, K, BLayout::RowMajor, 1, K>;

template <typename SrcT, typename DstT, typename OptsMaker>
__attribute__((noinline)) void run_single(SrcT *a_ptr, SrcT *b_ptr,
                                          DstT *d_ptr, OptsMaker maker) {
  constexpr int kM = TM, kN = TN, kK = TK;
  using gm_a = global_tensor<SrcT, RowMajor<kM, kK>>;
  using gm_b = global_tensor<SrcT, RowMajor<kK, kN>>;
  using gm_c = global_tensor<DstT, RowMajor<kM, kN>>;
  using tile_a = TileLeft<SrcT, kM, kK>;
  using tile_b = TileRight<SrcT, kK, kN>;
  using tile_d = Tile<Location::Vec, DstT, kM, kN>;

  gm_a gA(a_ptr);
  gm_b gB(b_ptr);
  gm_c gC(d_ptr);
  tile_a tA;
  tile_b tB;
  tile_d tD;

  BENCHSTART;
  TLOAD(tA, gA);
  TLOAD(tB, gB);
  auto options = maker();
  TMATMUL(tD, tA, tB, options);
  TSTORE(gC, tD);
  BENCHEND;
}

// TMATMUL-family driver: TLOADs A,B; declares tD; the lambda kernel(tD,tA,tB)
// calls the specific op (TMATMUL_BIAS / TMATMUL_ACC / TMATMUL_MX / ...) with any
// aux tiles (C / bias / scales) captured from the enclosing scope. A/B/C are
// full M x ... tiles (>=512 B at 32x32), so no padding is needed here.
template <typename SrcT, typename DstT, typename Kernel>
__attribute__((noinline)) void run_matmul(SrcT *a_ptr, SrcT *b_ptr,
                                         DstT *d_ptr, Kernel kernel) {
  constexpr int kM = TM, kN = TN, kK = TK;
  using gm_a = global_tensor<SrcT, RowMajor<kM, kK>>;
  using gm_b = global_tensor<SrcT, RowMajor<kK, kN>>;
  using gm_d = global_tensor<DstT, RowMajor<kM, kN>>;
  using tile_a = TileLeft<SrcT, kM, kK>;
  using tile_b = TileRight<SrcT, kK, kN>;
  using tile_d = Tile<Location::Vec, DstT, kM, kN>;

  gm_a gA(a_ptr);
  gm_b gB(b_ptr);
  gm_d gD(d_ptr);
  tile_a tA;
  tile_b tB;
  tile_d tD;

  BENCHSTART;
  TLOAD(tA, gA);
  TLOAD(tB, gB);
  kernel(tD, tA, tB);
  TSTORE(gD, tD);
  BENCHEND;
}

// TGEMV-family driver (M=1): vec=Left(1xK), mtx=Right(KxN), d=Vec(1xN).
// vec and d are padded to >=512 B physical (valid 1xK / 1xN) per the
// tile-register contract; mtx is KxN (already >=512 B at 32x32). The lambda
// kernel(tD,tMtx,tVec) calls the specific TGEMV op with captured aux tiles.
template <typename SrcT, typename DstT, typename Kernel>
__attribute__((noinline)) void run_gemv(SrcT *vec_ptr, SrcT *mtx_ptr,
                                        DstT *d_ptr, Kernel kernel) {
  constexpr int kK = TK, kN = TN;
  using gm_vec = global_tensor<SrcT, RowMajor<1, kK>>;
  using gm_mtx = global_tensor<SrcT, RowMajor<kK, kN>>;
  using gm_d = global_tensor<DstT, RowMajor<1, kN>>;
  using tile_vec = Tile<Location::Left, SrcT, 16, kK, BLayout::RowMajor, 1, kK>;
  using tile_mtx = TileRight<SrcT, kK, kN>;
  using tile_d = Tile<Location::Vec, DstT, 16, kN, BLayout::RowMajor, 1, kN>;

  gm_vec gV(vec_ptr);
  gm_mtx gMx(mtx_ptr);
  gm_d gD(d_ptr);
  tile_vec tVec;
  tile_mtx tMtx;
  tile_d tD;

  BENCHSTART;
  TLOAD(tVec, gV);
  TLOAD(tMtx, gMx);
  kernel(tD, tMtx, tVec);
  TSTORE(gD, tD);
  BENCHEND;
}

template <typename SrcT, typename DstT>
struct buf_t {
  static constexpr size_t kAlign = 4096;
  static constexpr size_t kAlignMask = ~(kAlign - 1);
  alignas(16) uint8_t a_raw[TM * TK * sizeof(SrcT) + 2 * kAlign];
  alignas(16) uint8_t b_raw[TK * TN * sizeof(SrcT) + 2 * kAlign];
  alignas(16) uint8_t d_raw[TM * TN * sizeof(DstT) + 2 * kAlign];
  SrcT *a;
  SrcT *b;
  DstT *d;
  buf_t() {
    a = (SrcT *)(((uint64_t)&a_raw[0] & kAlignMask) + kAlign);
    b = (SrcT *)(((uint64_t)&b_raw[0] & kAlignMask) + kAlign);
    d = (DstT *)(((uint64_t)&d_raw[0] & kAlignMask) + kAlign);
  }
};

int main() {
  // One mode per binary, selected by -D<MODE> from the Makefile.

// --- parameter-free conversion -------------------------------------------
#if defined(KEEP_ACC)
  buf_t<__half, float> buf;
  run_single<__half, float>(buf.a, buf.b, buf.d,
                             [] { return fixp::keep_acc(); });
#elif defined(KEEP_ACC_RELU)
  buf_t<__half, float> buf;
  run_single<__half, float>(buf.a, buf.b, buf.d,
                             [] { return fixp::keep_acc().relu(); });
#elif defined(F16)
  buf_t<__half, __half> buf;
  run_single<__half, __half>(buf.a, buf.b, buf.d,
                              [] { return fixp::f16(); });
#elif defined(F16_RELU)
  buf_t<__half, __half> buf;
  run_single<__half, __half>(buf.a, buf.b, buf.d,
                              [] { return fixp::f16().relu(); });
#elif defined(BF16)
  buf_t<__half, __bf16> buf;
  run_single<__half, __bf16>(buf.a, buf.b, buf.d,
                              [] { return fixp::bf16(); });
#elif defined(BF16_RELU)
  buf_t<__half, __bf16> buf;
  run_single<__half, __bf16>(buf.a, buf.b, buf.d,
                              [] { return fixp::bf16().relu(); });

// --- scalar quant descriptor modes ---------------------------------------
#elif defined(S_REQS8)
  buf_t<__half, int8_t> buf;
  run_single<__half, int8_t>(buf.a, buf.b, buf.d, [] {
    return fixp::scalar<FixpPreQuantMode::REQS8Pre>(mk_desc(1, 0, 9));
  });
#elif defined(S_DEQF16)
  buf_t<__half, __half> buf;
  run_single<__half, __half>(buf.a, buf.b, buf.d, [] {
    return fixp::scalar<FixpPreQuantMode::DEQF16>(mk_desc(1, 0, 0));
  });
#elif defined(S_SHIFTS16)
  buf_t<__half, int16_t> buf;
  run_single<__half, int16_t>(buf.a, buf.b, buf.d, [] {
    return fixp::scalar<FixpPreQuantMode::SHIFTS322S16>(mk_desc(1, 0, 17));
  });
#elif defined(S_QF_S4)
  buf_t<__half, __int4x2> buf;
  run_single<__half, __int4x2>(buf.a, buf.b, buf.d, [] {
    return fixp::scalar<FixpPreQuantMode::QF322S4Pre>(mk_desc(1, 0, 5));
  });
#elif defined(S_QF_S16)
  buf_t<__half, int16_t> buf;
  run_single<__half, int16_t>(buf.a, buf.b, buf.d, [] {
    return fixp::scalar<FixpPreQuantMode::QF322S16Pre>(mk_desc(1, 0, 17));
  });
#elif defined(S_QF_S8)
  buf_t<__half, int8_t> buf;
  run_single<__half, int8_t>(buf.a, buf.b, buf.d, [] {
    return fixp::s8(mk_desc(1, 0, 9));
  });
#elif defined(S_QF_HIF8)
  buf_t<__half, __hif8> buf;
  run_single<__half, __hif8>(buf.a, buf.b, buf.d, [] {
    return fixp::scalar<FixpPreQuantMode::QF322HIF8Pre>(mk_desc(1, 0, 9));
  });
#elif defined(S_QF_FP8)
  buf_t<__half, __fp8_e4m3> buf;
  run_single<__half, __fp8_e4m3>(buf.a, buf.b, buf.d, [] {
    return fixp::scalar<FixpPreQuantMode::QF322FP8Pre>(mk_desc(1, 0, 9));
  });
#elif defined(S_QF_F32)
  buf_t<__half, float> buf;
  run_single<__half, float>(buf.a, buf.b, buf.d, [] {
    return fixp::scalar<FixpPreQuantMode::QF322F32Pre>(mk_desc(1, 0, 9));
  });
#elif defined(S_QF_F16)
  buf_t<__half, __half> buf;
  run_single<__half, __half>(buf.a, buf.b, buf.d, [] {
    return fixp::scalar<FixpPreQuantMode::QF322F16Pre>(mk_desc(1, 0, 9));
  });
#elif defined(S_QF_BF16)
  buf_t<__half, __bf16> buf;
  run_single<__half, __bf16>(buf.a, buf.b, buf.d, [] {
    return fixp::scalar<FixpPreQuantMode::QF322BF16Pre>(mk_desc(1, 0, 9));
  });
#elif defined(S_QS_BF16)
  buf_t<__half, __bf16> buf;
  run_single<__half, __bf16>(buf.a, buf.b, buf.d, [] {
    return fixp::scalar<FixpPreQuantMode::QS322BF16Pre>(mk_desc(1, 0, 9));
  });

// --- vector quant parameter tile modes ------------------------------------
#elif defined(V_REQS8)
  buf_t<__half, int8_t> buf;
  par_tile_t<TN> quant;
  run_single<__half, int8_t>(buf.a, buf.b, buf.d, [&] {
    return fixp::vector<FixpPreQuantMode::VREQS8Pre>(quant);
  });
#elif defined(V_DEQF16)
  buf_t<__half, __half> buf;
  par_tile_t<TN> quant;
  run_single<__half, __half>(buf.a, buf.b, buf.d, [&] {
    return fixp::vector<FixpPreQuantMode::VDEQF16>(quant);
  });
#elif defined(V_SHIFTS16)
  buf_t<__half, int16_t> buf;
  par_tile_t<TN> quant;
  run_single<__half, int16_t>(buf.a, buf.b, buf.d, [&] {
    return fixp::vector<FixpPreQuantMode::VSHIFTS322S16>(quant);
  });
#elif defined(V_QF_S4)
  buf_t<__half, __int4x2> buf;
  par_tile_t<TN> quant;
  run_single<__half, __int4x2>(buf.a, buf.b, buf.d, [&] {
    return fixp::vector<FixpPreQuantMode::VQF322S4Pre>(quant);
  });
#elif defined(V_QF_S16)
  buf_t<__half, int16_t> buf;
  par_tile_t<TN> quant;
  run_single<__half, int16_t>(buf.a, buf.b, buf.d, [&] {
    return fixp::vector<FixpPreQuantMode::VQF322S16Pre>(quant);
  });
#elif defined(V_QF_S8)
  buf_t<__half, int8_t> buf;
  par_tile_t<TN> quant;
  run_single<__half, int8_t>(buf.a, buf.b, buf.d,
                              [&] { return fixp::s8(quant); });
#elif defined(V_QF_HIF8)
  buf_t<__half, __hif8> buf;
  par_tile_t<TN> quant;
  run_single<__half, __hif8>(buf.a, buf.b, buf.d, [&] {
    return fixp::vector<FixpPreQuantMode::VQF322HIF8Pre>(quant);
  });
#elif defined(V_QF_F16)
  buf_t<__half, __half> buf;
  par_tile_t<TN> quant;
  run_single<__half, __half>(buf.a, buf.b, buf.d, [&] {
    return fixp::vector<FixpPreQuantMode::VQF322F16Pre>(quant);
  });
#elif defined(V_QF_BF16)
  buf_t<__half, __bf16> buf;
  par_tile_t<TN> quant;
  run_single<__half, __bf16>(buf.a, buf.b, buf.d, [&] {
    return fixp::vector<FixpPreQuantMode::VQF322BF16Pre>(quant);
  });
#elif defined(V_QF_FP8)
  buf_t<__half, __fp8_e4m3> buf;
  par_tile_t<TN> quant;
  run_single<__half, __fp8_e4m3>(buf.a, buf.b, buf.d, [&] {
    return fixp::vector<FixpPreQuantMode::VQF322FP8Pre>(quant);
  });
#elif defined(V_QF_F32)
  buf_t<__half, float> buf;
  par_tile_t<TN> quant;
  run_single<__half, float>(buf.a, buf.b, buf.d, [&] {
    return fixp::vector<FixpPreQuantMode::VQF322F32Pre>(quant);
  });
#elif defined(V_QS_BF16)
  buf_t<__half, __bf16> buf;
  par_tile_t<TN> quant;
  run_single<__half, __bf16>(buf.a, buf.b, buf.d, [&] {
    return fixp::vector<FixpPreQuantMode::VQS322BF16Pre>(quant);
  });

// --- ReLU / quant combinations ---------------------------------------------
#elif defined(S8_RELU)
  buf_t<__half, int8_t> buf;
  run_single<__half, int8_t>(buf.a, buf.b, buf.d, [] {
    return fixp::s8(mk_desc(1, 0, 9)).relu();
  });
#elif defined(S8_LRELU)
  buf_t<__half, int8_t> buf;
  run_single<__half, int8_t>(buf.a, buf.b, buf.d, [] {
    return fixp::s8(mk_desc(1, 0, 9)).lrelu(1);
  });
#elif defined(V_S8_RELU)
  buf_t<__half, int8_t> buf;
  par_tile_t<TN> quant;
  run_single<__half, int8_t>(buf.a, buf.b, buf.d, [&] {
    return fixp::s8(quant).relu();
  });
#elif defined(F16_PRELU)
  buf_t<__half, __half> buf;
  par_tile_t<TN> prelu;
  run_single<__half, __half>(buf.a, buf.b, buf.d, [&] {
    return fixp::f16().prelu(prelu);
  });
#elif defined(S8_PRELU)
  buf_t<__half, int8_t> buf;
  par_tile_t<TN> prelu;
  run_single<__half, int8_t>(buf.a, buf.b, buf.d, [&] {
    return fixp::s8(mk_desc(1, 0, 9)).prelu(prelu);
  });

// --- RowMax / GroupMax / MaxAbs --------------------------------------------
#elif defined(ROWMAX)
  buf_t<__half, float> buf;
  row_max_tile_t<TM> row_out;
  run_single<__half, float>(buf.a, buf.b, buf.d, [&] {
    return fixp::keep_acc().row_max(row_out);
  });
#elif defined(ROWMAX_INIT)
  buf_t<__half, float> buf;
  row_max_tile_t<TM> row_in;
  row_max_tile_t<TM> row_out;
  run_single<__half, float>(buf.a, buf.b, buf.d, [&] {
    return fixp::keep_acc().row_max(row_in, row_out);
  });
#elif defined(GROUPMAX_8)
  buf_t<__half, float> buf;
  group_max_tile_t<TM, (TN + 7) / 8> group_out;
  run_single<__half, float>(buf.a, buf.b, buf.d, [&] {
    return fixp::keep_acc().group_max<8>(group_out);
  });
#elif defined(GROUPMAX_16)
  buf_t<__half, float> buf;
  group_max_tile_t<TM, (TN + 15) / 16> group_out;
  run_single<__half, float>(buf.a, buf.b, buf.d, [&] {
    return fixp::keep_acc().group_max<16>(group_out);
  });
#elif defined(GROUPMAX_128)
  buf_t<__half, float> buf;
  group_max_tile_t<TM, (TN + 127) / 128> group_out;
  run_single<__half, float>(buf.a, buf.b, buf.d, [&] {
    return fixp::keep_acc().group_max<128>(group_out);
  });
#elif defined(ROWGROUP_MAXABS)
  buf_t<__half, float> buf;
  row_max_tile_t<TM> row_in;
  row_max_tile_t<TM> row_out;
  group_max_tile_t<TM, (TN + 7) / 8> group_out;
  run_single<__half, float>(buf.a, buf.b, buf.d, [&] {
    return fixp::keep_acc()
        .row_max(row_in, row_out)
        .group_max<8>(group_out)
        .max_abs();
  });
#elif defined(F16_GROUPMAX)
  buf_t<__half, __half> buf;
  group_max_tile_t<TM, (TN + 15) / 16> group_out;
  run_single<__half, __half>(buf.a, buf.b, buf.d, [&] {
    return fixp::f16().group_max<16>(group_out);
  });
#elif defined(S8_ROWMAX)
  buf_t<__half, int8_t> buf;
  row_max_tile_t<TM> row_out;
  run_single<__half, int8_t>(buf.a, buf.b, buf.d, [&] {
    return fixp::s8(mk_desc(1, 0, 9)).row_max(row_out);
  });

// === operation-family coverage (param-free keep_acc) =====================
// Verifies each op's BSTART.CUBE mnemonic + math operand stream (C / Bias /
// Scale tiles). Mode macros are short labels (BIAS/ACC/MX/GEMV/...) so they
// never collide with the op function names (TMATMUL_BIAS / TGEMV / ...).
#elif defined(BIAS)                       // D = A*B + Bias
  buf_t<__half, float> buf;
  bias_tile_t<TN> bias;
  run_matmul<__half, float>(buf.a, buf.b, buf.d,
      [&](auto &tD, auto &tA, auto &tB) {
        TMATMUL_BIAS(tD, tA, tB, bias, fixp::keep_acc());
      });
#elif defined(ACC)                        // D = C + A*B
  buf_t<__half, float> buf;
  acc_tile_t<TM, TN> cacc;
  run_matmul<__half, float>(buf.a, buf.b, buf.d,
      [&](auto &tD, auto &tA, auto &tB) {
        TMATMUL_ACC(tD, cacc, tA, tB, fixp::keep_acc());
      });
#elif defined(MX)                         // C = (A*ScaleA)*(B*ScaleB)
  buf_t<__half, float> buf;
  scale_a_tile_t<TM, TK> sa;
  scale_b_tile_t<TK, TN> sb;
  run_matmul<__half, float>(buf.a, buf.b, buf.d,
      [&](auto &tD, auto &tA, auto &tB) {
        TMATMUL_MX(tD, tA, sa, tB, sb, fixp::keep_acc());
      });
#elif defined(MXBIAS)                     // D = (A*ScaleA)*(B*ScaleB) + Bias
  buf_t<__half, float> buf;
  scale_a_tile_t<TM, TK> sa;
  scale_b_tile_t<TK, TN> sb;
  bias_tile_t<TN> bias;
  run_matmul<__half, float>(buf.a, buf.b, buf.d,
      [&](auto &tD, auto &tA, auto &tB) {
        TMATMUL_MX_BIAS(tD, tA, sa, tB, sb, bias, fixp::keep_acc());
      });
#elif defined(MXACC)                      // D = C + (A*ScaleA)*(B*ScaleB)
  buf_t<__half, float> buf;
  acc_tile_t<TM, TN> cacc;
  scale_a_tile_t<TM, TK> sa;
  scale_b_tile_t<TK, TN> sb;
  run_matmul<__half, float>(buf.a, buf.b, buf.d,
      [&](auto &tD, auto &tA, auto &tB) {
        TMATMUL_MX_ACC(tD, cacc, tA, sa, tB, sb, fixp::keep_acc());
      });
#elif defined(GEMV)                       // D = mtx * vec (M=1)
  buf_t<__half, float> buf;
  run_gemv<__half, float>(buf.a, buf.b, buf.d,
      [&](auto &tD, auto &tMtx, auto &tVec) {
        TGEMV(tD, tMtx, tVec, fixp::keep_acc());
      });
#elif defined(GEMV_BIAS)
  buf_t<__half, float> buf;
  bias_tile_t<TN> bias;
  run_gemv<__half, float>(buf.a, buf.b, buf.d,
      [&](auto &tD, auto &tMtx, auto &tVec) {
        TGEMV_BIAS(tD, tMtx, tVec, bias, fixp::keep_acc());
      });
#elif defined(GEMV_ACC)
  buf_t<__half, float> buf;
  bias_tile_t<TN> cacc;    // GEMV accumulator C is 1 x N
  run_gemv<__half, float>(buf.a, buf.b, buf.d,
      [&](auto &tD, auto &tMtx, auto &tVec) {
        TGEMV_ACC(tD, cacc, tMtx, tVec, fixp::keep_acc());
      });
#elif defined(GEMV_MX)
  buf_t<__half, float> buf;
  scale_b_tile_t<TK, TN> smtx;   // scale for mtx: K x N
  scale_vec_tile_t<TK> svec;     // scale for vec: 1 x K
  run_gemv<__half, float>(buf.a, buf.b, buf.d,
      [&](auto &tD, auto &tMtx, auto &tVec) {
        TGEMV_MX(tD, tMtx, smtx, tVec, svec, fixp::keep_acc());
      });
#elif defined(GEMV_MX_BIAS)
  buf_t<__half, float> buf;
  scale_b_tile_t<TK, TN> smtx;
  scale_vec_tile_t<TK> svec;
  bias_tile_t<TN> bias;
  run_gemv<__half, float>(buf.a, buf.b, buf.d,
      [&](auto &tD, auto &tMtx, auto &tVec) {
        TGEMV_MX_BIAS(tD, tMtx, smtx, tVec, svec, bias, fixp::keep_acc());
      });
#elif defined(GEMV_MX_ACC)
  buf_t<__half, float> buf;
  bias_tile_t<TN> cacc;
  scale_b_tile_t<TK, TN> smtx;
  scale_vec_tile_t<TK> svec;
  run_gemv<__half, float>(buf.a, buf.b, buf.d,
      [&](auto &tD, auto &tMtx, auto &tVec) {
        TGEMV_MX_ACC(tD, cacc, tMtx, smtx, tVec, svec, fixp::keep_acc());
      });

// === full-options spot-check (s8 scalar quant on non-TMATMUL ops) =========
#elif defined(BIAS_S8)
  buf_t<__half, int8_t> buf;
  bias_tile_t<TN> bias;
  run_matmul<__half, int8_t>(buf.a, buf.b, buf.d,
      [&](auto &tD, auto &tA, auto &tB) {
        TMATMUL_BIAS(tD, tA, tB, bias, fixp::s8(mk_desc(1, 0, 9)));
      });
#elif defined(ACC_S8)
  buf_t<__half, int8_t> buf;
  acc_tile_t<TM, TN> cacc;
  run_matmul<__half, int8_t>(buf.a, buf.b, buf.d,
      [&](auto &tD, auto &tA, auto &tB) {
        TMATMUL_ACC(tD, cacc, tA, tB, fixp::s8(mk_desc(1, 0, 9)));
      });
#elif defined(MX_S8)
  buf_t<__half, int8_t> buf;
  scale_a_tile_t<TM, TK> sa;
  scale_b_tile_t<TK, TN> sb;
  run_matmul<__half, int8_t>(buf.a, buf.b, buf.d,
      [&](auto &tD, auto &tA, auto &tB) {
        TMATMUL_MX(tD, tA, sa, tB, sb, fixp::s8(mk_desc(1, 0, 9)));
      });
#elif defined(GEMV_S8)
  buf_t<__half, int8_t> buf;
  run_gemv<__half, int8_t>(buf.a, buf.b, buf.d,
      [&](auto &tD, auto &tMtx, auto &tVec) {
        TGEMV(tD, tMtx, tVec, fixp::s8(mk_desc(1, 0, 9)));
      });
#elif defined(GEMV_MX_S8)
  buf_t<__half, int8_t> buf;
  scale_b_tile_t<TK, TN> smtx;
  scale_vec_tile_t<TK> svec;
  run_gemv<__half, int8_t>(buf.a, buf.b, buf.d,
      [&](auto &tD, auto &tMtx, auto &tVec) {
        TGEMV_MX(tD, tMtx, smtx, tVec, svec, fixp::s8(mk_desc(1, 0, 9)));
      });

// === Shared-Right B (B.IOS) ===============================================
#elif defined(SHARED)                      // TMATMUL with SharedTile<Right> B
  buf_t<__half, float> buf;
  run_matmul<__half, float>(buf.a, buf.b, buf.d,
      [&](auto &tD, auto &tA, auto &tB) {
        SharedTile<std::decay_t<decltype(tB)>> b_shared(tB);
        TMATMUL(tD, tA, b_shared, fixp::keep_acc());
      });
#elif defined(S8_SHARED)                   // Shared B + s8 scalar quant
  buf_t<__half, int8_t> buf;
  run_matmul<__half, int8_t>(buf.a, buf.b, buf.d,
      [&](auto &tD, auto &tA, auto &tB) {
        SharedTile<std::decay_t<decltype(tB)>> b_shared(tB);
        TMATMUL(tD, tA, b_shared, fixp::s8(mk_desc(1, 0, 9)));
      });

// === edge-case FPATR / operand-stream features ============================
#elif defined(LRELU_ONLY)                  // LReLU without scalar quant -> [zero,lrelu]
  buf_t<__half, float> buf;
  run_matmul<__half, float>(buf.a, buf.b, buf.d,
      [&](auto &tD, auto &tA, auto &tB) {
        TMATMUL(tD, tA, tB, fixp::keep_acc().lrelu(1));
      });
#elif defined(VQF_S8_PRELU)               // vector-quant + PReLU (SrcMask=6)
  buf_t<__half, int8_t> buf;
  par_tile_t<TN> quant, prelu;
  run_matmul<__half, int8_t>(buf.a, buf.b, buf.d,
      [&](auto &tD, auto &tA, auto &tB) {
        TMATMUL(tD, tA, tB, fixp::s8(quant).prelu(prelu));
      });
#elif defined(LEGACY3)                     // 3-param no-options TMATMUL(c,a,b)
  buf_t<__half, float> buf;
  run_matmul<__half, float>(buf.a, buf.b, buf.d,
      [&](auto &tD, auto &tA, auto &tB) {
        TMATMUL(tD, tA, tB);
      });

#else
#error "no fixp mode selected: pass -D<KEEP_ACC|F16|BF16|S_*|V_*|BIAS|ACC|MX|GEMV|...> (see compile.all)"
#endif

  return 0;
}
