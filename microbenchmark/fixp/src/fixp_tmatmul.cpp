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
// The compile-all mode label "MX" is passed as -DMX.  Undefine it before
// including TileOP headers because the current API also uses MX as an internal
// template parameter name.
#if defined(MX)
#define FIXP_MODE_MX
#undef MX
#endif

#include <common/pto_tileop.hpp>
#include <cstdint>
#include <type_traits>

#include "benchmark.h"
#ifdef CROSS_MODEL_CORPUS
#include "../../common/cross_model_result.hpp"
#endif

#ifdef RES_CHECK
static volatile int g_numeric_failure = 0;
#endif
#if defined(RES_CHECK) || defined(CROSS_MODEL_CORPUS)
__attribute__((noinline)) void fill_bytes(void *p, size_t n, uint8_t value) {
  volatile uint8_t source = value;
  auto *bytes = reinterpret_cast<uint8_t *>(p);
  for (size_t i = 0; i < n; ++i) bytes[i] = source;
}
#endif
#ifdef RES_CHECK
template <typename T>
__attribute__((noinline)) void check_zero_result(const T *p, int n) {
  const auto *bytes = reinterpret_cast<const uint8_t *>(p);
  for (size_t i = 0; i < static_cast<size_t>(n) * sizeof(T); ++i)
    if (bytes[i] != 0) {
      g_numeric_failure = 1;
      return;
    }
}
#else
template <typename T>
inline void check_zero_result(const T *, int) {}
#endif

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

// PTO v0.58 persistent Local CUBE layouts.  A and D/C use the same CUBE_M
// family; the right matrix operand always uses CUBE_N8.
template <typename T, int M, int K, int VM = M, int VK = K>
using cube_left_t = std::conditional_t<
    (M <= 16), CubeTileM16<T, M, K, VM, VK>,
    CubeTileM32<T, M, K, VM, VK>>;

template <typename T, int M, int N, int VM = M, int VN = N>
using cube_acc_t = std::conditional_t<
    (M <= 16), CubeAccumulatorM16<T, M, N, VM, VN>,
    CubeAccumulatorM32<T, M, N, VM, VN>>;

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

template <typename T, int N>
using typed_bias_tile_t =
    Tile<Location::Vec, T, 4, N, BLayout::RowMajor, 1, N>;

// FP32 accumulator C tile for ACC ops. It must match D's CUBE_M layout.
template <int M, int N>
using acc_tile_t = cube_acc_t<float, M, N>;

template <int N>
using gemv_acc_tile_t = CubeAccumulatorM16<float, 16, N, 1, N>;

template <typename T, int N>
using typed_gemv_acc_tile_t = CubeAccumulatorM16<T, 16, N, 1, N>;

// CScale is an ACC-only U8 mathematical source in CUBE_M32 layout. Its valid
// shape is M x 1 while the physical carrier keeps the accumulator box width.
template <int M, int N>
using cscale_tile_t =
    Tile<Location::Vec, uint8_t, M, N, BLayout::CubeM32, M, 1>;

// MX scale tiles use one E8M0 value per 32 elements along K.  Physical shapes
// are padded to at least 512 B while valid shapes follow the matrix contract:
// ScaleA=M x ceil(K/32), ScaleB=ceil(K/32) x N.
template <int M, int K>
using scale_a_tile_t =
    Tile<Location::Vec, __fp8_e8m0, M, 16, BLayout::RowMajor,
         M, (K + 31) / 32>;
template <int K, int N>
using scale_b_tile_t =
    Tile<Location::Vec, __fp8_e8m0, 16, N, BLayout::RowMajor,
         (K + 31) / 32, N>;
// TGEMV_MX vector scale: valid 1 x ceil(K/32), padded to 512 B.
template <int K>
using scale_vec_tile_t =
    Tile<Location::Vec, __fp8_e8m0, 16, 32, BLayout::RowMajor,
         1, (K + 31) / 32>;

// Auxiliary matrix operands are real Tile operands, not C++ parameter
// objects.  Keep their creation, definition and use in one inlined scope;
// passing an uninitialised Tile through a captured lambda makes it spill as a
// scalar S64 object and loses the Tile descriptor required by B.FPATR.
template <typename TileT>
__attribute__((always_inline)) inline void load_aux(TileT &tile,
                                                    uint8_t *address) {
  using gm_t = global_tensor<typename TileT::DType,
                             RowMajor<TileT::ValidRow, TileT::ValidCol>>;
  gm_t gm(reinterpret_cast<typename TileT::DType *>(address));
  if constexpr (TileT::IsCubeLayout)
    TLOAD_CUBE(tile, gm);
  else
    TLOAD(tile, gm);
}

template <typename SrcT, typename DstT, typename OptsMaker>
__attribute__((noinline)) void run_single(SrcT *a_ptr, SrcT *b_ptr,
                                          DstT *d_ptr, OptsMaker maker) {
  constexpr int kM = TM, kN = TN, kK = TK;
  using gm_a = global_tensor<SrcT, RowMajor<kM, kK>>;
  using gm_b = global_tensor<SrcT, RowMajor<kK, kN>>;
  using gm_c = global_tensor<DstT, RowMajor<kM, kN>>;
  using tile_a = cube_left_t<SrcT, kM, kK>;
  using tile_b = CubeTileN8<SrcT, kK, kN>;
  using tile_d = cube_acc_t<DstT, kM, kN>;

  gm_a gA(a_ptr);
  gm_b gB(b_ptr);
  gm_c gC(d_ptr);
  tile_a tA;
  tile_b tB;
  tile_d tD;

  BENCHSTART;
  TLOAD_CUBE(tA, gA);
  TLOAD_CUBE(tB, gB);
  auto options = maker();
  TMATMUL(tD, tA, tB, options);
  TSTORE_CUBE(gC, tD);
  BENCHEND;
  check_zero_result(d_ptr, kM * kN);
#ifdef CROSS_MODEL_CORPUS
  publish_cross_model_result(d_ptr, kM * kN);
#endif
}

// TMATMUL-family driver: Cube-loads A/B, declares Cube D, then invokes
// kernel(tD,tA,tB). The callback calls TMATMUL_BIAS / TMATMUL_ACC /
// TMATMUL_MX / ... with any auxiliary tiles captured from the enclosing scope.
template <typename SrcT, typename DstT, typename Kernel>
__attribute__((noinline)) void run_matmul(SrcT *a_ptr, SrcT *b_ptr,
                                         DstT *d_ptr, Kernel kernel) {
  constexpr int kM = TM, kN = TN, kK = TK;
  using gm_a = global_tensor<SrcT, RowMajor<kM, kK>>;
  using gm_b = global_tensor<SrcT, RowMajor<kK, kN>>;
  using gm_d = global_tensor<DstT, RowMajor<kM, kN>>;
  using tile_a = cube_left_t<SrcT, kM, kK>;
  using tile_b = CubeTileN8<SrcT, kK, kN>;
  using tile_d = cube_acc_t<DstT, kM, kN>;

  gm_a gA(a_ptr);
  gm_b gB(b_ptr);
  gm_d gD(d_ptr);
  tile_a tA;
  tile_b tB;
  tile_d tD;

  BENCHSTART;
  TLOAD_CUBE(tA, gA);
  TLOAD_CUBE(tB, gB);
  kernel(tD, tA, tB);
  TSTORE_CUBE(gD, tD);
  BENCHEND;
  check_zero_result(d_ptr, kM * kN);
#ifdef CROSS_MODEL_CORPUS
  publish_cross_model_result(d_ptr, kM * kN);
#endif
}

// Mixed A/B dtypes are legal when both belong to the same numeric class.
// This driver also enables the asymmetric MX scale-mask cases.
template <typename AT, typename BT, typename DstT, typename Kernel>
__attribute__((noinline)) void run_matmul_mixed(AT *a_ptr, BT *b_ptr,
                                               DstT *d_ptr, Kernel kernel) {
  constexpr int kM = TM, kN = TN, kK = TK;
  using gm_a = global_tensor<AT, RowMajor<kM, kK>>;
  using gm_b = global_tensor<BT, RowMajor<kK, kN>>;
  using gm_d = global_tensor<DstT, RowMajor<kM, kN>>;
  using tile_a = cube_left_t<AT, kM, kK>;
  using tile_b = CubeTileN8<BT, kK, kN>;
  using tile_d = cube_acc_t<DstT, kM, kN>;

  gm_a gA(a_ptr); gm_b gB(b_ptr); gm_d gD(d_ptr);
  tile_a tA; tile_b tB; tile_d tD;
  BENCHSTART;
  TLOAD_CUBE(tA, gA);
  TLOAD_CUBE(tB, gB);
  kernel(tD, tA, tB);
  TSTORE_CUBE(gD, tD);
  BENCHEND;
  check_zero_result(d_ptr, kM * kN);
#ifdef CROSS_MODEL_CORPUS
  publish_cross_model_result(d_ptr, kM * kN);
#endif
}

// A Shared matrix operand selects the four-PE cooperative contract.  Publish
// the Core-total A (group_M=4*TM) and B directly from GM; D remains the local
// per-PE TMxTN accumulator.  A default-constructed SharedTile is only an
// opaque handle and does not define or ready any Shared bytes.
template <typename DstT, typename Kernel>
__attribute__((noinline)) void run_matmul_shared(__half *a_ptr, __half *b_ptr,
                                                DstT *d_ptr, Kernel kernel) {
  constexpr int kM = TM, kN = TN, kK = TK;
  using gm_a = global_tensor<__half, RowMajor<4 * kM, kK>>;
  using gm_b = global_tensor<__half, RowMajor<kK, kN>>;
  using gm_d = global_tensor<DstT, RowMajor<kM, kN>>;
  using tile_a_desc = SharedMatrixLeft<__half, 4 * kM, kK>;
  using tile_b_desc = SharedMatrixRight<__half, kK, kN>;
  using tile_d = cube_acc_t<DstT, kM, kN>;

  gm_a gA(a_ptr);
  gm_b gB(b_ptr);
  gm_d gD(d_ptr);
  tile_d tD;

  BENCHSTART;
  // Each of the four PEs publishes its own quarter.  A multi-PE Shared TLOAD
  // with mask=15 would require a B.ASSEMBLE destination instead.
  SharedTile<tile_a_desc> tA = TLOAD<tile_a_desc, 1>(gA);
  SharedTile<tile_b_desc> tB = TLOAD<tile_b_desc, 1>(gB);
  kernel(tD, tA, tB);
  TSTORE_CUBE(gD, tD);
  BENCHEND;
  check_zero_result(d_ptr, kM * kN);
#ifdef CROSS_MODEL_CORPUS
  publish_cross_model_result(d_ptr, kM * kN);
#endif
}

// [TEMP] Shared TLOAD A/B + TMATMUL with NO destination TSTORE.  The matmul
// result stays in the local tile register and is never written back to GM;
// used to inspect the cooperative TLOAD + TMATMUL bundle in isolation.
template <typename DstT, typename Kernel>
__attribute__((noinline)) void run_matmul_shared_nostore(__half *a_ptr,
                                                        __half *b_ptr,
                                                        Kernel kernel) {
  constexpr int kM = TM, kN = TN, kK = TK;
  using gm_a = global_tensor<__half, RowMajor<4 * kM, kK>>;
  using gm_b = global_tensor<__half, RowMajor<kK, kN>>;
  using tile_a_desc = SharedMatrixLeft<__half, 4 * kM, kK>;
  using tile_b_desc = SharedMatrixRight<__half, kK, kN>;
  using tile_d = cube_acc_t<DstT, kM, kN>;

  gm_a gA(a_ptr);
  gm_b gB(b_ptr);
  tile_d tD;

  BENCHSTART;
  SharedTile<tile_a_desc> tA = TLOAD<tile_a_desc, 1>(gA);
  SharedTile<tile_b_desc> tB = TLOAD<tile_b_desc, 1>(gB);
  kernel(tD, tA, tB);
  BENCHEND;
}

template <bool TransA, bool TransB, typename DstT, typename Kernel>
__attribute__((noinline)) void run_matmul_shared_transpose(
    __half *a_ptr, __half *b_ptr, DstT *d_ptr, Kernel kernel) {
  constexpr int kGroupM = 4 * TM, kN = TN, kK = TK;
  constexpr int kARows = TransA ? kK : kGroupM;
  constexpr int kACols = TransA ? kGroupM : kK;
  constexpr int kBRows = TransB ? kN : kK;
  constexpr int kBCols = TransB ? kK : kN;
  using gm_a = global_tensor<__half, RowMajor<kARows, kACols>>;
  using gm_b = global_tensor<__half, RowMajor<kBRows, kBCols>>;
  using gm_d = global_tensor<DstT, RowMajor<TM, kN>>;
  using tile_a_desc = SharedMatrixLeft<__half, kARows, kACols>;
  using tile_b_desc = SharedMatrixRight<__half, kBRows, kBCols>;
  using tile_d = cube_acc_t<DstT, TM, kN>;

  gm_a gA(a_ptr); gm_b gB(b_ptr); gm_d gD(d_ptr); tile_d tD;
  BENCHSTART;
  SharedTile<tile_a_desc> tA = TLOAD<tile_a_desc, 1>(gA);
  SharedTile<tile_b_desc> tB = TLOAD<tile_b_desc, 1>(gB);
  kernel(tD, tA, tB);
  TSTORE_CUBE(gD, tD);
  BENCHEND;
  check_zero_result(d_ptr, TM * kN);
#ifdef CROSS_MODEL_CORPUS
  publish_cross_model_result(d_ptr, TM * kN);
#endif
}

// TGEMV-family driver (M=1): vec=CUBE_M16(1xK valid), mtx=CUBE_N8(KxN),
// D=CUBE_M16(1xN valid). The lambda kernel(tD,tMtx,tVec) calls the specific
// TGEMV op with captured auxiliary tiles.
template <typename SrcT, typename DstT, typename Kernel>
__attribute__((noinline)) void run_gemv(SrcT *vec_ptr, SrcT *mtx_ptr,
                                        DstT *d_ptr, Kernel kernel) {
  constexpr int kK = TK, kN = TN;
  using gm_vec = global_tensor<SrcT, RowMajor<1, kK>>;
  using gm_mtx = global_tensor<SrcT, RowMajor<kK, kN>>;
  using gm_d = global_tensor<DstT, RowMajor<1, kN>>;
  using tile_vec = CubeTileM16<SrcT, 16, kK, 1, kK>;
  using tile_mtx = CubeTileN8<SrcT, kK, kN>;
  using tile_d = CubeAccumulatorM16<DstT, 16, kN, 1, kN>;

  gm_vec gV(vec_ptr);
  gm_mtx gMx(mtx_ptr);
  gm_d gD(d_ptr);
  tile_vec tVec;
  tile_mtx tMtx;
  tile_d tD;

  BENCHSTART;
  TLOAD_CUBE(tVec, gV);
  TLOAD_CUBE(tMtx, gMx);
  kernel(tD, tMtx, tVec);
  TSTORE_CUBE(gD, tD);
  BENCHEND;
  check_zero_result(d_ptr, kN);
#ifdef CROSS_MODEL_CORPUS
  publish_cross_model_result(d_ptr, kN);
#endif
}

template <typename VecT, typename MtxT, typename DstT, typename Kernel>
__attribute__((noinline)) void run_gemv_mixed(VecT *vec_ptr, MtxT *mtx_ptr,
                                             DstT *d_ptr, Kernel kernel) {
  constexpr int kK = TK, kN = TN;
  using gm_vec = global_tensor<VecT, RowMajor<1, kK>>;
  using gm_mtx = global_tensor<MtxT, RowMajor<kK, kN>>;
  using gm_d = global_tensor<DstT, RowMajor<1, kN>>;
  using tile_vec = CubeTileM16<VecT, 16, kK, 1, kK>;
  using tile_mtx = CubeTileN8<MtxT, kK, kN>;
  using tile_d = CubeAccumulatorM16<DstT, 16, kN, 1, kN>;

  gm_vec gV(vec_ptr); gm_mtx gMx(mtx_ptr); gm_d gD(d_ptr);
  tile_vec tVec; tile_mtx tMtx; tile_d tD;
  BENCHSTART;
  TLOAD_CUBE(tVec, gV);
  TLOAD_CUBE(tMtx, gMx);
  kernel(tD, tMtx, tVec);
  TSTORE_CUBE(gD, tD);
  BENCHEND;
  check_zero_result(d_ptr, kN);
#ifdef CROSS_MODEL_CORPUS
  publish_cross_model_result(d_ptr, kN);
#endif
}

template <typename SrcT, typename DstT>
struct buf_t {
  static constexpr size_t kAlign = 4096;
  static constexpr size_t kAlignMask = ~(kAlign - 1);
  static constexpr size_t kAuxBytes = 64 * 1024;
  // Shared-A coverage needs a Core-total 4*TM x TK allocation; ordinary
  // modes simply use the first TM x TK region.
  alignas(16) uint8_t a_raw[4 * TM * TK * sizeof(SrcT) + 2 * kAlign];
  alignas(16) uint8_t b_raw[TK * TN * sizeof(SrcT) + 2 * kAlign];
  alignas(16) uint8_t d_raw[TM * TN * sizeof(DstT) + 2 * kAlign];
  alignas(16) uint8_t aux_raw[kAuxBytes + 2 * kAlign];
  SrcT *a;
  SrcT *b;
  DstT *d;
  uint8_t *aux;
  buf_t() {
    a = (SrcT *)(((uint64_t)&a_raw[0] & kAlignMask) + kAlign);
    b = (SrcT *)(((uint64_t)&b_raw[0] & kAlignMask) + kAlign);
    d = (DstT *)(((uint64_t)&d_raw[0] & kAlignMask) + kAlign);
    aux = (uint8_t *)(((uint64_t)&aux_raw[0] & kAlignMask) + kAlign);
#if defined(RES_CHECK) || defined(CROSS_MODEL_CORPUS)
    fill_bytes(a, 4 * TM * TK * sizeof(SrcT), 0);
    fill_bytes(b, TK * TN * sizeof(SrcT), 0);
    // A non-zero destination catches a missing or partial store. Zero A/B and
    // zero auxiliary operands make zero the common oracle for every mode.
    fill_bytes(d, TM * TN * sizeof(DstT), 1);
    fill_bytes(aux, kAuxBytes, 0);
#endif
#ifndef RES_CHECK
    auto *descriptors = reinterpret_cast<uint64_t *>(aux);
    for (size_t i = 0; i < kAuxBytes / sizeof(uint64_t); ++i)
      descriptors[i] = mk_desc(1, 0, 9);
#endif
  }
};

template <typename AT, typename BT, typename DstT>
struct mixed_buf_t {
  static constexpr size_t kAlign = 4096;
  static constexpr size_t kAlignMask = ~(kAlign - 1);
  static constexpr size_t kAuxBytes = 64 * 1024;
  alignas(16) uint8_t a_raw[4 * TM * TK * sizeof(AT) + 2 * kAlign];
  alignas(16) uint8_t b_raw[TK * TN * sizeof(BT) + 2 * kAlign];
  alignas(16) uint8_t d_raw[TM * TN * sizeof(DstT) + 2 * kAlign];
  alignas(16) uint8_t aux_raw[kAuxBytes + 2 * kAlign];
  AT *a;
  BT *b;
  DstT *d;
  uint8_t *aux;
  mixed_buf_t() {
    a = reinterpret_cast<AT *>((reinterpret_cast<uint64_t>(&a_raw[0]) &
                                kAlignMask) + kAlign);
    b = reinterpret_cast<BT *>((reinterpret_cast<uint64_t>(&b_raw[0]) &
                                kAlignMask) + kAlign);
    d = reinterpret_cast<DstT *>((reinterpret_cast<uint64_t>(&d_raw[0]) &
                                  kAlignMask) + kAlign);
    aux = reinterpret_cast<uint8_t *>((reinterpret_cast<uint64_t>(&aux_raw[0]) &
                                       kAlignMask) + kAlign);
#if defined(RES_CHECK) || defined(CROSS_MODEL_CORPUS)
    fill_bytes(a, 4 * TM * TK * sizeof(AT), 0);
    fill_bytes(b, TK * TN * sizeof(BT), 0);
    fill_bytes(d, TM * TN * sizeof(DstT), 1);
    fill_bytes(aux, kAuxBytes, 0);
#endif
#ifndef RES_CHECK
    auto *descriptors = reinterpret_cast<uint64_t *>(aux);
    for (size_t i = 0; i < kAuxBytes / sizeof(uint64_t); ++i)
      descriptors[i] = mk_desc(1, 0, 9);
#endif
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

// --- matrix numeric-class and mixed-input contracts ----------------------
#elif defined(SIGNED_KEEP_ACC)
  buf_t<int8_t, int32_t> buf;
  run_matmul<int8_t, int32_t>(buf.a, buf.b, buf.d,
      [&](auto &d, auto &a, auto &b) { TMATMUL(d, a, b); });
#elif defined(SIGNED_ACC)
  buf_t<int8_t, int32_t> buf;
  run_matmul<int8_t, int32_t>(buf.a, buf.b, buf.d,
      [&](auto &d, auto &a, auto &b) {
        cube_acc_t<int32_t, TM, TN> c; load_aux(c, buf.aux);
        TMATMUL_ACC(d, c, a, b);
      });
#elif defined(SIGNED_BIAS)
  buf_t<int8_t, int32_t> buf;
  run_matmul<int8_t, int32_t>(buf.a, buf.b, buf.d,
      [&](auto &d, auto &a, auto &b) {
        typed_bias_tile_t<int32_t, TN> bias; load_aux(bias, buf.aux);
        TMATMUL_BIAS(d, a, b, bias);
      });
#elif defined(UNSIGNED_KEEP_ACC)
  buf_t<uint8_t, uint32_t> buf;
  run_matmul<uint8_t, uint32_t>(buf.a, buf.b, buf.d,
      [&](auto &d, auto &a, auto &b) { TMATMUL(d, a, b); });
#elif defined(UNSIGNED_ACC)
  buf_t<uint8_t, uint32_t> buf;
  run_matmul<uint8_t, uint32_t>(buf.a, buf.b, buf.d,
      [&](auto &d, auto &a, auto &b) {
        cube_acc_t<uint32_t, TM, TN> c; load_aux(c, buf.aux);
        TMATMUL_ACC(d, c, a, b);
      });
#elif defined(UNSIGNED_BIAS)
  buf_t<uint8_t, uint32_t> buf;
  run_matmul<uint8_t, uint32_t>(buf.a, buf.b, buf.d,
      [&](auto &d, auto &a, auto &b) {
        typed_bias_tile_t<uint32_t, TN> bias; load_aux(bias, buf.aux);
        TMATMUL_BIAS(d, a, b, bias);
      });
#elif defined(MIXED_FLOAT)
  mixed_buf_t<__half, __bf16, float> buf;
  run_matmul_mixed<__half, __bf16, float>(buf.a, buf.b, buf.d,
      [&](auto &d, auto &a, auto &b) { TMATMUL(d, a, b); });
#elif defined(GEMV_MIXED_FLOAT)
  mixed_buf_t<__half, __bf16, float> buf;
  run_gemv_mixed<__half, __bf16, float>(buf.a, buf.b, buf.d,
      [&](auto &d, auto &mtx, auto &vec) { TGEMV(d, mtx, vec); });

// --- scalar quant descriptor modes ---------------------------------------
#elif defined(S_REQS8)
  buf_t<int8_t, int8_t> buf;
  run_single<int8_t, int8_t>(buf.a, buf.b, buf.d, [] {
    return fixp::scalar<FixpPreQuantMode::REQS8Pre>(mk_desc(1, 0, 9));
  });
#elif defined(S_DEQF16)
  buf_t<int8_t, __half> buf;
  run_single<int8_t, __half>(buf.a, buf.b, buf.d, [] {
    return fixp::scalar<FixpPreQuantMode::DEQF16>(mk_desc(1, 0, 0));
  });
#elif defined(S_SHIFTS16)
  buf_t<int8_t, int16_t> buf;
  run_single<int8_t, int16_t>(buf.a, buf.b, buf.d, [] {
    return fixp::scalar<FixpPreQuantMode::SHIFTS322S16>(mk_desc(1, 0, 17));
  });
#elif defined(S_QF_S4)
  buf_t<int8_t, __int4x2> buf;
  run_single<int8_t, __int4x2>(buf.a, buf.b, buf.d, [] {
    return fixp::scalar<FixpPreQuantMode::QF322S4Pre>(mk_desc(1, 0, 5));
  });
#elif defined(S_QF_S16)
  buf_t<int8_t, int16_t> buf;
  run_single<int8_t, int16_t>(buf.a, buf.b, buf.d, [] {
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
  buf_t<int8_t, __bf16> buf;
  run_single<int8_t, __bf16>(buf.a, buf.b, buf.d, [] {
    return fixp::scalar<FixpPreQuantMode::QS322BF16Pre>(mk_desc(1, 0, 9));
  });

// --- vector quant parameter tile modes ------------------------------------
#elif defined(V_REQS8)
  buf_t<int8_t, int8_t> buf;
  run_matmul<int8_t, int8_t>(buf.a, buf.b, buf.d, [&](auto &d, auto &a, auto &b) {
    par_tile_t<TN> quant; load_aux(quant, buf.aux);
    TMATMUL(d, a, b, fixp::vector<FixpPreQuantMode::VREQS8Pre>(quant));
  });
#elif defined(V_DEQF16)
  buf_t<int8_t, __half> buf;
  run_matmul<int8_t, __half>(buf.a, buf.b, buf.d, [&](auto &d, auto &a, auto &b) {
    par_tile_t<TN> quant; load_aux(quant, buf.aux);
    TMATMUL(d, a, b, fixp::vector<FixpPreQuantMode::VDEQF16>(quant));
  });
#elif defined(V_SHIFTS16)
  buf_t<int8_t, int16_t> buf;
  run_matmul<int8_t, int16_t>(buf.a, buf.b, buf.d, [&](auto &d, auto &a, auto &b) {
    par_tile_t<TN> quant; load_aux(quant, buf.aux);
    TMATMUL(d, a, b, fixp::vector<FixpPreQuantMode::VSHIFTS322S16>(quant));
  });
#elif defined(V_QF_S4)
  buf_t<int8_t, __int4x2> buf;
  run_matmul<int8_t, __int4x2>(buf.a, buf.b, buf.d, [&](auto &d, auto &a, auto &b) {
    par_tile_t<TN> quant; load_aux(quant, buf.aux);
    TMATMUL(d, a, b, fixp::vector<FixpPreQuantMode::VQF322S4Pre>(quant));
  });
#elif defined(V_QF_S16)
  buf_t<int8_t, int16_t> buf;
  run_matmul<int8_t, int16_t>(buf.a, buf.b, buf.d, [&](auto &d, auto &a, auto &b) {
    par_tile_t<TN> quant; load_aux(quant, buf.aux);
    TMATMUL(d, a, b, fixp::vector<FixpPreQuantMode::VQF322S16Pre>(quant));
  });
#elif defined(V_QF_S8)
  buf_t<__half, int8_t> buf;
  run_matmul<__half, int8_t>(buf.a, buf.b, buf.d, [&](auto &d, auto &a, auto &b) {
    par_tile_t<TN> quant; load_aux(quant, buf.aux);
    TMATMUL(d, a, b, fixp::s8(quant));
  });
#elif defined(V_QF_HIF8)
  buf_t<__half, __hif8> buf;
  run_matmul<__half, __hif8>(buf.a, buf.b, buf.d, [&](auto &d, auto &a, auto &b) {
    par_tile_t<TN> quant; load_aux(quant, buf.aux);
    TMATMUL(d, a, b, fixp::vector<FixpPreQuantMode::VQF322HIF8Pre>(quant));
  });
#elif defined(V_QF_F16)
  buf_t<__half, __half> buf;
  run_matmul<__half, __half>(buf.a, buf.b, buf.d, [&](auto &d, auto &a, auto &b) {
    par_tile_t<TN> quant; load_aux(quant, buf.aux);
    TMATMUL(d, a, b, fixp::vector<FixpPreQuantMode::VQF322F16Pre>(quant));
  });
#elif defined(V_QF_BF16)
  buf_t<__half, __bf16> buf;
  run_matmul<__half, __bf16>(buf.a, buf.b, buf.d, [&](auto &d, auto &a, auto &b) {
    par_tile_t<TN> quant; load_aux(quant, buf.aux);
    TMATMUL(d, a, b, fixp::vector<FixpPreQuantMode::VQF322BF16Pre>(quant));
  });
#elif defined(V_QF_FP8)
  buf_t<__half, __fp8_e4m3> buf;
  run_matmul<__half, __fp8_e4m3>(buf.a, buf.b, buf.d, [&](auto &d, auto &a, auto &b) {
    par_tile_t<TN> quant; load_aux(quant, buf.aux);
    TMATMUL(d, a, b, fixp::vector<FixpPreQuantMode::VQF322FP8Pre>(quant));
  });
#elif defined(V_QF_F32)
  buf_t<__half, float> buf;
  run_matmul<__half, float>(buf.a, buf.b, buf.d, [&](auto &d, auto &a, auto &b) {
    par_tile_t<TN> quant; load_aux(quant, buf.aux);
    TMATMUL(d, a, b, fixp::vector<FixpPreQuantMode::VQF322F32Pre>(quant));
  });
#elif defined(V_QS_BF16)
  buf_t<int8_t, __bf16> buf;
  run_matmul<int8_t, __bf16>(buf.a, buf.b, buf.d, [&](auto &d, auto &a, auto &b) {
    par_tile_t<TN> quant; load_aux(quant, buf.aux);
    TMATMUL(d, a, b, fixp::vector<FixpPreQuantMode::VQS322BF16Pre>(quant));
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
  run_matmul<__half, int8_t>(buf.a, buf.b, buf.d, [&](auto &d, auto &a, auto &b) {
    par_tile_t<TN> quant; load_aux(quant, buf.aux);
    TMATMUL(d, a, b, fixp::s8(quant).relu());
  });
#elif defined(F16_PRELU)
  buf_t<__half, __half> buf;
  run_matmul<__half, __half>(buf.a, buf.b, buf.d, [&](auto &d, auto &a, auto &b) {
    par_tile_t<TN> prelu; load_aux(prelu, buf.aux);
    TMATMUL(d, a, b, fixp::f16().prelu(prelu));
  });
#elif defined(S8_PRELU)
  buf_t<__half, int8_t> buf;
  run_matmul<__half, int8_t>(buf.a, buf.b, buf.d, [&](auto &d, auto &a, auto &b) {
    par_tile_t<TN> prelu; load_aux(prelu, buf.aux);
    TMATMUL(d, a, b, fixp::s8(mk_desc(1, 0, 9)).prelu(prelu));
  });

// --- RowMax / GroupMax / MaxAbs --------------------------------------------
#elif defined(ROWMAX)
  buf_t<__half, float> buf;
  run_matmul<__half, float>(buf.a, buf.b, buf.d, [&](auto &d, auto &a, auto &b) {
    row_max_tile_t<TM> row_out;
    TMATMUL(d, a, b, fixp::keep_acc().row_max(row_out));
  });
#elif defined(ROWMAX_INIT)
  buf_t<__half, float> buf;
  run_matmul<__half, float>(buf.a, buf.b, buf.d, [&](auto &d, auto &a, auto &b) {
    row_max_tile_t<TM> row_in; load_aux(row_in, buf.aux);
    row_max_tile_t<TM> row_out;
    TMATMUL(d, a, b, fixp::keep_acc().row_max(row_in, row_out));
  });
#elif defined(GROUPMAX_8)
  buf_t<__half, float> buf;
  run_matmul<__half, float>(buf.a, buf.b, buf.d, [&](auto &d, auto &a, auto &b) {
    group_max_tile_t<TM, (TN + 7) / 8> group_out;
    TMATMUL(d, a, b, fixp::keep_acc().group_max<8>(group_out));
  });
#elif defined(GROUPMAX_16)
  buf_t<__half, float> buf;
  run_matmul<__half, float>(buf.a, buf.b, buf.d, [&](auto &d, auto &a, auto &b) {
    group_max_tile_t<TM, (TN + 15) / 16> group_out;
    TMATMUL(d, a, b, fixp::keep_acc().group_max<16>(group_out));
  });
#elif defined(GROUPMAX_32)
  buf_t<__half, float> buf;
  run_matmul<__half, float>(buf.a, buf.b, buf.d, [&](auto &d, auto &a, auto &b) {
    group_max_tile_t<TM, (TN + 31) / 32> group_out;
    TMATMUL(d, a, b, fixp::keep_acc().group_max<32>(group_out));
  });
#elif defined(GROUPMAX_48)
  buf_t<__half, float> buf;
  run_matmul<__half, float>(buf.a, buf.b, buf.d, [&](auto &d, auto &a, auto &b) {
    group_max_tile_t<TM, (TN + 47) / 48> group_out;
    TMATMUL(d, a, b, fixp::keep_acc().group_max<48>(group_out));
  });
#elif defined(GROUPMAX_64)
  buf_t<__half, float> buf;
  run_matmul<__half, float>(buf.a, buf.b, buf.d, [&](auto &d, auto &a, auto &b) {
    group_max_tile_t<TM, (TN + 63) / 64> group_out;
    TMATMUL(d, a, b, fixp::keep_acc().group_max<64>(group_out));
  });
#elif defined(GROUPMAX_80)
  buf_t<__half, float> buf;
  run_matmul<__half, float>(buf.a, buf.b, buf.d, [&](auto &d, auto &a, auto &b) {
    group_max_tile_t<TM, (TN + 79) / 80> group_out;
    TMATMUL(d, a, b, fixp::keep_acc().group_max<80>(group_out));
  });
#elif defined(GROUPMAX_96)
  buf_t<__half, float> buf;
  run_matmul<__half, float>(buf.a, buf.b, buf.d, [&](auto &d, auto &a, auto &b) {
    group_max_tile_t<TM, (TN + 95) / 96> group_out;
    TMATMUL(d, a, b, fixp::keep_acc().group_max<96>(group_out));
  });
#elif defined(GROUPMAX_112)
  buf_t<__half, float> buf;
  run_matmul<__half, float>(buf.a, buf.b, buf.d, [&](auto &d, auto &a, auto &b) {
    group_max_tile_t<TM, (TN + 111) / 112> group_out;
    TMATMUL(d, a, b, fixp::keep_acc().group_max<112>(group_out));
  });
#elif defined(GROUPMAX_128)
  buf_t<__half, float> buf;
  run_matmul<__half, float>(buf.a, buf.b, buf.d, [&](auto &d, auto &a, auto &b) {
    group_max_tile_t<TM, (TN + 127) / 128> group_out;
    TMATMUL(d, a, b, fixp::keep_acc().group_max<128>(group_out));
  });
#elif defined(ROWGROUP_MAXABS)
  buf_t<__half, float> buf;
  run_matmul<__half, float>(buf.a, buf.b, buf.d, [&](auto &d, auto &a, auto &b) {
    row_max_tile_t<TM> row_in; load_aux(row_in, buf.aux);
    row_max_tile_t<TM> row_out;
    group_max_tile_t<TM, (TN + 7) / 8> group_out;
    TMATMUL(d, a, b, fixp::keep_acc()
        .row_max(row_in, row_out)
        .group_max<8>(group_out)
        .max_abs());
  });
#elif defined(F16_GROUPMAX)
  buf_t<__half, __half> buf;
  run_matmul<__half, __half>(buf.a, buf.b, buf.d, [&](auto &d, auto &a, auto &b) {
    group_max_tile_t<TM, (TN + 15) / 16> group_out;
    TMATMUL(d, a, b, fixp::f16().group_max<16>(group_out));
  });
#elif defined(S8_ROWMAX)
  buf_t<__half, int8_t> buf;
  run_matmul<__half, int8_t>(buf.a, buf.b, buf.d, [&](auto &d, auto &a, auto &b) {
    row_max_tile_t<TM> row_out;
    TMATMUL(d, a, b, fixp::s8(mk_desc(1, 0, 9)).row_max(row_out));
  });

// === operation-family coverage (param-free keep_acc) =====================
// Verifies each op's BSTART.CUBE mnemonic + math operand stream (C / Bias /
// Scale tiles). Mode macros are short labels (BIAS/ACC/MX/GEMV/...) so they
// never collide with the op function names (TMATMUL_BIAS / TGEMV / ...).
#elif defined(BIAS)                       // D = A*B + Bias
  buf_t<__half, float> buf;
  run_matmul<__half, float>(buf.a, buf.b, buf.d,
      [&](auto &tD, auto &tA, auto &tB) {
        bias_tile_t<TN> bias; load_aux(bias, buf.aux);
        TMATMUL_BIAS(tD, tA, tB, bias, fixp::keep_acc());
      });
#elif defined(ACC)                        // D = C + A*B
  buf_t<__half, float> buf;
  run_matmul<__half, float>(buf.a, buf.b, buf.d,
      [&](auto &tD, auto &tA, auto &tB) {
        acc_tile_t<TM, TN> cacc; load_aux(cacc, buf.aux);
        TMATMUL_ACC(tD, cacc, tA, tB, fixp::keep_acc());
      });
#elif defined(ACC_CSCALE)
  buf_t<__half, float> buf;
  run_matmul<__half, float>(buf.a, buf.b, buf.d,
      [&](auto &tD, auto &tA, auto &tB) {
        acc_tile_t<TM, TN> cacc; load_aux(cacc, buf.aux);
        cscale_tile_t<TM, TN> cscale; load_aux(cscale, buf.aux);
        TMATMUL_ACC(tD, cacc, tA, tB, fixp::keep_acc().cscale(cscale));
      });
#elif defined(MX_SCALE0)
  mixed_buf_t<__half, __bf16, float> buf;
  run_matmul_mixed<__half, __bf16, float>(buf.a, buf.b, buf.d,
      [&](auto &tD, auto &tA, auto &tB) {
        TMATMUL_MX(tD, tA, tB, fixp::keep_acc());
      });
#elif defined(MX_SCALE_A)
  mixed_buf_t<__fp8_e4m3, __half, float> buf;
  run_matmul_mixed<__fp8_e4m3, __half, float>(buf.a, buf.b, buf.d,
      [&](auto &tD, auto &tA, auto &tB) {
        scale_a_tile_t<TM, TK> sa; load_aux(sa, buf.aux);
        TMATMUL_MX(tD, tA, sa, tB, fixp::keep_acc());
      });
#elif defined(MX_SCALE_B)
  mixed_buf_t<__bf16, __fp8_e5m2, float> buf;
  run_matmul_mixed<__bf16, __fp8_e5m2, float>(buf.a, buf.b, buf.d,
      [&](auto &tD, auto &tA, auto &tB) {
        scale_b_tile_t<TK, TN> sb; load_aux(sb, buf.aux);
        TMATMUL_MX(tD, tA, tB, sb, fixp::keep_acc());
      });
#elif defined(FIXP_MODE_MX)               // C = (A*ScaleA)*(B*ScaleB)
  buf_t<__fp8_e4m3, float> buf;
  run_matmul<__fp8_e4m3, float>(buf.a, buf.b, buf.d,
      [&](auto &tD, auto &tA, auto &tB) {
        scale_a_tile_t<TM, TK> sa; load_aux(sa, buf.aux);
        scale_b_tile_t<TK, TN> sb; load_aux(sb, buf.aux);
        TMATMUL_MX(tD, tA, sa, tB, sb, fixp::keep_acc());
      });
#elif defined(MXBIAS)                     // D = (A*ScaleA)*(B*ScaleB) + Bias
  buf_t<__fp8_e4m3, float> buf;
  run_matmul<__fp8_e4m3, float>(buf.a, buf.b, buf.d,
      [&](auto &tD, auto &tA, auto &tB) {
        scale_a_tile_t<TM, TK> sa; load_aux(sa, buf.aux);
        scale_b_tile_t<TK, TN> sb; load_aux(sb, buf.aux);
        bias_tile_t<TN> bias; load_aux(bias, buf.aux);
        TMATMUL_MX_BIAS(tD, tA, sa, tB, sb, bias, fixp::keep_acc());
      });
#elif defined(MXACC)                      // D = C + (A*ScaleA)*(B*ScaleB)
  buf_t<__fp8_e4m3, float> buf;
  run_matmul<__fp8_e4m3, float>(buf.a, buf.b, buf.d,
      [&](auto &tD, auto &tA, auto &tB) {
        acc_tile_t<TM, TN> cacc; load_aux(cacc, buf.aux);
        scale_a_tile_t<TM, TK> sa; load_aux(sa, buf.aux);
        scale_b_tile_t<TK, TN> sb; load_aux(sb, buf.aux);
        TMATMUL_MX_ACC(tD, cacc, tA, sa, tB, sb, fixp::keep_acc());
      });
#elif defined(MXACC_CSCALE)
  buf_t<__half, float> buf;
  run_matmul<__half, float>(buf.a, buf.b, buf.d,
      [&](auto &tD, auto &tA, auto &tB) {
        acc_tile_t<TM, TN> cacc; load_aux(cacc, buf.aux);
        cscale_tile_t<TM, TN> cscale; load_aux(cscale, buf.aux);
        TMATMUL_MX_ACC(tD, cacc, tA, tB,
                       fixp::keep_acc().cscale(cscale));
      });
#elif defined(GEMV)                       // D = mtx * vec (M=1)
  buf_t<__half, float> buf;
  run_gemv<__half, float>(buf.a, buf.b, buf.d,
      [&](auto &tD, auto &tMtx, auto &tVec) {
        TGEMV(tD, tMtx, tVec, fixp::keep_acc());
      });
#elif defined(GEMV_BIAS)
  buf_t<__half, float> buf;
  run_gemv<__half, float>(buf.a, buf.b, buf.d,
      [&](auto &tD, auto &tMtx, auto &tVec) {
        bias_tile_t<TN> bias; load_aux(bias, buf.aux);
        TGEMV_BIAS(tD, tMtx, tVec, bias, fixp::keep_acc());
      });
#elif defined(GEMV_ACC)
  buf_t<__half, float> buf;
  run_gemv<__half, float>(buf.a, buf.b, buf.d,
      [&](auto &tD, auto &tMtx, auto &tVec) {
        gemv_acc_tile_t<TN> cacc; load_aux(cacc, buf.aux);
        TGEMV_ACC(tD, cacc, tMtx, tVec, fixp::keep_acc());
      });
#elif defined(GEMV_MX_SCALE0)
  mixed_buf_t<__half, __bf16, float> buf;
  run_gemv_mixed<__half, __bf16, float>(buf.a, buf.b, buf.d,
      [&](auto &tD, auto &tMtx, auto &tVec) {
        TGEMV_MX(tD, tMtx, tVec, fixp::keep_acc());
      });
#elif defined(GEMV_MX_SCALE_A)
  mixed_buf_t<__fp8_e4m3, __half, float> buf;
  run_gemv_mixed<__fp8_e4m3, __half, float>(buf.a, buf.b, buf.d,
      [&](auto &tD, auto &tMtx, auto &tVec) {
        scale_vec_tile_t<TK> svec; load_aux(svec, buf.aux);
        TGEMV_MX(tD, tMtx, tVec, svec, fixp::keep_acc());
      });
#elif defined(GEMV_MX_SCALE_B)
  mixed_buf_t<__bf16, __fp8_e5m2, float> buf;
  run_gemv_mixed<__bf16, __fp8_e5m2, float>(buf.a, buf.b, buf.d,
      [&](auto &tD, auto &tMtx, auto &tVec) {
        scale_b_tile_t<TK, TN> smtx; load_aux(smtx, buf.aux);
        TGEMV_MX(tD, tMtx, smtx, tVec, fixp::keep_acc());
      });
#elif defined(GEMV_MX)
  buf_t<__fp8_e4m3, float> buf;
  run_gemv<__fp8_e4m3, float>(buf.a, buf.b, buf.d,
      [&](auto &tD, auto &tMtx, auto &tVec) {
        scale_b_tile_t<TK, TN> smtx; load_aux(smtx, buf.aux);
        scale_vec_tile_t<TK> svec; load_aux(svec, buf.aux);
        TGEMV_MX(tD, tMtx, smtx, tVec, svec, fixp::keep_acc());
      });
#elif defined(GEMV_MX_BIAS)
  buf_t<__fp8_e4m3, float> buf;
  run_gemv<__fp8_e4m3, float>(buf.a, buf.b, buf.d,
      [&](auto &tD, auto &tMtx, auto &tVec) {
        scale_b_tile_t<TK, TN> smtx; load_aux(smtx, buf.aux);
        scale_vec_tile_t<TK> svec; load_aux(svec, buf.aux);
        bias_tile_t<TN> bias; load_aux(bias, buf.aux);
        TGEMV_MX_BIAS(tD, tMtx, smtx, tVec, svec, bias, fixp::keep_acc());
      });
#elif defined(GEMV_MX_ACC)
  buf_t<__fp8_e4m3, float> buf;
  run_gemv<__fp8_e4m3, float>(buf.a, buf.b, buf.d,
      [&](auto &tD, auto &tMtx, auto &tVec) {
        gemv_acc_tile_t<TN> cacc; load_aux(cacc, buf.aux);
        scale_b_tile_t<TK, TN> smtx; load_aux(smtx, buf.aux);
        scale_vec_tile_t<TK> svec; load_aux(svec, buf.aux);
        TGEMV_MX_ACC(tD, cacc, tMtx, smtx, tVec, svec, fixp::keep_acc());
      });

// === full-options spot-check (s8 scalar quant on non-TMATMUL ops) =========
#elif defined(BIAS_S8)
  buf_t<__half, int8_t> buf;
  run_matmul<__half, int8_t>(buf.a, buf.b, buf.d,
      [&](auto &tD, auto &tA, auto &tB) {
        bias_tile_t<TN> bias; load_aux(bias, buf.aux);
        TMATMUL_BIAS(tD, tA, tB, bias, fixp::s8(mk_desc(1, 0, 9)));
      });
#elif defined(ACC_S8)
  buf_t<__half, int8_t> buf;
  run_matmul<__half, int8_t>(buf.a, buf.b, buf.d,
      [&](auto &tD, auto &tA, auto &tB) {
        acc_tile_t<TM, TN> cacc; load_aux(cacc, buf.aux);
        TMATMUL_ACC(tD, cacc, tA, tB, fixp::s8(mk_desc(1, 0, 9)));
      });
#elif defined(MX_S8)
  buf_t<__fp8_e4m3, int8_t> buf;
  run_matmul<__fp8_e4m3, int8_t>(buf.a, buf.b, buf.d,
      [&](auto &tD, auto &tA, auto &tB) {
        scale_a_tile_t<TM, TK> sa; load_aux(sa, buf.aux);
        scale_b_tile_t<TK, TN> sb; load_aux(sb, buf.aux);
        TMATMUL_MX(tD, tA, sa, tB, sb, fixp::s8(mk_desc(1, 0, 9)));
      });
#elif defined(MXBIAS_S8)
  buf_t<__fp8_e4m3, int8_t> buf;
  run_matmul<__fp8_e4m3, int8_t>(buf.a, buf.b, buf.d,
      [&](auto &tD, auto &tA, auto &tB) {
        scale_a_tile_t<TM, TK> sa; load_aux(sa, buf.aux);
        scale_b_tile_t<TK, TN> sb; load_aux(sb, buf.aux);
        bias_tile_t<TN> bias; load_aux(bias, buf.aux);
        TMATMUL_MX_BIAS(tD, tA, sa, tB, sb, bias,
                        fixp::s8(mk_desc(1, 0, 9)));
      });
#elif defined(MXACC_S8)
  buf_t<__fp8_e4m3, int8_t> buf;
  run_matmul<__fp8_e4m3, int8_t>(buf.a, buf.b, buf.d,
      [&](auto &tD, auto &tA, auto &tB) {
        acc_tile_t<TM, TN> cacc; load_aux(cacc, buf.aux);
        scale_a_tile_t<TM, TK> sa; load_aux(sa, buf.aux);
        scale_b_tile_t<TK, TN> sb; load_aux(sb, buf.aux);
        TMATMUL_MX_ACC(tD, cacc, tA, sa, tB, sb,
                       fixp::s8(mk_desc(1, 0, 9)));
      });
#elif defined(GEMV_S8)
  buf_t<__half, int8_t> buf;
  run_gemv<__half, int8_t>(buf.a, buf.b, buf.d,
      [&](auto &tD, auto &tMtx, auto &tVec) {
        TGEMV(tD, tMtx, tVec, fixp::s8(mk_desc(1, 0, 9)));
      });
#elif defined(GEMV_BIAS_S8)
  buf_t<__half, int8_t> buf;
  run_gemv<__half, int8_t>(buf.a, buf.b, buf.d,
      [&](auto &tD, auto &tMtx, auto &tVec) {
        bias_tile_t<TN> bias; load_aux(bias, buf.aux);
        TGEMV_BIAS(tD, tMtx, tVec, bias,
                   fixp::s8(mk_desc(1, 0, 9)));
      });
#elif defined(GEMV_ACC_S8)
  buf_t<__half, int8_t> buf;
  run_gemv<__half, int8_t>(buf.a, buf.b, buf.d,
      [&](auto &tD, auto &tMtx, auto &tVec) {
        gemv_acc_tile_t<TN> cacc; load_aux(cacc, buf.aux);
        TGEMV_ACC(tD, cacc, tMtx, tVec,
                  fixp::s8(mk_desc(1, 0, 9)));
      });
#elif defined(GEMV_MX_S8)
  buf_t<__fp8_e4m3, int8_t> buf;
  run_gemv<__fp8_e4m3, int8_t>(buf.a, buf.b, buf.d,
      [&](auto &tD, auto &tMtx, auto &tVec) {
        scale_b_tile_t<TK, TN> smtx; load_aux(smtx, buf.aux);
        scale_vec_tile_t<TK> svec; load_aux(svec, buf.aux);
        TGEMV_MX(tD, tMtx, smtx, tVec, svec, fixp::s8(mk_desc(1, 0, 9)));
      });
#elif defined(GEMV_MX_BIAS_S8)
  buf_t<__fp8_e4m3, int8_t> buf;
  run_gemv<__fp8_e4m3, int8_t>(buf.a, buf.b, buf.d,
      [&](auto &tD, auto &tMtx, auto &tVec) {
        scale_b_tile_t<TK, TN> smtx; load_aux(smtx, buf.aux);
        scale_vec_tile_t<TK> svec; load_aux(svec, buf.aux);
        bias_tile_t<TN> bias; load_aux(bias, buf.aux);
        TGEMV_MX_BIAS(tD, tMtx, smtx, tVec, svec, bias,
                      fixp::s8(mk_desc(1, 0, 9)));
      });
#elif defined(GEMV_MX_ACC_S8)
  buf_t<__fp8_e4m3, int8_t> buf;
  run_gemv<__fp8_e4m3, int8_t>(buf.a, buf.b, buf.d,
      [&](auto &tD, auto &tMtx, auto &tVec) {
        gemv_acc_tile_t<TN> cacc; load_aux(cacc, buf.aux);
        scale_b_tile_t<TK, TN> smtx; load_aux(smtx, buf.aux);
        scale_vec_tile_t<TK> svec; load_aux(svec, buf.aux);
        TGEMV_MX_ACC(tD, cacc, tMtx, smtx, tVec, svec,
                     fixp::s8(mk_desc(1, 0, 9)));
      });

// === Cooperative Shared A/B (B.IOS) =======================================
#elif defined(SHARED)                      // four-PE Shared A/B TMATMUL
  buf_t<__half, float> buf;
  run_matmul_shared<float>(buf.a, buf.b, buf.d,
      [&](auto &tD, auto &tA, auto &tB) {
        TMATMUL(tD, tA, tB, fixp::keep_acc());
      });
#elif defined(S8_SHARED)                   // Shared A/B + s8 scalar quant
  buf_t<__half, int8_t> buf;
  run_matmul_shared<int8_t>(buf.a, buf.b, buf.d,
      [&](auto &tD, auto &tA, auto &tB) {
        TMATMUL(tD, tA, tB, fixp::s8(mk_desc(1, 0, 9)));
      });
#elif defined(SHARED_ACC)                  // Shared A/B + local accumulator C
  buf_t<__half, float> buf;
  run_matmul_shared<float>(buf.a, buf.b, buf.d,
      [&](auto &tD, auto &tA, auto &tB) {
        acc_tile_t<TM, TN> cacc; load_aux(cacc, buf.aux);
        TMATMUL_ACC(tD, cacc, tA, tB, fixp::keep_acc());
      });
#elif defined(SHARED_BIAS)                 // Shared A/B + local bias
  buf_t<__half, float> buf;
  run_matmul_shared<float>(buf.a, buf.b, buf.d,
      [&](auto &tD, auto &tA, auto &tB) {
        bias_tile_t<TN> bias; load_aux(bias, buf.aux);
        TMATMUL_BIAS(tD, tA, tB, bias, fixp::keep_acc());
      });
#elif defined(SHARED_ROWMAX)               // Shared A/B + RowMax output
  buf_t<__half, float> buf;
  run_matmul_shared<float>(buf.a, buf.b, buf.d,
      [&](auto &tD, auto &tA, auto &tB) {
        // Cooperative Shared-A exposes the Core-total M dimension.
        row_max_tile_t<4 * TM> row_out;
        TMATMUL(tD, tA, tB, fixp::keep_acc().row_max(row_out));
      });
#elif defined(SHARED_GROUPMAX_8)            // Shared A/B + GroupMax<8> output
  buf_t<__half, float> buf;
  run_matmul_shared<float>(buf.a, buf.b, buf.d,
      [&](auto &tD, auto &tA, auto &tB) {
        group_max_tile_t<4 * TM, (TN + 7) / 8> group_out;
        TMATMUL(tD, tA, tB,
                fixp::keep_acc().group_max<8>(group_out));
      });
#elif defined(SHARED_ROWMAX_GROUPMAX_8)    // Shared A/B + RowMax + GroupMax<8>
  buf_t<__half, float> buf;
  run_matmul_shared<float>(buf.a, buf.b, buf.d,
      [&](auto &tD, auto &tA, auto &tB) {
        // Cooperative Shared-A exposes the Core-total M dimension.
        row_max_tile_t<4 * TM> row_out;
        group_max_tile_t<4 * TM, (TN + 7) / 8> group_out;
        TMATMUL(tD, tA, tB, fixp::keep_acc()
            .row_max(row_out)
            .group_max<8>(group_out));
      });
#elif defined(SHARED_KEEP_ACC_RELU)        // Shared A/B + keep_acc + ReLU
  buf_t<__half, float> buf;
  run_matmul_shared<float>(buf.a, buf.b, buf.d,
      [&](auto &tD, auto &tA, auto &tB) {
        TMATMUL(tD, tA, tB, fixp::keep_acc().relu());
      });
#elif defined(SHARED_F16)                   // Shared A/B + f16 destination
  buf_t<__half, __half> buf;
  run_matmul_shared<__half>(buf.a, buf.b, buf.d,
      [&](auto &tD, auto &tA, auto &tB) {
        TMATMUL(tD, tA, tB, fixp::f16());
      });
#elif defined(SHARED_F16_RELU)              // Shared A/B + f16 + ReLU
  buf_t<__half, __half> buf;
  run_matmul_shared<__half>(buf.a, buf.b, buf.d,
      [&](auto &tD, auto &tA, auto &tB) {
        TMATMUL(tD, tA, tB, fixp::f16().relu());
      });
#elif defined(SHARED_BF16)                  // Shared A/B + bf16 destination
  buf_t<__half, __bf16> buf;
  run_matmul_shared<__bf16>(buf.a, buf.b, buf.d,
      [&](auto &tD, auto &tA, auto &tB) {
        TMATMUL(tD, tA, tB, fixp::bf16());
      });
#elif defined(SHARED_BF16_RELU)             // Shared A/B + bf16 + ReLU
  buf_t<__half, __bf16> buf;
  run_matmul_shared<__bf16>(buf.a, buf.b, buf.d,
      [&](auto &tD, auto &tA, auto &tB) {
        TMATMUL(tD, tA, tB, fixp::bf16().relu());
      });
#elif defined(SHARED_S8_RELU)               // Shared A/B + s8 scalar quant + ReLU
  buf_t<__half, int8_t> buf;
  run_matmul_shared<int8_t>(buf.a, buf.b, buf.d,
      [&](auto &tD, auto &tA, auto &tB) {
        TMATMUL(tD, tA, tB, fixp::s8(mk_desc(1, 0, 9)).relu());
      });
#elif defined(SHARED_S8_LRELU)              // Shared A/B + s8 + LReLU
  buf_t<__half, int8_t> buf;
  run_matmul_shared<int8_t>(buf.a, buf.b, buf.d,
      [&](auto &tD, auto &tA, auto &tB) {
        TMATMUL(tD, tA, tB, fixp::s8(mk_desc(1, 0, 9)).lrelu(1));
      });
#elif defined(SHARED_V_S8_RELU)             // Shared A/B + vector s8 quant + ReLU
  buf_t<__half, int8_t> buf;
  run_matmul_shared<int8_t>(buf.a, buf.b, buf.d,
      [&](auto &tD, auto &tA, auto &tB) {
        par_tile_t<TN> quant; load_aux(quant, buf.aux);
        TMATMUL(tD, tA, tB, fixp::s8(quant).relu());
      });
#elif defined(SHARED_F16_PRELU)             // Shared A/B + f16 + PReLU
  buf_t<__half, __half> buf;
  run_matmul_shared<__half>(buf.a, buf.b, buf.d,
      [&](auto &tD, auto &tA, auto &tB) {
        par_tile_t<TN> prelu; load_aux(prelu, buf.aux);
        TMATMUL(tD, tA, tB, fixp::f16().prelu(prelu));
      });
#elif defined(SHARED_S8_PRELU)              // Shared A/B + s8 + PReLU
  buf_t<__half, int8_t> buf;
  run_matmul_shared<int8_t>(buf.a, buf.b, buf.d,
      [&](auto &tD, auto &tA, auto &tB) {
        par_tile_t<TN> prelu; load_aux(prelu, buf.aux);
        TMATMUL(tD, tA, tB, fixp::s8(mk_desc(1, 0, 9)).prelu(prelu));
      });
#elif defined(SHARED_ROWMAX_INIT)           // Shared A/B + RowMax(in,out)
  // NOTE: compiles (RowIn==RowOut==EffectiveM=4*TM, the cooperative group_M),
  // but gfrun rejects the shared RowMaxIn form: the model asserts RowMaxIn
  // must be a Local Mx1 tile with validRow=min(16,lb0)<=32, while the
  // compiler's static_assert demands RowIn.ValidRow==EffectiveM=4*TM. No
  // dimension satisfies both — a toolchain/model gap for shared RowMaxInit.
  buf_t<__half, float> buf;
  run_matmul_shared<float>(buf.a, buf.b, buf.d,
      [&](auto &tD, auto &tA, auto &tB) {
        row_max_tile_t<4 * TM> row_in; load_aux(row_in, buf.aux);
        row_max_tile_t<4 * TM> row_out;
        TMATMUL(tD, tA, tB, fixp::keep_acc().row_max(row_in, row_out));
      });
#elif defined(SHARED_GROUPMAX_16)           // Shared A/B + GroupMax<16>
  buf_t<__half, float> buf;
  run_matmul_shared<float>(buf.a, buf.b, buf.d,
      [&](auto &tD, auto &tA, auto &tB) {
        group_max_tile_t<4 * TM, (TN + 15) / 16> group_out;
        TMATMUL(tD, tA, tB, fixp::keep_acc().group_max<16>(group_out));
      });
#elif defined(SHARED_GROUPMAX_32)           // Shared A/B + GroupMax<32>
  buf_t<__half, float> buf;
  run_matmul_shared<float>(buf.a, buf.b, buf.d,
      [&](auto &tD, auto &tA, auto &tB) {
        group_max_tile_t<4 * TM, (TN + 31) / 32> group_out;
        TMATMUL(tD, tA, tB, fixp::keep_acc().group_max<32>(group_out));
      });
#elif defined(SHARED_GROUPMAX_48)           // Shared A/B + GroupMax<48>
  buf_t<__half, float> buf;
  run_matmul_shared<float>(buf.a, buf.b, buf.d,
      [&](auto &tD, auto &tA, auto &tB) {
        group_max_tile_t<4 * TM, (TN + 47) / 48> group_out;
        TMATMUL(tD, tA, tB, fixp::keep_acc().group_max<48>(group_out));
      });
#elif defined(SHARED_GROUPMAX_64)           // Shared A/B + GroupMax<64>
  buf_t<__half, float> buf;
  run_matmul_shared<float>(buf.a, buf.b, buf.d,
      [&](auto &tD, auto &tA, auto &tB) {
        group_max_tile_t<4 * TM, (TN + 63) / 64> group_out;
        TMATMUL(tD, tA, tB, fixp::keep_acc().group_max<64>(group_out));
      });
#elif defined(SHARED_GROUPMAX_80)           // Shared A/B + GroupMax<80>
  buf_t<__half, float> buf;
  run_matmul_shared<float>(buf.a, buf.b, buf.d,
      [&](auto &tD, auto &tA, auto &tB) {
        group_max_tile_t<4 * TM, (TN + 79) / 80> group_out;
        TMATMUL(tD, tA, tB, fixp::keep_acc().group_max<80>(group_out));
      });
#elif defined(SHARED_GROUPMAX_96)           // Shared A/B + GroupMax<96>
  buf_t<__half, float> buf;
  run_matmul_shared<float>(buf.a, buf.b, buf.d,
      [&](auto &tD, auto &tA, auto &tB) {
        group_max_tile_t<4 * TM, (TN + 95) / 96> group_out;
        TMATMUL(tD, tA, tB, fixp::keep_acc().group_max<96>(group_out));
      });
#elif defined(SHARED_GROUPMAX_112)          // Shared A/B + GroupMax<112>
  buf_t<__half, float> buf;
  run_matmul_shared<float>(buf.a, buf.b, buf.d,
      [&](auto &tD, auto &tA, auto &tB) {
        group_max_tile_t<4 * TM, (TN + 111) / 112> group_out;
        TMATMUL(tD, tA, tB, fixp::keep_acc().group_max<112>(group_out));
      });
#elif defined(SHARED_GROUPMAX_128)          // Shared A/B + GroupMax<128>
  buf_t<__half, float> buf;
  run_matmul_shared<float>(buf.a, buf.b, buf.d,
      [&](auto &tD, auto &tA, auto &tB) {
        group_max_tile_t<4 * TM, (TN + 127) / 128> group_out;
        TMATMUL(tD, tA, tB, fixp::keep_acc().group_max<128>(group_out));
      });
#elif defined(SHARED_ROWGROUP_MAXABS)       // Shared A/B + RowMax + GroupMax<8> + MaxAbs
  // NOTE: same shared-RowMaxInit gap as SHARED_ROWMAX_INIT — compiles at
  // 4*TM but gfrun rejects the shared RowMaxIn (Local Mx1 vs cooperative
  // EffectiveM contradiction). FPATR encoding is still valid.
  buf_t<__half, float> buf;
  run_matmul_shared<float>(buf.a, buf.b, buf.d,
      [&](auto &tD, auto &tA, auto &tB) {
        row_max_tile_t<4 * TM> row_in; load_aux(row_in, buf.aux);
        row_max_tile_t<4 * TM> row_out;
        group_max_tile_t<4 * TM, (TN + 7) / 8> group_out;
        TMATMUL(tD, tA, tB, fixp::keep_acc()
            .row_max(row_in, row_out)
            .group_max<8>(group_out)
            .max_abs());
      });
#elif defined(SHARED_F16_GROUPMAX)           // Shared A/B + f16 + GroupMax<16>
  // NOTE: compiles + FPATR-valid, but gfrun rejects the cooperative collective
  // when a non-keep_acc PreQuant (f16/s8 destination) is combined with a
  // max-reduction (RowMax/GroupMax). The single-PE twin passes, so this is a
  // cooperative-path model gap, not a test-code bug. keep_acc+GroupMax on
  // shared (shared_groupmax_16) works; only the f16/s8 PreQuant breaks it.
  buf_t<__half, __half> buf;
  run_matmul_shared<__half>(buf.a, buf.b, buf.d,
      [&](auto &tD, auto &tA, auto &tB) {
        group_max_tile_t<4 * TM, (TN + 15) / 16> group_out;
        TMATMUL(tD, tA, tB, fixp::f16().group_max<16>(group_out));
      });
#elif defined(SHARED_S8_ROWMAX)              // Shared A/B + s8 + RowMax
  // NOTE: same cooperative+PreQuant+max-reduction gap as SHARED_F16_GROUPMAX.
  // Compiles + FPATR-valid; single-PE twin passes; gfrun rejects the
  // collective on the cooperative path. keep_acc+RowMax (shared_rowmax) works.
  buf_t<__half, int8_t> buf;
  run_matmul_shared<int8_t>(buf.a, buf.b, buf.d,
      [&](auto &tD, auto &tA, auto &tB) {
        row_max_tile_t<4 * TM> row_out;
        TMATMUL(tD, tA, tB, fixp::s8(mk_desc(1, 0, 9)).row_max(row_out));
      });
#elif defined(SHARED_ACC_CSCALE)             // Shared A/B + ACC + CScale
  buf_t<__half, float> buf;
  run_matmul_shared<float>(buf.a, buf.b, buf.d,
      [&](auto &tD, auto &tA, auto &tB) {
        acc_tile_t<TM, TN> cacc; load_aux(cacc, buf.aux);
        cscale_tile_t<TM, TN> cscale; load_aux(cscale, buf.aux);
        TMATMUL_ACC(tD, cacc, tA, tB, fixp::keep_acc().cscale(cscale));
      });
#elif defined(SHARED_NOSTORE_ROWMAX)      // [TEMP] Shared TLOAD + TMATMUL row_max, no TSTORE
  buf_t<__half, float> buf;
  run_matmul_shared_nostore<float>(buf.a, buf.b,
      [&](auto &tD, auto &tA, auto &tB) {
        row_max_tile_t<4 * TM> row_out;
        TMATMUL(tD, tA, tB, fixp::keep_acc().row_max(row_out));
      });
#elif defined(SHARED_NOSTORE_GROUPMAX)    // [TEMP] Shared TLOAD + TMATMUL group_max<8>, no TSTORE
  buf_t<__half, float> buf;
  run_matmul_shared_nostore<float>(buf.a, buf.b,
      [&](auto &tD, auto &tA, auto &tB) {
        group_max_tile_t<4 * TM, (TN + 7) / 8> group_out;
        TMATMUL(tD, tA, tB, fixp::keep_acc().group_max<8>(group_out));
      });
#elif defined(SHARED_NOSTORE_ROWGROUPMAX) // [TEMP] Shared TLOAD + TMATMUL row_max+group_max<8>, no TSTORE
  buf_t<__half, float> buf;
  run_matmul_shared_nostore<float>(buf.a, buf.b,
      [&](auto &tD, auto &tA, auto &tB) {
        row_max_tile_t<4 * TM> row_out;
        group_max_tile_t<4 * TM, (TN + 7) / 8> group_out;
        TMATMUL(tD, tA, tB, fixp::keep_acc()
            .row_max(row_out)
            .group_max<8>(group_out));
      });
#elif defined(TRANS_A)
  buf_t<__half, float> buf;
  run_matmul_shared_transpose<true, false, float>(buf.a, buf.b, buf.d,
      [&](auto &tD, auto &tA, auto &tB) {
        TMATMUL(tD, tA, tB, fixp::keep_acc().transpose_a());
      });
#elif defined(TRANS_B)
  buf_t<__half, float> buf;
  run_matmul_shared_transpose<false, true, float>(buf.a, buf.b, buf.d,
      [&](auto &tD, auto &tA, auto &tB) {
        TMATMUL(tD, tA, tB, fixp::keep_acc().transpose_b());
      });
#elif defined(TRANS_AB)
  buf_t<__half, float> buf;
  run_matmul_shared_transpose<true, true, float>(buf.a, buf.b, buf.d,
      [&](auto &tD, auto &tA, auto &tB) {
        TMATMUL(tD, tA, tB,
                fixp::keep_acc().transpose_a().transpose_b());
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
  run_matmul<__half, int8_t>(buf.a, buf.b, buf.d,
      [&](auto &tD, auto &tA, auto &tB) {
        par_tile_t<TN> quant, prelu;
        load_aux(quant, buf.aux);
        load_aux(prelu, buf.aux);
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

#ifdef RES_CHECK
  return g_numeric_failure;
#else
  return 0;
#endif
}
