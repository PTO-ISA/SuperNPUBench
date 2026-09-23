#pragma once

// Dynamic-shape sibling of conv2d_img2col.hpp: the same NCHW-in / NHWC-out
// convolution over the same PTO v0.58.6 convolution-dedicated transports
// (TIMG2COL_SPART<DN2ND> + weight TLOAD<OIHW2NK> + cooperative Shared
// TMATMUL + TSTORE_CUBE), but with every shape number that the transports
// accept through GPR operands moved from template parameters to a runtime
// tiling descriptor:
//
//   * TIMG2COLParams param0/param1/param2 (H, W, Cin, KH, KW, pads,
//     dilations, strides, rowStart) are packed at runtime from tiling[];
//   * the weight TLOAD shape/start words (cin, cout, kernel, n_start) are
//     packed at runtime (the API already carries ValidK/ValidN in GPRs);
//   * the M-block / N-block loop bounds (Ho*Wo/GroupM, Co/TileN) are
//     runtime loop-trip counts;
//   * every GM access is addressed at runtime: the tensors are
//     global_tensor<float, RowMajor<-1,-1>> constructed with runtime
//     (rows, cols), and each output block store rebases the pointer
//     arithmetically (the static global_iterator is unusable here because
//     it requires compile-time RowStride).
//
// What must stay compile-time, and why (TileOP API 05f71fd surface):
//   * K: the cooperative Shared-Left x Shared-Right TMATMUL resolves
//     M/N/K from the operands' compile-time valid dimensions
//     (resolve_matmul_shape rejects DYNAMIC valid shapes; N and K are
//     encoded as B.DIM immediates), and TIMG2COL_SPART likewise encodes
//     ValidRow/ValidCol/TotalCol as immediates. K is therefore a
//     capacity property of the two Shared tiles, passed as the KPhys
//     template parameter. The runtime shape must produce exactly
//         K = KH*KW*ceil(Cin/c0)*c0 == KPhys  (c0 = 256/elem_bits)
//     which is guard-checked below. One instantiation serves every
//     (Cin, KH, KW) bucket with that K: e.g. KPhys=32 admits 2x2 kernels
//     with Cin<=8, 1x2/2x1 kernels with 9<=Cin<=16, and 1x1 kernels with
//     25<=Cin<=32, at any H/W/stride/pad/dilation.
//   * GroupM / TileN: CUBE row-block geometry of the cooperative
//     TMATMUL (LB0 = group_M comes from the Shared A type's ValidRow).
//
// The static version's static_asserts that only constrain the
// compile-time tile geometry are kept as static_asserts (they are
// properties of KPhys/GroupM/TileN); the shape-dependent ones become
// runtime guards that fail loudly (printf + trap, the TileOP runtime
// guard convention):
//   * positive dims, non-negative pads, stride/dilation >= 1;
//   * param-word field widths (H/W/Cin/Co < 2^16, KH/KW < 2^8,
//     pads < 2^8, dilations < 2^5, strides < 2^6);
//   * Ho > 0 and Wo > 0 (padded feature map covers the receptive field,
//     non-empty output);
//   * K == KPhys (which also pins K power-of-two and c0-aligned);
//   * Ho*Wo % GroupM == 0 and Co % TileN == 0 (full tiles only: the
//     tail-row/tail-column handling would need dynamic valid shapes on
//     the TMATMUL operands, which the API rejects; identical constraint
//     to the static version).
//
// The known llvm movr S-reg aliasing hazard is inherited unchanged from
// the static version: the Shared handles defined by the PE0-only
// publications are given empty "=Sr" asm definitions on the skipping
// path so the register allocator never inserts GPR->Shared copies
// (c.movr cannot encode the S file; llvm >= 679005ba1294 fails loudly
// instead of silently clobbering sp - issue #205).
//
// tiling descriptor layout (int64_t words, one NCHW plane set / batch):
//   tiling[0]  = Cin       tiling[7]  = StrideW
//   tiling[1]  = InHeight   tiling[8]  = PadTop
//   tiling[2]  = InWidth    tiling[9]  = PadBottom
//   tiling[3]  = OutChannels tiling[10] = PadLeft
//   tiling[4]  = KernelH    tiling[11] = PadRight
//   tiling[5]  = KernelW    tiling[12] = DilH
//   tiling[6]  = StrideH    tiling[13] = DilW
//
// Requires the same toolchain floor as conv2d_img2col.hpp (llvm >= 69d6e6d8
// + the 679005ba1294 movr Shared fix, TileOP-API >= f7c4a45).

#include <common/pto_tileop.hpp>
#include <cstdint>
#include <type_traits>

namespace supernpu::conv2d_img2col {

using namespace pto;

namespace detail {

constexpr bool is_pow2(int v) { return v > 0 && (v & (v - 1)) == 0; }

}  // namespace detail

// NCHW in, NHWC out convolution, runtime shapes. See the file header for
// the tiling layout and the compile-time/runtime split.
//
//   output_nhwc: [Ho][Wo][Co] row major (one batch plane)
//   input_nchw:  [Cin][H][W] row major (one batch plane)
//   weight_oihw: [Co][Cin][KH][KW] row major (PyTorch conv2d weight)
//   tiling:      14-word shape descriptor (see file header)
template <int KPhys, int GroupM = 64, int TileN = 16>
void conv2d_img2col_dyn(float *output_nhwc, float *input_nchw,
                        float *weight_oihw, const int64_t *tiling) {
  constexpr int kPeNum = 4;
  constexpr int kElemBits = 32;
  constexpr int kC0 = 256 / kElemBits;
  // Cooperative TMATMUL per-PE rows: group_M <= 64 -> 16, else 32.
  constexpr int kLocalM = GroupM <= 64 ? 16 : 32;

  static_assert(detail::is_pow2(KPhys) && KPhys % kC0 == 0,
                "KPhys must be a power of two and a multiple of the C0 "
                "channel carrier (cooperative TMATMUL validates pow2 K; "
                "the transports require C0-aligned K windows)");
  static_assert(GroupM == 64 || GroupM == 128,
                "GroupM must be 64 (16 rows/PE) or 128 (32 rows/PE): the "
                "output row-block map assumes full four-PE groups");
  static_assert(detail::is_pow2(TileN), "TileN must be a power of two");
  static_assert(detail::is_pow2(GroupM * KPhys * 4) &&
                    GroupM * KPhys * 4 >= 128 && GroupM * KPhys * 4 <= 256 * 1024,
                "Shared im2col A capacity must be a power-of-two SizeCode");
  static_assert(detail::is_pow2(TileN * KPhys * 4) &&
                    TileN * KPhys * 4 >= 128 && TileN * KPhys * 4 <= 256 * 1024,
                "Shared weight capacity must be a power-of-two SizeCode");

  // ---- runtime shape and derived quantities ----
  const int64_t ci = tiling[0];
  const int64_t hi = tiling[1];
  const int64_t wi = tiling[2];
  const int64_t co = tiling[3];
  const int64_t kh = tiling[4];
  const int64_t kw = tiling[5];
  const int64_t sh = tiling[6];
  const int64_t sw = tiling[7];
  const int64_t pt = tiling[8];
  const int64_t pb = tiling[9];
  const int64_t pl = tiling[10];
  const int64_t pr = tiling[11];
  const int64_t dh = tiling[12];
  const int64_t dw = tiling[13];

  const int64_t C1 = (ci + kC0 - 1) / kC0;
  const int64_t K = kh * kw * C1 * kC0;
  const int64_t EffH = (kh - 1) * dh + 1;
  const int64_t EffW = (kw - 1) * dw + 1;
  const int64_t Ho = (hi + pt + pb - EffH) / sh + 1;
  const int64_t Wo = (wi + pl + pr - EffW) / sw + 1;
  const int64_t M = Ho * Wo;
  const int64_t Mb = M / GroupM;
  const int64_t Nb = co / TileN;

  // ---- runtime guards (the static version's static_asserts) ----
  // Fail loudly on the first violated contract; gfrun surfaces the printf
  // before the trap aborts the run.
  if (ci <= 0 || hi <= 0 || wi <= 0 || co <= 0 || kh <= 0 || kw <= 0 ||
      pt < 0 || pb < 0 || pl < 0 || pr < 0 || sh < 1 || sw < 1 ||
      dh < 1 || dw < 1) {
    __builtin_printf(
        "conv2d_img2col_dyn: dims must be positive and pads/strides/dilations "
        "non-negative (stride/dilation >= 1)\n");
    __builtin_trap();
  }
  // Param-word field widths: H/W/Cin/Co are 16-bit, KH/KW 8-bit, pads
  // 8-bit, dilations 5-bit, strides 6-bit (PTO-BSTART-TIMG2COL-PARAMS-001
  // and the weight TLOAD shape word).
  if (hi > 0xffff || wi > 0xffff || ci > 0xffff || co > 0xffff ||
      kh > 0xff || kw > 0xff || pt > 0xff || pb > 0xff || pl > 0xff ||
      pr > 0xff || dh > 0x1f || dw > 0x1f || sh > 0x3f || sw > 0x3f) {
    __builtin_printf(
        "conv2d_img2col_dyn: shape exceeds the param-word field widths\n");
    __builtin_trap();
  }
  if (Ho <= 0 || Wo <= 0) {
    __builtin_printf(
        "conv2d_img2col_dyn: padded feature map is smaller than the "
        "receptive field (empty convolution output)\n");
    __builtin_trap();
  }
  if (M > 0x7fffffff || M * co > 0x7fffffff) {
    __builtin_printf(
        "conv2d_img2col_dyn: Ho*Wo exceeds the 31-bit GM addressing range\n");
    __builtin_trap();
  }
  if (K != KPhys) {
    __builtin_printf(
        "conv2d_img2col_dyn: runtime K = KH*KW*ceil(Cin/c0)*c0 = %lld does "
        "not match the compiled KPhys = %d; instantiate conv2d_img2col_dyn "
        "with KPhys = %lld\n",
        static_cast<long long>(K), KPhys, static_cast<long long>(K));
    __builtin_trap();
  }
  if (M % GroupM != 0) {
    __builtin_printf(
        "conv2d_img2col_dyn: Ho*Wo = %lld is not a multiple of GroupM = %d "
        "(tail row groups need dynamic TMATMUL valid shapes, unsupported)\n",
        static_cast<long long>(M), GroupM);
    __builtin_trap();
  }
  if (co % TileN != 0) {
    __builtin_printf(
        "conv2d_img2col_dyn: Co = %lld is not a multiple of TileN = %d\n",
        static_cast<long long>(co), TileN);
    __builtin_trap();
  }

  // TIMG2COL parameter words (PTO-BSTART-TIMG2COL-PARAMS-001), packed at
  // runtime (the wrapper keeps them in GPRs):
  //   param0: inputH[15:0] inputW[31:16] cin[47:32] kernelH[55:48]
  //           kernelW[63:56]
  //   param1: padTop[7:0] padLeft[15:8] padBottom[23:16] padRight[31:24]
  //           dilationH[36:32] dilationW[41:37] strideH[47:42] strideW[53:48]
  //   param2: rowStart[31:0] colStart[63:32]
  const uint64_t kParam0 =
      static_cast<uint64_t>(hi) |
      (static_cast<uint64_t>(wi) << 16) |
      (static_cast<uint64_t>(ci) << 32) |
      (static_cast<uint64_t>(kh) << 48) |
      (static_cast<uint64_t>(kw) << 56);
  const uint64_t kParam1 =
      static_cast<uint64_t>(pt) |
      (static_cast<uint64_t>(pl) << 8) |
      (static_cast<uint64_t>(pb) << 16) |
      (static_cast<uint64_t>(pr) << 24) |
      (static_cast<uint64_t>(dh) << 32) |
      (static_cast<uint64_t>(dw) << 37) |
      (static_cast<uint64_t>(sh) << 42) |
      (static_cast<uint64_t>(sw) << 48);

  // Dynamic GM descriptors: only data() (and, for the store, the runtime
  // row stride GetStrideBytes(3) = Co*4) feed the transports.
  using gmA = global_tensor<float, RowMajor<-1, -1>>;
  using gmW = global_tensor<float, RowMajor<-1, -1>>;
  using gmC = global_tensor<float, RowMajor<-1, -1>>;

  // Shared im2col A matrix: the complete [GroupM, KPhys] group, published
  // by PE0 through the single-issuer TIMG2COL variant (ValidRow doubles as
  // the cooperative TMATMUL LB0 group_M).
  using tileA = SharedTile<SharedMatrixLeft<float, GroupM, KPhys>>;
  // Shared weight B: physical RowMajor [N, K] (TransB=0 publishes the
  // logical [K, N] B operand, pto-spec #257).
  using tileW = SharedTile<SharedMatrixRight<float, TileN, KPhys>>;
  // Per-PE CUBE row slice of the [GroupM, TileN] output block.
  using tileC16 = CubeAccumulatorM16<float, kLocalM, TileN>;
  using tileC32 = CubeAccumulatorM32<float, kLocalM, TileN>;
  using tileC = std::conditional_t<(kLocalM <= 16), tileC16, tileC32>;

  const uint32_t tid = get_thread_idx();

  for (int64_t i = 0; i < Mb; ++i) {
    for (int64_t j = 0; j < Nb; ++j) {
      tileC tC;
      // Declared outside the branch: every PE's TMATMUL binds the same
      // compile-time Shared slots; only PE0 defines them.
      tileA tA;
      tileW tW;
      // Singleton Shared publications run on PE0 only; PEs 1..3 proceed
      // straight to the cooperative TMATMUL rendezvous (matched by slot
      // id and dimensions, not by program counter).
      if (tid == 0) {
        gmA gA(input_nchw, static_cast<int>(ci), static_cast<int>(hi * wi));
        gmW gW(weight_oihw, static_cast<int>(co),
               static_cast<int>(ci * kh * kw));
        // im2col rows [i*GroupM, (i+1)*GroupM) of the NCHW planes; one
        // B.IOS publishes the complete [GroupM, KPhys] Shared parent.
        TIMG2COL_SPART<DN2ND>(
            tA, gA,
            TIMG2COLParams{kParam0, kParam1,
                           static_cast<uint64_t>(i * GroupM)},
            /*PEMask=*/8);
        // Weight block j: the OIHW [j*TileN, (j+1)*TileN) x [0, KPhys)
        // window; the shape/start words are packed from the runtime
        // tiling (cin/cout/kernel ride GPRs, issue #195 contract).
        TLOAD<OIHW2NK, /*PEMask=*/8>(
            tW, gW,
            make_weight_tload_params(
                static_cast<uint16_t>(ci), static_cast<uint16_t>(co),
                static_cast<uint8_t>(kh), static_cast<uint8_t>(kw),
                /*n_start=*/static_cast<uint32_t>(j * TileN)));
      } else {
        // The two Shared handles are defined by the PE0-only
        // publications above; without a definition on this path the
        // register allocator repairs their live ranges with
        // GPR->Shared copies, which the ISA cannot encode (llvm >=
        // 679005ba1294 turns that into a compile-time error; older
        // llvm aliased S1 onto sp and crashed at the epilogue). These
        // empty asm statements define both handles on the skipping
        // path without emitting any instruction, so each handle keeps
        // the S register its publication assigned it.
        asm volatile("" : "=Sr"(tA.handle_ref()));
        asm volatile("" : "=Sr"(tW.handle_ref()));
      }
      // Cooperative Shared-A / Shared-B TMATMUL: LB0 = GroupM (derived
      // from the Shared A type); each PE computes its kLocalM-row slice.
      TMATMUL(tC, tA, tW);
      // Map (group, PE) to the per-PE CUBE row slice of the [Ho*Wo, Co]
      // NHWC output. The static global_iterator is unusable with
      // RowMajor<-1, -1> (compile-time RowStride), so the block base is
      // computed arithmetically; the dynamic gmC supplies the runtime
      // row stride (GetStrideBytes(3) = Co elements) for TSTORE_CUBE.
      if (static_cast<int>(tid) * kLocalM < GroupM) {
        float *block = output_nhwc +
                       (i * kPeNum + static_cast<int64_t>(tid)) * kLocalM * co +
                       j * TileN;
        gmC gC(block, static_cast<int>(M), static_cast<int>(co));
        TSTORE_CUBE(gC, tC);
      }
    }
  }
}

}  // namespace supernpu::conv2d_img2col
