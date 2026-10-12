#pragma once

// conv2d implemented with the TileOP-API C++ wrappers for the PTO v0.58.6
// convolution-dedicated GM transports (NCHW input flavor):
//
//   * TIMG2COL_SPART<DN2ND, 8>(): GM NCHW feature map -> Shared Left
//     [GroupM, K] im2col matrix, single-issuer form (TileOP issue #194).
//     The Shared ND output is the only TIMG2COL surface that accepts the
//     DN (channel-major) GM addressing (B.DATR DN2ND: gm index =
//     channel * H * W + spatial), so the NCHW input planes are consumed
//     directly. One B.IOS from PE0 publishes the complete Shared parent
//     without the B.ASSEMBLE generation protocol; PEMask=8 encodes the
//     digit-reversed text "1000" and selects PE0, so the publication call
//     sits in the tid==0 branch. The "=Sr" handle def inside that branch
//     is the pattern the clang-15 coalescer used to abort on (fixed by
//     llvm abeccf4937e8, "Fix coalescer abort when a Shared handle lives
//     across a block"). The instruction owns the padding/stride/dilation
//     window contract; out-of-window taps read as zero.
//   * TLOAD<OIHW2NK, 8>(): GM OIHW weight (the PyTorch conv2d weight
//     layout) -> Shared Right [N, K] with the K order [kh][kw][c1][c0]
//     that pairs directly with the im2col A matrix. Both transports
//     project K in the same canonical order.
//   * TMATMUL(tC, tA, tW): cooperative Shared-Left / Shared-Right form
//     (the 3-arg wrapper; LB0 = group_M comes from the Shared A type's
//     ValidRow). Each PE computes its 16/32-row slice of the [GroupM,
//     TileN] output block; the gfrun rendezvous matches the Shared
//     operands by slot id and dimensions, not by program counter, so the
//     PE0-only publications meet every PE's TMATMUL.
//   * TSTORE_CUBE(): the PE CUBE row slice -> GM row-major [Ho*Wo, Co],
//     i.e. the NHWC output layout (row index h*Wo + w, column index
//     channel).
//
// Why the im2col A matrix rides Shared storage (and not a Local CUBE
// shard as in the earlier NHWC flavor):
//   * The direct-Local TIMG2COL forms are pinned to the ND (NHWC) source
//     addressing by the schema's ND2M16/ND2M32 B.DATR layouts; NCHW
//     sources are only routable through the Shared ND output.
//   * The cooperative TMATMUL accepts the Shared-Left/Shared-Right pair
//     directly, with the destination staying the per-PE Local accumulator
//     block, so no per-PE im2col shard materialization is needed.
//
// Known llvm hazard worked around below (movr S-reg aliasing): when a
// Shared handle produced by an "=Sr" asm def inside the tid==0 branch is
// consumed after the join, the register allocator repairs the live range
// with GPR->Shared copies; c.movr's 5-bit destination field cannot encode
// the S file and aliases the GPR file (S0==zero, S1==sp), so an S1
// repair silently clobbers sp and the epilogue faults. The else branch
// defines both handles through empty asm statements (no instructions), so
// the allocator never inserts the copies. With a single branch-crossing
// Shared handle the repair lands in S0==zero and is discarded by
// hardware - the earlier NHWC flavor relied on that accident.
//
// Requires llvm >= 69d6e6d8 (BSTART.TIMG2COL mnemonic, zero B.IOR
// selectors, B.DATR numeric layout aliases, Shared-handle coalescer fix)
// and TileOP-API >= f7c4a45 (TIMG2COL SharedND output + TIMG2COL_SPART,
// issue #194; the weight-TLOAD ISA-name B.DATR spelling and zero-folding
// opaque guards, PRs #188/#196).

#include <common/pto_tileop.hpp>
#include <cstdint>
#include <type_traits>

namespace supernpu::conv2d_img2col {

using namespace pto;

namespace detail {

constexpr bool is_pow2(int v) { return v > 0 && (v & (v - 1)) == 0; }

}  // namespace detail

// NCHW in, NHWC out convolution using the convolution-dedicated transports.
//
//   input_nchw:  [Batch][InChannels][InHeight][InWidth] row major
//   weight_oihw: [OutChannels][InChannels][KernelH][KernelW] row major
//                (PyTorch conv2d weight layout, consumed as-is)
//   output_nhwc: [Batch][OutH][OutW][OutChannels] row major
//
// Constraints (statically enforced):
//   * fp32 element type;
//   * K = KernelH*KernelW*ceil(Cin/c0)*c0 (c0 = 256/elem_bits) must be a
//     power of two and a multiple of c0 (the cooperative TMATMUL validates
//     pow2 M/N/K and the transports require C0-aligned K windows);
//   * Ho*Wo must be a multiple of GroupM (full row groups);
//   * OutChannels must be a multiple of TileN;
//   * GroupM must be 64 (16 rows/PE) or 128 (32 rows/PE): the output
//     row-block map (i*kPeNum + tid) assumes the group spans all four
//     PEs;
//   * the Shared capacities (GroupM*K and TileN*K elements) must be
//     power-of-two byte sizes inside the 128 B..256 KiB SizeCode range
//     (the transports require the destination capacity to match the
//     written range exactly, which any power-of-two size in range
//     satisfies).
template <int InChannels, int InHeight, int InWidth,
          int OutChannels, int KernelH, int KernelW,
          int StrideH, int StrideW,
          int PadTop, int PadBottom, int PadLeft, int PadRight,
          int DilH = 1, int DilW = 1,
          int GroupM = 64, int TileN = 16>
void conv2d_img2col(float *output_nhwc, float *input_nchw,
                    float *weight_oihw) {
  constexpr int kPeNum = 4;
  constexpr int kElemBits = 32;
  constexpr int kC0 = 256 / kElemBits;
  constexpr int kC1 = (InChannels + kC0 - 1) / kC0;
  constexpr int kK = KernelH * KernelW * kC1 * kC0;
  constexpr int kEffH = (KernelH - 1) * DilH + 1;
  constexpr int kEffW = (KernelW - 1) * DilW + 1;
  constexpr int kOutH =
      (InHeight + PadTop + PadBottom - kEffH) / StrideH + 1;
  constexpr int kOutW =
      (InWidth + PadLeft + PadRight - kEffW) / StrideW + 1;
  constexpr int kM = kOutH * kOutW;
  constexpr int kMb = kM / GroupM;
  constexpr int kNb = OutChannels / TileN;
  // Cooperative TMATMUL per-PE rows: group_M <= 64 -> 16, else 32.
  constexpr int kLocalM = GroupM <= 64 ? 16 : 32;

  static_assert(InHeight + PadTop + PadBottom >= kEffH &&
                    InWidth + PadLeft + PadRight >= kEffW,
                "padded feature map is smaller than the receptive field");
  static_assert(kOutH > 0 && kOutW > 0, "empty convolution output");
  static_assert(detail::is_pow2(kK),
                "K = KH*KW*ceil(Cin/c0)*c0 must be a power of two "
                "(cooperative TMATMUL validates pow2 K)");
  static_assert(kK % kC0 == 0,
                "K must be a multiple of the C0 channel carrier");
  static_assert(GroupM == 64 || GroupM == 128,
                "GroupM must be 64 (16 rows/PE) or 128 (32 rows/PE): the "
                "output row-block map assumes full four-PE groups");
  static_assert(detail::is_pow2(TileN), "TileN must be a power of two");
  static_assert(kM % GroupM == 0,
                "Ho*Wo must be a multiple of GroupM (no tail row group)");
  static_assert(OutChannels % TileN == 0,
                "OutChannels must be a multiple of TileN");
  static_assert(detail::is_pow2(GroupM * kK * 4) &&
                    GroupM * kK * 4 >= 128 && GroupM * kK * 4 <= 256 * 1024,
                "Shared im2col A capacity must be a power-of-two SizeCode");
  static_assert(detail::is_pow2(TileN * kK * 4) &&
                    TileN * kK * 4 >= 128 && TileN * kK * 4 <= 256 * 1024,
                "Shared weight capacity must be a power-of-two SizeCode");

  // TIMG2COL parameter words (PTO-BSTART-TIMG2COL-PARAMS-001):
  //   param0: inputH[15:0] inputW[31:16] cin[47:32] kernelH[55:48]
  //           kernelW[63:56]
  //   param1: padTop[7:0] padLeft[15:8] padBottom[23:16] padRight[31:24]
  //           dilationH[36:32] dilationW[41:37] strideH[47:42] strideW[53:48]
  //   param2: rowStart[31:0] colStart[63:32]
  constexpr uint64_t kParam0 =
      static_cast<uint64_t>(InHeight) |
      (static_cast<uint64_t>(InWidth) << 16) |
      (static_cast<uint64_t>(InChannels) << 32) |
      (static_cast<uint64_t>(KernelH) << 48) |
      (static_cast<uint64_t>(KernelW) << 56);
  constexpr uint64_t kParam1 =
      static_cast<uint64_t>(PadTop) |
      (static_cast<uint64_t>(PadLeft) << 8) |
      (static_cast<uint64_t>(PadBottom) << 16) |
      (static_cast<uint64_t>(PadRight) << 24) |
      (static_cast<uint64_t>(DilH) << 32) |
      (static_cast<uint64_t>(DilW) << 37) |
      (static_cast<uint64_t>(StrideH) << 42) |
      (static_cast<uint64_t>(StrideW) << 48);

  using gmA = global_tensor<float, RowMajor<InChannels, InHeight * InWidth>>;
  using gmW =
      global_tensor<float, RowMajor<OutChannels, InChannels * KernelH * KernelW>>;
  using gmC = global_tensor<float, RowMajor<kM, OutChannels>>;

  // Shared im2col A matrix: the complete [GroupM, K] group, published by
  // PE0 through the single-issuer TIMG2COL variant (ValidRow doubles as
  // the cooperative TMATMUL LB0 group_M).
  using tileA = SharedTile<SharedMatrixLeft<float, GroupM, kK>>;
  // Shared weight B: physical RowMajor [N, K] (TransB=0 publishes the
  // logical [K, N] B operand, pto-spec #257).
  using tileW = SharedTile<SharedMatrixRight<float, TileN, kK>>;
  // Per-PE CUBE row slice of the [GroupM, TileN] output block.
  using tileC16 = CubeAccumulatorM16<float, kLocalM, TileN>;
  using tileC32 = CubeAccumulatorM32<float, kLocalM, TileN>;
  using tileC = std::conditional_t<(kLocalM <= 16), tileC16, tileC32>;

  using itC = global_iterator<gmC, tileC>;

  const uint32_t tid = get_thread_idx();

  itC gIterC(output_nhwc);

  for (int i = 0; i < kMb; ++i) {
    for (int j = 0; j < kNb; ++j) {
      tileC tC;
      // Declared outside the branch: every PE's TMATMUL binds the same
      // compile-time Shared slots; only PE0 defines them.
      tileA tA;
      tileW tW;
      // Singleton Shared publications run on PE0 only; PEs 1..3 proceed
      // straight to the cooperative TMATMUL rendezvous (matched by slot
      // id and dimensions, not by program counter).
      if (tid == 0) {
        gmA gA(input_nchw);
        gmW gW(weight_oihw);
        // im2col rows [i*GroupM, (i+1)*GroupM) of the NCHW planes; one
        // B.IOS publishes the complete [GroupM, K] Shared parent.
        TIMG2COL_SPART<DN2ND>(
            tA, gA,
            TIMG2COLParams{kParam0, kParam1,
                           static_cast<uint64_t>(i * GroupM)},
            /*PEMask=*/8);
        // Weight block j: the OIHW [nStart, nStart+TileN) x [0, K) window.
        TLOAD<OIHW2NK, /*PEMask=*/8>(
            tW, gW,
            make_weight_tload_params(InChannels, OutChannels, KernelH,
                                     KernelW, /*n_start=*/j * TileN));
      } else {
        // The two Shared handles are defined by the PE0-only
        // publications above; without a definition on this path the
        // register allocator repairs their live ranges with
        // GPR->Shared copies, which the ISA cannot encode. llvm >=
        // 679005ba1294 (issue #205) turns that into a compile-time
        // "cannot copy a Shared register" fatal error (older llvm
        // silently aliased S1 onto sp and crashed at the epilogue).
        // These empty asm statements define both handles on the
        // skipping path without emitting any instruction, so each
        // handle keeps the S register its publication assigned it.
        asm volatile("" : "=Sr"(tA.handle_ref()));
        asm volatile("" : "=Sr"(tW.handle_ref()));
      }
      // Cooperative Shared-A / Shared-B TMATMUL: LB0 = GroupM (derived
      // from the Shared A type); each PE computes its kLocalM-row slice.
      TMATMUL(tC, tA, tW);
      // Map (group, PE) to the per-PE CUBE row slice of the [kM, Co]
      // output.
      if (static_cast<int>(tid) * kLocalM < GroupM) {
        auto gC = gIterC(i * kPeNum + static_cast<int>(tid), j);
        TSTORE_CUBE(gC, tC);
      }
    }
  }
}

}  // namespace supernpu::conv2d_img2col
