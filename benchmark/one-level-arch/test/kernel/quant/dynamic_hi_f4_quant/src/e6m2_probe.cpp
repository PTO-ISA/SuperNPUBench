#include <common/pto_tileop.hpp>
#include <cstdint>

// ============================================================================
// e6m2 emit-capability probe for dynamic_hi_f4_quant (L1 base scale cast).
// ============================================================================
// GOAL: verify whether one-level `TCVT(dst=e6m2, src=bf16)` — the §2.1 step
//   `E6M2 = cast_to_E6M2(SF_BF16)` — can be emitted at the Tile API layer.
//
// EMIT-only check (runtime blocked by toolchain<->emulator skew); disassemble
// with `make ... TYPE=E6M2_PROBE diss` and inspect the BSTART.TEPL TCVT block.
//
// FINDING (empirically confirmed 2026-08-14, see RECORD.md 问题3): the e6m2
// one-level cast is NOT emittable with a correct dst type on the current
// toolchain. The gap is TWO-fold:
//
//   (1) jcore/type.hpp registers NO `type_traits<__fp8_e6m2>` (and none for
//       `__fp8_e6m2x2`). The natural code below — a plain
//       `Tile<Location::Vec, __fp8_e6m2, ...>` fed to TCVT/TSTORE — fails to
//       compile:  "no member named 'TypeCode' in 'type_traits<__fp8_e6m2>'"
//       at template_asm.hpp:120 (TCVT dst) and :1794 (TSTORE src).
//       => Stage 1 (STAGE2 undefined) does NOT compile. This is the honest
//          one-level path and documents the hard stop.
//
//   (2) __type_code (jcore/type.hpp:7-39) has NO `__type_fp8_e6m2` enum entry
//       at all. The "e6m2" mnemonic string lives only in the CVT cast macros
//       (template_asm.hpp:781/813); the hardware dst TYPE field of TCVT/TSTORE
//       is driven by `type_traits<>::TypeCode` (an "i"() immediate). With no
//       correct code, we cannot tag the dst as e6m2.
//       => Stage 2 (-DSTAGE2) supplies a PLACEHOLDER type_traits<__fp8_e6m2>
//          borrowing code 13 (= e8m0) purely to push past the type_traits
//          wall. It compiles and emits a real `BSTART.TEPL TCVT, BF16` /
//          `B.DATR e8m0, ...` — but the dst is tagged **e8m0, NOT e6m2**,
//          proving the cast is mis-typed. TSTORE likewise emits `TSTORE, e8m0`.
//
// CONCLUSION: one-level TCVT->e6m2 is a real toolchain HEADER gap (缺陷所在仓:
// linx-toolchain-build / Linx-TileOP-API). Correct emit needs BOTH an
// `__type_fp8_e6m2` enum code and a matching type_traits specialization
// upstream. Operator workaround (RECORD 问题3 规避方案): compute the e6m2 base
// bits ourselves via bf16->e6m2 bit reconstruction (exponent rebias + 2-bit
// mantissa round) and store as raw uint8 — same bit-reconstruction philosophy
// as the recip M2-LUT (RECORD 问题4) — sidestepping the CVT dst-type entirely.
// ============================================================================
using namespace pto;

#ifdef STAGE2
// Stage-2 ONLY: manually inject the missing registration with a placeholder
// TypeCode (13 = e8m0) to characterize the depth of the gap. This does NOT make
// e6m2 a legal end-to-end dtype; the emitted dst type is wrong (e8m0).
template<> struct type_traits<__fp8_e6m2>
    : public type_traits_base<13 /*PLACEHOLDER = e8m0, NOT e6m2*/, 8> {};
#endif

static __bf16  x[8 * 32] __attribute__((aligned(4096))) = {};
static uint8_t y[8 * 32] __attribute__((aligned(4096))) = {};

int main() {
    using tile_b = Tile<Location::Vec, __bf16,     8, 32, BLayout::RowMajor>;
    using tile_e = Tile<Location::Vec, __fp8_e6m2, 8, 32, BLayout::RowMajor>;

    using gm_b = global_tensor<__bf16,  RowMajor<8, 32>>;
    using gm_e = global_tensor<uint8_t, RowMajor<8, 32>>;

    global_iterator<gm_b, tile_b> xi(x);
    global_iterator<gm_e, tile_e> yi(reinterpret_cast<uint8_t *>(y));

    auto gx = xi(0, 0);
    auto gy = yi(0, 0);

    tile_b xq;
    TLOAD(xq, gx);
    tile_e eq;
    TCVT(eq, xq); // bf16 -> e6m2: hard-fails at type_traits unless -DSTAGE2
    TSTORE(gy, eq);
    return 0;
}
