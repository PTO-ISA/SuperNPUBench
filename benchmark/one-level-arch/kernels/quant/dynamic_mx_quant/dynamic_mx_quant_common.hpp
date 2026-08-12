#ifndef SUPERNPU_DYNAMIC_MX_QUANT_COMMON_HPP
#define SUPERNPU_DYNAMIC_MX_QUANT_COMMON_HPP

#include <common/pto_tileop.hpp>
#include <cstdint>
#include <type_traits>

namespace supernpu::tile_isa::mxquant {

// --- bf16 (E8M7) domain, mirrors AscendC dynamic_mx_quant_common.h -----------
constexpr uint16_t BF16_EXP_MASK = 0x7F80; // BF16_MAX_EXP
constexpr uint16_t BF16_ABS_MASK = 0x7FFF;
constexpr uint16_t BF16_EXP_BIAS = 0x7F00; // scale bias / recip base
constexpr uint16_t BF16_SHR_NUM = 7;
constexpr uint16_t BF16_NAN_PATTERN = 0x7F81; // BF16_NAN_CUSTOM (recip NaN)
constexpr uint16_t BF16_SPECIAL_EXP = 0x0040; // BF16_SPECIAL_EXP_THRESHOLD
constexpr uint16_t BF16_ADD_VALUE_MAN1 = 0x003F; // DynRange rounding add-value

// emax = biased-domain exponent (BF16 E8M7, i.e. <<7) of the output dtype's
// max normal value. Mirrors AscendC dynamic_mx_quant_common.h:81-90.
constexpr uint16_t FP8_E4M3_EMAX = 0x0400; // 448  = 1.75 * 2^8  -> exp 8
constexpr uint16_t FP8_E5M2_EMAX = 0x0780; // 57344= 1.75 * 2^15 -> exp 15
constexpr uint16_t FP4_E2M1_EMAX = 0x0100; // 6    = 1.5  * 2^2  -> exp 2
constexpr uint16_t FP4_E1M2_EMAX = 0x0000; // exp 0

constexpr float FP8_E4M3_INV_DST_MAX = 0.002232142857f; // 1/448
constexpr float FP8_E5M2_INV_DST_MAX = 0.000017438616f; // 1/57344
constexpr uint16_t FP8_NAN_BYTE = 0x00FF; // FP8_DEFAULT_MAX_EXP (E8M0 scale NaN)

// --- fp32 (E8M23) domain, cuBLAS path ---------------------------------------
constexpr uint32_t FP32_EXP_MASK = 0x7F800000; // FP32_MX_MAX_EXP
constexpr uint32_t FP32_MANTISSA_MASK = 0x007FFFFF;
constexpr uint32_t FP32_SHR_NUM = 23;
constexpr uint32_t FP32_EXP_BIAS_CUBLAS = 0x00007F00; // recip base (bf16 bias, low16)
constexpr uint32_t FP32_NAN_PACK = 0x00007F81; // recip NaN low16
constexpr uint32_t FP32_FP8_NAN = 0x000000FF; // scale-byte NaN
constexpr uint32_t FP32_NUMBER_254 = 0x000000FE;
constexpr uint32_t FP32_NUMBER_HALF = 0x00400000;
constexpr float CLAMP_MIN = 1e-12f; // maxLowBound_
// Default maxLowBound as an fp32 bit pattern. This backend's clang rejects
// floating-point non-type template parameters ("non-type template argument of
// type 'float' is not yet supported"), so maxLowBound is threaded as a uint32_t
// NTTP (its fp32 bits) and rebuilt via __builtin_bit_cast at the clamp site.
constexpr uint32_t CLAMP_MIN_BITS = 0x2b8cbcccu; // bits of 1e-12f

// emax derived FROM the output tile element type, exactly as AscendC does
// (GetFp4MaxExp<T>() / GetFp8MaxExp<T>()); the caller supplies only OutT.
template <typename OutT>
constexpr uint16_t emax_bits() {
    if constexpr (std::is_same_v<OutT, __fp8_e4m3>) {
        return FP8_E4M3_EMAX;
    } else if constexpr (std::is_same_v<OutT, __fp8_e5m2>) {
        return FP8_E5M2_EMAX;
    } else if constexpr (std::is_same_v<OutT, __fp4_e2m1x2>) {
        return FP4_E2M1_EMAX;
    } else {
        static_assert(std::is_same_v<OutT, __fp4_e1m2x2>,
                      "emax_bits: unsupported output dtype");
        return FP4_E1M2_EMAX;
    }
}

// cuBLAS 1/dstMax, keyed on output dtype (cuBLAS is FP8-only).
template <typename OutT>
constexpr float inv_dst_max() {
    if constexpr (std::is_same_v<OutT, __fp8_e4m3>) {
        return FP8_E4M3_INV_DST_MAX;
    } else {
        static_assert(std::is_same_v<OutT, __fp8_e5m2>,
                      "inv_dst_max: cuBLAS supports only FP8 output");
        return FP8_E5M2_INV_DST_MAX;
    }
}

// ---------------------------------------------------------------------------
// Compile-time tile-size derivation (TileM / TileN / R_sub) from operator inputs
// and the INPUT dtype's binding-tile budget. Mirrors the budget model documented
// in README「验证边界」/ DESIGN §7.5: a fixed 8192-byte binding-tile budget bound by
// the WIDEST tile that crosses LOAD/STORE. OCP stays in the input/uint16 (≤ input
// width) domain, so its binding width is sizeof(InT). cuBLAS currently pushes an
// fp32 (4B) tile through a scratch-HBM reinterpret roundtrip (问题4), so its binding
// width is 4B until the compiler exposes a register-level reinterpret (kRegBitcast),
// after which it falls back to sizeof(InT).
// NOTE: InT here only sizes/validates the tile budget; the kernels' DATA PATH is
// still bf16-only (each kernel static_asserts InT == __bf16). fp32/fp16 input data
// paths (32b-domain handling) are a separate, not-yet-landed item.
constexpr int  kTileBudgetBytes = 8192;
constexpr bool kRegBitcast = false; // flip true once 问题4 (reg reinterpret) lands

template <typename InT, bool IsCublas>
constexpr int binding_tile_bytes() {
    if constexpr (IsCublas && !kRegBitcast) {
        return 4; // fp32 scratch-HBM roundtrip binds a 32b tile
    } else {
        return static_cast<int>(sizeof(InT)); // bf16/fp16 = 2, fp32 = 4
    }
}

template <typename InT, bool IsCublas>
constexpr int tile_elem_budget() {
    return kTileBudgetBytes / binding_tile_bytes<InT, IsCublas>();
}

// Non-tail contiguous-axis (TileN) 32B-column-alignment lower bound on the packed
// output: fp8 -> TileN % 32; fp4 packs 2/byte -> TileN % 64.
template <typename OutT>
constexpr int nontail_align_lower() {
    if constexpr (std::is_same_v<OutT, __fp4_e2m1x2> || std::is_same_v<OutT, __fp4_e1m2x2>) {
        return 64;
    } else {
        return 32;
    }
}

// Tail-axis: largest PHYSICAL TileM whose tile [TileM, Contig] (input-width) fits
// the budget, clamped to [tilem_min, M]. Contig is the physical contiguous width
// per one-block-per-tile iteration: BlockSize for the cublas/plain tail, PW =
// ceil(BlockSize/64)*64 for the padded-physical fp4 tail. tilem_min keeps the
// physical tile >= 512B (LinxV5 sub-512B spill crash); the full/tail split already
// boxes the valid rows, so a physical TileM > M is safe (valid rows = min(M, TileM)).
template <int M, int Contig, typename InT, bool IsCublas>
constexpr int max_tilem() {
    constexpr int budget = tile_elem_budget<InT, IsCublas>();
    constexpr int min_elems = 512 / static_cast<int>(sizeof(InT)); // >= 512B floor
    constexpr int tilem_min = (min_elems + Contig - 1) / Contig;    // ceil
    constexpr int tilem_max = budget / Contig;
    int t = M;
    if (t > tilem_max) t = tilem_max;
    if (t < tilem_min) t = tilem_min;
    if (t < 1) t = 1;
    return t;
}

// Non-tail: largest align-multiple TileN with BlockSize*TileN <= budget, further
// capped at align-rounded Post so a small Post doesn't inflate TileN (N_tail covers
// the remainder). Returns 0 when no legal TileN exists (< align lower bound) ->
// caller routes to the bigbs split-reduce kernel.
template <int BlockSize, int Post, typename OutT, typename InT, bool IsCublas>
constexpr int pick_tilen() {
    constexpr int align  = nontail_align_lower<OutT>();
    constexpr int budget = tile_elem_budget<InT, IsCublas>();
    int cap = budget / BlockSize;                       // TileSize upper on TileN
    int postcap = ((Post + align - 1) / align) * align; // no need to exceed Post
    if (postcap < cap) cap = postcap;
    return (cap / align) * align;                       // floor to align (0 if cap < align)
}

// bigbs: largest divisor of BlockSize with R_sub*TileN <= budget (the CURRENT
// budget, not the formal 4096 the bigbs kernel static_asserts against). R_sub |
// BlockSize is required by 方案A (splits the reduce axis into R_sub-row sub-chunks).
template <int BlockSize, int TileN, typename InT, bool IsCublas>
constexpr int max_rsub() {
    constexpr int budget = tile_elem_budget<InT, IsCublas>();
    int cap = budget / TileN;
    if (cap > BlockSize) cap = BlockSize;
    for (int r = cap; r >= 1; --r) {
        if (BlockSize % r == 0) return r;
    }
    return 1;
}

// ---------------------------------------------------------------------------
// ================== 规避方案 (WORKAROUND) ==================
// 理想写法是用寄存器级 reinterpret / bitcast 语法（不经内存、不做数值转换）把
// float tile 的位型当整数用、或反过来。但当前编译器工具链**尚未支持**该语法形式
// （TCAST 在 -D__linx 下未声明，且它是数值转换非位重解释）——这是编译器侧待解决项，
// 与问题3（linx 缺 4-参 CmpMode）同类。详见 RECORD.md「问题4：编译器尚未支持
// reinterpret（位重解释）语法形式」。此处以 scratch-HBM 字节别名往返作为等价规避，代价是每次
// 多一次 HBM store+load 与一块静态 buffer；编译器补齐后可替换为寄存器级写法。
// ----------------------------------------------------------
// Bit reinterpret via scratch-HBM byte-aliasing. linx (-D__linx) has no
// register bitcast (TCAST is #ifndef __linx); TLOAD/TSTORE move raw bytes keyed
// only on element byte-width, so aliasing two same-width globals over the same
// scratch bytes performs a true reinterpret. Slot picks a distinct buffer so
// two live reinterprets never share storage.
// ValidR/ValidC (defaulted to R/C) carry the valid region so a partial tile
// reinterprets only its live lanes. When ValidR==R && ValidC==C the boxed type
// collapses to the NoneBox default, so existing full-tile callers are
// unaffected. A boxed valid=1 (ValidC==1 col-reduced, or ValidR==1 row-reduced)
// reinterprets only the per-block scalar strip: store and load share the same
// box so the same RowMajor offsets round-trip losslessly.
template <int Slot, int R, int C, int ValidR = R, int ValidC = C>
inline void reinterpret_u16_to_bf16(
    pto::Tile<pto::Location::Vec, uint16_t, R, C, pto::BLayout::RowMajor, ValidR, ValidC> &src,
    pto::Tile<pto::Location::Vec, __bf16,   R, C, pto::BLayout::RowMajor, ValidR, ValidC> &dst) {
    using namespace pto;
    static uint8_t buf[R * C * sizeof(uint16_t)] __attribute__((aligned(4096)));
    using gm_u = global_tensor<uint16_t, RowMajor<R, C>>;
    using gm_b = global_tensor<__bf16,   RowMajor<R, C>>;
    global_iterator<gm_u, Tile<Location::Vec, uint16_t, R, C, BLayout::RowMajor, ValidR, ValidC>> wi(reinterpret_cast<uint16_t *>(buf));
    global_iterator<gm_b, Tile<Location::Vec, __bf16,   R, C, BLayout::RowMajor, ValidR, ValidC>> ri(reinterpret_cast<__bf16 *>(buf));
    auto gw = wi(0, 0);
    TSTORE(gw, src);
    auto gr = ri(0, 0);
    TLOAD(dst, gr);
}

template <int Slot, int R, int C, int ValidR = R, int ValidC = C>
inline void reinterpret_f32_to_u32(
    pto::Tile<pto::Location::Vec, float,    R, C, pto::BLayout::RowMajor, ValidR, ValidC> &src,
    pto::Tile<pto::Location::Vec, uint32_t, R, C, pto::BLayout::RowMajor, ValidR, ValidC> &dst) {
    using namespace pto;
    static uint8_t buf[R * C * sizeof(uint32_t)] __attribute__((aligned(4096)));
    using gm_f = global_tensor<float,    RowMajor<R, C>>;
    using gm_u = global_tensor<uint32_t, RowMajor<R, C>>;
    global_iterator<gm_f, Tile<Location::Vec, float,    R, C, BLayout::RowMajor, ValidR, ValidC>> wi(reinterpret_cast<float *>(buf));
    global_iterator<gm_u, Tile<Location::Vec, uint32_t, R, C, BLayout::RowMajor, ValidR, ValidC>> ri(reinterpret_cast<uint32_t *>(buf));
    auto gw = wi(0, 0);
    TSTORE(gw, src);
    auto gr = ri(0, 0);
    TLOAD(dst, gr);
}

// ---------------------------------------------------------------------------
// OCP / DynRange scale-byte + recip finalize (bf16/uint16 domain). Mirrors the
// tail of AscendC ComputeScaleOcp / ComputeScaleDynamicDtypeRange:
//   scaleValue = sharedExp >> 7;   select finite? scaleValue : 0x00ff
//   halfScale  = 0x7f00 - sharedExp
//   select finite? half : 0x7f81;  select (sharedExp!=0)? half : 0;
//   select (sharedExp==0x7f00)? 0x0040 : half
// eq_inf carries (xMaxExpOnly == expMask) i.e. NOT finite. TSEL(dst,mask,src) =
// mask ? src : dst.
template <int R, int C, int ValidR = R, int ValidC = C>
inline void finalize_scale_recip_u16(
    pto::Tile<pto::Location::Vec, uint16_t, R, C, pto::BLayout::RowMajor, ValidR, ValidC> &shared_exp,
    pto::Tile<pto::Location::Vec, uint16_t, R, C, pto::BLayout::RowMajor, ValidR, ValidC> &eq_inf,
    pto::Tile<pto::Location::Vec, uint16_t, R, C, pto::BLayout::RowMajor, ValidR, ValidC> &scale_byte,
    pto::Tile<pto::Location::Vec, uint16_t, R, C, pto::BLayout::RowMajor, ValidR, ValidC> &recip_out) {
    using namespace pto;
    using tile_u16 = Tile<Location::Vec, uint16_t, R, C, BLayout::RowMajor, ValidR, ValidC>;

    TSHRS(scale_byte, shared_exp, BF16_SHR_NUM);
    tile_u16 nan_byte;
    TEXPANDS(nan_byte, FP8_NAN_BYTE);
    TSEL(scale_byte, eq_inf, nan_byte); // finite? scale : 0x00ff

    tile_u16 eq_zero;
    TCMPS(eq_zero, shared_exp, static_cast<uint16_t>(0));
    tile_u16 eq_special;
    TCMPS(eq_special, shared_exp, BF16_EXP_BIAS);

    tile_u16 bias_t;
    TEXPANDS(bias_t, BF16_EXP_BIAS);
    TSUB(recip_out, bias_t, shared_exp); // 0x7f00 - sharedExp

    tile_u16 nan16;
    TEXPANDS(nan16, BF16_NAN_PATTERN);
    TSEL(recip_out, eq_inf, nan16); // finite? half : 0x7f81

    tile_u16 zero_t;
    TEXPANDS(zero_t, static_cast<uint16_t>(0));
    TSEL(recip_out, eq_zero, zero_t); // (sharedExp!=0)? half : 0

    tile_u16 special_t;
    TEXPANDS(special_t, BF16_SPECIAL_EXP);
    TSEL(recip_out, eq_special, special_t); // (sharedExp==0x7f00)? 0x0040 : half
}

// cuBLAS scale-byte + recip from an already-reduced fp32 amax. Mirrors AscendC
// ComputeScaleCublas (dynamic_mx_quant_tail_axis_fp8.h:685-782). GT/LT are
// EQ-emulated (TCMP/TCMPS are EQ-only on linx): a<K == min(a,K-1)==a,
// a>K == max(a,K+1)==a, a>0 == !(a==0). MaxLowBoundBits mirrors AscendC
// maxLowBound_ (Reg::Maxs(max32, max32, maxLowBound_)); it is a per-invocation
// clamp threshold supplied as a template arg (fp32 bit pattern, since this clang
// rejects float NTTPs), defaulting to the bits of CLAMP_MIN. max_abs here is the
// per-block reduced amax as a boxed valid=1 tile (ValidC==1 col-reduced / tail,
// or ValidR==1 row-reduced / not_tail): the bit-math runs on the single live
// lane per block, mirroring AscendC's per-block scalar register math, and the
// valid=1 scale_byte/recip flow straight out to a 1-byte-per-block store and a
// fused TROWEXPANDMUL/TCOLEXPANDMUL in the data pass (no pre-broadcast).
template <typename OutT, int R, int C, uint32_t MaxLowBoundBits = CLAMP_MIN_BITS, int ValidR = R, int ValidC = C>
inline void compute_cublas_core(
    pto::Tile<pto::Location::Vec, float,    R, C, pto::BLayout::RowMajor, ValidR, ValidC> &max_abs,
    pto::Tile<pto::Location::Vec, uint16_t, R, C, pto::BLayout::RowMajor, ValidR, ValidC> &scale_byte,
    pto::Tile<pto::Location::Vec, uint16_t, R, C, pto::BLayout::RowMajor, ValidR, ValidC> &recip_out) {
    using namespace pto;
    using tile_u32 = Tile<Location::Vec, uint32_t, R, C, BLayout::RowMajor, ValidR, ValidC>;

    // ================== 规避方案 (WORKAROUND) ==================
    // 本段用 min/max + 默认-EQ 的 3-参 TCMP/TCMPS 模拟 GT/LT/NE 比较。
    // 理想写法是直接用带 CmpMode 的 4-参 TCMP/TCMPS 1:1 对照 AscendC
    // ComputeScaleCublas（见文件末尾注释保留的 IDEAL 版本），但 -D__linx 构建
    // 下随附头文件 jcore/template_asm.hpp 只提供 3-参 mode-less 的 TCMP/TCMPS，
    // 带 CmpMode 的 4-参重载仅存在于 jcore/TCmp.hpp（linx 未包含）。
    // 详见 RECORD.md「问题3：linx 缺失带 CmpMode 的 TCMP/TCMPS」。
    // 这是 linx 工具链侧需要补齐的能力；补齐后可切换到 IDEAL 版本。
    // 语义等价映射：
    //   a<b  == TMINS(t,a,b-1); TCMP(m,t,a)   （默认 EQ：t==a 即 a<=b-1 即 a<b）
    //   a>b  == TMAXS(t,a,b+1); TCMP(m,t,a)
    //   a!=b == TNOT(TCMPS(m,a,b))
    // ===========================================================

    // raw amax bits (pre-clamp) for finite / nonzero masks
    tile_u32 raw;
    reinterpret_f32_to_u32<0, R, C, ValidR, ValidC>(max_abs, raw);
    tile_u32 tmp;
    TMINS(tmp, raw, static_cast<uint32_t>(FP32_EXP_MASK - 1));
    tile_u32 finite; // raw < 0x7f800000
    TCMP(finite, tmp, raw);
    tile_u32 eq_zero;
    TCMPS(eq_zero, raw, static_cast<uint32_t>(0));

    // clamp + scale (float), then re-view scaled bits
    TMAXS(max_abs, max_abs, __builtin_bit_cast(float, MaxLowBoundBits));
    TMULS(max_abs, max_abs, inv_dst_max<OutT>());
    tile_u32 s32;
    reinterpret_f32_to_u32<1, R, C, ValidR, ValidC>(max_abs, s32);

    tile_u32 exp32;
    TSHRS(exp32, s32, FP32_SHR_NUM);
    tile_u32 man32;
    TANDS(man32, s32, FP32_MANTISSA_MASK);

    // p0 = (exp>0) && (exp<254) && (man>0)
    tile_u32 exp0;
    TCMPS(exp0, exp32, static_cast<uint32_t>(0));
    tile_u32 man0;
    TCMPS(man0, man32, static_cast<uint32_t>(0));
    tile_u32 t_lt;
    TMINS(t_lt, exp32, static_cast<uint32_t>(FP32_NUMBER_254 - 1));
    tile_u32 exp_lt254;
    TCMP(exp_lt254, t_lt, exp32);
    tile_u32 not_exp0;
    TNOT(not_exp0, exp0);
    tile_u32 not_man0;
    TNOT(not_man0, man0);
    tile_u32 pa;
    TAND(pa, not_exp0, exp_lt254);
    TAND(pa, pa, not_man0);
    // p1 = (exp==0) && (man>0x400000)
    tile_u32 t_gt;
    TMAXS(t_gt, man32, static_cast<uint32_t>(FP32_NUMBER_HALF + 1));
    tile_u32 man_gt;
    TCMP(man_gt, t_gt, man32);
    tile_u32 pb;
    TAND(pb, exp0, man_gt);
    tile_u32 roundup;
    TOR(roundup, pa, pb);

    // extractExp = roundup? exp+1 : exp ; finite? .. : 0xff ; nonzero? .. : 0
    tile_u32 exp_p1;
    TADDS(exp_p1, exp32, static_cast<uint32_t>(1));
    tile_u32 sel;
    TADDS(sel, exp32, static_cast<uint32_t>(0));
    TSEL(sel, roundup, exp_p1);
    tile_u32 nanb;
    TEXPANDS(nanb, FP32_FP8_NAN);
    tile_u32 fsel;
    TADDS(fsel, nanb, static_cast<uint32_t>(0));
    TSEL(fsel, finite, sel); // finite? sel : 0xff
    tile_u32 zero_t;
    TEXPANDS(zero_t, static_cast<uint32_t>(0));
    TSEL(fsel, eq_zero, zero_t); // (raw==0)? 0 : ..
    TCVT(scale_byte, fsel); // narrow low16 (values < 2^16)

    // recip = 0x7f00 - (extractExp<<7) ; finite? .. : 0x7f81 ; nonzero? .. : 0
    tile_u32 sh;
    TSHLS(sh, fsel, static_cast<uint32_t>(BF16_SHR_NUM));
    tile_u32 bias;
    TEXPANDS(bias, FP32_EXP_BIAS_CUBLAS);
    tile_u32 half;
    TSUB(half, bias, sh);
    tile_u32 rnan;
    TEXPANDS(rnan, FP32_NAN_PACK);
    tile_u32 rsel;
    TADDS(rsel, rnan, static_cast<uint32_t>(0));
    TSEL(rsel, finite, half);
    TSEL(rsel, eq_zero, zero_t);
    TCVT(recip_out, rsel);

/* === IDEAL VERSION (blocked): direct-CmpMode, mirrors AscendC ComputeScaleCublas
   1:1 (Compare<LT>/<NE>/<GT>/<EQ>) via the 4-arg TCMPS(dst,src,s,CmpMode) that is
   documented in docs/intrinsics/tcmps.md. Does NOT compile on -D__linx today —
   jcore/template_asm.hpp only ships the 3-arg mode-less TCMP/TCMPS; the 4-arg
   CmpMode overloads live in jcore/TCmp.hpp which is not included under __linx.
   Switch to this once linx exposes the 4-arg overload (see RECORD.md 问题3).

    tile_u32 raw;
    reinterpret_f32_to_u32<0, R, C, ValidR>(max_abs, raw);
    tile_u32 finite;  // raw < 0x7f800000   (AscendC:741 Compare<LT>)
    TCMPS(finite, raw, FP32_EXP_MASK, CmpMode::LT);
    tile_u32 nonzero; // raw != 0           (AscendC:742 Compare<NE>)
    TCMPS(nonzero, raw, static_cast<uint32_t>(0), CmpMode::NE);

    TMAXS(max_abs, max_abs, __builtin_bit_cast(float, MaxLowBoundBits));
    TMULS(max_abs, max_abs, inv_dst_max<OutT>());
    tile_u32 s32;
    reinterpret_f32_to_u32<1, R, C, ValidR, ValidC>(max_abs, s32);

    tile_u32 exp32;
    TSHRS(exp32, s32, FP32_SHR_NUM);
    tile_u32 man32;
    TANDS(man32, s32, FP32_MANTISSA_MASK);

    // p0 = (exp>0) && (exp<254) && (man>0)   (AscendC:751-755)
    tile_u32 p0a; TCMPS(p0a, exp32, static_cast<uint32_t>(0),                CmpMode::GT);
    tile_u32 p0b; TCMPS(p0b, exp32, static_cast<uint32_t>(FP32_NUMBER_254),  CmpMode::LT);
    tile_u32 p0c; TCMPS(p0c, man32, static_cast<uint32_t>(0),                CmpMode::GT);
    tile_u32 pa;
    TAND(pa, p0a, p0b);
    TAND(pa, pa, p0c);
    // p1 = (exp==0) && (man>0x400000)        (AscendC:757-759)
    tile_u32 p1a; TCMPS(p1a, exp32, static_cast<uint32_t>(0),                CmpMode::EQ);
    tile_u32 p1b; TCMPS(p1b, man32, static_cast<uint32_t>(FP32_NUMBER_HALF), CmpMode::GT);
    tile_u32 pb;
    TAND(pb, p1a, p1b);
    tile_u32 roundup;
    TOR(roundup, pa, pb);

    tile_u32 exp_p1;
    TADDS(exp_p1, exp32, static_cast<uint32_t>(1));
    tile_u32 sel;
    TADDS(sel, exp32, static_cast<uint32_t>(0));
    TSEL(sel, roundup, exp_p1);          // roundup? exp+1 : exp
    tile_u32 nanb;
    TEXPANDS(nanb, FP32_FP8_NAN);
    TSEL(nanb, finite, sel);             // finite? sel : 0xff
    tile_u32 extract;
    TEXPANDS(extract, static_cast<uint32_t>(0));
    TSEL(extract, nonzero, nanb);        // nonzero? .. : 0   (AscendC:765)
    TCVT(scale_byte, extract);           // narrow low16

    tile_u32 sh;
    TSHLS(sh, extract, static_cast<uint32_t>(BF16_SHR_NUM));
    tile_u32 bias;
    TEXPANDS(bias, FP32_EXP_BIAS_CUBLAS);
    tile_u32 half;
    TSUB(half, bias, sh);
    tile_u32 rnan;
    TEXPANDS(rnan, FP32_NAN_PACK);
    TSEL(rnan, finite, half);            // finite? half : 0x7f81
    tile_u32 rsel;
    TEXPANDS(rsel, static_cast<uint32_t>(0));
    TSEL(rsel, nonzero, rnan);           // nonzero? .. : 0
    TCVT(recip_out, rsel);
*/
}

// ===========================================================================
// Tail-axis variants (reduce along BlockSize columns via TROWMAX).
// ===========================================================================
template <typename OutT, int TileM, int BlockSize>
void compute_ocp_scale_tail(
    Tile<Location::Vec, uint16_t, TileM, BlockSize, BLayout::RowMajor> &x_u16,
    Tile<Location::Vec, uint16_t, TileM, BlockSize, BLayout::RowMajor> &scale_byte,
    Tile<Location::Vec, uint16_t, TileM, BlockSize, BLayout::RowMajor> &recip_out) {
    using namespace pto;
    using tile_u16 = Tile<Location::Vec, uint16_t, TileM, BlockSize, BLayout::RowMajor>;
    constexpr uint16_t EMAX = emax_bits<OutT>();

    tile_u16 exp_bits;
    TANDS(exp_bits, x_u16, BF16_EXP_MASK);
    tile_u16 max_exp;
    TROWMAX(max_exp, exp_bits);

    tile_u16 eq_inf; // (xMaxExp == expMask), computed pre-clamp
    TCMPS(eq_inf, max_exp, BF16_EXP_MASK);

    TMAXS(max_exp, max_exp, EMAX); // clamp up to emax (== invalidData select)
    tile_u16 emax_tile;
    TEXPANDS(emax_tile, EMAX);
    tile_u16 shared_exp;
    TSUB(shared_exp, max_exp, emax_tile);

    finalize_scale_recip_u16<TileM, BlockSize>(shared_exp, eq_inf, scale_byte, recip_out);
}

// Padded-physical, column-boxed OCP tail scale (used by the col-box fp4 tail).
// Same math as compute_ocp_scale_tail_boxed but the PHYSICAL tile width PW is
// decoupled from the valid BlockSize: value tiles are physical PW / valid
// BlockSize, so TROWMAX reduces over ValidCol=BlockSize only (cpu_sim
// TRowMax.hpp:13 loops j<ValidCol), keeping per-block reduction independent
// while PW pads the register width so the downstream fp4 output tile (PW/2
// bytes) is 32B-column-aligned. When PW == BlockSize this is identical to the
// boxed variant below. ValidM carries the live-row count.
template <typename OutT, int TileM, int PW, int BlockSize, int ValidM = TileM>
void compute_ocp_scale_tail_boxed_pw(
    Tile<Location::Vec, uint16_t, TileM, PW, BLayout::RowMajor, ValidM, BlockSize> &x_u16,
    Tile<Location::Vec, uint16_t, TileM, PW, BLayout::RowMajor, ValidM, 1> &scale_byte,
    Tile<Location::Vec, uint16_t, TileM, PW, BLayout::RowMajor, ValidM, 1> &recip_out) {
    using namespace pto;
    using tile_full = Tile<Location::Vec, uint16_t, TileM, PW, BLayout::RowMajor, ValidM, BlockSize>;
    using tile_box  = Tile<Location::Vec, uint16_t, TileM, PW, BLayout::RowMajor, ValidM, 1>;
    constexpr uint16_t EMAX = emax_bits<OutT>();

    tile_full exp_bits;
    TANDS(exp_bits, x_u16, BF16_EXP_MASK);
    tile_box max_exp;
    TROWMAX(max_exp, exp_bits);            // reduce ValidCol=BlockSize -> valid col=1

    tile_box eq_inf;
    TCMPS(eq_inf, max_exp, BF16_EXP_MASK);

    TMAXS(max_exp, max_exp, EMAX);
    tile_box emax_tile;
    TEXPANDS(emax_tile, EMAX);
    tile_box shared_exp;
    TSUB(shared_exp, max_exp, emax_tile);

    finalize_scale_recip_u16<TileM, PW, ValidM, 1>(shared_exp, eq_inf, scale_byte, recip_out);
}

// Boxed (valid col=1) OCP tail scale. Same math as compute_ocp_scale_tail but
// the col reduction (TROWMAX) collapses into a valid-col=1 tile so scale_byte
// and recip stay boxed (one per-row scalar per block) — mirrors the boxed cuBLAS
// tail so the caller stores one E8M0 byte per block (compact) and fuses the recip
// via TROWEXPANDMUL in the data pass. ValidM carries the live-row count (tail
// blocks pass M_tail; full tiles use the default -> box collapses to NoneBox).
// NOTE: superseded by compute_ocp_scale_tail_boxed_pw above (the col-box fp4
// tail now uses padded-physical tiles). Kept for reference / rollback.
#if 0
template <typename OutT, int TileM, int BlockSize, int ValidM = TileM>
void compute_ocp_scale_tail_boxed(
    Tile<Location::Vec, uint16_t, TileM, BlockSize, BLayout::RowMajor, ValidM, BlockSize> &x_u16,
    Tile<Location::Vec, uint16_t, TileM, BlockSize, BLayout::RowMajor, ValidM, 1> &scale_byte,
    Tile<Location::Vec, uint16_t, TileM, BlockSize, BLayout::RowMajor, ValidM, 1> &recip_out) {
    using namespace pto;
    using tile_full = Tile<Location::Vec, uint16_t, TileM, BlockSize, BLayout::RowMajor, ValidM, BlockSize>;
    using tile_box  = Tile<Location::Vec, uint16_t, TileM, BlockSize, BLayout::RowMajor, ValidM, 1>;
    constexpr uint16_t EMAX = emax_bits<OutT>();

    tile_full exp_bits;
    TANDS(exp_bits, x_u16, BF16_EXP_MASK);
    tile_box max_exp;
    TROWMAX(max_exp, exp_bits);            // reduce cols -> valid col=1

    tile_box eq_inf;
    TCMPS(eq_inf, max_exp, BF16_EXP_MASK);

    TMAXS(max_exp, max_exp, EMAX);
    tile_box emax_tile;
    TEXPANDS(emax_tile, EMAX);
    tile_box shared_exp;
    TSUB(shared_exp, max_exp, emax_tile);

    finalize_scale_recip_u16<TileM, BlockSize, ValidM, 1>(shared_exp, eq_inf, scale_byte, recip_out);
}
#endif

// ValidM (defaulted to TileM) carries the valid row count. Full tiles use the
// default (boxed type collapses to NoneBox); a tail invocation passes ValidM =
// M_tail so every intermediate operates only on the live rows, while the
// PHYSICAL tile shape stays TileM x BlockSize (keeps the 512B/32B store legal).
template <typename OutT, int TileM, int BlockSize, uint32_t MaxLowBoundBits = CLAMP_MIN_BITS, int ValidM = TileM>
void compute_cublas_scale_tail(
    Tile<Location::Vec, __bf16,   TileM, BlockSize, BLayout::RowMajor, ValidM, BlockSize> &x_bf16,
    Tile<Location::Vec, uint16_t, TileM, BlockSize, BLayout::RowMajor, ValidM, 1> &scale_byte,
    Tile<Location::Vec, uint16_t, TileM, BlockSize, BLayout::RowMajor, ValidM, 1> &recip_out) {
    using namespace pto;
    using tile_bf16   = Tile<Location::Vec, __bf16, TileM, BlockSize, BLayout::RowMajor, ValidM, BlockSize>;
    using tile_bf16_1 = Tile<Location::Vec, __bf16, TileM, BlockSize, BLayout::RowMajor, ValidM, 1>;
    using tile_f32_1  = Tile<Location::Vec, float,  TileM, BlockSize, BLayout::RowMajor, ValidM, 1>;
    // Reduce the block amax in the INPUT dtype (bf16) and cast ONLY the reduced
    // per-row amax to fp32, mirroring AscendC ComputeScaleCublas (reduce max in
    // T, Cast<float> the reduced max, tail_axis_fp8.h:736-739). The reduced result
    // stays boxed valid col=1 through the whole cuBLAS core: no TROWEXPAND, the
    // per-row scalar drives the bit-math and the fused data-pass TROWEXPANDMUL.
    tile_bf16 abs_x;
    TABS(abs_x, x_bf16);
    tile_bf16_1 max_r;
    TROWMAX(max_r, abs_x);          // reduce cols -> ValidCol == 1 (bf16)
    tile_f32_1 max_f;
    TCVT(max_f, max_r);             // cast only the reduced per-row amax
    compute_cublas_core<OutT, TileM, BlockSize, MaxLowBoundBits, ValidM, 1>(max_f, scale_byte, recip_out);
}

template <typename OutT, int TileM, int BlockSize>
void compute_dynamic_range_scale_tail(
    Tile<Location::Vec, uint16_t, TileM, BlockSize, BLayout::RowMajor> &x_u16,
    Tile<Location::Vec, uint16_t, TileM, BlockSize, BLayout::RowMajor> &scale_byte,
    Tile<Location::Vec, uint16_t, TileM, BlockSize, BLayout::RowMajor> &recip_out) {
    using namespace pto;
    using tile_u16 = Tile<Location::Vec, uint16_t, TileM, BlockSize, BLayout::RowMajor>;
    constexpr uint16_t EMAX = emax_bits<OutT>();
    constexpr uint16_t ADD_VALUE = BF16_ADD_VALUE_MAN1;

    tile_u16 abs_x;
    TANDS(abs_x, x_u16, BF16_ABS_MASK);
    tile_u16 max_abs;
    TROWMAX(max_abs, abs_x); // xMaxExp (full abs bits)

    tile_u16 xexp_only;
    TANDS(xexp_only, max_abs, BF16_EXP_MASK);
    tile_u16 eq_inf;
    TCMPS(eq_inf, xexp_only, BF16_EXP_MASK);

    // invalid = xexp_only < emax  ==  min(xexp_only, emax-1) == xexp_only
    tile_u16 t_lt;
    TMINS(t_lt, xexp_only, static_cast<uint16_t>(EMAX - 1));
    tile_u16 invalid;
    TCMP(invalid, t_lt, xexp_only);

    tile_u16 x_add; // (xMaxExp + addValue) exponent bits
    TADDS(x_add, max_abs, ADD_VALUE);
    TANDS(x_add, x_add, BF16_EXP_MASK);
    tile_u16 emax_tile;
    TEXPANDS(emax_tile, EMAX);
    TSEL(x_add, invalid, emax_tile); // invalid? emax : x_add

    tile_u16 shared_exp;
    TSUB(shared_exp, x_add, emax_tile);

    finalize_scale_recip_u16<TileM, BlockSize>(shared_exp, eq_inf, scale_byte, recip_out);
}

// ===========================================================================
// Not-tail-axis variants (reduce along BlockSize rows via TCOLMAX).
// ===========================================================================
template <typename OutT, int BlockSize, int TileN>
void compute_ocp_scale_not_tail(
    Tile<Location::Vec, uint16_t, BlockSize, TileN, BLayout::RowMajor> &x_u16,
    Tile<Location::Vec, uint16_t, BlockSize, TileN, BLayout::RowMajor> &scale_byte,
    Tile<Location::Vec, uint16_t, BlockSize, TileN, BLayout::RowMajor> &recip_out) {
    using namespace pto;
    using tile_u16 = Tile<Location::Vec, uint16_t, BlockSize, TileN, BLayout::RowMajor>;
    constexpr uint16_t EMAX = emax_bits<OutT>();

    tile_u16 exp_bits;
    TANDS(exp_bits, x_u16, BF16_EXP_MASK);
    tile_u16 max_exp;
    TCOLMAX(max_exp, exp_bits);

    tile_u16 eq_inf;
    TCMPS(eq_inf, max_exp, BF16_EXP_MASK);

    TMAXS(max_exp, max_exp, EMAX);
    tile_u16 emax_tile;
    TEXPANDS(emax_tile, EMAX);
    tile_u16 shared_exp;
    TSUB(shared_exp, max_exp, emax_tile);

    finalize_scale_recip_u16<BlockSize, TileN>(shared_exp, eq_inf, scale_byte, recip_out);
}

// Boxed (valid row=1) OCP not-tail scale. Same math as compute_ocp_scale_not_tail
// but the row reduction (TCOLMAX) collapses into a valid-row=1 tile so scale_byte
// and recip stay boxed (one per-column scalar per block-row) — mirrors the cuBLAS
// core so the caller can store one E8M0 byte per block (compact, reduce-axis
// collapsed + even-aligned) instead of a broadcast tile.
template <typename OutT, int BlockSize, int TileN, int ValidN = TileN>
void compute_ocp_scale_not_tail_boxed(
    Tile<Location::Vec, uint16_t, BlockSize, TileN, BLayout::RowMajor, BlockSize, ValidN> &x_u16,
    Tile<Location::Vec, uint16_t, BlockSize, TileN, BLayout::RowMajor, 1, ValidN> &scale_byte,
    Tile<Location::Vec, uint16_t, BlockSize, TileN, BLayout::RowMajor, 1, ValidN> &recip_out) {
    using namespace pto;
    using tile_full = Tile<Location::Vec, uint16_t, BlockSize, TileN, BLayout::RowMajor, BlockSize, ValidN>;
    using tile_box  = Tile<Location::Vec, uint16_t, BlockSize, TileN, BLayout::RowMajor, 1, ValidN>;
    constexpr uint16_t EMAX = emax_bits<OutT>();

    tile_full exp_bits;
    TANDS(exp_bits, x_u16, BF16_EXP_MASK);
    tile_box max_exp;
    TCOLMAX(max_exp, exp_bits);            // reduce rows -> valid row=1

    tile_box eq_inf;
    TCMPS(eq_inf, max_exp, BF16_EXP_MASK);

    TMAXS(max_exp, max_exp, EMAX);
    tile_box emax_tile;
    TEXPANDS(emax_tile, EMAX);
    tile_box shared_exp;
    TSUB(shared_exp, max_exp, emax_tile);

    finalize_scale_recip_u16<BlockSize, TileN, 1, ValidN>(shared_exp, eq_inf, scale_byte, recip_out);
}

template <typename OutT, int BlockSize, int TileN, uint32_t MaxLowBoundBits = CLAMP_MIN_BITS, int ValidN = TileN>
void compute_cublas_scale_not_tail(
    Tile<Location::Vec, __bf16,   BlockSize, TileN, BLayout::RowMajor, BlockSize, ValidN> &x_bf16,
    Tile<Location::Vec, uint16_t, BlockSize, TileN, BLayout::RowMajor, 1, ValidN> &scale_byte,
    Tile<Location::Vec, uint16_t, BlockSize, TileN, BLayout::RowMajor, 1, ValidN> &recip_out) {
    using namespace pto;
    using tile_bf16   = Tile<Location::Vec, __bf16, BlockSize, TileN, BLayout::RowMajor, BlockSize, ValidN>;
    using tile_bf16_1 = Tile<Location::Vec, __bf16, BlockSize, TileN, BLayout::RowMajor, 1, ValidN>;
    using tile_f32_1  = Tile<Location::Vec, float,  BlockSize, TileN, BLayout::RowMajor, 1, ValidN>;
    // Reduce block amax in the bf16 VALUE domain (TABS + bf16 TCOLMAX), then cast
    // ONLY the reduced per-column amax to fp32. The reduced result stays boxed
    // valid row=1 through the whole cuBLAS core: no TCOLEXPAND, the per-column
    // scalar drives the bit-math and the fused TCOLEXPANDMUL.
    //
    // DOMAIN NOTE vs AscendC: AscendC's ComputeScaleCublas reduces in the uint16
    // ABS-BIT domain (And(x, BF16_ABS_MASK) + uint16 Max). This bf16 value-domain
    // reduce is byte-identical to AscendC for all FINITE inputs: for non-negative
    // bf16 the abs-bit order is monotonic in magnitude, so value-max == bit-max.
    // The single-pass plain kernel can use bf16 TCOLMAX directly (no cross-chunk
    // seed, so no bf16 TEXPANDS -- which would crash the LinxV5 backend, see the
    // bigbs kernel). The ONE unverified edge is a block containing NaN: AscendC's
    // bit-max deterministically selects the NaN bits -> compute_cublas_core's
    // `finite` mask -> 0xff scale; this bf16 TCOLMAX depends on the hardware's
    // NaN-propagation semantics for max (untestable under the current skew). If
    // strict NaN-parity is ever required, switch this reduce to the uint16 abs-bit
    // domain (as the bigbs kernel does) at the cost of one reinterpret roundtrip.
    tile_bf16 abs_x;
    TABS(abs_x, x_bf16);
    tile_bf16_1 max_r;
    TCOLMAX(max_r, abs_x);          // reduce rows -> ValidRow == 1 (bf16)
    tile_f32_1 max_f;
    TCVT(max_f, max_r);             // cast only the reduced per-column amax
    compute_cublas_core<OutT, BlockSize, TileN, MaxLowBoundBits, 1, ValidN>(max_f, scale_byte, recip_out);
}

template <typename OutT, int BlockSize, int TileN>
void compute_dynamic_range_scale_not_tail(
    Tile<Location::Vec, uint16_t, BlockSize, TileN, BLayout::RowMajor> &x_u16,
    Tile<Location::Vec, uint16_t, BlockSize, TileN, BLayout::RowMajor> &scale_byte,
    Tile<Location::Vec, uint16_t, BlockSize, TileN, BLayout::RowMajor> &recip_out) {
    using namespace pto;
    using tile_u16 = Tile<Location::Vec, uint16_t, BlockSize, TileN, BLayout::RowMajor>;
    constexpr uint16_t EMAX = emax_bits<OutT>();
    constexpr uint16_t ADD_VALUE = BF16_ADD_VALUE_MAN1;

    tile_u16 abs_x;
    TANDS(abs_x, x_u16, BF16_ABS_MASK);
    tile_u16 max_abs;
    TCOLMAX(max_abs, abs_x);

    tile_u16 xexp_only;
    TANDS(xexp_only, max_abs, BF16_EXP_MASK);
    tile_u16 eq_inf;
    TCMPS(eq_inf, xexp_only, BF16_EXP_MASK);

    tile_u16 t_lt;
    TMINS(t_lt, xexp_only, static_cast<uint16_t>(EMAX - 1));
    tile_u16 invalid;
    TCMP(invalid, t_lt, xexp_only);

    tile_u16 x_add;
    TADDS(x_add, max_abs, ADD_VALUE);
    TANDS(x_add, x_add, BF16_EXP_MASK);
    tile_u16 emax_tile;
    TEXPANDS(emax_tile, EMAX);
    TSEL(x_add, invalid, emax_tile);

    tile_u16 shared_exp;
    TSUB(shared_exp, x_add, emax_tile);

    finalize_scale_recip_u16<BlockSize, TileN>(shared_exp, eq_inf, scale_byte, recip_out);
}

} // namespace supernpu::tile_isa::mxquant

#endif
