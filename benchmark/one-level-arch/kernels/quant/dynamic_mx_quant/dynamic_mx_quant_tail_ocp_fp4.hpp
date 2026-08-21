#ifndef SUPERNPU_DYNAMIC_MX_QUANT_TAIL_OCP_FP4_HPP
#define SUPERNPU_DYNAMIC_MX_QUANT_TAIL_OCP_FP4_HPP

#include "quant/dynamic_mx_quant/dynamic_mx_quant_common.hpp"

namespace supernpu::tile_isa::mxquant {

// Tail-axis, OCP scale (scaleAlg=0), FP4 output (E2M1 default, E1M2 valid).
// emax derived from OutT (fp4_e2m1 -> 0x0100). Two-pass (full + M_tail) structure
// mirrors dynamic_mx_quant_tail_cublas_fp8: everything is inlined, the tail is an
// `if constexpr (M_tail>0)` block with M_tail baked into the tile aliases.
//
// SCHEME (padded-physical column-box). A single MX block of fp4 output is
// BlockSize/2 packed bytes (16B at BlockSize=32), which fails the 32B column
// alignment (pto_tile.hpp:408) whenever BlockSize/2 % 32 != 0. Instead of packing
// two blocks per output tile, we PAD the PHYSICAL tile width to
//   PW = next multiple of 64 >= BlockSize
// and COLUMN-BOX every op to the real BlockSize:
//   - value/scale tiles (uint16/bf16/fp32): physical PW, valid BlockSize. TROWMAX,
//     TCVT and TROWEXPANDMUL operate on ValidCol=BlockSize only (cpu_sim
//     TRowMax.hpp:13 loops j<ValidCol), so each block's reduction/scale stays
//     independent and correct; the PW padding is register-only.
//   - fp4 output tile: physical PW/2 bytes (32B-aligned because PW % 64 == 0),
//     valid BlockSize/2 bytes.
//   - boxed TLOAD/TSTORE transfer ValidCol columns, NOT the physical PW
//     (TLoadBackend blk_tload count = GetValidCol()), so the trailing block
//     reads/writes only its real BlockSize columns -> no HBM over-read past N.
//   - base-pointer fold places each block at its true column (value input
//     x + kb*BlockSize; fp4 output y + kb*(BlockSize/2)), since the physical
//     width PW != the per-block stride BlockSize, so we cannot let the iterator
//     step kb by the tile width.
// No scratch-HBM concat, no 2-block pairing. When BlockSize % 64 == 0, PW ==
// BlockSize and the col-box collapses to a full tile. Needs only N % BlockSize == 0.
//
// ALTERNATIVE (kept, not deleted): the SAME tile-splitting problem also has a
// 2-block scratch-HBM concat solution — reduce two [TileM,BlockSize] blocks
// independently, TSTORE both fp32 halves into a scratch cat_buf, TLOAD the
// [TileM,2*BlockSize] tile and emit ONE fp32->fp4 TCVT of width BlockSize; an odd
// trailing block is a synthesized zero block. That version lives in
// dynamic_mx_quant_tail_ocp_fp4.hpp.bak. The two schemes are behaviorally
// equivalent but mutually exclusive; which one to keep is DEFERRED until the
// toolchain<->emulator skew is resolved and both can be runtime-compared. This
// file (padded-physical column-box) is the current default.
//
// ODD numKb (scale even-pad): the AscendC scale layout is even-block-aligned
// (golden _pad_to_even pads with 2^-127, whose E8M0 byte is 0x00). We process
// only the numKb real blocks, so when numKb is odd we explicitly write one 0x00
// E8M0 byte to the padding scale column scale[numKb]. The data output is
// naturally N/2 real bytes/row (no padding data emitted).
//
// TileM is NOT a caller knob: it is DERIVED at compile time from M + the InT
// binding-tile budget. CRITICAL: this kernel binds ONE MX block per tile at the
// PADDED physical width PW (not BlockSize), so the budget contig axis passed to
// max_tilem is PW, not BlockSize (max_tilem<M, PW, InT, /*IsCublas=*/false>()).
// InT drives BOTH the budget (a wider input dtype shrinks TileM) AND the compute
// domain: the scale-pass reduce and data-pass mul are InT-dispatched (half in
// half domain, fp32 in fp32 domain, bf16 pre-cast to fp32), mirroring the
// newcalc probe. See the scale-path comment for the TROWMAX/TABS whitelist.
template <int M, int N, int BlockSize = 32, typename OutT = __fp4_e2m1x2,
          typename InT = __bf16>
void dynamic_mx_quant_tail_ocp_fp4(InT *x, OutT *y, uint8_t *scale) {
    static_assert(M > 0 && N > 0, "dim must be positive");
    static_assert(N % BlockSize == 0, "N must be multiple of BlockSize");
    static_assert(BlockSize % 32 == 0,
                  "fp4 output block is BlockSize/2 packed bytes; BlockSize must be "
                  "a multiple of 32 so the padded physical fp4 width PW/2 is "
                  "32B-column-aligned (pto_tile.hpp:408)");
    static_assert(std::is_same_v<InT, __bf16> || std::is_same_v<InT, __half> ||
                      std::is_same_v<InT, float>,
                  "InT must be one of {__bf16, __half, float}");

    using namespace pto;

    // Physical tile width padded to the next multiple of 64 so the packed fp4
    // output tile (physical PW/2 bytes) is 32B-column-aligned. BlockSize % 64 == 0
    // -> PW == BlockSize (col-box collapses to a full tile).
    constexpr int PW = ((BlockSize + 63) / 64) * 64;
    // Contig axis for the budget is PW (one padded block per tile), NOT BlockSize.
    constexpr int TileM  = max_tilem<M, PW, InT, /*IsCublas=*/false>();
    constexpr int full_m = M / TileM;
    constexpr int M_tail = M % TileM;
    constexpr int numKb  = N / BlockSize;
    // AscendC scale layout: uint8 E8M0, one byte per block, compact [M, scaleCols]
    // with the block count even-aligned. Mirrors dynamic_mx_quant_tail_axis.h:217.
    constexpr int scaleCols = ((numKb + 1) / 2) * 2;
    constexpr bool oddTail  = (numKb % 2) != 0;   // padding scale col must be 0x00

    uint8_t *y_u8 = reinterpret_cast<uint8_t *>(y);

    using gm_x  = global_tensor<InT,      RowMajor<M, N>>;
    using gm_y  = global_tensor<uint8_t,  RowMajor<M, N / 2>>;
    using gm_s  = global_tensor<__fp8_e8m0, RowMajor<M, scaleCols>>;

    // Full-tile pass (ValidRow == TileM; boxed row collapses to NoneBox).
    {
        using tile_x         = Tile<Location::Vec, InT,      TileM, PW,     BLayout::RowMajor, TileM, BlockSize>;
        using tile_f         = Tile<Location::Vec, float,    TileM, PW,     BLayout::RowMajor, TileM, BlockSize>;
        using tile_maxh      = Tile<Location::Vec, __half,   TileM, PW,     BLayout::RowMajor, TileM, 1>;
        using tile_maxf      = Tile<Location::Vec, float,    TileM, PW,     BLayout::RowMajor, TileM, 1>;
        using tile_recip_bf1 = Tile<Location::Vec, __bf16,   TileM, PW,     BLayout::RowMajor, TileM, 1>;
        using tile_recip_f1  = Tile<Location::Vec, float,    TileM, PW,     BLayout::RowMajor, TileM, 1>;
        using tile_se8m0     = Tile<Location::Vec, __fp8_e8m0, TileM, PW,     BLayout::RowMajor, TileM, 1>;
        using tile_o         = Tile<Location::Vec, OutT,     TileM, PW / 2, BLayout::RowMajor, TileM, BlockSize / 2>;

        for (int m = 0; m < full_m; ++m) {
            for (int kb = 0; kb < numKb; ++kb) {
                // --- scale path: value-domain reduce (mirrors newcalc probe) ---
                // TROWMAX 的 emulator 白名单只含 FP16/FP32/INT32
                // (AccumulateBlockInfo.cpp:550)，拒绝 BF16/UINT16；TABS 白名单同样
                // 只含 FP16/FP32(:229)。故按 InT 分派归约域：
                //   half -> half 域 TABS+TROWMAX;
                //   fp32 -> fp32 域 TABS+TROWMAX;
                //   bf16 -> 先 TCVT bf16->fp32，再在 fp32 域归约（bf16 同时被
                //           TROWMAX 与 TABS 拒绝，转 fp32 一并规避）。
                // 归约得每 block 的 |max|，TCVT 到 bf16 后取指数位得 2^E_max。
                global_iterator<gm_x, tile_x> x_iter(x + kb * BlockSize);
                auto gx = x_iter(m, 0);
                tile_x xin;
                TLOAD(xin, gx);

                tile_recip_bf1 max_bf;
                if constexpr (std::is_same_v<InT, __half>) {
                    tile_x abs_h;
                    TABS(abs_h, xin);
                    tile_maxh max_h;
                    TROWMAX(max_h, abs_h);
                    TCVT(max_bf, max_h);                  // half -> bf16
                } else if constexpr (std::is_same_v<InT, float>) {
                    tile_x abs_f;
                    TABS(abs_f, xin);
                    tile_maxf max_f;
                    TROWMAX(max_f, abs_f);
                    TCVT(max_bf, max_f);                  // fp32 -> bf16
                } else {
                    tile_f xf32;
                    TCVT(xf32, xin);                      // bf16 -> fp32（规避 TROWMAX/TABS 拒 bf16）
                    tile_f abs_f;
                    TABS(abs_f, xf32);
                    tile_maxf max_f;
                    TROWMAX(max_f, abs_f);
                    TCVT(max_bf, max_f);                  // fp32 -> bf16
                }

                // 清尾数只留 2^E_max，乘 2^-emax -> shared = 2^(E_max-emax)。
                // emax 由 OutT 派生（recip_emax_bits<OutT>()），不硬编码探针的 e4m3 值。
                auto max_u16 = reinterpret_tile<uint16_t>(max_bf);
                TANDS(max_u16, max_u16, BF16_EXP_MASK);
                tile_recip_bf1 shared_bf;
                TMULS(shared_bf, max_bf,
                      __builtin_bit_cast(__bf16, recip_emax_bits<OutT>()));
                tile_se8m0 scale_e8m0;
                TCVT(scale_e8m0, shared_bf);              // bf16 -> e8m0 直转
                global_iterator<gm_s, tile_se8m0> s_iter(reinterpret_cast<__fp8_e8m0 *>(scale) + kb);
                auto gs = s_iter(m, 0);
                TSTORE(gs, scale_e8m0);

                // --- recip finalize：位补主路径 + inf/zero/special 三 Select ---
                // 内联 common::finalize_recip_u16（规避问题8：tile 作真实函数入参会被
                // lower 成 S64 栈往返、gfrun 拒），逐行对齐 AscendC ocp_new
                // ComputeScaleOcp (bak/..._ocp_new.h:90-94)。scale byte 靠上方
                // TCVT(scale_e8m0) 的 Cast<e8m0>(inf)→0xff，无需 Select；recip 保留
                // 三类特殊值写回（探针是 MINIMAL 版故意省略，业务 kernel 必须补齐）：
                //   inf/nan (max_exp==0x7f80) -> 0x7f81；全零 (max_exp==0) -> 0；
                //   special (shared==0x7f00) -> 0x0040（位补公式在此点算出 0，须修正）。
                // 主路径 recip = 0x7f00 - shared：二元 TSUB（收 uint16；标量 TSUBS 不收）。
                // eq_inf/eq_zero 取自 pre-multiply 的 max_exp（= 清尾数后的 max_u16，
                // TMULS 写 shared_bf 未改 max_bf，视图仍有效）。
                // TSUB/TSEL 要求 dst/src 同 tile_shape（jcore/TSub.hpp:51、
                // template_asm.hpp:5465 三参同型）；唯 TCMPS 允许 out≠in。故所有参与
                // TSUB/TSEL 的 uint16 量都以 reinterpret_tile<uint16_t> 视图落在同一
                // tile_recip_bf1 载体上——同源 Tile 类型的等宽视图彼此同型
                // （ReinterpretedTileView<uint16_t,tile_recip_bf1>），与 shared_u16/
                // max_u16 一致；k_u16 复用一块常数载体逐 Select 重填。
                auto shared_u16 = reinterpret_tile<uint16_t>(shared_bf);
                tile_recip_bf1 recip_bf, eqinf_bf, eqzero_bf, eqspc_bf, k_bf;
                auto recip_u16  = reinterpret_tile<uint16_t>(recip_bf);
                auto eq_inf     = reinterpret_tile<uint16_t>(eqinf_bf);
                auto eq_zero    = reinterpret_tile<uint16_t>(eqzero_bf);
                auto eq_special = reinterpret_tile<uint16_t>(eqspc_bf);
                auto k_u16      = reinterpret_tile<uint16_t>(k_bf);
                TCMPS(eq_inf,     max_u16,    BF16_EXP_MASK);              // NOT finite
                TCMPS(eq_zero,    max_u16,    static_cast<uint16_t>(0));   // all-zero block
                TCMPS(eq_special, shared_u16, BF16_EXP_BIAS);             // shared==0x7f00
                TEXPANDS(k_u16, BF16_EXP_BIAS);
                TSUB(recip_u16, k_u16, shared_u16);                       // 0x7f00 - shared
                TEXPANDS(k_u16, BF16_NAN_PATTERN);
                TSEL(recip_u16, eq_inf, k_u16);                           // inf 命中 -> 0x7f81
                TEXPANDS(k_u16, static_cast<uint16_t>(0));
                TSEL(recip_u16, eq_zero, k_u16);                          // 全零命中 -> 0
                TEXPANDS(k_u16, BF16_SPECIAL_EXP);
                TSEL(recip_u16, eq_special, k_u16);                       // special 命中 -> 0x0040
                tile_recip_f1 recip_f;
                TCVT(recip_f, recip_bf);                                  // bf16 -> fp32

                // --- data path: col-boxed load, fp32 scale, narrowed fp4 store ---
                tile_x xq;
                TLOAD(xq, gx);
                tile_o oq;
                if constexpr (std::is_same_v<InT, float>) {
                    TROWEXPANDMUL(xq, xq, recip_f);     // fp32 domain mul (no pre-cast)
                    TCVT(oq, xq);                       // fp32 -> fp4
                } else {
                    tile_f xf;
                    TCVT(xf, xq);                       // bf16/half -> fp32
                    TROWEXPANDMUL(xf, xf, recip_f);     // per-row scalar broadcast-mul
                    TCVT(oq, xf);                       // fp32 (valid BlockSize) -> fp4 (valid BlockSize/2 bytes)
                }
                global_iterator<gm_y, tile_o> y_iter(y_u8 + kb * (BlockSize / 2));
                auto gy = y_iter(m, 0);
                TSTORE(gy, oq);                       // narrowed store at byte kb*(BlockSize/2)
            }
            // Even-pad the odd trailing scale column with a 0x00 E8M0 byte
            // (golden _pad_to_even uses 2^-127 == E8M0 0x00).
            if constexpr (oddTail) {
                tile_se8m0 zpad;
                TEXPANDS(zpad, __builtin_bit_cast(__fp8_e8m0, static_cast<uint8_t>(0)));
                global_iterator<gm_s, tile_se8m0> zs_iter(reinterpret_cast<__fp8_e8m0 *>(scale) + numKb);
                auto gzs = zs_iter(m, 0);
                TSTORE(gzs, zpad);
            }
        }
    }

    // Tail rows: M_tail (< TileM), boxed to ValidRow = M_tail; row block full_m.
    if constexpr (M_tail > 0) {
        using tile_x         = Tile<Location::Vec, InT,      TileM, PW,     BLayout::RowMajor, M_tail, BlockSize>;
        using tile_f         = Tile<Location::Vec, float,    TileM, PW,     BLayout::RowMajor, M_tail, BlockSize>;
        using tile_maxh      = Tile<Location::Vec, __half,   TileM, PW,     BLayout::RowMajor, M_tail, 1>;
        using tile_maxf      = Tile<Location::Vec, float,    TileM, PW,     BLayout::RowMajor, M_tail, 1>;
        using tile_recip_bf1 = Tile<Location::Vec, __bf16,   TileM, PW,     BLayout::RowMajor, M_tail, 1>;
        using tile_recip_f1  = Tile<Location::Vec, float,    TileM, PW,     BLayout::RowMajor, M_tail, 1>;
        using tile_se8m0     = Tile<Location::Vec, __fp8_e8m0, TileM, PW,     BLayout::RowMajor, M_tail, 1>;
        using tile_o         = Tile<Location::Vec, OutT,     TileM, PW / 2, BLayout::RowMajor, M_tail, BlockSize / 2>;

        for (int kb = 0; kb < numKb; ++kb) {
            // scale path: value-domain reduce (mirrors newcalc probe; TROWMAX/TABS
            // 白名单只含 FP16/FP32 -> half 走 half 域, fp32 走 fp32 域, bf16 先转 fp32)。
            global_iterator<gm_x, tile_x> x_iter(x + kb * BlockSize);
            auto gx = x_iter(full_m, 0);
            tile_x xin;
            TLOAD(xin, gx);

            tile_recip_bf1 max_bf;
            if constexpr (std::is_same_v<InT, __half>) {
                tile_x abs_h;
                TABS(abs_h, xin);
                tile_maxh max_h;
                TROWMAX(max_h, abs_h);
                TCVT(max_bf, max_h);                  // half -> bf16
            } else if constexpr (std::is_same_v<InT, float>) {
                tile_x abs_f;
                TABS(abs_f, xin);
                tile_maxf max_f;
                TROWMAX(max_f, abs_f);
                TCVT(max_bf, max_f);                  // fp32 -> bf16
            } else {
                tile_f xf32;
                TCVT(xf32, xin);                      // bf16 -> fp32（规避 TROWMAX/TABS 拒 bf16）
                tile_f abs_f;
                TABS(abs_f, xf32);
                tile_maxf max_f;
                TROWMAX(max_f, abs_f);
                TCVT(max_bf, max_f);                  // fp32 -> bf16
            }

            auto max_u16 = reinterpret_tile<uint16_t>(max_bf);
            TANDS(max_u16, max_u16, BF16_EXP_MASK);
            tile_recip_bf1 shared_bf;
            TMULS(shared_bf, max_bf,
                  __builtin_bit_cast(__bf16, recip_emax_bits<OutT>()));
            tile_se8m0 scale_e8m0;
            TCVT(scale_e8m0, shared_bf);              // bf16 -> e8m0 直转
            global_iterator<gm_s, tile_se8m0> s_iter(reinterpret_cast<__fp8_e8m0 *>(scale) + kb);
            auto gs = s_iter(full_m, 0);
            TSTORE(gs, scale_e8m0);

            // recip finalize：位补主路径 + inf/zero/special 三 Select
            // 内联 common::finalize_recip_u16（规避问题8），对齐 ocp_new:90-94；
            // scale byte 靠上方 Cast<e8m0>(inf)→0xff，recip 补三类特殊值写回。
            // 同型约束见 full-loop 展开注释：TSUB/TSEL 三参同 tile_shape，全部 uint16
            // 量以 reinterpret_tile<uint16_t> 视图落在同一 tile_recip_bf1 载体上。
            auto shared_u16 = reinterpret_tile<uint16_t>(shared_bf);
            tile_recip_bf1 recip_bf, eqinf_bf, eqzero_bf, eqspc_bf, k_bf;
            auto recip_u16  = reinterpret_tile<uint16_t>(recip_bf);
            auto eq_inf     = reinterpret_tile<uint16_t>(eqinf_bf);
            auto eq_zero    = reinterpret_tile<uint16_t>(eqzero_bf);
            auto eq_special = reinterpret_tile<uint16_t>(eqspc_bf);
            auto k_u16      = reinterpret_tile<uint16_t>(k_bf);
            TCMPS(eq_inf,     max_u16,    BF16_EXP_MASK);              // NOT finite
            TCMPS(eq_zero,    max_u16,    static_cast<uint16_t>(0));   // all-zero block
            TCMPS(eq_special, shared_u16, BF16_EXP_BIAS);             // shared==0x7f00
            TEXPANDS(k_u16, BF16_EXP_BIAS);
            TSUB(recip_u16, k_u16, shared_u16);                      // 0x7f00 - shared
            TEXPANDS(k_u16, BF16_NAN_PATTERN);
            TSEL(recip_u16, eq_inf, k_u16);                           // inf 命中 -> 0x7f81
            TEXPANDS(k_u16, static_cast<uint16_t>(0));
            TSEL(recip_u16, eq_zero, k_u16);                          // 全零命中 -> 0
            TEXPANDS(k_u16, BF16_SPECIAL_EXP);
            TSEL(recip_u16, eq_special, k_u16);                       // special 命中 -> 0x0040
            tile_recip_f1 recip_f;
            TCVT(recip_f, recip_bf);                                  // bf16 -> fp32

            tile_x xq;
            TLOAD(xq, gx);
            tile_o oq;
            if constexpr (std::is_same_v<InT, float>) {
                TROWEXPANDMUL(xq, xq, recip_f);     // fp32 domain mul (no pre-cast)
                TCVT(oq, xq);                       // fp32 -> fp4
            } else {
                tile_f xf;
                TCVT(xf, xq);                       // bf16/half -> fp32
                TROWEXPANDMUL(xf, xf, recip_f);
                TCVT(oq, xf);
            }
            global_iterator<gm_y, tile_o> y_iter(y_u8 + kb * (BlockSize / 2));
            auto gy = y_iter(full_m, 0);
            TSTORE(gy, oq);
        }
        if constexpr (oddTail) {
            tile_se8m0 zpad;
            TEXPANDS(zpad, __builtin_bit_cast(__fp8_e8m0, static_cast<uint8_t>(0)));
            global_iterator<gm_s, tile_se8m0> zs_iter(reinterpret_cast<__fp8_e8m0 *>(scale) + numKb);
            auto gzs = zs_iter(full_m, 0);
            TSTORE(gzs, zpad);
        }
    }
}

} // namespace supernpu::tile_isa::mxquant

#endif
