#ifndef SUPERNPU_DYNAMIC_MX_QUANT_NONTAIL_OCP_FP4_BIGBS_HPP
#define SUPERNPU_DYNAMIC_MX_QUANT_NONTAIL_OCP_FP4_BIGBS_HPP

#include "multi_thread/quant/dynamic_mx_quant/dynamic_mx_quant_common.hpp"

namespace supernpu::tile_isa::mxquant {

// Non-tail-axis, OCP scale (scaleAlg=0), FP4 output — LARGE-BlockSize branch
// (方案A: split reduce axis + register accumulation). See RECORD 问题2 /
// DESIGN §7.5.
//
// Why a separate kernel: the plain non-tail kernel loads the whole
// [BlockSize, TileN] block in one shot, so the contiguous axis TileN carries
// BOTH the fp4 32B alignment LOWER bound (TileN % 64 == 0) and the TileSize
// UPPER bound (Rows*Cols*sizeof <= 8192 -> TileN <= 4096/BlockSize on the 16b
// input tile). Valid TileN exists iff lower <= upper, which fails once
// BlockSize >= 128 (upper < 64). This branch breaks the Rows*Cols product by
// tiling the REDUCE axis (rows, length BlockSize) into R_sub-row sub-chunks:
// TileSize now binds R_sub*TileN <= 4096 with R_sub a free knob, so TileN can
// always meet the 64 alignment regardless of BlockSize.
//
// Pass 1 (reduce): for each R_sub sub-chunk, TANDS(exp) + TCOLMAX -> partial
// [1, TileN], accumulate into a register-resident running-max via elementwise
// TMAX (max is associative, so cross-sub-chunk accumulation is exact). No shared
// memory: the accumulator is loop-carried within one PE-thread. After the last
// sub-chunk, finalize once (ocp_scale_from_maxexp_not_tail_boxed) -> scale_byte
// + recip, store one E8M0 byte per block.
//
// Pass 2 (data): reinterpret recip -> bf16 -> fp32 inv_scale (per-column
// scalar, valid row=1); for each R_sub sub-chunk re-load x, TCVT bf16->fp32,
// TCOLEXPANDMUL broadcasts the per-column inv_scale to all R_sub rows, TCVT
// fp32->fp4, TSTORE. The same per-column scale applies to every sub-chunk row.
//
// scale layout: E8M0 1 byte/block, planar [scaleRows, Post] — same as
// dynamic_mx_quant_nontail_ocp_fp4. This IS the PTO-ISA Shared B-scale [G,N]
// contract (ADR-0101); NO parity interleave needed (RECORD 问题5 dissolved).

// Supported BlockSize range (方案A, split reduce axis):
//   BlockSize ∈ { multiples of R_sub, R_sub..∞ }  (R_sub | BlockSize)
// Unlike the plain `dynamic_mx_quant_nontail_ocp_fp4` (single-load, capped at
// BlockSize ≤ 64 because fp4 alignment lower bound 64 > TileSize upper 4096/BS
// once BS ≥ 128), this branch decouples them: TileSize now binds R_sub*TileN
// (a free knob) instead of BlockSize*TileN, so ANY BlockSize is legal as long
// as R_sub | BlockSize and R_sub*TileN ≤ 4096. Intended for the BS ≥ 128 range
// the plain kernel cannot cover; small BS also works but the plain kernel is
// cheaper there (no split/re-read). Default R_sub=32, TileN=64.
template <int Axis, int Post, int BlockSize, int TileN = 64, int R_sub = 32,
          typename OutT = __fp4_e2m1x2, typename InT = __bf16>
void dynamic_mx_quant_nontail_ocp_fp4_bigbs(InT *x, OutT *y, uint8_t *scale) {
    static_assert(Axis > 0 && Post > 0, "dims must be positive");
    static_assert(Axis % BlockSize == 0, "Axis must be multiple of BlockSize");
    static_assert(Post % TileN == 0, "Post must be multiple of TileN");
    static_assert(TileN % 64 == 0,
                  "fp4 output tile is plain RowMajor NoneBox: (TileN/2)*8 % 256 == 0 "
                  "requires TileN a multiple of 64 (>=2 MX blocks along Post)");
    // BlockSize range: any multiple of R_sub (方案A decouples reduce axis).
    static_assert(BlockSize % R_sub == 0,
                  "BlockSize must be a multiple of R_sub (方案A splits the reduce axis "
                  "into R_sub-row sub-chunks); supported BlockSize = R_sub, 2*R_sub, ...");
    // 16b sub-chunk input tile: R_sub*TileN elems, <=4096 (Rows*Cols*2 <=8192).
    static_assert(R_sub * TileN <= 4096,
                  "sub-chunk R_sub*TileN must be <= 4096 (16b TileSize budget)");

    constexpr int numKb  = Axis / BlockSize;
    constexpr int numSub = BlockSize / R_sub;
    constexpr int numN   = Post / TileN;
    // reduce-axis block count, even-aligned (padding block-row left zero).
    constexpr int scaleRows = ((numKb + 1) / 2) * 2;

    using namespace pto;

    // Sub-chunk data/input tiles: physical [R_sub, TileN].
    using tile_x  = Tile<Location::Vec, InT,      R_sub, TileN, BLayout::RowMajor>;
    using tile_f  = Tile<Location::Vec, float,    R_sub, TileN, BLayout::RowMajor>;
    // fp4 输出 tile：ELEMENT-列形（physical Cols=TileN，与源 fp32 同列），TCVT 走 fp4 打包
    //   specialization；存储侧 Post/2 字节经 gm_y byte 域 + 字节基址折叠 y_iter。旧
    //   [R_sub, TileN/2] 字节列约定会让 TCVT 落 ordinary 路径编译崩（同 nontail_ocp_fp4 plain /
    //   tail_ocp_fp4 的 element-列迁移，0.58.3 工具链头已删 32B 列对齐 assert）。
    using tile_o  = Tile<Location::Vec, OutT,     R_sub, TileN, BLayout::RowMajor>;
    // colReduce（TCOLMAX）输出行向量 + running-max 累加器：physical row=1（physCol=validCol=TileN，
    //   模型 Block.cpp:2349 反推恰得 row=1；见 RECORD 问题22 补充）。
    using tile_maxf      = Tile<Location::Vec, float,    1, TileN, BLayout::RowMajor, 1, TileN>;
    using tile_se8m0     = Tile<Location::Vec, __fp8_e8m0, 1, TileN, BLayout::RowMajor, 1, TileN>;
    using tile_recip_bf1 = Tile<Location::Vec, __bf16,   1, TileN, BLayout::RowMajor, 1, TileN>;
    using tile_recip_f1  = Tile<Location::Vec, float,    1, TileN, BLayout::RowMajor, 1, TileN>;

    using gm_x  = global_tensor<InT,      RowMajor<Axis, Post>>;
    using gm_y  = global_tensor<uint8_t,  RowMajor<Axis, Post / 2>>;
    // scale: uint8 E8M0, compact planar [scaleRows, Post], one byte per block.
    using gm_s  = global_tensor<__fp8_e8m0, RowMajor<scaleRows, Post>>;

    global_iterator<gm_x,  tile_x>  x_iter(x);

    for (int kb = 0; kb < numKb; ++kb) {
        for (int n = 0; n < numN; ++n) {
            // ---- Pass 1: 值域分裂归约（fp32 统一累加器，跨 R_sub sub-chunk）----
            // 半/bf16 先 TCVT 到 fp32 再 TABS（TABS 白名单仅 FP16/FP32，bf16 被拒；
            // half→fp32 无损，max 结果一致），fp32 直接 TABS；避免 half-scalar 种子风险。
            tile_maxf max_acc;
            TEXPANDS(max_acc, 0.0f);                          // running-max 种子（abs>=0，0 为幺元）
            for (int s = 0; s < numSub; ++s) {
                // absolute sub-chunk row-block index; R_sub == tile height ==
                // iterator i-stride, so no base-pointer fold needed for x.
                const int rb = kb * numSub + s;
                auto gx = x_iter(rb, n);
                tile_x xin;
                TLOAD(xin, gx);
                tile_f abs_f;
                if constexpr (std::is_same_v<InT, float>) {
                    TABS(abs_f, xin);                         // xin 即 float
                } else {
                    tile_f xf; TCVT(xf, xin);                 // half/bf16 -> fp32
                    TABS(abs_f, xf);
                }
                tile_maxf partial;
                TCOLMAX(partial, abs_f);                      // reduce 行 -> valid row=1
                TMAX(max_acc, max_acc, partial);              // 跨 sub-chunk 累加（max 结合律）
            }

            // ---- 取指数 + 乘 2^-emax + 直转 e8m0 ----
            tile_recip_bf1 max_bf;
            TCVT(max_bf, max_acc);                            // fp32 -> bf16
            auto max_u16 = reinterpret_tile<uint16_t>(max_bf);
            TANDS(max_u16, max_u16, BF16_EXP_MASK);
            tile_recip_bf1 shared_bf;
            TMULS(shared_bf, max_bf, __builtin_bit_cast(__bf16, recip_emax_bits<OutT>()));
            tile_se8m0 scale_e8m0;
            TCVT(scale_e8m0, shared_bf);                      // bf16 -> e8m0（inf/nan->0xff 硬件）

            // ---- finalize_recip_u16 内联（问题8）：同载体 uint16 视图（同 plain）----
            auto shared_u16 = reinterpret_tile<uint16_t>(shared_bf);
            tile_recip_bf1 recip_bf, eqinf_bf, eqzero_bf, eqspc_bf, k_bf;
            auto recip_u16  = reinterpret_tile<uint16_t>(recip_bf);
            auto eq_inf     = reinterpret_tile<uint16_t>(eqinf_bf);
            auto eq_zero    = reinterpret_tile<uint16_t>(eqzero_bf);
            auto eq_special = reinterpret_tile<uint16_t>(eqspc_bf);
            auto k_u16      = reinterpret_tile<uint16_t>(k_bf);
            TCMPS(eq_inf,     max_u16,    BF16_EXP_MASK);              // NOT finite
            TCMPS(eq_zero,    max_u16,    static_cast<uint16_t>(0));   // 全零块
            TCMPS(eq_special, shared_u16, BF16_EXP_BIAS);             // shared==0x7f00
            TEXPANDS(k_u16, BF16_EXP_BIAS);
            TSUB(recip_u16, k_u16, shared_u16);                       // 0x7f00 - shared
            TEXPANDS(k_u16, BF16_NAN_PATTERN);   TSEL(recip_u16, eq_inf, k_u16);       // inf -> 0x7f81
            TEXPANDS(k_u16, static_cast<uint16_t>(0)); TSEL(recip_u16, eq_zero, k_u16);// 全零 -> 0
            TEXPANDS(k_u16, BF16_SPECIAL_EXP);   TSEL(recip_u16, eq_special, k_u16);   // special -> 0x0040

            // Compact scale store: fold block-row index (kb) into the base
            // pointer (iterator i-stride is physical tile height, not 1). Each
            // block-row writes TileN bytes at scale + kb*Post. The new OCP path
            // produces the E8M0 byte directly (Cast<bf16->e8m0>), so it stores
            // with no narrowing TCVT.
            global_iterator<gm_s, tile_se8m0> s_iter(
                reinterpret_cast<__fp8_e8m0 *>(scale) + kb * Post);
            auto gs = s_iter(0, n);
            // stored as PLAIN planar [scaleRows, Post] = PTO-ISA Shared B-scale
            // [G,N] (ADR-0101). NO parity interleave: AscendC's Reg::Interleave /
            // [ceil(numKb/2), Post, 2] zip is an Ascend packing convention, not the
            // PTO-ISA scale contract (RECORD 问题5 dissolved 2026-09-03).
            TSTORE(gs, scale_e8m0);

            // ---- Pass 2: apply per-column inv_scale, cast to fp4 ----
            tile_recip_f1 inv_scale_f;
            TCVT(inv_scale_f, recip_bf);                      // 问题4 消除：recip_bf 直接转 fp32

            for (int s = 0; s < numSub; ++s) {
                const int rb = kb * numSub + s;
                auto gx = x_iter(rb, n);
                // fp4 输出 element-列形对字节域 gm_y：字节基址折叠——子块行 rb 偏
                //   rb*R_sub 行 × (Post/2) 字节行距，列块 n 偏 n*(TileN/2) 字节。
                global_iterator<gm_y, tile_o> y_iter(
                    reinterpret_cast<uint8_t *>(y) + rb * R_sub * (Post / 2) + n * (TileN / 2));
                auto gy = y_iter(0, 0);
                tile_x xq;
                TLOAD(xq, gx);
                tile_o oq;
                if constexpr (std::is_same_v<InT, float>) {
                    TCOLEXPANDMUL(xq, xq, inv_scale_f); // fp32 domain mul (no pre-cast)
                    TCVT(oq, xq); // fp32 -> packed fp4_e2m1x2 (Post halved)
                } else {
                    tile_f xf;
                    TCVT(xf, xq); // bf16/half -> fp32
                    TCOLEXPANDMUL(xf, xf, inv_scale_f); // per-column scalar broadcast
                    TCVT(oq, xf); // fp32 -> packed fp4_e2m1x2 (Post halved)
                }
                TSTORE(gy, oq);
            }
        }
    }
}

} // namespace supernpu::tile_isa::mxquant

#endif
