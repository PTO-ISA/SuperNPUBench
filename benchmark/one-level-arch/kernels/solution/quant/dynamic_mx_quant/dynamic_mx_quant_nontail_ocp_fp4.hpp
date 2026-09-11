#ifndef SUPERNPU_DYNAMIC_MX_QUANT_NONTAIL_OCP_FP4_HPP
#define SUPERNPU_DYNAMIC_MX_QUANT_NONTAIL_OCP_FP4_HPP

#include "solution/quant/dynamic_mx_quant/dynamic_mx_quant_common.hpp"

namespace supernpu::tile_isa::mxquant {

// Non-tail-axis, OCP scale (scaleAlg=0), FP4 output (E2M1 default, E1M2 valid).
// Quantize axis is rows (TCOLMAX); fp4 packs 2/byte along the contiguous Post
// axis, so the output tile is [BlockSize, TileN/2] and gm_y is
// RowMajor<Axis, Post/2>. emax derived from OutT.
//
// fp4 output tile [BlockSize, TileN/2] is plain RowMajor NoneBox; the 32B column
// alignment (pto_tile.hpp:649, RECORD problem 3) requires (TileN/2)*8 % 256 == 0
// -> TileN % 64 == 0, i.e. one tile spans >=2 MX blocks along Post. The packed
// axis (Post) is orthogonal to the reduce axis (rows), so this widening does not
// touch the per-column TCOLMAX reduce. Default TileN=64 = 2 blocks.
//
// scale: E8M0 1 byte/block, planar [scaleRows, Post] with
// scaleRows = evenAlign(numKb) (reduce-axis collapsed by BlockSize + even-aligned)
// — same as dynamic_mx_quant_nontail_cublas_fp8. This IS the PTO-ISA Shared
// B-scale [G,N] contract (ADR-0101 / pto-spec d0ce06ad; matmul_shared_lowp.hpp
// consumes plain RowMajor). NO parity interleave — AscendC's [ceil(numKb/2),
// Post, 2] zip is an Ascend packing convention, not the PTO-ISA scale contract
// (RECORD 问题5 dissolved 2026-09-03). See DESIGN §5.3 / README.
// Supported BlockSize range: ALL block sizes (single-load path). The whole
// [BlockSize, TileN] block is loaded in ONE tile. The fp4 output tile is plain
// RowMajor NoneBox, so TileN carries the 32B store-granularity lower bound
// (TileN % 64 == 0, one tile spans ≥2 MX blocks along Post). The former 8KB tile
// ceiling that forced a split-reduce `_bigbs` kernel at large BlockSize is GONE:
// the toolchain header now admits any tile whose StorageBytes is a power-of-2 in
// [128B, 256KB] (pto_tile.hpp TilesizeCode), so a single [BlockSize, TileN] block
// fits directly even at BlockSize=128 ([128,64] bf16 = 16KB). The 方案A
// split-reduce kernel has been RETIRED — this plain path is the sole implementation
// (BS=128 plain verified byte-exact == old bigbs, 2026-09-08).
//
// This is the single-load implementation, kept behind an internal name. The public
// entry `dynamic_mx_quant_nontail_ocp_fp4` (below) DERIVES TileN at compile time
// from Post + the InT budget and always routes here. TileN stays an explicit param
// so the dispatcher can feed the derived value.
template <int Axis, int Post, int BlockSize = 32, int TileN = 64, typename OutT = __fp4_e2m1x2,
          typename InT = __bf16, int kPeNum = 1>
static void nontail_ocp_fp4_plain(InT *x, OutT *y, uint8_t *scale) {
    static_assert(Axis > 0 && Post > 0, "dims must be positive");
    static_assert(kPeNum == 1 || kPeNum == 4,
                  "kPeNum must be 1 (single PE) or 4 (SoftCore.h kCorePeCount)");
    static_assert(Axis % BlockSize == 0, "Axis must be multiple of BlockSize");
    static_assert(Post % 64 == 0,
                  "fp4 Post must be a multiple of 64 (align = 2 MX blocks): 自由轴 N 按物理 "
                  "TileN 列块分块 + N_tail(=Post%TileN) boxed 尾块覆盖余列; N_tail 须为 64 的倍数 "
                  "(fp4 打包 32B 列对齐)。TileN 为 BlockSize 派生常量(=128), 与 Post 无关。");
    static_assert(TileN % 64 == 0,
                  "fp4 output tile is plain RowMajor NoneBox: (TileN/2)*8 % 256 == 0 "
                  "requires TileN a multiple of 64 (>=2 MX blocks along Post)");
    // Single-load tile-size ceiling: the largest live tile is the fp32 working
    // copy [BlockSize, TileN] (32b), so BlockSize*TileN*4 must stay within the
    // 256KB TilesizeCode ceiling AND be a power-of-2 (enforced by the Tile type's
    // TilesizeCode). BlockSize*TileN <= 65536 keeps 32b tiles <= 256KB; BS=128,
    // TileN=64 (8192) has ample margin. The old 4096 cap (8KB header ceiling) is
    // retired along with the 方案A split-reduce kernel.
    static_assert(BlockSize * TileN <= 65536,
                  "non-tail OCP-FP4 single-load tile exceeds the 256KB TilesizeCode "
                  "ceiling (BlockSize*TileN*4 <= 256KB). Reduce TileN or tile the "
                  "reduce axis for BlockSize this large.");

    constexpr int numKb     = Axis / BlockSize;
    constexpr int numN_full = Post / TileN;      // 满 TileN 列块数
    constexpr int N_tail    = Post % TileN;      // 余列（boxed 尾块，物理仍 TileN）
    // reduce-axis block count, even-aligned (padding block-row left zero).
    constexpr int scaleRows = ((numKb + 1) / 2) * 2;

    using namespace pto;

    using gm_x  = global_tensor<InT,      RowMajor<Axis, Post>>;
    using gm_y  = global_tensor<uint8_t,  RowMajor<Axis, Post / 2>>;
    // scale: E8M0, compact planar [scaleRows, Post], one byte per block.
    using gm_s  = global_tensor<__fp8_e8m0, RowMajor<scaleRows, Post>>;

    // SPMD 4-PE：按**块行 kb**（归约块索引，= 输出行块 / scale 行）连续切分给 kPeNum 个
    //   PE，每个 PE 只算 [kb_begin, kb_end) 段的块行——写不重叠的 y 行块 + scale 行，无 barrier。
    //   kPeNum=1（默认）→ tid=0 跑全部块行；kPeNum=4 须 4 线程跑。与 nontail_cublas_fp8 同范式。
    const uint32_t tid = get_thread_idx();          // 0..kPeNum-1
    if (static_cast<int>(tid) >= kPeNum) return;    // 多余线程空转（kPeNum=1 只保留 tid 0）
    const int kb_base  = numKb / kPeNum;
    const int kb_rem   = numKb % kPeNum;
    const int SubKb    = kb_base + (static_cast<int>(tid) < kb_rem ? 1 : 0);
    const int kb_begin = (static_cast<int>(tid) < kb_rem)
                             ? static_cast<int>(tid) * (kb_base + 1)
                             : kb_rem * (kb_base + 1) + (static_cast<int>(tid) - kb_rem) * kb_base;
    const int kb_end   = kb_begin + SubKb;

    // 单个列块的完整计算（scale + data）。**物理 Cols 恒 = TileN**（BlockSize 派生常量，使
    //   最窄 tile [1,TileN] u8/e8m0 = TileN 字节 >=128B 最小 TSize，DerivedRows 不被撑高 →
    //   满足 pto-spec PTO-TILE-TCVT physical-Row 契约）；**ValidCols** = 该列块有效列数（满块
    //   = TileN；尾块 = N_tail，boxed，只触碰前 N_tail 列）。列块索引 n 用物理 TileN 定位，
    //   对称于 tail 的物理 TileM 恒定 + boxed ValidRow。
    auto process_tile = [&]<int ValidCols>(int kb, int n) {
        // fp4 输出 tile 为 ELEMENT-列形（physical Cols=TileN），TCVT 源/目标 physical/valid
        //   Cols 一致、走 fp4 打包 specialization；存储侧 byte 域（gm_y = Post/2 字节 + 字节
        //   基址折叠 y_iter）。ValidCols=N_tail 时只打包/落盘前 N_tail/2 字节。
        using tile_x  = Tile<Location::Vec, InT,    BlockSize, TileN, BLayout::RowMajor, BlockSize, ValidCols>;
        using tile_f  = Tile<Location::Vec, float,  BlockSize, TileN, BLayout::RowMajor, BlockSize, ValidCols>;
        using tile_o  = Tile<Location::Vec, OutT,   BlockSize, TileN, BLayout::RowMajor, BlockSize, ValidCols>;
        // colReduce（TCOLMAX）输出行向量：physical [1, TileN]、ValidCol=ValidCols（physical row=1，
        //   模型 Block.cpp:2349 反推得 row=1）。下游 scale/recip 同 physical Cols=TileN。
        using tile_maxh      = Tile<Location::Vec, __half,   1, TileN, BLayout::RowMajor, 1, ValidCols>;
        using tile_maxf      = Tile<Location::Vec, float,    1, TileN, BLayout::RowMajor, 1, ValidCols>;
        using tile_se8m0     = Tile<Location::Vec, __fp8_e8m0, 1, TileN, BLayout::RowMajor, 1, ValidCols>;
        using tile_recip_bf1 = Tile<Location::Vec, __bf16,   1, TileN, BLayout::RowMajor, 1, ValidCols>;
        using tile_recip_f1  = Tile<Location::Vec, float,    1, TileN, BLayout::RowMajor, 1, ValidCols>;

        global_iterator<gm_x, tile_x> x_iter(x);
        auto gx  = x_iter(kb, n);
        // 字节基址折叠定位：块行 kb 偏 kb*BlockSize 行 ×(Post/2) 字节行距，列块 n 偏 n*(TileN/2) 字节。
        global_iterator<gm_y, tile_o> y_iter(
            reinterpret_cast<uint8_t *>(y) + kb * BlockSize * (Post / 2) + n * (TileN / 2));
        auto gy  = y_iter(0, 0);
        // Compact scale: 把块行 kb 折进基址（迭代器 i-stride = 物理 tile 高，非 1）。
        global_iterator<gm_s, tile_se8m0> s_iter(reinterpret_cast<__fp8_e8m0 *>(scale) + kb * Post);
        auto gs = s_iter(0, n);

        // --- 值域归约（InT 分派，TABS 白名单 FP16/FP32 → bf16 先转 fp32）---
        tile_x xin;
        TLOAD(xin, gx);
        tile_recip_bf1 max_bf;
        if constexpr (std::is_same_v<InT, __half>) {
            tile_x abs_h; TABS(abs_h, xin);
            tile_maxh max_h; TCOLMAX(max_h, abs_h);   // reduce 行 -> valid row=1
            TCVT(max_bf, max_h);                        // half -> bf16
        } else if constexpr (std::is_same_v<InT, float>) {
            tile_x abs_f; TABS(abs_f, xin);
            tile_maxf max_f; TCOLMAX(max_f, abs_f);
            TCVT(max_bf, max_f);                        // fp32 -> bf16
        } else { // bf16
            tile_f xf32; TCVT(xf32, xin);               // bf16 -> fp32（规避 TABS 拒 bf16）
            tile_f abs_f; TABS(abs_f, xf32);
            tile_maxf max_f; TCOLMAX(max_f, abs_f);
            TCVT(max_bf, max_f);                        // fp32 -> bf16
        }
        // --- 清尾数留 2^E_max，乘 2^-emax，直转 e8m0 ---
        auto max_u16 = reinterpret_tile<uint16_t>(max_bf);
        TANDS(max_u16, max_u16, BF16_EXP_MASK);
        tile_recip_bf1 shared_bf;
        TMULS(shared_bf, max_bf, __builtin_bit_cast(__bf16, recip_emax_bits<OutT>()));
        tile_se8m0 scale_e8m0;
        TCVT(scale_e8m0, shared_bf);                    // bf16 -> e8m0（inf/nan->0xff 硬件）

        // --- finalize_recip_u16 内联（问题8）：同载体 uint16 视图（抄 tail_ocp_fp4）---
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

        // scale_e8m0：E8M0 字节由 Cast<bf16->e8m0> 直出，1 字节/块（无窄化 TCVT）。存 PLAIN
        //   planar [scaleRows, Post] = PTO-ISA Shared B-scale [G,N]（ADR-0101，无交织，问题5）。
        TSTORE(gs, scale_e8m0);

        tile_recip_f1 inv_scale_f;
        TCVT(inv_scale_f, recip_bf);                    // 问题4 消除：recip_bf 直接转 fp32

        tile_x xq;
        TLOAD(xq, gx);
        tile_o oq;
        if constexpr (std::is_same_v<InT, float>) {
            TCOLEXPANDMUL(xq, xq, inv_scale_f); // fp32 domain mul (no pre-cast)
            TCVT(oq, xq); // fp32 -> packed fp4_e2m1x2 (Post halved)
        } else {
            tile_f xf;
            TCVT(xf, xq); // bf16/half -> fp32
            TCOLEXPANDMUL(xf, xf, inv_scale_f); // per-column scalar broadcast-mul
            TCVT(oq, xf); // fp32 -> packed fp4_e2m1x2 (Post halved)
        }
        TSTORE(gy, oq);
    };

    for (int kb = kb_begin; kb < kb_end; ++kb) {
        for (int n = 0; n < numN_full; ++n)
            process_tile.template operator()<TileN>(kb, n);      // 满列块
        if constexpr (N_tail > 0)
            process_tile.template operator()<N_tail>(kb, numN_full); // boxed 尾列块
    }
}

// Public entry: TileN is NOT a caller knob. It is DERIVED at compile time from
// Post + the InT budget (pick_tilen) and always routes to the single-load plain
// path (方案A split-reduce retired — the 256KB TilesizeCode ceiling admits a single
// [BlockSize, TileN] block at any BlockSize). pick_tilen returns the "preferred"
// TileN under the legacy 8192B soft budget; when that yields no legal aligned TileN
// (large BlockSize, the old bigbs trigger), fall back to the minimal legal aligned
// tile TileN=align — BlockSize*align as a single block sits within 256KB. Small
// BlockSize keeps its preferred TileN unchanged → existing cases see zero change.
// InT drives BOTH the budget AND the compute domain: scale-reduce and data paths
// are InT-dispatched (bf16/half/fp32) via `if constexpr`.
template <int Axis, int Post, int BlockSize = 32, typename OutT = __fp4_e2m1x2,
          typename InT = __bf16, int kPeNum = 1>
void dynamic_mx_quant_nontail_ocp_fp4(InT *x, OutT *y, uint8_t *scale) {
    static_assert(std::is_same_v<InT, __bf16> || std::is_same_v<InT, __half> ||
                      std::is_same_v<InT, float>,
                  "InT must be one of {__bf16, __half, float}");
    constexpr int align  = nontail_align_lower<OutT>();
    constexpr int TileN0 = pick_tilen<BlockSize, Post, OutT, InT, /*IsCublas=*/false>();
    constexpr int TileN  = (TileN0 >= align) ? TileN0 : align;
    nontail_ocp_fp4_plain<Axis, Post, BlockSize, TileN, OutT, InT, kPeNum>(x, y, scale);
}

} // namespace supernpu::tile_isa::mxquant

#endif
