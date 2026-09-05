#ifndef SUPERNPU_DYNAMIC_MX_QUANT_NONTAIL_OCP_FP4_HPP
#define SUPERNPU_DYNAMIC_MX_QUANT_NONTAIL_OCP_FP4_HPP

#include "quant/dynamic_mx_quant/dynamic_mx_quant_common.hpp"
#include "quant/dynamic_mx_quant/dynamic_mx_quant_nontail_ocp_fp4_bigbs.hpp"

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
// Supported BlockSize range (plain single-load path): BlockSize ∈ {32, 64}.
// The whole [BlockSize, TileN] block is loaded in ONE tile, so the contiguous
// axis TileN carries BOTH the fp4 32B alignment LOWER bound (TileN % 64 == 0,
// i.e. TileN ≥ 64) and the TileSize UPPER bound (16b input tile:
// BlockSize*TileN*2 ≤ 8192 → TileN ≤ 4096/BlockSize). A legal TileN exists iff
// 64 ≤ 4096/BlockSize → BlockSize ≤ 64. BlockSize ≥ 96 (next multiple of 32:
// 96*64=6144 > 4096) has NO legal TileN here → use
// dynamic_mx_quant_nontail_ocp_fp4_bigbs (方案A, splits the reduce axis).
//
// This is the PLAIN single-load implementation, kept behind an internal name.
// The public entry `dynamic_mx_quant_nontail_ocp_fp4` (below) DERIVES TileN at
// compile time from Post + the InT budget and AUTO-ROUTES to this plain path when
// a legal TileN exists, or to the 方案A split-reduce `_bigbs` kernel when it does
// not (large BlockSize). TileN stays an explicit param here so the dispatcher can
// feed the derived value.
template <int Axis, int Post, int BlockSize = 32, int TileN = 64, typename OutT = __fp4_e2m1x2,
          typename InT = __bf16, int kPeNum = 1>
static void nontail_ocp_fp4_plain(InT *x, OutT *y, uint8_t *scale) {
    static_assert(Axis > 0 && Post > 0, "dims must be positive");
    static_assert(kPeNum == 1 || kPeNum == 4,
                  "kPeNum must be 1 (single PE) or 4 (SoftCore.h kCorePeCount)");
    static_assert(Axis % BlockSize == 0, "Axis must be multiple of BlockSize");
    static_assert(Post % TileN == 0, "Post must be multiple of TileN");
    static_assert(TileN % 64 == 0,
                  "fp4 output tile is plain RowMajor NoneBox: (TileN/2)*8 % 256 == 0 "
                  "requires TileN a multiple of 64 (>=2 MX blocks along Post)");
    // BlockSize range: single-load path is capped at BlockSize ≤ 64. With
    // TileN ≥ 64, the 16b input tile budget BlockSize*TileN ≤ 4096 forces
    // BlockSize ≤ 64. BlockSize ≥ 96 -> no legal TileN; use the _bigbs kernel.
    // The whole [BlockSize, TileN] input block is loaded in ONE tile (value-domain
    // reduce), so the 16b input tile is the binding budget; large BS still needs
    // _bigbs (split reduce axis).
    static_assert(BlockSize * TileN <= 4096,
                  "plain non-tail OCP-FP4 supports BlockSize ∈ {32,64} only (16b input "
                  "tile BlockSize*TileN <= 4096, and TileN >= 64 forces BlockSize <= 64). "
                  "For BlockSize >= 96 use dynamic_mx_quant_nontail_ocp_fp4_bigbs "
                  "(方案A, split reduce axis).");

    constexpr int numKb = Axis / BlockSize;
    constexpr int numN  = Post / TileN;
    // reduce-axis block count, even-aligned (padding block-row left zero).
    constexpr int scaleRows = ((numKb + 1) / 2) * 2;

    using namespace pto;

    using tile_x  = Tile<Location::Vec, InT,    BlockSize, TileN, BLayout::RowMajor>;
    using tile_f  = Tile<Location::Vec, float,  BlockSize, TileN, BLayout::RowMajor>;
    // fp4 输出 tile：ELEMENT-列形（physical Cols=TileN，与源 fp32 tile 同列数），故 TCVT
    //   源/目标 physical/valid Rows/Cols 完全一致、走 fp4 打包 specialization；gfrun 按
    //   BytesOf(fp4) 打包两个 4bit/字节（SuperScalarModel 31f7a8f）。存储侧仍是 Post/2 字节
    //   （gm_y byte 域 + 字节基址折叠 y_iter）。旧的 [BlockSize, TileN/2] 字节列约定会让
    //   TCVT 落到 ordinary 路径（dst Cols=TileN/2≠src Cols=TileN）而编译崩——与 tail_ocp_fp4
    //   同一 element-列迁移（0.58.3 工具链头已删 32B 列对齐 assert，无需 PW-padding）。
    using tile_o  = Tile<Location::Vec, OutT,   BlockSize, TileN, BLayout::RowMajor>;
    // colReduce（TCOLMAX）输出为行向量：physical row=1（对齐模型 Block.cpp:2349 —— colReduce
    //   按 dst->size/(validCol·elem) 反推物理行，physical row=1 时反推得 1，自洽）。下游 scale/
    //   recip 载体同取 physical row=1（TCOLEXPANDMUL 的一行广播源）。见 RECORD 问题22 补充。
    using tile_maxh      = Tile<Location::Vec, __half,   1, TileN, BLayout::RowMajor, 1, TileN>;
    using tile_maxf      = Tile<Location::Vec, float,    1, TileN, BLayout::RowMajor, 1, TileN>;
    using tile_se8m0     = Tile<Location::Vec, __fp8_e8m0, 1, TileN, BLayout::RowMajor, 1, TileN>;
    using tile_recip_bf1 = Tile<Location::Vec, __bf16,   1, TileN, BLayout::RowMajor, 1, TileN>;
    using tile_recip_f1  = Tile<Location::Vec, float,    1, TileN, BLayout::RowMajor, 1, TileN>;

    using gm_x  = global_tensor<InT,      RowMajor<Axis, Post>>;
    using gm_y  = global_tensor<uint8_t,  RowMajor<Axis, Post / 2>>;
    // scale: E8M0, compact planar [scaleRows, Post], one byte per block.
    using gm_s  = global_tensor<__fp8_e8m0, RowMajor<scaleRows, Post>>;

    global_iterator<gm_x,  tile_x>  x_iter(x);

    // SPMD 4-PE：按**块行 kb**（归约块索引，= 输出行块 / scale 行）连续切分给 kPeNum 个
    //   PE，每个 PE 只算 [kb_begin, kb_end) 段的块行——写不重叠的 y 行块 + scale 行，无 barrier。
    //   块行切分不改任何 tile 形状（不同于 tail 的 M 行 boxed 尾块），故用**运行期**值即可。
    //   kPeNum=1（默认）→ tid=0 跑全部块行，与旧单线程行为等价（现有 driver 零回归）。
    //   kPeNum=4 须 4 线程跑（gfrun -s softcore.multiThreadNum=4）；单线程只写 1/kPeNum 输出。
    //   与 nontail_cublas_fp8 同一切分范式。
    const uint32_t tid = get_thread_idx();          // 0..kPeNum-1
    if (static_cast<int>(tid) >= kPeNum) return;    // 多余线程空转（kPeNum=1 只保留 tid 0）
    const int kb_base  = numKb / kPeNum;
    const int kb_rem   = numKb % kPeNum;
    const int SubKb    = kb_base + (static_cast<int>(tid) < kb_rem ? 1 : 0);
    const int kb_begin = (static_cast<int>(tid) < kb_rem)
                             ? static_cast<int>(tid) * (kb_base + 1)
                             : kb_rem * (kb_base + 1) + (static_cast<int>(tid) - kb_rem) * kb_base;
    const int kb_end   = kb_begin + SubKb;

    for (int kb = kb_begin; kb < kb_end; ++kb) {
        for (int n = 0; n < numN; ++n) {
            auto gx  = x_iter(kb, n);
            // fp4 输出 element-列形 tile 对字节域 gm_y：字节基址折叠定位（同 tail_ocp_fp4）——
            //   块行 kb 偏 kb*BlockSize 行 × (Post/2) 字节行距，列块 n 偏 n*(TileN/2) 字节。
            global_iterator<gm_y, tile_o> y_iter(
                reinterpret_cast<uint8_t *>(y) + kb * BlockSize * (Post / 2) + n * (TileN / 2));
            auto gy  = y_iter(0, 0);
            // Compact scale: fold block-row index (kb) into the base pointer since
            // the iterator's i-stride is the PHYSICAL tile height, not 1. Each
            // block-row writes TileN bytes at scale + kb*Post.
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

            // --- finalize_recip_u16 内联（问题8）：同载体 uint16 视图（抄 tail_ocp_fp4:174-193）---
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

            // scale_e8m0 boxed valid row=1: E8M0 byte produced directly by
            // Cast<bf16->e8m0>, store 1 byte/block (no narrowing TCVT).
            // stored as PLAIN planar [scaleRows, Post] = PTO-ISA Shared B-scale
            // [G,N] (ADR-0101). NO parity interleave: AscendC's [ceil(numKb/2),
            // Post, 2] zip is an Ascend packing convention, not the PTO-ISA scale
            // contract (RECORD 问题5 dissolved 2026-09-03).
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
        }
    }
}

// Public entry: TileN is NOT a caller knob. It is DERIVED at compile time from
// Post + the InT budget (pick_tilen). If a legal TileN >= the fp4 64B lower bound
// exists, route to the plain single-load path; otherwise (large BlockSize leaves
// no legal TileN) auto-route to the 方案A split-reduce `_bigbs` kernel with a
// budget-derived R_sub. `if constexpr` guarantees the untaken branch is not
// instantiated. InT drives BOTH the budget AND the compute domain: scale-reduce and
// data paths are InT-dispatched (bf16/half/fp32) via `if constexpr`.
template <int Axis, int Post, int BlockSize = 32, typename OutT = __fp4_e2m1x2,
          typename InT = __bf16, int kPeNum = 1>
void dynamic_mx_quant_nontail_ocp_fp4(InT *x, OutT *y, uint8_t *scale) {
    static_assert(std::is_same_v<InT, __bf16> || std::is_same_v<InT, __half> ||
                      std::is_same_v<InT, float>,
                  "InT must be one of {__bf16, __half, float}");
    constexpr int TileN = pick_tilen<BlockSize, Post, OutT, InT, /*IsCublas=*/false>();
    if constexpr (TileN >= nontail_align_lower<OutT>()) {
        nontail_ocp_fp4_plain<Axis, Post, BlockSize, TileN, OutT, InT, kPeNum>(x, y, scale);
    } else {
        // 大 BlockSize 自动路由到 _bigbs（方案A 切归约轴）；bigbs 场景块行少、并行度有限,
        // 不做 4-PE（与 nontail_cublas_fp8 一致，kPeNum 不透传）。
        constexpr int BigTileN = nontail_align_lower<OutT>();
        constexpr int Rsub = max_rsub<BlockSize, BigTileN, InT, /*IsCublas=*/false>();
        dynamic_mx_quant_nontail_ocp_fp4_bigbs<Axis, Post, BlockSize, BigTileN, Rsub, OutT, InT>(x, y, scale);
    }
}

} // namespace supernpu::tile_isa::mxquant

#endif
