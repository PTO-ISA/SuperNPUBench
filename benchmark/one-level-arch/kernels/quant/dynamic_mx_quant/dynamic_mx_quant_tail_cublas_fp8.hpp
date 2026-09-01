#ifndef SUPERNPU_DYNAMIC_MX_QUANT_TAIL_CUBLAS_FP8_HPP
#define SUPERNPU_DYNAMIC_MX_QUANT_TAIL_CUBLAS_FP8_HPP

#include "quant/dynamic_mx_quant/dynamic_mx_quant_common.hpp"

namespace supernpu::tile_isa::mxquant {

// Tail-axis, cuBLAS scale (scaleAlg=1), FP8 output (E4M3 default, E5M2 valid).
// cuBLAS consumes the fp32 VALUE view (amax via TABS+TROWMAX, guarded exponent
// extract). Two-pass structure keeps peak live tiles low.
//
// SPMD 4-PE (kPeNum) —— 与 tail_ocp_fp8 同构的两级切分骨架：
//   L1 (行切分, 定 SubM)：把 M 行按连续段均分给 kPeNum 个 PE，每个 PE 只算自己那
//     [row_begin, row_begin+SubM) 段；行余数 (M%kPeNum) 前几个 PE 各多 1 行 (起点仍连续)。
//   L2 (段内 tiling, 定 TileM)：TileM = max_tilem<M,BlockSize,InT,cublas>()，段内先
//     seg_full 个 full-tile 再 (若有) 一个 seg_tail 行 boxed 尾块。
//   实现：kPeNum 编译期已知 → 按 TID 编译期展开 (模板 lambda run_pe<Pe>)，令
//     row_begin/SubM/seg_tail 均为 constexpr (满足 boxed 尾块 validRow 编译期常量要求)；
//     运行期 switch(get_thread_idx()) 分派。kPeNum=1 (默认) 时只 PE0 跑全部 M 行，行为
//     与旧的 full_m+M_tail 单线程结构完全等价 (现有单线程 driver / 字节校验零回归)。
//   kPeNum=4 时必须用 4 线程跑：gfrun -s softcore.multiThreadNum=4；单线程只写 1/4 输出。
//
// BlockSize range: UNBOUNDED. Unlike the non-tail kernels, the tail contiguous
// axis is BlockSize (the quant axis, always a multiple of 32 -> naturally 32B
// aligned), and the free axis is M rows tiled by the derived TileM. There is no
// alignment-vs-TileSize conflict on a single axis: the binding tile budget
// (TileM*BlockSize) is met by shrinking TileM, so ANY BlockSize is legal.
//
// TileM is NOT a caller knob: it is DERIVED at compile time from M + the InT
// binding-tile budget (max_tilem<M, BlockSize, InT, /*IsCublas=*/true>()), clamped
// to [tilem_min(>=512B tile), budget/BlockSize] and to M. InT drives BOTH the budget
// (a wider input dtype shrinks TileM) AND the compute domain: scale-reduce and data
// paths are InT-dispatched (bf16/half/fp32) via `if constexpr` (static_assert below).
template <int M, int K, int BlockSize = 32, typename OutT = __fp8_e4m3,
          typename InT = __bf16, uint32_t MaxLowBoundBits = 0x2b8cbcccu,
          int kPeNum = 1>
void dynamic_mx_quant_tail_cublas_fp8(InT *x, OutT *y, uint8_t *scale) {
    static_assert(M > 0 && K > 0, "dim must be positive");
    static_assert(K % BlockSize == 0, "K must be multiple of BlockSize");
    static_assert(std::is_same_v<InT, __bf16> || std::is_same_v<InT, __half> ||
                      std::is_same_v<InT, float>,
                  "InT must be one of {__bf16, __half, float}");
    static_assert(kPeNum == 1 || kPeNum == 4,
                  "kPeNum must be 1 (single PE) or 4 (SoftCore.h kCorePeCount)");

    constexpr int numKb  = K / BlockSize;
    // AscendC scale layout: uint8 E8M0, one byte per block, compact [M, scaleCols]
    // with the block count even-aligned (scaleColNum_ = CeilDiv(numKb,2)*2). The
    // trailing padding column is left zero. Mirrors dynamic_mx_quant_tail_axis_fp8.h:168.
    constexpr int scaleCols = ((numKb + 1) / 2) * 2;

    using namespace pto;

    uint8_t *y_u8 = reinterpret_cast<uint8_t *>(y);
    const uint32_t tid = get_thread_idx();          // 0..kPeNum-1

    using gm_x = global_tensor<InT,     RowMajor<M, K>>;
    using gm_y = global_tensor<uint8_t, RowMajor<M, K>>;
    using gm_s = global_tensor<uint8_t, RowMajor<M, scaleCols>>;

    // 单个 tile-行块的完整计算 (scale pass + data pass)。TileMv = 该 tile 物理行高
    //   (per-PE 编译期常量, 由 max_tilem 推得)；ValidRows = 该 tile 活跃行数 (full-tile:
    //   TileMv；尾块: seg_tail<TileMv, boxed)；row0 = 该 tile 全局起始行。物理 tile 恒
    //   TileMv×BlockSize (logicalTileBytes>=512B, 避免 sub-512B spill)，boxed ValidRows
    //   只触碰活跃行。base 指针按 row0 偏移 (对照 tail_ocp_fp8::process_tile)。
    //   列向量中间 tile (reduce 输出及其下游) 声明 physical Cols=1 —— 匹配 codex 模型
    //   rowReduce 无条件 col=1 (对照 tail_ocp_fp8::process_tile 注释 / Block.cpp)，令
    //   reduce 输出与下游 binary/compare TEPL 两侧 physical Cols 全等，绕过
    //   IsCompatibleOperationDataTile 的 source->col==physicalCol 契约 (否则 physical
    //   Cols=BlockSize 会撞 ValidateBasicBinaryTepl:748)。全宽 tile (tile_x/f/o) 仍 Cols=BlockSize。
    auto process_tile = [&]<int TileMv, int ValidRows>(int row0) {
        using tile_x        = Tile<Location::Vec, InT,      TileMv, BlockSize, BLayout::RowMajor, ValidRows, BlockSize>;
        using tile_f        = Tile<Location::Vec, float,    TileMv, BlockSize, BLayout::RowMajor, ValidRows, BlockSize>;
        using tile_o        = Tile<Location::Vec, OutT,     TileMv, BlockSize, BLayout::RowMajor, ValidRows, BlockSize>;
        // Compact scale store: the cuBLAS core emits scale_byte/recip already boxed
        // valid col=1 (one per-row scalar per block), narrow straight to uint8, store
        // one byte per block. Physical Cols=1 (see note above).
        using tile_sred     = Tile<Location::Vec, uint16_t, TileMv, 1, BLayout::RowMajor, ValidRows, 1>;
        using tile_sstore   = Tile<Location::Vec, uint8_t,  TileMv, 1, BLayout::RowMajor, ValidRows, 1>;
        using tile_recip_f1 = Tile<Location::Vec, float,    TileMv, 1, BLayout::RowMajor, ValidRows, 1>;
        using tile_in1      = Tile<Location::Vec, InT,      TileMv, 1, BLayout::RowMajor, ValidRows, 1>;
        using tile_u32_1    = Tile<Location::Vec, uint32_t, TileMv, 1, BLayout::RowMajor, ValidRows, 1>;

        for (int kb = 0; kb < numKb; ++kb) {
            global_iterator<gm_x, tile_x> x_iter(x + row0 * K + kb * BlockSize);
            auto gx = x_iter(0, 0);
            global_iterator<gm_y, tile_o> y_iter(y_u8 + row0 * K + kb * BlockSize);
            auto gy = y_iter(0, 0);
            // Compact scale: base pointer folds row0 (row stride scaleCols) and the
            // block index kb (per-row byte column); index (0,0) writes ValidRows rows.
            global_iterator<gm_s, tile_sstore> s_iter(scale + row0 * scaleCols + kb);
            auto gs = s_iter(0, 0);

            // ComputeScale pass: reduce the amax in InT domain, cast only the reduced
            // per-row scalar to fp32 (mirrors AscendC; no whole-tile CVT).
            tile_sred scale_byte;
            tile_sred recip;
            tile_x xq_s;
            TLOAD(xq_s, gx);
            // ================================================================
            // 内联展开：等价于 common::compute_cublas_scale_tail<OutT,InT,TileMv,
            // BlockSize,MaxLowBoundBits,ValidRows> + common::compute_cublas_core（IDEAL
            // CmpMode 版）。就地展开以规避 RECORD 问题8（tile 作真实函数入参 → S64 栈往返
            // → gfrun 拒）。两处规避已换正式方案：
            //   · reinterpret_f32_to_u32（scratch-HBM，问题4）→ reinterpret_tile<>（零指令视图）
            //   · GT/LT/NE 的 min/max+默认-EQ 模拟（问题3）→ 带 CmpMode 的原生 TCMPS
            // scale 无需交织（尾轴块行行内已连续，compact 平铺即等价，问题5）。
            // -- compute_cublas_scale_tail：InT 域 TABS+TROWMAX，仅把归约量转 fp32 --
            tile_x abs_x;
            TABS(abs_x, xq_s);
            tile_recip_f1 max_f;
            if constexpr (std::is_same_v<InT, float>) {
                TROWMAX(max_f, abs_x);      // fp32：直接归约到 fp32（免前置 cast）
            } else {
                tile_in1 max_r;
                TROWMAX(max_r, abs_x);      // reduce cols -> valid col=1（InT 域）
                TCVT(max_f, max_r);         // bf16/half -> fp32（仅归约后的 per-row 标量）
            }
            // -- compute_cublas_core（IDEAL CmpMode 版，对照 AscendC ComputeScaleCublas）--
            // finite/nonzero 掩码须在原地 clamp 前从 raw 视图算完（视图与 max_f 同寄存器）。
            auto raw = reinterpret_tile<uint32_t>(max_f);        // 问题4 正式方案：零指令
            tile_u32_1 finite;
            TCMPS<CmpMode::LT>(finite, raw, FP32_EXP_MASK);      // raw < 0x7f800000
            tile_u32_1 nonzero;
            TCMPS<CmpMode::NE>(nonzero, raw, static_cast<uint32_t>(0));
            TMAXS(max_f, max_f, __builtin_bit_cast(float, MaxLowBoundBits)); // 原地 clamp
            TMULS(max_f, max_f, inv_dst_max<OutT>());
            // clamp 后再开视图（零指令），再用 u32->u32 恒等 TCVT 把位型物化到真实 uint32
            // tile：后续 TSHRS/TANDS/TAND/TOR/TSEL 都是单模板参（dst/src 必须同类型），视图
            // 类型 ≠ 真实 tile，故须先物化一次；相比 scratch-HBM 往返，这里只一条寄存器级 TCVT。
            auto s32v = reinterpret_tile<uint32_t>(max_f);
            tile_u32_1 s32;
            TCVT(s32, s32v);
            tile_u32_1 exp32;
            TSHRS(exp32, s32, FP32_SHR_NUM);
            tile_u32_1 man32;
            TANDS(man32, s32, FP32_MANTISSA_MASK);
            // extractExp = ((exp>0 && exp<254 && man>0) || (exp==0 && man>0x400000))
            //                ? exp+1 : exp
            // PTO ISA 合规写法（PTO-REQ-TEPL-COMPARISON-001，pto-spec）：TCMP/TCMPS 产出**packed
            // predicate Tile**（位打包，元素 i 占 byte i/8 的 bit i%8），TSEL 的 mask 源**必须**是
            // packed predicate，而 TAND 规范上**只作用于 integer 数据、显式 reject packed 格式**——故
            // 不能用数据域 TAND/TOR 组合 compare 掩码（那是旧仿真器纵容的不合规写法）。ISA 也无 predicate
            // 组合指令,复合条件（&& / ||）只能经**嵌套 TSEL**逐个 predicate 串联,每个 TSEL 只吃单个直接
            // compare predicate。逐值等价旧 TAND/TOR+单 TSEL（这些 tile 全是掩码、不承载数值）。
            // 详见 README「cuBLAS 守卫掩码的 PTO ISA 合规写法」小节。
            tile_u32_1 exp_p1;
            TADDS(exp_p1, exp32, static_cast<uint32_t>(1));
            tile_u32_1 sel;
            TADDS(sel, exp32, static_cast<uint32_t>(0));   // 默认 extractExp = exp
            // p0 = (exp>0)&&(exp<254)&&(man>0) ? exp+1 : sel —— 嵌套 c3→c2→c1 gate
            tile_u32_1 c1; TCMPS<CmpMode::GT>(c1, exp32, static_cast<uint32_t>(0));
            tile_u32_1 c2; TCMPS<CmpMode::LT>(c2, exp32, FP32_NUMBER_254);
            tile_u32_1 c3; TCMPS<CmpMode::GT>(c3, man32, static_cast<uint32_t>(0));
            tile_u32_1 n3; TADDS(n3, sel, static_cast<uint32_t>(0)); TSEL(n3, c3, exp_p1); // c3? e+1 : e
            tile_u32_1 n2; TADDS(n2, sel, static_cast<uint32_t>(0)); TSEL(n2, c2, n3);     // c2? n3 : e
            TSEL(sel, c1, n2);                                                             // c1? n2 : e = p0?e+1:e
            // p1 = (exp==0)&&(man>0x400000) ? exp+1 : sel —— 从含 p0 的 sel 起做 OR
            tile_u32_1 c4; TCMPS<CmpMode::EQ>(c4, exp32, static_cast<uint32_t>(0));
            tile_u32_1 c5; TCMPS<CmpMode::GT>(c5, man32, FP32_NUMBER_HALF);
            tile_u32_1 u5; TADDS(u5, sel, static_cast<uint32_t>(0)); TSEL(u5, c5, exp_p1); // c5? e+1 : sel
            TSEL(sel, c4, u5);                                                             // c4? u5 : sel = p1?e+1:sel
            // finite? .. : 0xff ; nonzero? .. : 0
            tile_u32_1 nanb;
            TEXPANDS(nanb, FP32_FP8_NAN);
            TSEL(nanb, finite, sel);        // finite? sel : 0xff
            tile_u32_1 extract;
            TEXPANDS(extract, static_cast<uint32_t>(0));
            TSEL(extract, nonzero, nanb);   // nonzero? .. : 0
            TCVT(scale_byte, extract);      // narrow low16
            // recip = 0x7f00 - (extractExp<<7) ; finite? .. : 0x7f81 ; nonzero? .. : 0
            tile_u32_1 sh;
            TSHLS(sh, extract, static_cast<uint32_t>(BF16_SHR_NUM));
            tile_u32_1 bias;
            TEXPANDS(bias, FP32_EXP_BIAS_CUBLAS);
            tile_u32_1 half;
            TSUB(half, bias, sh);
            tile_u32_1 rnan;
            TEXPANDS(rnan, FP32_NAN_PACK);
            TSEL(rnan, finite, half);       // finite? half : 0x7f81
            tile_u32_1 rsel;
            TEXPANDS(rsel, static_cast<uint32_t>(0));
            TSEL(rsel, nonzero, rnan);      // nonzero? .. : 0
            TCVT(recip, rsel);
            // ================================================================
            // scale_byte already boxed valid col=1; narrow to uint8, store 1 byte/block.
            tile_sstore scale_u8;
            TCVT(scale_u8, scale_byte);
            TSTORE(gs, scale_u8); // store scale early; scale_byte now dead

            // 问题4 正式方案：reinterpret_tile 零指令把 recip(uint16) 视为 bf16，替代
            // scratch-HBM 的 reinterpret_u16_to_bf16。recip 为具名 uint16 lvalue，满足视图约束。
            auto inv_bf16 = reinterpret_tile<__bf16>(recip);
            tile_recip_f1 inv_scale_f;
            TCVT(inv_scale_f, inv_bf16);

            // ComputeData pass: reload the value view now.
            tile_x xq;
            TLOAD(xq, gx);
            tile_o oq;
            if constexpr (std::is_same_v<InT, float>) {
                // fp32 input already in the compute domain: mul in fp32 directly
                // (mirrors AscendC ComputeData fp32 branch, no pre-cast).
                TROWEXPANDMUL(xq, xq, inv_scale_f);
                TCVT(oq, xq);
            } else {
                tile_f xf;
                TCVT(xf, xq); // bf16/half -> fp32
                TROWEXPANDMUL(xf, xf, inv_scale_f); // per-row scalar broadcast-mul
                TCVT(oq, xf);
            }
            TSTORE(gy, oq);
        }
    };

    // 单个 PE (编译期常量 Pe) 的驱动：算自己那段连续行 [row_begin, row_begin+SubM)，
    // 段内先 seg_full 个 TileM full-tile，再 (若有) 一个 seg_tail 行的 boxed 尾块。
    //   row_base = M / kPeNum;  row_rem = M % kPeNum
    //   Pe < row_rem -> SubM = row_base+1, row_begin = Pe*(row_base+1)  (前几个 PE 各多 1 行)
    //   Pe >= row_rem-> SubM = row_base,   row_begin = row_rem*(row_base+1)+(Pe-row_rem)*row_base
    // row_begin/SubM/seg_tail 全为 constexpr → 满足 boxed 尾块 validRow 编译期常量要求。
    auto run_pe = [&]<int Pe>() {
        constexpr int row_base  = M / kPeNum;
        constexpr int row_rem   = M % kPeNum;
        constexpr int SubM      = row_base + (Pe < row_rem ? 1 : 0);
        constexpr int row_begin = (Pe < row_rem)
                                      ? Pe * (row_base + 1)
                                      : row_rem * (row_base + 1) + (Pe - row_rem) * row_base;
        if constexpr (SubM > 0) {
            constexpr int TileM    = max_tilem<M, BlockSize, InT, /*IsCublas=*/true>();
            constexpr int seg_full = SubM / TileM;   // SubM>TileM 时循环多个 full-tile
            constexpr int seg_tail = SubM % TileM;   // 余行 (< TileM), boxed 尾块
            for (int lm = 0; lm < seg_full; ++lm) {
                process_tile.template operator()<TileM, TileM>(row_begin + lm * TileM);
            }
            if constexpr (seg_tail > 0) {
                process_tile.template operator()<TileM, seg_tail>(row_begin + seg_full * TileM);
            }
        }
    };

    // 运行期按 tid 分派到编译期展开的 per-PE 实例 (kPeNum 编译期已知)。kPeNum=1 时只 case0。
    switch (static_cast<int>(tid)) {
        case 0: run_pe.template operator()<0>(); break;
        case 1: if constexpr (kPeNum > 1) run_pe.template operator()<1>(); break;
        case 2: if constexpr (kPeNum > 2) run_pe.template operator()<2>(); break;
        case 3: if constexpr (kPeNum > 3) run_pe.template operator()<3>(); break;
        default: break;
    }
}

} // namespace supernpu::tile_isa::mxquant

#endif
