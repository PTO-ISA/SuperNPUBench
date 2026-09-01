#ifndef SUPERNPU_DYNAMIC_MX_QUANT_TAIL_OCP_FP4_HPP
#define SUPERNPU_DYNAMIC_MX_QUANT_TAIL_OCP_FP4_HPP

#include <common/pto_tileop.hpp>
#include <cstdint>
#include "quant/dynamic_mx_quant/dynamic_mx_quant_common.hpp"

namespace supernpu::tile_isa::mxquant {

namespace tail_ocp_fp4_detail {
// 向下取 2 的幂：物理 tile 字节数须为 2 的幂（TSize 编码约束），否则 LLVM 后端 getSimpleVT 崩。
constexpr int pow2_floor(int v) {
    if (v < 1) return 0;
    int p = 1;
    while (p * 2 <= v) p *= 2;
    return p;
}
// 最大 TileM —— 仅由 blockSize + 输入宽度决定，与 SubM 无关：
//   budgetMax = 8192/(BS*sizeof fp32)  —— data pass 的 fp32 中间量 tile <= 8KB（绑定约束）
//   floorMin  = 512/(BS*inBytes)       —— 物理 tile >= 512B（避免 LinxV5 sub-512B spill）
//   TileMmax = pow2_floor(budgetMax) 抬到 floorMin。BS=32 -> 64。
constexpr int tilem_max(int blockSize, int inBytes) {
    const int budgetMax = 8192 / (blockSize * static_cast<int>(sizeof(float)));   // 64 @ BS=32
    const int floorMin  = (512 / inBytes + blockSize - 1) / blockSize;            // 8 @ BS=32/bf16
    int t = pow2_floor(budgetMax);
    if (t > 0 && t < floorMin) t = floorMin;
    return t;
}
} // namespace tail_ocp_fp4_detail

// ===========================================================================
// TAIL-OCP-FP4 正式 kernel（固定 SPMD 4-PE）—— bf16/half/fp32 in / fp4(e2m1) out /
// e8m0 scale / BlockSize=32 / 倒数位补主路径 + inf/zero/special 三守卫（TCMPS+TSEL，
// fp4 原有语义，不随 fp8 母本省略）。
//
// 与 tail_ocp_fp8 母本逐结构对齐（两级切分 + get_thread_idx SPMD），差异仅在输出：
//   - 输出 fp4 打包（每字节 2 个 4bit 元素）→ 每行 N/2 字节（gm_y = RowMajor<M, N/2>），
//     TSTORE 落 kb*(BlockSize/2) 字节偏移；emax 由 OutT 派生（recip_emax_bits<OutT>()）。
//   - InT 分派 reduce：bf16 原生域 / half·fp32 经 fp32 域（floor 指数避 narrowing round 进位）。
//
// 关键：reduce 输出的列向量 tile（max/shared/scale/recip）声明 **physical Cols=1**，匹配
//   model d8903938 rowReduce 无条件 col=1（Block.cpp:2069）；下游标量逻辑 op（TANDS/TXORS/
//   TSUBS）的 physicalCol 才与 source tile 实际 col 一致，绕过 ValidateScalarLogicalTepl
//   （AccumulateBlockInfo.cpp:596）的 col 校验。旧 PW-padding 方案已废弃（32B 列对齐 assert
//   在 0.58.3 工具链头已删，PW 无必要，且 PW≠1 恰是运行崩根因）。
//
// SPMD 语义：runtime 把 [0,multiThreadNum) 所有线程 reset 到同一 entry PC，靠 kernel 内
//   get_thread_idx()（0..3）自我按 M 行切分，写不重叠的 M 行，无 barrier。**必须 4 线程跑**
//   （gfrun -s softcore.multiThreadNum=4）；单线程跑只会写 1/4 输出。
//
// 两级切分：L1 行切分定 SubM（前 row_rem 个 PE 各多 1 行，连续块）；L2 段内 tiling 定 TileM
//   （blocksize 决定的固定上限 tilem_max，BS=32→64，2 的幂），SubM 只决定循环次数：
//   seg_full = SubM/TileM 个 full-tile + seg_tail = SubM%TileM 余行 boxed 尾块。
// ===========================================================================
template <int M, int N, int BlockSize = 32, typename OutT = __fp4_e2m1x2,
          typename InT = __bf16>
void dynamic_mx_quant_tail_ocp_fp4(InT *x, OutT *y, uint8_t *scale) {
    static_assert(M > 0 && N > 0, "dim must be positive");
    static_assert(N % BlockSize == 0, "N must be multiple of BlockSize");
    static_assert(BlockSize % 32 == 0,
                  "fp4 output block is BlockSize/2 packed bytes; BlockSize must be "
                  "a multiple of 32 so the packed row is 32B-column-aligned");
    static_assert(std::is_same_v<InT, __bf16> || std::is_same_v<InT, __half> ||
                      std::is_same_v<InT, float>,
                  "InT must be one of {__bf16, __half, float}");

    using namespace pto;

    // emax 由 OutT 派生（recip_emax_bits<__fp4_e2m1x2>() = BF16_ONE - FP4_E2M1_EMAX）。
    constexpr uint16_t RECIP_EMAX = recip_emax_bits<OutT>();

    constexpr int numKb     = N / BlockSize;
    // AscendC scale 布局：uint8 E8M0，每 block 一字节，block 数偶对齐。
    constexpr int scaleCols = ((numKb + 1) / 2) * 2;
    constexpr bool oddTail  = (numKb % 2) != 0;   // 奇尾 padding scale 列须写 0x00
    constexpr int kPeNum    = 4;  // SoftCore.h kCorePeCount，multiThreadNum 仅 1|4 合法
    const uint32_t tid = get_thread_idx();          // 0..3

    uint8_t *y_u8 = reinterpret_cast<uint8_t *>(y);

    using gm_x = global_tensor<InT,        RowMajor<M, N>>;
    using gm_y = global_tensor<uint8_t,    RowMajor<M, N / 2>>;
    using gm_s = global_tensor<__fp8_e8m0, RowMajor<M, scaleCols>>;

    // 单个 tile-行块的完整计算（scale pass + data pass）。ValidRows = 活跃行数
    //   （full-tile: TileMv；尾块: seg_tail<TileMv，boxed）。row0 = 全局起始行。
    auto process_tile = [&]<int TileMv, int ValidRows>(int row0) {
        // 全宽 tile：physical Cols=BlockSize。列向量 reduce tile：physical Cols=1
        //   （匹配 model rowReduce 无条件 col=1，见头部注释）。
        using t_x   = Tile<Location::Vec, InT,        TileMv, BlockSize, BLayout::RowMajor, ValidRows, BlockSize>;
        using t_hb  = Tile<Location::Vec, __half,     TileMv, 1,         BLayout::RowMajor, ValidRows, 1>;
        using t_bfb = Tile<Location::Vec, __bf16,     TileMv, 1,         BLayout::RowMajor, ValidRows, 1>;
        using t_e8b = Tile<Location::Vec, __fp8_e8m0, TileMv, 1,         BLayout::RowMajor, ValidRows, 1>;
        using t_fb  = Tile<Location::Vec, float,      TileMv, 1,         BLayout::RowMajor, ValidRows, 1>;
        using t_f   = Tile<Location::Vec, float,      TileMv, BlockSize, BLayout::RowMajor, ValidRows, BlockSize>;
        // fp4 输出 tile：ELEMENT-列形（physical BlockSize，valid BlockSize），gfrun 按
        //   BytesOf(fp4) 打包两个 4bit/字节（SuperScalarModel 31f7a8f）。
        using t_o   = Tile<Location::Vec, OutT,       TileMv, BlockSize, BLayout::RowMajor, ValidRows, BlockSize>;

        for (int kb = 0; kb < numKb; ++kb) {
            // === scale pass：value-domain reduce（InT 分派），floor 指数 ===
            // ⚠ 指数抽取 round-mode 缺口：块 |max| 恰落 2^k 正下方时，narrowing TCVT(_->bf16)
            //   默认 round-to-nearest 会进位越 2^k → 指数抬高一档、量化 2× 偏低，与 golden 截断
            //   语义失配。故 half/fp32 先在更宽域 mask(floor 到 2^E，无进位)再窄化；bf16 原生
            //   无 narrowing，直接取指数天然与截断一致。
            global_iterator<gm_x, t_x> x_iter(x + row0 * N + kb * BlockSize);
            auto gx = x_iter(0, 0);
            t_x xin; TLOAD(xin, gx);

            t_bfb max_bf;
            if constexpr (std::is_same_v<InT, __half>) {
                t_x  abs_h;  TABS(abs_h, xin);
                t_hb max_h;  TROWMAX(max_h, abs_h);            // half 域归约
                t_fb max_f;  TCVT(max_f, max_h);               // half -> fp32（精确加宽）
                auto max_u32 = reinterpret_tile<uint32_t>(max_f);
                TANDS(max_u32, max_u32, FP32_EXP_MASK);        // fp32 域 floor 到 2^E（无进位）
                TCVT(max_bf, max_f);                           // fp32 -> bf16（尾数=0，精确）
            } else if constexpr (std::is_same_v<InT, float>) {
                t_x  abs_f;  TABS(abs_f, xin);
                t_fb max_f;  TROWMAX(max_f, abs_f);            // fp32 域归约
                auto max_u32 = reinterpret_tile<uint32_t>(max_f);
                TANDS(max_u32, max_u32, FP32_EXP_MASK);        // fp32 域 floor（无进位）
                TCVT(max_bf, max_f);                           // fp32 -> bf16（尾数=0，精确）
            } else {
                t_x  abs_bf; TABS(abs_bf, xin);                // bf16 原生
                TROWMAX(max_bf, abs_bf);                       // bf16 域归约 -> max_bf
                auto max_u16 = reinterpret_tile<uint16_t>(max_bf);
                TANDS(max_u16, max_u16, BF16_EXP_MASK);        // 直接取指数（无转换->无进位）
            }

            // shared = max * 2^-emax = 2^(E_max - emax)
            t_bfb shared_bf;
            TMULS(shared_bf, max_bf, __builtin_bit_cast(__bf16, RECIP_EMAX));
            t_e8b scale_e8m0; TCVT(scale_e8m0, shared_bf);     // bf16 -> e8m0 直转（须在位补前）
            global_iterator<gm_s, t_e8b> s_iter(
                reinterpret_cast<__fp8_e8m0 *>(scale) + row0 * scaleCols + kb);
            auto gs = s_iter(0, 0); TSTORE(gs, scale_e8m0);

            // === recip finalize：位补主路径 + inf/zero/special 三守卫（TCMPS+TSEL）===
            // 逐行对齐 AscendC ocp_new ComputeScaleOcp：主路径 recip = 0x7F00 - shared（二元
            //   TSUB 收 uint16），再对三类特殊 max/shared 用 TSEL 写回哨兵值：
            //     inf/nan (max_exp==0x7F80) -> 0x7F81；全零块 (max_exp==0) -> 0；
            //     special (shared==0x7F00) -> 0x0040（位补公式在此点算出 0，须修正）。
            //   eq_inf/eq_zero 取自 pre-multiply 的 max_u16（floor 后 max_bf 视图，TMULS 未改
            //   max_bf）。所有 uint16 量以 reinterpret_tile<uint16_t> 落在同型 t_bfb 载体，满足
            //   TSUB/TSEL 三参同 tile_shape（唯 TCMPS 允许 out≠in）。守卫是 fp4 kernel 原有语义，
            //   不随 fp8 母本一并省略（fp8 缺守卫是缺陷，非契约）；cmode 头修复后 TCMPS 原生可编。
            auto max_u16    = reinterpret_tile<uint16_t>(max_bf);
            auto shared_u16 = reinterpret_tile<uint16_t>(shared_bf);
            t_bfb recip_bf, eqinf_bf, eqzero_bf, eqspc_bf, k_bf;
            auto recip_u16  = reinterpret_tile<uint16_t>(recip_bf);
            auto eq_inf     = reinterpret_tile<uint16_t>(eqinf_bf);
            auto eq_zero    = reinterpret_tile<uint16_t>(eqzero_bf);
            auto eq_special = reinterpret_tile<uint16_t>(eqspc_bf);
            auto k_u16      = reinterpret_tile<uint16_t>(k_bf);
            TCMPS(eq_inf,     max_u16,    BF16_EXP_MASK);              // NOT finite
            TCMPS(eq_zero,    max_u16,    static_cast<uint16_t>(0));   // all-zero block
            TCMPS(eq_special, shared_u16, BF16_EXP_BIAS);             // shared==0x7F00
            TEXPANDS(k_u16, BF16_EXP_BIAS);
            TSUB(recip_u16, k_u16, shared_u16);                       // 0x7F00 - shared
            TEXPANDS(k_u16, BF16_NAN_PATTERN);
            TSEL(recip_u16, eq_inf, k_u16);                           // inf 命中 -> 0x7F81
            TEXPANDS(k_u16, static_cast<uint16_t>(0));
            TSEL(recip_u16, eq_zero, k_u16);                          // 全零命中 -> 0
            TEXPANDS(k_u16, BF16_SPECIAL_EXP);
            TSEL(recip_u16, eq_special, k_u16);                       // special 命中 -> 0x0040
            t_fb recip_f; TCVT(recip_f, recip_bf);                    // bf16 -> fp32
            t_o oq;
            if constexpr (std::is_same_v<InT, float>) {
                TROWEXPANDMUL(xin, xin, recip_f);              // fp32 域直乘（无预转）
                TCVT(oq, xin);                                 // fp32 -> fp4
            } else {
                t_f xf; TCVT(xf, xin);                         // bf16/half -> fp32
                TROWEXPANDMUL(xf, xf, recip_f);                // 逐行标量广播乘
                TCVT(oq, xf);                                  // fp32 -> fp4
            }
            global_iterator<gm_y, t_o> y_iter(y_u8 + row0 * (N / 2) + kb * (BlockSize / 2));
            auto gy = y_iter(0, 0); TSTORE(gy, oq);            // 打包窄化落 kb*(BlockSize/2) 字节
        }
        // 奇尾 scale 列补 0x00 E8M0（golden _pad_to_even 用 2^-127 == E8M0 0x00）。
        if constexpr (oddTail) {
            t_e8b zpad;
            TEXPANDS(zpad, __builtin_bit_cast(__fp8_e8m0, static_cast<uint8_t>(0)));
            global_iterator<gm_s, t_e8b> zs_iter(
                reinterpret_cast<__fp8_e8m0 *>(scale) + row0 * scaleCols + numKb);
            auto gzs = zs_iter(0, 0); TSTORE(gzs, zpad);
        }
    };

    // 单个 PE（编译期常量 Pe）：算自己那段连续行 [row_begin, row_begin+SubM)。
    //   前 row_rem 个 PE 各多 1 行，起点仍连续。row_begin/SubM/seg_tail 全 constexpr。
    auto run_pe = [&]<int Pe>() {
        constexpr int row_base  = M / kPeNum;
        constexpr int row_rem   = M % kPeNum;
        constexpr int SubM      = row_base + (Pe < row_rem ? 1 : 0);
        constexpr int row_begin = (Pe < row_rem)
                                      ? Pe * (row_base + 1)
                                      : row_rem * (row_base + 1) + (Pe - row_rem) * row_base;
        if constexpr (SubM > 0) {
            constexpr int TileM    = tail_ocp_fp4_detail::tilem_max(BlockSize, sizeof(InT)); // BS=32 -> 64
            constexpr int seg_full = SubM / TileM;   // SubM>TileM 时循环多个 full-tile
            constexpr int seg_tail = SubM % TileM;   // 余行（< TileM），boxed 尾块
            for (int lm = 0; lm < seg_full; ++lm) {
                process_tile.template operator()<TileM, TileM>(row_begin + lm * TileM);
            }
            if constexpr (seg_tail > 0) {
                process_tile.template operator()<TileM, seg_tail>(row_begin + seg_full * TileM);
            }
        }
    };

    // 运行期按 tid 分派到编译期展开的 per-PE 实例（kPeNum 编译期已知 = 4）。
    switch (static_cast<int>(tid)) {
        case 0: run_pe.template operator()<0>(); break;
        case 1: run_pe.template operator()<1>(); break;
        case 2: run_pe.template operator()<2>(); break;
        case 3: run_pe.template operator()<3>(); break;
        default: break;
    }
}

} // namespace supernpu::tile_isa::mxquant

#endif
