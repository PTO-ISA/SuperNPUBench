#ifndef SUPERNPU_DYNAMIC_MX_QUANT_TAIL_OCP_FP4_DYN_HPP
#define SUPERNPU_DYNAMIC_MX_QUANT_TAIL_OCP_FP4_DYN_HPP

#include <common/pto_tileop.hpp>
#include <cstdint>
#include "solution/quant/dynamic_mx_quant/dynamic_mx_quant_common.hpp"
// 复用静态版 tail_ocp_fp4_detail::{pow2_floor, tilem_max}（纯 BlockSize/inBytes 函数，
// 编译期常量），避免重复定义。静态 kernel 模板未实例化则不产码。
#include "solution/quant/dynamic_mx_quant/dynamic_mx_quant_tail_ocp_fp4.hpp"

namespace supernpu::tile_isa::mxquant {

// ===========================================================================
// TAIL-OCP-FP4 —— 运行期动态 shape 版（DYNAMIC SHAPE）
//
// 与静态 dynamic_mx_quant_tail_ocp_fp4 逐 op 等价（InT 分派值域归约 / fp4(e2m1) 打包
// 输出 / e8m0 scale / 倒数位补主路径 + inf/zero/special 三守卫），唯一区别：**M、N 在
// 编译期不可知，运行期由 tiling 指针传入，全部切分参数运行期计算**。BlockSize 是属性
// → 保留模板参（编译期）。动态化范式与 tail_ocp_fp8_dyn 完全一致：
//   · physical tile 形状（TileM×列宽）仍编译期锁定 —— TileM = tilem_max(BlockSize,
//     sizeof InT)，纯由 BlockSize+输入宽度决定，与 M/N 无关。
//   · Valid 有效尺寸全部下放运行期：数据/reduce tile 声明 ValidRow=-1（DYNAMIC），
//     构造时传运行期 vr。列 valid 保持编译期静态（BlockSize 或 1）—— B.DIM 立即数
//     要求列维编译期可知。
//   · global_tensor 用 RowMajor<-1,-1>，构造传运行期 shape；行 stride 以 dynamicCol
//     作为行 stride（pto_tile.hpp），故多行 strided 列块 load/store 正确。不能用
//     global_iterator（依赖编译期 RowStride）。
//   · full-tile 与尾块共用同一 Valid=-1 类型，仅 ctor 传不同 vr → 消除 boxed 编译期
//     特例，process_tile 退化为普通运行期 lambda，PE 分派退化为运行期公式。
//
// fp4 专属（不随 fp8 母本省略）：
//   · InT 分派 reduce：bf16 原生取指数；half/fp32 经 fp32 域 mask floor（避 narrowing
//     round 进位）再窄化。
//   · 打包输出：每字节 2 个 4bit 元素 → 每行 N/2 字节（gm_y 列 = N/2 字节域），TSTORE
//     落 kb*(BlockSize/2) 字节偏移；emax 由 OutT 派生。
//   · inf/zero/special 三守卫（TCMPS 出 predicate + 嵌套 TSEL 写哨兵）。
//   · oddTail scale 补列（numKb 奇数）改运行期 if。
//
// SPMD：kPeNum=1（默认，单 PE 全算）/ kPeNum=4（按 tid 切 4 段，须 gfrun -s
//   softcore.multiThreadNum=4）。tiling[0]=M（行/自由轴），tiling[1]=N（尾轴/量化轴，
//   N%BlockSize==0）。
// ===========================================================================
template <int BlockSize = 32, typename OutT = __fp4_e2m1x2, typename InT = __bf16,
          int kPeNum = 1>
void dynamic_mx_quant_tail_ocp_fp4_dyn(InT *x, OutT *y, uint8_t *scale,
                                       const int64_t *tiling) {
    static_assert(BlockSize % 32 == 0,
                  "fp4 output block is BlockSize/2 packed bytes; BlockSize must be "
                  "a multiple of 32 so the packed row is 32B-column-aligned");
    static_assert(std::is_same_v<InT, __bf16> || std::is_same_v<InT, __half> ||
                      std::is_same_v<InT, float>,
                  "InT must be one of {__bf16, __half, float}");
    static_assert(kPeNum == 1 || kPeNum == 4,
                  "kPeNum must be 1 (single PE) or 4 (SoftCore kCorePeCount)");

    using namespace pto;

    // emax 由 OutT 派生（逐值等价静态版）。
    constexpr uint16_t RECIP_EMAX = recip_emax_bits<OutT>();

    // physical tile 行高 —— 仅由 BlockSize + 输入宽度决定（编译期），与运行期 M/N 无关。
    constexpr int TileM = tail_ocp_fp4_detail::tilem_max(BlockSize, sizeof(InT)); // BS=32 -> 64

    // ---- 运行期 shape 与派生量 ----
    const int64_t M         = tiling[0];
    const int64_t N         = tiling[1];
    const int64_t numKb     = N / BlockSize;
    const int64_t scaleCols = ((numKb + 1) / 2) * 2; // even-align 补列
    const bool    oddTail   = (numKb % 2) != 0;      // 奇尾 padding scale 列须写 0x00

    const uint32_t tid = get_thread_idx();
    if (static_cast<int>(tid) >= kPeNum) return; // 冗余 PE 不发指令

    uint8_t     *y_u8     = reinterpret_cast<uint8_t *>(y);
    __fp8_e8m0  *scale_e8 = reinterpret_cast<__fp8_e8m0 *>(scale);

    // 动态 global_tensor：RowMajor<-1,-1>。x 行 stride=N；y 打包字节域行 stride=N/2；
    //   scale compact 行 stride=scaleCols。
    using gm_x = global_tensor<InT,        RowMajor<-1, -1>>;
    using gm_y = global_tensor<uint8_t,    RowMajor<-1, -1>>;
    using gm_s = global_tensor<__fp8_e8m0, RowMajor<-1, -1>>;

    // 动态 Valid tile：physical [TileM, 列]，ValidRow=-1（运行期 ctor 传 vr），列静态。
    //   全宽 tile physical Cols=BlockSize；列向量 reduce tile physical Cols=1（匹配 model
    //   rowReduce 无条件 col=1，令 reduce→下游 physical 列全等，绕过契约，见静态版注释）。
    using t_x   = Tile<Location::Vec, InT,        TileM, BlockSize, BLayout::RowMajor, -1, BlockSize>;
    using t_hb  = Tile<Location::Vec, __half,     TileM, 1,         BLayout::RowMajor, -1, 1>;
    using t_bfb = Tile<Location::Vec, __bf16,     TileM, 1,         BLayout::RowMajor, -1, 1>;
    using t_e8b = Tile<Location::Vec, __fp8_e8m0, TileM, 1,         BLayout::RowMajor, -1, 1>;
    using t_fb  = Tile<Location::Vec, float,      TileM, 1,         BLayout::RowMajor, -1, 1>;
    using t_f   = Tile<Location::Vec, float,      TileM, BlockSize, BLayout::RowMajor, -1, BlockSize>;
    using t_o   = Tile<Location::Vec, OutT,       TileM, BlockSize, BLayout::RowMajor, -1, BlockSize>;

    // 单个 tile-行块的完整计算（scale pass + recip finalize + data pass）。row0 = 全局
    //   起始行；validRows = 活跃行数（full-tile: TileM；尾块: seg_tail<TileM）。全部 tile
    //   用同一 Valid=-1 类型，ctor 传 vr。
    auto process_tile = [&](int64_t row0, int64_t validRows) {
        const size_t vr = static_cast<size_t>(validRows);
        for (int64_t kb = 0; kb < numKb; ++kb) {
            // === scale pass：value-domain reduce（InT 分派），floor 指数 ===
            gm_x gx(x + row0 * N + kb * BlockSize,
                    static_cast<int>(M), static_cast<int>(N));
            t_x xin(vr); TLOAD(xin, gx);

            t_bfb max_bf(vr);
            if constexpr (std::is_same_v<InT, __half>) {
                t_x  abs_h(vr);  TABS(abs_h, xin);
                t_hb max_h(vr);  TROWMAX(max_h, abs_h);       // half 域归约
                t_fb max_f(vr);  TCVT(max_f, max_h);          // half -> fp32（精确加宽）
                auto max_u32 = reinterpret_tile<uint32_t>(max_f);
                TANDS(max_u32, max_u32, FP32_EXP_MASK);       // fp32 域 floor 到 2^E（无进位）
                TCVT(max_bf, max_f);                          // fp32 -> bf16（尾数=0，精确）
            } else if constexpr (std::is_same_v<InT, float>) {
                t_x  abs_f(vr);  TABS(abs_f, xin);
                t_fb max_f(vr);  TROWMAX(max_f, abs_f);       // fp32 域归约
                auto max_u32 = reinterpret_tile<uint32_t>(max_f);
                TANDS(max_u32, max_u32, FP32_EXP_MASK);       // fp32 域 floor（无进位）
                TCVT(max_bf, max_f);                          // fp32 -> bf16（尾数=0，精确）
            } else {
                t_x  abs_bf(vr); TABS(abs_bf, xin);           // bf16 原生
                TROWMAX(max_bf, abs_bf);                      // bf16 域归约 -> max_bf
                auto max_u16 = reinterpret_tile<uint16_t>(max_bf);
                TANDS(max_u16, max_u16, BF16_EXP_MASK);       // 直接取指数（无转换->无进位）
            }

            // shared = max * 2^-emax = 2^(E_max - emax)
            t_bfb shared_bf(vr);
            TMULS(shared_bf, max_bf, __builtin_bit_cast(__bf16, RECIP_EMAX));
            t_e8b scale_e8m0(vr); TCVT(scale_e8m0, shared_bf); // bf16 -> e8m0 直转（须在位补前）
            gm_s gs(scale_e8 + row0 * scaleCols + kb,
                    static_cast<int>(M), static_cast<int>(scaleCols));
            TSTORE(gs, scale_e8m0);

            // === recip finalize：位补主路径 + inf/zero/special 三守卫（TCMPS+TSEL）===
            // 逐行对齐静态版 / AscendC ocp_new ComputeScaleOcp（见静态头注释）。eq_inf/
            //   eq_zero 取自 pre-multiply 的 max_u16（floor 后 max_bf 视图，TMULS 未改 max_bf）。
            auto max_u16    = reinterpret_tile<uint16_t>(max_bf);
            auto shared_u16 = reinterpret_tile<uint16_t>(shared_bf);
            t_bfb recip_bf(vr), eqinf_bf(vr), eqzero_bf(vr), eqspc_bf(vr), k_bf(vr);
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
            t_fb recip_f(vr); TCVT(recip_f, recip_bf);                // bf16 -> fp32

            // === data pass ===
            t_o oq(vr);
            if constexpr (std::is_same_v<InT, float>) {
                TROWEXPANDMUL(xin, xin, recip_f);             // fp32 域直乘（无预转，复用 xin）
                TCVT(oq, xin);                                // fp32 -> fp4
            } else {
                t_f xf(vr); TCVT(xf, xin);                    // bf16/half -> fp32
                TROWEXPANDMUL(xf, xf, recip_f);               // 逐行标量广播乘
                TCVT(oq, xf);                                 // fp32 -> fp4
            }
            gm_y gy(y_u8 + row0 * (N / 2) + kb * (BlockSize / 2),
                    static_cast<int>(M), static_cast<int>(N / 2));
            TSTORE(gy, oq);                                   // 打包窄化落 kb*(BlockSize/2) 字节
        }
        // 奇尾 scale 列补 0x00 E8M0（golden _pad_to_even 用 2^-127 == E8M0 0x00）。
        if (oddTail) {
            t_e8b zpad(vr);
            TEXPANDS(zpad, __builtin_bit_cast(__fp8_e8m0, static_cast<uint8_t>(0)));
            gm_s gzs(scale_e8 + row0 * scaleCols + numKb,
                     static_cast<int>(M), static_cast<int>(scaleCols));
            TSTORE(gzs, zpad);
        }
    };

    // ---- L1 行切分（运行期公式）：前 row_rem 个 PE 各多 1 行，起点连续 ----
    const int64_t row_base = M / kPeNum;
    const int64_t row_rem  = M % kPeNum;
    const int64_t itid     = static_cast<int64_t>(tid);
    const int64_t SubM     = row_base + (itid < row_rem ? 1 : 0);
    const int64_t row_begin = (itid < row_rem)
                                  ? itid * (row_base + 1)
                                  : row_rem * (row_base + 1) + (itid - row_rem) * row_base;
    if (SubM == 0) return; // M<kPeNum 时的空 PE

    // ---- L2 段内 tiling（运行期）：seg_full 个 full-tile + 可选 seg_tail 尾块 ----
    const int64_t seg_full = SubM / TileM;
    const int64_t seg_tail = SubM % TileM;
    for (int64_t lm = 0; lm < seg_full; ++lm) {
        process_tile(row_begin + lm * TileM, TileM);
    }
    if (seg_tail > 0) {
        process_tile(row_begin + seg_full * TileM, seg_tail);
    }
}

} // namespace supernpu::tile_isa::mxquant

#endif
