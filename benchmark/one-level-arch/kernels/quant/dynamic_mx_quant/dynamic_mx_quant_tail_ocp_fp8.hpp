#ifndef SUPERNPU_DYNAMIC_MX_QUANT_TAIL_OCP_FP8_HPP
#define SUPERNPU_DYNAMIC_MX_QUANT_TAIL_OCP_FP8_HPP

#include <common/pto_tileop.hpp>
#include <cstdint>
#include "quant/dynamic_mx_quant/dynamic_mx_quant_common.hpp"

namespace supernpu::tile_isa::mxquant {

namespace tail_ocp_fp8_detail {
// 向下取 2 的幂：物理 tile 字节数必须是 2 的幂 (TSize 编码约束)，否则 LLVM 后端
// getSimpleVT 断言崩溃 (M=12/24/48 实证)。v<1 返回 0 (SubM=0 的空 PE)。
constexpr int pow2_floor(int v) {
    if (v < 1) return 0;
    int p = 1;
    while (p * 2 <= v) p *= 2;
    return p;
}
// 最大可支持 TileM —— **仅由 blocksize 决定, 与 SubM 无关**：
//   budgetMax = 8192/(BS*sizeof fp32)  —— data pass 的 fp32 中间量 tile <= 8KB (绑定约束)
//   floorMin  = 512/(BS*sizeof half)   —— 物理 tile >= 512B (避免 LinxV5 sub-512B spill)
//   TileMmax = pow2_floor(budgetMax) 抬到 floorMin。BS=32 -> 64。
// SubM 只决定循环次数 (seg_full = SubM/TileMmax), 不改 TileM 本身 —— 不能把 SubM 当 tile 行
// (SubM 可远大于 budgetMax, 直接当行数会爆 8KB)。
constexpr int tilem_max(int blockSize) {
    const int budgetMax = 8192 / (blockSize * static_cast<int>(sizeof(float)));   // 64 @ BS=32
    const int floorMin  = (512 / static_cast<int>(sizeof(__half)) + blockSize - 1) / blockSize; // 8 @ BS=32
    int t = pow2_floor(budgetMax);
    if (t > 0 && t < floorMin) t = floorMin;
    return t;
}
} // namespace tail_ocp_fp8_detail

// ===========================================================================
// TAIL-OCP-FP8 正式 kernel (固定 SPMD 4-PE) —— half in / e4m3 out / e8m0 scale /
// BlockSize=32 / 位补求倒数。逐元素算法与 single-PE 探针
// probe_dynamic_mx_quant_tail_ocp_fp8_newcalc 完全一致，唯一区别：把外层 M-tile
// 循环按 get_thread_idx() 切成 4 份，每个 PE-线程只算自己那 1/4 的 M 行。
//
// 动机 (源码确证)：单线程版把全部 full_m*numKb 个 tile-block 压在 Thread0/PE0 的一条私有
//   Vector ALU 流水上 (Core.cpp: vecTops[i] 每 PE 私有 aluPipe/fmaPipe/lnexpPipe)，导致
//   Vector 引擎 union≈总周期、BRob Full Stall 77%。本 kernel 8 个算子在 BS=32/half·fp32
//   配置下全落 PE 私有通路 (TROWMAX 经 IsRowReduceTree 降级私有 ALU、TROWEXPANDMUL 降级
//   私有 FMA、其余 ALU/FMA)，无一占用跨 PE SHARED 单例 → 按 M 切 4 线程近线性加速。
//
// SPMD 语义：runtime 把 [0,multiThreadNum) 所有线程 reset 到同一 entry PC (main.cpp)，
//   靠 kernel 内 get_thread_idx() (=SYS_LXLCID, 0..3) 自我切分，写不重叠的 M 行，无 barrier。
//   必须用 4 线程跑 (gfrun -s softcore.multiThreadNum=4 / gfsim --conf fourpe)；单线程跑本
//   变体只会写 1/4 输出。
//
// 两级切分模型：
//   L1 (行切分, 定 SubM)：把 M 行尽量均分给 kPeNum 个 PE，每个 PE 拿一段 **连续** 行区间
//     [row_begin, row_begin+SubM)；行数余数 (M%kPeNum) 按连续块规则分给前几个 PE (前 row_rem
//     个各多 1 行, 起点仍连续, 不散布)，负载均衡 Δ≤1，每 PE 数据连续利于 TMA/缓存。
//   L2 (段内 tiling, 定 TileM)：TileM = **仅由 blocksize 决定的固定上限** (tilem_max, BS=32→64,
//     受 fp32 中间量 tile<=8KB 约束, 且取 2 的幂避免 LLVM getSimpleVT 崩)；**不能把 SubM 当 tile
//     行** (SubM 可远大于 64, 直接当行会爆 8KB)。SubM 只决定循环次数：
//     seg_full = SubM/TileM 个 full-tile + seg_tail = SubM%TileM 余行 boxed 尾块。
//   例 M=680,4PE: 各 170 行 = 2×64 full + 42 tail；M=1024,4PE: 各 256 = 4×64 full + 0 tail。
//   实现：kPeNum 编译期已知 → 按 TID 编译期展开 (模板 lambda)，令 SubM/row_begin/seg_tail 均为
//   constexpr，满足 boxed 尾块 tile 的 validRow 编译期常量要求；运行期 switch(tid) 分派。
//
// 约束：BS=32/half·fp32 (守住 TROWMAX/TROWEXPANDMUL 私有通路两道门)。M/full_m 任意 (行粒度
//   均分, 尾块段内自理)。注：seg_tail>0 (M 非 TileM 整除) 会触碰 boxed sub-TileM reduce→TCVT
//   reduce→TCVT 形状契约缺陷 (全 mx_quant 家族共有, 见 RECORD 问题22 /
//   ISSUE_reduce_output_stride_tail.md)，该缺陷独立于本切分模型。
// ===========================================================================
template <int M, int N, int BlockSize = 32>
void dynamic_mx_quant_tail_ocp_fp8(__half *x, __fp8_e4m3 *y, uint8_t *scale) {
    static_assert(M > 0 && N > 0, "dim must be positive");
    static_assert(N % BlockSize == 0, "N must be multiple of BlockSize");
    static_assert(BlockSize % 32 == 0,
                  "fp8 block = BlockSize bytes; BlockSize must be a multiple of "
                  "32 so the output tile is 32B-column-aligned");

    using namespace pto;

    // FP32_EXP_MASK / recip_emax_bits 复用 common.hpp（逐值等价，非改数值）：
    //   FP32_EXP_MASK (common:38) = 0x7F800000 — fp32 指数位域，清尾数+符号 = floor 到 2^E。
    //   recip_emax_bits<__fp8_e4m3>() (common:77) = BF16_ONE(0x3f80) - FP8_E4M3_EMAX(0x0400)
    //     = 0x3b80 = bf16 位型 2^-8 (emax_dst=8)，等于原硬编码常量。
    constexpr uint16_t RECIP_EMAX = recip_emax_bits<__fp8_e4m3>(); // 0x3b80
    // 倒数位补常量 (同单线程版): recip_bits = 0x7F00 - shared_bits, 分两步 TXORS(0xFFFF)+TSUBS(0x80FF)。
    // 保持内联：common 走 finalize_recip_u16 的 TSUB 形式（含特殊值 TSEL），不提供这两步位补常量。
    constexpr uint16_t RECIP_XOR_NOT   = 0xFFFF; // 按位取反 (int16: -1)
    constexpr uint16_t RECIP_COMPL_SUB = 0x80FF; // 0xFFFF - 0x7F00 (int16 补码减法, 低16位不变)

    // TileM 不再全局推导 —— 移入 run_pe 按 per-PE SubM 推 (见 tail_ocp_fp8_detail::tilem_max)。
    constexpr int numKb     = N / BlockSize;
    constexpr int scaleCols = ((numKb + 1) / 2) * 2;
    constexpr int kPeNum    = 4;  // SoftCore.h kCorePeCount，multiThreadNum 仅 1|4 合法
    const uint32_t tid = get_thread_idx();          // 0..3

    uint8_t *y_u8 = reinterpret_cast<uint8_t *>(y);

    using gm_x = global_tensor<__half,     RowMajor<M, N>>;
    using gm_y = global_tensor<uint8_t,    RowMajor<M, N>>;
    using gm_s = global_tensor<__fp8_e8m0, RowMajor<M, scaleCols>>;

    // 单个 tile-行块的完整计算 (scale pass + data pass)。ValidRows = 该 tile 活跃行数
    //   (full-tile: TileM；尾块: seg_tail<TileM，boxed)。row0 = 该 tile 的全局起始行。
    //   物理 tile 恒 TileM×BlockSize (logicalTileBytes>=512B，避免 sub-512B spill)，boxed
    //   ValidRow=ValidRows 只触碰活跃行。base 指针按 row0 偏移，段起点可非 TileM 对齐。
    // TileMv = 该 PE 的物理 tile 行高 (per-PE 编译期常量, 由 tilem_max 推得, 2 的幂)。
    auto process_tile = [&]<int TileMv, int ValidRows>(int row0) {
        using t_h   = Tile<Location::Vec, __half,     TileMv, BlockSize, BLayout::RowMajor, ValidRows, BlockSize>;
        using t_hb  = Tile<Location::Vec, __half,     TileMv, BlockSize, BLayout::RowMajor, ValidRows, 1>;
        using t_bfb = Tile<Location::Vec, __bf16,     TileMv, BlockSize, BLayout::RowMajor, ValidRows, 1>;
        using t_e8b = Tile<Location::Vec, __fp8_e8m0, TileMv, BlockSize, BLayout::RowMajor, ValidRows, 1>;
        using t_fb  = Tile<Location::Vec, float,      TileMv, BlockSize, BLayout::RowMajor, ValidRows, 1>;
        using t_f   = Tile<Location::Vec, float,      TileMv, BlockSize, BLayout::RowMajor, ValidRows, BlockSize>;
        using t_o   = Tile<Location::Vec, __fp8_e4m3, TileMv, BlockSize, BLayout::RowMajor, ValidRows, BlockSize>;

        for (int kb = 0; kb < numKb; ++kb) {
            // === scale pass ===
            global_iterator<gm_x, t_h> x_iter(x + row0 * N + kb * BlockSize);
            auto gx = x_iter(0, 0);
            t_h xh;      TLOAD(xh, gx);
            t_h abs_h;   TABS(abs_h, xh);
            t_hb max_h;  TROWMAX(max_h, abs_h);
            t_fb max_f;  TCVT(max_f, max_h);                     // half -> fp32（精确无舍入）
            auto max_u32 = reinterpret_tile<uint32_t>(max_f);
            TANDS(max_u32, max_u32, FP32_EXP_MASK);              // floor 到 2^E_max（清尾数+符号）
            t_bfb max_bf; TCVT(max_bf, max_f);                   // fp32 -> bf16（尾数=0，精确）
            t_bfb shared_bf;
            TMULS(shared_bf, max_bf, __builtin_bit_cast(__bf16, RECIP_EMAX)); // 2^(E_max-8)
            t_e8b scale_e8m0; TCVT(scale_e8m0, shared_bf);       // bf16 -> e8m0 直转
            global_iterator<gm_s, t_e8b> s_iter(
                reinterpret_cast<__fp8_e8m0 *>(scale) + row0 * scaleCols + kb);
            auto gs = s_iter(0, 0); TSTORE(gs, scale_e8m0);

            // === data pass (NEWCALC: 位补求倒数, 复用 xh) ===
            auto sh_u16 = reinterpret_tile<int16_t>(shared_bf);
            TXORS(sh_u16, sh_u16, RECIP_XOR_NOT);                // 0xFFFF - bits (重打 I16 标签)
            TSUBS(sh_u16, sh_u16, RECIP_COMPL_SUB);              // -> 0x7F00 - bits = 2^(8-E_max)
            t_fb recip_f; TCVT(recip_f, shared_bf);              // bf16 -> fp32
            t_f xf;       TCVT(xf, xh);                          // half -> fp32 (复用 xh)
            TROWEXPANDMUL(xf, xf, recip_f);                      // x * (1/scale)
            t_o oq;       TCVT(oq, xf);                          // fp32 -> e4m3
            global_iterator<gm_y, t_o> y_iter(y_u8 + row0 * N + kb * BlockSize);
            auto gy = y_iter(0, 0); TSTORE(gy, oq);
        }
    };
    // 单个 PE (编译期常量 Pe) 的驱动：算自己那段连续行 [row_begin, row_begin+my_rows)，
    // 段内先 seg_full 个 TileM full-tile，再 (若有) 一个 seg_tail 行的 boxed 尾块。
    //   row_base = M / kPeNum;  row_rem = M % kPeNum
    //   Pe < row_rem -> my_rows = row_base+1, row_begin = Pe*(row_base+1)  (前几个 PE 各多 1 行)
    //   Pe >= row_rem-> my_rows = row_base,   row_begin = row_rem*(row_base+1)+(Pe-row_rem)*row_base
    // row_begin/my_rows/seg_tail 全为 constexpr → 满足 boxed 尾块 validRow 编译期常量要求。
    auto run_pe = [&]<int Pe>() {
        // L1: 行切分 -> SubM (本 PE 连续行段行数), 前 row_rem 个 PE 各多 1 行。
        constexpr int row_base  = M / kPeNum;
        constexpr int row_rem   = M % kPeNum;
        constexpr int SubM      = row_base + (Pe < row_rem ? 1 : 0);
        constexpr int row_begin = (Pe < row_rem)
                                      ? Pe * (row_base + 1)
                                      : row_rem * (row_base + 1) + (Pe - row_rem) * row_base;
        // L2: TileM = blocksize 决定的固定上限 (与 SubM 无关)；SubM 只决定循环次数。
        //     SubM=0 (M<kPeNum 时的空 PE) -> 不发 tile。
        if constexpr (SubM > 0) {
            constexpr int TileM    = tail_ocp_fp8_detail::tilem_max(BlockSize);  // BS=32 -> 64
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

    // 运行期按 tid 分派到编译期展开的 per-PE 实例 (kPeNum 编译期已知 = 4)。
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
