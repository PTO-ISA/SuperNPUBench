#ifndef SUPERNPU_DYNAMIC_MX_QUANT_TAIL_OCP_FP8_HPP
#define SUPERNPU_DYNAMIC_MX_QUANT_TAIL_OCP_FP8_HPP

#include <common/pto_tileop.hpp>
#include <cstdint>
#include "solution/quant/dynamic_mx_quant/dynamic_mx_quant_common.hpp"

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
// row-reduce 源列切分粒度（#585 / ASL v0.58.6.0：row-reduce 源 tile 物理字节 <= 2048）。
//   全宽 reduce 源 [TileMv, BlockSize] 常超 2048B（half [128,32]=8192B）→ 沿 BlockSize 切列成
//   nPart 个 [TileMv, RSC] 子块（各 <=2048B），逐块 TROWMAX 出 [TileMv,1] partial，再 TMAX 合并。
//   RSC = pow2_floor(2048/(TileMv*sizeof(InT)))，钳到 [1,BlockSize] 且整除 BlockSize。对齐官方
//   标准归约 kernel reducemax_rowvec.hpp「切列 + TMAX 累积」（行归约不能拆行 subview：
//   B.SUBVIEW/B.ASSEMBLE 仅 CUBE 布局，行 band 输出 <128B 非法分片）。
constexpr int reduce_slice_cols(int tileM, int inBytes, int blockSize) {
    int rc = pow2_floor(2048 / (tileM * inBytes));
    if (rc < 1) rc = 1;
    if (rc > blockSize) rc = pow2_floor(blockSize);
    while (blockSize % rc != 0) rc /= 2;
    return rc < 1 ? 1 : rc;
}
// 最大可支持 TileM —— **仅由 blocksize 决定, 与 SubM 无关**：
//   physical 行高须 >= 128（floorRows）：reduce 输出下游有 e8m0 列向量 tile [TileM,1]，
//     只占 TileM 字节；当 TileM*1B < 128B 最小 TSize 时被 padding 撑高 → 其 capacity 派生
//     physical Row 翻倍，违反 pto-spec PTO-TILE-TCVT「ordinary TCVT 源/目的 physical Row
//     相等」(Row = DerivedTileRows(capacity,Col,dtype)=cap*8/(Col*bits))，触发 TileOP #42
//     的 static_assert(SrcDerivedRows==DstDerivedRows)。e8m0=8-bit → TileM >= 128B/1B = 128
//     即最窄列 tile 恰达 128B、DerivedRows=TileM，与其它 16/32-bit 列 tile 全等 → 全链合法。
//   上限受 fp32 数据 tile [TileM,BlockSize] <= 256KB 约束：BS=32 → [128,32]=16KB，安全。
//   TileMmax = 满足上限的最大 2 的幂，但至少 floorRows(128)。BS=32 -> 128。
// SubM 只决定循环次数 (seg_full = SubM/TileMmax), 不改 TileM 本身 —— 不能把 SubM 当 tile 行。
// 注：TileM 由 32(旧 gfsim #605 规避的 4KB 预算) 提到 128 是为满足上述 TCVT physical-Row 契约;
//     纯 tiling 参数，功能/精度逐字节不变。旧的「广播源 [TileM,1] 单-128B-CELL」#605 规避
//     被此契约取代(dmxq 不在 gfsim pass-list，该 gfsim-only 权衡可接受)。
constexpr int tilem_max(int blockSize) {
    const int floorRows = 128;   // e8m0 列 tile [TileM,1] 达 128B 最小 TSize 的下限
    const int budgetMax = 262144 / (blockSize * static_cast<int>(sizeof(float)));  // [TileM,BS]fp32<=256KB
    int t = pow2_floor(budgetMax);
    if (t < floorRows) t = floorRows;
    // 数据集偏小时无需超大 tile：钉到 floorRows(128) 即满足契约，避免 boxed 尾块浪费容量。
    if (t > floorRows) t = floorRows;
    return t;
}
} // namespace tail_ocp_fp8_detail

// ===========================================================================
// TAIL-OCP-FP8 正式 kernel (固定 SPMD 4-PE) —— InT(bf16/half/fp32) in / e4m3 out / e8m0 scale /
// BlockSize=32 / recip = 0x7F00 - shared 主路径 (TEXPANDS+TSUB，计算算法与 tail_ocp_fp4 统一)。
// ⚠ 暂未补 fp4 的 inf/zero/special 三守卫 (TCMPS+TSEL)：随机基准数据不触发，output byte-exact；
//   缺守卫的三种边界见下方 recip finalize 处 TODO。原逐元素算法源自 single-PE 探针
//   probe_dynamic_mx_quant_tail_ocp_fp8_newcalc，唯一区别：外层 M-tile 循环按 get_thread_idx()
//   切成 4 份，每个 PE-线程只算自己那 1/4 的 M 行。
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
template <int M, int N, int BlockSize = 32, typename InT = __half>
void dynamic_mx_quant_tail_ocp_fp8(InT *x, __fp8_e4m3 *y, uint8_t *scale) {
    static_assert(M > 0 && N > 0, "dim must be positive");
    static_assert(N % BlockSize == 0, "N must be multiple of BlockSize");
    static_assert(BlockSize % 32 == 0,
                  "fp8 block = BlockSize bytes; BlockSize must be a multiple of "
                  "32 so the output tile is 32B-column-aligned");
    static_assert(std::is_same_v<InT, __bf16> || std::is_same_v<InT, __half> ||
                      std::is_same_v<InT, float>,
                  "InT must be one of {__bf16, __half, float}");

    using namespace pto;

    // FP32_EXP_MASK / recip_emax_bits 复用 common.hpp（逐值等价，非改数值）：
    //   FP32_EXP_MASK (common:38) = 0x7F800000 — fp32 指数位域，清尾数+符号 = floor 到 2^E。
    //   recip_emax_bits<__fp8_e4m3>() (common:77) = BF16_ONE(0x3f80) - FP8_E4M3_EMAX(0x0400)
    //     = 0x3b80 = bf16 位型 2^-8 (emax_dst=8)，等于原硬编码常量。
    constexpr uint16_t RECIP_EMAX = recip_emax_bits<__fp8_e4m3>(); // 0x3b80
    // recip 主路径 = 0x7F00 - shared（TEXPANDS(BF16_EXP_BIAS)+TSUB，与 tail_ocp_fp4 统一）。
    //   旧的两步位补 TXORS(0xFFFF)+TSUBS(0x80FF) 已弃用：算同一结果，但 int16 视图上的
    //   TXORS/TSUBS 会触发 gfrun res_check openat 落盘缺陷（实证：换 TEXPANDS+TSUB 后 openat
    //   恢复、output byte-exact）。BF16_EXP_BIAS(=0x7F00) 见 common.hpp。

    // TileM 不再全局推导 —— 移入 run_pe 按 per-PE SubM 推 (见 tail_ocp_fp8_detail::tilem_max)。
    constexpr int numKb     = N / BlockSize;
    constexpr int scaleCols = ((numKb + 1) / 2) * 2;
    constexpr bool oddTail  = (numKb % 2) != 0;   // 奇尾 padding scale 列须写 0x00（与 fp4 统一）
    constexpr int kPeNum    = 4;  // SoftCore.h kCorePeCount，multiThreadNum 仅 1|4 合法
    const uint32_t tid = get_thread_idx();          // 0..3

    uint8_t *y_u8 = reinterpret_cast<uint8_t *>(y);

    using gm_x = global_tensor<InT,        RowMajor<M, N>>;
    using gm_y = global_tensor<uint8_t,    RowMajor<M, N>>;
    using gm_s = global_tensor<__fp8_e8m0, RowMajor<M, scaleCols>>;

    // 单个 tile-行块的完整计算 (scale pass + data pass)。ValidRows = 该 tile 活跃行数
    //   (full-tile: TileM；尾块: seg_tail<TileM，boxed)。row0 = 该 tile 的全局起始行。
    //   物理 tile 恒 TileM×BlockSize (logicalTileBytes>=512B，避免 sub-512B spill)，boxed
    //   ValidRow=ValidRows 只触碰活跃行。base 指针按 row0 偏移，段起点可非 TileM 对齐。
    // TileMv = 该 PE 的物理 tile 行高 (per-PE 编译期常量, 由 tilem_max 推得, 2 的幂)。
    auto process_tile = [&]<int TileMv, int ValidRows>(int row0) {
        // 列向量中间 tile (reduce 输出及其下游) 声明 physical Cols=1 —— 匹配 model
        //   d8903938 rowReduce 无条件 col=1 (Block.cpp:2069)，令 reduce→TCVT 两侧
        //   physical Cols 全等，绕过 ValidateOperandContract「TCVT matching logical
        //   shapes」契约 (RECORD 问题22 / ISSUE_reduce_output_stride_tail.md)。0.58.3
        //   工具链头已删 32B 列对齐 static_assert，故 Cols=1 可直接构造。
        //   全宽 tile (t_h/t_f/t_o) 仍 physical Cols=BlockSize。
        using t_h   = Tile<Location::Vec, InT,        TileMv, BlockSize, BLayout::RowMajor, ValidRows, BlockSize>;
        using t_hb  = Tile<Location::Vec, InT,        TileMv, 1,         BLayout::RowMajor, ValidRows, 1>;
        using t_bfb = Tile<Location::Vec, __bf16,     TileMv, 1,         BLayout::RowMajor, ValidRows, 1>;
        using t_e8b = Tile<Location::Vec, __fp8_e8m0, TileMv, 1,         BLayout::RowMajor, ValidRows, 1>;
        using t_fb  = Tile<Location::Vec, float,      TileMv, 1,         BLayout::RowMajor, ValidRows, 1>;
        using t_f   = Tile<Location::Vec, float,      TileMv, BlockSize, BLayout::RowMajor, ValidRows, BlockSize>;
        using t_o   = Tile<Location::Vec, __fp8_e4m3, TileMv, BlockSize, BLayout::RowMajor, ValidRows, BlockSize>;

        // #585 列分区归约（half 域）：源 [TileMv,BlockSize] 超 2048B → 沿列切 nPart 个 [TileMv,RSC]
        //   子块各 <=2048B，逐块 TROWMAX→[TileMv,1] partial，TMAX 合并；输出满行、不触 #42。
        constexpr int RSC   = tail_ocp_fp8_detail::reduce_slice_cols(TileMv, sizeof(InT), BlockSize);
        constexpr int nPart = BlockSize / RSC;
        using t_hc  = Tile<Location::Vec, InT, TileMv, RSC, BLayout::RowMajor, ValidRows, RSC>;
        auto col_part_rowmax = [&](int colBase, t_hb &max_in) {
            global_iterator<gm_x, t_hc> it0(x + row0 * N + colBase);
            auto g0 = it0(0, 0);
            t_hc xs0; TLOAD(xs0, g0);
            t_hc as0; TABS(as0, xs0);
            TROWMAX(max_in, as0);
            for (int p = 1; p < nPart; ++p) {
                global_iterator<gm_x, t_hc> itp(x + row0 * N + colBase + p * RSC);
                auto gp = itp(0, 0);
                t_hc xsp; TLOAD(xsp, gp);
                t_hc asp; TABS(asp, xsp);
                t_hb pm; TROWMAX(pm, asp);
                TMAX(max_in, max_in, pm);
            }
        };

        for (int kb = 0; kb < numKb; ++kb) {
            // === scale pass：value-domain reduce（InT 分派），floor 指数（与 fp4 统一）===
            // ⚠ round-mode 缺口：half/fp32 先在 fp32 域 mask floor 再窄化（避 narrowing 进位越
            //   2^k）；bf16 原生无 narrowing，直接取指数天然与截断一致。
            global_iterator<gm_x, t_h> x_iter(x + row0 * N + kb * BlockSize);
            auto gx = x_iter(0, 0);
            t_h xin; TLOAD(xin, gx);                             // 全宽 load 供 data pass 复用

            t_bfb max_bf;
            if constexpr (std::is_same_v<InT, __half>) {
                t_hb max_h; col_part_rowmax(kb * BlockSize, max_h);  // half 域列分区归约（#585）
                t_fb max_f; TCVT(max_f, max_h);                      // half -> fp32（精确加宽）
                auto max_u32 = reinterpret_tile<uint32_t>(max_f);
                TANDS(max_u32, max_u32, FP32_EXP_MASK);              // fp32 域 floor（无进位）
                TCVT(max_bf, max_f);                                 // fp32 -> bf16（尾数=0，精确）
            } else if constexpr (std::is_same_v<InT, float>) {
                t_hb max_f; col_part_rowmax(kb * BlockSize, max_f);  // fp32 域列分区归约（t_hb=InT=fp32）
                auto max_u32 = reinterpret_tile<uint32_t>(max_f);
                TANDS(max_u32, max_u32, FP32_EXP_MASK);              // fp32 域 floor（无进位）
                TCVT(max_bf, max_f);                                 // fp32 -> bf16（尾数=0，精确）
            } else {  // bf16：原生取指数（无转换 -> 无进位）
                col_part_rowmax(kb * BlockSize, max_bf);             // bf16 域列分区归约 -> max_bf
                auto max_u16 = reinterpret_tile<uint16_t>(max_bf);
                TANDS(max_u16, max_u16, BF16_EXP_MASK);
            }

            // shared = max * 2^-emax = 2^(E_max - emax)
            t_bfb shared_bf;
            TMULS(shared_bf, max_bf, __builtin_bit_cast(__bf16, RECIP_EMAX));
            t_e8b scale_e8m0; TCVT(scale_e8m0, shared_bf);       // bf16 -> e8m0 直转（须在 recip 前）
            global_iterator<gm_s, t_e8b> s_iter(
                reinterpret_cast<__fp8_e8m0 *>(scale) + row0 * scaleCols + kb);
            auto gs = s_iter(0, 0); TSTORE(gs, scale_e8m0);

            // === recip finalize：主路径 recip = 0x7F00 - shared（TEXPANDS+TSUB，与 fp4 统一）===
            // 写独立 recip_bf tile（不就地改 shared_bf），使 shared_u16/max_u16 可留给守卫读。
            // TODO(inf/zero/special 三守卫)：fp4 在此处对三类特殊 max/shared 用 TCMPS+TSEL 写哨兵
            //   （inf/nan max_exp==0x7F80 -> 0x7F81；全零块 max_exp==0 -> 0；shared==0x7F00 ->
            //   0x0040）。本 kernel 暂不补。缺守卫的后果：① shared==0x7F00 饱和点主路径算出 0
            //   -> 该 block 输出错；② inf/nan 输入 recip 错；③ 全零块防御缺失（output 仍 0）。
            //   随机基准数据三者都不触发 -> output byte-exact。补齐只需照抄 fp4 的 6 行 TCMPS/TSEL。
            auto shared_u16 = reinterpret_tile<uint16_t>(shared_bf);
            t_bfb recip_bf, k_bf;
            auto recip_u16 = reinterpret_tile<uint16_t>(recip_bf);
            auto k_u16     = reinterpret_tile<uint16_t>(k_bf);
            TEXPANDS(k_u16, BF16_EXP_BIAS);                      // 0x7F00
            TSUB(recip_u16, k_u16, shared_u16);                  // 0x7F00 - shared = 2^(8-E_max)
            t_fb recip_f; TCVT(recip_f, recip_bf);               // bf16 -> fp32

            // === data pass（InT 分派，与 fp4 统一）===
            t_o oq;
            if constexpr (std::is_same_v<InT, float>) {
                TROWEXPANDMUL(xin, xin, recip_f);                // fp32 域直乘（无预转）
                TCVT(oq, xin);                                   // fp32 -> e4m3
            } else {
                t_f xf; TCVT(xf, xin);                           // bf16/half -> fp32
                TROWEXPANDMUL(xf, xf, recip_f);                  // 逐行标量广播乘
                TCVT(oq, xf);                                    // fp32 -> e4m3
            }
            global_iterator<gm_y, t_o> y_iter(y_u8 + row0 * N + kb * BlockSize);
            auto gy = y_iter(0, 0); TSTORE(gy, oq);
        }
        // 奇尾 scale 列补 0x00 E8M0（golden _pad_to_even 用 2^-127 == E8M0 0x00）——与 fp4 统一。
        if constexpr (oddTail) {
            t_e8b zpad;
            TEXPANDS(zpad, __builtin_bit_cast(__fp8_e8m0, static_cast<uint8_t>(0)));
            global_iterator<gm_s, t_e8b> zs_iter(
                reinterpret_cast<__fp8_e8m0 *>(scale) + row0 * scaleCols + numKb);
            auto gzs = zs_iter(0, 0); TSTORE(gzs, zpad);
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
