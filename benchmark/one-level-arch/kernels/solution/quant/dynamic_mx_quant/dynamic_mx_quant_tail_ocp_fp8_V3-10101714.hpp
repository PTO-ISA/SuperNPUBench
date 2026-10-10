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
// BlockSize=32 / recip = 0x7F00 - shared 主路径 (TSUB) + inf/zero/special 三守卫 (TCMPS+TSEL)，
// 计算算法与 tail_ocp_fp4 完全统一。三守卫见下方 recip finalize：inf/nan->0x7F81 / 全零->0 /
// special(shared==0x7F00)->0x0040。原逐元素算法源自 single-PE 探针
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
// 2D 切分模型（V3，与 dyn 版同公式；替代旧「L1 均分行 + L2 段内 tiling」两级模型）：
//   M 按 TileM=32 切成 rowTiles 个行 tile，行方向铺 peRows=min(rowTiles,4) 个 PE（tile
//   粒度均分、连续区间，前 rem 个 lane 各 +1 tile）；行方向用不满 4 PE 时，剩余 PE 切到
//   N 轴（peCols=4/peRows：rowTiles=1→1×4、=2→2×2、=3→3×1〔PE3 闲〕、>=4→4×1）。
//   kb（列块）区间在 colLane 间均分（前 rem 个各 +1 kb，连续区间）——kb 块沿 N 两两独立，
//   y/scale 以全局 kb 寻址，PE 间 store 天然不重叠；oddTail 补齐列由持有最末 kb 的 PE 写。
//   动机（2026-10-10 性能分析）：瓶颈 = per-kb 串行链周期 × 每 PE 链数（PipeView 实测链
//   ~350cyc、Vector BLOCK Retire 均值 ~1.2K；STQ 99.61% 在等生产者数据）。旧均分下 M<128
//   每 PE 扫全 N 且 tile 半幅（M=64：4PE×16行×512链）；2D 切分让链数/PE 随 peCols 减少
//   （M=64 → 4PE 各 32 行×256 链满幅，实测 1.45×）。
//   实现：kPeNum 编译期已知 → 按 TID 编译期展开 (模板 lambda)，切分参数均为 constexpr，
//   满足 boxed 尾块 tile 的 validRow 编译期常量要求；运行期 switch(tid) 分派。
//
// 约束：BS=32/half·fp32 (守住 TROWMAX/TROWEXPANDMUL 私有通路两道门)。M/N 任意 (N%BS==0，
//   尾块 tile 自理)。注：M%32≠0 的最后全局 tile 会触碰 boxed sub-TileM reduce→TCVT
//   形状契约缺陷 (全 mx_quant 家族共有, 见 RECORD 问题22 /
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

    // 【M32】TSTORE_CUBE 要求 GM 与 CUBE tile dtype 一致 → 输出 global 用 __fp8_e4m3（非 uint8 视图）。
    using gm_x = global_tensor<InT,        RowMajor<M, N>>;
    using gm_y = global_tensor<__fp8_e4m3, RowMajor<M, N>>;
    using gm_s = global_tensor<__fp8_e8m0, RowMajor<M, scaleCols>>;

    // 单个 tile-行块的完整计算 (scale pass + data pass)。ValidRows = 该 tile 活跃行数
    //   (full-tile: TileM；尾块: M%32≠0 的最后全局 tile，boxed)。row0 = 该 tile 的全局起始行。
    //   物理 tile 恒 TileM×BlockSize (logicalTileBytes>=512B，避免 sub-512B spill)，boxed
    //   ValidRow=ValidRows 只触碰活跃行。base 指针按 row0 偏移，段起点可非 TileM 对齐。
    // TileMv = 该 PE 的物理 tile 行高 (固定 32，CUBE_M32 cell)。kb 区间 [kbBegin, kbBegin+kbCount)
    //   为 2D 切分下本 PE 分到的列块范围（全局 kb 索引，store 天然不重叠）；
    //   ownsLastKb = 本 PE 持有最末 kb（oddTail 补齐列归属）。
    auto process_tile = [&]<int TileMv, int ValidRows>(int row0, int kbBegin,
                                                       int kbCount, bool ownsLastKb) {
        // 【M32 版】编译期 ValidRows（满足 TREDUCEPREFIXVIEW asm 立即数约束）。
        //   #311 归约输出宽 carrier（physical Col=BlockSize、valid Col=1）→ TREDUCEPREFIXVIEW 取首个
        //   128B CELL 窄视图 → TADD(zero+view) materialize 成普通窄 tile → 标量链全在窄 tile 上。
        //   inf/zero/special 三守卫（TCMPS/TSEL）见下方 recip finalize；CUBE_M32 上的 compare-select
        //   CELL 几何已由 SuperScalarModel #785 修复，守卫功能完整。参考 kernels/basic_op/fa/fa_lowp.hpp。
        using t_blk = VecTileM32<InT,        TileMv, BlockSize, ValidRows, BlockSize>; // 输入/abs（宽）
        using t_rw  = VecTileM32<InT,        TileMv, BlockSize, ValidRows, 1>;         // reduce 宽 carrier（#311）
        using t_rin = VecTileM32<InT,        TileMv, 1,         ValidRows, 1>;         // 窄 InT（view SubTile + materialize）
        using t_row = VecTileM32<__bf16,     TileMv, 1,         ValidRows, 1>;         // 窄 bf16（shared/recip）
        using t_frw = VecTileM32<float,      TileMv, 1,         ValidRows, 1>;         // 窄 fp32（floor/recip_f）
        using t_e8b = VecTileM32<__fp8_e8m0, TileMv, 1,         ValidRows, 1>;         // scale（窄）
        using t_f   = VecTileM32<float,      TileMv, BlockSize, ValidRows, BlockSize>; // data-pass fp32
        using t_o   = VecTileM32<__fp8_e4m3, TileMv, BlockSize, ValidRows, BlockSize>; // 输出

        for (int kb = kbBegin; kb < kbBegin + kbCount; ++kb) {
            // === scale pass：M32 全宽 load 一次（[TileMv,BlockSize]=2048B 免 #585 切片），行归约 ===
            global_iterator<gm_x, t_blk> x_iter(x + row0 * N + kb * BlockSize);
            auto gx = x_iter(0, 0);
            t_blk xin; TLOAD(xin, gx);                           // 全宽 load 供 data pass 复用
            t_blk absx; TABS(absx, xin);                         // |x|

            // #311 归约 + TREDUCEPREFIXVIEW 消费（照 fa_lowp）：宽 carrier 归约 → 零拷贝窄视图 →
            //   TADD(zero+view) materialize 成普通窄 InT tile。
            t_rw  max_w; TROWMAX(max_w, absx);                   // 宽 carrier 归约（#311 ✓）
            auto  max_view = TREDUCEPREFIXVIEW<t_rin>(max_w);    // 零拷贝窄视图（首个 128B CELL）
            t_rin zero_in;  TEXPANDS(zero_in, static_cast<InT>(0.0f));
            t_rin max_in;   TADD(max_in, zero_in, max_view);     // materialize → 普通窄 InT（view 不能被 reinterpret，必需）

            // floor 指数 → shared + 收集 inf/zero 守卫 predicate（读 floor 后 max 位型；OCP/fp4 统一）。
            t_row shared_bf, eqinf_bf, eqzero_bf;
            auto eq_inf  = reinterpret_tile<uint16_t>(eqinf_bf);
            auto eq_zero = reinterpret_tile<uint16_t>(eqzero_bf);
            if constexpr (std::is_same_v<InT, __bf16>) {
                auto max_u16 = reinterpret_tile<uint16_t>(max_in);
                TANDS(max_u16, max_u16, BF16_EXP_MASK);          // bf16 floor
                TCMPS(eq_inf,  max_u16, BF16_EXP_MASK);          // NOT finite (max_exp==0x7F80)
                TCMPS(eq_zero, max_u16, static_cast<uint16_t>(0)); // all-zero block
                TMULS(shared_bf, max_in, __builtin_bit_cast(__bf16, RECIP_EMAX)); // 2^(E_max-8)
            } else {
                t_frw max_f; TCVT(max_f, max_in);                // half/fp32 -> 窄 fp32
                auto max_u32 = reinterpret_tile<uint32_t>(max_f);
                TANDS(max_u32, max_u32, FP32_EXP_MASK);          // fp32 域 floor
                t_row max_bf; TCVT(max_bf, max_f);               // 窄 fp32 -> 窄 bf16（尾数=0）
                auto maxbf_u16 = reinterpret_tile<uint16_t>(max_bf);
                TCMPS(eq_inf,  maxbf_u16, BF16_EXP_MASK);
                TCMPS(eq_zero, maxbf_u16, static_cast<uint16_t>(0));
                TMULS(shared_bf, max_bf, __builtin_bit_cast(__bf16, RECIP_EMAX));
            }
            t_e8b scale_e8m0; TCVT(scale_e8m0, shared_bf);       // bf16 -> e8m0
            global_iterator<gm_s, t_e8b> s_iter(
                reinterpret_cast<__fp8_e8m0 *>(scale) + row0 * scaleCols + kb);
            auto gs = s_iter(0, 0); TSTORE(gs, scale_e8m0);

            // === recip finalize：主路径 recip = 0x7F00 - shared（TEXPANDS+TSUB）+ inf/zero/special 三守卫 ===
            //   与 V1(RowMajor)/tail_ocp_fp4 统一：inf/nan->0x7F81 / 全零->0 / special(shared==0x7F00)->0x0040。
            //   CUBE_M32 上的 TCMPS/TSEL：早期 gfrun compare-select 校验器用 RowMajor `col==physicalCol`
            //     拒 CUBE_M32 cell 对齐列而崩；已由 SuperScalarModel #785（按 ASL 实现 CUBE PredicateCell）
            //     修复，全量 dmxq gfrun 逐用例 PASS。bench_small 数据无特殊值，守卫为 compute-only
            //     （TLSU-bound 下不增墙钟）。
            auto shared_u16 = reinterpret_tile<uint16_t>(shared_bf);
            t_row recip_bf, eqspc_bf, k_bf;
            auto recip_u16  = reinterpret_tile<uint16_t>(recip_bf);
            auto eq_special = reinterpret_tile<uint16_t>(eqspc_bf);
            auto k_u16      = reinterpret_tile<uint16_t>(k_bf);
            TCMPS(eq_special, shared_u16, BF16_EXP_BIAS);        // shared==0x7F00
            TEXPANDS(k_u16, BF16_EXP_BIAS);                      // 0x7F00
            TSUB(recip_u16, k_u16, shared_u16);                  // 0x7F00 - shared = 2^(8-E_max)
            TEXPANDS(k_u16, BF16_NAN_PATTERN);
            TSEL(recip_u16, eq_inf, k_u16);                      // inf/nan -> 0x7F81
            TEXPANDS(k_u16, static_cast<uint16_t>(0));
            TSEL(recip_u16, eq_zero, k_u16);                     // all-zero -> 0
            TEXPANDS(k_u16, BF16_SPECIAL_EXP);
            TSEL(recip_u16, eq_special, k_u16);                  // special -> 0x0040
            t_frw recip_f; TCVT(recip_f, recip_bf);              // 窄 bf16 -> 窄 fp32

            // === data pass（InT 分派）===
            t_o oq;
            if constexpr (std::is_same_v<InT, float>) {
                TROWEXPANDMUL(xin, xin, recip_f);                // fp32 域直乘
                TCVT(oq, xin);                                   // fp32 -> e4m3
            } else {
                t_f xf; TCVT(xf, xin);                           // bf16/half -> fp32
                TROWEXPANDMUL(xf, xf, recip_f);                  // 逐行标量广播乘
                TCVT(oq, xf);                                    // fp32 -> e4m3
            }
            global_iterator<gm_y, t_o> y_iter(y + row0 * N + kb * BlockSize);
            auto gy = y_iter(0, 0); TSTORE(gy, oq);
        }
        // 奇尾 scale 列补 0x00 E8M0（numKb 奇数）：2D 切分下同一行 tile 的 kb 区间分属多个
        //   PE，补齐列只由持有最末 kb 区间者写（ownsLastKb），保证每行恰好写一次。
        //   构造链（两个约束叠加）：① TEXPANDS dtype 白名单无 E8M0 → 先产 bf16 再 TCVT；
        //   ② e8m0 无法表示 0（模型 ConvertFloatToE8M0：零输入→0xFF invalid），而 ADR-0101
        //   契约 pad = 2^-127（byte 0x00）→ 用 bf16 subnormal 0x0040（=2^-127）经 TCVT
        //   精确落 byte 0x00。V1/V2 存档用 TEXPANDS-e8m0 非法元组（奇 numKb 触发 emulator
        //   断言；历史用例 numKb 全偶从未暴露）。
        if constexpr (oddTail) {
            if (ownsLastKb) {
                t_row zpad_bf;
                TEXPANDS(zpad_bf, __builtin_bit_cast(__bf16, static_cast<uint16_t>(0x0040)));  // 2^-127
                t_e8b zpad;
                TCVT(zpad, zpad_bf);     // bf16 2^-127 -> e8m0 0x00
                global_iterator<gm_s, t_e8b> zs_iter(
                    reinterpret_cast<__fp8_e8m0 *>(scale) + row0 * scaleCols + numKb);
                auto gzs = zs_iter(0, 0); TSTORE(gzs, zpad);
            }
        }
    };
    // 单个 PE (编译期常量 Pe) 的驱动（V3 2D 切分，与 dyn 版同公式）：
    //   M 按 TileM=32 切 rowTiles 个行 tile，行方向铺 peRows=min(rowTiles,4) 个 PE（tile
    //   粒度均分、连续区间：前 rem 个 lane 各 +1 tile）；行方向用不满 4 PE 时剩余 PE 切到
    //   N 轴（peCols=4/peRows，仅整除时>1：rowTiles=1→1×4、=2→2×2、=3→3×1〔PE3 闲〕、
    //   >=4→4×1）。kb 区间在 colLane 间均分（前 rem 个各 +1 kb）。全部 constexpr → 满足
    //   boxed 尾块 validRow 编译期常量要求（TREDUCEPREFIXVIEW 立即数约束）。
    //   注：rowTiles>=4 且不被 4 整除时 tile 粒度有 ±1 tile 负载差（如 M=160 → 2:1:1:1），
    //   换取全部满幅链；此类形状旧均分更均衡，属已知取舍。
    auto run_pe = [&]<int Pe>() {
        constexpr int TileM = 32;  // 【M32】CUBE_M32 cell 行高固定 32
        constexpr int rowTiles = (M + TileM - 1) / TileM;
        constexpr int peRows   = rowTiles < kPeNum ? rowTiles : kPeNum;
        constexpr int peCols   = kPeNum / peRows;             // 整除：4/3→1（PE3 闲）
        constexpr int rowLane  = Pe / peCols;
        constexpr int colLane  = Pe % peCols;
        if constexpr (rowLane < peRows) {
            // 行方向：rowTiles 均分到 peRows 个 lane（前 rem 个各 +1 tile，连续 tile 区间）。
            constexpr int tilesBase   = rowTiles / peRows;
            constexpr int tilesRem    = rowTiles % peRows;
            constexpr int myTiles     = tilesBase + (rowLane < tilesRem ? 1 : 0);
            constexpr int myTileBegin =
                rowLane * tilesBase + (rowLane < tilesRem ? rowLane : tilesRem);
            // 列方向：numKb 均分到 peCols 个 lane（前 rem 个各 +1 kb，连续 kb 区间）。
            constexpr int kbBase  = numKb / peCols;
            constexpr int kbRem   = numKb % peCols;
            constexpr int myKb    = kbBase + (colLane < kbRem ? 1 : 0);
            constexpr int kbBegin = colLane * kbBase + (colLane < kbRem ? colLane : kbRem);
            constexpr bool ownsLastKb = (kbBegin + myKb == numKb);   // oddTail 补齐列归属
            if constexpr (myTiles > 0 && myKb > 0) {
                // 最后全局 tile 由拥有它的 lane 单独处理：M%32≠0 → boxed ValidRows=M%32；
                //   M%32==0 → lastValid=TileM（等价 full，仍走第二调用）。**不能**写成
                //   `if constexpr (ownLastTile && M%TileM != 0) { ...<TileM, M%TileM>... }`——
                //   clang-15(linx) 怪癖：泛型 lambda 内 if constexpr 的条件若**值依赖该
                //   lambda 模板参 Pe**（ownLastTile 经 rowLane←Pe）且为 false，丢弃分支中
                //   的 .template operator()<...> 调用仍被实例化（探针实证：连 `if constexpr
                //   (Pe > 3)` 都丢弃失败；条件仅依赖外层 M 时丢弃正常，见
                //   /tmp/opencode/probe_ifconstexpr.cpp 写法 B2/B4 vs B7/B8）。ownLastTile
                //   天然 Pe 依赖无法规避 → 唯一稳健不变式：run_pe 内**所有**模板调用点的
                //   实参对每个 Pe 实例化都必须合法（lastValid>=1 恒成立）。
                //   注：旧 V2 的 `if constexpr (seg_tail > 0)` 同受此怪癖影响（seg_tail 经
                //   SubM←Pe 依赖 Pe；M=128 时 seg_tail=0 丢弃失败会实例化 <32,0>）——只是
                //   V2 从未编译过 M%128==0 形状而未暴露。
                constexpr bool ownLastTile = (myTileBegin + myTiles == rowTiles);
                constexpr int lastValid = (M % TileM) != 0 ? (M % TileM) : TileM;
                constexpr int myFull = ownLastTile ? myTiles - 1 : myTiles;
                for (int t = 0; t < myFull; ++t) {
                    process_tile.template operator()<TileM, TileM>(
                        (myTileBegin + t) * TileM, kbBegin, myKb, ownsLastKb);
                }
                if constexpr (ownLastTile) {
                    process_tile.template operator()<TileM, lastValid>(
                        (rowTiles - 1) * TileM, kbBegin, myKb, ownsLastKb);
                }
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
