#ifndef SUPERNPU_DYNAMIC_HI_F4_QUANT_TAIL_H
#define SUPERNPU_DYNAMIC_HI_F4_QUANT_TAIL_H

#include <common/pto_tileop.hpp>
#include <cstdint>
#include <type_traits>

// ============================================================================
// dynamic_hi_f4_quant —— 尾轴（axis=-1）encode，路线 A（手工多遍）整体流程框架
// ============================================================================
// 权威格式/算法见 DESIGN.md §1/§2.1；缺口与规避见 RECORD.md。
//
// ⚠️ 本文件是「算法整体流程框架」，不是可运行的完整实现。按用户 2026-08-14 决策：
//    「先按照算法实现整体流程框架，绕过不可实现的部分，代码中预留位置，待后续与
//     硬件商讨适配方案。同时如实记录问题」。
//
// 框架把 §2.1 的每一步逐条铺开。步骤分两类，代码中严格区分：
//   【REAL】   —— 现有 linx one-level 指令可发射、且规避已定，直接写真实 tile-op。
//   【占位】   —— 缺口步骤（RECORD 问题3/4/10），无可发射的 one-level 路径 / 规避细节
//                待与硬件商讨。用带 `_PLACEHOLDER` 后缀的 helper 预留位置，函数体是
//                **类型合法但数值不正确**的退化 stand-in（醒目标注），编译占位、待替换。
//
// 关键布局洞察（RECORD 问题10 的正解方向）：block 级中间量**不用窄列 reshape**，
// 一律用「物理宽 = BlockSize(64) + ValidCol 收窄」声明，列宽天然满足 32B 对齐
// （bf16 Cols%16、uint8/uint32 亦合法）。于是缺口**只剩两个算子本身**：
//   (1) 分段/相邻归约（64→16→8）—— TROWMAX 只能整条 valid 行归约，无相邻分段（问题10）；
//   (2) 分组广播乘（8→64 / 16→64 / 8→16）—— TROWEXPANDMUL 只能 1→C，无多对多（问题10/2）。
// 其余（整块 Vmax、1→C 广播、compare-select、位打包、e1m2 输出 cast）均可发射。
//
// 另有两处「已定但未写」的位重构规避，本框架同样以占位 helper 预留（含已定公式注释）：
//   - bf16→e6m2 位重构（问题3，one-level TCVT→e6m2 不可发射）；
//   - e6m2 倒数 M2-LUT 位重构（问题4，无 e6m2 recip / 无反向 cast）。
//
// 验证边界：工具链↔仿真器 skew 下新编 ELF 无法稳定 gfrun/gfsim，本框架仅 op-review +
// 编译/反汇编级别，无运行期数值验证。占位步骤未替换前，数值输出不可信。
// ============================================================================

namespace supernpu::tile_isa::hif4quant {

using namespace pto;

// --- 权威常量（DESIGN §2.1/§2.2）--------------------------------------------
// 1/7：满量程 base×(最大 boost 4)×(e1m2 最大幅值 1.75)=base×7，令块内 max 落到满量程。
constexpr float HIF4_INV7   = 0.142857142857f; // 1/7
constexpr float HIF4_HALF   = 0.5f;            // 2^-1，boost 位命中时的因子
constexpr float HIF4_ONE    = 1.0f;            // 2^0，boost 位未命中时的因子
constexpr float HIF4_THR_L2 = 4.0f;            // 二级阈值（binade 边界）
constexpr float HIF4_THR_L3 = 2.0f;            // 三级阈值

// e6m2 recip 的 M2-LUT（DESIGN §3.1 recip 行 / RECORD 问题4）：
// LUT[m2] = bf16 bits of 1/(1+m2/4)，m2=0..3 → {1, 0.8, 0.6667, 0.5714}
constexpr uint16_t HIF4_RECIP_LUT[4] = {0x3F80, 0x3F4D, 0x3F2B, 0x3F12};

// ⚠️ bf16 标量必须以「原始位模式」materialize（RECORD 问题11）：LinxV5 后端对 float→bf16
//    的运行期转换（fptrunc → i16）在指令选择期触发 PromoteInteger 崩溃
//    （LegalizeIntegerTypes.cpp:57，probe bf16s.cpp 实测）。连 `static_cast<__bf16>(常量 float)`
//    也不折叠、照样崩。故所有喂给 tile-scalar op（TMULS/TMAXS/TEXPANDS…）的 bf16 立即数
//    一律走 __builtin_bit_cast(位模式)——与 dynamic_mx_quant 的 __builtin_bit_cast(__bf16,·) 同法。
constexpr uint16_t HIF4_ONE_B    = 0x3F80; // 1.0
constexpr uint16_t HIF4_HALF_B   = 0x3F00; // 0.5
// 注：bf16 0.0 不可作 tile-scalar 立即数（会落到 zero 寄存器，问题11）；需 0 tile 用 t−t 产生。
constexpr uint16_t HIF4_INV7_B   = 0x3E12; // 1/7
constexpr uint16_t HIF4_THR_L2_B = 0x4080; // 4.0（二级阈值）
constexpr uint16_t HIF4_THR_L3_B = 0x4000; // 2.0（三级阈值）
inline __bf16 hif4_bf16c(uint16_t bits) { return __builtin_bit_cast(__bf16, bits); }

// ============================================================================
// 占位 helper（缺口，待与硬件商讨） —— 详见每个函数头的 RECORD 引用
// ============================================================================

// -------- 占位 1：相邻分段 max（问题10）------------------------------------
// 语义（权威 §2.1）：把 valid=(GroupSize*OutValid) 列按**相邻 GroupSize 个一组**取 max，
// 产出 valid=OutValid 列（组内 max 写到该组的 col）。GroupSize=4：64→16（Vmax16）；
// GroupSize=2：16→8（Vmax8）。
//
// 缺口：TROWMAX 只能对整条 ValidCol 归约成 [R,1]，**无相邻分段归约**。既定的「零成本
// TRESHAPE 化整为零 [.,4]」规避在 one-level 不可声明（问题10 双墙：32B 列对齐 + TRESHAPE
// header gap，probe segreduce_probe.cpp 实测）。候选正解（待决策/待验证，问题10）：
//   (A) strided HBM load 把每 block 按 stride 拆成宽≥16 列子 tile + 逐元素 TMAX 树（保相邻语义）；
//   (B) 采纳 emulator strided（转置）分组语义（§2.4，列宽天然合法、偏离相邻 spec）。
//
// ⚠️ PLACEHOLDER 函数体：退化为「整块 max 广播回各组」（TROWMAX→TROWEXPAND），
//    **数值不正确**（丢失组间差异），仅为类型合法、让框架流程贯通。待替换为 (A)/(B)。
template <int TileM, int PW, int ValidR, int OutValid, int GroupSize,
          typename DType = __bf16>
inline void seg_max_adjacent_PLACEHOLDER(
    Tile<Location::Vec, DType, TileM, PW, BLayout::RowMajor, ValidR, GroupSize * OutValid> &src,
    Tile<Location::Vec, DType, TileM, PW, BLayout::RowMajor, ValidR, OutValid> &dst) {
    // TODO(hardware/问题10): 用 (A) strided 子 tile + TMAX 树 或 (B) emulator strided 语义
    //                        替换本退化实现，保 §2.1 相邻分组语义。
    Tile<Location::Vec, DType, TileM, PW, BLayout::RowMajor, ValidR, 1> blkmax;
    TROWMAX(blkmax, src);       // 整块 max（非相邻分段）—— 退化 stand-in
    TROWEXPAND(dst, blkmax);    // 广播回 OutValid 列
}

// -------- 占位 2：分组广播乘（问题10 / 问题2）------------------------------
// 语义（权威 §2.1）：把 valid=NumFactor 的每行因子，按**相邻 GroupSize 个元素共享**广播
// 乘到 valid=(NumFactor*GroupSize) 列的 data 上。用于 2^-E1_8(8→64)、2^-E1_16(16→64)、
// 以及 E1_16 阈值里 2^-E1_8(8→16)。
//
// 缺口：TROWEXPANDMUL 只支持 1→C（src1 为 [R,1] 逐行标量）；**无多对多分组广播**。窄列
// reshape 规避同样被问题10 双墙阻断。候选正解同占位 1 的 (A)/(B)。
//
// ⚠️ PLACEHOLDER 函数体：退化为「取因子 col0 一个标量 1→C 广播乘」（丢失其余 NumFactor-1
//    个因子），**数值不正确**，仅类型合法。待替换。
template <int TileM, int PW, int ValidR, int NumFactor, int GroupSize,
          typename DType = __bf16>
inline void group_bcast_mul_PLACEHOLDER(
    Tile<Location::Vec, DType, TileM, PW, BLayout::RowMajor, ValidR, NumFactor * GroupSize> &data_inout,
    Tile<Location::Vec, DType, TileM, PW, BLayout::RowMajor, ValidR, NumFactor> &factor) {
    // TODO(hardware/问题10): 用 (A)/(B) 实现相邻分组广播乘，替换本退化实现。
    Tile<Location::Vec, DType, TileM, PW, BLayout::RowMajor, ValidR, 1> f0;
    TROWMAX(f0, factor);        // 取一个代表标量（退化：应逐组取对应因子）
    TROWEXPANDMUL(data_inout, data_inout, f0); // 1→C 广播乘 stand-in
}

// -------- 占位 3：bf16 → e6m2 位重构（问题3，one-level TCVT→e6m2 不可发射）---
// 已定公式（DESIGN §3.1 cast 行）：指数重偏置 bf16 bias 127→e6m2 bias 48 + 2bit 尾数带
// 舍入截断，用整数/bf16 位域 TSHRS/TANDS/TADDS/TSELS 拼出 8bit e6m2 位，存裸 uint8。
// 只作用于块级 [TileM, valid1] 的 base，成本可忽略。
//
// ⚠️ PLACEHOLDER：位重构细节（舍入、次正规、饱和）待逐位对齐 emulator bfloat16_to_e6m2
//    后落地；此处仅取 bf16 高字节做占位，**数值不正确**。上游补 type_traits<__fp8_e6m2>
//    + __type_code e6m2 后可回退原生 TCVT(e6m2,bf16)。
template <int TileM, int PW, int ValidR>
inline void bf16_to_e6m2_bits_PLACEHOLDER(
    Tile<Location::Vec, __bf16,   TileM, PW, BLayout::RowMajor, ValidR, 1> &sf_bf16,
    Tile<Location::Vec, uint8_t,  TileM, PW, BLayout::RowMajor, ValidR, 1> &e6m2_byte) {
    // TODO(hardware/问题3): exp rebias 127->48 + 2bit mantissa round，逐位对齐 emulator。
    (void)sf_bf16;
    TEXPANDS(e6m2_byte, static_cast<uint8_t>(0)); // 占位：应为重构后的 e6m2 字节
}

// -------- 占位 4：e6m2 倒数 → bf16（问题4，无 e6m2 recip / 无反向 cast）------
// 已定 M2-LUT 位重构（DESIGN §3.1 recip 行）：ea 为 e6m2 uint8，exp6=(bits>>2)&0x3F(bias 48)、
// m2=bits&3；1/ea=2^-(exp6-48)·1/(1+m2/4)。bf16 位：rec_bits = LUT[m2] + ((48-exp6)<<7)。
// op 序：TSHRS/TANDS 取 exp6、TSHLS(7)+(6144-·) 得指数项、TANDS+两次 TSELS 选 LUT、TADD 合并、
// 字节别名 reinterpret u16→bf16。仅作用于块级 [TileM, valid1]。
//
// ⚠️ PLACEHOLDER：LUT 选择/指数项细节待落地；此处占位输出 bf16 1.0，**数值不正确**。
template <int TileM, int PW, int ValidR>
inline void e6m2_recip_bf16_PLACEHOLDER(
    Tile<Location::Vec, uint8_t, TileM, PW, BLayout::RowMajor, ValidR, 1> &e6m2_byte,
    Tile<Location::Vec, __bf16,  TileM, PW, BLayout::RowMajor, ValidR, 1> &rec_bf16) {
    // TODO(hardware/问题4): M2-LUT 位重构倒数，替换占位。
    (void)e6m2_byte;
    TEXPANDS(rec_bf16, hif4_bf16c(HIF4_ONE_B)); // 占位：应为 1/E6M2(bf16)
}

// ============================================================================
// REAL helper：compare(≥K) → E1 位 + 2^-E1 因子（问题5，原生 TCMPS<GE> + 双 TSEL）
// ============================================================================
// t≥K 用原生 TCMPS<CmpMode::GE>(mask,t,K)（v0.58 起可用，发射 B.DATR Zero,ge）。
// 早期无带 CmpMode 的比较时靠 TMAXS(tmp,t,K);TCMP(mask,tmp,t) 模拟 GE；现已切原生。
//   E1 位（bf16 0/1）：TEXPANDS(e,0); TSEL(e,mask,one)  → e=(t≥K)?1:0
//   因子 2^-E1     ：TEXPANDS(f,1); TSEL(f,mask,half) → f=(t≥K)?0.5:1.0
// TSEL(dst,mask,src)=mask?src:dst（in-place）。t 非负有限、无 NaN。
// K 以 bf16 位模式作**编译期模板参数**传入（Kbits）作 tile-scalar 立即数；
// 运行期 float→bf16 会崩后端（问题11），故走位模式；TCMPS 与 TMAXS 同为 tile-scalar，同法安全。
template <int TileM, int PW, int ValidR, int V, uint16_t Kbits>
inline void ge_threshold_e1_factor(
    Tile<Location::Vec, __bf16, TileM, PW, BLayout::RowMajor, ValidR, V> &t,
    Tile<Location::Vec, __bf16, TileM, PW, BLayout::RowMajor, ValidR, V> &e1_out,   // 0/1
    Tile<Location::Vec, __bf16, TileM, PW, BLayout::RowMajor, ValidR, V> &fac_out) { // 2^-E1
    Tile<Location::Vec, __bf16, TileM, PW, BLayout::RowMajor, ValidR, V> mask, one_t, half_t;
    TCMPS<pto::CmpMode::GE>(mask, t, hif4_bf16c(Kbits)); // mask = (t≥K)
    TEXPANDS(one_t,  hif4_bf16c(HIF4_ONE_B));
    TEXPANDS(half_t, hif4_bf16c(HIF4_HALF_B));
    // E1 = mask ? 1 : 0  —— 初值 0，TSEL(e1,mask,one)=mask?one:e1。
    // ⚠️ 用 t−t 产 0（t 非负有限、无 NaN）而非 TEXPANDS(bf16 0.0)：后者标量 0 被分配到硬件
    //    zero 寄存器 → `B.IOR [zero],[]` 指令匹配失败（问题11，volatile 反折叠对 bf16-0 无效）。
    TSUB(e1_out, t, t);                  // e1_out = 0（bf16）
    TSEL(e1_out, mask, one_t);          // (t≥K)?1:0
    // 2^-E1 = mask ? 0.5 : 1.0 —— 初值 1，TSEL(fac,mask,half)
    TEXPANDS(fac_out, hif4_bf16c(HIF4_ONE_B));
    TSEL(fac_out, mask, half_t);        // (t≥K)?0.5:1.0
}

// ============================================================================
// REAL helper：把 8 个 E1_8(0/1) 与 16 个 E1_16(0/1) 位打包成 32bit scale word
// ============================================================================
// word = e6m2 | (L2<<8) | (L3<<16)（小端，DESIGN §0/§2.5；相邻位序，问题8 运行期须对齐）。
// 位打包：pos = TCI(0..V-1)；pow2 = 1<<pos（TSHL）；bits = E1_uint * pow2；TROWSUM → 整数。
// E1 由 bf16 0/1 经 TCVT → uint16 得到（0/1 精确）。全 emittable。
template <int TileM, int PW, int ValidR>
inline void pack_hif4_scale_word(
    Tile<Location::Vec, uint8_t,  TileM, PW, BLayout::RowMajor, ValidR, 1>  &e6m2_byte,
    Tile<Location::Vec, __bf16,   TileM, PW, BLayout::RowMajor, ValidR, 8>  &e1_8,
    Tile<Location::Vec, __bf16,   TileM, PW, BLayout::RowMajor, ValidR, 16> &e1_16,
    Tile<Location::Vec, uint32_t, TileM, PW, BLayout::RowMajor, ValidR, 1>  &word_out) {
    // L2：8 个 1bit → 一个字节值 [0,255]
    Tile<Location::Vec, uint32_t, TileM, PW, BLayout::RowMajor, ValidR, 8> e8u, pos8, pow8, bits8;
    TCVT(e8u, e1_8);                                  // bf16 0/1 → uint32 0/1
    TCI(pos8, static_cast<uint32_t>(0));              // 0,1,2,...,7（逐列递增）
    Tile<Location::Vec, uint32_t, TileM, PW, BLayout::RowMajor, ValidR, 8> one8;
    TEXPANDS(one8, static_cast<uint32_t>(1));
    TSHL(pow8, one8, pos8);                           // 2^pos
    TMUL(bits8, e8u, pow8);
    Tile<Location::Vec, uint32_t, TileM, PW, BLayout::RowMajor, ValidR, 1> l2v;
    TROWSUM(l2v, bits8);                              // Σ E1_8[i]<<i
    // L3：16 个 1bit → 值 [0,65535]
    Tile<Location::Vec, uint32_t, TileM, PW, BLayout::RowMajor, ValidR, 16> e16u, pos16, pow16, bits16;
    TCVT(e16u, e1_16);
    TCI(pos16, static_cast<uint32_t>(0));
    Tile<Location::Vec, uint32_t, TileM, PW, BLayout::RowMajor, ValidR, 16> one16;
    TEXPANDS(one16, static_cast<uint32_t>(1));
    TSHL(pow16, one16, pos16);
    TMUL(bits16, e16u, pow16);
    Tile<Location::Vec, uint32_t, TileM, PW, BLayout::RowMajor, ValidR, 1> l3v;
    TROWSUM(l3v, bits16);
    // e6m2 字节 → uint32 低字节
    Tile<Location::Vec, uint32_t, TileM, PW, BLayout::RowMajor, ValidR, 1> e6u;
    TCVT(e6u, e6m2_byte);                             // uint8 → uint32
    // word = e6u | (l2<<8) | (l3<<16)
    TSHLS(l2v, l2v, static_cast<uint32_t>(8));
    TSHLS(l3v, l3v, static_cast<uint32_t>(16));
    TORS(word_out, e6u, static_cast<uint32_t>(0));    // word = e6u
    TOR(word_out, word_out, l2v);
    TOR(word_out, word_out, l3v);
}

// ============================================================================
// kernel：dynamic_hi_f4_quant 尾轴 encode（路线 A 框架）
// ============================================================================
// 输入 x：InT ∈ {__bf16, __half}（DESIGN §0；fp16 先 TCVT→bf16 进 bf16 域，无 fp32）。
// 输出 y：hifloat4 ≡ e1m2（__fp4_e1m2x2，2/字节）；scale：uint32 32bit 容器/块（问题9）。
// BlockSize 固定 64。尾轴 compact 平铺，每行 = numKb 个 64-elem block。
//
// 一块一 tile 迭代：所有 block 级中间量物理宽 PW=64，ValidCol 收窄（合法宽度）。
template <int M, int N, int BlockSize = 64, typename OutT = __fp4_e1m2x2,
          typename InT = __bf16>
void dynamic_hi_f4_quant_tail(InT *x, OutT *y, uint32_t *scale) {
    static_assert(M > 0 && N > 0, "dim must be positive");
    static_assert(BlockSize == 64, "hi_f4 BlockSize is fixed 64 (DESIGN §1.1)");
    static_assert(N % BlockSize == 0, "N must be multiple of BlockSize(64)");
    static_assert(std::is_same_v<InT, __bf16> || std::is_same_v<InT, __half>,
                  "InT must be __bf16 or __half (DESIGN §0)");
    static_assert(std::is_same_v<OutT, __fp4_e1m2x2>,
                  "y is hifloat4 (≡e1m2, __fp4_e1m2x2)");

    constexpr int PW    = BlockSize;         // 64：block 物理宽，天然 32B 对齐
    constexpr int HALF  = BlockSize / 2;     // 32：e1m2 打包字节宽
    constexpr int numKb = N / BlockSize;     // 每行 block 数
    // TileM：一块一 tile，contig 轴 = PW。沿用 mx_quant 预算模型的等价占位（此处简化为 8）。
    // TODO: 接入 max_tilem<M, PW, InT, false>() 统一预算（当前 hi_f4 无 common 预算头，先固定）。
    constexpr int TileM  = (M < 8) ? M : 8;
    constexpr int full_m = M / TileM;
    constexpr int M_tail = M % TileM;

    using gm_x = global_tensor<InT,      RowMajor<M, N>>;
    using gm_y = global_tensor<OutT,     RowMajor<M, N / 2>>;
    using gm_s = global_tensor<uint32_t, RowMajor<M, numKb>>;

    // 单块处理（ValidRow=VR 由 full/tail 传入），逐 op 铺开 §2.1。
    auto process_block = [&]<int VR>(int m_row, int kb) {
        using tile_v  = Tile<Location::Vec, __bf16, TileM, PW, BLayout::RowMajor, VR, BlockSize>;
        using tile_b1 = Tile<Location::Vec, __bf16, TileM, PW, BLayout::RowMajor, VR, 1>;
        using tile_b8 = Tile<Location::Vec, __bf16, TileM, PW, BLayout::RowMajor, VR, 8>;
        using tile_b16= Tile<Location::Vec, __bf16, TileM, PW, BLayout::RowMajor, VR, 16>;
        using tile_e6 = Tile<Location::Vec, uint8_t, TileM, PW, BLayout::RowMajor, VR, 1>;
        using tile_w  = Tile<Location::Vec, uint32_t, TileM, PW, BLayout::RowMajor, VR, 1>;
        using tile_o  = Tile<Location::Vec, OutT,     TileM, HALF, BLayout::RowMajor, VR, HALF>;

        // --- 载入 + 统一到 bf16 域 -------------------------------------------
        global_iterator<gm_x, Tile<Location::Vec, InT, TileM, PW, BLayout::RowMajor, VR, BlockSize>>
            x_iter(x + kb * BlockSize);
        auto gx = x_iter(m_row, 0);
        tile_v V;
        if constexpr (std::is_same_v<InT, __bf16>) {
            TLOAD(V, gx);
        } else { // __half → bf16（DESIGN §0）
            Tile<Location::Vec, __half, TileM, PW, BLayout::RowMajor, VR, BlockSize> Vh;
            TLOAD(Vh, gx);
            TCVT(V, Vh);
        }

        // --- §2.1 三级归约（相邻分组）---------------------------------------
        tile_v Vabs;
        TABS(Vabs, V);                                        // 【REAL】|V|
        tile_b16 Vmax16;
        seg_max_adjacent_PLACEHOLDER<TileM, PW, VR, 16, 4>(Vabs, Vmax16);   // 【占位问题10】64→16
        tile_b8 Vmax8;
        seg_max_adjacent_PLACEHOLDER<TileM, PW, VR, 8, 2>(Vmax16, Vmax8);   // 【占位问题10】16→8
        tile_b1 Vmax;
        TROWMAX(Vmax, Vmax8);                                 // 【REAL】整块 max → 1

        // --- §2.1 一级 base scale：SF=Vmax/7 → E6M2 → rec -------------------
        tile_b1 SF;
        TMULS(SF, Vmax, hif4_bf16c(HIF4_INV7_B));             // 【REAL】SF_BF16
        tile_e6 e6m2;
        bf16_to_e6m2_bits_PLACEHOLDER<TileM, PW, VR>(SF, e6m2); // 【占位问题3】L1 base 字节
        tile_b1 rec;
        e6m2_recip_bf16_PLACEHOLDER<TileM, PW, VR>(e6m2, rec);  // 【占位问题4】rec=1/E6M2

        // --- §2.1 二级 E1_8 =(Vmax8*rec ≥4)?1:0 -----------------------------
        // TROWEXPANDMUL(dst,src0,src1[R,1]) = src0 逐行×标量 → t8 = Vmax8 * rec（rec 1→8）。
        tile_b8 t8, e1_8, f8;
        TROWEXPANDMUL(t8, Vmax8, rec);                        // 【REAL】t8 = Vmax8 * rec
        ge_threshold_e1_factor<TileM, PW, VR, 8, HIF4_THR_L2_B>(t8, e1_8, f8); // 【REAL】E1_8, 2^-E1_8

        // --- §2.1 三级 E1_16=(Vmax16*rec*2^-E1_8 ≥2)?1:0 --------------------
        tile_b16 t16, e1_16, f16;
        TROWEXPANDMUL(t16, Vmax16, rec);                      // 【REAL】Vmax16*rec（rec 1→16）
        group_bcast_mul_PLACEHOLDER<TileM, PW, VR, 8, 2>(t16, f8); // 【占位问题10】×2^-E1_8（8→16）
        ge_threshold_e1_factor<TileM, PW, VR, 16, HIF4_THR_L3_B>(t16, e1_16, f16); // 【REAL】E1_16, 2^-E1_16

        // --- §2.1 逐元素归一化 Vin=V*rec*2^-E1_8*2^-E1_16（保符号）----------
        tile_v Vin;
        TROWEXPANDMUL(Vin, V, rec);                           // 【REAL】V*rec（rec 1→64，带符号 V）
        group_bcast_mul_PLACEHOLDER<TileM, PW, VR, 8, 8>(Vin, f8);   // 【占位问题10】×2^-E1_8（8→64）
        group_bcast_mul_PLACEHOLDER<TileM, PW, VR, 16, 4>(Vin, f16); // 【占位问题10】×2^-E1_16（16→64）

        // --- §2.1 输出 y=cast_to_E1M2(Vin) ----------------------------------
        tile_o oq;
        TCVT(oq, Vin);                                        // 【REAL】bf16 → e1m2（≡hifloat4，含符号）
        global_iterator<gm_y, tile_o> y_iter(y + kb * HALF);
        auto gy = y_iter(m_row, 0);
        TSTORE(gy, oq);

        // --- scale 打包 32bit 容器（问题9）+ 存 -----------------------------
        tile_w word;
        pack_hif4_scale_word<TileM, PW, VR>(e6m2, e1_8, e1_16, word); // 【REAL】word=e6m2|(L2<<8)|(L3<<16)
        global_iterator<gm_s, tile_w> s_iter(scale + kb);
        auto gs = s_iter(m_row, 0);
        TSTORE(gs, word);
    };

    // 主 pass（满行）+ 尾 pass（M_tail 行，ValidRow 收窄）
    for (int m = 0; m < full_m; ++m)
        for (int kb = 0; kb < numKb; ++kb)
            process_block.template operator()<TileM>(m, kb);
    if constexpr (M_tail > 0)
        for (int kb = 0; kb < numKb; ++kb)
            process_block.template operator()<M_tail>(full_m, kb);
}

} // namespace supernpu::tile_isa::hif4quant

#endif // SUPERNPU_DYNAMIC_HI_F4_QUANT_TAIL_H
