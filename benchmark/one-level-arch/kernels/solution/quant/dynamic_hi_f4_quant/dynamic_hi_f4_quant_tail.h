#ifndef SUPERNPU_DYNAMIC_HI_F4_QUANT_TAIL_H
#define SUPERNPU_DYNAMIC_HI_F4_QUANT_TAIL_H

#include <common/pto_tileop.hpp>
#include <cstdint>
#include <type_traits>

// ============================================================================
// dynamic_hi_f4_quant —— 尾轴 encode,V1 蓝本 + ROW_MAJOR
// ============================================================================
// 权威格式/算法见 pto-spec asl/arch/data-types/formats/{e6m2,rcpe6m2,hif4-scale,hif4x2}.asl。
// 数值验证:test/solution/quant/dynamic_hi_f4_quant/src/{gen_hif4_golden,hif4_compare,
//   hif4_res_check,run_hif4_precision}(host golden + res_check 重建 MSE,多 seed pass)。
//
// 2026-09-18 布局定案 = ROW_MAJOR(全程 Vec/RowMajor):
//   - 归约:`[32,4] → [32,1]` 行归约,输出 col=1,下游 `[32,1]` 消费 physCol=1 天然匹配
//     (mxquant 同款,过 gfrun)。SSM #685 只在归约 dst 声明成物理 `[32,2]` 时才失配——
//     本实现全程 `[32,1]`,不触发;也无需 CUBE #311 TREDUCEPREFIXVIEW。
//   - fp4/hif4 输出走 RowMajor(新 LLVM c9d40c88 gate 掉了 CUBE_M32 的 fp4 B.DATR,
//     `B.DATR CUBE_M32, {HiF4x2,e1m2x2}` Match Instruction Error;RowMajor fp4 可发射,dmxq 同款)。
//   - scale word:走 **uint16 域**(bf16→uint16 同宽,规避 narrowing→u32 的 #119 physical-row
//     契约),lo/hi 两个 uint16 拼成 U32/块,分两半 store。
//
// 结构:abs→三级相邻 max(16 组 [32,4] standalone 载 + TROWMAX + TMAX 树)→ SF=Vmax/7
//   → e6m2 量化:重构 e6m2 值到 bf16(quantize_e6m2)→ TRECIP 取倒数(问题3/4;rcpe6m2
//   单次末端舍入,替代旧 cvt+recip 两次舍入)→ L2/L3(TCMPS<GE>+TSEL)→ 带符号归一化
//   (每 4-col 组重载+TROWEXPANDMUL+TCVT→fp4+store)→ scale word(uint16 lo/hi)。
//   y≡e1m2(hif4x2 正名待 CUBE fp4 gate 放开)。
// ============================================================================

namespace supernpu::tile_isa::hif4quant {

using namespace pto;

constexpr uint16_t HIF4_INV7_B   = 0x3E12; // 1/7
constexpr uint16_t HIF4_ONE_B    = 0x3F80; // 1.0
constexpr uint16_t HIF4_HALF_B   = 0x3F00; // 0.5
constexpr uint16_t HIF4_THR_L2_B = 0x4080; // 4.0
constexpr uint16_t HIF4_THR_L3_B = 0x4000; // 2.0
inline __bf16 hif4_bf16c(uint16_t b) { return __builtin_bit_cast(__bf16, b); }

// ROW_MAJOR 列向量 [32,1] / uint16 [32,1]
using Row    = Tile<Location::Vec, __bf16,    32, 1, BLayout::RowMajor>;
using U16Row = Tile<Location::Vec, uint16_t,  32, 1, BLayout::RowMajor>;

// ============================================================================
// bf16 → e6m2 量化(问题3/4):一次舍入产出
//   (a) q_bf16 = e6m2 量化后的 **bf16 值**(尾数舍到 2 位)= E6M2FiniteValue,精确;
//   (b) e6m2_byte = 打包 e6m2 8bit 码(供 scale word 的 bits[7:0])。
// e6m2 值 =(1+m2/4)·2^(exp6−48),1+m2/4∈{1,1.25,1.5,1.75} 在 bf16 精确 → q_bf16 无
// 二次舍入;rec 由 TRECIP(q_bf16) 给出 = 1/E6M2FiniteValue(rcpe6m2 语义:单次末端舍入,
// 不再 cvt+recip 两次舍入)。尾数舍入 = round-half-up @ bit4(golden 逐位同规则)。
// bf16: [sign15][exp8=14:7][mant7=6:0];e6m2: exp6=exp8−79,m2=mant7[6:5]。
//   e6m2_byte = ((exp8·4)|m2) − 316 = ((exp8−79)<<2)|m2。
// ============================================================================
inline void quantize_e6m2(Row &sf, Row &q_bf16, U16Row &e6m2_byte) {
    q_bf16 = sf;
    auto q = reinterpret_tile<uint16_t>(q_bf16);
    TADDS(q, q, (uint16_t)0x10);       // round-half-up @ bit4
    TANDS(q, q, (uint16_t)0xFFE0);     // 截尾数到 [6:5]:q_bf16 = e6m2 量化值(bf16)
    TCVT(e6m2_byte, q);                // uint16 视图 → plain uint16(发 lb2 盖章)
    TSHRS(e6m2_byte, e6m2_byte, (uint16_t)5);
    TADDS(e6m2_byte, e6m2_byte, (uint16_t)(0x10000 - 316));
    TANDS(e6m2_byte, e6m2_byte, (uint16_t)0xFF);
}

// ============================================================================
// (t ≥ K) ? 0.5 : 1.0(bf16 因子)+ E1 位(uint16 0/1)—— TCMPS<GE> + TSEL(整数域)
// ============================================================================
// ⚠ 模型要求 TSEL 的 tuple dtype 为**整数**(IsLogicalIntegerTeplDataType)。故因子选择
//    在 uint16 **位模式**上做(0.5=0x3F00,1.0=0x3F80,选完 reinterpret 回 bf16 即正确值),
//    E1 位直接选整数 0/1。与 dmxq 同款(TSEL 走 uint16)。
template <uint16_t Kbits>
inline void ge_factor_and_bit(Row &t, Row &fac_out, U16Row &e1_out) {
    Row pred; TCMPS<pto::CmpMode::GE>(pred, t, hif4_bf16c(Kbits));
    // 因子:fac_out(bf16)位模式 = pred ? 0.5 : 1.0
    auto fo = reinterpret_tile<uint16_t>(fac_out); TEXPANDS(fo, HIF4_ONE_B);
    Row halfbf; auto ho = reinterpret_tile<uint16_t>(halfbf); TEXPANDS(ho, HIF4_HALF_B);
    TSEL(fo, pred, ho);
    // E1 位(uint16 整数):pred ? 1 : 0
    TEXPANDS(e1_out, (uint16_t)0);
    U16Row one; TEXPANDS(one, (uint16_t)1);
    TSEL(e1_out, pred, one);
}

// ============================================================================
// kernel(V1 蓝本,ROW_MAJOR,BS=64)
// ============================================================================
template <int M, int N, int BlockSize = 64, typename OutT = __fp4_e1m2x2,
          typename InT = __bf16>
void dynamic_hi_f4_quant_tail(InT *x, OutT *y, uint32_t *scale) {
    static_assert(M > 0 && N > 0, "dim must be positive");
    static_assert(BlockSize == 64, "hi_f4 BlockSize is fixed 64");
    static_assert(N % BlockSize == 0, "N must be a multiple of 64");
    static_assert(std::is_same_v<InT, __bf16> || std::is_same_v<InT, __half>,
                  "InT must be __bf16 or __half");

    constexpr int numKb = N / BlockSize;
    constexpr int HALF  = BlockSize / 2;
    constexpr int TileM = (M < 32) ? M : 32;   // 一次处理 TileM 行(≤32)
    uint8_t  *y_u8 = reinterpret_cast<uint8_t *>(y);
    uint16_t *s16  = reinterpret_cast<uint16_t *>(scale);

    // 全形状 global tensor:行 stride 由 Cols 决定(RowStride=Cols),配 global_iterator
    // 定位 (row0,offset) 才能正确跨行。否则 RowMajor<TileM,·> 行距被错设成 tile 宽 → 多行
    // 读写互相踩踏(dmxq tail_cublas 同款 idiom)。scale 存成 [M,2*numKb] uint16(每块 lo/hi
    // 两个 uint16 = [M,numKb] uint32 小端),行距 2*numKb。
    using gm_x   = global_tensor<InT,      RowMajor<M, N>>;
    using gm_y   = global_tensor<uint8_t,  RowMajor<M, N / 2>>;
    using gm_s16 = global_tensor<uint16_t, RowMajor<M, 2 * numKb>>;

    auto process_tile = [&](int row0) {
        using GrpIn = Tile<Location::Vec, InT,    TileM, 4, BLayout::RowMajor>;   // [TileM,4]
        using GrpBf = Tile<Location::Vec, __bf16, TileM, 4, BLayout::RowMajor>;
        using OutGrp= Tile<Location::Vec, OutT,   TileM, 4, BLayout::RowMajor>;
        using RowT  = Tile<Location::Vec, __bf16, TileM, 1, BLayout::RowMajor>;
        using U16T  = Tile<Location::Vec, uint16_t, TileM, 1, BLayout::RowMajor>;

      for (int kb = 0; kb < numKb; ++kb) {
        // --- 三级相邻 max:16 组 [TileM,4] → TROWMAX [TileM,1];TMAX 树 ---
        RowT m16[16];
#pragma clang loop unroll(full)
        for (int g = 0; g < 16; ++g) {
            global_iterator<gm_x, GrpIn> it(x + row0 * N + kb * BlockSize + g * 4);
            auto gg = it(0, 0);
            GrpIn xg; TLOAD(xg, gg);
            GrpBf ag;
            if constexpr (std::is_same_v<InT, __bf16>) { TABS(ag, xg); }
            else { GrpBf t; TCVT(t, xg); TABS(ag, t); }
            TROWMAX(m16[g], ag);
        }
        RowT m8[8];
#pragma clang loop unroll(full)
        for (int g = 0; g < 8; ++g) TMAX(m8[g], m16[2 * g], m16[2 * g + 1]);
        RowT m4[4];
#pragma clang loop unroll(full)
        for (int g = 0; g < 4; ++g) TMAX(m4[g], m8[2 * g], m8[2 * g + 1]);
        RowT mA, mB, vmax;
        TMAX(mA, m4[0], m4[1]); TMAX(mB, m4[2], m4[3]); TMAX(vmax, mA, mB);

        // --- base scale + 倒数(rcpe6m2:重构 e6m2 值到 bf16 再 TRECIP,单次舍入)---
        RowT sf; TMULS(sf, vmax, hif4_bf16c(HIF4_INV7_B));
        RowT q_bf16; U16T e6m2_byte; quantize_e6m2(sf, q_bf16, e6m2_byte);
        RowT rec;  TRECIP(rec, q_bf16);

        // --- L2 / L3 ---
        RowT f8[8]; U16T e1_8[8];
#pragma clang loop unroll(full)
        for (int g = 0; g < 8; ++g) {
            RowT t8; TMUL(t8, m8[g], rec);
            ge_factor_and_bit<HIF4_THR_L2_B>(t8, f8[g], e1_8[g]);
        }
        RowT f16[16]; U16T e1_16[16];
#pragma clang loop unroll(full)
        for (int g = 0; g < 16; ++g) {
            RowT t16; TMUL(t16, m16[g], rec); TMUL(t16, t16, f8[g / 2]);
            ge_factor_and_bit<HIF4_THR_L3_B>(t16, f16[g], e1_16[g]);
        }

        // --- 归一化 + 输出(每 4-col 组:重载 signed + 广播乘 + TCVT fp4 + store)---
#pragma clang loop unroll(full)
        for (int g = 0; g < 16; ++g) {
            RowT sg; TMUL(sg, rec, f8[g / 2]); TMUL(sg, sg, f16[g]);
            global_iterator<gm_x, GrpIn> it(x + row0 * N + kb * BlockSize + g * 4);
            auto gg = it(0, 0);
            GrpIn xg; TLOAD(xg, gg);
            GrpBf xbf;
            if constexpr (std::is_same_v<InT, __bf16>) { xbf = xg; }
            else { TCVT(xbf, xg); }
            GrpBf zg; TROWEXPANDMUL(zg, xbf, sg);
            OutGrp oq; TCVT(oq, zg);
            global_iterator<gm_y, OutGrp> oit(y_u8 + row0 * (N / 2) + kb * HALF + g * 2);
            auto gy = oit(0, 0);
            TSTORE(gy, oq);
        }

        // --- scale word(uint16 域,规避 #119):lo = e6m2_byte | E1_8<<8;hi = E1_16 16 位 ---
        // e6m2_byte 已是 bits[7:0],直接把 E1_8 位 OR 进去(就地成 lo)。
        U16T &lo = e6m2_byte;
#pragma clang loop unroll(full)
        for (int g = 0; g < 8; ++g) {
            U16T b; TSHLS(b, e1_8[g], (uint16_t)(8 + g)); TOR(lo, lo, b);
        }
        U16T hi; TEXPANDS(hi, (uint16_t)0);
#pragma clang loop unroll(full)
        for (int g = 0; g < 16; ++g) {
            U16T b; TSHLS(b, e1_16[g], (uint16_t)g); TOR(hi, hi, b);
        }
        // 存两个 uint16 到 scale([M,2*numKb] uint16,行距 2*numKb):块 (row,kb) 的
        // lo→列 2*kb、hi→列 2*kb+1,合成 U32/块 = lo | hi<<16(小端)。
        global_iterator<gm_s16, U16T> lo_it(s16 + row0 * (2 * numKb) + 2 * kb);
        global_iterator<gm_s16, U16T> hi_it(s16 + row0 * (2 * numKb) + 2 * kb + 1);
        auto gslo = lo_it(0, 0); auto gshi = hi_it(0, 0);
        TSTORE(gslo, lo); TSTORE(gshi, hi);
      }
    };

    for (int r = 0; r < M; r += TileM) process_tile(r);
}

} // namespace supernpu::tile_isa::hif4quant

#endif // SUPERNPU_DYNAMIC_HI_F4_QUANT_TAIL_H
