#ifndef SUPERNPU_PROBE_DYNAMIC_MX_QUANT_TAIL_OCP_FP8_HPP
#define SUPERNPU_PROBE_DYNAMIC_MX_QUANT_TAIL_OCP_FP8_HPP

#include <common/pto_tileop.hpp>
#include <cstdint>

namespace supernpu::tile_isa::mxquant {

// ===========================================================================
// MINIMAL PROBE — 穿刺验证编译器/model 能否发射并跑通「half 输入 + e4m3 输出 +
// BlockSize=32 + OCP 新公式 (Cast<bf16->e8m0> 直转)」这条尾轴指令链。
//
// 刻意最小化，NOT numerically faithful:
//   - 无 helper：TileM 推导、scratch-HBM 位重解释、scale/data 全链均内联在本函数。
//   - 只实现核心 mul + Cast<bf16->e8m0> 直转 + TRECIP 归一化；
//     跳过 inf/nan/zero/special 边界 select、奇数 numKb 的 0x00 偶对齐 pad、
//     uint16 域 recip 精确 finalize（生产 kernel 才需要）。
//   - 输入固定 __half、输出固定 __fp8_e4m3、BlockSize 默认 32。
//
// 归约走 half 值域（TABS/TROWMAX 用 FP16，落在 emulator 白名单内），直转留 bf16：
// 避开 U16-TROWMAX 被 emulator 拒绝的运行期断言（rel0812 缺陷2）。对正规数
// 「max|x| 的指数」== 「max 指数」，与原「取指数位再 max」等价。
//
// 穿刺目标发射点（反汇编核对）：
//   TABS half | TROWMAX half | TCVT half->bf16 | TANDS u16 | TCVT bf16->e8m0 (直转) |
//   TCVT half->fp32 | TRECIP fp32 | TROWEXPANDMUL | TCVT fp32->e4m3，
//   且无 32B 列对齐 / round 断言。
//
// fp8 让尾轴比 fp4 简单：e4m3 是 1 字节/元素，一个 block = BlockSize 字节，
// 天然 32B 列对齐 → 无需 fp4 tail 的 PW 物理补齐 / 列装箱。
// ===========================================================================
template <int M, int N, int BlockSize = 32>
void probe_dynamic_mx_quant_tail_ocp_fp8(__half *x, __fp8_e4m3 *y, uint8_t *scale) {
    static_assert(M > 0 && N > 0, "dim must be positive");
    static_assert(N % BlockSize == 0, "N must be multiple of BlockSize");
    static_assert(BlockSize % 32 == 0,
                  "fp8 block = BlockSize bytes; BlockSize must be a multiple of "
                  "32 so the output tile is 32B-column-aligned");

    using namespace pto;

    // --- 本地常量（bf16 域），避免依赖 common.hpp ---
    constexpr uint16_t BF16_EXP_MASK = 0x7F80; // bf16 指数位域 (取 2^E_max)
    // 2^-emax for e4m3: emax_bits(e4m3)=0x0400, bf16(1.0)=0x3f80 -> 0x3f80-0x0400.
    constexpr uint16_t RECIP_EMAX = 0x3b80;    // bf16 位型 = 2^-8

    // --- 内联 TileM 推导（OCP 绑定宽 = sizeof(half)=2，budget 8192B）---
    constexpr int kBudgetElems = 8192 / static_cast<int>(sizeof(__half)); // 4096
    constexpr int kMinElems    = 512  / static_cast<int>(sizeof(__half)); // 256 (>=512B floor)
    constexpr int kTilemMax    = kBudgetElems / BlockSize;
    constexpr int kTilemMin    = (kMinElems + BlockSize - 1) / BlockSize;
    constexpr int TileM = []{
        int t = M;
        if (t > kTilemMax) t = kTilemMax;
        if (t < kTilemMin) t = kTilemMin;
        if (t < 1) t = 1;
        return t;
    }();
    constexpr int full_m = M / TileM;
    constexpr int M_tail = M % TileM;
    constexpr int numKb  = N / BlockSize;
    // 极简：scale 紧凑平铺 [M, numKb]，无偶对齐 pad。
    constexpr int scaleCols = numKb;

    uint8_t *y_u8 = reinterpret_cast<uint8_t *>(y);

    using gm_x = global_tensor<__half,     RowMajor<M, N>>;
    using gm_y = global_tensor<uint8_t,    RowMajor<M, N>>;
    using gm_s = global_tensor<__fp8_e8m0, RowMajor<M, scaleCols>>;

    // ---- Full-tile pass (ValidRow == TileM) ----
    {
        using tile_h    = Tile<Location::Vec, __half,     TileM, BlockSize, BLayout::RowMajor>;
        using tile_hb   = Tile<Location::Vec, __half,     TileM, BlockSize, BLayout::RowMajor, TileM, 1>;
        using tile_u16b = Tile<Location::Vec, uint16_t,   TileM, BlockSize, BLayout::RowMajor, TileM, 1>;
        using tile_bfb  = Tile<Location::Vec, __bf16,     TileM, BlockSize, BLayout::RowMajor, TileM, 1>;
        using tile_e8b  = Tile<Location::Vec, __fp8_e8m0, TileM, BlockSize, BLayout::RowMajor, TileM, 1>;
        using tile_fb   = Tile<Location::Vec, float,      TileM, BlockSize, BLayout::RowMajor, TileM, 1>;
        using tile_f    = Tile<Location::Vec, float,      TileM, BlockSize, BLayout::RowMajor>;
        using tile_o    = Tile<Location::Vec, __fp8_e4m3, TileM, BlockSize, BLayout::RowMajor>;

        // scratch-HBM 位重解释缓冲（linx 无寄存器 bitcast，经字节别名往返）。
        static uint8_t buf_bu[TileM * BlockSize * sizeof(uint16_t)] __attribute__((aligned(4096)));
        static uint8_t buf_ub[TileM * BlockSize * sizeof(uint16_t)] __attribute__((aligned(4096)));

        for (int m = 0; m < full_m; ++m) {
            for (int kb = 0; kb < numKb; ++kb) {
                // === scale pass (归约走 half 值域, 直转留 bf16) ===
                global_iterator<gm_x, tile_h> x_iter(x + kb * BlockSize);
                auto gx = x_iter(m, 0);
                tile_h xh;
                TLOAD(xh, gx);
                // half 值域求 block max magnitude: FP16 的 TABS/TROWMAX 在 emulator
                // 白名单内 (AccumulateBlockInfo.cpp:229/525), 规避 U16-TROWMAX 断言
                // (rel0812 缺陷2)。对正规数 "max|x| 的指数" == "max 指数", 等价原算法。
                tile_h abs_h;
                TABS(abs_h, xh);
                tile_hb max_h;
                TROWMAX(max_h, abs_h);                // FP16 归约 -> valid col=1 (max|x|)
                tile_bfb max_bf;
                TCVT(max_bf, max_h);                  // half -> bf16 (仅每 block 的 max 标量)

                // 清尾数, 只留指数位 (2^E_max): 经 u16 boxed 域 TANDS (U16-TANDS 合法)
                // [reinterpret 验证 2026-08-17] 用 v0.58 register 级 reinterpret_tile
                // 取代下面注释掉的 scratch-HBM 位重解释往返 (bf16->u16)。零指令、
                // 无 HBM 往返;TANDS 就地清尾数,写回 max_bf 底层存储。
                auto max_u16 = reinterpret_tile<uint16_t>(max_bf);
                /* [reinterpret 验证] 原 scratch-HBM 位重解释 (bf16->u16),暂注释便于恢复:
                tile_u16b max_u16;
                {
                    using gb = global_tensor<__bf16,   RowMajor<TileM, BlockSize>>;
                    using gu = global_tensor<uint16_t, RowMajor<TileM, BlockSize>>;
                    global_iterator<gb, tile_bfb>  wi(reinterpret_cast<__bf16 *>(buf_bu));
                    global_iterator<gu, tile_u16b> ri(reinterpret_cast<uint16_t *>(buf_bu));
                    auto gw = wi(0, 0); TSTORE(gw, max_bf);
                    auto gr = ri(0, 0); TLOAD(max_u16, gr);
                }
                */
                TANDS(max_u16, max_u16, BF16_EXP_MASK);
                // [reinterpret 验证] max_u16 是 max_bf 存储的 u16 视图,上一步 TANDS 已就地
                // 清尾数;回到 bf16 域即 max_bf 本身 (同类型,无需再 reinterpret),位型即
                // 2^E_max。取引用别名保持零指令、无 HBM 往返 (TMULS 要求 dst/src 同实体类型)。
                tile_bfb &max_expbf = max_bf;
                /* [reinterpret 验证] 原 scratch-HBM 位重解释 (u16->bf16),暂注释便于恢复:
                tile_bfb max_expbf;
                {
                    using gu = global_tensor<uint16_t, RowMajor<TileM, BlockSize>>;
                    using gb = global_tensor<__bf16,   RowMajor<TileM, BlockSize>>;
                    global_iterator<gu, tile_u16b> wi(reinterpret_cast<uint16_t *>(buf_ub));
                    global_iterator<gb, tile_bfb>  ri(reinterpret_cast<__bf16 *>(buf_ub));
                    auto gw = wi(0, 0); TSTORE(gw, max_u16);
                    auto gr = ri(0, 0); TLOAD(max_expbf, gr);
                }
                */
                tile_bfb shared_bf;
                TMULS(shared_bf, max_expbf, __builtin_bit_cast(__bf16, RECIP_EMAX)); // 2^(E_max-8)
                tile_e8b scale_e8m0;
                TCVT(scale_e8m0, shared_bf);          // *** bf16 -> e8m0 直转 (穿刺核心) ***
                global_iterator<gm_s, tile_e8b> s_iter(reinterpret_cast<__fp8_e8m0 *>(scale) + kb);
                auto gs = s_iter(m, 0); TSTORE(gs, scale_e8m0);

                // === data pass (TRECIP + 广播乘) ===
                tile_fb shared_f;
                TCVT(shared_f, shared_bf);            // bf16 -> fp32
                tile_fb recip_f;
                TRECIP(recip_f, shared_f);            // 1/scale (逐 block 标量)

                tile_h xh2;
                TLOAD(xh2, gx);
                tile_f xf;
                TCVT(xf, xh2);                        // half -> fp32
                TROWEXPANDMUL(xf, xf, recip_f);       // x * (1/scale)
                tile_o oq;
                TCVT(oq, xf);                         // fp32 -> e4m3
                global_iterator<gm_y, tile_o> y_iter(y_u8 + kb * BlockSize);
                auto gy = y_iter(m, 0); TSTORE(gy, oq);
            }
        }
    }

    // ---- Tail rows: M_tail (< TileM), boxed to ValidRow = M_tail ----
    if constexpr (M_tail > 0) {
        using tile_h    = Tile<Location::Vec, __half,     TileM, BlockSize, BLayout::RowMajor, M_tail, BlockSize>;
        using tile_hb   = Tile<Location::Vec, __half,     TileM, BlockSize, BLayout::RowMajor, M_tail, 1>;
        using tile_u16b = Tile<Location::Vec, uint16_t,   TileM, BlockSize, BLayout::RowMajor, M_tail, 1>;
        using tile_bfb  = Tile<Location::Vec, __bf16,     TileM, BlockSize, BLayout::RowMajor, M_tail, 1>;
        using tile_e8b  = Tile<Location::Vec, __fp8_e8m0, TileM, BlockSize, BLayout::RowMajor, M_tail, 1>;
        using tile_fb   = Tile<Location::Vec, float,      TileM, BlockSize, BLayout::RowMajor, M_tail, 1>;
        using tile_f    = Tile<Location::Vec, float,      TileM, BlockSize, BLayout::RowMajor, M_tail, BlockSize>;
        using tile_o    = Tile<Location::Vec, __fp8_e4m3, TileM, BlockSize, BLayout::RowMajor, M_tail, BlockSize>;

        static uint8_t buf_bu[TileM * BlockSize * sizeof(uint16_t)] __attribute__((aligned(4096)));
        static uint8_t buf_ub[TileM * BlockSize * sizeof(uint16_t)] __attribute__((aligned(4096)));

        for (int kb = 0; kb < numKb; ++kb) {
            global_iterator<gm_x, tile_h> x_iter(x + kb * BlockSize);
            auto gx = x_iter(full_m, 0);
            tile_h xh;
            TLOAD(xh, gx);
            // half 值域求 block max magnitude (FP16 TABS/TROWMAX 白名单内, 规避 U16-TROWMAX)
            tile_h abs_h;
            TABS(abs_h, xh);
            tile_hb max_h;
            TROWMAX(max_h, abs_h);
            tile_bfb max_bf;
            TCVT(max_bf, max_h);

            // [reinterpret 验证] register 级 reinterpret 取代 scratch-HBM 往返 (bf16->u16)
            auto max_u16 = reinterpret_tile<uint16_t>(max_bf);
            /* [reinterpret 验证] 原 scratch-HBM 位重解释 (bf16->u16),暂注释便于恢复:
            tile_u16b max_u16;
            {
                using gb = global_tensor<__bf16,   RowMajor<TileM, BlockSize>>;
                using gu = global_tensor<uint16_t, RowMajor<TileM, BlockSize>>;
                global_iterator<gb, tile_bfb>  wi(reinterpret_cast<__bf16 *>(buf_bu));
                global_iterator<gu, tile_u16b> ri(reinterpret_cast<uint16_t *>(buf_bu));
                auto gw = wi(0, 0); TSTORE(gw, max_bf);
                auto gr = ri(0, 0); TLOAD(max_u16, gr);
            }
            */
            TANDS(max_u16, max_u16, BF16_EXP_MASK);
            // [reinterpret 验证] max_u16 是 max_bf 存储的 u16 视图,TANDS 已就地清尾数;
            // 回到 bf16 域即 max_bf 本身 (同类型,无需再 reinterpret)。取引用别名零指令。
            tile_bfb &max_expbf = max_bf;
            /* [reinterpret 验证] 原 scratch-HBM 位重解释 (u16->bf16),暂注释便于恢复:
            tile_bfb max_expbf;
            {
                using gu = global_tensor<uint16_t, RowMajor<TileM, BlockSize>>;
                using gb = global_tensor<__bf16,   RowMajor<TileM, BlockSize>>;
                global_iterator<gu, tile_u16b> wi(reinterpret_cast<uint16_t *>(buf_ub));
                global_iterator<gb, tile_bfb>  ri(reinterpret_cast<__bf16 *>(buf_ub));
                auto gw = wi(0, 0); TSTORE(gw, max_u16);
                auto gr = ri(0, 0); TLOAD(max_expbf, gr);
            }
            */
            tile_bfb shared_bf;
            TMULS(shared_bf, max_expbf, __builtin_bit_cast(__bf16, RECIP_EMAX));
            tile_e8b scale_e8m0;
            TCVT(scale_e8m0, shared_bf);
            global_iterator<gm_s, tile_e8b> s_iter(reinterpret_cast<__fp8_e8m0 *>(scale) + kb);
            auto gs = s_iter(full_m, 0); TSTORE(gs, scale_e8m0);

            tile_fb shared_f;
            TCVT(shared_f, shared_bf);
            tile_fb recip_f;
            TRECIP(recip_f, shared_f);

            tile_h xh2;
            TLOAD(xh2, gx);
            tile_f xf;
            TCVT(xf, xh2);
            TROWEXPANDMUL(xf, xf, recip_f);
            tile_o oq;
            TCVT(oq, xf);
            global_iterator<gm_y, tile_o> y_iter(y_u8 + kb * BlockSize);
            auto gy = y_iter(full_m, 0); TSTORE(gy, oq);
        }
    }
}

} // namespace supernpu::tile_isa::mxquant

#endif
