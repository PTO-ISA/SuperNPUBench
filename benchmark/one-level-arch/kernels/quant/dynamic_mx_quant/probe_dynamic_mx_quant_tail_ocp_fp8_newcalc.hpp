#ifndef SUPERNPU_PROBE_DYNAMIC_MX_QUANT_TAIL_OCP_FP8_NEWCALC_HPP
#define SUPERNPU_PROBE_DYNAMIC_MX_QUANT_TAIL_OCP_FP8_NEWCALC_HPP

#include <common/pto_tileop.hpp>
#include <cstdint>

namespace supernpu::tile_isa::mxquant {

// ===========================================================================
// MINIMAL PROBE (NEWCALC) — 与 probe_dynamic_mx_quant_tail_ocp_fp8.hpp 接口、入参、
// 输出完全一致 (half in / e4m3 out / e8m0 scale / BlockSize=32)，只把 data pass 的
// 倒数求法从 "TCVT bf16->fp32 + TRECIP" 换成 "位补 (bit-complement) 求倒数"。
//
// 新流程动机 (对齐用户公式 1/mx_scale = reinterpret<bf16>(complement(shared_exp))):
//   OCP shared scale 恒为 2 的幂 (清尾数后 mant=0)，故其倒数亦为 2 的幂，
//   在浮点位型域可由 "指数字段取补" 精确得到，无需真正的浮点除法/倒数单元。
//   bf16 位型: 2^k 的 bits = (k+127)<<7。倒数 2^(-k) 的 bits = (-k+127)<<7 =
//   (254 - (k+127))<<7 = 0x7F00 - bits。  (254<<7 = 0x7F00)
//   → recip_bits = 0x7F00 - shared_bits，用 TNOT (0xFFFF-x) + TSUBS(0x80FF) 实现：
//     0xFFFF - x - 0x80FF = 0x7F00 - x。 (纯 2 的幂输入，指数域不借位，恒 >0 不回绕)
//
// 为什么不直接照公式在 e8m0 字节域做 255-b 再 cvt<bf16>(reinterpret<e8m0>)：
//   emulator 功能模型只实现了正向 bf16->e8m0 (FloatPointUtils #253)，**没有**逆向
//   e8m0->浮点 的 elementwise TCVT (funcMap 无 SF8 源条目)，故 cvt<bf16>(e8m0) 会命中
//   "Not support such type convert yet"。因此改在 bf16 指数字段域做补 (0x7F00-bits)，
//   与 e8m0 字节域的 254-b 数值等价 (e8m0 与 bf16 共享 8-bit 指数、同 bias 127)，且全程
//   走已实现的 bf16<->fp32 与整数 TNOT/TSUBS，绕开逆向 TCVT 缺口。
//
// 注: 公式里的 255-b 是 "对 e8m0 字节 (含 bias 127) 直接取 8-bit 补"，隐含 dst emax=7；
//   本 kernel 与原 kernel 一致取 emax=8 (e4m3)，故为 254-b (= 0x7F00-bits)。若要精确复刻
//   公式的 255-b，把常量换成 0x7F80 即可 (对应 scale 用 RECIP_EMAX=2^-7)。
//
// scale pass 与原探针逐字节相同 (floor 取指数 -> 减 emax -> bf16->e8m0 直转 -> store)。
// ===========================================================================
template <int M, int N, int BlockSize = 32>
void probe_dynamic_mx_quant_tail_ocp_fp8_newcalc(__half *x, __fp8_e4m3 *y, uint8_t *scale) {
    static_assert(M > 0 && N > 0, "dim must be positive");
    static_assert(N % BlockSize == 0, "N must be multiple of BlockSize");
    static_assert(BlockSize % 32 == 0,
                  "fp8 block = BlockSize bytes; BlockSize must be a multiple of "
                  "32 so the output tile is 32B-column-aligned");

    using namespace pto;

    constexpr uint16_t BF16_EXP_MASK = 0x7F80; // bf16 指数位域 (取 2^E_max)
    constexpr uint16_t RECIP_EMAX    = 0x3b80; // bf16 位型 = 2^-8 (emax_dst=8, e4m3)
    // 倒数位补常量: recip_bits = 0x7F00 - shared_bits。分两步:
    //   (1) XOR 0xFFFF (= 按位取反 NOT) -> 0xFFFF - bits
    //   (2) 减 0x80FF                    -> 0x7F00 - bits = 2^(8-E) bf16 位型
    // 用 TXORS(scalar-logical) 而非 TNOT(basic-unary): reinterpret 视图的运行期 dtype 标签
    // 仍是产出它的 TMULS 的 BF16，emulator 的 basic-unary 校验 (AccumulateBlockInfo.cpp:425)
    // 仍是严格 dtype 相等 (BF16!=I16 会误杀)，而 scalar-logical 校验 (:463) 已放宽为位宽相等；
    // 且 TXORS 作为产出指令把该 tile 运行期标签重打成 I16，后续 TSUBS 见 I16==I16 即合法。
    // 位重解释用 int16 (非 uint16): TSUBS 的 ISA profile 仅允许 INT16/INT32/FP16/FP32
    // (TEPLEngine.cpp:2149)，U16 不在内；补码减法低 16 位与无符号一致，位型精确等价。
    constexpr uint16_t RECIP_XOR_NOT   = 0xFFFF; // 按位取反 (int16: -1)
    constexpr uint16_t RECIP_COMPL_SUB = 0x80FF; // 0xFFFF - 0x7F00 (int16 补码减法, 低16位不变)

    // 按最大中间量 (data pass 的 fp32 tile_f) 预算, 而非 half: tile_f 物理字节 =
    // TileM*BlockSize*4, 而 TSize 只编码 <=8KB 的 2 的幂, 故须 TileM*BlockSize*4<=8192。
    // 用 sizeof(__half) 预算会漏掉这翻倍, 令 TileM 撑到 128 -> tile_f=16KB -> 不可编码。
    constexpr int kBudgetElems = 8192 / static_cast<int>(sizeof(float)); // 2048 (fp32-budgeted)
    constexpr int kMinElems    = 512  / static_cast<int>(sizeof(__half)); // 256
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

        for (int m = 0; m < full_m; ++m) {
            for (int kb = 0; kb < numKb; ++kb) {
                // === scale pass (与原探针一致) ===
                global_iterator<gm_x, tile_h> x_iter(x + kb * BlockSize);
                auto gx = x_iter(m, 0);
                tile_h xh;
                TLOAD(xh, gx);
                tile_h abs_h;
                TABS(abs_h, xh);
                tile_hb max_h;
                TROWMAX(max_h, abs_h);
                tile_bfb max_bf;
                TCVT(max_bf, max_h);                  // half -> bf16 (每 block max)

                auto max_u16 = reinterpret_tile<uint16_t>(max_bf);
                TANDS(max_u16, max_u16, BF16_EXP_MASK);   // 清尾数, 只留 2^E_max
                tile_bfb &max_expbf = max_bf;
                tile_bfb shared_bf;
                TMULS(shared_bf, max_expbf, __builtin_bit_cast(__bf16, RECIP_EMAX)); // 2^(E_max-8)
                tile_e8b scale_e8m0;
                TCVT(scale_e8m0, shared_bf);          // bf16 -> e8m0 直转
                global_iterator<gm_s, tile_e8b> s_iter(reinterpret_cast<__fp8_e8m0 *>(scale) + kb);
                auto gs = s_iter(m, 0); TSTORE(gs, scale_e8m0);

                // === data pass (NEWCALC: 位补求倒数, 取代 TRECIP) ===
                // shared_bf = 2^(E_max-8) 恒为 2 的幂; 其倒数位型 = 0x7F00 - bits。
                // 就地在 u16 视图上做补 (镜像 scale pass 的 TANDS 就地写法), 回到 bf16
                // 即 shared_bf 本身 (同实体, 无需再 reinterpret)。
                auto sh_u16 = reinterpret_tile<int16_t>(shared_bf);
                TXORS(sh_u16, sh_u16, RECIP_XOR_NOT);          // 0xFFFF - bits (并重打 U16 标签)
                TSUBS(sh_u16, sh_u16, RECIP_COMPL_SUB);        // -> 0x7F00 - bits = 2^(8-E_max) bf16
                tile_bfb &recip_bf = shared_bf;
                tile_fb recip_f;
                TCVT(recip_f, recip_bf);              // bf16 -> fp32 (逆向已实现方向)

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

        for (int kb = 0; kb < numKb; ++kb) {
            global_iterator<gm_x, tile_h> x_iter(x + kb * BlockSize);
            auto gx = x_iter(full_m, 0);
            tile_h xh;
            TLOAD(xh, gx);
            tile_h abs_h;
            TABS(abs_h, xh);
            tile_hb max_h;
            TROWMAX(max_h, abs_h);
            tile_bfb max_bf;
            TCVT(max_bf, max_h);

            auto max_u16 = reinterpret_tile<uint16_t>(max_bf);
            TANDS(max_u16, max_u16, BF16_EXP_MASK);
            tile_bfb &max_expbf = max_bf;
            tile_bfb shared_bf;
            TMULS(shared_bf, max_expbf, __builtin_bit_cast(__bf16, RECIP_EMAX));
            tile_e8b scale_e8m0;
            TCVT(scale_e8m0, shared_bf);
            global_iterator<gm_s, tile_e8b> s_iter(reinterpret_cast<__fp8_e8m0 *>(scale) + kb);
            auto gs = s_iter(full_m, 0); TSTORE(gs, scale_e8m0);

            // data pass (NEWCALC: 位补求倒数, 就地补于 u16 视图)
            auto sh_u16 = reinterpret_tile<uint16_t>(shared_bf);
            TXORS(sh_u16, sh_u16, RECIP_XOR_NOT);
            TSUBS(sh_u16, sh_u16, RECIP_COMPL_SUB);
            tile_bfb &recip_bf = shared_bf;
            tile_fb recip_f;
            TCVT(recip_f, recip_bf);

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
