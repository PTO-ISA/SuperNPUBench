#ifndef SUPERNPU_PROBE_DYNAMIC_MX_QUANT_TAIL_OCP_FP8_NEWCALC_MT_HPP
#define SUPERNPU_PROBE_DYNAMIC_MX_QUANT_TAIL_OCP_FP8_NEWCALC_MT_HPP

#include <common/pto_tileop.hpp>
#include <cstdint>

namespace supernpu::tile_isa::mxquant {

// ===========================================================================
// MULTI-THREAD (SPMD 4-PE) 变体 —— 逐元素算法与 probe_dynamic_mx_quant_tail_ocp_fp8_newcalc
// 完全一致 (half in / e4m3 out / e8m0 scale / BlockSize=32 / 位补求倒数)，唯一区别：
// 把外层 M-tile 循环按 get_thread_idx() 切成 4 份，���个 PE-线程只算自己那 1/4 的 M 行。
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
// 约束：BS=32/half·fp32 (守住 TROWMAX/TROWEXPANDMUL 私有通路两道门)；M%TileM==0 (无 tail)；
//   full_m%4==0 (4 线程均分，无余数)。见对应 PERF/plan 文档。
// ===========================================================================
template <int M, int N, int BlockSize = 32>
void probe_dynamic_mx_quant_tail_ocp_fp8_newcalc_mt(__half *x, __fp8_e4m3 *y, uint8_t *scale) {
    static_assert(M > 0 && N > 0, "dim must be positive");
    static_assert(N % BlockSize == 0, "N must be multiple of BlockSize");
    static_assert(BlockSize % 32 == 0,
                  "fp8 block = BlockSize bytes; BlockSize must be a multiple of "
                  "32 so the output tile is 32B-column-aligned");

    using namespace pto;

    constexpr uint32_t FP32_EXP_MASK = 0x7F800000u; // fp32 指数位域 (清尾数+符号 = floor 到 2^E, 无进位)
    constexpr uint16_t RECIP_EMAX    = 0x3b80; // bf16 位型 = 2^-8 (emax_dst=8, e4m3)
    // 倒数位补常量 (同单线程版): recip_bits = 0x7F00 - shared_bits, 分两步 TXORS(0xFFFF)+TSUBS(0x80FF)。
    constexpr uint16_t RECIP_XOR_NOT   = 0xFFFF; // 按位取反 (int16: -1)
    constexpr uint16_t RECIP_COMPL_SUB = 0x80FF; // 0xFFFF - 0x7F00 (int16 补码减法, 低16位不变)

    // 按最大中间量 (data pass 的 fp32 tile_f) 预算, 而非 half (见单线程版注释)。
    constexpr int kBudgetElems = 8192 / static_cast<int>(sizeof(float));  // 2048 (fp32-budgeted)
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
    constexpr int numKb  = N / BlockSize;
    constexpr int scaleCols = ((numKb + 1) / 2) * 2;

    // === SPMD 线程切分 ===
    constexpr int kPeNum = 4;  // SoftCore.h kCorePeCount，multiThreadNum 仅 1|4 合法
    static_assert(M % TileM == 0,
                  "MT variant requires no M_tail (M must be a multiple of TileM)");
    static_assert(full_m % kPeNum == 0,
                  "MT variant requires full_m divisible by 4 (even split across PEs)");
    const uint32_t tid = get_thread_idx();          // 0..3
    constexpr int m_per_t = full_m / kPeNum;         // 每线程 M-tile 数
    const int m_begin = static_cast<int>(tid) * m_per_t;
    const int m_end   = m_begin + m_per_t;

    uint8_t *y_u8 = reinterpret_cast<uint8_t *>(y);

    using gm_x = global_tensor<__half,     RowMajor<M, N>>;
    using gm_y = global_tensor<uint8_t,    RowMajor<M, N>>;
    using gm_s = global_tensor<__fp8_e8m0, RowMajor<M, scaleCols>>;

    using tile_h    = Tile<Location::Vec, __half,     TileM, BlockSize, BLayout::RowMajor>;
    using tile_hb   = Tile<Location::Vec, __half,     TileM, BlockSize, BLayout::RowMajor, TileM, 1>;
    using tile_bfb  = Tile<Location::Vec, __bf16,     TileM, BlockSize, BLayout::RowMajor, TileM, 1>;
    using tile_e8b  = Tile<Location::Vec, __fp8_e8m0, TileM, BlockSize, BLayout::RowMajor, TileM, 1>;
    using tile_fb   = Tile<Location::Vec, float,      TileM, BlockSize, BLayout::RowMajor, TileM, 1>;
    using tile_f    = Tile<Location::Vec, float,      TileM, BlockSize, BLayout::RowMajor>;
    using tile_o    = Tile<Location::Vec, __fp8_e4m3, TileM, BlockSize, BLayout::RowMajor>;

    // 每线程只遍历自己的 M-tile 区间 [m_begin, m_end)；写不重叠的 M 行。
    for (int m = m_begin; m < m_end; ++m) {
        for (int kb = 0; kb < numKb; ++kb) {
            // === scale pass (与单线程版逐字一致) ===
            global_iterator<gm_x, tile_h> x_iter(x + kb * BlockSize);
            auto gx = x_iter(m, 0);
            tile_h xh;
            TLOAD(xh, gx);
            tile_h abs_h;
            TABS(abs_h, xh);
            tile_hb max_h;
            TROWMAX(max_h, abs_h);
            tile_fb max_f;
            TCVT(max_f, max_h);                       // half -> fp32（精确无舍入）
            auto max_u32 = reinterpret_tile<uint32_t>(max_f);
            TANDS(max_u32, max_u32, FP32_EXP_MASK);   // floor 到 2^E_max（清尾数+符号，无进位）
            tile_bfb max_bf;
            TCVT(max_bf, max_f);                       // fp32 -> bf16（尾数=0，转换精确）
            tile_bfb &max_expbf = max_bf;
            tile_bfb shared_bf;
            TMULS(shared_bf, max_expbf, __builtin_bit_cast(__bf16, RECIP_EMAX)); // 2^(E_max-8)
            tile_e8b scale_e8m0;
            TCVT(scale_e8m0, shared_bf);          // bf16 -> e8m0 直转
            global_iterator<gm_s, tile_e8b> s_iter(reinterpret_cast<__fp8_e8m0 *>(scale) + kb);
            auto gs = s_iter(m, 0); TSTORE(gs, scale_e8m0);

            // === data pass (NEWCALC: 位补求倒数, 复用 scale pass 的 xh) ===
            auto sh_u16 = reinterpret_tile<int16_t>(shared_bf);
            TXORS(sh_u16, sh_u16, RECIP_XOR_NOT);          // 0xFFFF - bits (并重打 U16 标签)
            TSUBS(sh_u16, sh_u16, RECIP_COMPL_SUB);        // -> 0x7F00 - bits = 2^(8-E_max) bf16
            tile_bfb &recip_bf = shared_bf;
            tile_fb recip_f;
            TCVT(recip_f, recip_bf);              // bf16 -> fp32

            tile_f xf;
            TCVT(xf, xh);                         // half -> fp32 (复用 xh, 免第二次 TLOAD)
            TROWEXPANDMUL(xf, xf, recip_f);       // x * (1/scale)
            tile_o oq;
            TCVT(oq, xf);                         // fp32 -> e4m3
            global_iterator<gm_y, tile_o> y_iter(y_u8 + kb * BlockSize);
            auto gy = y_iter(m, 0); TSTORE(gy, oq);
        }
    }
}

} // namespace supernpu::tile_isa::mxquant

#endif
