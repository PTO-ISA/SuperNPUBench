// =============================================================================
// multilayer_recompute_pto.hpp — 多层重算 GEMM 累加链（tile 版）
// =============================================================================
//
// 【功能】
//   反向重算阶段逐层重算前向中间激活：每层 acc += comb_mix * residual，复用 ACC 累加器，
//   末尾导出。避免前向存储所有激活。
//
// 【源端】TileKernels/tile_kernels/mhc/multilayer_recompute_kernel.py
//   源端用 T.gemm(clear_accum=False) 累加 + warp spec 软流水。
//
// 【迁移映射】
//   T.gemm(C += A*B)          → TMATMUL_ACC(acc, A, B)（ACC 既输入又输出累加）
//   T.gemm(C = A*B, 清零)    → TMATMUL(acc, A, B)（首层清零起算）
//   ACC → 普通 tile            → ACCCVT（导出，含类型转换）
//   A/B 须 boxed tile         → TileLeft / TileRight；C 须 TileAcc
//   warp spec 软流水          → PTO 一层无对应，改显式 serial 层循环 + ACC 链式复用
//   间接寻址(指针表)          → 标量核心从 ptrs[l] 取地址构造 global_tensor 视图
//
// 【约束】Mhc/TileH 须 16 的倍数（float boxed tile 对齐）。
//
// 【算法步骤】
//   首层: TLOAD(A0)+TLOAD(B0)→TMATMUL(acc, A0, B0)   （acc 清零起算）
//   每后续层 l: TLOAD(A_l)+TLOAD(B_l)→TMATMUL_ACC(acc, A_l, B_l)  （acc += A_l*B_l）
//   末尾: ACCCVT(out, acc)→TSTORE
// =============================================================================
#ifndef SUPERNPU_MULTILAYER_RECOMPUTE_PTO_HPP
#define SUPERNPU_MULTILAYER_RECOMPUTE_PTO_HPP
#include <common/pto_tileop.hpp>
#include <cstddef>
#include <cstdint>

namespace supernpu::tile_isa {

template <int Mhc, int TileH, int NumLayers>
void multilayer_recompute(float *initial_residual, float *const *comb_mix_ptrs,
                          float *const *layer_input_ptrs, float *out_residual) {
    static_assert(Mhc > 0 && TileH > 0 && NumLayers > 0, "dim must be positive");
    static_assert(Mhc % 16 == 0, "Mhc must be multiple of 16");
    static_assert(TileH % 16 == 0, "TileH must be multiple of 16");
    using namespace pto;
    using gm_r = global_tensor<float, RowMajor<Mhc, TileH>>;
    using gm_m = global_tensor<float, RowMajor<Mhc, Mhc>>;
    using tile_left = TileLeft<float, Mhc, Mhc>;        // A (Mhc×Mhc) Left box
    using tile_right = TileRight<float, Mhc, TileH>;    // B (Mhc×TileH) Right box
    using tile_acc = TileAcc<float, Mhc, TileH>;        // C ACC
    using tile_out = Tile<Location::Vec, float, Mhc, TileH, BLayout::RowMajor, Mhc, TileH>;
    using it_r = global_iterator<gm_r, tile_right>;
    using it_m = global_iterator<gm_m, tile_left>;
    using it_o = global_iterator<gm_r, tile_out>;
    it_r res_iter(initial_residual); it_o out_iter(out_residual);

    tile_acc acc;                                        // 链式累加器，整条层链复用
    // 首层：TMATMUL 清零起算 acc = A0 * B0
    {
        tile_left a; tile_right b;
        auto ga = it_m(const_cast<float*>(comb_mix_ptrs[0]))(0, 0);  // 间接寻址取本层 A
        auto gb = res_iter(0, 0);                                   // B = 初始残差
        TLOAD(a, ga); TLOAD(b, gb);
        TMATMUL(acc, a, b);                            // acc = A0 * B0（清零起算）
    }
    // 后续层：TMATMUL_ACC 累加 acc += A_l * B_l
    for (int l = 1; l < NumLayers; ++l) {
        tile_left a; tile_right b;
        auto ga = it_m(const_cast<float*>(comb_mix_ptrs[l]))(0, 0);
        auto gb = it_r(const_cast<float*>(layer_input_ptrs[l]))(0, 0);
        TLOAD(a, ga); TLOAD(b, gb);
        TMATMUL_ACC(acc, a, b);                        // acc += A_l * B_l
    }
    auto gout = out_iter(0, 0);
    tile_out o;
    ACCCVT(o, acc);                                    // ACC → 普通 tile（导出）
    TSTORE(gout, o);
}

} // namespace supernpu::tile_isa
#endif
