#include <common/pto_tileop.hpp>
#include <cstdint>
#ifdef RES_CHECK
#include "fileop.h"
#endif

// ============================================================================
// TABS(bf16) / TROWMAX(uint16) dtype 白名单探针 —— 只含 TLOAD-TABS-TROWMAX-TSTORE。
//
// 验证 RECORD 记录的两个旧缺陷是否已在当前 emulator 基线解除：
//   (1) TABS 是否支持 bf16   —— 旧白名单只含 FP16/FP32，拒 BF16。
//   (2) TROWMAX 是否支持 uint16 —— 旧白名单只含 FP16/FP32/INT32，拒 UINT16/BF16。
//
// 单链 dtype 过渡靠零指令 reinterpret_tile（不算一条 op），一次覆盖两点，
// 也正是 dynamic_mx_quant 旧「指数位域归约」的真实用法（值域 TABS + uint16 视图 TROWMAX）：
//   TLOAD  bf16 xin
//   TABS   bf16 -> bf16 abs           <-- 验证 TABS 支持 bf16
//   reinterpret_tile<uint16_t>(abs)   <-- 零指令视图，op 域切到 uint16
//   TROWMAX uint16 -> uint16 max(每行) <-- 验证 TROWMAX 支持 uint16
//   TSTORE uint16 每行 max
//
// 追加尾段验证问题17（TCMPS UINT32）：TCVT bf16->fp32、reinterpret_tile<uint32_t> 视图、
//   TCMPS(raw, 0x7F800000, LT) 在 uint32 域比较、TSTORE 掩码。旧 compare/select 白名单
//   IsCompareSelectTeplDataType 的 TCMPS 分支缺 UINT32，gfrun 跑通即证已含（commit 50afe316）。
//
// gfrun 若跑到底不 assert，即证两白名单均已放宽（此 emulator 单一 dtype 门
// AccumulateBlockInfo.cpp:669 ASSERT(IsReduceAndExpandTeplDataType/BasicUnary...)）。
// ============================================================================
using namespace pto;

constexpr int R = 8;
constexpr int C = 32;   // bf16(2B): R*C=256 元素、512B ≥ spill 下限；Cols%16==0 满足 32B 列对齐

static __bf16   x[R * C] __attribute__((aligned(4096))) = {};
static uint16_t y[R * 1] __attribute__((aligned(4096))) = {};
static uint32_t z[R * C] __attribute__((aligned(4096))) = {};   // 问题17：TCMPS UINT32 掩码输出

int main() {
#ifdef RES_CHECK
    readBinaryFile(CHK_DIR "/input.bin", (uint8_t *)x, sizeof(x));
#endif

    // 源 bf16：physical=valid=[R,C]。归约结果 uint16：physical [R,C]、valid [R,1]。
    using tile_bf  = Tile<Location::Vec, __bf16,   R, C, BLayout::RowMajor, R, C>;
    using tile_max = Tile<Location::Vec, uint16_t, R, C, BLayout::RowMajor, R, 1>;

    using gm_x = global_tensor<__bf16,   RowMajor<R, C>>;
    using gm_y = global_tensor<uint16_t, RowMajor<R, 1>>;

    global_iterator<gm_x, tile_bf>  x_iter(x);
    global_iterator<gm_y, tile_max> y_iter(y);

    auto gx = x_iter(0, 0);   // 具名 lvalue（TLOAD/TSTORE 拒临时量）
    auto gy = y_iter(0, 0);

    tile_bf xin;
    TLOAD(xin, gx);

    tile_bf absb;
    TABS(absb, xin);                                 // <-- TABS bf16

    auto abs_u16 = reinterpret_tile<uint16_t>(absb); // 零指令视图：bf16 -> uint16

    tile_max maxu;
    TROWMAX(maxu, abs_u16);                           // <-- TROWMAX uint16（每行 max）

    TSTORE(gy, maxu);

    // ---- 问题17 验证：TCMPS 作用于 UINT32（旧 compare/select 白名单缺 UINT32）----
    // fp32 位型上做整型比较（抽指数位），源经零指令 reinterpret_tile<uint32_t> 视图。
    // gfrun 若不 assert 即证白名单已含 UINT32（emulator commit 50afe316）。
    using tile_f32 = Tile<Location::Vec, float,    R, C, BLayout::RowMajor, R, C>;
    using tile_u32 = Tile<Location::Vec, uint32_t, R, C, BLayout::RowMajor, R, C>;
    tile_f32 xf;
    TCVT(xf, xin);                                     // bf16 -> fp32
    auto raw = reinterpret_tile<uint32_t>(xf);         // 零指令视图 fp32 -> uint32
    tile_u32 finite;
    TCMPS(finite, raw, static_cast<uint32_t>(0x7F800000));  // <-- TCMPS UINT32（3参，默认模式）

    using gm_z = global_tensor<uint32_t, RowMajor<R, C>>;
    global_iterator<gm_z, tile_u32> z_iter(z);
    auto gz = z_iter(0, 0);
    TSTORE(gz, finite);                                // 落地防 DCE

#ifdef RES_CHECK
    writeBinaryFile(CHK_DIR "/output.bin", (uint8_t *)y, sizeof(y));
#endif
    return 0;
}
