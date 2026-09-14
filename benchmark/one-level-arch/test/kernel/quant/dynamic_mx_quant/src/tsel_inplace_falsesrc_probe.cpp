#include <common/pto_tileop.hpp>
#include <cstdint>
#ifdef RES_CHECK
#include "fileop.h"
#endif

// ============================================================================
// 就地 TSEL 的 false-source(prior-dst)探针 —— 问题18 更深一层的最简复现。
//
// TSEL(dst, mask, trueSrc) 就地语义: dst = mask ? trueSrc : dst_prior。
// 当 mask 全 0 时，结果应 == dst 的**先前值**（一个 no-op）。但当前 LinxV5 后端的
// 就地 lowering 没把 dst 的先前值绑定进 TSEL 的 dst 寄存器 → emulator 侧
// SoftCore::ExecuteTSEL 读 dstTile[0] 作 false-source 时读到 0 → dst 被清零。
//
// 与已有 tsel_inplace_probe（mask=1，测 dst=trueSrc）互补：那个只走 true 分支、
// 不触碰 prior-dst，故崩溃修复(ab822e7a+1f398190)后即通过；本探针 mask=0 专测
// prior-dst 读取，暴露"读到 0"这一层。
//
//   x     = 0x1234  (prior dst value)
//   mask  = 0       (全假谓词)
//   k     = 0xABCD  (true-source，不应被选中)
//   TSEL(x, mask, k)  -> 期望每元素 x == 0x1234；缺陷时 x == 0x0000。
//
// 单条 TSEL 即触发，与 dynamic_mx_quant 的 finalize 无关。
// ============================================================================
using namespace pto;

constexpr int R = 8;
constexpr int C = 16;   // uint16(2B): R*C=128 元素≥64；Cols%16==0 满足 32B 列对齐

static uint16_t y[R * C] __attribute__((aligned(4096))) = {};

int main() {
    using tile_u16 = Tile<Location::Vec, uint16_t, R, C, BLayout::RowMajor, R, C>;
    using gm_y = global_tensor<uint16_t, RowMajor<R, C>>;

    global_iterator<gm_y, tile_u16> y_iter(y);
    auto gy = y_iter(0, 0);   // 具名 lvalue（TSTORE 拒临时量）

    tile_u16 x;
    TEXPANDS(x, static_cast<uint16_t>(0x1234));    // prior dst value（false-source）
    tile_u16 mask;
    TEXPANDS(mask, static_cast<uint16_t>(0));       // 全假谓词
    tile_u16 k;
    TEXPANDS(k, static_cast<uint16_t>(0xABCD));     // true-source（不应被选中）

    TSEL(x, mask, k);   // 就地: x = mask ? k : x_prior; mask=0 -> x 应保持 0x1234

    TSTORE(gy, x);

#ifdef RES_CHECK
    writeBinaryFile(CHK_DIR "/output.bin", (uint8_t *)y, sizeof(y));
#endif
    return 0;
}
