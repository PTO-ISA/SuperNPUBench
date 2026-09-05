#include <common/pto_tileop.hpp>
#include <cstdint>
#ifdef RES_CHECK
#include "fileop.h"
#endif

// ============================================================================
// 就地 TSEL 建模契约探针（只含一条 TSEL 计算）—— 问题18 最简复现。
//
// TSEL(dst, mask, trueSrc) 被工具链 (template_asm.hpp TSEL, TEPL 26) lower 成
// 一条 B.IOT，两个 tile 源 (mask=src0, trueSrc=src1)，->dst 兼隐式就地 false-source：
//     dst = mask ? trueSrc : dst_prior
// 而 emulator 把 TSEL 建模为「显式三源 / 两拍 B.IOT」，于是:
//   (a) validate 侧 ValidateCompareSelectTepl 首拍要求 inst->dsts.empty()，
//       而工具链的单拍带 dst(->%0) → 命中断言；
//   (b) execute 侧 ExecuteTSEL 无条件读 block->srcTile[2]，而工具链只有 2 个源
//       → 越界。
// 单条 TSEL 即可触发，与 dynamic_mx_quant 的 finalize_recip_u16 无关。
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
    TEXPANDS(x, static_cast<uint16_t>(0xABCD));   // 就地 false-source 种子
    tile_u16 mask;
    TEXPANDS(mask, static_cast<uint16_t>(1));      // 全真谓词
    tile_u16 k;
    TEXPANDS(k, static_cast<uint16_t>(0x1234));    // true-source 常量

    TSEL(x, mask, k);   // 就地: x = mask ? k : x_prior   <-- 问题18

    TSTORE(gy, x);

#ifdef RES_CHECK
    writeBinaryFile(CHK_DIR "/output.bin", (uint8_t *)y, sizeof(y));
#endif
    return 0;
}
