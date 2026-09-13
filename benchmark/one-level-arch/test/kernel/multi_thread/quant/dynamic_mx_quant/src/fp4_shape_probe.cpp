#include <common/pto_tileop.hpp>
#include <cstdint>
#ifdef RES_CHECK
#include "fileop.h"
#endif

// ============================================================================
// TCVT 形状契约探针（只含一条 TCVT 计算：TLOAD -> TCVT(fp32->打包fp4) -> TSTORE）
//
// 几何对齐真实 kernel dynamic_mx_quant_tail_ocp_fp4<M=8,BlockSize=32>:
//   PW = ((BlockSize+63)/64)*64 = 64
//   源 fp32 tile physical Cols = PW = 64
//   目标 fp4 tile:
//     - 默认(打包正确)      : physical Cols = PW/2 = 32  （__fp4_e2m1x2 每元素=1字节=2个fp4）
//     - -DWIDEN(加宽骗断言)  : physical Cols = PW   = 64
//
// 两个断言层:
//   (1) pto_tile.hpp:722 32B 列对齐: fp4(bits=8) 需 Cols%32==0  -> 32、64 都过
//   (2) template_asm.hpp:115 TCVT_T TileLogicalShapeMatch: out::Cols==in::Cols
//         打包(32) vs 源(64)  -> 32!=64  编译期崩（本探针默认变体）
//         加宽(64) vs 源(64)  -> 64==64  过（-DWIDEN 变体，编译通过但数据错）
// ============================================================================
using namespace pto;

constexpr int R  = 8;
constexpr int PW = 64;
#ifdef WIDEN
constexpr int OCOL = PW;        // 64: physical Cols 与源一致 -> static_assert 过
#else
constexpr int OCOL = PW / 2;    // 32: 打包正确 -> static_assert Cols 32!=64 崩
#endif

static float   x[R * PW]   __attribute__((aligned(4096))) = {};
static uint8_t y[R * OCOL] __attribute__((aligned(4096))) = {};

int main() {
#ifdef RES_CHECK
    readBinaryFile(CHK_DIR "/input.bin", (uint8_t *)x, sizeof(x));
#endif

    // 源 fp32：physical=valid=[8,64]，避免 col-box 干扰，聚焦形状契约本身。
    using tile_f = Tile<Location::Vec, float,        R, PW,   BLayout::RowMajor, R, PW>;
    // 目标 fp4：physical=valid=[8,OCOL]。
    using tile_o = Tile<Location::Vec, __fp4_e2m1x2, R, OCOL, BLayout::RowMajor, R, OCOL>;

    using gm_x = global_tensor<float,   RowMajor<R, PW>>;
    using gm_y = global_tensor<uint8_t, RowMajor<R, OCOL>>;

    global_iterator<gm_x, tile_f> x_iter(x);
    global_iterator<gm_y, tile_o> y_iter(y);

    auto gx = x_iter(0, 0);       // 具名 lvalue（TLOAD/TSTORE 拒临时量）
    auto gy = y_iter(0, 0);

    tile_f xf;
    TLOAD(xf, gx);
    tile_o oq;
    TCVT(oq, xf);                 // 唯一计算：fp32 -> 打包 fp4
    TSTORE(gy, oq);

#ifdef RES_CHECK
    writeBinaryFile(CHK_DIR "/output.bin", (uint8_t *)y, sizeof(y));
#endif
    return 0;
}
