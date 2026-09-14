#include <common/pto_tileop.hpp>
#include <cstdint>
#ifdef RES_CHECK
#include "fileop.h"
#endif

// ============================================================================
// TROWMAX -> TCVT 形状契约探针（physical Cols=1 迁移）。
//
// 背景：当前 model d8903938 的 rowReduce 分支无条件把 reduce 输出 col 设为 1
//   (Block.cpp:2069 `const uint64_t stride = 1;`)。旧 kernel 把 reduce 输出及其
//   下游 TCVT 目标声明成 physical Cols=BlockSize / ValidCol=1，运行期 reduce.col=1
//   ≠ TCVT dst.col=BlockSize → ValidateOperandContract「PTO 0.58 TCVT requires
//   matching source/destination logical shapes」崩（连 full-tile 都崩）。
//
// 本探针把 reduce 输出 (max_h) 与 TCVT 目标 (max_f) 都声明成 physical Cols=1，
//   验证在当前工具链(0.58.3, 32B 列对齐 static_assert 已移除)编得过、且当前 gfrun
//   跑到底 R2=0 不再撞 TCVT 形状契约。跑通即证「physical=1 迁移」是 kernel 侧正解。
//
//   TLOAD  half [R,BS]
//   TABS   half -> half
//   TROWMAX half [R,BS] -> half [R,1]     <-- reduce 输出 physical Cols=1
//   TCVT   half [R,1] -> fp32 [R,1]        <-- 曾崩的那条；两侧 physical Cols=1
//   TSTORE fp32 [R,1] 每行 max
// ============================================================================
using namespace pto;

#ifndef R
#define R 64          // TileM (full-tile, validRow==R)
#endif
#ifndef BS
#define BS 32         // BlockSize
#endif

static __half x[R * BS] __attribute__((aligned(4096))) = {};
static float  y[R * 1]  __attribute__((aligned(4096))) = {};

int main() {
#ifdef RES_CHECK
    readBinaryFile(CHK_DIR "/input.bin", (uint8_t *)x, sizeof(x));
#endif

    // 输入：physical=valid=[R,BS]。reduce 输出/TCVT 目标：physical Cols=1、ValidCol=1。
    using t_h   = Tile<Location::Vec, __half, R, BS, BLayout::RowMajor, R, BS>;
    using t_hb  = Tile<Location::Vec, __half, R, 1,  BLayout::RowMajor, R, 1>;
    using t_fb  = Tile<Location::Vec, float,  R, 1,  BLayout::RowMajor, R, 1>;
    using t_bfb = Tile<Location::Vec, __bf16,     R, 1,  BLayout::RowMajor, R, 1>;
    using t_e8b = Tile<Location::Vec, __fp8_e8m0, R, 1,  BLayout::RowMajor, R, 1>;

    using gm_x = global_tensor<__half,      RowMajor<R, BS>>;
    using gm_y = global_tensor<__fp8_e8m0,  RowMajor<R, 1>>;

    global_iterator<gm_x, t_h>   x_iter(x);
    global_iterator<gm_y, t_e8b> y_iter(reinterpret_cast<__fp8_e8m0 *>(y));
    auto gx = x_iter(0, 0);
    auto gy = y_iter(0, 0);

    t_h xh;      TLOAD(xh, gx);
    t_h abs_h;   TABS(abs_h, xh);
    t_hb max_h;  TROWMAX(max_h, abs_h);      // reduce -> physical Cols=1
    t_fb max_f;  TCVT(max_f, max_h);         // reduce->TCVT (已验证过)
    auto max_u32 = reinterpret_tile<uint32_t>(max_f);
    TANDS(max_u32, max_u32, static_cast<uint32_t>(0x7F800000)); // 就地 scalar-logical
    t_bfb max_bf; TCVT(max_bf, max_f);       // fp32[R,1]->bf16[R,1]
    t_e8b sc;     TCVT(sc, max_bf);          // bf16[R,1]->e8m0[R,1] (1B, 疑似崩点)
    TSTORE(gy, sc);

#ifdef RES_CHECK
    writeBinaryFile(CHK_DIR "/output.bin", (uint8_t *)y, sizeof(y));
#endif
    return 0;
}
