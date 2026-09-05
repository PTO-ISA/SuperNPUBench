#include "guard_common.hpp"
#include "guard_io.h"
#include <utility>
// TileOP-API doc guard: TileArray region API — TileArray / TASSEMBLY (+ TCVT
// slot producer). Source: docs/tileop-usage/range-modifiers.md "TileArray region
// API" (NEW on remote branch linx, commit fa24eae).
// NOTE(doc-gap): 文档示例不可直接落地:
//   (1) `TCVT(destinations[0][2], source_tile)` 用临时量,但 region TCVT 签名是
//       `TCVT(TileArrayOutputRef& dst, In& src)`(非 const 左值引用),须先绑具名左值;
//   (2) 示例源变量名 source(=TPARTVIEW 视图)与 TCVT 用的 source_tile 不一致;实测
//       用 TPARTVIEW 的父 strided 子视图作源 → gfrun `raw tile spill source does not
//       fit the carrier shape`。正解:源须为独立紧凑 Fragment Tile。
//   (3) 即便改用独立紧凑 Fragment 作源,gfrun 仍断言 `RawTileSourceFits(source,
//       shape) && "raw tile spill source does not fit the carrier shape"`——region
//       producer 写入 carrier slot 的形状契约与紧凑 Fragment 不符。结合文档自身
//       "until the ... path is implemented and validated" 告警,且该特性是远程
//       linx 顶端 commit,判定:**新 API 在当前 env_test 模型上尚不可运行(run-fail)**。
// 本 demo 保留以看护该新接口:编译需绕过文档示例的左值绑定缺口,运行期受阻于模型
// 未就绪。待模型侧补齐后再验证组装布局。
constexpr int PM = 32, PN = 64, FN = 16, NF = PN / FN;   // 4 fragments of 32x16
static float gin[PM * PN], gout[PM * PN];                // gin = 4 contiguous 32x16 blocks
int main() {
    guard_read_bin(CHK_DIR "/in_a.bin", gin, sizeof(gin));
    using Parent   = Tile<Location::Vec, float, PM, PN, BLayout::RowMajor>;
    using Fragment = Tile<Location::Vec, float, PM, FN, BLayout::RowMajor>;

    region::TileArray<Fragment, 1, NF> dst;
    Fragment src0, src1, src2, src3;
    iter_t<float, PM, FN> g0(gin + 0 * PM * FN);
    iter_t<float, PM, FN> g1(gin + 1 * PM * FN);
    iter_t<float, PM, FN> g2(gin + 2 * PM * FN);
    iter_t<float, PM, FN> g3(gin + 3 * PM * FN);
    auto g0v = g0(0, 0); auto g1v = g1(0, 0);
    auto g2v = g2(0, 0); auto g3v = g3(0, 0);
    TLOAD(src0, g0v); TLOAD(src1, g1v);
    TLOAD(src2, g2v); TLOAD(src3, g3v);

    auto d0 = dst[0][0]; auto d1 = dst[0][1];
    auto d2 = dst[0][2]; auto d3 = dst[0][3];
    BENCHSTART;
    TCVT(d0, src0); TCVT(d1, src1);
    TCVT(d2, src2); TCVT(d3, src3);
    Parent result = TASSEMBLY<Parent>(std::move(dst));
    BENCHEND;

    iter_t<float, PM, PN> go(gout);
    auto go0 = go(0, 0);
    TSTORE(go0, result);
    guard_dump_bin(CHK_DIR "/out.bin", gout, sizeof(gout));
    return 0;
}
