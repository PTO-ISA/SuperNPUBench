// S 组 —— strided：从一个宽矩阵里抠出 8x128 的块。
//
// 源是 8 x 512 int32 的 GM，块只取每行前 128 列，于是
//   strideGm = 512 * 4 = 2048 B  >  validCol * sizeof(D) = 512 B
// 每行搬完要跳过 1536 B 才到下一行的起点。P0 那档 stride 恰等于行长，
// 地址生成退化成一段连续搬运，根本不检验行距 —— 这一档才真的用上 B.IOR
// 的 src1（行距），也就是之前 isa/MInst.cpp 操作数次序反了的那个字段。
//
// 目的侧仍是连续的（stride == 行长），所以比对失败可以先怀疑读侧。
//
// 行间填充（每行第 128..511 列）打的是 TLSU_TAG_PAD。它们绝不该出现在结果
// 里 —— 一旦出现，就是行距算小了、把填充区当有效数据搬了进来。
#include "tlsu_bench.hpp"

#define RESULT_SIZE TLSU_PHYS_BYTES
TLSU_RESULT_BUFFER(RESULT_SIZE);

constexpr int M = 8;
constexpr int N = 128;
constexpr int GM_STRIDE = 512;   // 元素数；字节行距 = 512 * 4 = 2048

alignas(256) constinit auto gSrc =
    MakeTlsuPattern<int32_t, M, N, GM_STRIDE>(TLSU_TAG_A);

int main()
{
    int32_t *dst = TLSU_REGION(0);

    // RowMajor 的第三个模板参数就是行距（layout.hpp:119），默认等于列数。
    // 显式给成 GM_STRIDE 即得到一个宽矩阵上的窗口视图。
    using gm_src_t = global_tensor<int32_t, RowMajor<M, N, GM_STRIDE>>;
    using gm_dst_t = global_tensor<int32_t, RowMajor<M, N>>;
    using tile_t = Tile<Location::Vec, int32_t, M, N, BLayout::RowMajor>;

    global_iterator<gm_src_t, tile_t> gA(gSrc.v);
    global_iterator<gm_dst_t, tile_t> gC(dst);
    // 两个 iterator 的行距不同，返回类型也就不同，不能写在同一条 auto 声明里。
    auto gA0 = gA(0, 0);
    auto gC0 = gC(0, 0);
    tile_t tA;

    BENCHSTART;
    TLOAD(tA, gA0);
    TSTORE(gC0, tA);
    BENCHEND;

    TlsuDrain();
    tlsu_finish(1);
    return 0;
}
