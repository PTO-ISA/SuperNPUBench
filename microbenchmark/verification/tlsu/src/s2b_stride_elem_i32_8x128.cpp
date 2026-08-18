// S 组 —— stride 单位的判别用例：行距取 160 元素，故意不是 128 的 4 倍。
//
// s2_strided 用的是 512 元素行距。int32 是 4 字节，512 元素 = 2048 字节，而
// 128 列的行长恰好也是 512 字节 —— 于是"把 512 当字节用"这个错误正好退化成
// 一次稠密搬运。虽然 PAD 标记仍能让它显形，但失败签名长得像"行距被忽略"，
// 而不是"单位搞错了"。
//
// 这一档把两者拆开：
//   行距 160 元素 = 640 字节；行长 128 元素 = 512 字节；行间空隙 32 元素。
//   若哪一侧按字节解释，行距就成了 160 字节 = 40 元素 —— 比行长 128 还小，
//   第 1 行起每行都会退回到上一行内部，结果里出现大量重复的低列号元素。
// 两种错误的 dump 完全不同，一眼可分。
//
// 元素编码 0xT_rrr_cccc 自带行列号，所以"读回了第几行第几列"是直接可读的，
// 不需要反推地址。
#include "tlsu_bench.hpp"

#define RESULT_SIZE TLSU_PHYS_BYTES
TLSU_RESULT_BUFFER(RESULT_SIZE);

constexpr int M = 8;
constexpr int N = 128;
// 元素数。刻意选成与 sizeof(int32_t) 无倍数关系的值，见上面的说明。
constexpr int GM_STRIDE = 160;

alignas(256) constinit auto gSrc =
    MakeTlsuPattern<int32_t, M, N, GM_STRIDE>(TLSU_TAG_A);

int main()
{
    int32_t *dst = TLSU_REGION(0);

    using gm_src_t = global_tensor<int32_t, RowMajor<M, N, GM_STRIDE>>;
    using gm_dst_t = global_tensor<int32_t, RowMajor<M, N>>;
    using tile_t = Tile<Location::Vec, int32_t, M, N, BLayout::RowMajor>;

    global_iterator<gm_src_t, tile_t> gA(gSrc.v);
    global_iterator<gm_dst_t, tile_t> gC(dst);
    // 行距不同 -> 返回类型不同，不能合并成一条 auto 声明。
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
