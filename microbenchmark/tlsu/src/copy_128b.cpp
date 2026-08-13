// TLSU 端到端 P0 用例：一条 128 字节的 TLOAD + 回写。
//
// 32 x fp32 = 128 B，恰好等于 Streaming L2 后端的 BEAT_SIZE，是能构造的最小
// 完整搬运：连续、对齐、不跨 cacheline（CACHELINE_SIZE=256），一个 beat 一次
// 写回。用 TSTORE 把 tile 写回 GM，使结果成为架构可见。
//
// 判据不在 ELF 内：gfrun（黄金）与 gfsim 各跑一遍，比对 cross_model_result
// 区域的内存 dump。本用例只负责产生激励并正常收尾。
// 比对由 SuperScalarModel 的 scripts/run_tlsu_compare.sh 驱动。
#include <common/pto_tileop.hpp>
#include "benchmark.h"
#include "tlsu_finish.h"

using namespace pto;

#define RESULT_SIZE 4096
TLSU_RESULT_BUFFER(RESULT_SIZE);

// 形状可由 -DTLSU_M / -DTLSU_N 覆盖。默认 4x8 fp32 = 128 B。
//
// 不要用 M=1 的退化形状：编译器会发出 dstTile 为空的 TLOAD，gfrun 和 gfsim
// 都报 "TLOAD with empty dstTile!" 并 abort。
#ifndef TLSU_M
#define TLSU_M 4
#endif
#ifndef TLSU_N
#define TLSU_N 8
#endif

constexpr int M = TLSU_M;
constexpr int N = TLSU_N;  // M * N * sizeof(float) = 128 B

// 源数据放 .bss 并由程序自己填，避免依赖 .data 的初值加载。
// .bss 可写且已建 bank：ELF 加载器对每个 SHF_ALLOC 段都 fn_create，
// 只有 SHF_EXECINSTR 的段才受写保护。
static float src[M * N];

int main()
{
    float *dst = (float *)cross_model_result;

    for (int i = 0; i < M * N; ++i) {
        src[i] = (float)(i + 1);
    }
    for (int i = 0; i < M * N; ++i) {
        dst[i] = 0.0f;
    }

    using gm_t = global_tensor<float, RowMajor<M, N>>;
    using tile_t = Tile<Location::Vec, float, M, N, BLayout::RowMajor>;
    using iter_t = global_iterator<gm_t, tile_t>;

    iter_t gA(src), gC(dst);
    auto gA0 = gA(0, 0), gC0 = gC(0, 0);
    tile_t tA;

    // 算子名由工具链版本决定：-mlxbc 会把工具链自带的
    // lib/clang/<ver>/include/tileop-api/ 前置，遮蔽仓库里的
    // benchmark/two-level-arch/include/common/（显式 -I 也压不过）。
    // PTO 0.57.1 那版叫 TCOPYIN/TCOPYOUT，v0.58 起恢复为 TLOAD/TSTORE。
    BENCHSTART;
    TLOAD(tA, gA0);    // GM -> TR
    TSTORE(gC0, tA);   // TR -> GM
    BENCHEND;

    // 排空窗口。finisher 一旦退休就终止仿真，此前尚未全局可见的写会从
    // dump 中丢失 —— 实测：把一笔标量 store 紧贴 tlsu_finish() 放置，
    // 它在 dump 里就不见了；挪到 tile op 之前即正常。
    // 所有用例都必须在收尾前留出排空窗口。
    static volatile int spin;
    for (int k = 0; k < 20000; ++k) {
        spin = spin + 1;
    }

    tlsu_finish(1);
    return 0;
}
