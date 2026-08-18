// TLSU 端到端 P0 用例：8x128 int32 的 copy（TLOAD + TSTORE）。
//
// 8 x 128 x int32 = 4096 B。每行 128 x 4 = 512 B，正好 2 条 cacheline
// （CACHELINE_SIZE=256）、4 个 beat（BEAT_SIZE=128）；4 行共 8 条 cacheline。
// 比单 beat 的粒度更能覆盖真实的多行、多 beat、跨 cacheline 切分逻辑。
//
// 行数取 8 而非 4，是为了让有效区正好填满 tile 寄存器的物理容量。
// gfrun 的 TLOAD 循环上界是 totalRow = tileTotalSize / (totalCol * eleSize)
// （TMAEngine.cpp:157），由 tile 物理大小反推，validRow 只用于一个断言、
// 不限制搬运范围。B.IOT 给这个 tile 分的 size class 是 4KB，128 列 int32
// 即 512 B/行，所以它一定会搬 8 行。源对象若只有 4 行就会被读越界。
// 用 TSTORE 把 tile 写回 GM，使结果成为架构可见。
//
// 判据不在 ELF 内：gfrun（黄金）与 gfsim 各跑一遍，比对 cross_model_result
// 区域的内存 dump。本用例只负责产生激励并正常收尾。
// 比对由 SuperScalarModel 的 scripts/run_tlsu_compare.sh 驱动。
#include <common/pto_tileop.hpp>
#include "benchmark.h"
#include "tlsu_finish.h"

using namespace pto;

#define RESULT_SIZE 16384
TLSU_RESULT_BUFFER(RESULT_SIZE);

// 形状可由 -DTLSU_M / -DTLSU_N 覆盖。默认 8x128 int32 = 4096 B。
//
// 不要用 M=1 的退化形状：编译器会发出 dstTile 为空的 TLOAD，gfrun 和 gfsim
// 都报 "TLOAD with empty dstTile!" 并 abort。
#ifndef TLSU_M
#define TLSU_M 8
#endif
#ifndef TLSU_N
#define TLSU_N 128
#endif

constexpr int M = TLSU_M;
constexpr int N = TLSU_N;  // M * N * sizeof(int32_t) = 4096 B

// 源数据必须烘进 ELF，不能在程序启动后用标量 store 初始化。
//
// 标量访存目前没有 shared 路径：标量 store 走私有域（TLSU 前端的
// CommitStoreToL1 → L1D），而 TLOAD 读的是 shared/GM。用标量循环填 src，
// 那些字节到不了 tile load 能看见的地方，TLOAD 读回来的是未初始化内存。
//
// 所以 src 是带初值的全局变量，由 ELF 加载器直接把 .data 的内容灌进内存
// （ELF.cpp 对 SHT_PROGBITS 段逐字节 fn_load），完全不经过标量写路径。
// constinit 保证它是静态初始化而非运行期动态初始化。
struct SrcTile {
    int32_t v[M * N];
};

// 每个元素编码自己的行列号：0x00rr00cc（行列均从 1 起）。
// 于是 hex dump 里一眼能看出某个字节来自 tile 的哪一格，
// 地址错位、行距算错、beat 顺序颠倒都会立刻显形。0 表示从未被写。
static constexpr SrcTile MakeSrc()
{
    SrcTile d{};
    for (int r = 0; r < M; ++r) {
        for (int c = 0; c < N; ++c) {
            d.v[r * N + c] = ((r + 1) << 16) | (c + 1);
        }
    }
    return d;
}

alignas(256) constinit SrcTile gSrc = MakeSrc();

int main()
{
    // 目的缓冲在 .bss，加载时已是零，不需要（也不能）用标量 store 清零。
    int32_t *dst = (int32_t *)cross_model_result;
    int32_t *src = gSrc.v;

    using gm_t = global_tensor<int32_t, RowMajor<M, N>>;
    using tile_t = Tile<Location::Vec, int32_t, M, N, BLayout::RowMajor>;
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
